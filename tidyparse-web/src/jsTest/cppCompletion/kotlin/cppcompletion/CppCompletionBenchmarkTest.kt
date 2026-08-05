package cppcompletion

import CppClangdCompletionGroup
import cppClangdAstContextDto
import cppCompletionContextDto
import cppCompletionContextFromDto
import cppEditorStatementSnapshot
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import ai.hypergraph.kaliningraph.parsing.freeze
import com.ionspin.kotlin.bignum.integer.BigInteger
import kotlinx.browser.window
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.async
import kotlinx.coroutines.await
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.promise
import kotlinx.coroutines.withTimeout
import org.w3c.fetch.RequestInit
import kotlin.js.Promise
import kotlin.math.roundToInt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

private const val CPP_ROUTE = "/__cpp_completion"
private const val CPP_PRECISION_SAMPLES = 100
private const val CPP_SAMPLES_PER_LENGTH = 10
private const val CPP_DISPLAY_SAMPLES = 3
private const val CPP_SERVICE_TIMEOUT_MILLIS = 8_000
// Context extraction is deliberately prefetched. Its browser deadline must therefore cover the
// bounded queue wait as well as clangd and the compiler-backed declaration oracle; ordinary bridge
// requests retain the tighter timeout above.
private const val CPP_CONTEXT_REQUEST_TIMEOUT_MILLIS = 20_000
private const val CPP_COMPILE_TIMEOUT_MILLIS = 30_000
private const val CPP_MAX_COMPILE_REQUEST = 1920
// Candidate-weighted sharding keeps all compiler workers busy. Measurements on the committed
// fixtures put 3k blocks at the best point between repeated fixture parsing and oversized clang
// processes; it also leaves ample headroom below the native bridge's per-process timeout.
private const val CPP_MAX_CANDIDATE_BLOCKS_PER_FIXTURE_BUNDLE = 3_000
// Leave a small final compiler wave while the larger first wave overlaps the remaining CFG work.
private const val CPP_COMPILE_WAVE_CURSOR_TARGET = 560
private const val CPP_CASE_DEADLINE_RESERVE_MILLIS = 2_000L
// Keep clangd's serialized document queue busy while the browser derives the current statement's
// cursor quotients. Four statements cover that overlap without allowing compiler-backed
// declaration oracles to build an unproductive backlog behind the serialized clangd document.
private const val CPP_CONTEXT_LOOKAHEAD_STATEMENTS = 4
private const val CPP_MIN_AGGREGATE_PRECISION = 99.0
private const val CPP_MIN_CASE_PRECISION = 95.0

data class CppFixture(val name: String, val source: String)
data class CompileResult(val compiled: Boolean, val timedOut: Boolean, val diagnostics: String)
data class CppCandidateBundle(val source: String, val candidates: Int)

data class BenchmarkStatus(
  val enabled: Boolean,
  val clangd: String?,
  val compiler: String?,
  val fixtures: List<String>,
  val samplesPerInstance: Int,
  val startInstance: Int,
  val maxInstances: Int?,
  val timeLimitMillis: Long
)

/** Minimal browser client for the native test-only clangd/clang++ middleware. */
class CppBenchmarkService {
  suspend fun status(): BenchmarkStatus {
    val json = request("$CPP_ROUTE/status")
    return BenchmarkStatus(
      enabled = json.enabled as? Boolean ?: false,
      clangd = json.clangd as? String,
      compiler = json.compiler as? String,
      fixtures = jsonArray(json.fixtures).mapNotNull { it as? String },
      samplesPerInstance = (json.samplesPerInstance as? Number)?.toInt() ?: CPP_PRECISION_SAMPLES,
      startInstance = (json.startInstance as? Number)?.toInt() ?: 0,
      maxInstances = (json.maxInstances as? Number)?.toInt(),
      timeLimitMillis = (json.timeLimitMillis as? Number)?.toLong() ?: 180_000L
    )
  }

  suspend fun fixtures(): List<CppFixture> = jsonArray(request("$CPP_ROUTE/fixtures").fixtures).map {
    CppFixture(it.name as String, it.source as String)
  }

  suspend fun context(
    source: String,
    line: Int,
    character: Int,
    fixture: String? = null
  ): CppCompletionContext {
    val payload = js("({})")
    payload.source = source
    payload.line = line
    payload.character = character
    if (fixture != null) payload.fixture = fixture
    val json = request("$CPP_ROUTE/context", payload, CPP_CONTEXT_REQUEST_TIMEOUT_MILLIS)
    val snapshot = requireNotNull(cppEditorStatementSnapshot(source, line, character)) {
      "The benchmark cursor is not a completable C++ statement location"
    }
    val completionGroups = jsonArray(json.completionGroups).map { group ->
      CppClangdCompletionGroup(
        result = group,
        receiverMember = group.receiverMember as? Boolean ?: false,
        receiverOperator = group.receiverOperator as? String
      )
    }
    val ast = cppClangdAstContextDto(json.ast, source, line, character)
    return cppCompletionContextFromDto(cppCompletionContextDto(
      source = source,
      completionGroups = completionGroups,
      signatures = json.signatures,
      hover = json.hover,
      diagnostics = json.diagnostics,
      ast = ast,
      snapshot = snapshot
    ))
  }

  /** Test-only native enrichment; the scored benchmark intentionally uses [context]. */
  suspend fun oracleContext(
    source: String,
    line: Int,
    character: Int,
    fixture: String? = null
  ): CppCompletionContext {
    val payload = js("({})")
    payload.source = source
    payload.line = line
    payload.character = character
    payload.mode = "oracle"
    if (fixture != null) payload.fixture = fixture
    return cppCompletionContextFromDto(
      request("$CPP_ROUTE/context", payload, CPP_CONTEXT_REQUEST_TIMEOUT_MILLIS)
    )
  }

  suspend fun compile(sources: List<String>): List<CompileResult> {
    require(sources.isNotEmpty())
    return sources.chunked(CPP_MAX_COMPILE_REQUEST).flatMap { batch ->
      val payload = js("({})")
      payload.sources = batch.toTypedArray()
      jsonArray(request("$CPP_ROUTE/compile", payload, CPP_COMPILE_TIMEOUT_MILLIS).results).map {
        CompileResult(
          compiled = it.ok as? Boolean ?: false,
          timedOut = it.timedOut as? Boolean ?: false,
          diagnostics = it.diagnostics as? String ?: ""
        )
      }.also { check(it.size == batch.size) }
    }
  }

  /** Expands globally numbered candidate diagnostics from each block-isolated fixture bundle. */
  suspend fun compileCandidateBundles(bundles: List<CppCandidateBundle>): List<List<CompileResult>> {
    require(bundles.isNotEmpty())
    require(bundles.all { it.candidates > 0 })
    return compile(bundles.map { it.source }).zip(bundles) { result, bundle ->
      when {
        result.compiled -> List(bundle.candidates) { CompileResult(true, false, "") }
        result.timedOut -> List(bundle.candidates) { result }
        else -> candidateResults(bundle.candidates, result.diagnostics)
      }
    }
  }

  @Suppress("UNCHECKED_CAST_TO_EXTERNAL_INTERFACE")
  private suspend fun request(
    path: String,
    payload: dynamic = null,
    timeoutMillis: Int = CPP_SERVICE_TIMEOUT_MILLIS
  ): dynamic {
    val controller = js("new AbortController()")
    val options = js("({})")
    options.signal = controller.signal
    if (payload != null) {
      options.method = "POST"
      options.headers = js("({ 'Content-Type': 'application/json' })")
      options.body = JSON.stringify(payload)
    }
    val timeout = window.setTimeout({ controller.abort() }, timeoutMillis)
    try {
      val response = window.fetch(path, options.unsafeCast<RequestInit>()).await()
      val text = response.text().await()
      if (!response.ok) error("$path failed: ${serviceMessage(text)}")
      return JSON.parse(text)
    } finally {
      window.clearTimeout(timeout)
    }
  }

  private fun serviceMessage(body: String): String = try {
    JSON.parse<dynamic>(body).error as? String ?: body.take(500)
  } catch (_: Throwable) {
    body.take(500)
  }
}

private val CPP_CANDIDATE_ERROR = Regex(
  "(?:^|[/\\\\])cpp_completion_[0-9]+_candidate_([0-9]+)\\.cpp:" +
    "[0-9]+(?::[0-9]+)?:\\s*(?:fatal\\s+)?error:"
)
private val CPP_ANY_ERROR = Regex(
  "(?:^|[/\\\\])[^:\\n]+:[0-9]+(?::[0-9]+)?:\\s*(?:fatal\\s+)?error:"
)

private fun candidateResults(count: Int, diagnostics: String): List<CompileResult> {
  val errorLines = diagnostics.lineSequence().filter(CPP_ANY_ERROR::containsMatchIn).toList()
  val failures = errorLines.mapNotNull { line ->
    CPP_CANDIDATE_ERROR.find(line)?.groupValues?.get(1)?.toIntOrNull()?.let { it to line }
  }.groupBy({ it.first }, { it.second })
  if (errorLines.isEmpty() || failures.values.sumOf { it.size } != errorLines.size)
    return List(count) { CompileResult(false, false, diagnostics) }
  return List(count) { candidate ->
    failures[candidate]?.let { CompileResult(false, false, it.joinToString("\n")) }
      ?: CompileResult(true, false, "")
  }
}

data class CompletionScore(
  val fixture: String,
  val line: Int,
  val prefixTokens: Int,
  val precision: Double,
  val recall: Boolean,
  val contextMillis: Long,
  val grammarMillis: Long,
  val inspectedDerivations: String,
  val cfgStats: String = "",
  val compiledSamples: Int = 0,
  val sampledCompletions: Int = 0,
  val contextFacts: String = "",
  val originalCode: String = "",
  val statementTokens: Int = 0,
  val groundTruth: String = "",
  val displayedSamples: List<String> = emptyList(),
  val rejectedSample: String? = null,
  val failure: String? = null
)

data class CompletionReport(
  val totalStatements: Int,
  val totalInstances: Int,
  val selectedInstances: Int,
  val samplesPerInstance: Int,
  val scores: List<CompletionScore>,
  val elapsedMillis: Long,
  val stoppedAtDeadline: Boolean,
  val compilerLogicalCandidates: Int = 0,
  val compilerUniqueCandidates: Int = 0,
  val compilerBundles: Int = 0,
  val compilerMillis: Long = 0
) {
  val precision: Double get() = if (scores.isEmpty()) 0.0 else scores.sumOf { it.precision } / scores.size
  val recall: Double get() = if (scores.isEmpty()) 0.0 else 100.0 * scores.count { it.recall } / scores.size

  fun render(): String = buildString {
    val failures = scores.count { it.failure != null }
    val compiled = scores.sumOf { it.compiledSamples }
    val drawn = scores.sumOf { it.sampledCompletions }
    val sampleCap = scores.size * samplesPerInstance
    appendLine("C++ completion benchmark")
    val statementGroups = scores.groupBy { it.fixture to it.line }
    val completeStatements = statementGroups.values.count { statementScores ->
      val tokenCount = statementScores.first().statementTokens
      statementScores.mapTo(linkedSetOf()) { it.prefixTokens } == (0..tokenCount).toSet()
    }
    val partialStatements = statementGroups.size - completeStatements
    val absentStatements = totalStatements - statementGroups.size
    appendLine(
      "statements=$completeStatements/$totalStatements " +
        "(partial=$partialStatements, absent=$absentStatements), " +
        "coverage=${scores.size}/$totalInstances (selected=$selectedInstances), " +
        "precision=${precision.percent()}, recall=${recall.percent()}, elapsed=${elapsedMillis}ms, " +
        "compiledSamples=$compiled/$drawn (cap=$sampleCap), failures=$failures, " +
        "deadlineStop=$stoppedAtDeadline, compilerCandidates=$compilerUniqueCandidates/" +
        "$compilerLogicalCandidates, compilerBundles=$compilerBundles, compiler=${compilerMillis}ms"
    )
    scores.groupBy { it.fixture }.forEach { (fixture, fixtureScores) ->
      val fixtureStatements = fixtureScores.map { it.line }.distinct().size
      appendLine(
        "$fixture: $fixtureStatements statements, ${fixtureScores.size} instances, " +
          "precision=${fixtureScores.map { it.precision }.average().percent()}, " +
        "recall=${(100.0 * fixtureScores.count { it.recall } / fixtureScores.size).percent()}"
      )
    }
    scores.groupBy { it.fixture to it.line }.values.forEach { statementScores ->
      val first = statementScores.first()
      appendLine()
      appendLine("Orig code: ${first.originalCode} [${first.fixture}:${first.line + 1}]")
      appendLine("CFG stats:")
      statementScores.sortedBy { it.prefixTokens }.forEach { score ->
        appendLine(
          "  index ${score.prefixTokens}: ${score.cfgStats.ifEmpty { "unavailable" }}, " +
            "precision=${score.precision.percent()}, recall=${if (score.recall) "100%" else "0%"}"
        )
        score.rejectedSample?.let { appendLine("    rejected: $it") }
      }
      statementScores.sortedBy { it.prefixTokens }.forEach { score ->
        appendLine(
          "${score.displayedSamples.size} Samples at index ${score.prefixTokens}: " +
            "shortest lengths first"
        )
        score.displayedSamples.forEach { appendLine("  $it") }
      }
    }
    scores.filter { !it.recall || it.failure != null }.forEach { score ->
      appendLine(
        "miss ${score.fixture}:${score.line + 1} prefix=${score.prefixTokens}: " +
          (score.failure ?: "groundTruth=${score.groundTruth}")
      )
    }
  }.trimEnd()
}

private data class BenchmarkCase(val fixture: CppFixture, val truncation: CppTruncation)

/** Compiler-equivalence key: C++ token whitespace and genuinely fresh spellings are immaterial. */
private fun compilerCandidateKey(prefix: List<CppToken>, sample: CppCompletionSample): String {
  val freshSlots = linkedMapOf<String, Int>()
  return buildString {
    (prefix.map { it.text } + sample.tokens).forEach { token ->
      val freshSlot = token.takeIf(sample.freshNames::contains)?.let { name ->
        freshSlots.getOrPut(name) { freshSlots.size }
      }
      if (freshSlot == null) append("T${token.length}:$token") else append("F$freshSlot")
      append('\u0000')
    }
  }
}

private data class PreparedStatementGrammar(
  val context: CppCompletionContext,
  val grammar: PreparedCppCompletionGrammar,
  val contextMillis: Long,
  val preparationMillis: Long
)

private data class PreparedPrefixGrammar(val grammar: PreparedCppCompletionGrammar, val preparationMillis: Long)

private data class PendingCompletion(
  val scoreIndex: Int,
  val case: BenchmarkCase,
  val exactRecall: Boolean,
  val contextMillis: Long,
  val grammarMillis: Long,
  val inspectedDerivations: String,
  val cfgStats: String,
  val contextFacts: String,
  val samples: List<CppCompletionSample>,
  val displayedSamples: List<String>,
  val candidateLines: List<String>,
  val candidateKeys: List<String> = candidateLines,
  val sampleCandidates: Int
) {
  init { require(candidateKeys.size == candidateLines.size) }

  val candidateCount: Int get() = candidateLines.size
}

private data class StatementCompilePlan(
  val id: Int,
  val completions: List<PendingCompletion>,
  val uniqueLines: List<String>,
  val resultIndexes: List<List<Int>>
) {
  val case: BenchmarkCase get() = completions.first().case
  val candidateCount: Int get() = uniqueLines.size
}

private data class StatementCompileSlice(
  val plan: StatementCompilePlan,
  val candidateStart: Int,
  val candidateEnd: Int
) {
  val candidateCount: Int get() = candidateEnd - candidateStart
  val candidateLines: List<String>
    get() = plan.uniqueLines.subList(candidateStart, candidateEnd)
}

private fun packFixtureCompilePlans(
  plans: List<StatementCompilePlan>,
  maxCandidateBlocks: Int = CPP_MAX_CANDIDATE_BLOCKS_PER_FIXTURE_BUNDLE
): List<List<StatementCompileSlice>> {
  require(maxCandidateBlocks > 0)
  return plans.groupBy { it.case.fixture }.values.flatMap { fixturePlans ->
    buildList {
      var packed = mutableListOf<StatementCompileSlice>()
      var weight = 0
      fixturePlans.forEach { plan ->
        require(plan.candidateCount > 0)
        var candidateStart = 0
        while (candidateStart < plan.candidateCount) {
          if (weight == maxCandidateBlocks) {
            add(packed)
            packed = mutableListOf()
            weight = 0
          }
          val candidateEnd = minOf(
            plan.candidateCount,
            candidateStart + maxCandidateBlocks - weight
          )
          packed += StatementCompileSlice(plan, candidateStart, candidateEnd)
          weight += candidateEnd - candidateStart
          candidateStart = candidateEnd
        }
      }
      if (packed.isNotEmpty()) add(packed)
    }
  }
}

private class CppCompletionBenchmark(
  private val service: CppBenchmarkService,
  private val grammar: CppCompletionGrammar,
  private val startInstance: Int,
  private val maxInstances: Int?,
  private val samplesPerInstance: Int,
  private val timeLimitMillis: Long
) {
  suspend fun run(fixtures: List<CppFixture>): CompletionReport = coroutineScope {
    val clock = TimeSource.Monotonic.markNow()
    val all = benchmarkCases(fixtures)
    assertTrue(
      all.all { it.truncation.line.tokens.size <= CPP_MAX_STATEMENT_TOKENS },
      "Every selected statement must fit the finite $CPP_MAX_STATEMENT_TOKENS-token language"
    )
    require(startInstance in 0..all.size)
    val remaining = all.drop(startInstance)
    val selected = maxInstances?.let(remaining::take) ?: remaining
    val scores = arrayOfNulls<CompletionScore>(selected.size)
    val pending = mutableListOf<PendingCompletion>()
    val preparedStatements =
      mutableMapOf<Pair<String, Int>, Deferred<Result<PreparedStatementGrammar>>>()
    val statementsWithQueuedLookahead = mutableSetOf<Pair<String, Int>>()
    val preparedPrefixes = mutableMapOf<Triple<String, Int, List<String>>, PreparedPrefixGrammar>()
    val exactRecallByGrammar = mutableMapOf<PreparedCppCompletionGrammar, Boolean>()
    var compilerLogicalCandidates = 0
    var compilerUniqueCandidates = 0
    var compilerBundles = 0
    var compilerMillis = 0L

    fun queueStatement(case: BenchmarkCase): Deferred<Result<PreparedStatementGrammar>> {
      val key = case.fixture.name to case.truncation.line.number
      return preparedStatements.getOrPut(key) {
        async {
          runCatching {
            val deletion = cppTruncations(case.truncation.line).first()
            val source = truncateCppSource(case.fixture.source, deletion)
            val contextClock = TimeSource.Monotonic.markNow()
            val context = service.context(
              source,
              deletion.line.number,
              deletion.prefixText.length,
              case.fixture.name
            )
            val contextMillis = contextClock.elapsedNow().inWholeMilliseconds
            val preparationClock = TimeSource.Monotonic.markNow()
            val prepared = grammar.prepare(context)
            PreparedStatementGrammar(
              context,
              prepared,
              contextMillis,
              preparationClock.elapsedNow().inWholeMilliseconds
            )
          }
        }
      }
    }

    suspend fun prepareStatement(case: BenchmarkCase): PreparedStatementGrammar =
      queueStatement(case).await().getOrThrow()

    val selectedStatements = selected
      .filter { it.truncation.suffix.isNotEmpty() }
      .distinctBy { it.fixture.name to it.truncation.line.number }
    val statementOrdinals = selectedStatements.withIndex().associate { (index, case) ->
      (case.fixture.name to case.truncation.line.number) to index
    }

    fun queueContextWindow(firstOrdinal: Int) {
      val end = minOf(
        selectedStatements.size,
        firstOrdinal + CPP_CONTEXT_LOOKAHEAD_STATEMENTS
      )
      for (ordinal in firstOrdinal until end) queueStatement(selectedStatements[ordinal])
    }

    fun prefixBinders(prefix: List<CppToken>): List<String> = buildList {
      val structuredOpen = prefix.indexOfFirst { it.text == "[" }.takeIf { open ->
        open >= 0 && prefix.take(open).any { it.text == "auto" }
      }
      prefix.forEachIndexed { index, token ->
        if (token.kind != CppTokenKind.IDENTIFIER) return@forEachIndexed
        val previous = prefix.getOrNull(index - 1)?.text
        if (
          previous == "using" || previous == "&" ||
          previous in setOf("bool", "char", "short", "int", "long", "float", "double") ||
          structuredOpen != null && index > structuredOpen && previous in setOf("[", ",")
        ) add(token.text)
      }
    }.distinct()

    fun preparePrefix(
      case: BenchmarkCase,
      context: CppCompletionContext,
      prefix: List<CppToken>,
      binders: List<String>
    ): PreparedPrefixGrammar {
      val key = Triple(case.fixture.name, case.truncation.line.number, binders)
      preparedPrefixes[key]?.let { return it }
      val clock = TimeSource.Monotonic.markNow()
      val result = PreparedPrefixGrammar(
        grammar.prepare(context, prefix),
        clock.elapsedNow().inWholeMilliseconds
      )
      preparedPrefixes[key] = result
      return result
    }

    suspend fun scorePending(preparedBatch: List<PendingCompletion>) {
      // Cursor locations on one statement first share unique completed lines. Several statement
      // plans from the same fixture then share one translation unit, bounded by candidate weight so
      // clang remains below its per-process deadline. Draw multiplicity is restored from the index
      // maps after compilation; only physical parsing and identical statement-local lines are
      // deduplicated.
      val statementPlans = preparedBatch
        .groupBy { it.case.fixture.name to it.case.truncation.line.number }
        .values
        .mapIndexed { planId, statementCompletions ->
          val uniqueIndexes = linkedMapOf<String, Int>()
          val uniqueLines = mutableListOf<String>()
          val resultIndexes = statementCompletions.map { prepared ->
            prepared.candidateLines.zip(prepared.candidateKeys).map { (line, key) ->
              uniqueIndexes.getOrPut(key) {
                uniqueLines += line
                uniqueLines.lastIndex
              }
            }
          }
          StatementCompilePlan(planId, statementCompletions, uniqueLines, resultIndexes)
        }
      val bundlePlans = packFixtureCompilePlans(statementPlans)
      compilerLogicalCandidates += preparedBatch.sumOf { it.candidateCount }
      compilerUniqueCandidates += statementPlans.sumOf { it.candidateCount }
      compilerBundles += bundlePlans.size
      val compilerClock = TimeSource.Monotonic.markNow()
      val uniqueResults = service.compileCandidateBundles(bundlePlans.map { slices ->
        bundleCppCandidates(
          slices.first().plan.case.fixture,
          slices.map { CppStatementCandidates(it.plan.case, it.candidateLines) }
        )
      })
      compilerMillis += compilerClock.elapsedNow().inWholeMilliseconds
      val resultsByPlan = Array(statementPlans.size) { planId ->
        arrayOfNulls<CompileResult>(statementPlans[planId].candidateCount)
      }
      val compilation = mutableMapOf<Int, List<CompileResult>>()
      bundlePlans.zip(uniqueResults).forEach { (slices, results) ->
        check(results.size == slices.sumOf { it.candidateCount })
        var resultOffset = 0
        slices.forEach { slice ->
          repeat(slice.candidateCount) { localIndex ->
            resultsByPlan[slice.plan.id][slice.candidateStart + localIndex] =
              results[resultOffset + localIndex]
          }
          resultOffset += slice.candidateCount
        }
      }
      statementPlans.forEach { plan ->
        val statementResults = resultsByPlan[plan.id].map { checkNotNull(it) }
        plan.completions.zip(plan.resultIndexes).forEach { (prepared, indexes) ->
          compilation[prepared.scoreIndex] = indexes.map(statementResults::get)
        }
      }
      preparedBatch.forEach { prepared ->
        val candidates = compilation.getValue(prepared.scoreIndex)
        check(candidates.size == prepared.candidateCount)
        val sampleCompilation = candidates.take(prepared.sampleCandidates)
        val freshCompilation = candidates.drop(prepared.sampleCandidates)
        val compiledSamples = sampleCompilation.count { it.compiled }
        val rejectedIndex = sampleCompilation.indexOfFirst { !it.compiled }
        val truncation = prepared.case.truncation
        scores[prepared.scoreIndex] = CompletionScore(
          fixture = prepared.case.fixture.name,
          line = truncation.line.number,
          prefixTokens = truncation.prefix.size,
          precision = if (sampleCompilation.isEmpty()) 0.0
            else 100.0 * compiledSamples / sampleCompilation.size,
          recall = prepared.exactRecall || freshCompilation.any { it.compiled },
          contextMillis = prepared.contextMillis,
          grammarMillis = prepared.grammarMillis,
          inspectedDerivations = prepared.inspectedDerivations,
          cfgStats = prepared.cfgStats,
          compiledSamples = compiledSamples,
          sampledCompletions = prepared.samples.size,
          contextFacts = prepared.contextFacts,
          originalCode = truncation.line.text.trim(),
          statementTokens = truncation.line.tokens.size,
          groundTruth = truncation.suffix.joinToString(" ") { it.text },
          displayedSamples = prepared.displayedSamples,
          rejectedSample = rejectedIndex.takeIf { it >= 0 }?.let { indexInSamples ->
            val completion = prepared.samples[indexInSamples].tokens.joinToString(" ")
            val diagnostic = sampleCompilation[indexInSamples].diagnostics.compactDiagnostic()
            if (diagnostic.isEmpty()) completion else "$completion [$diagnostic]"
          }
        )
      }
    }

    var compilationJob: Deferred<Unit>? = null
    suspend fun dispatchPending() {
      if (pending.isEmpty()) return
      val preparedBatch = pending.toList()
      pending.clear()
      // Only one scoring request is in flight. It can use the compiler's configured worker pool
      // while this coroutine prepares the next wave with clangd; awaiting here bounds retained
      // source text and prevents HTTP requests from multiplying compiler concurrency.
      compilationJob?.await()
      compilationJob = async { scorePending(preparedBatch) }
    }

    // Start the first semantic request before baseline validation. clangd and clang++ are separate
    // native processes, and the compiler bridge keeps a process-wide permit pool, so this hides
    // the remaining cold-context/base-CFG setup without increasing either system's concurrency.
    queueContextWindow(0)
    fixtures.chunked(160).forEach { batch ->
      service.compile(batch.map { it.source }).zip(batch).forEach { (result, fixture) ->
        check(result.compiled) { "${fixture.name} is not a valid baseline:\n${result.diagnostics}" }
      }
    }
    var deadlineStop = false
    for ((scoreIndex, case) in selected.withIndex()) {
      if (clock.elapsedNow().inWholeMilliseconds + CPP_CASE_DEADLINE_RESERVE_MILLIS >= timeLimitMillis) {
        deadlineStop = true
        break
      }
      val truncation = case.truncation
      val statementPrepared = if (truncation.suffix.isEmpty()) null else try {
        prepareStatement(case)
      } catch (error: Throwable) {
        scores[scoreIndex] = failed(case, 0, 0, error.message ?: "clangd context failed")
        continue
      }
      // Keep a small ordered context window in flight while this statement's CPU-heavy quotient
      // counting and sampling proceed. clangd still serializes document versions, but once one LSP
      // query completes its declaration probes can overlap the next LSP query and browser CFG work.
      // The deferred map prevents duplicate requests for every remaining boundary of a statement.
      val statementKey = case.fixture.name to truncation.line.number
      if (statementsWithQueuedLookahead.add(statementKey)) {
        statementOrdinals[statementKey]?.let { ordinal ->
          queueContextWindow(ordinal + 1)
        }
      }
      val context = if (statementPrepared == null) {
        // The endpoint quotient is exactly epsilon and needs no semantic facts. Its short-first
        // length-zero draws still reach the compiler-backed scorer (and hit the fixture cache).
        CppCompletionContext(emptySet())
      } else statementPrepared.context
      val contextMillis = statementPrepared?.contextMillis ?: 0
      val binders = prefixBinders(truncation.prefix)
      val activeGrammar = when {
        statementPrepared == null -> null
        binders.isEmpty() -> PreparedPrefixGrammar(
          statementPrepared.grammar,
          statementPrepared.preparationMillis
        )
        else -> preparePrefix(case, context, truncation.prefix, binders)
      }
      if (activeGrammar != null && activeGrammar.preparationMillis > CPP_CFG_BUDGET_MILLIS) {
        scores[scoreIndex] = failed(
          case,
          contextMillis,
          activeGrammar.preparationMillis,
          "Base CFG ${activeGrammar.preparationMillis}ms > ${CPP_CFG_BUDGET_MILLIS}ms",
          context.summary()
        )
        continue
      }
      val grammarClock = TimeSource.Monotonic.markNow()
      val language = when {
        activeGrammar == null -> grammar.generate(context, truncation.prefix)
        else -> activeGrammar.grammar.generate(truncation.prefix)
      }
      val grammarMillis = grammarClock.elapsedNow().inWholeMilliseconds
      if (grammarMillis > CPP_CFG_BUDGET_MILLIS) {
        scores[scoreIndex] = failed(
          case,
          contextMillis,
          grammarMillis,
          "CFG ${grammarMillis}ms > ${CPP_CFG_BUDGET_MILLIS}ms",
          context.summary()
        )
        continue
      }

      // Every cursor language is an exact left quotient of its prepared grammar, hence
      // suffix in prefix^-1(G) iff the unchanged full statement is in G. Cache that invariant
      // once per ordinary/binder-sensitive grammar variant; guarded fresh matching remains
      // cursor-specific below when exact membership is false.
      val exactRecall = activeGrammar?.grammar?.let { preparedGrammar ->
        exactRecallByGrammar.getOrPut(preparedGrammar) {
          preparedGrammar.recognizes(truncation.line.tokens)
        }
      } ?: true
      val freshMatches = if (exactRecall) emptyList() else language.freshMatches(truncation.suffix)
      val samplePreparationClock = TimeSource.Monotonic.markNow()
      val sampler = CppCompletionSampler(
        language,
        context.identifiers + truncation.line.tokens.map { it.text },
        Random(case.fixture.name.hashCode() * 31 + truncation.line.number * 17 + truncation.prefix.size)
      )
      sampler.prepare(samplesPerInstance)
      val preparedSamplesMillis = samplePreparationClock.elapsedNow().inWholeMilliseconds
      val samples = sampler.sample(samplesPerInstance)
      check(samples.size <= samplesPerInstance) {
        "Short-first sampler exceeded its cap: ${samples.size}/$samplesPerInstance"
      }
      check(samples.zipWithNext().all { (left, right) -> left.length <= right.length }) {
        "Short-first samples are not ordered by exact terminal length"
      }
      val lengthHistogram = samples.groupingBy { it.length }.eachCount()
      check(lengthHistogram.values.all { it <= CPP_SAMPLES_PER_LENGTH }) {
        "A terminal-length slice exceeded $CPP_SAMPLES_PER_LENGTH samples: $lengthHistogram"
      }
      val displayedSamples = shortestDisplayedSamples(case, samples)
      check(displayedSamples.size == minOf(CPP_DISPLAY_SAMPLES, samples.size))
      val sampleLines = samples.map { sample -> completedLine(case, sample.tokens) }
      val freshNames = FreshCppNames(
        context.identifiers + truncation.line.tokens.map { it.text },
        Random(case.fixture.name.hashCode() xor truncation.prefix.size)
      )
      val freshLines = freshMatches.map { match ->
        val replacements = match.groups.associateWith { freshNames.next() }
        completedLine(
          case,
          truncation.suffix.mapIndexed { indexInSuffix, token ->
            replacements.entries.firstOrNull { indexInSuffix in it.key }?.value ?: token.text
          }
        )
      }
      val candidateLines = sampleLines + freshLines
      val candidateKeys = samples.map { sample -> compilerCandidateKey(truncation.prefix, sample) } + freshLines
      val prepared = PendingCompletion(
        scoreIndex = scoreIndex,
        case = case,
        exactRecall = exactRecall,
        contextMillis = contextMillis,
        grammarMillis = grammarMillis,
        inspectedDerivations = sampler.inspectedDerivationCount.toString(),
        cfgStats = "${language.bounded.structuralStats()}, rules=${language.syntax.size}, " +
          "maxTokens=${language.templateTokens}, " +
          "${if (sampler.coversFullBound) "total" else "inspected"}Derivations" +
          "[${sampler.inspectedLengths}]=${sampler.inspectedDerivationCount}, " +
          "generated=${grammarMillis}ms, base=${activeGrammar?.preparationMillis ?: 0}ms, " +
          "phases=d${language.conditioningMetrics.derivativeMillis}/" +
          "r${language.conditioningMetrics.reachableMillis}/" +
          "b${language.conditioningMetrics.boundedMillis}ms, " +
          "contextualized=${contextMillis}ms, preparedSamples=${preparedSamplesMillis}ms, " +
          "sampleLengths=${lengthHistogram.entries.joinToString(prefix = "[", postfix = "]") { "${it.key}:${it.value}" }}, " +
          "context=${context.summary()}",
        contextFacts = context.summary(),
        samples = samples,
        displayedSamples = displayedSamples,
        candidateLines = candidateLines,
        candidateKeys = candidateKeys,
        sampleCandidates = sampleLines.size
      )
      if (prepared.candidateCount == 0) {
        scores[scoreIndex] = failed(case, contextMillis, grammarMillis, "CFG has no sampled derivations")
      } else {
        pending += prepared
      }
      if (truncation.suffix.isEmpty()) {
        // Do not split a physical statement between compiler waves: statement-level bundling is
        // what removes the repeated fixture body and duplicate completed lines.
        if (pending.size >= CPP_COMPILE_WAVE_CURSOR_TARGET) dispatchPending()
        // Cursor cases are contiguous by statement. Release its large conditioner workspace after
        // the endpoint so the exhaustive corpus sweep retains only the one-line lookahead.
        preparedStatements.remove(statementKey)
        preparedPrefixes.keys.removeAll { (fixture, line, _) ->
          fixture == statementKey.first && line == statementKey.second
        }
        exactRecallByGrammar.clear()
      }
    }
    dispatchPending()
    compilationJob?.await()

    val completedScores = scores.filterNotNull()

    CompletionReport(
      totalStatements = all.map { it.fixture.name to it.truncation.line.number }.distinct().size,
      totalInstances = all.size,
      selectedInstances = selected.size,
      samplesPerInstance = samplesPerInstance,
      scores = completedScores,
      elapsedMillis = clock.elapsedNow().inWholeMilliseconds,
      stoppedAtDeadline = deadlineStop,
      compilerLogicalCandidates = compilerLogicalCandidates,
      compilerUniqueCandidates = compilerUniqueCandidates,
      compilerBundles = compilerBundles,
      compilerMillis = compilerMillis
    )
  }

  private fun failed(
    case: BenchmarkCase,
    context: Long,
    grammar: Long,
    reason: String,
    contextFacts: String = ""
  ) =
    CompletionScore(
      fixture = case.fixture.name,
      line = case.truncation.line.number,
      prefixTokens = case.truncation.prefix.size,
      precision = 0.0,
      recall = false,
      contextMillis = context,
      grammarMillis = grammar,
      inspectedDerivations = "0",
      contextFacts = contextFacts,
      originalCode = case.truncation.line.text.trim(),
      statementTokens = case.truncation.line.tokens.size,
      groundTruth = case.truncation.suffix.joinToString(" ") { it.text },
      failure = reason
    )

  private fun completedLine(case: BenchmarkCase, suffix: List<String>): String {
    // Keep the source spelling before the cursor. CPP14Lexer deliberately exposes a shift as two
    // adjacent `<` tokens for the parser; joining those raw prefix tokens with spaces changes `<<`
    // into the ill-formed `< <`. Sampled suffix terminals are already materialized spellings.
    val prefix = case.truncation.prefixText
    val joinsShift = prefix.endsWith('<') && suffix.firstOrNull() == "<"
    val separator = if (prefix.isNotBlank() && suffix.isNotEmpty() && !joinsShift) " " else ""
    return prefix + separator + suffix.renderCppTokens()
  }

  /** Shows one representative from each shortest slice, then fills from those slices if needed. */
  private fun shortestDisplayedSamples(
    case: BenchmarkCase,
    samples: List<CppCompletionSample>
  ): List<String> {
    val representatives = linkedSetOf<Int>()
    var previousLength: Int? = null
    samples.forEachIndexed { index, sample ->
      if (sample.length != previousLength && representatives.size < CPP_DISPLAY_SAMPLES) {
        representatives += index
        previousLength = sample.length
      }
    }
    samples.indices.forEach { index ->
      if (representatives.size < minOf(CPP_DISPLAY_SAMPLES, samples.size)) representatives += index
    }
    return representatives.sortedBy { samples[it].length }.map { index ->
      val sample = samples[index]
      "length ${sample.length}: ${completedLine(case, sample.tokens).trim()}"
    }
  }

}

private const val CPP_BUNDLE_MARKER = "__TIDYPARSE_CPP_COMPLETION_BUNDLE__"
private val CPP_UNBRACED_CONTROL = Regex("(?:for|if|while|switch)\\s*\\(.*\\)\\s*|else|do")

private data class CppStatementCandidates(val case: BenchmarkCase, val candidates: List<String>)

/**
 * Compiles each candidate in its own lexical block while retaining one copy of the fixture. Every
 * replaced statement is followed by its original baseline line, so declarations and side effects
 * needed by later candidate sites remain in their original scope. Replacements are applied from
 * the bottom of the source upward to keep all lexer-provided offsets stable.
 *
 * The native bridge rewrites [CPP_BUNDLE_MARKER] to the enclosing source index. Candidate numbers
 * are global within this fixture bundle, making diagnostics unambiguous across statement sites.
 */
private fun bundleCppCandidates(
  fixture: CppFixture,
  statements: List<CppStatementCandidates>
): CppCandidateBundle {
  require(statements.isNotEmpty())
  require(statements.all { it.case.fixture == fixture && it.candidates.isNotEmpty() })
  require(statements.map { it.case.truncation.line.number }.distinct().size == statements.size)

  var candidateOffset = 0
  val replacements = statements.map { statement ->
    val offset = candidateOffset
    candidateOffset += statement.candidates.size
    statement to offset
  }
  var source = fixture.source
  replacements.sortedByDescending { (statement, _) -> statement.case.truncation.line.start }
    .forEach { (statement, offset) ->
      val line = statement.case.truncation.line
      val indent = line.text.takeWhile { it == ' ' || it == '\t' }
      val previous = fixture.source.substring(0, line.start).lineSequence()
        .lastOrNull { it.isNotBlank() }.orEmpty().trim()
      val controlledBody = CPP_UNBRACED_CONTROL.matches(previous)
      val replacement = buildString {
        append(indent).append("{\n")
        statement.candidates.forEachIndexed { localIndex, candidate ->
          append(indent).append("  {\n")
          append("#line ${line.number + 1} \"")
            .append(CPP_BUNDLE_MARKER).append("_candidate_")
            .append(offset + localIndex).append(".cpp\"\n")
          append(candidate).append('\n')
          append(indent).append("  }\n")
        }
        if (controlledBody) {
          append("#line ${line.number + 1} \"")
            .append(CPP_BUNDLE_MARKER).append("_baseline.cpp\"\n")
          append(line.text).append('\n')
          append(indent).append('}')
        } else {
          append(indent).append("}\n")
          append("#line ${line.number + 1} \"")
            .append(CPP_BUNDLE_MARKER).append("_baseline.cpp\"\n")
          append(line.text)
        }
      }
      source = source.replaceRange(line.start, line.contentEnd, replacement)
    }
  return CppCandidateBundle(source, candidateOffset)
}

private fun List<String>.renderCppTokens(): String = buildString {
  var index = 0
  while (index < size) {
    if (
      this@renderCppTokens[index] == "<" &&
      this@renderCppTokens.getOrNull(index + 1) == "<"
    ) {
      append("<<")
      index += 2
    } else {
      append(this@renderCppTokens[index++])
    }
    if (index < size) append(' ')
  }
}

class CppCompletionBenchmarkTest {
  @Test
  fun reportGroupsCfgStatsAndSamplesByOriginalStatement() {
    fun score(index: Int, line: Int = 4) = CompletionScore(
      fixture = "sample.cpp",
      line = line,
      prefixTokens = index,
      precision = 100.0,
      recall = true,
      contextMillis = 1,
      grammarMillis = 2,
      inspectedDerivations = "7",
      cfgStats = "CFG(|Σ|=3, |V|=4, |P|=5), rules=8, maxTokens=32, " +
        "inspectedDerivations[0..2]=7, generated=2ms, preparedSamples=1ms",
      originalCode = "items.push_back(value);",
      displayedSamples = listOf("items.push_back(value);", "items.clear();", "return;")
    )
    val rendered = CompletionReport(
      3, 4, 4, 100,
      listOf(
        score(0).copy(statementTokens = 2),
        score(1).copy(statementTokens = 2),
        score(2).copy(statementTokens = 2),
        score(0, line = 5).copy(statementTokens = 2)
      ),
      3, false
    ).render()

    assertTrue("statements=1/3 (partial=1, absent=1)" in rendered)
    assertEquals(2, rendered.split("Orig code:").size - 1)
    assertTrue("CFG stats:\n  index 0:" in rendered)
    assertTrue("3 Samples at index 0:" in rendered)
    assertTrue("3 Samples at index 2:" in rendered)
  }

  @Test
  fun bundledCompilerDiagnosticsRemainCandidateSpecific() {
    val classified = candidateResults(
      3,
      "cpp_completion_7_candidate_1.cpp:4:9: error: unknown identifier"
    )
    assertEquals(listOf(true, false, true), classified.map { it.compiled })
    assertTrue("candidate_1" in classified[1].diagnostics)
    assertTrue(candidateResults(3, "cpp_completion_7_baseline.cpp:4:9: error: broken").none { it.compiled })
  }

  @Test
  fun compilerCandidateKeysAlphaNormalizeOnlyFreshIdentifiers() {
    val first = CppCompletionSample(
      listOf("int", "freshId_aaaaaaaaaaaa", "=", "freshId_aaaaaaaaaaaa", ";"),
      setOf("freshId_aaaaaaaaaaaa")
    )
    val renamed = CppCompletionSample(
      listOf("int", "freshId_bbbbbbbbbbbb", "=", "freshId_bbbbbbbbbbbb", ";"),
      setOf("freshId_bbbbbbbbbbbb")
    )
    val distinctBinders = CppCompletionSample(
      listOf("int", "freshId_cccccccccccc", "=", "freshId_dddddddddddd", ";"),
      setOf("freshId_cccccccccccc", "freshId_dddddddddddd")
    )

    assertEquals(compilerCandidateKey(emptyList(), first), compilerCandidateKey(emptyList(), renamed))
    assertTrue(compilerCandidateKey(emptyList(), first) != compilerCandidateKey(emptyList(), distinctBinders))
    assertTrue(
      compilerCandidateKey(emptyList(), first) !=
        compilerCandidateKey(emptyList(), first.copy(tokens = first.tokens.toMutableList().also { it[0] = "long" }))
    )
  }

  @Test
  fun oneFixtureBundleMapsCandidatesAcrossMultipleStatementSites() {
    val source = """
      int main() {
        int value = 1;
        if (value)
          value += 2;
        int copy = value;
        return copy;
      }
    """.trimIndent()
    val fixture = CppFixture("multi.cpp", source)
    fun case(code: String): BenchmarkCase {
      val line = cppLines(source).single { it.text.trim() == code }
      return BenchmarkCase(fixture, cppTruncations(line).first())
    }

    val bundle = bundleCppCandidates(
      fixture,
      listOf(
        CppStatementCandidates(
          case("int value = 1;"),
          listOf("int value = 3;", "int value = 4;")
        ),
        CppStatementCandidates(case("value += 2;"), listOf("value += 5;")),
        CppStatementCandidates(case("int copy = value;"), listOf("int copy = value + 1;"))
      )
    )

    assertEquals(4, bundle.candidates)
    (0 until bundle.candidates).forEach { index ->
      assertEquals(1, bundle.source.split("_candidate_$index.cpp").size - 1)
    }
    assertEquals(3, bundle.source.split("_baseline.cpp").size - 1)
    assertTrue(bundle.source.indexOf("int value = 1;") < bundle.source.indexOf("if (value)"))
    assertTrue(bundle.source.indexOf("value += 2;") < bundle.source.indexOf("int copy = value;"))
    // The unbraced if still controls the generated outer block, which contains its baseline line.
    assertTrue(Regex("if \\(value\\)\\s+\\{[\\s\\S]*value \\+= 2;\\s+\\}").containsMatchIn(bundle.source))

    val classified = candidateResults(
      bundle.candidates,
      "cpp_completion_11_candidate_2.cpp:4:9: error: rejected second-site candidate"
    )
    assertEquals(listOf(true, true, false, true), classified.map { it.compiled })
  }

  @Test
  fun fixtureBundlePackingIsCandidateBoundedAndReassemblesStatementSlices() {
    val source = """
      int main() {
        int first = 1;
        int second = 2;
      }
    """.trimIndent()
    val fixture = CppFixture("packing.cpp", source)
    fun plan(id: Int, code: String, candidates: Int): StatementCompilePlan {
      val line = cppLines(source).single { it.text.trim() == code }
      val case = BenchmarkCase(fixture, cppTruncations(line).first())
      val completion = PendingCompletion(
        scoreIndex = id,
        case = case,
        exactRecall = true,
        contextMillis = 0,
        grammarMillis = 0,
        inspectedDerivations = "1",
        cfgStats = "",
        contextFacts = "",
        samples = emptyList(),
        displayedSamples = emptyList(),
        candidateLines = emptyList(),
        sampleCandidates = 0
      )
      return StatementCompilePlan(
        id,
        listOf(completion),
        List(candidates) { "$code // candidate $it" },
        emptyList()
      )
    }

    val packed = packFixtureCompilePlans(
      listOf(plan(0, "int first = 1;", 5), plan(1, "int second = 2;", 2)),
      maxCandidateBlocks = 3
    )

    assertEquals(listOf(3, 3, 1), packed.map { bundle -> bundle.sumOf { it.candidateCount } })
    assertEquals(
      listOf((0 until 5).toList(), (0 until 2).toList()),
      (0..1).map { planId ->
        packed.flatten().filter { it.plan.id == planId }
          .flatMap { it.candidateStart until it.candidateEnd }
      }
    )
  }

  @Test
  fun mapsUnresolvedAndEnclosingCallableContextFacts() {
    val json = js("({})")
    json.identifiers = arrayOf("visible")
    json.unresolvedIdentifiers = arrayOf("later_name", "other_name", "later_name")
    json.requiredIdentifier = "later_name"
    json.requiredTypes = arrayOf("Widget", "const Widget &", "Widget")
    json.probedRequiredTypes = arrayOf("Widget", "const Widget &", "double")
    json.enclosingReturnType = "const Widget &"
    json.enclosingClassType = "Factory"
    json.thisType = "const Factory *"
    json.mutableFields = arrayOf("cache")
    json.types = arrayOf(js("({ name: 'AbstractBase', type: 'AbstractBase', kind: 'class', abstract: true })"))

    val context = cppCompletionContextFromDto(json)

    assertEquals(setOf("later_name", "other_name"), context.unresolvedIdentifiers)
    assertEquals("later_name", context.requiredIdentifier)
    assertEquals(setOf("Widget", "const Widget &"), context.requiredTypes)
    assertEquals(setOf("Widget", "const Widget &", "double"), context.probedRequiredTypes)
    assertEquals("const Widget &", context.enclosingReturnType)
    assertEquals("Factory", context.enclosingClassType)
    assertEquals("const Factory *", context.thisType)
    assertEquals(setOf("cache"), context.mutableFields)
    assertTrue(context.types.single().abstract)
  }

  @Test
  fun nestedTemplateDeclarationIsRecognizedAtEveryTypeBoundary() {
    val line = cppLines("std::vector<std::unique_ptr<Animal>> animals;").single()
    val context = CppCompletionContext(
      identifiers = setOf("std", "vector", "unique_ptr", "make_unique", "Animal", "animals"),
      typeNames = setOf("Animal"),
      types = listOf(CppReference("Animal", type = "Animal", kind = "class", source = "ast")),
      unresolvedIdentifiers = setOf("animals"),
      requiredIdentifier = "animals",
      requiredTypes = setOf("std::vector<std::unique_ptr<Animal>>")
    )
    cppTruncations(line).forEach { truncation ->
      val language = CppCompletionGrammar().generate(context, truncation.prefix)
      assertTrue(
        language.recognizes(truncation.suffix),
        "Nested-template declaration rejected at token index ${truncation.prefix.size}: " +
          projectCppTokens(line.tokens)
      )
    }
  }

  @Test
  fun unprobedDeclarationTypesSurviveANonexhaustiveRequiredTypeOracle() {
    val context = CppCompletionContext(
      identifiers = setOf("std", "string", "value"),
      sourceIdentifiers = setOf("std", "string", "value"),
      headers = setOf("string"),
      requiredIdentifier = "value",
      requiredTypes = setOf("int"),
      probedRequiredTypes = setOf("int", "double")
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    fun statement(source: String) = cppLines(source).single().tokens

    assertTrue(language.recognizes(statement("int value;")))
    assertFalse(language.recognizes(statement("double value;")))
    assertTrue(
      language.recognizes(statement("std::string value;")),
      "A successful partial probe universe must not exclude an unprobed source-valid type"
    )
  }

  @Test
  fun hexadecimalDigitsAndUserSuffixesRetainTheirLiteralCategories() {
    val tokens = cppLines(
      "0xdead 0xFFu 123_people 0b1010 0755 .5 1.0e3;"
    ).single().tokens.dropLast(1)

    assertEquals(
      listOf(
        CppTokenKind.INTEGER,
        CppTokenKind.INTEGER,
        CppTokenKind.USER_DEFINED_INTEGER,
        CppTokenKind.INTEGER,
        CppTokenKind.INTEGER,
        CppTokenKind.FLOATING,
        CppTokenKind.FLOATING
      ),
      tokens.map(CppToken::kind)
    )
  }

  @Test
  fun stringStreamInsertionHasTheStaticOstreamResultType() {
    val identifiers = setOf("std", "ostringstream", "ostream", "move", "out", "name", "copy")
    val values = listOf(
      CppReference("out", type = "std::ostringstream", kind = "variable", source = "ast"),
      CppReference("name", type = "std::string", kind = "variable", source = "ast")
    )
    val base = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      headers = setOf("sstream"),
      values = values,
      types = listOf(
        CppReference("ostringstream", type = "std::ostringstream", kind = "class", source = "ast")
      )
    )
    val chainedInsertion = cppLines("out << name << \" (\";").single().tokens
    assertTrue(
      CppCompletionGrammar().generate(base, emptyList()).recognizes(chainedInsertion),
      "A derived output stream must remain usable throughout a chained insertion"
    )

    val declarationContext = base.copy(
      unresolvedIdentifiers = setOf("copy"),
      requiredIdentifier = "copy",
      requiredTypes = setOf("std::ostringstream")
    )
    val generator = CppCompletionGrammar()
    assertTrue(
      generator.generate(declarationContext, emptyList()).recognizes(
        cppLines("std::ostringstream copy = std::move(out);").single().tokens
      ),
      "The string stream object itself remains movable"
    )
    assertFalse(
      generator.generate(declarationContext, emptyList()).recognizes(
        cppLines("std::ostringstream copy = std::move(out << name);").single().tokens
      ),
      "operator<< returns ostream&, so moving its result cannot initialize an ostringstream"
    )
  }

  @Test
  fun standardOstreamRemainsUsableButIsNotAssignable() {
    val identifiers = setOf("std", "cout", "ostream")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      values = listOf(
        CppReference("std::cout", type = "std::ostream", kind = "variable", source = "ast")
      )
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    fun statement(source: String): List<CppToken> = cppLines(source).single().tokens

    assertTrue(language.recognizes(statement("std::cout << 0;")))
    assertFalse(
      language.recognizes(statement("std::cout = std::cout;")),
      "basic_ostream's protected/deleted assignment must not enter a typesafe statement CFG"
    )
    val declarationLanguage = CppCompletionGrammar().generate(
      context.copy(
        unresolvedIdentifiers = setOf("copy"),
        requiredIdentifier = "copy",
        requiredTypes = setOf("std::ostream")
      ),
      emptyList()
    )
    assertFalse(declarationLanguage.recognizes(statement("std::ostream copy = std::cout;")))
    assertTrue(declarationLanguage.recognizes(statement("const std::ostream& copy = std::cout;")))
  }

  @Test
  fun incompleteStandardLibraryTypesAreNotDeclarationCandidates() {
    val identifiers = setOf("std", "ostringstream", "out")
    val base = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = setOf("std", "out"),
      unresolvedIdentifiers = setOf("out"),
      requiredIdentifier = "out"
    )
    val statement = cppLines("std::ostringstream out;").single().tokens

    assertFalse(
      CppCompletionGrammar().generate(base, emptyList()).recognizes(statement),
      "<iostream> only forward-declares basic_ostringstream; <sstream> is required"
    )
    assertTrue(
      CppCompletionGrammar().generate(base.copy(headers = setOf("sstream")), emptyList())
        .recognizes(statement)
    )
  }

  @Test
  fun constReceiversExcludeNonconstUserMethods() {
    val size = CppReference(
      "size", returnType = "std::size_t", kind = "method",
      detail = "std::size_t size() const", ownerType = "Document", source = "ast"
    )
    val titled = CppReference(
      "titled", returnType = "Document &", parameters = listOf(CppParameter(type = "const char *")),
      kind = "method", detail = "Document &titled(const char *)",
      ownerType = "Document", source = "ast"
    )
    val context = CppCompletionContext(
      identifiers = setOf("document", "Document", "size", "titled"),
      sourceIdentifiers = setOf("document", "Document", "size", "titled"),
      values = listOf(CppReference("document", type = "const Document &", kind = "variable", source = "ast")),
      types = listOf(CppReference("Document", type = "Document", kind = "class", source = "ast")),
      membersByType = listOf(CppTypeMembers("Document", listOf(size, titled)))
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes(cppLines("document.size();").single().tokens))
    assertFalse(language.recognizes(cppLines("document.titled(\"\").size();").single().tokens))
  }

  @Test
  fun unqualifiedIteratorAliasesDoNotBecomeVectorElementTypes() {
    val vehicle = CppReference("Vehicle", type = "Vehicle", kind = "class", source = "ast")
    val begin = CppReference(
      "begin", returnType = "iterator", kind = "method", detail = "iterator begin()",
      ownerType = "std::vector<Vehicle>", source = "completion"
    )
    val context = CppCompletionContext(
      identifiers = setOf("std", "vector", "Vehicle", "iterator", "items"),
      sourceIdentifiers = setOf("std", "vector", "Vehicle", "items"),
      headers = setOf("vector"),
      types = listOf(vehicle),
      membersByType = listOf(CppTypeMembers("std::vector<Vehicle>", listOf(begin))),
      unresolvedIdentifiers = setOf("items"),
      requiredIdentifier = "items"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes(cppLines("std::vector<Vehicle> items;").single().tokens))
    assertFalse(language.recognizes(cppLines("std::vector<iterator> items;").single().tokens))
  }

  @Test
  fun placeholderCompletionTypesDoNotBecomePointerDeclarations() {
    val language = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = setOf("id", "values", "LocalRecord"),
        sourceIdentifiers = setOf("id", "values", "LocalRecord"),
        values = listOf(
          CppReference("id", type = "type", kind = "variable", source = "completion"),
          // libc++ recovery signatures can leak a compound placeholder rather than a bare `Tp`.
          CppReference("values", type = "const Ep *", kind = "variable", source = "ast")
        ),
        types = listOf(
          CppReference(
            "value_type", type = "value_type", kind = "typeAlias", source = "ast"
          ),
          // Recovery records sometimes carry a header-template placeholder in `type` while their
          // visible declaration name is source-local. The unspellable payload must not become a
          // user-declared type merely because the record name itself occurs in the file.
          CppReference(
            "LocalRecord", type = "Tp", kind = "class", source = "ast"
          )
        ),
        requiredIdentifier = "values"
      ),
      emptyList()
    )

    assertFalse(language.recognizes(cppLines("type* values;").single().tokens))
    assertFalse(language.recognizes(cppLines("value_type* values;").single().tokens))
    assertFalse(language.recognizes(cppLines("Tp* values;").single().tokens))
    assertFalse(language.recognizes(cppLines("const Ep* values;").single().tokens))
  }

  @Test
  fun stringAppendRequiresAMutableReceiverButAcceptsPrvalues() {
    val identifiers = setOf("std", "string", "append", "text", "makeText", "readText")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      values = listOf(CppReference("text", type = "std::string", kind = "variable", source = "ast")),
      types = listOf(CppReference("string", type = "std::string", kind = "class", source = "ast")),
      functions = listOf(
        CppReference("makeText", returnType = "std::string", kind = "function", source = "ast"),
        CppReference("readText", returnType = "const std::string &", kind = "function", source = "ast")
      )
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    fun statement(source: String): List<CppToken> = cppLines(source).single().tokens

    assertTrue(
      language.recognizes(statement("text.append(\" mutable\");")),
      "Known std::string::append must accept a mutable string lvalue"
    )
    assertTrue(
      language.recognizes(statement("makeText().append(\" prvalue\");")),
      "Known std::string::append must accept a string prvalue"
    )
    assertFalse(
      language.recognizes(statement("readText().append(\" forbidden\");")),
      "A const std::string& result must not become a mutable append receiver"
    )
    assertTrue(language.recognizes(statement("(&text)->append(\" pointer\");")))
    assertFalse(
      language.recognizes(statement("&text->append(\" precedence bug\");")),
      "Address-of has lower precedence than arrow member access"
    )
    assertFalse(
      language.recognizes(statement("text.append(nullptr);")),
      "basic_string text operations reject nullptr even though it converts to an ordinary pointer"
    )
  }

  @Test
  fun constMethodsOnlyExposeMutableFieldsAsModifiableLvalues() {
    fun context(mutableFields: Set<String>) = CppCompletionContext(
      identifiers = setOf("std", "string", "append", "name_"),
      sourceIdentifiers = setOf("std", "string", "append", "name_"),
      values = listOf(CppReference("name_", type = "std::string", kind = "field", source = "completion")),
      thisType = "const Route *",
      mutableFields = mutableFields
    )
    val statement = cppLines("name_.append(\"suffix\");").single().tokens

    assertFalse(
      CppCompletionGrammar().generate(context(emptySet()), emptyList()).recognizes(statement),
      "An ordinary field is a const lvalue inside a const-qualified member function"
    )
    assertTrue(
      CppCompletionGrammar().generate(context(setOf("name_")), emptyList()).recognizes(statement),
      "A field declared mutable remains modifiable inside a const-qualified member function"
    )
  }

  @Test
  fun abstractRecordsCanOnlyBeDeclaredThroughReferencesOrPointers() {
    val animal = CppReference("Animal", type = "Animal", kind = "class", source = "ast", abstract = true)
    val dog = CppReference("Dog", type = "Dog", kind = "class", source = "ast")
    val context = CppCompletionContext(
      identifiers = setOf("std", "move", "Animal", "Dog", "dog", "base", "copy"),
      sourceIdentifiers = setOf("std", "move", "Animal", "Dog", "dog", "base", "copy"),
      values = listOf(
        CppReference("dog", type = "Dog", kind = "variable", source = "ast"),
        CppReference("base", type = "Animal *", kind = "variable", source = "ast")
      ),
      types = listOf(animal, dog),
      conversions = listOf(CppConversion("Dog", "Animal")),
      requiredIdentifier = "copy",
      requiredTypes = setOf("Animal")
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    fun statement(source: String) = cppLines(source).single().tokens

    assertTrue(language.recognizes(statement("const Animal& copy = dog;")))
    assertFalse(
      language.recognizes(statement("Animal copy = dog;")),
      "An abstract record cannot be instantiated by value"
    )
    assertFalse(
      language.recognizes(statement("auto copy = *base;")),
      "auto must not silently deduce an abstract by-value record"
    )
    val expressionLanguage = CppCompletionGrammar().generate(
      context.copy(requiredIdentifier = null, requiredTypes = emptySet()),
      emptyList()
    )
    assertTrue(expressionLanguage.recognizes(statement("true ? *base : *base;")))
    assertFalse(
      expressionLanguage.recognizes(statement("true ? *base : std::move(*base);")),
      "Mixing an abstract lvalue and xvalue would require materializing the abstract record"
    )
  }

  @Test
  fun moveOnlyConditionalsDoNotMixLvaluesAndXvalues() {
    val context = CppCompletionContext(
      identifiers = setOf("std", "unique_ptr", "Bicycle", "bicycle", "move"),
      sourceIdentifiers = setOf("std", "unique_ptr", "Bicycle", "bicycle", "move"),
      values = listOf(CppReference("bicycle", type = "std::unique_ptr<Bicycle>", kind = "variable", source = "ast"))
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    fun statement(source: String) = cppLines(source).single().tokens

    assertTrue(language.recognizes(statement("true ? bicycle : bicycle;")))
    assertTrue(language.recognizes(
      statement("true ? std::move(bicycle) : std::move(bicycle);")
    ))
    assertFalse(
      language.recognizes(statement("true ? bicycle : std::move(bicycle);")),
      "A mixed move-only conditional requires an implicitly deleted copy"
    )
  }

  @Test
  fun scopedValuesHideUnqualifiedCallableCompletions() {
    val context = CppCompletionContext(
      identifiers = setOf("index"),
      sourceIdentifiers = setOf("index"),
      values = listOf(CppReference("index", type = "int", kind = "variable", source = "ast")),
      functions = listOf(
        CppReference(
          "index", returnType = "int", parameters = listOf(CppParameter(type = "int")),
          kind = "function", source = "completion"
        )
      )
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertFalse(
      language.recognizes(cppLines("index(0);").single().tokens),
      "A local object shadows an unqualified function with the same spelling"
    )
  }

  @Test
  fun dereferencedPointersAreParenthesizedBeforeDotMemberAccess() {
    val identifiers = setOf("raw_vehicle", "Vehicle", "range")
    val member = CppReference(
      "range", returnType = "int", kind = "method", detail = "int range() const",
      ownerType = "Vehicle", source = "ast"
    )
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      values = listOf(CppReference("raw_vehicle", type = "Vehicle *", kind = "variable", source = "ast")),
      types = listOf(CppReference("Vehicle", type = "Vehicle", kind = "class", source = "ast")),
      membersByType = listOf(CppTypeMembers("Vehicle", listOf(member)))
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    fun statement(source: String) = cppLines(source).single().tokens

    assertTrue(language.recognizes(statement("raw_vehicle->range();")))
    assertTrue(language.recognizes(statement("(*raw_vehicle).range();")))
    assertFalse(
      language.recognizes(statement("*raw_vehicle.range();")),
      "Unary dereference has lower precedence than postfix member access"
    )
  }

  @Test
  fun standardFactoryTemplatesRetainTheirExplicitTypeArgument() {
    val identifiers = setOf("std", "make_shared", "make_unique", "Node")
    val node = CppReference("Node", type = "Node", kind = "class", source = "ast")
    val constructor = CppReference(
      "Node", returnType = "Node", kind = "constructor", ownerType = "Node", source = "ast"
    )
    val misleadingTemplateCompletion = CppReference(
      "std::make_shared", returnType = "std::shared_ptr<Node>", kind = "function",
      source = "completion"
    )
    val language = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = identifiers,
        sourceIdentifiers = identifiers,
        types = listOf(node),
        functions = listOf(constructor, misleadingTemplateCompletion)
      ),
      emptyList()
    )
    fun statement(source: String) = cppLines(source).single().tokens

    assertTrue(language.recognizes(statement("std::make_shared<Node>();")))
    assertFalse(
      language.recognizes(statement("std::make_shared();")),
      "A deduced clang completion must not erase make_shared's required template argument"
    )
  }

  @Test
  fun expandedLibraryTypesAcceptValuesConstructedThroughASourceAliasAtEveryBoundary() {
    val record = "std::tuple<int,std::string,double>"
    val records = "std::map<int,$record>"
    val identifiers = setOf("std", "map", "tuple", "string", "Record", "records", "emplace")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      headers = setOf("map", "tuple", "string"),
      values = listOf(CppReference("records", type = records, kind = "variable", source = "ast")),
      types = listOf(
        CppReference(
          "Record", type = record, kind = "typeAlias", detail = record, source = "source"
        )
      )
    )
    val line = cppLines("records.emplace(7, Record{7, \"Noor\", 88.5});").single()
    val prepared = CppCompletionGrammar().prepare(context)

    cppTruncations(line).forEach { truncation ->
      assertTrue(
        prepared.generate(truncation.prefix).recognizes(truncation.suffix),
        "The source alias and clang's expanded tuple spelling must agree at index ${truncation.prefix.size}"
      )
    }
  }

  @Test
  fun implicitThisFieldsRemainScopedMutableReceiversAtEveryBoundary() {
    // Preserve clang/libc++'s function-signature spacing; it is semantically significant to this
    // regression because the compact standard type is also present in the same context.
    val function = "std::function<int (int)>"
    val steps = "std::vector<$function>"
    val identifiers = setOf(
      "std", "function", "vector", "move", "Pipeline", "steps_", "step", "push_back"
    )
    val field = CppReference(
      "steps_", type = steps, kind = "field", receiverMember = true,
      ownerType = "Pipeline", source = "completion"
    )
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      headers = setOf("functional", "utility", "vector"),
      values = listOf(CppReference("step", type = function, kind = "variable", source = "ast")),
      types = listOf(CppReference("Pipeline", type = "Pipeline", kind = "class", source = "ast")),
      completions = listOf(field),
      thisType = "Pipeline *"
    )
    val line = cppLines("steps_.push_back(std::move(step));").single()
    val prepared = CppCompletionGrammar().prepare(context)

    cppTruncations(line).forEach { truncation ->
      assertTrue(
        prepared.generate(truncation.prefix).recognizes(truncation.suffix),
        "The implicit-this field must remain a receiver at index ${truncation.prefix.size}"
      )
    }
    assertFalse(
      CppCompletionGrammar().generate(context.copy(thisType = "const Pipeline *"), emptyList())
        .recognizes(line.tokens),
      "An ordinary field of a const implicit object is not a mutable receiver"
    )
  }

  @Test
  fun variantQueriesRetainTheirCorrelatedTemplateAlternative() {
    val variant = "std::variant<std::monostate,int,std::string>"
    val base = CppCompletionContext(
      identifiers = setOf("std", "variant", "monostate", "string", "payload"),
      sourceIdentifiers = setOf("std", "variant", "monostate", "string", "payload"),
      headers = setOf("variant", "string"),
      values = listOf(CppReference("payload", type = variant, kind = "variable", source = "ast"))
    )
    fun statement(source: String) = cppLines(source).single().tokens

    val holds = CppCompletionGrammar().generate(
      base.copy(requiredIdentifier = "textual", requiredTypes = setOf("bool")),
      emptyList()
    )
    assertTrue(holds.recognizes(
      statement("bool textual = std::holds_alternative<std::string>(payload);")
    ))

    val get = CppCompletionGrammar().generate(
      base.copy(requiredIdentifier = "text", requiredTypes = setOf("const std::string *")),
      emptyList()
    )
    assertTrue(get.recognizes(
      statement("const std::string* text = std::get_if<std::string>(&payload);")
    ))
  }

  @Test
  fun duplicateVariantAlternativesExcludeTypeBasedQueries() {
    val variant = "std::variant<int,int>"
    val identifiers = setOf("std", "variant", "payload")
    val language = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = identifiers,
        sourceIdentifiers = identifiers,
        headers = setOf("variant"),
        values = listOf(CppReference("payload", type = variant, kind = "variable", source = "ast"))
      ),
      emptyList()
    )
    fun statement(source: String) = cppLines(source).single().tokens

    assertFalse(
      language.recognizes(statement("std::holds_alternative<int>(payload);")),
      "A type-based variant observer is ill-formed unless its alternative occurs exactly once"
    )
    assertFalse(
      language.recognizes(statement("std::get_if<int>(&payload);")),
      "get_if<T> has the same unique-alternative requirement as holds_alternative<T>"
    )
    val declarationLanguage = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = identifiers + "choice",
        sourceIdentifiers = identifiers + "choice",
        headers = setOf("variant"),
        values = listOf(CppReference("payload", type = variant, kind = "variable", source = "ast")),
        requiredIdentifier = "choice",
        requiredTypes = setOf(variant),
        probedRequiredTypes = setOf(variant)
      ),
      emptyList()
    )
    assertFalse(
      declarationLanguage.recognizes(statement("std::variant<int,int> choice = 0;")),
      "A converting variant constructor is ambiguous when the selected type occurs twice"
    )
  }

  @Test
  fun finiteStreamChainsStopAfterExactlyTwelveInsertions() {
    val identifiers = setOf("std", "cout")
    val language = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = identifiers,
        sourceIdentifiers = identifiers,
        headers = setOf("iostream"),
        values = listOf(
          CppReference("std::cout", type = "std::ostream", kind = "variable", source = "ast")
        )
      ),
      emptyList()
    )
    fun insertionChain(length: Int): List<CppToken> =
      cppLines("std::cout" + " << 0".repeat(length) + ";").single().tokens

    assertTrue(language.recognizes(insertionChain(12)))
    assertFalse(
      language.recognizes(insertionChain(13)),
      "The dedicated stream tier must not compose with itself past its documented finite bound"
    )
  }

  @Test
  fun visitRequiresAConstructibleTemporaryAndExactSafeOverloads() {
    val variant = "std::variant<std::monostate,int,std::string>"
    val identifiers = setOf("std", "variant", "monostate", "string", "visit", "Describe", "payload")
    val constructor = CppReference(
      "Describe", returnType = "Describe", kind = "constructor", ownerType = "Describe",
      detail = "void ()", source = "ast"
    )
    val safeOverloads = listOf(
      CppReference(
        "operator()", returnType = "std::string", parameters = listOf(CppParameter(type = "std::monostate")),
        kind = "method", detail = "std::string (std::monostate) const", ownerType = "Describe", source = "ast"
      ),
      CppReference(
        "operator()", returnType = "std::string", parameters = listOf(CppParameter(type = "int")),
        kind = "method", detail = "std::string (int) const", ownerType = "Describe", source = "ast"
      ),
      CppReference(
        "operator()", returnType = "std::string",
        parameters = listOf(CppParameter(type = "const std::string &")), kind = "method",
        detail = "std::string (const std::string &) const", ownerType = "Describe", source = "ast"
      )
    )
    val base = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      headers = setOf("variant", "string"),
      values = listOf(CppReference("payload", type = variant, kind = "variable", source = "ast")),
      types = listOf(CppReference("Describe", type = "Describe", kind = "class", source = "ast")),
      functions = listOf(constructor),
      membersByType = listOf(CppTypeMembers("Describe", safeOverloads))
    )
    val visit = cppLines("std::visit(Describe{}, payload);").single().tokens

    assertTrue(CppCompletionGrammar().generate(base, emptyList()).recognizes(visit))
    assertTrue(
      CppCompletionGrammar().generate(
        base.copy(functions = emptyList(), defaultConstructibleTypes = setOf("Describe")),
        emptyList()
      ).recognizes(visit),
      "A successful compiler declaration probe is sufficient evidence for the visitor temporary"
    )
    assertFalse(
      CppCompletionGrammar().generate(base.copy(functions = emptyList()), emptyList()).recognizes(visit),
      "A fieldless-looking record is not proof that its default constructor is usable"
    )
    val mismatchedReturn = safeOverloads.toMutableList().also {
      it[2] = it[2].copy(
        returnType = "const std::string &",
        detail = "const std::string & (const std::string &) const"
      )
    }
    assertFalse(
      CppCompletionGrammar().generate(
        base.copy(membersByType = listOf(CppTypeMembers("Describe", mismatchedReturn))),
        emptyList()
      ).recognizes(visit),
      "visit requires one exact common return type, including cv/ref category"
    )
    val nonConstReference = safeOverloads.toMutableList().also {
      it[1] = it[1].copy(
        parameters = listOf(CppParameter(type = "int &")),
        detail = "std::string (int &) const"
      )
    }
    assertFalse(
      CppCompletionGrammar().generate(
        base.copy(membersByType = listOf(CppTypeMembers("Describe", nonConstReference))),
        emptyList()
      ).recognizes(visit),
      "An unrestricted variant expression cannot safely feed an alternative through non-const T&"
    )
  }

  @Test
  fun browserFactsCompleteVisitAtTheActualIncompleteCursor(): Promise<Unit> = MainScope().promise {
    val service = CppBenchmarkService()
    if (service.status().clangd == null) return@promise
    val source = """
      #include <iostream>
      #include <optional>
      #include <string>
      #include <variant>

      struct Describe {
          std::string operator()(std::monostate) const { return "empty"; }
          std::string operator()(int value) const { return std::to_string(value); }
          std::string operator()(const std::string& value) const { return value; }
      };

      int main() {
          std::optional<std::string> nickname = std::nullopt;
          nickname.emplace("Ada");
          std::string display = nickname.value_or("anonymous");
          std::variant<std::monostate, int, std::string> payload = std::monostate{};
          payload = std::string{"ready"};
          bool textual = std::holds_alternative<std::string>(payload);
          const std::string* text = std::get_if<std::string>(&payload);
          std::string rendered = std::visit(
      }
    """.trimIndent()
    val lines = source.lines()
    val line = lines.indexOfFirst { "std::string rendered" in it }
    val character = lines[line].length
    val snapshot = requireNotNull(cppEditorStatementSnapshot(source, line, character))
    val context = service.context(source, line, character, "visit_browser_parity.cpp")
    val completions = CppCompletionGrammar().generate(context, snapshot.tokens).shortestCompletions(
      prefixText = snapshot.prefixText,
      identifiersInFile = context.identifiers,
      limit = CPP_MAX_INTERACTIVE_COMPLETIONS,
      random = Random(snapshot.seed)
    )

    assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.size)
    assertTrue(
      completions.any { it.tokens == listOf("Describe", "{", "}", ",", "payload", ")", ";") },
      "Browser-fact completions omitted `Describe{}, payload);`: " +
        completions.joinToString { it.tokens.joinToString(" ") }
    )
  }

  @Test
  fun browserFactsCompleteTryEmplaceAtTheActualIncompleteCursor(): Promise<Unit> = MainScope().promise {
    val service = CppBenchmarkService()
    if (service.status().clangd == null) return@promise
    val source = """
      #include <iostream>
      #include <map>
      #include <set>
      #include <string>
      #include <tuple>

      int main() {
          using Record = std::tuple<int, std::string, double>;
          std::map<int, Record> records;
          records.emplace(7, Record{ '\0' , "" , 0.0 } ) ;
          records.try_emplace
      }
    """.trimIndent()
    val lines = source.lines()
    val line = lines.indexOfFirst { "records.try_emplace" in it }
    val character = lines[line].length
    val snapshot = requireNotNull(cppEditorStatementSnapshot(source, line, character))
    val context = service.context(source, line, character, "try_emplace_browser_parity.cpp")
    val completions = CppCompletionGrammar().generate(context, snapshot.tokens).shortestCompletions(
      prefixText = snapshot.prefixText,
      identifiersInFile = context.identifiers,
      limit = CPP_MAX_INTERACTIVE_COMPLETIONS,
      random = Random(snapshot.seed)
    )
    val expected = setOf(
      listOf("(", "0", ",", "0", ",", "\"\"", ",", "0", ")", ";"),
      listOf("(", "0", ",", "'\\0'", ",", "\"\"", ",", "'\\0'", ")", ";"),
      listOf("(", "0", ",", "'\\0'", ",", "\"\"", ",", "0.0", ")", ";"),
      listOf("(", "0", ",", "'\\0'", ",", "\"\"", ",", "0", ")", ";"),
      listOf("(", "'\\0'", ",", "'\\0'", ",", "\"\"", ",", "0", ")", ";"),
      listOf("(", "'\\0'", ",", "0", ",", "\"\"", ",", "'\\0'", ")", ";"),
      listOf("(", "'\\0'", ",", "0", ",", "\"\"", ",", "0.0", ")", ";"),
      listOf("(", "'\\0'", ",", "0", ",", "\"\"", ",", "0", ")", ";"),
      listOf("(", "0", ",", "0", ",", "\"\"", ",", "'\\0'", ")", ";"),
      listOf("(", "0", ",", "0", ",", "\"\"", ",", "0.0", ")", ";")
    )

    assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.size)
    assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.map { it.tokens }.distinct().size)
    assertTrue(completions.all { it.length == 10 })
    assertEquals(
      expected,
      completions.map { it.tokens }.toSet(),
      "The actual-cursor browser facts must expose the tuple-valued std::map overload"
    )
  }

  @Test
  fun browserFactsCompleteASecondMapSpecializationAtTheActualTypeBoundary(): Promise<Unit> =
    MainScope().promise {
      val service = CppBenchmarkService()
      if (service.status().clangd == null) return@promise
      val source = """
        #include <iostream>
        #include <map>
        #include <set>
        #include <string>
        #include <tuple>

        int main() {
            using Record = std::tuple<int, std::string, double>;
            std::map<int, Record> records;
            records.emplace(7, Record{ '\0' , "" , 0.0 } ) ;
            records . lower_bound ( '\0' ) ;
            std::map<int, std::string
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { "std::map<int, std::string" in it }
      val character = lines[line].length
      val snapshot = requireNotNull(cppEditorStatementSnapshot(source, line, character))
      val context = service.context(source, line, character, "partial_second_map_specialization.cpp")
      val completions = CppCompletionGrammar().generate(context, snapshot.tokens).shortestCompletions(
        prefixText = snapshot.prefixText,
        identifiersInFile = context.identifiers,
        limit = CPP_MAX_INTERACTIVE_COMPLETIONS,
        random = Random(snapshot.seed)
      )

      assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.size)
      assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.map { it.tokens }.distinct().size)
      assertTrue(completions.all { completion ->
        completion.tokens.firstOrNull() == ">" && completion.tokens.lastOrNull() == ";"
      })
    }

  @Test
  fun compilerSeparatesDefaultConstructionFromLaterDeclaratorTypeUseWhenAvailable(): Promise<Unit> =
    MainScope().promise {
      val service = CppBenchmarkService()
      val status = service.status()
      if (status.clangd == null || status.compiler == null) return@promise
      val fixture = service.fixtures().first { it.name == "optional_variants.cpp" }
      val line = cppStatementLines(fixture.source).single { "std::visit(Describe" in it.text }
      val deletion = cppTruncations(line).first()
      val context = service.oracleContext(
        truncateCppSource(fixture.source, deletion),
        line.number,
        deletion.prefixText.length,
        fixture.name
      )

      assertTrue(
        "Describe" in context.defaultConstructibleTypes,
        "The declaration itself compiles even though a later stream use rejects `Describe rendered`"
      )
      val prepared = CppCompletionGrammar().prepare(context)
      assertTrue(prepared.recognizes(line.tokens))
      cppTruncations(line).take(4).forEach { truncation ->
        assertTrue(prepared.generate(truncation.prefix).recognizes(truncation.suffix))
      }
    }

  @Test
  fun lazilyHashedResidualsPreserveFrozenCfgCountsAndRecognition() {
    val context = CppCompletionContext(
      identifiers = setOf("value"),
      sourceIdentifiers = setOf("value"),
      values = listOf(CppReference("value", type = "int", kind = "variable", source = "ast"))
    )
    val line = cppLines("value = value + 1;").single()
    val prepared = CppCompletionGrammar().prepare(context)
    assertTrue(prepared.recognizes(line.tokens))
    assertFalse(prepared.recognizes(cppLines("value = missing + 1;").single().tokens))
    cppTruncations(line).forEach { truncation ->
      val residual = prepared.generate(truncation.prefix)
      assertFalse(
        residual.syntax is MutableSet<*>,
        "A residual CFG must not publish its owned mutable construction set"
      )
      val frozen = residual.syntax.freeze().boundedAcyclic(
        maxLength = residual.templateTokens,
        startSymbol = residual.bounded.startSymbol
      )
      assertEquals(residual.syntax.toSet(), residual.syntax)
      val projectedSuffix = projectCppTokens(truncation.line.tokens)
        .drop(residual.projectedPrefix.size)
      if (projectedSuffix.isNotEmpty()) {
        assertFalse(
          "|Σ|=0" in residual.bounded.structuralStats(),
          "A productive nonempty residual must report its terminal alphabet"
        )
      }

      assertEquals(
        frozen.derivationCount,
        residual.derivationCount,
        "Freezing must not change the quotient's derivation multiplicity at index ${truncation.prefix.size}"
      )
      assertEquals(
        frozen.recognizes(projectedSuffix),
        residual.recognizes(truncation.suffix),
        "Freezing must not change quotient recognition at index ${truncation.prefix.size}"
      )
      val residualRandom = Random(40_000 + truncation.prefix.size)
      val frozenRandom = Random(40_000 + truncation.prefix.size)
      assertEquals(
        frozen.samplesByIncreasingLength(frozenRandom, sampleLimit = 12, samplesPerLength = 3)
          .toList(),
        residual.bounded.samplesByIncreasingLength(
          residualRandom,
          sampleLimit = 12,
          samplesPerLength = 3
        ).toList(),
        "Chunked residual publication must preserve seeded samples at index ${truncation.prefix.size}"
      )
    }
  }

  @Test
  fun completionSamplerMaterializesOneIdempotentScopedBatch() {
    val bounded = setOf("START" to listOf(CPP_FRESH)).boundedAcyclic(maxLength = 1)
    val language = CppSuffixGrammar(
      bounded = bounded,
      rawPrefix = emptyList(),
      projectedPrefix = emptyList(),
      templateTokens = 1
    )
    val sampler = CppCompletionSampler(language, setOf("alreadyUsed"), Random(90210))

    val first = sampler.sample(3)
    val repeated = sampler.sample(3)
    assertTrue(first === repeated, "Repeated access must return the one materialized batch")
    assertEquals(first, repeated)
    assertTrue(first.all { sample ->
      sample.freshNames.size == 1 && sample.tokens.single() in sample.freshNames
    })
    assertEquals(BigInteger.ONE, sampler.inspectedDerivationCount)
    assertEquals(0..1, sampler.inspectedLengths)
    assertTrue(sampler.coversFullBound)
    assertFailsWith<IllegalArgumentException> { sampler.sample(2) }

    val emptySampler = CppCompletionSampler(language, emptySet(), Random(90210))
    assertTrue(emptySampler.sample(0).isEmpty())
    assertEquals(BigInteger.ZERO, emptySampler.inspectedDerivationCount)
    assertEquals(IntRange.EMPTY, emptySampler.inspectedLengths)
    assertFalse(emptySampler.coversFullBound)
  }

  @Test
  fun clangdReportsFutureUnresolvedNamesAndDeepestCallableWhenAvailable(): Promise<Unit> =
    MainScope().promise {
      val service = CppBenchmarkService()
      if (service.status().clangd == null) return@promise
      val fixtures = service.fixtures()

      val animals = fixtures.first { it.name == "default_animals.cpp" }
      val dogDeclaration = cppStatementLines(animals.source).single { "Dog dog" in it.text }
      val dogDeletion = cppTruncations(dogDeclaration).first()
      val mainContext = service.oracleContext(
        truncateCppSource(animals.source, dogDeletion),
        dogDeclaration.number,
        dogDeletion.prefixText.length
      )
      assertTrue("dog" in mainContext.unresolvedIdentifiers)
      assertEquals("dog", mainContext.requiredIdentifier)
      assertTrue("Dog" in mainContext.requiredTypes)
      assertTrue("Animal" in mainContext.requiredTypes)
      assertTrue(mainContext.requiredTypes.all { it in mainContext.probedRequiredTypes })
      assertTrue("int" in mainContext.probedRequiredTypes)
      assertFalse("int" in mainContext.requiredTypes)
      assertEquals("int", mainContext.enclosingReturnType)
      assertEquals(null, mainContext.enclosingClassType)
      assertEquals(null, mainContext.thisType)

      val animalsDeclaration = cppStatementLines(animals.source).single {
        "std::vector<std::unique_ptr<Animal>> animals" in it.text
      }
      val animalsDeletion = cppTruncations(animalsDeclaration).first()
      val animalsContext = service.oracleContext(
        truncateCppSource(animals.source, animalsDeletion),
        animalsDeclaration.number,
        animalsDeletion.prefixText.length
      )
      assertEquals("animals", animalsContext.requiredIdentifier)
      assertTrue("std::vector<std::unique_ptr<Animal>>" in animalsContext.requiredTypes)
      assertFalse("std::vector<Dog>" in animalsContext.requiredTypes)

      // clangd's recovery AST exposes the range-for variable at some damaged-call boundaries but
      // falls back to a display-only completion spelling (`unique_ptr<Animal> const &`) at others.
      // The bridge must canonicalize that spelling so dereference remains type-safe everywhere.
      val introduce = cppStatementLines(animals.source).single { "introduce(*animal, 2)" in it.text }
      cppTruncations(introduce).filter { it.prefix.size in setOf(0, 4) }.forEach { truncation ->
        val context = service.oracleContext(
          truncateCppSource(animals.source, truncation),
          introduce.number,
          truncation.prefixText.length
        )
        assertTrue(
          context.values.any { reference ->
            reference.name == "animal" && "std::unique_ptr<Animal>" in reference.type.orEmpty()
          },
          "The range-for value needs a canonical smart-pointer type at index ${truncation.prefix.size}"
        )
        assertTrue(
          CppCompletionGrammar().generate(context, truncation.prefix).recognizes(truncation.suffix),
          "The dereferenced free call was rejected at index ${truncation.prefix.size}"
        )
      }

      val routes = fixtures.first { it.name == "fluent_routes.cpp" }
      val sort = cppStatementLines(routes.source).single { "std::sort(fleet.begin()" in it.text }
      val sortDeletion = cppTruncations(sort).first()
      val sortContext = service.oracleContext(
        truncateCppSource(routes.source, sortDeletion),
        sort.number,
        sortDeletion.prefixText.length
      )
      assertTrue("algorithm" in sortContext.headers, "The algorithm header must be reported")
      assertTrue(
        sortContext.values.any { it.name == "fleet" && "vector" in it.type.orEmpty() },
        "fleet was absent from ${sortContext.values.map { it.name to it.type }}"
      )
      assertTrue(
        sortContext.membersByType.flatMap { it.members }.any { it.name == "range" },
        "range was absent from ${sortContext.membersByType.map { it.type to it.members.map(CppReference::name) }}"
      )
      val sortLanguage = CppCompletionGrammar().generate(sortContext, sortDeletion.prefix)
      assertTrue(
        sortLanguage.sourceSyntax.any { (_, rhs) -> encodeIdentifier("sort") in rhs },
        "The algorithm header, vector value, and numeric pointee member must specialize std::sort"
      )

      val returnThis = cppStatementLines(routes.source).first { it.text.trim() == "return *this;" }
      val returnDeletion = cppTruncations(returnThis).first()
      val methodContext = service.oracleContext(
        truncateCppSource(routes.source, returnDeletion),
        returnThis.number,
        returnDeletion.prefixText.length
      )
      assertEquals("RouteBuilder &", methodContext.enclosingReturnType)
      assertEquals("RouteBuilder", methodContext.enclosingClassType)
      assertEquals("RouteBuilder *", methodContext.thisType)
    }

  @Test
  fun clangdRecoversScopedReceiverMembersFromANonemptyPartialStatement(): Promise<Unit> =
    MainScope().promise {
      val service = CppBenchmarkService()
      if (service.status().clangd == null) return@promise
      val source = """
        #include <vector>
        int main() {
          std::vector<long long> independent;
          independent.push_back(
          return 0;
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { "independent.push_back(" in it }
      val context = service.oracleContext(source, line, lines[line].length)
      val vectorMembers = context.membersByType
        .filter { "vector" in it.type && "long long" in it.type }
        .flatMap { it.members }

      assertTrue(context.values.any { it.name == "independent" })
      assertTrue(
        vectorMembers.any { it.name == "push_back" },
        "A direct mid-statement request must recover vector members without an earlier dot cursor"
      )
      assertEquals(null, context.receiver, "The isolated recovery probe must not replace the live receiver")

    }

  @Test
  fun everyStatementAndBoundaryIsEnumeratedWithoutWeakeningLockedCases(): Promise<Unit> = MainScope().promise {
    val noFacts = CppCompletionGrammar().generate(CppCompletionContext(emptySet()), emptyList())
    assertFalse(noFacts.isEmpty, "Builtin literal statements keep a zero-fact cursor grammar productive")
    assertTrue(isAcyclic(noFacts.syntax), "Even a zero-fact cursor grammar must be finite")
    val fixtures = CppBenchmarkService().fixtures()
    assertTrue(fixtures.size >= 12, "The completion corpus must retain at least twelve C++ fixtures")
    val corpusSource = fixtures.joinToString("\n") { it.source }
    val syntaxMisses = fixtures.flatMap { fixture ->
      cppStatementLines(fixture.source).mapNotNull { line ->
        if (cppSingleStatementSyntaxRecognizes(line.tokens)) null
        else "${fixture.name}:${line.number + 1}: ${line.text.trim()}"
      }
    }
    assertTrue(
      syntaxMisses.isEmpty(),
      "Pinned statement grammar rejected compiler-validated corpus lines:\n${syntaxMisses.joinToString("\n")}"
    )
    listOf(
      "using Record", "std::map<", "std::function<", "std::transform(", "std::ranges::transform(",
      "enum class", "std::optional<", "std::variant<", "std::weak_ptr<", "dynamic_cast<",
      "const_cast<", "reinterpret_cast<", "typeid(", "for (const auto& [", "[](int value)",
      "counters[1]", "operator|", "std::string_view"
    ).forEach { syntax ->
      assertTrue(syntax in corpusSource, "Expanded corpus lost the '$syntax' syntax family")
    }
    val all = benchmarkCases(fixtures)
    assertTrue(
      all.all { it.truncation.line.tokens.size <= CPP_MAX_STATEMENT_TOKENS },
      "Every expanded statement must fit the finite $CPP_MAX_STATEMENT_TOKENS-token horizon"
    )
    assertEquals(
      mapOf(
        "associative_records.cpp" to listOf(8, 9, 10, 11, 12, 13, 14, 15, 16),
        "callable_pipeline.cpp" to listOf(16, 17, 21, 22, 27, 28, 29, 30, 31),
        "container_algorithms.cpp" to listOf(10, 11, 12, 13, 14, 15, 16, 17, 18),
        "default_animals.cpp" to listOf(30, 32, 36, 37, 39, 40, 42, 43, 44, 47),
        "enum_bitmask.cpp" to listOf(12, 16, 20, 21, 22, 23, 24, 25, 26),
        "fluent_routes.cpp" to listOf(
          33, 34, 35, 45, 46, 50, 51, 55, 56, 64, 68, 69,
          70, 72, 73, 74, 75, 77, 78, 79, 80, 82, 84, 85
        ),
        "optional_variants.cpp" to listOf(13, 14, 15, 16, 17, 18, 19, 20, 21),
        "pointer_casts.cpp" to listOf(19, 20, 21, 22, 23, 24, 25, 26),
        "polymorphic_casts.cpp" to listOf(23, 24, 25, 26, 27, 28, 29, 30, 31),
        "raii_ownership.cpp" to listOf(15, 16, 17, 18, 19, 20, 21, 22, 23),
        "shared_documents.cpp" to listOf(
          34, 35, 39, 40, 49, 50, 51, 55, 56, 57, 59, 60,
          61, 62, 64, 65, 66, 67, 68
        ),
        "string_transformations.cpp" to listOf(9, 10, 11, 12, 13, 14, 15, 16, 17)
      ),
      all.groupBy { it.fixture.name }.mapValues { (_, cases) ->
        cases.map { it.truncation.line.number + 1 }.distinct()
      },
      "The complete physical statement-line inventory must remain selected"
    )
    assertEquals(2_141, all.size, "Every expanded statement token boundary must be scored")
    assertEquals(
      fixtures.sumOf { cppStatementLines(it.source).size },
      all.count { it.truncation.prefix.size == it.truncation.line.tokens.size }
    )
    all.groupBy { it.fixture.name to it.truncation.line.number }.values.forEach { cases ->
      assertEquals(
        (0..cases.first().truncation.line.tokens.size).toList(),
        cases.map { it.truncation.prefix.size }
      )
    }
    val selected = lockedCases(fixtures).map { case ->
      "${case.fixture.name}:${case.truncation.line.number + 1}:${case.truncation.prefix.size}"
    }
    val exhaustiveKeys = all.mapTo(hashSetOf()) { case ->
      "${case.fixture.name}:${case.truncation.line.number + 1}:${case.truncation.prefix.size}"
    }
    assertTrue(selected.all { it in exhaustiveKeys }, "Every locked case must be in the exhaustive corpus")
    assertEquals(
      listOf(
        "default_animals.cpp:32:0", "fluent_routes.cpp:69:0", "shared_documents.cpp:59:0",
        "default_animals.cpp:32:2", "fluent_routes.cpp:69:2", "shared_documents.cpp:59:2",
        "default_animals.cpp:43:2", "fluent_routes.cpp:64:2", "shared_documents.cpp:68:2",
        "default_animals.cpp:43:8", "fluent_routes.cpp:64:10", "shared_documents.cpp:68:13"
      ),
      selected,
      "Do not weaken or replace the scored C++ completion cases"
    )
  }

  @Test
  fun statementDiscoveryHandlesMultilineFunctionDeclarators() {
    val source = """
      struct Counter
      {
        int field;

        int increment(int value)
        {
          value += 1;
          return value;
        }
      };

      int main()
      {
        Counter counter;
        return counter.increment(1);
      }
    """.trimIndent()

    assertEquals(
      listOf("value += 1;", "return value;", "Counter counter;", "return counter.increment(1);"),
      cppStatementLines(source).map { it.text.trim() }
    )
  }

  @Test
  fun nativeGrammarRecognizesAndSamplesAComplexStatement(): Promise<Unit> = MainScope().promise {
    val service = CppBenchmarkService()
    val status = service.status()
    assertTrue(status.samplesPerInstance > 0)
    val fixtures = service.fixtures()
    val fixture = fixtures.first { it.name == "default_animals.cpp" }
    val target = cppSemicolonLines(fixture.source).first { "animal.speak()" in it.text }
    val generator = CppCompletionGrammar()
    val generated = listOf(0, 2, 3, 4, target.tokens.size).map { prefixTokens ->
      val truncation = cppTruncations(target).first { it.prefix.size == prefixTokens }
      val context = service.oracleContext(
        truncateCppSource(fixture.source, truncation),
        truncation.line.number,
        truncation.prefixText.length
      )
      val clock = TimeSource.Monotonic.markNow()
      val language = generator.generate(context, truncation.prefix)
      val elapsed = clock.elapsedNow().inWholeMilliseconds
      assertTrue(elapsed <= CPP_CFG_BUDGET_MILLIS, "p$prefixTokens CFG generation took ${elapsed}ms")
      assertTrue(
        language.recognizes(truncation.suffix),
        "p$prefixTokens grammar rejected: ${target.tokens.joinToString(" ") { it.text }}"
      )
      assertTrue(isAcyclic(language.syntax), "A generated cursor CFG must be finite and non-recursive")
      assertFalse(language.isEmpty, "Conditioned finite parse forest is empty")
      language to context
    }
    val samples = CppCompletionSampler(generated.first().first, generated.first().second.identifiers, Random(7)).sample(100)
    assertEquals(100, samples.size)
    assertTrue(samples.all { it.tokens.isNotEmpty() }, "The deletion CFG sampled an empty suffix")
    assertTrue(samples.zipWithNext().all { (left, right) -> left.length <= right.length })
    assertTrue(samples.groupingBy { it.length }.eachCount().values.all { it <= CPP_SAMPLES_PER_LENGTH })
    val endpointSamples = CppCompletionSampler(
      generated.last().first,
      generated.last().second.identifiers,
      Random(7)
    ).sample(100)
    assertEquals(CPP_SAMPLES_PER_LENGTH, endpointSamples.size)
    assertEquals(List(CPP_SAMPLES_PER_LENGTH) { emptyList() }, endpointSamples.map { it.tokens })
    assertEquals(List(CPP_SAMPLES_PER_LENGTH) { 0 }, endpointSamples.map { it.length })
  }

  @Test
  fun everyFixtureIsAValidTranslationUnitAndCompilerErrorsStaySeparated(): Promise<Unit> = MainScope().promise {
    val service = CppBenchmarkService()
    val status = service.status()
    if (status.compiler == null) return@promise
    val fixtures = service.fixtures()
    assertTrue(fixtures.size >= 12, "The completion corpus must retain at least twelve C++ fixtures")
    val first = fixtures.first()
    val invalid = first.source.replace("return 0;", "return missing_symbol;")
      .let { if (it == first.source) it + "\nmissing_symbol;\n" else it }
    val results = service.compile(fixtures.map { it.source } + invalid)
    fixtures.zip(results).forEach { (fixture, result) ->
      assertTrue(result.compiled, "${fixture.name} must compile before it can be benchmarked:\n${result.diagnostics}")
    }
    assertFalse(results.last().compiled)
  }

}

/** The only test selected by CPP_COMPLETION_BENCHMARK=1. */
class CppCompletionBenchmarkRunTest {
  @Test
  fun benchmarkCppCompletions(): Promise<Unit> = MainScope().promise {
    val service = CppBenchmarkService()
    val status = service.status()
    assertTrue(status.fixtures.isNotEmpty())
    if (!status.enabled) {
      println(
        "C++ completion benchmark ready. Run the focused command from cppCompletion/README.md " +
          "with CPP_COMPLETION_BENCHMARK=1."
      )
      return@promise
    }
    check(status.clangd != null) { "CPP_COMPLETION_BENCHMARK requires clangd" }
    check(status.compiler != null) { "CPP_COMPLETION_BENCHMARK requires clang++" }
    assertTrue(
      status.samplesPerInstance >= CPP_DISPLAY_SAMPLES,
      "At least $CPP_DISPLAY_SAMPLES precision draws are required to display three samples per cursor"
    )
    val fixtures = service.fixtures()
    val wallClock = TimeSource.Monotonic.markNow()
    val report = withTimeout((status.timeLimitMillis + 30_000L).milliseconds) {
      CppCompletionBenchmark(
        service = service,
        grammar = CppCompletionGrammar(),
        startInstance = status.startInstance,
        maxInstances = status.maxInstances,
        samplesPerInstance = status.samplesPerInstance,
        timeLimitMillis = status.timeLimitMillis
      ).run(fixtures)
    }
    val renderedReport = report.render()
    // A complete corpus report is intentionally detailed and can exceed one megabyte. Sending it
    // as one browser-console payload can overflow Karma's socket even though the benchmark itself
    // has finished successfully, so retain identical stdout while bounding each transport frame.
    renderedReport.lineSequence().chunked(64).forEach { lines -> println(lines.joinToString("\n")) }
    assertTrue(report.scores.isNotEmpty())
    assertEquals(report.selectedInstances, report.scores.size, "Benchmark did not score every selected case")
    if (status.startInstance == 0 && status.maxInstances == null) {
      assertEquals(
        report.totalInstances,
        report.selectedInstances,
        "An uncapped benchmark must score every statement boundary"
      )
    }
    assertFalse(report.stoppedAtDeadline, "Benchmark reached its internal deadline")
    assertTrue(
      report.scores.none { it.failure != null },
      "Benchmark contained failed instances:\n$renderedReport"
    )
    assertEquals(100.0, report.recall, "Every ground-truth continuation must be recognized")
    assertTrue(
      report.precision >= CPP_MIN_AGGREGATE_PRECISION,
      "Aggregate precision fell below $CPP_MIN_AGGREGATE_PRECISION%:\n$renderedReport"
    )
    assertTrue(
      report.scores.all { it.precision >= CPP_MIN_CASE_PRECISION },
      "A completion case fell below $CPP_MIN_CASE_PRECISION% precision:\n$renderedReport"
    )
    assertTrue(wallClock.elapsedNow().inWholeMilliseconds < status.timeLimitMillis + 30_000L)
  }
}

private fun benchmarkCases(fixtures: List<CppFixture>): List<BenchmarkCase> = fixtures.flatMap { fixture ->
  cppStatementLines(fixture.source).flatMap { line ->
    cppTruncations(line).map { BenchmarkCase(fixture, it) }
  }
}

private fun lockedCases(fixtures: List<CppFixture>): List<BenchmarkCase> =
  LOCKED_BENCHMARK_CASES.map { specification ->
    val fixture = fixtures.singleOrNull { it.name == specification.fixture }
      ?: error("Missing locked C++ fixture ${specification.fixture}")
    val line = cppSemicolonLines(fixture.source).singleOrNull { specification.lineNeedle in it.text }
      ?: error("Locked statement is missing or ambiguous: ${specification.fixture}:${specification.lineNeedle}")
    require(specification.prefixTokens in 0..line.tokens.size) {
      "Locked prefix ${specification.prefixTokens} is outside ${specification.fixture}:${line.number + 1}"
    }
    BenchmarkCase(
      fixture,
      cppTruncations(line).single { it.prefix.size == specification.prefixTokens }
    )
  }

private data class LockedBenchmarkCase(
  val fixture: String,
  val lineNeedle: String,
  val prefixTokens: Int
)

/** Stable difficult cases retained as an explicit subset of the exhaustive corpus. */
private val LOCKED_BENCHMARK_CASES = listOf(
  LockedBenchmarkCase("default_animals.cpp", "std::cout << animal.speak()", 0),
  LockedBenchmarkCase("fluent_routes.cpp", "builder.named(\"Harbor Loop\")", 0),
  LockedBenchmarkCase("shared_documents.cpp", "document.titled(\"Memory Notes\")", 0),
  LockedBenchmarkCase("default_animals.cpp", "std::cout << animal.speak()", 2),
  LockedBenchmarkCase("fluent_routes.cpp", "builder.named(\"Harbor Loop\")", 2),
  LockedBenchmarkCase("shared_documents.cpp", "document.titled(\"Memory Notes\")", 2),
  LockedBenchmarkCase("default_animals.cpp", "animals.push_back(std::make_unique<Dog>", 2),
  LockedBenchmarkCase("fluent_routes.cpp", "std::cout << vehicle.label()", 2),
  LockedBenchmarkCase("shared_documents.cpp", "std::cout << preview.append", 2),
  LockedBenchmarkCase("default_animals.cpp", "animals.push_back(std::make_unique<Dog>", 8),
  LockedBenchmarkCase("fluent_routes.cpp", "std::cout << vehicle.label()", 10),
  LockedBenchmarkCase("shared_documents.cpp", "std::cout << preview.append", 13)
)

private fun Double.percent(): String {
  val thousandths = (this * 1_000).roundToInt()
  val whole = thousandths / 1_000
  val fraction = (thousandths % 1_000).toString().padStart(3, '0').trimEnd('0')
  return if (fraction.isEmpty()) "$whole%" else "$whole.$fraction%"
}

private fun CppCompletionContext.summary(): String =
  "v${values.size}/t${types.size}/f${functions.size}/c${completions.size}/" +
    "m${membersByType.sumOf { it.members.size }}/x${conversions.size}/" +
    "u${unresolvedIdentifiers.size}/r${requiredTypes.size}:${probedRequiredTypes.size}"

private fun String.compactDiagnostic(): String = lineSequence()
  .map(String::trim)
  .filter { it.isNotEmpty() }
  .toList()
  .let { lines ->
    lines.firstOrNull { line -> Regex("(?:fatal )?error:", RegexOption.IGNORE_CASE).containsMatchIn(line) }
      ?: lines.firstOrNull()
  }
  ?.replace(Regex("\\s+"), " ")
  ?.take(120)
  .orEmpty()

private fun isAcyclic(grammar: Set<Pair<String, List<String>>>): Boolean {
  val nonterminals = grammar.mapTo(linkedSetOf()) { it.first }
  val edges = grammar.groupBy({ it.first }) { production ->
    production.second.filter { it in nonterminals }
  }.mapValues { (_, successors) -> successors.flatten().toSet() }
  val visiting = mutableSetOf<String>()
  val visited = mutableSetOf<String>()
  fun visit(symbol: String): Boolean {
    if (symbol in visited) return true
    if (!visiting.add(symbol)) return false
    if (edges[symbol].orEmpty().any { !visit(it) }) return false
    visiting.remove(symbol)
    visited.add(symbol)
    return true
  }
  return nonterminals.all(::visit)
}

private fun jsonArray(value: dynamic): List<dynamic> {
  val array = js("(value) => Array.isArray(value)")(value) as Boolean
  if (!array) return emptyList()
  return (0 until ((value.length as Number).toInt())).map { value[it] }
}
