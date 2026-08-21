package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.LED_BUFFER
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import kotlinx.coroutines.*
import org.kosat.round
import kotlin.time.DurationUnit.SECONDS
import kotlin.time.TimeSource

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

abstract class TidyEditor {
  // TODO: eliminate this completely
  open var cfg: CFG = setOf()
  var grammarFileCache: Int = 0
  var cache = mutableMapOf<Int, String>()
  var currentWorkHash = 0
  var epsilons = false
  var ntStubs = true

  abstract fun readDisplayText(): Σᐩ
  abstract fun readEditorText(): Σᐩ
  open fun getCaretPosition(): IntRange = TODO()
  open fun getLineBounds(): IntRange = TODO()
  fun getSelection(): Σᐩ = getCaretPosition().let {
    if (it.let { it.isEmpty() || it.last - it.first == 0 }) ""
    else readEditorText().substring(it).trim()
  }
  open fun setCaretPosition(range: IntRange): Unit = TODO()
  abstract fun currentLine(): Σᐩ
  open fun overwriteRegion(region: IntRange, s: Σᐩ): Unit = TODO()
  abstract fun writeDisplayText(s: Σᐩ)
  abstract fun writeDisplayText(s: (Σᐩ) -> Σᐩ)

  open fun getLatestCFG(): CFG {
    val grammar: String = getGrammarText()
    val hash = grammar.hashCode() + listOf(ntStubs, epsilons).hashCode()
    return try {
      if (hash != grammarFileCache || cfg.isEmpty()) {
        grammar.also { grammarFileCache = hash }
          .parseCFG(validate = true)
          .let {
            if (!ntStubs && !epsilons) it.noEpsilonOrNonterminalStubs
            else if (!ntStubs) it.noNonterminalStubs
            else if (!epsilons) it.noEpsilon
            else it
          }.also { cfg = it }
      } else { cfg }
    } catch (e: Exception) {
      writeDisplayText("<html><body><pre>${e.message!!}</pre></body></html>")
      emptySet()
    }
  }

  var runningJob: Job? = null
  open val stubMatcher = Regex("<\\S+>")

  fun handleTab() {
    val lineIdx = getLineBounds().first
    val line = currentLine()
    var firstPlaceholder = stubMatcher.find(line, (getCaretPosition().first - lineIdx + 1).coerceAtMost(line.length))
    if (firstPlaceholder == null) { firstPlaceholder = stubMatcher.find(line, 0) }
    if (firstPlaceholder == null) { setCaretPosition((lineIdx + line.length).let { it..it }); return }

    setCaretPosition((lineIdx + firstPlaceholder.range.first)..(lineIdx + firstPlaceholder.range.last + 1))
    handleInput() // This will update the completions view
  }

  open fun getApplicableContext(): Σᐩ =
    getSelection().let { if (it.isNotEmpty() && stubMatcher.matches(it)) it else currentLine() }

  open fun handleInput() {
    val caretInGrammar = caretInGrammar()
    val context = getApplicableContext()
    if (context.isEmpty()) return
    println("Applicable context:\n$context")
    val tokens = context.tokenizeByWhitespace()

    val cfg = if (caretInGrammar) CFGCFG(names = tokens.filter { it !in setOf("->", "|") }.toSet()) else getLatestCFG()

    if (cfg.isEmpty()) return

    var hasHole = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { hasHole = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, epsilons, ntStubs).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash.hashCode()
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    runningJob?.cancel()

    val scenario = when {
      tokens.size == 1 && stubMatcher.matches(tokens[0]) -> Scenario.STUB
      HOLE_MARKER in tokens -> Scenario.COMPLETION
      !hasHole && tokens in cfg.language -> Scenario.PARSEABLE
      else -> Scenario.REPAIR
    }

    runningJob = MainScope().launch {
      when (scenario) {
        Scenario.STUB -> cfg.enumNTSmall(tokens[0].stripStub())
        Scenario.COMPLETION -> cfg.enumSeqSmart(tokens)
        Scenario.PARSEABLE -> {
          val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
          writeDisplayText("$parsedPrefix$parseTree".also { cache[workHash] = it }); null
        }
        Scenario.REPAIR -> sampleGREUntilTimeout(tokens, cfg)
        else -> sequenceOf<Σᐩ>().also { println("Unhandled scenario: $scenario") }
      }?.let { candidates ->
        val metric = levAndLenMetric(tokens)
        val originalText = tokens.joinToString(" ")
        candidates.enumerateInteractively(
          workHash = workHash,
          keyOf = { it },
          metric = { metric(it.tokenizeByWhitespace()) },
          customDiff = { levenshteinAlign(originalText, it).paintDiffs() },
          reason = scenario.reason
        )
      }
    }
  }

  enum class Scenario(val reason: String, var data: List<Int>? = null) {
    STUB(stubGenPrefix), COMPLETION(holeGenPrefix),
    PARSEABLE(parsedPrefix), REPAIR(invalidPrefix),
    SUFFIX_COMPLETION(fwdCplPrefix);

    operator fun invoke(d: List<Int>): Scenario = apply { data = d }
  }

  protected suspend fun <T, K> Sequence<T>.enumerateInteractively(
    workHash: Int,
    keyOf: (T) -> K,
    metric: (T) -> Int,
    customDiff: suspend (T) -> String,
    resultsToPost: Int = MAX_DISP_RESULTS,
    timer: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow(),
    shouldContinue: () -> Boolean = { currentWorkHash == workHash && timer.hasTimeLeft() },
    postCompletionSummary: () -> String = { "." },
    reason: String = "Generic completions:\n\n"
  ) {
    val results = mutableSetOf<K>()
    val topResults = mutableListOf<Pair<T, Int>>()
    val iter = iterator()
    val startTime = TimeSource.Monotonic.markNow()

    while (true) {
      pause()
      if (!shouldContinue() || !iter.hasNext()) break
      val candidate = iter.next()
      val key = keyOf(candidate)
      if ((key is String && key.isEmpty()) || !results.add(key)) continue

      val score = metric(candidate)
      if (topResults.size < resultsToPost || score < topResults.last().second) {
        val location = topResults.binarySearch { it.second.compareTo(score) }
        topResults.add(if (location < 0) -location - 1 else location, candidate to score)
        if (topResults.size > resultsToPost) topResults.removeLast()
      }
    }

    if (currentWorkHash != workHash) return
    val throughput = (results.size / (startTime.elapsedNow().toDouble(SECONDS) + 0.001)).round(3)
    val moreResults = (results.size - topResults.size)
      .let { if (it == 0) "\n\n" else "\n\n...$it more, " }
    val renderedResults = coroutineScope {
      topResults.mapIndexed { index, (candidate, _) ->
        async {
          val result = "<span class=\"result-index\">${index.toString().padStart(2)}.) </span>${customDiff(candidate)}"
          if (index == 0) "<mark>$result</mark>" else result
        }
      }.awaitAll()
    }
    val summary = "$moreResults~$throughput res/s${postCompletionSummary()}"
    renderedResults.joinToString("\n", "", summary).let {
      writeDisplayText("$reason$it".also { rendered -> cache[workHash] = rendered })
    }
  }

  fun caretInGrammar(): Boolean =
    readEditorText().indexOf("---").let { it == -1 || getCaretPosition().start < it }

  open fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ = l2.joinToString(" ")
  open fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ> = TODO()
  open fun redecorateLines(cfg: CFG = setOf()) {}

  /** See: [JSTidyEditor.continuation] */
  open fun continuation(f: () -> Unit): Any = { f() }

  fun getGrammarText(): Σᐩ = readEditorText().substringBefore("---")
  fun getExampleText(): Σᐩ = readEditorText().substringAfter("---")

  fun currentGrammar(): CFG = try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

  fun currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()
}
