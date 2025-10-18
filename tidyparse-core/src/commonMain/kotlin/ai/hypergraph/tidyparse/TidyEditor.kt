package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.LED_BUFFER
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import ai.hypergraph.kaliningraph.repair.minimizeFix
import kotlinx.coroutines.*
import kotlin.math.absoluteValue
import kotlin.time.TimeSource

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

abstract class TidyEditor {
  // TODO: eliminate this completely
  open var cfg: CFG = setOf()
  var grammarFileCache: String = ""
  var cache = mutableMapOf<Int, String>()
  var currentWorkHash = 0
  var minimize = false
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
    return try {
      if (grammar != grammarFileCache || cfg.isEmpty()) {
        grammar.also { grammarFileCache = it }
          .parseCFG(validate = true)
          .let { if (ntStubs) it else it.noNonterminalStubs }
          .also { cfg = it }
      } else cfg
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

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, minimize, ntStubs).hashCode()
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
      }?.enumerateInteractively(workHash, tokens, reason = scenario.reason)
    }
  }

  enum class Scenario(val reason: String) {
    STUB(stubGenPrefix), COMPLETION(holeGenPrefix),
    PARSEABLE(parsedPrefix), REPAIR(invalidPrefix)
  }

  protected suspend fun Sequence<String>.enumerateInteractively(
    workHash: Int,
    origTks: List<String>,
    timer: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow(),
    metric: (List<String>) -> Int = { levenshtein(origTks, it) * 7919 +
        (origTks.sumOf { it.length } - it.sumOf { it.length }).absoluteValue },
    shouldContinue: () -> Boolean = { currentWorkHash == workHash && timer.hasTimeLeft() },
    customDiff: (String) -> String = { levenshteinAlign(origTks.joinToString(" "), it).paintDiffs() },
    recognizer: (String) -> Boolean = { it in cfg.language },
    postCompletionSummary: () -> String = { "." },
    reason: String = "Generic completions:\n\n"
  ) = let {
    if (!minimize || "_" in origTks) it
    else it.flatMap { minimizeFix(origTks, it.tokenizeByWhitespace()) { recognizer(this) } }
  }.enumerateCompletionsInteractively(
    metric = metric,
    shouldContinue = shouldContinue,
    postResults = { writeDisplayText("$invalidPrefix$it") },
    finally = {
      if (currentWorkHash == workHash) writeDisplayText("$reason$it".also { cache[workHash] = it })
      println("Enumeration completed in ${timer.elapsedNow().inWholeMilliseconds}ms")
    },
    customDiff = customDiff,
    postCompletionSummary = postCompletionSummary
  )

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