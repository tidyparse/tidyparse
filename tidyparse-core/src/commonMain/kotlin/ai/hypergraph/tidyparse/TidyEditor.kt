package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.parsing.contains
import ai.hypergraph.kaliningraph.repair.MAX_RADIUS
import ai.hypergraph.kaliningraph.repair.minimizeFix
import kotlinx.coroutines.delay
import org.kosat.round
import kotlin.math.absoluteValue
import kotlin.time.*
import kotlin.time.DurationUnit.SECONDS
import kotlinx.coroutines.*
import kotlin.time.Duration.Companion.nanoseconds

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
  abstract fun getCaretPosition(): Int
  abstract fun currentLine(): Σᐩ
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

  open fun handleInput() {
    val currentLine = currentLine().also { println("Current line is: $it") }
    if (currentLine.isBlank()) return
    val caretInGrammar = caretInGrammar()
    val tokens = currentLine.tokenizeByWhitespace()

    val cfg =
      if (caretInGrammar)
        CFGCFG(names = tokens.filter { it !in setOf("->", "|") }.toSet())
      else getLatestCFG()

    if (cfg.isEmpty()) return

    var containsUnk = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnk = true; "_" } }

    val workHash = abstractUnk.hashCode() + cfg.hashCode()
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    runningJob?.cancel()

    /* Completion */ if (HOLE_MARKER in tokens) {
      cfg.enumSeqSmart(tokens).enumerateInteractively(workHash, tokens)
    } else /* Parseable */ if (!containsUnk && tokens in cfg.language) {
      val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
      writeDisplayText("$parsedPrefix$parseTree".also { cache[workHash] = it })
    } else /* Repair */ Unit.also {
      runningJob = MainScope().launch {
        initiateSuspendableRepair(tokens, cfg).enumerateInteractively(workHash, tokens)
      }
    }
  }

  protected fun Sequence<String>.enumerateInteractively(
    workHash: Int,
    tokens: List<String>,
    timer: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow(),
    metric: (List<String>) -> Int = { levenshtein(tokens, it) * 7919 +
        (tokens.sumOf { it.length } - it.sumOf { it.length }).absoluteValue},
    shouldContinue: () -> Boolean = { currentWorkHash == workHash && timer.hasTimeLeft() },
    customDiff: (String) -> String = { levenshteinAlign(tokens.joinToString(" "), it).paintDiffs() }
  ) = this.let {
    if (!minimize) it
    else it.flatMap { minimizeFix(tokens, it.tokenizeByWhitespace()) { this in cfg.language } }
  }.enumerateCompletionsInteractively(
    metric = metric,
    shouldContinue = shouldContinue,
    postResults = { writeDisplayText("$invalidPrefix$it") },
    localContinuation = ::continuation,
    finally = {
      if (currentWorkHash == workHash)
        writeDisplayText("$invalidPrefix$it".also { cache[workHash] = it })
      println("Completed in ${timer.elapsedNow().inWholeMilliseconds}ms")
    },
    customDiff = customDiff
  )

  fun caretInGrammar(): Boolean =
    readEditorText().indexOf("---")
      .let { it == -1 || getCaretPosition() < it }

  open fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ = l2.joinToString(" ")
  abstract fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ>
  open fun redecorateLines(cfg: CFG = setOf()) {}

  /** See: [JSTidyEditor.continuation] */
  open fun continuation(f: () -> Unit): Any = { f() }

  fun getGrammarText(): Σᐩ = readEditorText().substringBefore("---")
  fun getExampleText(): Σᐩ = readEditorText().substringAfter("---")

  fun currentGrammar(): CFG =
    try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

  fun currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()
}