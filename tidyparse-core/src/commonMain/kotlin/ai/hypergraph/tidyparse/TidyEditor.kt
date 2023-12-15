package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import kotlin.math.absoluteValue
import kotlin.time.TimeSource

abstract class TidyEditor {
  // TODO: eliminate this completely
  var cfg: CFG = setOf()
  var grammarFileCache: String = ""
  val synthCache = LRUCache<Pair<String, CFG>, List<String>>()
  var cache = mutableMapOf<Int, String>()
  var currentWorkHash = 0
  val toTake = 30

  abstract fun readDisplayText(): Σᐩ
  abstract fun readEditorText(): Σᐩ
  abstract fun getCaretPosition(): Int
  abstract fun currentLine(): Σᐩ
  abstract fun writeDisplayText(s: Σᐩ)
  abstract fun writeDisplayText(s: (Σᐩ) -> Σᐩ)
  fun getLatestCFG(): CFG {
    val grammar: String = readEditorText().substringBefore("---")
    return if (grammar != grammarFileCache || cfg.isNotEmpty()) {
      grammarFileCache = grammar
      grammarFileCache.parseCFG().also { cfg = it }
    } else cfg
  }

  open fun handleInput() {
    val timer = TimeSource.Monotonic.markNow()
    val cfg = getLatestCFG()
    if (caretInGrammar()) { return }
    val line = currentLine()
    val tokens = line.tokenizeByWhitespace()
    val sanitized = tokens.joinToString(" ")
    val lineHash = tokens.hashCode() + grammarFileCache.hashCode()
    currentWorkHash = lineHash

    fun finally(it: String, action: String = "Completed") {
      cache[lineHash] = it
      writeDisplayText(it)
      println("$action in ${timer.elapsedNow().inWholeMilliseconds}ms")
    }
    fun shouldContinue() = currentWorkHash == lineHash && timer.hasTimeLeft()

    if (lineHash in cache) return writeDisplayText(cache[lineHash]!!)

    if (line.containsHole()) {
      writeDisplayText("Solving: $line\n".also { print(it) })

      return cfg.enumSeqSmart(tokens).distinct()
        .enumerateCompletionsInteractively(
          metric = { it.size * 7919 + it.sumOf { it.length } },
          shouldContinue = ::shouldContinue,
          finally = ::finally,
          localContinuation = ::continuation
        )
    }

    val q = cfg.parse(sanitized)?.prettyPrint()

    if (q != null) return writeDisplayText("$line\n\n$q")

//    val renderedStubs = cfg.parseWithStubs(sanitized).second.renderStubs()
//    writeDisplayText(render(cfg, emptyList(), this, stubs = renderedStubs, reason = no))
    writeDisplayText("Repairing: $line\n".also { print(it) })

    return cfg.fastRepairSeq(tokens)
      .enumerateCompletionsInteractively(
        metric = { levenshtein(tokens, it) * 7919 +
          (tokens.sumOf { it.length } - it.sumOf { it.length }).absoluteValue
        },
        shouldContinue = ::shouldContinue,
        finally = ::finally,
        localContinuation = ::continuation
      )
  }

  fun caretInGrammar(): Boolean =
    readEditorText().indexOf("---")
      .let { it == -1 || getCaretPosition() < it }
  open fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ = l2.joinToString(" ")
  abstract fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ>
  open fun redecorateLines(cfg: CFG) {}

  fun Sequence<String>.enumerateCompletionsInteractively(
    resultsToPost: Int = toTake,
    metric: (List<String>) -> Int,
    shouldContinue: () -> Boolean,
    postResults: (String) -> Unit = { writeDisplayText(it) },
    finally: (String) -> Unit = { postResults(it) },
    localContinuation: (() -> Unit) -> Any = { it() }
  ) {
    val results = mutableSetOf<String>()
    val topNResults = mutableListOf<Pair<String, Int>>()
    val iter = iterator()

    fun findNextCompletion() {
      if (!iter.hasNext() || !shouldContinue())
        return finally(topNResults.joinToString("\n", "", "\n...") { it.first })

      val next = iter.next()
      println("Found: $next")
      val isNew = next !in results
      if (next.isNotEmpty() && isNew) {
        results.add(next)
        if (topNResults.size < resultsToPost || next.length < topNResults.last().second) {
          val score = metric(next.tokenizeByWhitespace())
          val loc = topNResults.binarySearch { it.second.compareTo(score) }
          val idx = if (loc < 0) { -loc - 1 } else loc
          topNResults.add(idx, next to score)
          if (topNResults.size > resultsToPost) topNResults.removeLast()
          postResults(topNResults.joinToString("\n") { it.first })
        }
      }

      localContinuation(::findNextCompletion)
    }

    findNextCompletion()
  }

  open fun continuation(f: () -> Unit): Any = { f() }

  fun getGrammarText(): Σᐩ = readEditorText().substringBefore("---")

  fun currentGrammar(): CFG =
    try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

  fun currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()
}