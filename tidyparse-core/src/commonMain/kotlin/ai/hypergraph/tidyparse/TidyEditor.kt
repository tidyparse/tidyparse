package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import kotlin.math.absoluteValue
import kotlin.time.TimeSource

// TODO: eliminate this completely
var cfg: CFG = setOf()
var grammarFileCache: String = ""
val synthCache = LRUCache<Pair<String, CFG>, List<String>>()
var cache = mutableMapOf<Int, String>()
var currentWorkHash = 0
val toTake = 30

interface TidyEditor {
  fun readDisplayText(): Σᐩ
  fun readEditorText(): Σᐩ
  fun getCaretPosition(): Int
  fun currentLine(): Σᐩ
  fun writeDisplayText(s: Σᐩ)
  fun writeDisplayText(s: (Σᐩ) -> Σᐩ)
  fun getLatestCFG(): CFG {
    val grammar: String = readEditorText().substringBefore("---")
    return if (grammar != grammarFileCache || cfg.isNotEmpty()) {
      grammarFileCache = grammar
      grammarFileCache.parseCFG().also { cfg = it }
    } else cfg
  }

  fun handleInput() {
    val timer = TimeSource.Monotonic.markNow()
    val cfg = getLatestCFG()
    if (caretInGrammar()) { return }
    val line = currentLine()
    val tokens = line.tokenizeByWhitespace()
    val lineHash = tokens.hashCode()
    currentWorkHash = lineHash

    fun finally(it: String, action: String = "Completed") {
      cache[lineHash] = it
      writeDisplayText(it)
      println("$action in ${timer.elapsedNow().inWholeMilliseconds}")
    }
    fun shouldContinue() = currentWorkHash == lineHash && timer.hasTimeLeft()

    if (lineHash in cache) writeDisplayText(cache[lineHash]!!)
    else if ("_" in line) {
      writeDisplayText("Solving: $line\n".also { print(it) })

      cfg.enumSeqSmart(line.tokenizeByWhitespace()).distinct()
        .enumerateCompletionsInteractively(
          metric = { it.size * 7919 + it.sumOf { it.length } },
          shouldContinue = ::shouldContinue,
          finally = ::finally,
          localContinuation = ::continuation
        )
    }
    else if (cfg.parse(line)?.prettyPrint()?.also { writeDisplayText("$line\n\n$it") } != null) { }
    else {
      writeDisplayText("Repairing: $line\n".also { print(it) })

      cfg.fastRepairSeq(tokens)
        .enumerateCompletionsInteractively(
          metric = { levenshtein(tokens, it) * 7919 +
              (tokens.sumOf { it.length } - it.sumOf { it.length }).absoluteValue
          },
          shouldContinue = ::shouldContinue,
          finally = ::finally,
          localContinuation = ::continuation
        )
    }
  }

  fun caretInGrammar(): Boolean =
    readEditorText().indexOf("---")
      .let { it == -1 || getCaretPosition() < it }
  fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ = l2.joinToString(" ")
  fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ>
  fun redecorateLines(cfg: CFG) {}

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
      println("Found: ${next}")
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

  fun continuation(f: () -> Unit): Any = { f() }

  fun getGrammarText(): Σᐩ = readEditorText().substringBefore("---")

  fun currentGrammar(): CFG =
    try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

  fun currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()
}