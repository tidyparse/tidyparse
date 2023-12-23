package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import org.kosat.round
import kotlin.math.absoluteValue
import kotlin.time.*
import kotlin.time.DurationUnit.SECONDS
abstract class TidyEditor {
  // TODO: eliminate this completely
  var cfg: CFG = setOf()
  var grammarFileCache: String = ""
  val synthCache = LRUCache<Pair<String, CFG>, List<String>>()
  var cache = mutableMapOf<Int, String>()
  var currentWorkHash = 0
  val toTake = 27

  abstract fun readDisplayText(): Σᐩ
  abstract fun readEditorText(): Σᐩ
  abstract fun getCaretPosition(): Int
  abstract fun currentLine(): Σᐩ
  abstract fun writeDisplayText(s: Σᐩ)
  abstract fun writeDisplayText(s: (Σᐩ) -> Σᐩ)
  fun getLatestCFG(): CFG {
    val grammar: String = getGrammarText()
    return try {
      if (grammar != grammarFileCache || cfg.isNotEmpty()) {
        grammar.also { grammarFileCache = it }
          .parseCFG(validate = true).freeze().also { cfg = it }
      } else cfg
    } catch (e: Exception) { cfg }
  }

  open fun handleInput() {
    val timer = TimeSource.Monotonic.markNow()

    val currentLine = currentLine()
    if (currentLine.isBlank()) return
    val caretInGrammar = caretInGrammar()
    val tokens = currentLine.tokenizeByWhitespace()

    val cfg =
      (if (caretInGrammar)
        CFGCFG(names = tokens.filter { it !in setOf("->", "|") }.toSet())
      else getLatestCFG()).freeze()

    if (cfg.isEmpty()) return

    val sanitized = tokens.joinToString(" ")
    val workHash = sanitized.hashCode() + cfg.hashCode()
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    fun finally(it: String, action: String = "Completed") {
      if (currentWorkHash == workHash)
        writeDisplayText("$invalidPrefix$it".also { cache[workHash] = it })
      println("$action in ${timer.elapsedNow().inWholeMilliseconds}ms")
    }
    fun shouldContinue() = currentWorkHash == workHash && timer.hasTimeLeft()

    return if (sanitized.containsHole()) {
      cfg.enumSeqSmart(tokens).distinct()
        .enumerateCompletionsInteractively(
          metric = {
            levenshtein(tokens, it) * 7919 +
                (tokens.sumOf { it.length } - it.sumOf { it.length }).absoluteValue
          },
          shouldContinue = ::shouldContinue,
          finally = ::finally,
          localContinuation = ::continuation
        )
    }
    else if (tokens in cfg.language) {
      val parseTree = cfg.parse(sanitized)?.prettyPrint()
      writeDisplayText("$parsedPrefix$parseTree")
    }
    else cfg.fastRepairSeq(tokens)
      .enumerateCompletionsInteractively(
        metric = {
          levenshtein(tokens, it) * 7919 +
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
  open fun redecorateLines(cfg: CFG = setOf()) {}

  fun Sequence<String>.enumerateCompletionsInteractively(
    resultsToPost: Int = toTake,
    metric: (List<String>) -> Int,
    shouldContinue: () -> Boolean,
    postResults: (String) -> Unit = { writeDisplayText("$invalidPrefix$it") },
    finally: (String) -> Unit = { postResults(it) },
    localContinuation: (() -> Unit) -> Any = { it() }
  ) {
    val results = mutableSetOf<String>()
    val topNResults = mutableListOf<Pair<String, Int>>()
    val iter = iterator()
    val startTime = TimeSource.Monotonic.markNow()

    fun findNextCompletion() {
      var i = 0
      if (!iter.hasNext() || !shouldContinue()) {
        val throughput = (results.size /
            startTime.elapsedNow().toDouble(SECONDS)).round(3)
        val moreResults = (results.size - topNResults.size)
          .let { if (it == 0) "\n\n" else "\n\n...$it more" }
        val statistics = "$moreResults ~$throughput res/s."
        return finally(topNResults.joinToString("\n", "", statistics) {
          "${i++.toString().padStart(2)}.) ${it.first}"
        })
      }

      val next = iter.next()
      println("Found: $next")
      val isNew = next !in results
      if (next.isNotEmpty() && isNew) {
        results.add(next)
        val score = metric(next.tokenizeByWhitespace())
        if (topNResults.size < resultsToPost || score < topNResults.last().second) {
          val loc = topNResults.binarySearch { it.second.compareTo(score) }
          val idx = if (loc < 0) { -loc - 1 } else loc
          topNResults.add(idx, next to score)
          if (topNResults.size > resultsToPost) topNResults.removeLast()
          postResults(topNResults.joinToString("\n") { "${i++.toString().padStart(2)}.) ${it.first}" })
        }
      }

      localContinuation(::findNextCompletion)
    }

    findNextCompletion()
  }

  open fun continuation(f: () -> Unit): Any = { f() }

  fun getGrammarText(): Σᐩ = readEditorText().split("---").first()
  fun getExampleText(): Σᐩ = readEditorText().split("---").last()

  fun currentGrammar(): CFG =
    try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

  fun currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()
}