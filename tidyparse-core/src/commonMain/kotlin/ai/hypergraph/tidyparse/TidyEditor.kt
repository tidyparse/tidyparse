package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.minimizeFix
import org.kosat.round
import kotlin.math.absoluteValue
import kotlin.time.*
import kotlin.time.DurationUnit.SECONDS

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

abstract class TidyEditor {
  // TODO: eliminate this completely
  var cfg: CFG = setOf()
  var grammarFileCache: String = ""
  var cache = mutableMapOf<Int, String>()
  var currentWorkHash = 0
  var minimize = false
  var ntStubs = true
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
          .parseCFG(validate = true)
          .let { if (ntStubs) it else it.noNonterminalStubs }
          .also { cfg = it }
      } else cfg
    } catch (e: Exception) {
      writeDisplayText("<html><body><pre>${e.message!!}</pre></body></html>")
      emptySet()
    }
  }

  open fun handleInput() {
    val timer = TimeSource.Monotonic.markNow()

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

    fun finally(it: String, action: String = "Completed") {
      if (currentWorkHash == workHash)
        writeDisplayText("$invalidPrefix$it".also { cache[workHash] = it })
      println("$action in ${timer.elapsedNow().inWholeMilliseconds}ms")
    }
    fun shouldContinue() = currentWorkHash == workHash && timer.hasTimeLeft()

    fun rankingFun(l: List<String>): Int = levenshtein(tokens, l) * 7919 +
      (tokens.sumOf { it.length } - l.sumOf { it.length }).absoluteValue

    return if (HOLE_MARKER in tokens) {
      cfg.enumSeqSmart(tokens)
        .enumerateCompletionsInteractively(
          metric = ::rankingFun,
          shouldContinue = ::shouldContinue,
          finally = ::finally,
          localContinuation = ::continuation
        )
    }
    else if (!containsUnk && tokens in cfg.language) {
      val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
      writeDisplayText("$parsedPrefix$parseTree".also { cache[workHash] = it })
    }
    else FSA.intersectPTree(currentLine, cfg, FSA.LED(cfg, currentLine))
      ?.sampleStrWithoutReplacement()
//    else cfg
//      .barHillelRepair(tokens) // TODO: fix delay and replace fastRepairSeq
//      .fasterRepairSeq(abstractUnk, minimize = minimize)
      ?.enumerateCompletionsInteractively(
        metric = ::rankingFun,
        shouldContinue = ::shouldContinue,
        finally = ::finally,
        localContinuation = ::continuation
      ) ?: Unit
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
    var totalResults = 0

    fun findNextCompletion() {
      var i = 0
      if (!iter.hasNext() || !shouldContinue()) {
        val throughput = (results.size /
            startTime.elapsedNow().toDouble(SECONDS)).round(3)
        val throughputTot = (totalResults /
            startTime.elapsedNow().toDouble(SECONDS)).round(3)
        val summary = if (throughput != throughputTot)
          "~$throughput unique res/s, ~$throughputTot total res/s"
        else "~$throughput res/s"
        val moreResults = (results.size - topNResults.size)
          .let { if (it == 0) "\n\n" else "\n\n...$it more" }
        val statistics = "$moreResults $summary."
        return finally(topNResults.joinToString("\n", "", statistics) {
          "<span style=\"color: gray\" class=\"noselect\">${i++.toString().padStart(2)}.) </span>${it.first}"
        })
      }

      val next = iter.next()
      totalResults++
      if (next.isNotEmpty() && next !in results) {
        println("Found: $next")
        results.add(next)
        val score = metric(next.tokenizeByWhitespace())
        if (topNResults.size < resultsToPost || score < topNResults.last().second) {
          val currentLine = currentLine()
          val html = levenshteinAlign(currentLine, next).paintDiffs()
          val loc = topNResults.binarySearch { it.second.compareTo(score) }
          val idx = if (loc < 0) { -loc - 1 } else loc
          topNResults.add(idx, html to score)
          if (topNResults.size > resultsToPost) topNResults.removeLast()
          postResults(topNResults.joinToString("\n") {
            "<span style=\"color: gray\" class=\"noselect\">${i++.toString().padStart(2)}.) </span>${it.first}"
          })
        }
      }

      localContinuation(::findNextCompletion)
    }

    findNextCompletion()
  }

  open fun continuation(f: () -> Unit): Any = { f() }

  fun getGrammarText(): Σᐩ = readEditorText().substringBefore("---")
  fun getExampleText(): Σᐩ = readEditorText().substringAfter("---")

  fun currentGrammar(): CFG =
    try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

  fun currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()
}