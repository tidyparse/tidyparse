package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.parsing.contains
import ai.hypergraph.kaliningraph.repair.MAX_RADIUS
import ai.hypergraph.kaliningraph.repair.minimizeFix
import ai.hypergraph.kaliningraph.types.to
import kotlinx.coroutines.delay
import org.kosat.round
import kotlin.math.absoluteValue
import kotlin.time.*
import kotlin.time.DurationUnit.SECONDS
import kotlinx.coroutines.*
import kotlin.time.Duration.Companion.microseconds
import kotlin.time.Duration.Companion.nanoseconds

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

abstract class TidyEditor {
  // TODO: eliminate this completely
  var cfg: CFG = setOf()
  var grammarFileCache: String = ""
  var cache = mutableMapOf<Int, String>()
  var currentWorkHash = 0
  var minimize = false
  var ntStubs = true
  val toTake = 28

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

  private suspend fun initiateSuspendableRepair(
    brokenStr: List<Σᐩ>, cfg: CFG,
    metric: (List<String>) -> Int,
    shouldContinue: () -> Boolean,
    finally: (String) -> Unit,
  ) {
    var i = 0
    val upperBound = MAX_RADIUS * 2
    val monoEditBounds = cfg.maxParsableFragmentB(brokenStr, pad = upperBound)
    val bindex = cfg.bindex
    val bimap = cfg.bimap
    val prods = cfg.tripleIntProds
    val width = cfg.nonterminals.size
    suspend fun pause(freq: Int = 100_000) { if (i++ % freq == 0) { delay(100.nanoseconds) }}

    suspend fun nonemptyLevInt(cfg: CFG, levFSA: FSA): Boolean {
      val ap: Map<Pair<Int, Int>, Set<Int>> = levFSA.allPairs
      val dp = Array(levFSA.numStates) { Array(levFSA.numStates) { BooleanArray(width) { false } } }

      levFSA.allIndexedTxs0(cfg).forEach { (q0, nt, q1) -> dp[q0][q1][nt] = true }

      val startIdx = bindex[START_SYMBOL]

      // For pairs (p,q) in topological order
      for (dist in 0 until levFSA.numStates) {
        for (iP in 0 until levFSA.numStates - dist) {
          val p = iP
          val q = iP + dist
          if (p to q !in levFSA.allPairs) continue
          for ((A, /*->*/ B, C) in prods) {
            if (!dp[p][q][A]) {
              // Check possible midpoints r in [p+1, q-1]
              // or in general, r in levFSA.allPairs[p->q]
              for (r in ap[p to q]!!) {
                pause()
                if (dp[p][r][B] && dp[r][q][C]) {
                  if (p == 0 && A == startIdx && q in levFSA.finalIdxs) return true
                  dp[p][q][A] = true
                  break
                }
              }
            }
          }
        }
      }

      return false
    }

    val radius = (2 until upperBound).firstOrNull {
      nonemptyLevInt(cfg, makeLevFSA(brokenStr, it, monoEditBounds))
    } ?: upperBound

    val levFSA = makeLevFSA(brokenStr, radius + 1, monoEditBounds)

    val nStates = levFSA.numStates
    val startIdx = bindex[START_SYMBOL]

    // 1) Create dp array of parse trees
    val dp: Array<Array<Array<PTree?>>> = Array(nStates) { Array(nStates) { Array(width) { null } } }

    // 2) Initialize terminal productions A -> a
    for ((p, σ, q) in levFSA.allIndexedTxs1(cfg)) {
      val Aidxs = bimap.TDEPS[σ]!!.map { bindex[it] }
      for (Aidx in Aidxs) {
        pause()
        if (!shouldContinue()) return
        val newLeaf = PTree(root = bindex[Aidx], branches = PSingleton(σ))
        dp[p][q][Aidx] = newLeaf + dp[p][q][Aidx]
      }
    }

    // 3) CYK + Floyd Warshall parsing
    for (dist in 0 until nStates) {
      for (p in 0 until (nStates - dist)) {
        val q = p + dist
        if (p to q !in levFSA.allPairs) continue

        for (r in levFSA.allPairs[p to q]!!) {
          for ((Aidx, /*->*/ Bidx, Cidx) in prods) {
          // Check all possible midpoint states r in the DAG from p to q
            pause()
            val left = dp[p][r][Bidx]
            val right = dp[r][q][Cidx]
            if (left != null && right != null) {
              // Found a parse for A
              val newTree = PTree(bindex[Aidx], listOf(left to right))
              dp[p][q][Aidx] = newTree + dp[p][q][Aidx]
            }
          }
        }
      }
    }

    // 4) Gather final parse trees from dp[0][f][startIdx], for all final states f
    val allParses = levFSA.finalIdxs.mapNotNull { q -> dp[0][q][startIdx] }

    // 5) Combine them under a single "super‐root"
    PTree(START_SYMBOL, allParses.flatMap { forest -> forest.branches })
//      .toCFG.also { println("CFG Size: ${it.size}") }.toPTree().also { println("Words: ${it.totalTreesStr}") }
      .sampleStrWithoutReplacement()
      .let {
        if (!minimize) it
        else it.flatMap { minimizeFix(brokenStr, it.tokenizeByWhitespace()) { this in cfg.language } }
      }
      .enumerateCompletionsInteractively(
        metric = metric,
        shouldContinue = shouldContinue,
        finally = finally,
        localContinuation = ::continuation
      )
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
    else Unit.also {
      MainScope().launch {
        initiateSuspendableRepair(
          tokens, cfg,
          metric = ::rankingFun,
          shouldContinue = ::shouldContinue,
          finally = ::finally,
        )
      }
    }
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
          val result = "<span style=\"color: gray\" class=\"noselect\">${i++.toString().padStart(2)}.) </span>${it.first}"
          if (i == 1) "<mark>$result</mark>" else result
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