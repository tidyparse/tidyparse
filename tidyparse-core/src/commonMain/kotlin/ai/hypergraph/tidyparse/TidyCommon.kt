package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.image.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.*
import kotlinx.coroutines.delay
import org.kosat.round
import kotlin.math.ceil
import kotlin.math.max
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.time.DurationUnit.SECONDS
import kotlin.time.TimeSource

val CFG.renderedHTML by cache { renderCFGToHTML() }

fun CFG.renderCFGToHTML(tokens: Set<Σᐩ> = emptySet()): Σᐩ =
  (listOf(originalForm.summarize("Original form")) +
      (if (originalForm == nonparametricForm) listOf()
      else listOf(nonparametricForm.summarize("Nonparametric form"))) +
      listOf(summarize("Normal form"))
//      upwardClosure(tokens).let { closure ->
//        if (closure.size == size) listOf()
//        else listOf(closure.summarize("Upward closure")) +
//        listOf(filter { it.LHS !in closure.nonterminals }.summarize("Filtered"))
//      }
  )
  .let { rewriteSummary ->
    val maxLen = rewriteSummary.joinToString("\n").lines().maxOf { it.length }
    rewriteSummary.joinToString(delim(maxLen), "<pre>${delim(maxLen)}", "</pre>")
  }

fun CFG.summarize(name: Σᐩ): Σᐩ = "<b>$name</b> (" +
    "${nonterminals.size} nonterminal${if (1 < nonterminals.size) "s" else ""} / " +
    "${terminals.size} terminal${if (1 < terminals.size) "s" else ""} / " +
    "$size production${if (1 < size) "s" else ""})\n$prettyHTML"

fun delim(len: Int = 120) = List(len) { "─" }.joinToString("", "\n", "\n")

val CFG.prettyHTML by cache { prettyPrint().carveSeams().escapeHTML() }

// Determines whether a substitution is invariant w.r.t. NT membership
fun CFG.preservesNTInvariance(newNT: Σᐩ, oldTerminal: Σᐩ) = newNT in bimap[listOf(oldTerminal)]

val la = "<".escapeHTML()
val ra = ">".escapeHTML()
fun Σᐩ.treatAsNonterminal() = drop(la.length).dropLast(ra.length)

fun Σᐩ.dehtmlify(): Σᐩ =
  replace("&lt;", "<")
    .replace("&gt;", ">")
    .replace("&amp;", "&")
    .replace("&quot;", "\"")
    .replace("&apos;", "'")
    .replace("<span.*?>".toRegex(), "")
    .replace("</span>", "")

// Binary search for the max parsable fragment. Equivalent to the linear search, but faster
suspend fun CFG.maxParsableFragmentB(tokens: List<Σᐩ>, pad: Int = 3): Pair<Int, Int> {
  suspend fun <T> List<T>.binSearch(fromIndex: Int = 0, toIndex: Int = size, comparison: suspend (T) -> Int): Int {
    var low = fromIndex
    var high = toIndex - 1

    while (low <= high) {
      val mid = (low + high).ushr(1) // safe from overflows
      val midVal = get(mid)
      val cmp = comparison(midVal)

      if (cmp < 0)
        low = mid + 1
      else if (cmp > 0)
        high = mid - 1
      else
        return mid // key found
    }
    return -(low + 1)  // key not found
  }

  val boundsTimer = TimeSource.Monotonic.markNow()
  val monoEditBounds = ((1..tokens.size).toList().binSearch { i ->
    delay(100.nanoseconds)
    val blocked = blockForward(tokens, i, pad)
    val blockedInLang = blocked in language
    if (blockedInLang) -1 else {
      val blockedPrev = blockForward(tokens, i - 1, pad)
      val blockedPrevInLang = i == 1 || blockedPrev in language
      if (!blockedInLang && blockedPrevInLang) 0 else 1
    }
  }.let { if (it < 0) tokens.size else it + 1 }) to ((2..tokens.size).toList().binSearch { i ->
    delay(100.nanoseconds)
    val blocked = blockBackward(tokens, i, pad)
    val blockedInLang = blocked in language
    if (blockedInLang) -1 else {
      val blockedPrev = blockBackward(tokens, i - 1, pad)
      val blockedPrevInLang = i == 2 || blockedPrev in language
      if (!blockedInLang && blockedPrevInLang) 0 else 1
    }
  }.let { if (it < 0) 0 else (tokens.size - it - 2).coerceAtLeast(0) })

  val delta = monoEditBounds.run { second - first }.let { if (it < 0) "$it" else "+$it" }
  println("Mono-edit bounds (R=${monoEditBounds.first}, " +
      "L=${monoEditBounds.second})/${tokens.size} [delta=$delta] in ${boundsTimer.elapsedNow()}")

  return monoEditBounds
}

val MAX_DISP_RESULTS = 29

fun Sequence<Σᐩ>.enumerateCompletionsInteractively(
  resultsToPost: Int = MAX_DISP_RESULTS,
  metric: (List<Σᐩ>) -> Int,
  shouldContinue: () -> Boolean,
  postResults: (Σᐩ) -> Unit,
  finally: (Σᐩ) -> Unit = { postResults(it) },
  localContinuation: (() -> Unit) -> Any = { it() },
  customDiff: (String) -> String
) {
  val results = mutableSetOf<Σᐩ>()
  val topNResults = mutableListOf<Pair<Σᐩ, Int>>()
  val iter = iterator()
  val startTime = TimeSource.Monotonic.markNow()
  var totalResults = 0

  fun findNextCompletion() {
    var i = 0
    if (!iter.hasNext() || !shouldContinue()) {
      val throughput = (results.size /
          (startTime.elapsedNow().toDouble(SECONDS) + 0.001)).round(3)
      val throughputTot = (totalResults /
          (startTime.elapsedNow().toDouble(SECONDS) + 0.001)).round(3)
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
//      println("Found: $next")
      results.add(next)
      val score = metric(next.tokenizeByWhitespace())
      if (topNResults.size < resultsToPost || score < topNResults.last().second) {
        val html = customDiff(next)
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

suspend fun initiateSuspendableRepair(brokenStr: List<Σᐩ>, cfg: CFG): Sequence<Σᐩ> {
  var i = 0
  val upperBound = MAX_RADIUS * 2
  val monoEditBounds = cfg.maxParsableFragmentB(brokenStr, pad = upperBound)
  val timer = TimeSource.Monotonic.markNow()
  val bindex = cfg.bindex
  val width = cfg.nonterminals.size
  val vindex = cfg.vindex
  val ups = cfg.unitProductions
  val t2vs = cfg.tmToVidx
  val startIdx = bindex[START_SYMBOL]

  suspend fun pause(freq: Int = 300_000) { if (i++ % freq == 0) { delay(50.nanoseconds) }}

  suspend fun nonemptyLevInt(levFSA: FSA): Int? {
    val ap: List<List<List<Int>?>> = levFSA.allPairs
    val dp = Array(levFSA.numStates) { Array(levFSA.numStates) { BooleanArray(width) { false } } }

    levFSA.allIndexedTxs0(ups, bindex).forEach { (q0, nt, q1) -> dp[q0][q1][nt] = true }
    var min: Int = Int.MAX_VALUE

    // For pairs (p,q) in topological order
    for (dist: Int in 0 until levFSA.numStates) {
      for (iP: Int in 0 until levFSA.numStates - dist) {
        val p = iP
        val q = iP + dist
        if (ap[p][q] == null) continue
        val appq = ap[p][q]!!
        for ((A: Int, indexArray: IntArray) in vindex.withIndex()) {
          pause()
          outerloop@for(j: Int in 0..<indexArray.size step 2) {
            val B = indexArray[j]
            val C = indexArray[j + 1]
            for (r in appq)
              if (dp[p][r][B] && dp[r][q][C]) {
                dp[p][q][A] = true
                break@outerloop
              }
          }

          if (p == 0 && A == startIdx && q in levFSA.finalIdxs && dp[p][q][A])
            min = minOf(min, levFSA.idsToCoords[q]!!.second)
        }
      }
    }

    return if (min == Int.MAX_VALUE) null else min
  }

  val led = (3 until upperBound)
    .firstNotNullOfOrNull { nonemptyLevInt(makeLevFSA(brokenStr, it, monoEditBounds)) } ?: upperBound
  val radius = max(3, led) + LED_BUFFER

  println("Identified LED=$radius in ${timer.elapsedNow()}")

  val levFSA = makeLevFSA(brokenStr, radius, monoEditBounds)

  val nStates = levFSA.numStates
  val tms = cfg.tmLst.size
  val tmm = cfg.tmMap

  // 1) Create dp array of parse trees
  val dp: Array<Array<Array<GRE?>>> = Array(nStates) { Array(nStates) { Array(width) { null } } }

  // 2) Initialize terminal productions A -> a
  val aitx = levFSA.allIndexedTxs1(ups)
  for ((p, σ, q) in aitx) for (Aidx in t2vs[tmm[σ]!!])
    dp[p][q][Aidx] = ((dp[p][q][Aidx] as? GRE.SET) ?: GRE.SET(tms))
      .apply { pause(); s.set(tmm[σ]!!) }

  // 3) CYK + Floyd Warshall parsing
  for (dist in 0 until nStates) {
    for (p in 0 until (nStates - dist)) {
      val q = p + dist
      if (levFSA.allPairs[p][q] == null) continue
      val appq = levFSA.allPairs[p][q]!!
      for ((Aidx, indexArray) in vindex.withIndex()) {
//        println("${cfg.bindex[Aidx]}(${pm!!.ntLengthBounds[Aidx]}):${levFSA.stateLst[p]}-${levFSA.stateLst[q]}(${levFSA.SPLP(p, q)})")
        val rhsPairs = dp[p][q][Aidx]?.let { mutableListOf(it) } ?: mutableListOf()
        outerLoop@for (j in 0..<indexArray.size step 2) {
          pause()
          val Bidx = indexArray[j]
          val Cidx = indexArray[j + 1]
          for (r in appq) {
            val left = dp[p][r][Bidx]
            val right = dp[r][q][Cidx]
            if (left != null && right != null) {
              // Found a parse for A
              rhsPairs += left * right
//              if (rhsPairs.size > 10) break@outerLoop
            }
          }
        }

        if (rhsPairs.isNotEmpty()) dp[p][q][Aidx] = GRE.UNI(*rhsPairs.toTypedArray())
      }
    }
  }

  println("Completed parse matrix in: ${timer.elapsedNow()}")

  /* Too slow?
  // 4) Gather successful PTrees for each Levenshtein distance shell from LED to max radius
  //    and enumerate repairs in increasing order by Levenshtein distance
  val allParses = levFSA.finalIdxs.groupBy { levFSA.idsToCoords[it]!!.second }
    .let { distToFinalStates ->
      distToFinalStates.keys.sorted().map { dist ->
        distToFinalStates[dist]!!.mapNotNull { dp[0][it][startIdx]?.branches }.flatten().let {
          if (it.isEmpty()) emptySequence()
          else PTree(START_SYMBOL, it).sampleStrWithoutReplacement()
        }
      }.fold(emptySequence<Σᐩ>()) { acc, p -> acc + p }
    }.also { println("Took ${timer.elapsedNow()} parse for |A|=$nStates, |G|=${cfg.size}") }

  return allParses
   */

  // 4) Gather final parse trees from dp[0][f][startIdx], for all final states f
  val allParses = levFSA.finalIdxs.mapNotNull { q -> dp[0][q][startIdx] }

  // 5) Combine them under a single GRE
  return (if (allParses.isEmpty()) sequenceOf() else GRE.UNI(*allParses.toTypedArray()).words(cfg.tmLst))
    .also { println("Took ${timer.elapsedNow()} to parse with |σ|=${brokenStr.size}, |A|=$nStates, |G|=${cfg.size}") }
}

fun displayComparator(tokens: List<Σᐩ>): Comparator<Σᐩ> =
  compareBy(tokenwiseLevenshteinEdits(tokens)).thenBy { it.length }

fun tokenwiseLevenshteinEdits(tokens: List<Σᐩ>): (Σᐩ) -> Comparable<*> =
  { levenshtein(tokens.filterNot { it == "_" }, it.tokenizeByWhitespace()) }

fun List<Tree>.renderStubs(): Σᐩ =
  runningFold(setOf<Tree>()) { acc, t -> if (acc.any { t.span isSubsetOf it.span }) acc else acc + t }
    .last().sortedBy { it.span.first }
    .partition { it.terminal == null }
    .let { (branches, leaves) ->
      val (leafCols, branchCols) = 3 to 2
      "<pre>${delim()}<b>Parseable subtrees</b> (" +
          "${leaves.size} lea${if (leaves.size != 1) "ves" else "f"} / " +
          "${branches.size} branch${if (branches.size != 1) "es" else ""})</pre>\n\n" +
          leaves.mapIndexed { i, it -> "🌿\n└── " + it.prettyPrint().trim() }.let { asts ->
            FreeMatrix(ceil(asts.size.toDouble() / leafCols).toInt(), leafCols) { r, c ->
              if (r * leafCols + c < asts.size) asts[r * leafCols + c].ifBlank { "" } else ""
            }
          }.toHtmlTable() +
          branches.let { asts ->
            FreeMatrix(ceil(asts.size.toDouble() / branchCols).toInt(), branchCols) { r, c ->
              if (r * branchCols + c < asts.size)
                Tree("🌿", null, asts[r * branchCols + c], span = -1..-1)
                  .prettyPrint().ifBlank { "" } else ""
            }
          }.toHtmlTable()
    }

fun renderLite(
  solutions: List<Σᐩ>,
  editor: TidyEditor,
  reason: Σᐩ? = null,
  prompt: Σᐩ? = null,
  stubs: Σᐩ? = null,
  template: Σᐩ = prompt ?: editor.readDisplayText()
    .substringAfter("Solving: ").substringBefore("\n")
): Σᐩ = """
  <html>
  <body>
  <pre>${reason ?: "Synthesizing...\n"}
  """.trimIndent() +
    // TODO: legend
    solutions.joinToString("\n", "\n", "\n") + """🔍 Solving: $template
  
  ${if (reason != null) legend else ""}</pre>${stubs ?: ""}
  </body>
  </html>
  """.trimIndent()

fun render(
  cfg: CFG,
  solutions: List<Σᐩ>,
  editor: TidyEditor,
  reason: Σᐩ? = null,
  prompt: Σᐩ? = null,
  stubs: Σᐩ? = null,
  template: Σᐩ = prompt ?: editor.readDisplayText()
    .substringAfter("Solving: ").substringBefore("\n")
): Σᐩ = """
  <html>
  <body>
  <pre>${reason ?: "Synthesizing...\n"}
  """.trimIndent() +
    // TODO: legend
    solutions.joinToString("\n", "\n", "\n") + """🔍 Solving: $template
  
  ${if (reason != null) legend else ""}</pre>${stubs ?: ""}${cfg.renderedHTML}
  </body>
  </html>
  """.trimIndent()

fun TimeSource.Monotonic.ValueTimeMark.hasTimeLeft() =
  elapsedNow().inWholeMilliseconds < TIMEOUT_MS

fun updateProgress(query: Σᐩ, editor: TidyEditor) {
  val sanitized = query.escapeHTML()
  editor.writeDisplayText {
    it.replace(
      "Solving:.*\n".toRegex(),
      "Solving: $sanitized\n"
    )
  }
}

//    "$delim</pre>\n" +
//    GrammarToRRDiagram().run {
//      val grammar = BNFToGrammar().convert(
//        """
//        H2_SELECT =
//        'SELECT' [ 'TOP' term ] [ 'DISTINCT' | 'ALL' ] selectExpression {',' selectExpression} \
//        'FROM' tableExpression {',' tableExpression} [ 'WHERE' expression ] \
//        [ 'GROUP BY' expression {',' expression} ] [ 'HAVING' expression ] \
//        [ ( 'UNION' [ 'ALL' ] | 'MINUS' | 'EXCEPT' | 'INTERSECT' ) select ] [ 'ORDER BY' order {',' order} ] \
//        [ 'LIMIT' expression [ 'OFFSET' expression ] [ 'SAMPLE_SIZE' rowCountInt ] ] \
//        [ 'FOR UPDATE' ];
//        """.trimIndent()
//      )
//      RRDiagramToSVG().convert(grammar.rules.map { convert(it) }.last())
//    }

//fun CFG.toGrammar() = Grammar()

fun Σᐩ.sanitized(terminals: Set<Σᐩ>): Σᐩ =
  tokenizeByWhitespace().joinToString(" ") { if (it in terminals) it else "_" }

const val parsedPrefix = "✅ Current line parses! Tree:\n\n"
const val invalidPrefix = "❌ Current line invalid, possible fixes:\n\n"
const val ok = "<b>✅ Current line unambiguously parses! Parse tree:</b>\n"
const val ambig = "<b>⚠️ Current line parses, but is ambiguous:</b>\n"
const val no = "<b>❌ Current line invalid, possible fixes:</b>\n"
const val insertColor = "#AFFF9F"
const val changeColor = "#FFE585"
const val deleteColor = "#FFEEF2"
const val legend =
  "<span style=\"background-color: $insertColor\">  </span> : INSERTION   " +
      "<span style=\"background-color: $changeColor\">  </span> : SUBSTITUTION   " +
      "<span style=\"background-color: $deleteColor\">  </span> : DELETION"