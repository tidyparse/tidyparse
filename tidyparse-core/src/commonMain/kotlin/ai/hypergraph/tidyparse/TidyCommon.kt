package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.repairWithSparseGRE
import ai.hypergraph.kaliningraph.image.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.*
import kotlinx.coroutines.delay
import kotlin.math.ceil
import kotlin.time.Duration.Companion.nanoseconds
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

const val MAX_DISP_RESULTS = 29

var i = 0
suspend fun pause(freq: Int = 300_000) { if (i++ % freq == 0) { delay(50.nanoseconds) } }

suspend fun sampleGREUntilTimeout(tokens: List<String>, cfg: CFG) =
  repairWithSparseGRE(tokens, cfg)?.let {
    val clock = TimeSource.Monotonic.markNow()
    it.words(cfg.tmLst) { clock.hasTimeLeft() }
  } ?: emptySequence()

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

const val fwdCplPrefix = "-> Forward completion, possible continuations:\n\n"
const val ifxCplPrefix = ">< Infix completion, possible fillings:\n\n"
const val parsedPrefix = "✅ Current line parses! Tree:\n\n"
const val invalidPrefix = "❌ Current line invalid, possible fixes:\n\n"
const val stubGenPrefix = "&lt;/&gt; Stub generation, possible completions:\n\n"
const val holeGenPrefix = "___ Hole generation, possible completions:\n\n"
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
