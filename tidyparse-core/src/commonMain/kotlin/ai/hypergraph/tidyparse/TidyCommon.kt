package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.*
import ai.hypergraph.kaliningraph.parsing.* // TODO: Why is this not available?
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.parsing.prettyPrint
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import ai.hypergraph.kaliningraph.types.*
import kotlin.math.*
import kotlin.time.*

val CFG.renderedHTML by cache { renderCFGToHTML() }

fun CFG.renderCFGToHTML(tokens: Set<Σᐩ> = emptySet()): String =
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

fun CFG.summarize(name: String): String = "<b>$name</b> (" +
    "${nonterminals.size} nonterminal${if (1 < nonterminals.size) "s" else ""} / " +
    "${terminals.size} terminal${if (1 < terminals.size) "s" else ""} / " +
    "$size production${if (1 < size) "s" else ""})\n$prettyHTML"

fun delim(len: Int = 120) = List(len) { "─" }.joinToString("", "\n", "\n")

val CFG.prettyHTML by cache { prettyPrint().carveSeams().escapeHTML() }

// Determines whether a substitution is invariant w.r.t. NT membership
fun CFG.preservesNTInvariance(newNT: String, oldTerminal: String) =
  newNT in bimap[listOf(oldTerminal)]

val la = "<".escapeHTML()
val ra = ">".escapeHTML()
fun String.treatAsNonterminal() = drop(la.length).dropLast(ra.length)

fun String.dehtmlify(): String =
  replace("&lt;", "<")
    .replace("&gt;", ">")
    .replace("&amp;", "&")
    .replace("&quot;", "\"")
    .replace("&apos;", "'")
    .replace("<span.*?>".toRegex(), "")
    .replace("</span>", "")

fun displayComparator(tokens: List<String>): Comparator<String> =
  compareBy(tokenwiseLevenshteinEdits(tokens)).thenBy { it.length }

fun tokenwiseLevenshteinEdits(tokens: List<String>): (String) -> Comparable<*> =
  { levenshtein(tokens.filterNot { it == "_" }, it.tokenizeByWhitespace()) }

fun List<Tree>.renderStubs(): String =
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
  solutions: List<String>,
  editor: TidyEditor,
  reason: String? = null,
  prompt: String? = null,
  stubs: String? = null,
  template: String = prompt ?: editor.readDisplayText()
    .substringAfter("Solving: ").substringBefore("\n")
): String = """
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
  solutions: List<String>,
  editor: TidyEditor,
  reason: String? = null,
  prompt: String? = null,
  stubs: String? = null,
  template: String = prompt ?: editor.readDisplayText()
    .substringAfter("Solving: ").substringBefore("\n")
): String = """
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

fun String.synthesizeCachingAndDisplayProgress(tidyEditor: TidyEditor, cfg: CFG): List<String> {
  val sanitized: String = tokenizeByWhitespace().joinToString(" ") { if (it in cfg.terminals) it else "_" }

  val cacheResultOn: Pair<String, CFG> = sanitized to cfg

  val cached = synthCache[cacheResultOn]

  return if (cached?.isNotEmpty() == true) cached
  // Cache miss could be due to prior timeout or cold cache. Either way, we need to recompute
  else tidyEditor.repair(cfg, this).also { synthCache.put(cacheResultOn, it) }
}

fun updateProgress(query: String, editor: TidyEditor) {
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

fun String.sanitized(terminals: Set<Σᐩ>): String =
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