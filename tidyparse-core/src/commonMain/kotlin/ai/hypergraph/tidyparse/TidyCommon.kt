package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.*
import ai.hypergraph.kaliningraph.parsing.* // TODO: Why is this not available?
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.parsing.prettyPrint
import ai.hypergraph.kaliningraph.types.*
import kotlin.math.*
import kotlin.time.*

fun helper() = "helper"

fun CFG.renderCFGToHTML(): String =
  (listOf(originalForm.summarize("Original form")) +
      (if (originalForm == nonparametricForm) listOf()
      else listOf(nonparametricForm.summarize("Nonparametric form"))) +
      listOf(summarize("Normal form"))).let { rewriteSummary ->
    val maxLen = rewriteSummary.joinToString("\n").lines().maxOf { it.length }
    rewriteSummary.joinToString(delim(maxLen), "<pre>${delim(maxLen)}", "</pre>")
  }

fun CFG.summarize(name: String): String = "<b>$name</b> (" +
    "${nonterminals.size} nonterminal${if (1 < nonterminals.size) "s" else ""} / " +
    "${terminals.size} terminal${if (1 < terminals.size) "s" else ""} / " +
    "$size production${if (1 < size) "s" else ""})\n${prettyHTML}"

fun delim(len: Int = 120) = List(len) { "─" }.joinToString("", "\n", "\n")

val CFG.prettyHTML by cache { prettyPrint().carveSeams().escapeHTML() }

// Determines whether a substitution is invariant w.r.t. NT membership
fun CFG.preservesNTInvariance(newNT: String, oldTerminal: String) =
  newNT in bimap[listOf(oldTerminal)]

val la = "<".escapeHTML()
val ra = ">".escapeHTML()
fun String.treatAsNonterminal() = drop(la.length).dropLast(ra.length)
fun String.isNonterminal() = startsWith(la) && endsWith(ra)

fun String.dehtmlify(): String =
  replace("&lt;", "<").replace("&gt;", ">")

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()
var grammarFileCache: String = ""

fun displayComparator(tokens: List<String>): Comparator<String> =
  compareBy(tokenwiseLevenshteinEdits(tokens)).thenBy { it.length }

fun tokenwiseLevenshteinEdits(tokens: List<String>): (String) -> Comparable<*> =
  { levenshtein(tokens.filterNot { it.containsHole() }, it.tokenizeByWhitespace()) }

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

fun render(
  solutions: List<String>,
  editor: TidyEditor,
  reason: String? = null,
  prompt: String? = null,
  stubs: String? = null,
  template: String? = prompt ?: editor.readDisplayText()
    .substringAfter("Solving: ").substringBefore("\n")
): String = """
  <html>
  <body>
  <pre>${reason ?: "Synthesizing...\n"}
  """.trimIndent() +
    // TODO: legend
    solutions.joinToString("\n", "\n", "\n") + """🔍 Solving: $template
  
  ${if (reason != null ) legend else ""}</pre>${stubs ?: ""}${cfg.renderCFGToHTML()}
  </body>
  </html>
  """.trimIndent()

fun TidyEditor.tryToReconcile(currentLine: String, isInGrammar: Boolean, caretPos: Int) =
  try { reconcile(currentLine, isInGrammar, caretPos) } catch (_: Exception) { }

@OptIn(ExperimentalTime::class)
fun String.synthesizeCachingAndDisplayProgress(
  editor: TidyEditor,
  cfg: CFG,
  tokens: List<String> = tokenizeByWhitespace().map { if (it in cfg.terminals) it else "_" },
  sanitized: String = tokens.joinToString(" "),
  maxResults: Int = 20,
  // TODO: think about whether we really want to solve for variations in every case
  variations: List<Mutator> =
    listOf(
      { a, b -> a.everySingleHoleConfig() },
      { a, b -> a.increasingLengthChunks() }
    ),
): List<String> =
  synthCache.getOrPut(sanitized to cfg) {
    val t = TimeSource.Monotonic.markNow()
    val renderedStubs = if (containsHole()) null
    else cfg.parseWithStubs(sanitized).second.renderStubs()
    val reason = if (containsHole()) null else no
    editor.writeDisplayText(render(emptyList(), editor, stubs = renderedStubs, reason = reason))
    val solutions = mutableSetOf<Σᐩ>()

    editor.getOptimalSynthesizer(sanitized, variations).map {
      solutions.add(it)
      val htmlSolutions =
        solutions.sortedWith(displayComparator(tokens)).let { solutions ->
          if (containsHole()) solutions.map { it.escapeHTML() }
          else solutions
//        .also { it.map { println(diffAsLatex(tokens, it.tokenizeByWhitespace())) }; println() }
            .map { editor.diffAsHtml(tokens, it.tokenizeByWhitespace()) }
        }

      editor.writeDisplayText(render(htmlSolutions, editor, stubs = renderedStubs, reason = reason))
    }.takeWhile { solutions.size <= maxResults && t.elapsedNow().inWholeMilliseconds < TIMEOUT_MS }.toList()

    solutions.sortedWith(displayComparator(tokens)).toList()
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
fun TidyEditor.reconcile(
  currentLine: String,
  caretInGrammar: Boolean,
  caretPos: Int
) {
  if (currentLine.isBlank()) return
  val cfg =
    if (caretInGrammar)
      CFGCFG(
        names = currentLine.tokenizeByWhitespace()
          .filter { it !in setOf("->", "|") }.toSet()
      )
    else getLatestCFG()

  if (cfg == null) return

  var debugText = ""
  if (currentLine.containsHole()) {
      currentLine.synthesizeCachingAndDisplayProgress(this, cfg).let {
        debugText = "<pre><b>🔍 Found ${it.size} admissible solutions!</b>\n\n" +
            it.joinToString("\n") { it.escapeHTML() } + "</pre>"
      }
  } else {
    println("Parsing `$currentLine` with stubs!")
    val (parseForest, stubs) = cfg.parseWithStubs(currentLine)
    debugText = if (parseForest.isNotEmpty()) {
      if (parseForest.size == 1) "<pre>$ok\n🌳" + parseForest.first().prettyPrint() + "</pre>"
      else "<pre>$ambig\n🌳" + parseForest.joinToString("\n\n") { it.prettyPrint() } + "</pre>"
    } else {
      val exclude = stubs.allIndicesInsideParseableRegions()
      val repairs = currentLine.findRepairs(this, cfg, exclude, fishyLocations = listOf(caretPos))
      "<pre>$no" + repairs + "\n$legend</pre>" + stubs.renderStubs()
    }
  }

  // Append the CFG only if parse succeeds
  debugText += cfg.renderCFGToHTML()

//  println(cfg.original.graph.toString())
//  println(cfg.original.graph.toDot())
//  println(cfg.graph.toDot().alsoCopy())
//  cfg.graph.A.show()

  writeDisplayText("""
        <html>
        <body>
        $debugText
        </body>
        </html>
      """.trimIndent())
//    .also { it.show() }
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

fun String.findRepairs(editor: TidyEditor, cfg: CFG, exclusions: Set<Int>, fishyLocations: List<Int>): String =
  synthesizeCachingAndDisplayProgress(
    editor,
    cfg = cfg,
    tokens = tokenizeByWhitespace().map { if (it in cfg.terminals) it else "_" },
    variations = listOf({ a, b ->
      a.multiTokenSubstitutionsAndInsertions(
        numberOfEdits = 3,
        exclusions = b,
        fishyLocations = fishyLocations
      )
    })
  ).let {
    if (it.isEmpty()) ""
    else it.joinToString("\n", "\n", "\n") {
      editor.diffAsHtml(tokenizeByWhitespace(), it.tokenizeByWhitespace())
    }
  }


var cfg: CFG = setOf()

fun String.sanitized(): String =
  tokenizeByWhitespace().joinToString(" ") { if (it in cfg.terminals) it else "_" }


const val ok = "<b>✅ Current line unambiguously parses! Parse tree:</b>\n"
const val ambig = "<b>⚠️ Current line parses, but is ambiguous:</b>\n"
const val no = "<b>❌ Current line invalid, possible fixes:</b>\n"
const val insertColor = "#85FF7A"
const val changeColor = "#FFC100"
const val deleteColor = "#FFCCCB"
const val legend =
  "<span style=\"background-color: $insertColor\">  </span> : INSERTION   " +
      "<span style=\"background-color: $changeColor\">  </span> : SUBSTITUTION   " +
      "<span style=\"background-color: $deleteColor\">  </span> : DELETION"
