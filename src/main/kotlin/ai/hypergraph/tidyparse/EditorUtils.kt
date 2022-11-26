package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.carveSeams
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.image.toHtmlTable
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.cache
import ai.hypergraph.kaliningraph.types.isSubsetOf
import com.github.difflib.text.DiffRow.Tag.*
import com.github.difflib.text.DiffRowGenerator
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import com.intellij.util.concurrency.AppExecutorUtil
import prettyPrint
import java.awt.Color
import java.util.*
import java.util.concurrent.Future
import kotlin.math.ceil

var mostRecentQuery: String = ""
var promise: Future<*>? = null

// TODO: Do not re-compute all work on each keystroke, cache prior results
fun handle(currentLine: String, project: Project, editor: Editor, file: PsiFile): Future<*>? {
    val (caretPos, isInGrammar) = runReadAction {
      editor.caretModel.logicalPosition.column to
        editor.document.text.lastIndexOf("---").let { separator ->
          if (separator == -1) true else editor.caretModel.offset < separator
        }
    }
    val sanitized = currentLine.trim().tokenizeByWhitespace().joinToString(" ")
    if (file.name.endsWith(".tidy") && sanitized != mostRecentQuery) {
      mostRecentQuery = sanitized
      promise?.cancel(true)
      TidyToolWindow.text = ""
      promise = AppExecutorUtil.getAppExecutorService()
         .submit { file.tryToReconcile(sanitized, isInGrammar, caretPos) }

      ToolWindowManager.getInstance(project).getToolWindow("Tidyparse")
        ?.let { runReadAction { if (!it.isVisible) it.show() } }
    }
    return promise
  }

fun Editor.currentLine(): String =
  caretModel.let { it.visualLineStart to it.visualLineEnd }
    .let { (lineStart, lineEnd) ->
      document.getText(TextRange.create(lineStart, lineEnd))
    }

fun generateColors(n: Int): List<Color> =
  (0 until n).map { i ->
    Color.getHSBColor(i.toFloat() / n.toFloat(), 0.85f, 1.0f)
  }

private val latexDiffGenerator: DiffRowGenerator =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean -> if (f) "(*@\\hl{" else "}@*)" }
    .oldTag { _ -> "" }
    .build()

fun diffAsLatex(l1: List<String>, l2: List<String>): String =
  latexDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") {
    when (it.tag) {
      INSERT -> it.newLine.replace("\\hl", "\\hlgreen")
      CHANGE -> it.newLine.replace("\\hl", "\\hlorange")
      DELETE -> "(*@\\hlred{${it.oldLine}} @*)"
      else -> it.newLine
    }
  }.replace("&gt;", ">").replace("&lt;", "<")

private val htmlDiffGenerator: DiffRowGenerator =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean -> "<${if (f) "" else "/"}span>" }
    .oldTag { _ -> "" }
    .build()

fun diffAsHtml(l1: List<String>, l2: List<String>): String =
  htmlDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") {
    when (it.tag) {
      INSERT -> it.newLine.replace("<span>", "<span style=\"background-color: $insertColor\">")
      CHANGE -> it.newLine.replace("<span>", "<span style=\"background-color: $changeColor\">")
      DELETE -> "<span style=\"background-color: $deleteColor\">${List(it.oldLine.length) { " " }.joinToString("")}</span>"
      else -> it.newLine.replace("<span>", "<span style=\"background-color: #FFFF66\">")
    }
  }

private val plaintextDiffGenerator: DiffRowGenerator =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { _ -> "" }
    .oldTag { _ -> "" }
    .build()

// TODO: maybe add a dedicated SAT constraint instead of filtering out NT invariants after?
private fun CFG.overrideInvariance(l1: List<String>, l2: List<String>): String =
  plaintextDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") { dr ->
  val (old,new ) = dr.oldLine to dr.newLine
  // If repair substitutes a terminal with the same NT that it belongs to, do not treat as a diff
  if (new.isNonterminal() && preservesNTInvariance(new.treatAsNonterminal(), old)) old else new
}

// Determines whether a substitution is invariant w.r.t. NT membership
private fun CFG.preservesNTInvariance(newNT: String, oldTerminal: String) =
  newNT in bimap[listOf(oldTerminal)]

private val la = "<".escapeHTML()
private val ra = ">".escapeHTML()
private fun String.treatAsNonterminal() = drop(la.length).dropLast(ra.length)
private fun String.isNonterminal() = startsWith(la) && endsWith(ra)

val CFG.prettyHTML by cache { prettyPrint().carveSeams().escapeHTML() }

fun render(
  solutions: List<String>,
  reason: String? = null,
  prompt: String? = null,
  stubs: String? = null
): String = """
  <html>
  <body>
  <pre>${reason ?: "Synthesizing...\n"}
  """.trimIndent() +
  // TODO: legend
  solutions.joinToString("\n", "\n", "\n") + """🔍 Solving: ${
    prompt ?: TidyToolWindow.text.substringAfter("Solving: ").substringBefore("\n")
  }
  
  ${if (reason != null ) legend else ""}</pre>${stubs ?: ""}${cfg.renderCFGToHTML()}
  </body>
  </html>
  """.trimIndent()

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

fun String.sanitized(): String =
  tokenizeByWhitespace().joinToString(" ") { if (it in cfg.terminals) it else "_" }

fun String.synthesizeCachingAndDisplayProgress(
  cfg: CFG,
  tokens: List<String> = tokenizeByWhitespace().map { if (it in cfg.terminals) it else "_" },
  sanitized: String = tokens.joinToString(" "),
  maxResults: Int = 20,
  checkInterrupted: () -> Unit = { if (Thread.currentThread().isInterrupted) throw InterruptedException() },
    // TODO: think about whether we really want to solve for variations in every case
  variations: List<Mutator> =
    listOf(
      { a, b -> a.everySingleHoleConfig() },
      { a, b -> a.increasingLengthChunks() }
    ),
): List<String> =
  synthCache.getOrPut(sanitized to cfg) {
    val t = System.currentTimeMillis()
    val renderedStubs = if (containsHole()) null
      else cfg.parseWithStubs(sanitized).second.renderStubs()
    val reason = if (containsHole()) null else no
    TidyToolWindow.text = render(emptyList(), stubs = renderedStubs, reason = reason)
    val solutions = TreeSet(compareBy(tokenwiseEdits(tokens)).thenBy { it.length })

    fun String.updateSolutions(
      solutions: TreeSet<String>,
      cfg: CFG,
      tokens: List<String>,
      it: String
    ) =
      if (containsHole()) solutions.add(it)
      else solutions.add(cfg.overrideInvariance(tokens, it.tokenizeByWhitespace()))

    sanitized.synthesizeIncrementally(
      cfg = cfg,
      variations = variations,
      checkInterrupted = checkInterrupted,
      updateProgress = { query ->
        checkInterrupted().also { if ("Solving:" in TidyToolWindow.text) updateProgress(query) }
      }
    ).map {
//      updateSolutions(solutions, cfg, tokens, it)
      solutions.add(it)
      val htmlSolutions = if (containsHole()) solutions.map { it.escapeHTML() }
      else solutions
//        .also { it.map { println(diffAsLatex(tokens, it.tokenizeByWhitespace())) }; println() }
        .map { diffAsHtml(tokens, it.tokenizeByWhitespace()) }
      TidyToolWindow.text = render(htmlSolutions, stubs = renderedStubs, reason = reason)
    }.takeWhile { solutions.size <= maxResults && System.currentTimeMillis() - t < TIMEOUT_MS }.toList()

    solutions.toList()
  }

private fun tokenwiseEdits(tokens: List<String>): (String) -> Comparable<*> =
  { levenshtein(tokens.filterNot { it.containsHole() }, it.tokenizeByWhitespace()) }

fun List<String>.dittoSummarize() =
  listOf("", *toTypedArray())
    .windowed(2, 1).map { it[0] to it[1] }
    .map { (a, b) -> ditto(a, b) }

fun ditto(s1: String, s2: String): String =
  plaintextDiffGenerator.generateDiffRows(
    s1.tokenizeByWhitespace(),
    s2.tokenizeByWhitespace()
  ).joinToString("") {
    when (it.tag) {
      EQUAL -> ""
      else -> it.newLine + " "
    }
  }

fun updateProgress(query: String) {
  val sanitized = query.escapeHTML()
  TidyToolWindow.text =
    TidyToolWindow.text.replace(
      "Solving:.*\n".toRegex(),
      "Solving: $sanitized\n"
    )
}

fun Sequence<Tree>.bordersOfParsable(): Set<Int> =
  map { it.span }.flatMap { listOf(it.first, it.last) }.toSet()

fun PsiFile.tryToReconcile(currentLine: String, isInGrammar: Boolean, caretPos: Int) =
  try { reconcile(currentLine, isInGrammar, caretPos) } catch (_: Exception) { }

fun PsiFile.reconcile(currentLine: String, caretInGrammar: Boolean, caretPos: Int) {
  if (currentLine.isBlank()) return
  val cfg =
    if (caretInGrammar)
      CFGCFG(
        names = currentLine.tokenizeByWhitespace()
          .filter { it !in setOf("->", "|") }.toSet()
      )
    else recomputeGrammar()

  var debugText = ""
  if (currentLine.containsHole()) {
    synchronized(cfg) {
      currentLine.synthesizeCachingAndDisplayProgress(cfg).let {
        debugText = "<pre><b>🔍 Found ${it.size} admissible solutions!</b>\n\n" +
          it.joinToString("\n") { it.escapeHTML() } + "</pre>"
      }
    }
  } else {
    println("Parsing ${currentLine} with stubs!")
    val (parseForest, stubs) = cfg.parseWithStubs(currentLine)
    debugText = if (parseForest.isNotEmpty()) {
      if (parseForest.size == 1) "<pre>$ok\n🌳" + parseForest.first().prettyPrint() + "</pre>"
      else "<pre>$ambig\n🌳" + parseForest.joinToString("\n\n") { it.prettyPrint() } + "</pre>"
    } else {
      val exclude = stubs.allIndicesInsideParseableRegions()
      val repairs = currentLine.findRepairs(cfg, exclude, fishyLocations = listOf(caretPos))
      "<pre>$no" + repairs + "\n$legend</pre>" + stubs.renderStubs()
    }
  }

  // Append the CFG only if parse succeeds
  debugText += cfg.renderCFGToHTML()

//  println(cfg.original.graph.toString())
//  println(cfg.original.graph.toDot())
//  println(cfg.graph.toDot().alsoCopy())
//  cfg.graph.A.show()

  TidyToolWindow.text = """
        <html>
        <body>
        $debugText
        </body>
        </html>
      """.trimIndent()
//    .also { it.show() }
}

fun CFG.renderCFGToHTML(): String =
  (listOf(originalForm.summarize("Original form")) +
    (if (originalForm == nonparametricForm) listOf()
    else listOf(nonparametricForm.summarize("Nonparametric form"))) +
    listOf(summarize("Normal form"))
  ).let { rewriteSummary ->
    val maxLen = rewriteSummary.joinToString("\n").lines().maxOf { it.length }
    rewriteSummary.joinToString(delim(maxLen), "<pre>${delim(maxLen)}", "</pre>")
  }

fun CFG.summarize(name: String) = "<b>$name</b> (" +
    "${nonterminals.size} nonterminal${if (1 < nonterminals.size) "s" else ""} / " +
    "${terminals.size} terminal${if (1 < terminals.size) "s" else ""} / " +
    "$size production${if (1 < size) "s" else ""})\n${prettyHTML}"

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

fun String.findRepairs(cfg: CFG, exclusions: Set<Int>, fishyLocations: List<Int>): String =
  synthesizeCachingAndDisplayProgress(
    cfg = cfg,
    tokens = tokenizeByWhitespace().map { if (it in cfg.terminals) it else "_" },
    variations = listOf { a, b ->
      a.multiTokenSubstitutionsAndInsertions(
        numberOfEdits = 3,
        exclusions = b,
        fishyLocations = fishyLocations
      )
    }
  ).let {
    if (it.isEmpty()) ""
    else it.joinToString("\n", "\n", "\n") {
      diffAsHtml(tokenizeByWhitespace(), it.tokenizeByWhitespace())
    }
  }

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

var grammarFileCache: String = ""
lateinit var cfg: CFG
fun delim(len: Int = 120) = List(len) { "─" }.joinToString("", "\n", "\n")

fun PsiFile.recomputeGrammar(): CFG {
  val grammar: String = runReadAction { text.substringBefore("---") }
  return if (grammar != grammarFileCache || !::cfg.isInitialized) {
    grammarFileCache = grammar
    grammarFileCache.parseCFG().also { cfg = it }
  } else cfg
}

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

fun String.dehtmlify() =
  replace("&lt;", "<").replace("&gt;", ">")