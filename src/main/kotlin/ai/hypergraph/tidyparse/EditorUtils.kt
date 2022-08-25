package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.image.toHtmlTable
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.carveSeams
import ai.hypergraph.kaliningraph.containsHole
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.cache
import ai.hypergraph.kaliningraph.types.isSubsetOf
import com.github.difflib.text.DiffRow.Tag.*
import com.github.difflib.text.DiffRowGenerator
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import com.intellij.util.concurrency.AppExecutorUtil
import java.awt.Color
import java.util.*
import java.util.concurrent.Future
import kotlin.math.ceil

var cached: String = ""
var promise: Future<*>? = null

fun handle(currentLine: String, project: Project, editor: Editor, file: PsiFile) = TypedHandlerDelegate.Result.CONTINUE.also {
  val (caretPos, isInGrammar) = runReadAction {
    editor.caretModel.logicalPosition.column to
      (editor.caretModel.offset < editor.document.text.lastIndexOf("---"))
  }
  val sanitized = currentLine.trim().tokenizeByWhitespace().joinToString(" ")
  if (file.name.endsWith(".tidy") && sanitized != cached) {
    cached = sanitized
    promise?.cancel(true)
    TidyToolWindow.text = ""
    promise = AppExecutorUtil.getAppExecutorService()
      .submit { file.tryToReconcile(sanitized, isInGrammar, caretPos) }

    ToolWindowManager.getInstance(project).getToolWindow("Tidyparse")
      ?.let { if (!it.isVisible) it.show() }
  }
}

fun Editor.currentLine(): String =
  caretModel.let { it.visualLineStart to it.visualLineEnd }
    .let { (lineStart, lineEnd) ->
      document.getText(TextRange.create(lineStart, lineEnd))
    }

val AnActionEvent.editor: Editor get() =
  CommonDataKeys.EDITOR.getData(dataContext)!!

fun generateColors(n: Int): List<Color> =
  (0 until n).map { i ->
    Color.getHSBColor(i.toFloat() / n.toFloat(), 0.85f, 1.0f)
  }

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
fun CFG.overrideInvariance(l1: List<String>, l2: List<String>): String =
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

val CFG.prettyHTML by cache { pretty.toString().carveSeams().escapeHTML() }

fun render(
  solutions: List<String>,
  reason: String? = null,
  prompt: String? = null,
  stubs: String? = null
): String = """
  <html>
  <body style=\"font-family: JetBrains Mono\">
  <pre>${reason ?: "Synthesizing...\n"}
  """.trimIndent() +
  // TODO: legend
  solutions.joinToString("\n", "\n", "\n") +
  """🔍 Solving: ${
    prompt ?: TidyToolWindow.text.substringAfter("Solving: ").substringBefore("\n")
  }
  
${if (reason != null ) legend else ""}</pre>${stubs ?: ""}${cfg.renderCNFToHtml()}
      </body>
      </html>
  """.trimIndent()

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

fun String.synthesizeCachingAndDisplayProgress(
  cfg: CFG,
  tokens: List<String> = tokenizeByWhitespace(),
  sanitized: String = tokens.joinToString(" "),
  maxResults: Int = 20,
  variations: List<(String) -> Sequence<String>> =
    listOf(
      String::everySingleHoleConfig,
      String::increasingLengthChunks
    ),
): List<String> =
  synthCache.getOrPut(sanitized to cfg) {
    val renderedStubs = if (containsHole()) null
      else cfg.parseWithStubs(sanitized).second.renderStubs()
    val reason = if (containsHole()) null else no
    TidyToolWindow.text = render(emptyList(), stubs = renderedStubs, reason = reason)
    val solutions = TreeSet(compareBy(tokenwiseEdits(tokens)).thenBy { it.length })
    sanitized.synthesizeIncrementally(
      cfg = cfg,
      join = " ",
      variations = variations,
      cfgFilter = { true },
      updateProgress = { query ->
        if (Thread.currentThread().isInterrupted) throw InterruptedException()
        if ("Solving:" in TidyToolWindow.text) updateProgress(query)
      }
    ).map {
      updateSolutions(solutions, cfg, tokens, it)
      val htmlSolutions = if (containsHole()) solutions.map { it.escapeHTML() }
      else solutions.map { diffAsHtml(tokens, it.tokenizeByWhitespace()) }
      TidyToolWindow.text = render(htmlSolutions, stubs = renderedStubs, reason = reason)
    }.takeWhile { solutions.size <= maxResults }.toList()

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

private fun String.updateSolutions(
  solutions: TreeSet<String>,
  cfg: CFG,
  tokens: List<String>,
  it: String
) =
  if (containsHole()) solutions.add(it)
  else solutions.add(cfg.overrideInvariance(tokens, it.tokenizeByWhitespace()))

fun updateProgress(query: String) {
  val sanitized = query.escapeHTML()
  TidyToolWindow.text =
    TidyToolWindow.text.replace(
      "Solving:.*\n".toRegex(),
      "Solving: $sanitized\n"
    )
}

fun Sequence<Tree>.allIndicesInsideParseableRegions(): Set<Int> =
  map { it.span }.filter { 3 < it.last - it.first }
    .flatMap { (it.first + 1) until it.last }.toSet()

fun Sequence<Tree>.bordersOfParsable(): Set<Int> =
  map { it.span }.flatMap { listOf(it.first, it.last) }.toSet()

fun PsiFile.tryToReconcile(currentLine: String, isInGrammar: Boolean, caretPos: Int) =
  try { reconcile(currentLine, isInGrammar, caretPos) } catch (_: Exception) {}

fun PsiFile.reconcile(currentLine: String, isInGrammar: Boolean, caretPos: Int) {
  if (currentLine.isBlank()) return
  val cfg =
    if (isInGrammar)
      CFGCFG(
        names = currentLine.split(Regex("\\s+"))
          .filter { it.isNotBlank() && it !in setOf("->", "|") }.toSet()
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
  debugText += cfg.renderCNFToHtml()

  TidyToolWindow.text = """
        <html>
        <body style=\"font-family: JetBrains Mono\">
        $debugText
        </body>
        </html>
      """.trimIndent()
}

fun CFG.renderCNFToHtml(): String =
  "<pre>$delim<b>Normal form</b> (" +
    "${nonterminals.size} nonterminal${if (1 < nonterminals.size) "s" else ""} / " +
    "${terminals.size} terminal${if (1 < terminals.size) "s" else ""} / " +
    "$size production${if (1 < size) "s" else ""})" +
    "\n${prettyHTML}</pre>"

fun String.findRepairs(cfg: CFG, exclusions: Set<Int>, fishyLocations: List<Int>): String =
  synthesizeCachingAndDisplayProgress(
    cfg = cfg,
    tokens = tokenizeByWhitespace().map { if (it in cfg.terminals) it else "_" },
    variations = listOf {
      it.multiTokenSubstitutionsAndInsertions(
        numberOfEdits = 3,
        exclusions = exclusions,
        fishyLocations = fishyLocations
      )
    }
  ).let {
    if (it.isEmpty()) ""
    else it.joinToString("\n", "\n", "\n") {
      diffAsHtml(tokenizeByWhitespace(), it.tokenizeByWhitespace())
    }
  }

fun Sequence<Tree>.renderStubs(): String =
  runningFold(setOf<Tree>()) { acc, t -> if (acc.any { t.span isSubsetOf it.span }) acc else acc + t }
    .last().sortedBy { it.span.first }.map { it.prettyPrint() }.partition { it.contains('─') }
    .let { (branches, leaves) ->
      val (leafCols, branchCols) = 3 to 2
      "<pre>$delim<b>Parseable subtrees</b> (" +
        "${leaves.size} lea${if (leaves.size != 1) "ves" else "f"} / " +
        "${branches.size} branch${if (branches.size != 1) "es" else ""})</pre>\n\n" +
      leaves.mapIndexed { i, it -> "🌿── " + it.trim() }.let { asts ->
        FreeMatrix(ceil(asts.size.toDouble() / leafCols).toInt(), leafCols) { r, c ->
          if (r * leafCols + c < asts.size) asts[r * leafCols + c].ifBlank { "" } else ""
        }
      }.toHtmlTable() +
      branches.let { asts ->
        FreeMatrix(ceil(asts.size.toDouble() / branchCols).toInt(), branchCols) { r, c ->
          if (r * branchCols + c < asts.size) asts[r * branchCols + c].let { if (it.isNotBlank()) "🌿$it" else "" } else ""
        }
      }.toHtmlTable()
    }