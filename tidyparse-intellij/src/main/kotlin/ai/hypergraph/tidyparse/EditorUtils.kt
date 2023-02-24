package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import com.github.difflib.text.DiffRow.Tag.*
import com.github.difflib.text.DiffRowGenerator
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import com.intellij.util.concurrency.AppExecutorUtil
import java.awt.Color
import java.util.concurrent.Future

var mostRecentQuery: String = ""
var promise: Future<*>? = null

class IJTidyEditor(val editor: Editor, val psiFile: PsiFile): TidyEditor {
  override fun readDisplayText(): Σᐩ = TidyToolWindow.text

  override fun writeDisplayText(s: Σᐩ) { TidyToolWindow.text = s }
  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun readEditorText(): Σᐩ = editor.document.text

  override fun getCaretPosition(): Int = editor.caretModel.offset

  override fun getLatestCFG(): CFG = psiFile.recomputeGrammar()

  override fun getOptimalSynthesizer(sanitized: Σᐩ, variations: List<Mutator>): Sequence<Σᐩ> =
    sanitized.synthesizeIncrementally(
      cfg = cfg,
      variations = variations,
      updateProgress = { query ->
        if ("Solving:" in readDisplayText()) updateProgress(query, this)
      }
    )

  override fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ =
    htmlDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") {
      when (it.tag) {
        INSERT -> it.newLine.replace("<span>", "<span style=\"background-color: $insertColor\">")
        CHANGE -> it.newLine.replace("<span>", "<span style=\"background-color: $changeColor\">")
        DELETE -> "<span style=\"background-color: $deleteColor\">${List(it.oldLine.length) { " " }.joinToString("")}</span>"
        else -> it.newLine.replace("<span>", "<span style=\"background-color: #FFFF66\">")
      }
    }
}

// TODO: Do not re-compute all work on each keystroke, cache prior results
fun handle(currentLine: String, project: Project, editor: Editor, file: PsiFile): Future<*>? {
    val tidyEditor = IJTidyEditor(editor, file)
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
         .submit { tidyEditor.tryToReconcile(sanitized, isInGrammar, caretPos) }

      println(helper())
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

fun List<String>.dittoSummarize(): List<String> =
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

fun PsiFile.recomputeGrammar(): CFG {
  val grammar: String = runReadAction { text.substringBefore("---") }
  return if (grammar != grammarFileCache || cfg.isNotEmpty()) {
    grammarFileCache = grammar
    grammarFileCache.parseCFG().also { cfg = it }
  } else cfg
}

