package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.image.toHtmlTable
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.parsing.multiTokenSubstitutionsAndInsertions
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.isSubsetOf
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import org.jetbrains.concurrency.runAsync

var grammarFileCache: String? = ""
lateinit var cfg: CFG

val delim = List(50) { "─" }.joinToString("", "\n", "\n")

fun PsiFile.recomputeGrammar(): CFG =
  runReadAction {
    val grammar = text.substringBefore("---")
    if (grammar != grammarFileCache) {
      grammarFileCache = grammar
      ReadAction.compute<String, Exception> { grammarFileCache }
        .parseCFG().also { cfg = it }
    } else cfg
  }

val ok = "<pre><b>✅ Current line parses! Tree:</b></pre>\n"
val noMsg = "❌ Current line invalid, possible fixes:"
val no = "<pre><b>$noMsg</b></pre>\n"

class TidyKeyHandler : TypedHandlerDelegate() {
  override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile) =
    CONTINUE.also {
      if (file.name.endsWith(".tidy")) {
        file.reconcile(
          currentLine = editor.currentLine(),
          isInGrammar = editor.caretModel.offset < editor.document.text.lastIndexOf("---")
        )

        ToolWindowManager.getInstance(project).getToolWindow("Tidyparse")
          ?.let { if (!it.isVisible) it.show() }
      }
    }

  private fun Sequence<Tree>.allIndicesInsideParseableRegions(): Set<Int> =
    map { it.span }.filter { 3 < it.last - it.first }
      .flatMap { (it.first + 1) until it.last }.toSet()

  private fun PsiFile.reconcile(currentLine: String, isInGrammar: Boolean) =
    runAsync {
      if (currentLine.isBlank()) return@runAsync
      val cfg =
        if (isInGrammar)
          CFGCFG(names = currentLine.split(Regex("\\s+"))
            .filter { it.isNotBlank() && it !in setOf("->", "|") }.toSet()
          )
        else recomputeGrammar()

      var debugText = ""
      val cnf = "<pre>$delim<b>Chomsky normal form:</b></pre>\n${cfg.pretty.map { it.escapeHTML() }.toHtmlTable()}"
      if (currentLine.containsHole()) {
        synchronized(cfg) {
          currentLine.synthesizeCachingAndDisplayProgress(cfg).let {
            if (it.isNotEmpty()) debugText = "<pre><b>🔍 Found ${it.size} admissible solutions!</b>\n\n" + it.joinToString("\n").escapeHTML() + "</pre>"
          }
        }
      } else {
        val (parse, stubs) = cfg.parseWithStubs(currentLine)
        debugText = if (parse != null) ok + "<pre>" + parse.prettyPrint() + "</pre>"
        else no + currentLine.findRepairs(cfg, stubs.allIndicesInsideParseableRegions()) + stubs.renderStubs()
      }

      // Append the CFG only if parse succeeds
      debugText += cnf

      TidyToolWindow.textArea.text = """
        <html>
        <body style=\"font-family: JetBrains Mono\">
        $debugText
        </body>
        </html>
      """.trimIndent()
    }

  private fun String.findRepairs(cfg: CFG, exclusions: Set<Int>): String =
    synthesizeCachingAndDisplayProgress(
      cfg = cfg,
      variations = listOf { it.multiTokenSubstitutionsAndInsertions(numberOfEdits = 3, exclusions = exclusions) },
      allowNTs = true
    ).let {
      if (it.isNotEmpty())
        it.joinToString("\n", "<pre>", "\n") { diffAsHtml(this, it) } + "${delim}Partial AST branches:</pre>"
      else ""
    }

  fun Sequence<Tree>.renderStubs(): String =
    runningFold(setOf<Tree>()) { acc, t -> if (acc.any { t.span isSubsetOf it.span }) acc else acc + t }
      .last().map { it.prettyPrint() }.partition { it.contains('─') }
      .let { (trees, stubs) ->
        stubs.distinct().joinToString("  ", "<pre>", "</pre>\n") { it.trim() } +
        trees.let { asts -> if (asts.size % 2 == 1) asts + listOf("") else asts }
          .let { asts -> FreeMatrix(asts.size / 2, 2) { r, c -> asts[r * 2 + c] } }
          .toHtmlTable()
      }

  private fun String.containsHole(): Boolean =
    "_" in this || Regex("<[^\\s>]*>") in this
}