package ai.hypergraph.tidyparse

import ai.grazie.nlp.utils.takeNonWhitespaces
import ai.hypergraph.kaliningraph.parsing.*
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.grazie.utils.dropPrefix
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import org.jetbrains.concurrency.runAsync

var grammarFileCache: String? = ""
lateinit var cfg: CFG

fun PsiFile.recomputeGrammar(): CFG =
  runReadAction {
    val grammar = text.substringBefore("---")
    if (grammar != grammarFileCache) {
      grammarFileCache = grammar
      ReadAction.compute<String, Exception> { grammarFileCache }
        .parseCFG().also { cfg = it }
    } else cfg
  }

class TidyKeyHandler : TypedHandlerDelegate() {
  val ok = "✅ Current line parses!\n"
  val no = "❌ Current line invalid\n"

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

  private fun PsiFile.reconcile(currentLine: String, isInGrammar: Boolean) =
    runAsync {
      val cfg =
        if (isInGrammar)
          CFGCFG(names = currentLine.split(Regex("\\s+"))
            .filter { it.isNotBlank() && it !in setOf("->", "|") }.toSet()
          )
        else recomputeGrammar()

      var debugText = ""
      if (currentLine.containsHole()) {
        synchronized(cfg) {
          synth(currentLine, cfg).let {
            if (it.isNotEmpty()) debugText = it.joinToString("\n")
          }
        }
      } else {
        val parse = cfg.parse(currentLine)
        debugText = if (parse != null) ok + parse.prettyPrint()
        else no + TidyToolWindow.textArea.text.dropPrefix(ok).dropPrefix(no)
      }

      // Append the CFG only if parse succeeds
      if (!debugText.startsWith(no)) {
        val delim = List(50) { "─" }.joinToString("", "\n", "\n")
        debugText += delim + "Chomsky normal form:\n${cfg.prettyPrint(3)}"
      }

      TidyToolWindow.textArea.text = debugText
    }

  private fun String.containsHole(): Boolean =
    "_" in this || Regex("<[^\\s>]*>") in this
}