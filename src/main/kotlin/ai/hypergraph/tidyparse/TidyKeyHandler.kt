package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeFrom
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.updateSettings.impl.Product
import com.intellij.psi.PsiFile
import org.jetbrains.concurrency.runAsync

var grammarFileCache: String? = ""
lateinit var cfg: CFG

fun PsiFile.recomputeGrammar(): CFG =
  runReadAction {
    val grammar = text.substringBefore("---")
    if (grammar != grammarFileCache) {
      grammarFileCache = grammar
      ReadAction.compute<String, Exception> { grammarFileCache }.parseCFG()
    } else cfg
  }

class TidyKeyHandler : TypedHandlerDelegate() {
  val ok = "✅ Current line parses!\n"
  val no = "❌ Current line invalid\n"

  override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile) =
    CONTINUE.also {
      if (file.name.endsWith(".tidy"))
        file.reconcile(
          currentLine = editor.currentLine(),
          isInGrammar = editor.caretModel.offset < editor.document.text.lastIndexOf("---")
        )
    }

  private fun PsiFile.reconcile(currentLine: String, isInGrammar: Boolean) =
    runAsync {
      cfg =
        if (isInGrammar)
          CFGCFG(names = currentLine.split(Regex("\\s+"))
            .filter { it.isNotBlank() && it.contains("\\w") })
        else recomputeGrammar()

      var debugText = ""
      if ("_" !in currentLine && Regex("<[^\\s>]*>") !in currentLine) {
        val parse = cfg.parse(currentLine)
        debugText = if (parse != null) ok + parse.prettyPrint()
        else no + TidyToolWindow.textArea.text.drop(ok.length)
      } else synchronized(cfg) {
        currentLine.synthesizeFrom(cfg, " ").take(20).toList().shuffled()
          .let { if (it.isNotEmpty()) debugText = it.joinToString("\n") }
      }

      val delim = List(50) { "─" }.joinToString("", "\n", "\n")
      debugText += delim + "Chomsky normal form:\n${cfg.prettyPrint(3)}"
      TidyToolWindow.textArea.text = debugText
    }
}