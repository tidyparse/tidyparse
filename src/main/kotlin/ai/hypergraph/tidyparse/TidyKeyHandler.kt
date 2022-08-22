package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.parseCFG
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import com.intellij.util.concurrency.AppExecutorUtil
import java.util.concurrent.Future

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

val ok = "<b>✅ Current line parses! Tree:</b>\n"
val no = "<b>❌ Current line invalid, possible fixes:</b>\n"

class TidyKeyHandler : TypedHandlerDelegate() {
  var promise: Future<*>? = null
  override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile) =
    CONTINUE.also {
      val currentLine = runReadAction { editor.currentLine() }
      val isInGrammar = runReadAction { editor.caretModel.offset < editor.document.text.lastIndexOf("---") }
      if (file.name.endsWith(".tidy")) {
        promise?.cancel(true)
        TidyToolWindow.text = ""
        promise = AppExecutorUtil.getAppExecutorService()
          .submit { file.tryToReconcile(currentLine, isInGrammar) }

        ToolWindowManager.getInstance(project).getToolWindow("Tidyparse")
          ?.let { if (!it.isVisible) it.show() }
      }
    }
}