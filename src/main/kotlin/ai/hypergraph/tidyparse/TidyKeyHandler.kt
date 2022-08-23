package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.parsing.tokenizeByWhitespace
import com.intellij.codeInsight.editorActions.BackspaceHandlerDelegate
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
//  override fun beforeCharTyped(c: Char, project: Project, editor: Editor, file: PsiFile, fileType: FileType) =
//    handle(runReadAction { editor.currentLine() }.let {
//      val index = runReadAction { editor.caretModel.logicalPosition.column }
//      it.substring(0, index) + c + it.substring(index)
//    }, project, editor, file)

  override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile): Result =
    handle(runReadAction { editor.currentLine() }, project, editor, file)
}

class TidyBackspaceHandler : BackspaceHandlerDelegate() {
  override fun beforeCharDeleted(c: Char, file: PsiFile, editor: Editor) = Unit

  override fun charDeleted(c: Char, file: PsiFile, editor: Editor): Boolean =
    true.also { handle(runReadAction { editor.currentLine() }, editor.project!!, editor, file) }
}

var cached: String = ""
var promise: Future<*>? = null

private fun handle(currentLine: String, project: Project, editor: Editor, file: PsiFile) = CONTINUE.also {
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