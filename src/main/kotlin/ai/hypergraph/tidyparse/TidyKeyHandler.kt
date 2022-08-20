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
}