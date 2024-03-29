package ai.hypergraph.tidyparse

import com.intellij.codeInsight.editorActions.*
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFile

var fontScalingRatio = 1.0

class TidyKeyHandler : TypedHandlerDelegate() {
//  override fun beforeCharTyped(c: Char, project: Project, editor: Editor, file: PsiFile, fileType: FileType) =
//    handle(runReadAction { editor.currentLine() }.let {
//      val index = runReadAction { editor.caretModel.logicalPosition.column }
//      it.substring(0, index) + c + it.substring(index)
//    }, project, editor, file)

  override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile): Result =
    CONTINUE.also {
      fontScalingRatio = (editor.colorsScheme.editorFontSize / 16.0).coerceAtLeast(1.0)
      ijTidyEditor = IJTidyEditor(editor, file)
      ijTidyEditor.handle()
    }
}

lateinit var ijTidyEditor: IJTidyEditor

class TidyBackspaceHandler : BackspaceHandlerDelegate() {
  override fun beforeCharDeleted(c: Char, file: PsiFile, editor: Editor) = Unit

  override fun charDeleted(c: Char, file: PsiFile, editor: Editor): Boolean =
    true.also { IJTidyEditor(editor, file).handle() }
}