package ai.hypergraph.tidyparse

import com.intellij.codeInsight.lookup.LookupElement
import com.intellij.codeInsight.lookup.LookupEvent
import com.intellij.codeInsight.lookup.LookupListener
import com.intellij.codeInsight.lookup.LookupManager
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.DumbAwareAction
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange

class TidyRepairAction : DumbAwareAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val project = e.project ?: return
    val editor = e.getData(CommonDataKeys.EDITOR) ?: return
    val line = editor.currentDocumentLine().trim()

    project.webGpuStringService().runString(line).whenComplete { raw, err ->
      if (err != null) { err.showTidyParseError(project); return@whenComplete }
      ApplicationManager.getApplication().invokeLater { showTidyWebGpuCompletionLookup(project, editor, raw.lines()) }
    }
  }

  override fun update(e: AnActionEvent) {
    e.presentation.isEnabledAndVisible = e.project != null && e.getData(CommonDataKeys.EDITOR) != null
  }
}

internal fun showTidyWebGpuCompletionLookup(project: Project, editor: Editor, rawCompletions: Iterable<String>) {
  val replacementStart = editor.currentLineReplacementStart()
  val completions = rawCompletions.filter { it.isNotEmpty() }.distinct().take(10)

  if (completions.isEmpty()) { return }

  val lookup = LookupManager.getInstance(project)
    .showLookup(editor, completions.map { object : LookupElement() {
      override fun getLookupString(): String = it
    } }.toTypedArray<LookupElement>(), "") ?: return

  lookup.addLookupListener(object : LookupListener {
    override fun beforeItemSelected(event: LookupEvent): Boolean {
      val completion = event.item?.lookupString ?: return false
      WriteCommandAction.runWriteCommandAction(project) { editor.replaceLineSuffix(replacementStart, completion) }
      lookup.hideLookup(true)
      return false
    }
  })
}

internal fun Editor.currentDocumentLine(): String {
  val document = this.document
  val lineNumber = document.getLineNumber(caretModel.offset)
  return document.getText(TextRange(document.getLineStartOffset(lineNumber), document.getLineEndOffset(lineNumber)))
}

private fun Editor.currentLineReplacementStart(): Int {
  val document = this.document
  val lineNumber = document.getLineNumber(caretModel.offset)
  val lineStart = document.getLineStartOffset(lineNumber)
  val lineEnd = document.getLineEndOffset(lineNumber)
  val lineText = document.getText(TextRange(lineStart, lineEnd))
  val indentLength = lineText.indexOfFirst { !it.isWhitespace() }.let { if (it == -1) lineText.length else it }
  return lineStart + indentLength
}

private fun Editor.replaceLineSuffix(replacementStart: Int, completion: String) {
  val document = this.document
  val lineNumber = document.getLineNumber(replacementStart.coerceAtMost(document.textLength))
  val lineEnd = document.getLineEndOffset(lineNumber)

  document.replaceString(replacementStart, lineEnd, completion)
  caretModel.moveToOffset(replacementStart + completion.length)
}
