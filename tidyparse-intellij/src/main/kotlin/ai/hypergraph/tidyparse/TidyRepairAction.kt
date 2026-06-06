package ai.hypergraph.tidyparse

import com.intellij.codeInsight.lookup.LookupElement
import com.intellij.codeInsight.lookup.LookupEvent
import com.intellij.codeInsight.lookup.LookupListener
import com.intellij.codeInsight.lookup.LookupManager
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.application.runWriteAction
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.fileTypes.FileType
import com.intellij.openapi.project.DumbAwareAction
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFileFactory
import com.intellij.psi.codeStyle.CodeStyleManager

class TidyRepairAction : DumbAwareAction() {
  override fun actionPerformed(e: AnActionEvent) {
    val project = e.project ?: return
    val editor = e.getData(CommonDataKeys.EDITOR) ?: return
    if (!e.isPythonFile()) return

    val fileType = e.getData(CommonDataKeys.PSI_FILE)?.fileType
      ?: e.getData(CommonDataKeys.VIRTUAL_FILE)?.fileType
      ?: return
    val line = editor.currentDocumentLine().trim()

    project.webGpuStringService().repairPythonLine(line).whenComplete { raw, err ->
      if (err != null) { err.showTidyParseError(project); return@whenComplete }
      ApplicationManager.getApplication().invokeLater {
        try {
          showTidyPythonRepairLookup(project, editor, fileType, raw.lines())
        } catch (t: Throwable) {
          t.showTidyParseError(project)
        }
      }
    }
  }

  override fun update(e: AnActionEvent) {
    e.presentation.isEnabledAndVisible =
      e.project != null && e.getData(CommonDataKeys.EDITOR) != null && e.isPythonFile()
  }
}

internal fun showTidyPythonRepairLookup(project: Project, editor: Editor, fileType: FileType, rawCompletions: Iterable<String>) {
  val replacementStart = editor.currentLineReplacementStart()
  val completions = runWriteAction {
    formatPythonCompletions(project, fileType, rawCompletions.asSequence()
      .mapNotNull { it.withoutTrailingGrammarNewline().takeIf(String::isNotBlank) }
      .take(50)
      .toList())
      .distinct()
      .take(10)
  }

  if (completions.isEmpty()) return

  println("TidyParse formatted completions:\n${completions.joinToString("\n")}")

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

private fun AnActionEvent.isPythonFile(): Boolean =
  getData(CommonDataKeys.VIRTUAL_FILE)?.extension.equals("py", ignoreCase = true) ||
    getData(CommonDataKeys.PSI_FILE)?.name?.endsWith(".py", ignoreCase = true) == true

private fun String.withoutTrailingGrammarNewline(): String =
  replace(Regex("""(?:\s+NEWLINE)+\s*$"""), "")

private fun formatPythonCompletions(project: Project, fileType: FileType, completions: List<String>): List<String> {
  if (completions.isEmpty()) return emptyList()
  val separatorPrefix = "# __tidyparse_repair_"
  val tempFile = PsiFileFactory.getInstance(project)
    .createFileFromText("tidyparse_repairs.py", fileType, completions.mapIndexed { index, completion ->
      "$separatorPrefix${index}__\n$completion"
    }.joinToString("\n"))
  CodeStyleManager.getInstance(project).reformat(tempFile)

  val formatted = mutableListOf<String>()
  val current = mutableListOf<String>()
  tempFile.text.lines().forEach { line ->
    if (line.trim().startsWith(separatorPrefix)) {
      if (current.isNotEmpty()) {
        formatted += current.joinToString("\n").trim()
        current.clear()
      }
    } else {
      current += line
    }
  }
  if (current.isNotEmpty()) formatted += current.joinToString("\n").trim()

  check(formatted.size == completions.size) {
    "Python formatter changed repair count from ${completions.size} to ${formatted.size}"
  }
  return formatted
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
