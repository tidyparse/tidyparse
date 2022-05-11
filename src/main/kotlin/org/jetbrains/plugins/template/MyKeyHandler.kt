package org.jetbrains.plugins.template

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.parse
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.sat.synthesizeFromFPSolving
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import org.jetbrains.concurrency.runAsync

class MyKeyHandler : TypedHandlerDelegate() {
    lateinit var cfg: CFG
    var grammarLastModified = 0L
    val ok = "✅ Current line parses\n"
    val no = "❌ Current line invalid\n"

    override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile): Result {
        if (!file.name.endsWith(".tidy")) return CONTINUE
        val grammarFile = file.containingDirectory.files
            .firstOrNull { it.name.endsWith(".cfg") } ?: return CONTINUE

            //NotificationGroupManager.getInstance()
            //    .getNotificationGroup("Custom Notification Group")
            //    .createNotification(file.text.ifEmpty { "ε" }, NotificationType.INFORMATION)
            //    .notify(project)

        runAsync {
            if (grammarFile.modificationStamp != grammarLastModified)
                cfg = ReadAction.compute<String, Exception> { grammarFile.text }.parseCFG()

            val (lineStart, lineEnd) = editor.caretModel.let { it.visualLineStart to it.visualLineEnd }
            val currentLine = editor.document.getText(TextRange.create(lineStart, lineEnd))

            if ("_" !in currentLine)
                cfg.parse(currentLine)?.let { MyToolWindow.textArea.text = ok + it.prettyPrint() }
            else currentLine.synthesizeFromFPSolving(cfg).take(20).toList().shuffled().let {
                if (it.isNotEmpty()) MyToolWindow.textArea.text = it.joinToString("\n")
            }
        }

        return CONTINUE
    }
}