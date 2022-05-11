package org.jetbrains.plugins.template

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.parse
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.parsing.solveFixedpoint
import ai.hypergraph.kaliningraph.sat.synthesizeFromFPSolving
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.application.WriteAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import org.jetbrains.concurrency.runAsync

class MyKeyHandler : TypedHandlerDelegate() {
    lateinit var cfg: CFG
    var grammarLastModified = 0L
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

            if ("_" !in editor.document.text)
                cfg.parse(editor.document.text)?.let { MyToolWindow.textField.text = it.prettyPrint() }
            else editor.document.text.synthesizeFromFPSolving(cfg).take(40).toList().shuffled().let {
                if (it.isNotEmpty()) MyToolWindow.textField.text = it.joinToString("\n")
            }
        }

        return CONTINUE
    }
}