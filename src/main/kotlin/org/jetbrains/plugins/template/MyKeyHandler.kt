package org.jetbrains.plugins.template

import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile

class MyKeyHandler : TypedHandlerDelegate() {
    override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile): Result {
        //if (file.name.endsWith(".tidy"))
            //val grammarFile = file.containingDirectory.files
            //    .firstOrNull { it.name.endsWith(".cfg") }
            //NotificationGroupManager.getInstance()
            //    .getNotificationGroup("Custom Notification Group")
            //    .createNotification(file.text.ifEmpty { "ε" }, NotificationType.INFORMATION)
            //    .notify(project)
        MyToolWindow.textField.text = editor.document.text
        //ToolWindowManager.getInstance(project).getToolWindow("Tidyparse").contentManager
        MyToolWindow.textField.repaint()
        return CONTINUE
    }
}