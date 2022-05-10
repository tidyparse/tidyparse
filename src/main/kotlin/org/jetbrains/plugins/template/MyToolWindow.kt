package org.jetbrains.plugins.template

import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.content.ContentFactory.SERVICE
import javax.swing.JTextField

class MyToolWindowFactory: ToolWindowFactory {
    override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
        val contentFactory = SERVICE.getInstance()
        val content = contentFactory.createContent(MyToolWindow.textField, "", false)
        toolWindow.contentManager.addContent(content)
    }
}

object MyToolWindow {
    val textField = JTextField()
}