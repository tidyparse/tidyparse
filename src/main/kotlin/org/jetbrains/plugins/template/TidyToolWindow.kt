package org.jetbrains.plugins.template

import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.content.ContentFactory.SERVICE
import java.awt.Font
import javax.swing.JTextArea

class TidyToolWindowFactory : ToolWindowFactory {
  override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) =
    toolWindow.contentManager.addContent(
      SERVICE.getInstance().createContent(TidyToolWindow.panel, "", false)
    )
}

object TidyToolWindow {
  val textArea = JTextArea().apply {
    font = Font("JetBrains Mono", Font.PLAIN, 16)
    isEditable = false
  }

  val panel = JBScrollPane(textArea)
}