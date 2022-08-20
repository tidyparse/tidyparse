package ai.hypergraph.tidyparse

import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.content.ContentFactory
import java.awt.Font
import javax.swing.JTextPane

class TidyToolWindowFactory : ToolWindowFactory {
  override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) =
    toolWindow.contentManager.addContent(
      ContentFactory.SERVICE.getInstance().createContent(TidyToolWindow.panel, "", false)
    )
}

object TidyToolWindow {
  val textArea = JTextPane().apply {
    contentType = "text/html"
    font = Font("JetBrains Mono", Font.PLAIN, 16)
    isEditable = false
  }

  var text: String
    get() = textArea.text
    set(s) { textArea.text = s }

  val panel = JBScrollPane(textArea)
}