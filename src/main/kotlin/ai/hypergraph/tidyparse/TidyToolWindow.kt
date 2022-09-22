package ai.hypergraph.tidyparse

import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.content.ContentFactory
import org.jetbrains.plugins.notebooks.visualization.saveScrollingPosition
import java.awt.Font
import javax.swing.JTextPane
import javax.swing.text.DefaultCaret

class TidyToolWindowFactory : ToolWindowFactory {
  override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) =
    toolWindow.contentManager.addContent(
      ContentFactory.SERVICE.getInstance().createContent(TidyToolWindow.panel, "", false)
    )
}

object TidyToolWindow {
  private val textArea = JTextPane().apply {
    contentType = "text/html"
    font = Font("JetBrains Mono", Font.PLAIN, 16)
    isEditable = false
    (caret as DefaultCaret).updatePolicy = DefaultCaret.NEVER_UPDATE
  }

  @Volatile
  var lastUpdate = ""
  var text: String
    get() = lastUpdate
    set(s) {
      if (s.length != lastUpdate.length || s != lastUpdate) {
        lastUpdate = s
        invokeLater { textArea.text = s }
      }
    }

  val panel = JBScrollPane(textArea)//.apply { autoscrolls = false }
}