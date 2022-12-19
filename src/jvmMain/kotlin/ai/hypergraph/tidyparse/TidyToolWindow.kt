package ai.hypergraph.tidyparse

import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.components.JBScrollPane
import com.intellij.ui.content.ContentFactory
import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import javax.swing.JTextPane
import javax.swing.text.DefaultCaret
import javax.swing.text.DefaultCaret.NEVER_UPDATE

class TidyToolWindowFactory : ToolWindowFactory {
  override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) =
    toolWindow.contentManager.addContent(
      ContentFactory.getInstance().createContent(TidyToolWindow.panel, "", false)
    )
}

object TidyToolWindow {
  private val textArea =
    object: JTextPane() {
      override fun getPreferredSize(): Dimension {
        val (w, h) = super.getPreferredSize().let { it.width to it.height }
        return Dimension(w * fontScalingRatio.toInt(), h * fontScalingRatio.toInt())
      }

      override fun paint(g: Graphics?) {
        (g as? Graphics2D)?.scale(fontScalingRatio, fontScalingRatio)
        super.paintComponent(g)
        (g as? Graphics2D)?.scale(1.0, 1.0)
      }
    }.apply {
    contentType = "text/html"
    isEditable = false
    (caret as DefaultCaret).updatePolicy = NEVER_UPDATE
  }

  @Volatile
  var lastUpdate = ""
  var text: String
    get() = lastUpdate
    set(s) {
      if (s.length != lastUpdate.length || s != lastUpdate) {
        lastUpdate = s
        val t = Thread.currentThread()
        invokeLater { if(!t.isInterrupted) textArea.text = s }
      }
    }

  val panel = JBScrollPane(textArea)//.apply { autoscrolls = false }
}