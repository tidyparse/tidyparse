package ai.hypergraph.tidyparse

import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.util.TextRange
import java.awt.Color

fun Editor.currentLine(): String =
  caretModel.let { it.visualLineStart to it.visualLineEnd }
    .let { (lineStart, lineEnd) ->
      document.getText(TextRange.create(lineStart, lineEnd))
    }

val AnActionEvent.editor: Editor get() =
  CommonDataKeys.EDITOR.getData(dataContext)!!

fun generateColors(n: Int): List<Color> =
  (0 until n).map { i ->
    Color.getHSBColor(i.toFloat() / n.toFloat(), 0.85f, 1.0f)
  }