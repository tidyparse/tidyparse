package ai.hypergraph.tidyparse

import com.intellij.codeInsight.daemon.impl.HighlightVisitor
import com.intellij.codeInsight.daemon.impl.analysis.HighlightInfoHolder
import com.intellij.codeInsight.highlighting.HighlightManager
import com.intellij.openapi.application.WriteAction
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.editor.DefaultLanguageHighlighterColors
import com.intellij.openapi.editor.DefaultLanguageHighlighterColors.DOC_COMMENT
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.editor.colors.TextAttributesKey.createTempTextAttributesKey
import com.intellij.openapi.editor.colors.TextAttributesKey.createTextAttributesKey
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.intellij.psi.util.PsiEditorUtil
import java.awt.Color
import java.awt.Font
import kotlin.reflect.KProperty

class TidyHighlightVisitor : HighlightVisitor {
  private var highlightInfoHolder: HighlightInfoHolder? = null

  override fun suitableForFile(file: PsiFile) =
    file.fileType.defaultExtension == "tidy"

  private fun createAttributes(function: TextAttributes.() -> Unit) =
    DOC_COMMENT.defaultAttributes.clone().apply { function() }

  operator fun TextAttributes.getValue(tidyHighlightVisitor: TidyHighlightVisitor, property: KProperty<*>) =
    createTextAttributesKey(property.name, this)

  val BOLD_FONT by createAttributes { fontType = Font.BOLD }
  val BLUE_FONT by createAttributes { foregroundColor = Color.BLUE }
  val RED_FONT by createAttributes { foregroundColor = Color.RED }

  override fun visit(element: PsiElement) {
    invokeLater {
      val editor = PsiEditorUtil.findEditor(element)!!
      val off = editor.document.text.indexOf("---")
      val highlighter = HighlightManager.getInstance(editor.project)
      // Highlight BNF syntax
      Regex("(->| \\| )")
        .findAll(editor.document.text)
        .takeWhile { it.range.first < off }
        .forEach {
          highlighter.addRangeHighlight(
            editor, it.range.first, it.range.last + 1,
            RED_FONT, false, null
          )
        }

      // Highlight grammar
      highlighter.addRangeHighlight(editor, 0, off + 3,
        BOLD_FONT, false, null)

      // Highlight delimiter
      highlighter.addRangeHighlight(editor, off, off + 3,
        BLUE_FONT, false, null)
    }
  }

  override fun analyze(
    file: PsiFile,
    updateWholeFile: Boolean,
    holder: HighlightInfoHolder,
    action: Runnable
  ): Boolean {
    highlightInfoHolder = holder
    try { action.run() } catch (_: Exception) { }
    highlightInfoHolder = null
    return true
  }

  override fun clone() = TidyHighlightVisitor()
}