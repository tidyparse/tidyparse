package ai.hypergraph.tidyparse

import com.intellij.codeInsight.daemon.impl.HighlightVisitor
import com.intellij.codeInsight.daemon.impl.analysis.HighlightInfoHolder
import com.intellij.codeInsight.highlighting.HighlightManager
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.editor.DefaultLanguageHighlighterColors.DOC_COMMENT
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

  private fun createAttributes(function: TextAttributes.() -> Unit): TextAttributes =
    DOC_COMMENT.defaultAttributes.clone().apply { function() }

  operator fun TextAttributes.getValue(tidyHighlightVisitor: TidyHighlightVisitor, property: KProperty<*>) =
    createTextAttributesKey(property.name, createTempTextAttributesKey(property.name, this))

  val BOLD_FONT by createAttributes { fontType = Font.BOLD }
  val BLUE_FONT by createAttributes { foregroundColor = Color.BLUE }
  val RED_FONT by createAttributes { foregroundColor = Color.RED }

  fun RGB_FONT(color: Color, name: String = "${color.rgb}_font") =
    createTextAttributesKey(name, createTempTextAttributesKey(name,
      DOC_COMMENT.defaultAttributes.clone().apply { foregroundColor = color }))

  override fun visit(element: PsiElement) {
    invokeLater {
      val editor = PsiEditorUtil.findEditor(element)!!
      val divIdx = editor.document.text.indexOf("---")
      val highlighter = HighlightManager.getInstance(editor.project)
      // Highlight BNF syntax
      Regex("(->| \\| )")
        .findAll(editor.document.text)
        .takeWhile { it.range.first < divIdx }
        .forEach {
          highlighter.addRangeHighlight(
            editor, it.range.first, it.range.last + 1,
            RED_FONT, false, null
          )
        }

      // Highlight grammar
      highlighter.addRangeHighlight(editor, 0, divIdx + 3,
        BOLD_FONT, false, null)

//    val colorMap = cfg.terminals.zip(generateColors(cfg.terminals.size)).toMap()
//    colorMap.keys.forEach {
//      Regex(Regex.escape(it)).findAll(editor.document.text).forEach {
//        val font = colorMap[it.value]?.let { RGB_FONT(it) } ?: BOLD_FONT
//
//        // Highlight grammar
//        highlighter.addRangeHighlight(editor, it.range.first, it.range.last,
//          font, false, null)
//      }
//    }

      // Highlight delimiter
      highlighter.addRangeHighlight(editor, divIdx, divIdx + 3,
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