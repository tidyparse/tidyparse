package ai.hypergraph.tidyparse.cfg

import com.intellij.openapi.editor.*
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.editor.colors.TextAttributesKey.createTextAttributesKey
import com.intellij.openapi.fileTypes.SyntaxHighlighterBase
import com.intellij.psi.TokenType
import com.intellij.psi.tree.IElementType
import ai.hypergraph.tidyparse.psi.CfgInterfaceTypes

class CfgInterfaceSyntaxHighlighter : SyntaxHighlighterBase() {
  val SEPARATOR = createTextAttributesKey("ROS_INTERFACE_SEPARATOR", DefaultLanguageHighlighterColors.OPERATION_SIGN)
  val TYPE = createTextAttributesKey("ROS_INTERFACE_TYPE", DefaultLanguageHighlighterColors.KEYWORD)
  val KEY = createTextAttributesKey("ROS_INTERFACE_KEY", DefaultLanguageHighlighterColors.IDENTIFIER)
  val VALUE = createTextAttributesKey("ROS_INTERFACE_VALUE", DefaultLanguageHighlighterColors.STRING)
  val COMMENT = createTextAttributesKey("ROS_INTERFACE_COMMENT", DefaultLanguageHighlighterColors.LINE_COMMENT)
  val BAD_CHARACTER = createTextAttributesKey("ROS_INTERFACE_BAD_CHARACTER", HighlighterColors.BAD_CHARACTER)

  private val BAD_CHAR_KEYS = arrayOf(BAD_CHARACTER)
  private val SEPARATOR_KEYS = arrayOf(SEPARATOR)
  private val TYPE_KEYS = arrayOf(TYPE)
  private val KEY_KEYS = arrayOf(KEY)
  private val VALUE_KEYS = arrayOf(VALUE)
  private val COMMENT_KEYS = arrayOf(COMMENT)
  private val EMPTY_KEYS = arrayOf<TextAttributesKey>()

  override fun getHighlightingLexer() = CfgInterfaceLexerAdapter()

  override fun getTokenHighlights(tokenType: IElementType) =
    when (tokenType) {
      CfgInterfaceTypes.SEPARATOR -> SEPARATOR_KEYS
      CfgInterfaceTypes.KEY -> KEY_KEYS
      CfgInterfaceTypes.VALUE -> VALUE_KEYS
      CfgInterfaceTypes.COMMENT -> COMMENT_KEYS
      TokenType.BAD_CHARACTER -> BAD_CHAR_KEYS
      else -> EMPTY_KEYS
    }
}