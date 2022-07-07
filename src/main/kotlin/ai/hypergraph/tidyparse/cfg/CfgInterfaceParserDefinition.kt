package ai.hypergraph.tidyparse.cfg

import com.intellij.lang.*
import com.intellij.lang.ParserDefinition.SpaceRequirements
import com.intellij.openapi.project.Project
import com.intellij.psi.*
import com.intellij.psi.tree.*
import ai.hypergraph.tidyparse.psi.*

object CfgInterfaceParserDefinition : ParserDefinition {
  val WHITE_SPACES = TokenSet.create(TokenType.WHITE_SPACE)
  val COMMENTS = TokenSet.create(CfgInterfaceTypes.COMMENT)
  val FILE = IFileElementType(CfgInterfaceLanguage)

  override fun createLexer(project: Project) = CfgInterfaceLexerAdapter()

  override fun getWhitespaceTokens(): TokenSet = WHITE_SPACES

  override fun getCommentTokens(): TokenSet = COMMENTS

  override fun getStringLiteralElements(): TokenSet = TokenSet.EMPTY

  override fun createParser(project: Project) = CfgInterfaceParser()

  override fun getFileNodeType(): IFileElementType = FILE

  override fun createFile(viewProvider: FileViewProvider): PsiFile = CfgInterfaceFile(viewProvider)

  override fun spaceExistenceTypeBetweenTokens(left: ASTNode, right: ASTNode) = SpaceRequirements.MAY

  override fun createElement(node: ASTNode): PsiElement = CfgInterfaceTypes.Factory.createElement(node)
}