package ai.hypergraph.tidyparse.psi.impl

import ai.hypergraph.tidyparse.psi.CfgInterfaceElementFactory
import com.intellij.extapi.psi.ASTWrapperPsiElement
import com.intellij.lang.ASTNode
import com.intellij.psi.PsiElement
import ai.hypergraph.tidyparse.psi.CfgInterfaceNamedElement
import ai.hypergraph.tidyparse.psi.CfgInterfaceTypes.KEY
import ai.hypergraph.tidyparse.psi.CfgInterfaceTypes.VALUE

open class CfgInterfaceNamedElementImpl(node: ASTNode) : ASTWrapperPsiElement(node), CfgInterfaceNamedElement {
  override fun getKey() = node.findChildByType(KEY)?.text?.replace("\\\\ ".toRegex(), " ")

  override fun getValue() = node.findChildByType(VALUE)?.text

  override fun getName() = getKey()

  override fun setName(newName: String): PsiElement {
    val keyNode = node.findChildByType(KEY)
    if (keyNode != null) {
      val property = CfgInterfaceElementFactory.createProperty(project, newName)
      val newKeyNode = property.firstChild.node
      node.replaceChild(keyNode, newKeyNode)
    }

    return this
  }

  override fun getNameIdentifier() = node.findChildByType(KEY)?.psi
}