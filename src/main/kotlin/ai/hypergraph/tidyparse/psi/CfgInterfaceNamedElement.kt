package ai.hypergraph.tidyparse.psi

import com.intellij.psi.PsiNameIdentifierOwner

interface CfgInterfaceNamedElement : PsiNameIdentifierOwner {
  fun getKey(): String?

  fun getValue(): String?
}