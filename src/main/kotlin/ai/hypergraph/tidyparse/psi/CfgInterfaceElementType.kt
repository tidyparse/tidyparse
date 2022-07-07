package ai.hypergraph.tidyparse.psi

import ai.hypergraph.tidyparse.cfg.CfgInterfaceLanguage
import com.intellij.psi.tree.IElementType
import org.jetbrains.annotations.NonNls

class CfgInterfaceElementType : IElementType {
  constructor(@NonNls debugName: String) : super(debugName, CfgInterfaceLanguage)
}