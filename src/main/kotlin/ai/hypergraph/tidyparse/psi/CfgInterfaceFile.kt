package ai.hypergraph.tidyparse.psi

import ai.hypergraph.tidyparse.cfg.CfgInterfaceFileType
import ai.hypergraph.tidyparse.cfg.CfgInterfaceLanguage
import com.intellij.extapi.psi.PsiFileBase
import com.intellij.psi.FileViewProvider
import com.intellij.util.PlatformIcons

class CfgInterfaceFile(viewProvider: FileViewProvider) : PsiFileBase(viewProvider, CfgInterfaceLanguage) {
  override fun getFileType() = CfgInterfaceFileType

  override fun toString() = "CFG Interface File"

  override fun getIcon(flags: Int) = PlatformIcons.JAR_ICON
}