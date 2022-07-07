package ai.hypergraph.tidyparse.cfg

import com.intellij.openapi.fileTypes.LanguageFileType
import com.intellij.util.PlatformIcons

object CfgInterfaceFileType : LanguageFileType(CfgInterfaceLanguage) {
  override fun getName() = "CFG"

  override fun getDescription() = "Context Free Grammar"

  override fun getDefaultExtension() = "cfg"

  override fun getIcon() = PlatformIcons.JAR_ICON
}