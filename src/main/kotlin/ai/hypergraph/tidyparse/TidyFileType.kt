package ai.hypergraph.tidyparse

import com.intellij.openapi.fileTypes.FileType
import com.intellij.util.PlatformIcons

object TidyFileType: FileType {
  override fun getName() = "Tidy File"

  override fun getDescription() = "Tidy description"

  override fun getDefaultExtension() = "tidy"

  override fun getIcon() = PlatformIcons.EDIT

  override fun isBinary() = false
}