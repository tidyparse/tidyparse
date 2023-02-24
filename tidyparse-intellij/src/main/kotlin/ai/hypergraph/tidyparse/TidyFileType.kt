package ai.hypergraph.tidyparse

import com.intellij.openapi.fileTypes.FileType
import com.intellij.openapi.util.IconLoader
import javax.swing.Icon

class TidyFileType: FileType {
  override fun getName() = "Tidy File"

  override fun getDescription() = "Tidy description"

  override fun getDefaultExtension() = "tidy"

  override fun getIcon(): Icon = IconLoader.getIcon("META-INF/pluginIcon.png", javaClass.classLoader)

  override fun isBinary() = false
}