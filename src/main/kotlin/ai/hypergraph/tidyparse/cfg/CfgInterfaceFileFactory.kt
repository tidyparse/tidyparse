package ai.hypergraph.tidyparse.cfg

import com.intellij.openapi.fileTypes.*

object CfgInterfaceFileFactory : FileTypeFactory() {
  const val ROSINTERFACE_EXTENSIONS = "msg;srv"

  override fun createFileTypes(consumer: FileTypeConsumer) = consumer.consume(CfgInterfaceFileType, ROSINTERFACE_EXTENSIONS)
}