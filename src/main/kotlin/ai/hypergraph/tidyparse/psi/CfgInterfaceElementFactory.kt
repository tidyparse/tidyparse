package ai.hypergraph.tidyparse.psi

import ai.hypergraph.tidyparse.cfg.CfgInterfaceFileType
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFileFactory

object CfgInterfaceElementFactory {
  fun createProperty(project: Project, name: String, value: String): CfgInterfaceProperty {
    val file = createFile(project, "$name = $value")
    return file.firstChild as CfgInterfaceProperty
  }

  fun createProperty(project: Project, name: String): CfgInterfaceProperty {
    val file = createFile(project, name)
    return file.firstChild as CfgInterfaceProperty
  }

  fun createCRLF(project: Project) = createFile(project, "\n").firstChild!!

  fun createFile(project: Project, text: String): CfgInterfaceFile {
    val name = "dummy.CfgInterface"
    return PsiFileFactory.getInstance(project).createFileFromText(name, CfgInterfaceFileType, text) as CfgInterfaceFile
  }
}