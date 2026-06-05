package ai.hypergraph.tidyparse

import ai.hypergraph.tidyparse.template.BaseTest
import com.intellij.util.ui.UIUtil
class TidyRepairActionTest : BaseTest() {
  fun testSelectedCompletionReplacesIndentedLineSuffix() {
    myFixture.configureByText("scratch.txt", "    yield (<caret>")

    showTidyWebGpuCompletionLookup(project, myFixture.editor, listOf("yield ( ) NEWLINE"))
    UIUtil.dispatchAllInvocationEvents()

    assertEquals(listOf("yield ( ) NEWLINE"), myFixture.lookupElementStrings)

    myFixture.type('\n')
    UIUtil.dispatchAllInvocationEvents()

    assertEquals("    yield ( ) NEWLINE", myFixture.editor.document.text)
  }
}
