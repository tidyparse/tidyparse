package ai.hypergraph.tidyparse

import ai.hypergraph.tidyparse.template.BaseTest
import com.intellij.util.ui.UIUtil
class TidyRepairActionTest : BaseTest() {
  fun testSelectedPythonRepairReplacesLineSuffix() {
    myFixture.configureByText("scratch.py", "x=P(1-)<caret>")

    showTidyPythonRepairLookup(project, myFixture.editor, myFixture.file.fileType, listOf(
      "x=P ( )",
      "x = P( NAME )",
      "x = P( 1 )",
      "x = P( STRING )",
      "x = P",
      "x . P( 1 )",
      "x = P( 1 , )",
      "x = P( NAME , )",
      "x = ( 1 )",
      "x = P( STRING , )"
    ))
    UIUtil.dispatchAllInvocationEvents()

    assertEquals(listOf(
      "x = P()",
      "x = P(NAME)",
      "x = P(1)",
      "x = P(STRING)",
      "x = P",
      "x.P(1)",
      "x = P(1, )",
      "x = P(NAME, )",
      "x = (1)",
      "x = P(STRING, )"
    ), myFixture.lookupElementStrings)

    myFixture.type('\n')
    UIUtil.dispatchAllInvocationEvents()

    assertEquals("x = P()", myFixture.editor.document.text)
  }

  fun testFormattedMultilineRepairStaysOneCompletion() {
    myFixture.configureByText("scratch.py", "x<caret>")

    showTidyPythonRepairLookup(project, myFixture.editor, myFixture.file.fileType, listOf(
      "x=1; y=2",
      "z=3"
    ))
    UIUtil.dispatchAllInvocationEvents()

    assertEquals(listOf("x = 1;\ny = 2", "z = 3"), myFixture.lookupElementStrings)

    myFixture.type('\n')
    UIUtil.dispatchAllInvocationEvents()

    assertEquals("x = 1;\ny = 2", myFixture.editor.document.text)
  }

  fun testFormattedDuplicateRepairsAreCollapsed() {
    myFixture.configureByText("scratch.py", "x<caret>")

    showTidyPythonRepairLookup(project, myFixture.editor, myFixture.file.fileType, listOf(
      "x=P(1)",
      "x = P(1)",
      "x=P(1,)",
      "x = P(1,)"
    ))
    UIUtil.dispatchAllInvocationEvents()

    assertEquals(listOf("x = P(1)", "x = P(1, )"), myFixture.lookupElementStrings)
  }

  fun testJcefNewlineSentinelsAreRemovedAndResultsAreLimited() {
    myFixture.configureByText("scratch.py", "x=P(1-)<caret>")

    showTidyPythonRepairLookup(project, myFixture.editor, myFixture.file.fileType, listOf(
      "x=P ( ) NEWLINE",
      "x = P( 1 ) NEWLINE",
      "x = P( NAME ) NEWLINE",
      "x = P( STRING ) NEWLINE",
      "x = P NEWLINE",
      "x . P( 1 ) NEWLINE",
      "x = P( 1 , ) NEWLINE",
      "x = P( NAME , ) NEWLINE",
      "x = ( 1 ) NEWLINE",
      "x = P( STRING , ) NEWLINE",
      "x = 11 NEWLINE",
      "x = 12 NEWLINE"
    ))
    UIUtil.dispatchAllInvocationEvents()

    val lookupStrings = myFixture.lookupElementStrings ?: emptyList()
    assertEquals(10, lookupStrings.size)
    assertFalse(lookupStrings.any { it.contains("NEWLINE") })
    assertEquals("x = P()", lookupStrings.first())

    myFixture.type('\n')
    UIUtil.dispatchAllInvocationEvents()

    assertEquals("x = P()", myFixture.editor.document.text)
  }
}
