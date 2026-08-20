import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CppCompletionFormattingTest {
  @Test
  fun batchExtractionUsesClangFormatOutputForDisplayAndWholeStatementReplacement() {
    val raw = "    Shape & mutable_view = const_cast < Shape & > ( view ) ;"
    val batch = cppCompletionFormatBatch(listOf(raw), nonce = 17)
    val formattedScratch = batch.source.replace(
      "    Shape & mutable_view = const_cast < Shape & > ( view ) ;",
      "    Shape& mutable_view = const_cast<Shape&>(view);"
    )

    val completion = assertNotNull(extractCppFormattedCompletions(batch, formattedScratch)).single()
    assertEquals("    Shape& mutable_view = const_cast<Shape&>(view);", completion.replacementText)
    assertEquals("Shape& mutable_view = const_cast<Shape&>(view);", completion.displayText)
  }

  @Test
  fun extractionPreservesRelativeClangFormatIndentationForMultilineStatements() {
    val batch = cppCompletionFormatBatch(listOf("\tif(flag){run();}"), nonce = 23)
    val formattedScratch = batch.source.replace(
      "    if(flag){run();}",
      "    if (flag) {\n        run();\n    }"
    )

    val completion = assertNotNull(extractCppFormattedCompletions(batch, formattedScratch)).single()
    assertEquals("\tif (flag) {\n\t    run();\n\t}", completion.replacementText)
    assertEquals("if (flag) { run(); }", completion.displayText)
  }

  @Test
  fun lspFormattingEditsUseUtf16CoordinatesAndApplyFromTheEnd() {
    val source = "😀x\r\nbeta=2;\n"
    val result = applyCppFormatTextEdits(
      source,
      listOf(
        CppFormatTextEdit(
          CppFormatPosition(0, 2),
          CppFormatPosition(0, 3),
          "value"
        ),
        CppFormatTextEdit(
          CppFormatPosition(1, 4),
          CppFormatPosition(1, 5),
          " = "
        )
      )
    )

    assertEquals("😀value\r\nbeta = 2;\n", result)
  }

  @Test
  fun malformedMarkersAndOverlappingEditsAreRejected() {
    val batch = cppCompletionFormatBatch(listOf("value;"), nonce = 29)
    assertNull(extractCppFormattedCompletions(batch, batch.source.replace(batch.endMarkers.single(), "")))
    assertNull(
      applyCppFormatTextEdits(
        "value;",
        listOf(
          CppFormatTextEdit(CppFormatPosition(0, 0), CppFormatPosition(0, 3), "first"),
          CppFormatTextEdit(CppFormatPosition(0, 2), CppFormatPosition(0, 5), "second")
        )
      )
    )
  }

  @Test
  fun installedStyleDelegatesAllCompletionWhitespaceToClangFormat() {
    assertTrue("BasedOnStyle: LLVM" in CPP_CLANG_FORMAT_CONFIGURATION)
    assertTrue("Standard: Latest" in CPP_CLANG_FORMAT_CONFIGURATION)
    assertTrue("PointerAlignment: Left" in CPP_CLANG_FORMAT_CONFIGURATION)
    assertTrue("ReferenceAlignment: Pointer" in CPP_CLANG_FORMAT_CONFIGURATION)
    assertTrue("ColumnLimit: 0" in CPP_CLANG_FORMAT_CONFIGURATION)
  }
}
