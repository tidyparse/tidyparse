import cppcompletion.CppToken
import cppcompletion.CppTokenKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppCompletionWorkerProtocolTest {
  @Test
  fun requestHelperPublishesOnlyPlainStructuredCloneFields() {
    val source = "  value ="
    val snapshot = requireNotNull(cppEditorStatementSnapshot(source, 0, source.length))
    val completion = js("({ items: [{ label: 'value', detail: 'int', kind: 6 }] })")
    val request = cppCompletionWorkerRequest(
      cacheKey = "main.cpp@7:4",
      source = source,
      snapshot = snapshot.copy(seed = 73),
      facts = CppCompletionSemanticFacts(
        completionGroups = listOf(CppClangdCompletionGroup(completion))
      ),
      limit = 4
    )

    assertEquals("complete", request.type)
    assertEquals(0, request.id)
    assertEquals("main.cpp@7:4", request.cacheKey)
    assertEquals("  value =", request.statementPrefixText)
    assertEquals(source, request.source)
    assertEquals(4, request.limit)
    assertEquals("IDENTIFIER", request.prefixTokens[0].kind)
    assertEquals(2, request.prefixTokens[0].start)
    assertEquals("=", request.prefixTokens[1].text)
    assertTrue(js("Array.isArray(request.prefixTokens)") as Boolean)
    assertTrue(js("Array.isArray(request.facts.completionGroups)") as Boolean)
    assertFalse(
      request.facts.completionGroups[0].result === completion,
      "Semantic facts must be converted to an owned plain DTO"
    )
    assertEquals("value", request.facts.completionGroups[0].result.items[0].label)
    assertTrue(JSON.stringify(request).isNotBlank())
  }

  @Test
  fun requestHelperRejectsAnUnsafeInteractiveShape() {
    val snapshot = requireNotNull(cppEditorStatementSnapshot("return", 0, 6))
    assertFailsWith<IllegalArgumentException> {
      cppCompletionWorkerRequest("key", "return", snapshot, limit = 11)
    }
    assertFailsWith<IllegalArgumentException> {
      cppCompletionWorkerRequest("key", "first\nsecond", snapshot.copy(prefixText = "first\nsecond"))
    }
  }

  @Test
  fun workerRehydratesStructuredClonePrefixTokens() {
    val source = "value ="
    val snapshot = requireNotNull(cppEditorStatementSnapshot(source, 0, source.length))
    val request = cppCompletionWorkerRequest(
      cacheKey = "main.cpp@3:9",
      source = source,
      snapshot = snapshot.copy(seed = 7)
    )

    // JSON cloning mirrors the browser Worker structured-clone boundary while ensuring that the
    // parser receives ordinary JavaScript objects with no Kotlin prototype or helper methods.
    val cloned = cppCompletionJsonClone(request)
    val tokens = cppCompletionPrefixTokens(cloned.prefixTokens, cloned.statementPrefixText)

    assertEquals(2, tokens.size)
    assertEquals(CppToken("value", 0, 5, CppTokenKind.IDENTIFIER), tokens[0])
    assertEquals(CppToken("=", 6, 7, CppTokenKind.OTHER), tokens[1])
  }

  @Test
  fun completionWidgetWidthFitsLongestLabelWithoutLeavingViewport() {
    assertEquals(
      620.0,
      cppCompletionWidgetTargetWidth(
        longestLabelCharacters = 60,
        typicalHalfwidthCharacterWidth = 9.0,
        currentWidth = 430.0,
        maximumWidth = 1_200.0
      )
    )
    assertEquals(
      430.0,
      cppCompletionWidgetTargetWidth(10, 9.0, currentWidth = 430.0, maximumWidth = 1_200.0)
    )
    assertEquals(
      700.0,
      cppCompletionWidgetTargetWidth(100, 9.0, currentWidth = 430.0, maximumWidth = 700.0)
    )
  }

  @Test
  fun cppEditorDisablesMonacoBracketPairInsertion() {
    assertEquals("never", cppMonacoEditorOptions().autoClosingBrackets)
  }
}
