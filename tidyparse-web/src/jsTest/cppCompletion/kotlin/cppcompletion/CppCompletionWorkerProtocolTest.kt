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
    val semantic = js(
      """({
        schemaVersion: 1,
        context: { kind: 'Expression' },
        items: [{
          name: 'value',
          insertText: 'value',
          kind: 6,
          symbols: [{
            id: 'c:@value',
            qualifiedName: 'value',
            kind: 'VarDecl',
            provenance: { sema: true, index: false },
            isValue: true,
            type: 'int',
            canonicalType: 'int'
          }]
        }]
      })"""
    )
    val request = cppCompletionWorkerRequest(
      cacheKey = "main.cpp@7:4",
      snapshot = snapshot.copy(seed = 73),
      semantic = semantic,
      limit = 4
    )

    assertEquals("complete", request.type)
    assertEquals(0, request.id)
    assertEquals("main.cpp@7:4", request.cacheKey)
    assertEquals("  value =", request.statementPrefixText)
    assertEquals("  value =", request.semanticPrefixText)
    assertEquals(4, request.limit)
    assertEquals("IDENTIFIER", request.prefixTokens[0].kind)
    assertEquals(2, request.prefixTokens[0].start)
    assertEquals("=", request.prefixTokens[1].text)
    assertTrue(js("Array.isArray(request.prefixTokens)") as Boolean)
    assertFalse(
      request.semantic === semantic,
      "Semantic facts must be converted to an owned plain DTO"
    )
    assertEquals(1, request.semantic.schemaVersion)
    assertEquals("value", request.semantic.items[0].name)
    assertEquals("int", request.semantic.items[0].symbols[0].canonicalType)
    assertTrue(JSON.stringify(request).isNotBlank())
  }

  @Test
  fun requestHelperRejectsAnUnsafeInteractiveShape() {
    val snapshot = requireNotNull(cppEditorStatementSnapshot("return", 0, 6))
    val semantic = js("({ schemaVersion: 1, items: [] })")
    assertFailsWith<IllegalArgumentException> {
      cppCompletionWorkerRequest("key", snapshot, semantic, limit = 11)
    }
    assertFailsWith<IllegalArgumentException> {
      cppCompletionWorkerRequest("key", snapshot.copy(prefixText = "first\nsecond"), semantic)
    }
  }

  @Test
  fun workerRehydratesStructuredClonePrefixTokens() {
    val source = "std::string"
    val character = "std::str".length
    val snapshot = requireNotNull(cppEditorStatementSnapshot(source, 0, character))
    val request = cppCompletionWorkerRequest(
      cacheKey = "main.cpp@3:9",
      snapshot = snapshot.copy(seed = 7),
      semantic = js("({ schemaVersion: 1, items: [] })")
    )

    // JSON cloning mirrors the browser Worker structured-clone boundary while ensuring that the
    // parser receives ordinary JavaScript objects with no Kotlin prototype or helper methods.
    val cloned = cppCompletionJsonClone(request)
    val tokens = cppCompletionPrefixTokens(cloned.prefixTokens, cloned.statementPrefixText)

    assertEquals(3, tokens.size)
    assertEquals(CppToken("std", 0, 3, CppTokenKind.IDENTIFIER), tokens[0])
    assertEquals(CppToken("::", 3, 5, CppTokenKind.OTHER), tokens[1])
    assertEquals(CppToken("str", 5, 8, CppTokenKind.IDENTIFIER, "string"), tokens[2])
    assertEquals("str", snapshot.activeFragment?.text)
    assertEquals(listOf("std", "::"), snapshot.stableTokens.map(CppToken::text))
    assertEquals("std::", snapshot.stablePrefixText)
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
  fun cppEditorUsesOnlyExplicitCompletionWithoutBracketPairInsertion() {
    val options = cppMonacoEditorOptions()
    assertEquals("never", options.autoClosingBrackets)
    assertEquals(false, options.quickSuggestions)
    assertEquals(false, options.suggestOnTriggerCharacters)
  }
}
