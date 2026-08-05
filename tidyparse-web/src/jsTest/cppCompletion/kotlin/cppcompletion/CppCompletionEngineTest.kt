package cppcompletion

import cppEditorStatementSnapshot
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppCompletionEngineTest {
  @Test
  fun sharedEntryPointOwnsGenerationSamplingAndEditorRendering() {
    val prefixText = "std::"
    val prefix = cppLines(prefixText).single().tokens
    val context = CppCompletionContext(
      identifiers = setOf("std", "cout", "value"),
      sourceIdentifiers = setOf("std", "cout", "value"),
      headers = setOf("iostream"),
      values = listOf(
        CppReference("std::cout", type = "std::ostream", kind = "variable"),
        CppReference("value", type = "int", kind = "variable")
      )
    )
    val query = CppCompletionQuery(
      prefix = prefix,
      prefixText = prefixText,
      identifiersInFile = context.identifiers,
      seed = 20260804
    )

    val prepared = CppCompletionGrammar().prepare(context, prefix)
    val cachedPath = prepared.completeCppStatement(query)
    val uncachedPath = CppCompletionGrammar().completeCppStatement(context, query)

    assertEquals(cachedPath.suggestions, uncachedPath.suggestions)
    assertEquals(cachedPath.minimumTokenLength, uncachedPath.minimumTokenLength)
    assertTrue(cachedPath.suggestions.isNotEmpty())
    assertTrue(cachedPath.suggestions.all { suggestion ->
      suggestion.insertionText.isNotEmpty() &&
        suggestion.candidateText.startsWith(prefixText) &&
        suggestion.tokenLength == suggestion.tokens.size
    })
    assertEquals("std::cout;", cachedPath.suggestions.first().candidateText)
    assertEquals("cout;", cachedPath.suggestions.first().insertionText)
  }

  @Test
  fun productionQueryRejectsTransportShapesTheWorkerCannotSafelyExecute() {
    assertFailsWith<IllegalArgumentException> {
      CppCompletionQuery(emptyList(), "first\nsecond", emptySet())
    }
    assertFailsWith<IllegalArgumentException> {
      CppCompletionQuery(emptyList(), "", emptySet(), limit = 0)
    }
    assertFailsWith<IllegalArgumentException> {
      CppCompletionQuery(
        emptyList(), "", emptySet(), limit = CPP_MAX_INTERACTIVE_COMPLETIONS + 1
      )
    }
  }

  @Test
  fun productionPathCompletesAStatementPrefixBeyondTheSemanticTokenHorizon() {
    val prefixText = (0..49).joinToString(", ") { index -> "value$index" }
    val snapshot = assertNotNull(cppEditorStatementSnapshot(prefixText, 0, prefixText.length))
    assertTrue(
      snapshot.projectedTokens.size > CPP_MAX_STATEMENT_TOKENS,
      "The regression prefix must exceed the semantic lane's finite token horizon"
    )

    val execution = CppCompletionGrammar().completeCppStatement(
      context = CppCompletionContext(emptySet()),
      query = CppCompletionQuery(
        prefix = snapshot.tokens,
        prefixText = snapshot.prefixText,
        identifiersInFile = snapshot.tokens.mapTo(linkedSetOf(), CppToken::text),
        seed = 20260804
      )
    )

    assertEquals(1, execution.minimumTokenLength)
    assertTrue(
      execution.suggestions.any { it.tokens == listOf(";") },
      "The syntax lane must close a valid long expression when the semantic lane is out of range"
    )
  }
}
