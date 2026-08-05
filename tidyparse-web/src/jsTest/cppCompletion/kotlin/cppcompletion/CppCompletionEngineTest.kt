package cppcompletion

import cppEditorStatementSnapshot
import completionQuery
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppCompletionEngineTest {
  @Test
  fun sharedEntryPointOwnsGenerationSamplingAndEditorRendering() {
    val prefixText = "std::"
    val snapshot = assertNotNull(cppEditorStatementSnapshot(prefixText, 0, prefixText.length))
    val prefix = snapshot.stableTokens
    val context = CppCompletionContext(
      identifiers = setOf("std", "cout", "value"),
      sourceIdentifiers = setOf("std", "cout", "value"),
      headers = setOf("iostream"),
      values = listOf(
        CppReference("std::cout", type = "std::ostream", kind = "variable"),
        CppReference("value", type = "int", kind = "variable")
      )
    )
    val query = snapshot.completionQuery(context.identifiers, seed = 20260804)

    val prepared = CppCompletionGrammar().prepare(context, prefix)
    val cachedPath = prepared.completeCppStatement(query)
    val uncachedPath = CppCompletionGrammar().completeCppStatement(context, query)

    assertEquals(cachedPath.suggestions, uncachedPath.suggestions)
    assertEquals(cachedPath.minimumTokenLength, uncachedPath.minimumTokenLength)
    assertTrue(cachedPath.suggestions.isNotEmpty())
    assertTrue(cachedPath.suggestions.all { suggestion ->
      suggestion.candidateText.startsWith(prefixText) &&
        suggestion.tokenLength == suggestion.tokens.size
    })
    assertEquals("std::cout;", cachedPath.suggestions.first().candidateText)
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
      snapshot.tokens.size > CPP_MAX_STATEMENT_TOKENS,
      "The regression prefix must exceed the semantic lane's finite token horizon"
    )

    val execution = CppCompletionGrammar().completeCppStatement(
      context = CppCompletionContext(emptySet()),
      query = snapshot.completionQuery(
        snapshot.tokens.mapTo(linkedSetOf(), CppToken::text), seed = 20260804
      )
    )

    assertEquals(2, execution.minimumTokenLength)
    assertTrue(
      execution.suggestions.any { it.tokens == listOf("value49", ";") },
      "The syntax lane must close a valid long expression when the semantic lane is out of range"
    )
  }

  @Test
  fun partialIdentifierUsesTheFullGrammarInsteadOfPostFilteringSamples() {
    val source = "return payload;"
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 0, "return payl".length))
    val context = CppCompletionContext(
      identifiers = setOf("payload"),
      sourceIdentifiers = setOf("payload"),
      values = listOf(CppReference("payload", type = "int")),
      enclosingReturnType = "int"
    )

    val execution = CppCompletionGrammar().completeCppStatement(
      context,
      snapshot.completionQuery(context.identifiers, seed = 7)
    )

    assertTrue(execution.suggestions.isNotEmpty())
    assertTrue(execution.suggestions.all { it.candidateText.startsWith("return payl") })
    assertTrue(execution.suggestions.any { it.candidateText == "return payload;" })
    assertEquals(source.length, snapshot.replacementEndCharacter)
  }

  @Test
  fun initialStdFragmentFindsKnownAndSyntacticCompletions() {
    val context = CppCompletionContext(
      identifiers = setOf("std", "cout"),
      sourceIdentifiers = setOf("std", "cout"),
      headers = setOf("iostream", "string"),
      typeNames = setOf("std::string"),
      values = listOf(CppReference("std::cout", type = "std::ostream"))
    )
    val snapshot = assertNotNull(cppEditorStatementSnapshot("st", 0, 2))
    val execution = CppCompletionGrammar().completeCppStatement(
      context,
      snapshot.completionQuery(context.identifiers, seed = 11)
    )

    assertTrue(execution.suggestions.isNotEmpty())
    assertTrue(execution.suggestions.all { it.candidateText.startsWith("st") })
    assertTrue(execution.suggestions.any { it.candidateText.startsWith("std::") })
  }

  @Test
  fun qualifiedPartialTypeUsesSemanticIdentifierSpellings() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("std::str", 0, "std::str".length))
    val context = CppCompletionContext(
      identifiers = setOf("std", "string"),
      sourceIdentifiers = setOf("std", "string"),
      headers = setOf("string"),
      typeNames = setOf("std::string")
    )
    val execution = CppCompletionGrammar().completeCppStatement(
      context,
      snapshot.completionQuery(context.identifiers, seed = 13)
    )

    assertTrue(execution.suggestions.all { it.candidateText.startsWith("std::str") })
    assertTrue(execution.suggestions.any { it.candidateText.startsWith("std::string") })
  }

  @Test
  fun generatedSyntaxCompletesPartialFixedTokensWithoutSemanticFacts() {
    mapOf(
      "ret" to "return",
      "co_r" to "co_return",
      "tru" to "true",
      "nullp" to "nullptr"
    ).forEach { (typed, completed) ->
      val snapshot = assertNotNull(cppEditorStatementSnapshot(typed, 0, typed.length))
      val suggestions = CppCompletionGrammar().completeCppStatement(
        CppCompletionContext(emptySet()),
        snapshot.completionQuery(emptySet(), seed = typed.hashCode())
      ).suggestions

      assertTrue(
        suggestions.any { it.candidateText.startsWith(completed) },
        "Generated statement syntax did not extend `$typed` to `$completed`: " +
          suggestions.joinToString { it.candidateText }
      )
    }
  }
}
