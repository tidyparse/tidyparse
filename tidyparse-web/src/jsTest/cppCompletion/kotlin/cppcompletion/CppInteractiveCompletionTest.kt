package cppcompletion

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CppInteractiveCompletionTest {
  @Test
  fun interactiveCompletionsExpandToSuccessiveLengthsInShortestFirstOrder() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("SHORT"),
      "START" to listOf("LONG"),
      "SHORT" to listOf("x"),
      "LONG" to listOf("SHORT", "SHORT")
    )
    val language = suffixLanguage(grammar, maxLength = 2)

    val completions = language.shortestCompletions("", emptySet(), random = Random(7))

    assertEquals(listOf(listOf("x"), listOf("x", "x")), completions.map { it.tokens })
    assertEquals(listOf(1, 2), completions.map { it.length })
  }

  @Test
  fun interactiveLimitDeduplicatesAndOmitsEpsilon() {
    val ambiguous: CFG = linkedSetOf(
      "START" to listOf("LEFT"),
      "START" to listOf("RIGHT"),
      "LEFT" to listOf("same"),
      "RIGHT" to listOf("same")
    )
    assertEquals(
      listOf(listOf("same")),
      suffixLanguage(ambiguous, 1)
        .shortestCompletions("", emptySet(), limit = 10, random = Random(11))
        .map { it.tokens }
    )
    assertEquals(
      emptyList(),
      suffixLanguage(setOf("START" to emptyList()), 0)
        .shortestCompletions("complete;", emptySet(), random = Random(11))
    )
    assertEquals(
      emptyList(),
      suffixLanguage(ambiguous, 1)
        .shortestCompletions("", emptySet(), limit = 0, random = Random(11))
    )
    assertTrue(
      suffixLanguage(ambiguous, 1).shortestCompletions(
        "", emptySet(), limit = CPP_MAX_INTERACTIVE_COMPLETIONS + 100, random = Random(11)
      ).size <= CPP_MAX_INTERACTIVE_COMPLETIONS
    )
  }

  @Test
  fun injectedSeedMakesMaterializedCompletionsStableWithoutFreshNameDuplicates() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("NAME", "END"),
      "NAME" to listOf(CPP_FRESH),
      "END" to listOf(";")
    )
    val language = suffixLanguage(grammar, 2)

    val first = language.shortestCompletions("int", setOf("freshId_reserved"), 4, Random(90210))
    val repeated = language.shortestCompletions("int", setOf("freshId_reserved"), 4, Random(90210))

    assertEquals(first, repeated)
    assertEquals(1, first.size)
    assertTrue(first.all { it.insertionText.startsWith(" freshId_") })
  }

  @Test
  fun ambiguityCannotCrowdOutTenSourceDistinctShortestCompletions() {
    val grammar: CFG = buildSet {
      repeat(40) { index ->
        add("START" to listOf("AMBIGUOUS_$index"))
        add("AMBIGUOUS_$index" to listOf("same"))
      }
      repeat(12) { index ->
        add("START" to listOf("UNIQUE_$index"))
        add("UNIQUE_$index" to listOf("choice_$index"))
      }
    }

    val completions = suffixLanguage(grammar, 1)
      .shortestCompletions("std::", emptySet(), limit = 10, random = Random(31337))

    assertEquals(10, completions.size)
    assertEquals(10, completions.map { it.tokens }.distinct().size)
    assertTrue(completions.all { it.length == 1 })
    assertTrue(completions.none { it.insertionText.startsWith(' ') })
  }

  @Test
  fun alphaEquivalentBindersAreOneStructuralCompletion() {
    val first = "$CPP_BIND_PREFIX:first"
    val second = "$CPP_BIND_PREFIX:second"
    val grammar: CFG = linkedSetOf(
      "START" to listOf("LEFT"),
      "START" to listOf("RIGHT"),
      "LEFT" to listOf(first),
      "RIGHT" to listOf(second)
    )

    val completions = suffixLanguage(grammar, 1)
      .shortestCompletions("int", emptySet(), random = Random(44))

    assertEquals(1, completions.size)
    assertTrue(completions.single().tokens.single().startsWith("freshId_"))
  }

  @Test
  fun qualifiedStdPrefixProducesTenUsefulShortestFirstStatements() {
    val prefixText = "std::"
    val prefix = cppLines(prefixText).single().tokens
    val context = CppCompletionContext(
      identifiers = setOf("std", "cout", "value", "flag", "name"),
      sourceIdentifiers = setOf("std", "cout", "value", "flag", "name"),
      headers = setOf("iostream"),
      values = listOf(
        CppReference("std::cout", type = "std::ostream", kind = "variable"),
        CppReference("value", type = "int", kind = "variable"),
        CppReference("flag", type = "bool", kind = "variable"),
        CppReference("name", type = "const char *", kind = "variable")
      )
    )
    val completions = CppCompletionGrammar().generate(context, prefix)
      .shortestCompletions(prefixText, context.identifiers, random = Random(2026))

    assertEquals(10, completions.size)
    assertEquals(10, completions.map { it.tokens }.distinct().size)
    assertTrue(completions.zipWithNext().all { (left, right) -> left.length <= right.length })
    assertTrue(completions.all { !it.insertionText.startsWith(' ') })
    assertEquals(listOf("cout", ";"), completions.first().tokens)
  }

  @Test
  fun expandedTupleValuedMapCompletesTryEmplaceWithoutAliasDependentFacts() {
    val prefixText = "records.try_emplace"
    val prefix = cppLines(prefixText).single().tokens
    val identifiers = setOf(
      "std", "map", "tuple", "string", "Record", "records", "try_emplace"
    )
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      headers = setOf("map", "tuple", "string"),
      values = listOf(
        CppReference(
          "records",
          type = "std::map<int,std::tuple<int,std::string,double>>",
          kind = "variable",
          source = "ast"
        )
      )
    )
    val completions = CppCompletionGrammar().generate(context, prefix).shortestCompletions(
      prefixText = prefixText,
      identifiersInFile = identifiers,
      random = Random(2142905409)
    )

    assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.size)
    assertTrue(
      completions.all { it.tokens.firstOrNull() == "(" && it.tokens.lastOrNull() == ";" }
    )
  }

  @Test
  fun observedMapSpecializationDoesNotHideAnotherPartialDeclaration() {
    val prefixText = "std::map<int, std::string"
    val prefix = cppLines(prefixText).single().tokens
    val identifiers = setOf("std", "map", "tuple", "string", "Record", "records")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      headers = setOf("map", "tuple", "string"),
      values = listOf(
        CppReference(
          "records",
          type = "std::map<int,std::tuple<int,std::string,double>>",
          kind = "variable",
          source = "ast"
        )
      )
    )
    val completions = CppCompletionGrammar().generate(context, prefix).shortestCompletions(
      prefixText = prefixText,
      identifiersInFile = identifiers,
      random = Random(20260804)
    )

    assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.size)
    assertTrue(completions.all { it.tokens.firstOrNull() == ">" && it.tokens.lastOrNull() == ";" })
  }

  @Test
  fun observedStandardTemplateFamiliesRetainSiblingDeclarationCandidates() {
    data class Case(
      val header: String,
      val observedType: String,
      val partialType: String
    )
    val cases = listOf(
      Case("vector", "std::vector<int>", "std::vector<std::string"),
      Case("set", "std::set<std::string>", "std::set<int"),
      Case("optional", "std::optional<int>", "std::optional<std::string")
    )
    cases.forEach { case ->
      val identifiers = setOf("std", case.header, "string", "existing")
      val context = CppCompletionContext(
        identifiers = identifiers,
        sourceIdentifiers = identifiers,
        headers = setOf(case.header, "string"),
        values = listOf(
          CppReference("existing", type = case.observedType, kind = "variable", source = "ast")
        )
      )
      val prefix = cppLines(case.partialType).single().tokens
      val completions = CppCompletionGrammar().generate(context, prefix).shortestCompletions(
        prefixText = case.partialType,
        identifiersInFile = identifiers,
        random = Random(20260804)
      )

      assertEquals(
        CPP_MAX_INTERACTIVE_COMPLETIONS,
        completions.size,
        "An observed ${case.header} specialization suppressed `${case.partialType}`"
      )
      assertTrue(
        completions.all { it.tokens.firstOrNull() == ">" && it.tokens.lastOrNull() == ";" },
        "Unexpected ${case.header} completion: ${completions.joinToString { it.tokens.joinToString(" ") }}"
      )
    }
  }

  @Test
  fun directUniquePtrUseCompletesANestedVectorDeclaration() {
    val identifiers = setOf("std", "vector", "unique_ptr", "Widget", "widgets")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      headers = setOf("memory", "vector"),
      types = listOf(CppReference("Widget", type = "Widget", kind = "class", source = "ast")),
      values = listOf(
        CppReference("widgets", type = "std::vector<int>", kind = "variable", source = "ast")
      )
    )
    listOf(
      "std::vector<std::unique_ptr<Widget" to listOf(">", ">"),
      "std::vector<std::unique_ptr<Widget>" to listOf(">")
    ).forEach { (prefixText, closingTokens) ->
      val prefix = cppLines(prefixText).single().tokens
      val completions = CppCompletionGrammar().generate(context, prefix).shortestCompletions(
        prefixText = prefixText,
        identifiersInFile = identifiers,
        random = Random(20260804)
      )

      assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.size)
      assertTrue(completions.all { completion ->
        completion.tokens.take(closingTokens.size) == closingTokens &&
          completion.tokens.lastOrNull() == ";"
      })
    }
  }

  @Test
  fun rendererPreservesShiftTokensWithinAndAcrossTheSuffixBoundary() {
    assertEquals(
      "std :: cout << value ;",
      listOf("std", "::", "cout", "<", "<", "value", ";").renderCppTokens()
    )
    assertEquals(" value ;", renderCppCompletionSuffix("return", listOf("value", ";")))
    assertEquals("cout ;", renderCppCompletionSuffix("std::", listOf("cout", ";")))
    assertEquals("size ( )", renderCppCompletionSuffix("value.", listOf("size", "(", ")")))
    assertEquals("< value ;", renderCppCompletionSuffix("std::cout <", listOf("<", "value", ";")))
    assertEquals(
      "std::cout << value ;",
      "std::cout <" + renderCppCompletionSuffix("std::cout <", listOf("<", "value", ";"))
    )
    assertEquals("value ;", renderCppCompletionSuffix("  ", listOf("value", ";")))
  }

  @Test
  fun completionLabelsUseCompactCppFormattingWithoutChangingInsertionText() {
    assertEquals(
      "records.try_emplace(0, 0, \"\", 0.0);",
      formatCppCompletionLabel(
        prefixTokens = listOf("records", ".", "try_emplace"),
        suffixTokens = listOf("(", "0", ",", "0", ",", "\"\"", ",", "0.0", ")", ";")
      )
    )
    assertEquals(
      "std::visit(Describe{}, payload);",
      formatCppCompletionLabel(
        prefixTokens = listOf("std", "::", "visit", "("),
        suffixTokens = listOf("Describe", "{", "}", ",", "payload", ")", ";")
      )
    )
    assertEquals(
      "std::cout << value;",
      formatCppCompletionLabel(
        prefixTokens = listOf("std", "::", "cout", "<"),
        suffixTokens = listOf("<", "value", ";")
      )
    )
    assertEquals(
      "std::vector<std::vector<int>> values;",
      formatCppCompletionLabel(
        prefixTokens = emptyList(),
        suffixTokens = listOf(
          "std", "::", "vector", "<", "std", "::", "vector", "<", "int", ">", ">",
          "values", ";"
        )
      )
    )
    assertEquals(
      " ( 0 , 0 , \"\" , 0.0 ) ;",
      renderCppCompletionSuffix(
        "records.try_emplace",
        listOf("(", "0", ",", "0", ",", "\"\"", ",", "0.0", ")", ";")
      ),
      "Display formatting must not change the conservative insertion renderer"
    )
  }

  private fun suffixLanguage(grammar: CFG, maxLength: Int): CppSuffixGrammar = CppSuffixGrammar(
    bounded = grammar.boundedAcyclic(maxLength),
    rawPrefix = emptyList(),
    projectedPrefix = emptyList(),
    templateTokens = maxLength
  )
}
