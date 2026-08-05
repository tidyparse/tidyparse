package cppcompletion

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import cppEditorStatementSnapshot
import completionQuery
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
  fun tokenPrefixIsAppliedBeforeTheInteractiveSampleCap() {
    val grammar: CFG = buildSet {
      repeat(40) { index -> add("START" to listOf("other_$index")) }
      add("START" to listOf("return"))
    }
    val prefix = CppToken("ret", 0, 3, CppTokenKind.IDENTIFIER)

    val completions = suffixLanguage(grammar, 1).shortestCompletions(
      prefixText = "",
      identifiersInFile = emptySet(),
      limit = 10,
      random = Random(9),
      tokenPrefix = prefix
    )

    assertEquals(listOf(listOf("return")), completions.map { it.tokens })
    assertEquals("return", completions.single().insertionText)
  }

  @Test
  fun firstTerminalIntersectionHandlesUnitsBinariesAndNullableLeftChildren() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("MAYBE", "WORD"),
      "MAYBE" to emptyList(),
      "MAYBE" to listOf("OTHER"),
      "OTHER" to listOf("other"),
      "WORD" to listOf("RETURN", "END"),
      "RETURN" to listOf("return"),
      "END" to listOf(";")
    )
    fun complete(typed: String) = suffixLanguage(grammar, 3).shortestCompletions(
      prefixText = "",
      identifiersInFile = emptySet(),
      random = Random(3),
      tokenPrefix = CppToken(typed, 0, typed.length, CppTokenKind.IDENTIFIER)
    ).map(CppShortestCompletion::tokens)

    assertEquals(listOf(listOf("return", ";")), complete("ret"))
    assertEquals(listOf(listOf("other", "return", ";")), complete("oth"))
  }

  @Test
  fun everyGrammarTerminalKindUsesItsMatchingSourceSpelling() {
    listOf(
      Triple("return", "ret", "return"),
      Triple("&&", "an", "and"),
      Triple("@boolean", "tru", "true"),
      Triple("@nullptr", "nullp", "nullptr"),
      Triple(encodeIdentifier("std"), "st", "std")
    ).forEach { (terminal, typed, expected) ->
      val prefix = CppToken(typed, 0, typed.length, CppTokenKind.IDENTIFIER)
      val completion = suffixLanguage(setOf("START" to listOf(terminal)), 1)
        .shortestCompletions("", emptySet(), random = Random(1), tokenPrefix = prefix)
        .single()
      assertEquals(expected, completion.insertionText, "$typed must complete through $terminal")
    }
  }

  @Test
  fun partialLiteralsAndMaximalMunchOperatorsKeepACompleteTokenWitness() {
    listOf(
      Triple("\"text\"", 3, "\"text\""),
      Triple("'x'", 2, "'x'"),
      Triple("42_km", 2, "42_km"),
      Triple(">>=", 2, ">>=")
    ).forEach { (source, caret, expected) ->
      val snapshot = requireNotNull(cppEditorStatementSnapshot(source, 0, caret))
      val terminal = projectCppTokens(cppLines(source).single().tokens).single()
      val completion = suffixLanguage(setOf("START" to listOf(terminal)), 1)
        .shortestCompletions(
          snapshot.stablePrefixText,
          emptySet(),
          random = Random(2),
          tokenPrefix = snapshot.activeFragment
        ).single()
      assertEquals(expected, completion.insertionText, source)
    }
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
    val snapshot = requireNotNull(cppEditorStatementSnapshot(prefixText, 0, prefixText.length))
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
    val completions = CppCompletionGrammar().completeCppStatement(
      context, snapshot.completionQuery(context.identifiers, seed = 2026)
    ).suggestions

    assertEquals(10, completions.size)
    assertEquals(10, completions.map { it.candidateText }.distinct().size)
    assertTrue(completions.zipWithNext().all { (left, right) ->
      left.tokenLength <= right.tokenLength
    })
    assertEquals("std::cout;", completions.first().candidateText)
    assertEquals(listOf("::", "cout", ";"), completions.first().tokens)
  }

  @Test
  fun expandedTupleValuedMapCompletesPartialTryEmplaceWithoutAliasDependentFacts() {
    val text = "records.try_emp"
    val snapshot = requireNotNull(cppEditorStatementSnapshot(text, 0, text.length))
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
    val completions = CppCompletionGrammar().completeCppStatement(
      context, snapshot.completionQuery(identifiers, seed = 2142905409)
    ).suggestions

    assertEquals(CPP_MAX_INTERACTIVE_COMPLETIONS, completions.size)
    assertTrue(
      completions.all {
        it.tokens.firstOrNull() == "try_emplace" && it.tokens.lastOrNull() == ";"
      }
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
  fun lexicalCodecPreservesShiftTokensWithinAndAcrossTheSuffixBoundary() {
    assertEquals(
      "std::cout<<value;",
      listOf("std", "::", "cout", "<", "<", "value", ";").renderCppTokens()
    )
    assertEquals(" value;", renderCppCompletionSuffix("return", listOf("value", ";")))
    assertEquals("cout;", renderCppCompletionSuffix("std::", listOf("cout", ";")))
    assertEquals("size()", renderCppCompletionSuffix("value.", listOf("size", "(", ")")))
    assertEquals("<value;", renderCppCompletionSuffix("std::cout <", listOf("<", "value", ";")))
    assertEquals(
      "std::cout <<value;",
      "std::cout <" + renderCppCompletionSuffix("std::cout <", listOf("<", "value", ";"))
    )
    assertEquals("value;", renderCppCompletionSuffix("  ", listOf("value", ";")))
  }

  @Test
  fun lexicalCodecDoesNotSplitAValidCastAcrossTheCursorBoundary() {
    val prefix = "Shape& mutable_view = const_cast<Shape"
    val suffix = listOf("&", ">", "(", "view", ")", ";")

    assertEquals("&>(view);", renderCppCompletionSuffix(prefix, suffix))
    assertEquals(
      "Shape& mutable_view = const_cast<Shape&>(view);",
      prefix + renderCppCompletionSuffix(prefix, suffix)
    )
  }

  @Test
  fun lexicalCodecRetainsOnlyRequiredTokenSeparators() {
    assertEquals(" value;", renderCppCompletionSuffix("return", listOf("value", ";")))
    assertEquals("value;", renderCppCompletionSuffix("return ", listOf("value", ";")))
    assertEquals(" /right;", renderCppCompletionSuffix("left/", listOf("/", "right", ";")))
    assertEquals(" *right;", renderCppCompletionSuffix("left/", listOf("*", "right", ";")))
    assertEquals(" >value", renderCppCompletionSuffix("label:", listOf(">", "value")))
  }

  @Test
  fun lexicalCodecLeavesAllStylisticWhitespaceToClangFormat() {
    assertEquals(
      "records.try_emplace(0,0,\"\",0.0);",
      listOf(
        "records", ".", "try_emplace", "(", "0", ",", "0", ",", "\"\"", ",",
        "0.0", ")", ";"
      ).renderCppTokens()
    )
    assertEquals(
      "std::visit(Describe{},payload);",
      listOf("std", "::", "visit", "(", "Describe", "{", "}", ",", "payload", ")", ";")
        .renderCppTokens()
    )
    assertEquals(
      "std::cout<<value;",
      listOf("std", "::", "cout", "<", "<", "value", ";").renderCppTokens()
    )
    assertEquals(
      "std::vector<std::vector<int>>values;",
      listOf(
        "std", "::", "vector", "<", "std", "::", "vector", "<", "int", ">", ">",
        "values", ";"
      ).renderCppTokens()
    )
    assertEquals(
      "(0,0,\"\",0.0);",
      renderCppCompletionSuffix(
        "records.try_emplace",
        listOf("(", "0", ",", "0", ",", "\"\"", ",", "0.0", ")", ";")
      )
    )
  }

  private fun suffixLanguage(grammar: CFG, maxLength: Int): CppSuffixGrammar = CppSuffixGrammar(
    bounded = grammar.boundedAcyclic(maxLength),
    rawPrefix = emptyList(),
    projectedPrefix = emptyList(),
    templateTokens = maxLength
  )
}
