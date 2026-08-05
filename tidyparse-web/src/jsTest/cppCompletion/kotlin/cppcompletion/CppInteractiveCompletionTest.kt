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
  fun semaMemberSignatureCompletesAnArbitraryPartialMemberName() {
    val text = "archive.record_"
    val snapshot = requireNotNull(cppEditorStatementSnapshot(text, 0, text.length))
    val identifiers = setOf("Archive", "archive", "record_entry")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      values = listOf(semaValue("archive", "Archive")),
      membersByType = listOf(
        CppTypeMembers(
          "Archive",
          listOf(
            semaMethod(
              "Archive",
              "record_entry",
              "void",
              parameters = listOf(
                CppParameter(name = "key", type = "int", canonicalType = "int"),
                CppParameter(
                  name = "label",
                  type = "const char *",
                  canonicalType = "const char *"
                )
              )
            )
          )
        )
      )
    )
    val completions = CppCompletionGrammar().completeCppStatement(
      context, snapshot.completionQuery(identifiers, limit = 1, seed = 2142905409)
    ).suggestions

    val completion = completions.single()
    assertEquals("record_entry", completion.tokens.first())
    assertEquals(";", completion.tokens.last())
    assertTrue(completion.candidateText.startsWith("archive.record_entry("))
  }

  @Test
  fun semaTypeSpellingCompletesAnArbitraryNestedDeclaration() {
    val type = "atlas::Ledger<int,glyph::Label>"
    val prefixText = "atlas::Ledger<int, glyph::Lab"
    val snapshot = requireNotNull(cppEditorStatementSnapshot(prefixText, 0, prefixText.length))
    val identifiers = setOf("atlas", "Ledger", "glyph", "Label")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      types = listOf(semaType(type)),
      defaultConstructibleTypes = setOf(type)
    )
    val completion = CppCompletionGrammar().completeCppStatement(
      context,
      snapshot.completionQuery(identifiers, limit = 1, seed = 20260804)
    ).suggestions.single()

    assertEquals("Label", completion.tokens.first())
    assertEquals(";", completion.tokens.last())
    assertTrue(
      completion.candidateText.startsWith("atlas::Ledger<int, glyph::Label>"),
      completion.candidateText
    )
  }

  @Test
  fun independentlyReportedSemaSpecializationsRemainIndependent() {
    val observedType = "cosmos::Vault<int>"
    val offeredType = "cosmos::Vault<Signal>"
    val identifiers = setOf("cosmos", "Vault", "Signal", "existing")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      values = listOf(semaValue("existing", observedType)),
      types = listOf(
        semaType("Signal"),
        semaType(offeredType)
      ),
      defaultConstructibleTypes = setOf(offeredType)
    )
    val prefixText = "cosmos::Vault<Sign"
    val snapshot = requireNotNull(cppEditorStatementSnapshot(prefixText, 0, prefixText.length))
    val completion = CppCompletionGrammar().completeCppStatement(
      context,
      snapshot.completionQuery(identifiers, limit = 1, seed = 20260804)
    ).suggestions.single()

    assertEquals("Signal", completion.tokens.first())
    assertEquals(";", completion.tokens.last())
    assertTrue(completion.candidateText.startsWith(offeredType))
  }

  @Test
  fun explicitlyReportedNestedSemaSpecializationCompletesAtEveryClosingBoundary() {
    val type = "fabric::Bundle<fabric::Handle<Widget>>"
    val identifiers = setOf("fabric", "Bundle", "Handle", "Widget")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      types = listOf(
        semaType("Widget"),
        semaType(type)
      ),
      defaultConstructibleTypes = setOf(type)
    )
    listOf(
      "fabric::Bundle<fabric::Handle<Widget" to listOf(">", ">"),
      "fabric::Bundle<fabric::Handle<Widget>" to listOf(">")
    ).forEach { (prefixText, closingTokens) ->
      val prefix = cppLines(prefixText).single().tokens
      val completions = CppCompletionGrammar().generate(context, prefix).shortestCompletions(
        prefixText = prefixText,
        identifiersInFile = identifiers,
        limit = 1,
        random = Random(20260804)
      )

      val completion = completions.single()
      assertEquals(closingTokens, completion.tokens.take(closingTokens.size))
      assertEquals(";", completion.tokens.last())
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

  private fun semaType(type: String) = CppReference(
    name = type,
    type = type,
    kind = "class",
    source = "sema",
    id = "type:$type",
    qualifiedName = type,
    canonicalType = type,
    isType = true
  )

  private fun semaValue(name: String, type: String) = CppReference(
    name = name,
    type = type,
    kind = "variable",
    source = "sema",
    id = "value:$name",
    canonicalType = type,
    isValue = true
  )

  private fun semaMethod(
    owner: String,
    name: String,
    returnType: String,
    parameters: List<CppParameter>
  ) = CppReference(
    name = name,
    returnType = returnType,
    parameters = parameters,
    kind = "method",
    ownerType = owner,
    source = "sema",
    id = "$owner::$name",
    canonicalReturnType = returnType,
    canonicalOwnerType = owner,
    isCallable = true,
    isMember = true,
    isStatic = false
  )
}
