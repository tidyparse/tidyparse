package buildlogic

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppStatementGrammarGeneratorTest {
  private val lexer = """
    lexer grammar MiniLexer;
    LeftParen: '(';
    RightParen: ')';
    LeftBracket: '[';
    RightBracket: ']';
    Comma: ',';
    Semi: ';';
    Bang: '!';
    Star: '*';
    Identifier: [a-zA-Z_]+;
    Whitespace: [ \\t]+ -> skip;
  """.trimIndent()

  @Test
  fun desugarsEbnfLabelsPredicatesAndNegatedTokenSetsToExactCnf() {
    val parser = """
      parser grammar MiniParser;
      options { tokenVocab = MiniLexer; }
      statement
        : item? (Comma item)* tagged = item? { allowed() }? ~(LeftParen | RightParen)+ Semi
        ;
      item: Identifier | LeftParen item RightParen;
      unreachable: Bang;
    """.trimIndent()

    val generated = CppStatementGrammarGenerator.generate(parser, lexer)
    val grammar = generated.productions
    val nonterminals = grammar.mapTo(linkedSetOf(), Production::lhs)

    assertTrue(grammar.all { production ->
      production.rhs.size == 1 && production.rhs.single() !in nonterminals ||
        production.rhs.size == 2 && production.rhs.all(nonterminals::contains)
    }, "The build artifact must be epsilon-free and contain no variable-unit productions")
    assertFalse(grammar.any { it.lhs == "unreachable" }, "Only statement-reachable rules are retained")

    // `!` is in the lexer vocabulary and outside the negated parenthesis set. This shortest witness
    // also requires every optional/star/predicate node above to disappear through epsilon closure.
    assertTrue(grammar.recognizes(listOf("!", ";")))
    assertTrue(grammar.recognizes(listOf("!", "!", ";")), "The negated token-set `+` must repeat")
    assertTrue(
      grammar.recognizes(listOf("@identifier", ",", "@identifier", "@identifier", "!", ";")),
      "Optional, star, label assignment, and identifier projection must compose"
    )
    assertFalse(grammar.recognizes(listOf("(", ";")), "Negated token sets must exclude every named token")
  }

  @Test
  fun embedsInputHashesAndManifestInPackedKotlin() {
    val parser = "parser grammar MiniParser; statement: Identifier Semi;"
    val generated = CppStatementGrammarGenerator.generate(
      parser,
      lexer,
      parserSha256 = "parser-hash",
      lexerSha256 = "lexer-hash"
    )

    assertTrue("parser-hash" in generated.kotlinSource)
    assertTrue("lexer-hash" in generated.kotlinSource)
    assertTrue("nonterminalCount" in generated.kotlinSource)
    assertTrue("productions: IntArray" in generated.kotlinSource)
  }

  @Test
  fun ignoresAntlrNongreedyModifiersWithoutMakingOneOrMoreNullable() {
    val parser = """
      parser grammar MiniParser;
      options { tokenVocab = MiniLexer; }
      statement: Identifier+? Bang*? Semi;
    """.trimIndent()

    val grammar = CppStatementGrammarGenerator.generate(parser, lexer).productions

    assertTrue(grammar.recognizes(listOf("@identifier", ";")))
    assertTrue(grammar.recognizes(listOf("@identifier", "@identifier", "!", "!", ";")))
    assertFalse(
      grammar.recognizes(listOf(";")),
      "ANTLR `+?` is nongreedy one-or-more, not an optional one-or-more expression"
    )
    assertFalse(
      grammar.recognizes(listOf("!", ";")),
      "The nongreedy modifier must not make the preceding `Identifier+` nullable"
    )
  }

  @Test
  fun erasesSemanticPredicatesWithoutLeakingEmptyHelpersAsTerminals() {
    val parser = """
      parser grammar MiniParser;
      options { tokenVocab = MiniLexer; }
      statement: Identifier { allowed() }? Bang? Semi;
    """.trimIndent()

    val grammar = CppStatementGrammarGenerator.generate(parser, lexer).productions
    val nonterminals = grammar.mapTo(linkedSetOf(), Production::lhs)
    val terminals = grammar.asSequence()
      .flatMap { it.rhs.asSequence() }
      .filter { it !in nonterminals }
      .toSet()

    assertTrue(grammar.recognizes(listOf("@identifier", ";")))
    assertTrue(grammar.recognizes(listOf("@identifier", "!", ";")))
    assertFalse(grammar.recognizes(listOf(";")))
    assertFalse(
      terminals.any { it.startsWith("CPP14_EBNF_") || it.startsWith("CPP14_CNF_") },
      "Erased predicates must not leave internal helper symbols in the terminal alphabet: $terminals"
    )
  }

  @Test
  fun auditedAbstractArrayErratumComposesWithPointersDimensionsAttributesAndParameters() {
    val parser = """
      parser grammar MiniParser;
      statement: theTypeId Semi;
      theTypeId: Identifier abstractDeclarator?;
      abstractDeclarator: pointerAbstractDeclarator | noPointerAbstractDeclarator;
      pointerAbstractDeclarator: Star* (noPointerAbstractDeclarator | Star);
      noPointerAbstractDeclarator
        : LeftParen pointerAbstractDeclarator RightParen
          (parametersAndQualifiers | LeftBracket constantExpression? RightBracket attributeSpecifierSeq?)*
        ;
      parametersAndQualifiers: LeftParen RightParen;
      constantExpression: IntegerLiteral;
      attributeSpecifierSeq: LeftBracket LeftBracket Identifier RightBracket RightBracket;
    """.trimIndent()
    val grammar = CppStatementGrammarGenerator.generate(parser, lexer).productions

    listOf(
      listOf("@identifier", "[", "]", ";"),
      listOf("@identifier", "[", "@integer", "]", ";"),
      listOf("@identifier", "[", "]", "[", "@integer", "]", ";"),
      listOf("@identifier", "*", "[", "]", ";"),
      listOf("@identifier", "(", "*", ")", "[", "@integer", "]", ";"),
      listOf("@identifier", "[", "]", "(", ")", ";"),
      listOf("@identifier", "[", "]", "[", "[", "@identifier", "]", "]", ";")
    ).forEach { witness ->
      assertTrue(grammar.recognizes(witness), "Abstract-array erratum rejected $witness")
    }
  }

  @Test
  fun nongreedyMarkersDoNotChangeEbnfCardinality() {
    val parser = """
      parser grammar MiniParser;
      statement: Identifier+? Semi;
    """.trimIndent()
    val grammar = CppStatementGrammarGenerator.generate(parser, lexer).productions

    assertFalse(grammar.recognizes(listOf(";")), "ANTLR `+?` still requires at least one item")
    assertTrue(grammar.recognizes(listOf("@identifier", ";")))
    assertTrue(grammar.recognizes(listOf("@identifier", "@identifier", ";")))
  }

  @Test
  fun actionsAndPredicatesEraseWithoutLeakingGeneratedHelperTerminals() {
    val parser = """
      parser grammar MiniParser;
      statement: Identifier { action(); } { allowed() }? Semi;
    """.trimIndent()
    val grammar = CppStatementGrammarGenerator.generate(parser, lexer).productions
    val nonterminals = grammar.mapTo(linkedSetOf(), Production::lhs)
    val terminals = grammar.flatMap { it.rhs }.filterNot(nonterminals::contains)

    assertTrue(grammar.recognizes(listOf("@identifier", ";")))
    assertFalse(terminals.any { it.startsWith("CPP14_") }, "Empty-only helper leaked as a terminal")
  }

  @Test
  fun suppliesEveryStandardAlternativeOperatorTokenMissingFromThePinnedLexer() {
    val operatorLexer = """
      lexer grammar OperatorLexer;
      And: '&';
      Or: '|';
      Tilde: '~';
      Caret: '^';
      AndAssign: '&=';
      OrAssign: '|=';
      XorAssign: '^=';
      NotEqual: '!=';
      Semi: ';';
    """.trimIndent()
    val parser = """
      parser grammar OperatorParser;
      statement: (And | Or | Tilde | Caret | AndAssign | OrAssign | XorAssign | NotEqual) Semi;
    """.trimIndent()
    val grammar = CppStatementGrammarGenerator.generate(parser, operatorLexer).productions
    val spellings = mapOf(
      "&" to "bitand",
      "|" to "bitor",
      "~" to "compl",
      "^" to "xor",
      "&=" to "and_eq",
      "|=" to "or_eq",
      "^=" to "xor_eq",
      "!=" to "not_eq"
    )

    spellings.forEach { (punctuator, alternative) ->
      assertTrue(grammar.recognizes(listOf(punctuator, ";")), "Lost primary spelling '$punctuator'")
      assertTrue(grammar.recognizes(listOf(alternative, ";")), "Missing alternative token '$alternative'")
    }
  }

  @Test
  fun keepsUserDefinedLiteralCategoriesDistinctAndProductive() {
    val parser = """
      parser grammar LiteralParser;
      statement
        : LeftParen ordinaryLiteral RightParen Semi
        | Bang UserDefinedLiteral Semi
        | Star specializedUserDefinedLiteral Semi
        ;
      ordinaryLiteral: IntegerLiteral | FloatingLiteral | CharacterLiteral | StringLiteral;
      specializedUserDefinedLiteral
        : UserDefinedIntegerLiteral
        | UserDefinedFloatingLiteral
        | UserDefinedCharacterLiteral
        | UserDefinedStringLiteral
        ;
    """.trimIndent()
    val grammar = CppStatementGrammarGenerator.generate(parser, lexer).productions
    val ordinary = listOf("@integer", "@floating", "@character", "@string")
    val userDefined = listOf("@ud_integer", "@ud_floating", "@ud_character", "@ud_string")

    ordinary.forEach { terminal ->
      assertTrue(grammar.recognizes(listOf("(", terminal, ")", ";")))
      assertFalse(grammar.recognizes(listOf("!", terminal, ";")), "UDL slot accepted ordinary $terminal")
    }
    userDefined.forEach { terminal ->
      assertTrue(grammar.recognizes(listOf("!", terminal, ";")), "Aggregate UDL slot rejected $terminal")
      assertTrue(grammar.recognizes(listOf("*", terminal, ";")), "Specialized UDL slot rejected $terminal")
      assertFalse(
        grammar.recognizes(listOf("(", terminal, ")", ";")),
        "Ordinary-only literal slot accepted $terminal"
      )
    }
  }

  @Test
  fun treatsFinalAndOverrideAsContextualIdentifiersWithoutRemovingDedicatedTokens() {
    val contextualLexer = """
      lexer grammar ContextualLexer;
      Final: 'final';
      Override: 'override';
      Identifier: [a-zA-Z_]+;
      LeftParen: '(';
      Semi: ';';
    """.trimIndent()
    val parser = """
      parser grammar ContextualParser;
      statement
        : Identifier Semi
        | Final LeftParen Override Semi
        ;
    """.trimIndent()
    val grammar = CppStatementGrammarGenerator.generate(parser, contextualLexer).productions

    assertTrue(grammar.recognizes(listOf("@identifier", ";")))
    assertTrue(grammar.recognizes(listOf("final", ";")), "`final` must remain usable as an identifier")
    assertTrue(grammar.recognizes(listOf("override", ";")), "`override` must remain usable as an identifier")
    assertTrue(
      grammar.recognizes(listOf("final", "(", "override", ";")),
      "Dedicated Final/Override grammar token paths must remain productive"
    )
  }
}

/** Tiny independent recognizer used only to verify the generator's normalized output language. */
private fun Set<Production>.recognizes(tokens: List<String>): Boolean {
  if (tokens.isEmpty()) return false
  val nonterminals = mapTo(linkedSetOf(), Production::lhs).toList()
  val index = nonterminals.withIndex().associate { it.value to it.index }
  val chart = Array(tokens.size + 1) {
    Array(tokens.size + 1) { BooleanArray(nonterminals.size) }
  }
  forEach { production ->
    if (production.rhs.size == 1 && production.rhs.single() !in index) {
      tokens.forEachIndexed { tokenIndex, terminal ->
        if (terminal == production.rhs.single()) chart[tokenIndex][tokenIndex + 1][index.getValue(production.lhs)] = true
      }
    }
  }
  for (span in 2..tokens.size) for (begin in 0..tokens.size - span) {
    val end = begin + span
    for (split in begin + 1 until end) forEach { production ->
      if (production.rhs.size == 2 &&
        chart[begin][split][index.getValue(production.rhs[0])] &&
        chart[split][end][index.getValue(production.rhs[1])]
      ) chart[begin][end][index.getValue(production.lhs)] = true
    }
  }
  return chart[0][tokens.size][index.getValue("START")]
}
