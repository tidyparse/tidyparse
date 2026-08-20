package cppcompletion

import cppEditorStatementSnapshot
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSyntaxCompletionTotalityTest {
  @Test
  fun syntaxFloorKeepsEveryGeneratedBoundaryProductiveAndAcceptsTheFullStatementWitness() {
    val statements = generatedSyntaxStatements()
    assertTrue(statements.size >= 16, "The syntax matrix must retain broad recursive coverage")

    statements.forEach(::assertSyntaxCompletionAtEveryBoundary)
  }

  @Test
  fun reportedNestedStdPrefixCompletesFromTheSemanticIdentifierInventory() {
    val statement = "using Record = std::tuple<int, std::string, std::string>;"
    val reportedPrefix = "using Record = std::tuple<int, std::string, std"
    val line = cppLines(statement).single()
    val truncation = cppTruncations(line).singleOrNull { it.prefixText == reportedPrefix }

    assertNotNull(truncation, "The reported caret must be an exact C++ token boundary")
    assertEquals(listOf("::", "string", ">", ";"), truncation.suffix.map(CppToken::text))
    val snapshot = assertNotNull(
      cppEditorStatementSnapshot(statement, 0, reportedPrefix.length)
    )
    val identifiers = line.tokens.filter { it.kind == CppTokenKind.IDENTIFIER }
      .mapTo(linkedSetOf(), CppToken::text)
    val syntaxResidual = cppSingleStatementSyntaxCompletion(
      snapshot.stableTokens, snapshot.activeFragment, identifiers
    )

    assertNotNull(syntaxResidual, "The production syntax floor returned no residual")
    assertFalse(syntaxResidual.isEmpty, "The production syntax residual was empty")
    assertTrue(
      syntaxResidual.recognizes(listOf(requireNotNull(snapshot.activeFragment)) + truncation.suffix),
      "The full-statement syntax check rejected `${truncation.suffixText()}` after `$reportedPrefix`"
    )
    val completions = syntaxResidual.shortestCompletions(
      prefixText = reportedPrefix,
      identifiersInFile = truncation.prefixIdentifiers(),
      limit = 1,
      random = Random(0x51A7),
      tokenPrefix = snapshot.activeFragment
    )
    assertTrue(completions.isNotEmpty(), "The explicit syntax oracle produced no sampled completion")
    assertTrue(completions.single().tokens.isNotEmpty())
  }

  @Test
  fun syntaxProjectionPreservesNamedCastTemplateClosersAtEveryBoundary() {
    assertSyntaxCompletionAtEveryBoundary("static_cast<std::vector<int>>(value);")
  }

  @Test
  fun syntaxProjectionUsesTheLexerCategoryForUniversalCharacterNameIdentifiers() {
    val universalIdentifier = "\\" + "u03B1"
    val statement = "int $universalIdentifier = 0;"
    val tokens = cppLines(statement).single().tokens

    assertEquals(CppTokenKind.IDENTIFIER, tokens[1].kind)
    assertEquals(
      listOf("int", CPP_SYNTAX_IDENTIFIER, "=", CPP_INTEGER, ";"),
      projectCppCompletionTokens(tokens, CppProjectionMode.SYNTAX)
    )
    assertTrue(
      cppSingleStatementSyntaxRecognizes(tokens),
      "The pinned lexer admits the UCN identifier, so syntax projection must retain it"
    )
  }

  @Test
  fun syntaxProjectionRecognizesEveryStandardAlternativeOperatorSpelling() {
    val statements = listOf(
      "flag = left and right;",
      "flag = left or right;",
      "flag = not left;",
      "value = left bitand right;",
      "value = left bitor right;",
      "value = left xor right;",
      "value = compl left;",
      "value and_eq mask;",
      "value or_eq mask;",
      "value xor_eq mask;",
      "flag = left not_eq right;"
    )

    statements.forEach { statement ->
      assertTrue(
        cppSingleStatementSyntaxRecognizes(cppLines(statement).single().tokens),
        "Standard alternative operator spelling was rejected: `$statement`"
      )
    }
  }

  @Test
  fun contextualIdentifiersAndDigraphPunctuatorsRetainTheirStandardSyntax() {
    listOf(
      "int final = 1;",
      "int override = 1;",
      "struct Base { virtual void run() final; };",
      "int values<:2:> = <%1, 2%>;"
    ).forEach { statement ->
      assertTrue(
        cppSingleStatementSyntaxRecognizes(cppLines(statement).single().tokens),
        "Contextual keyword or digraph statement was rejected: `$statement`"
      )
    }
  }

  @Test
  fun splitGreaterTerminalsRenderAsAValidContiguousToken() {
    assertEquals(">>", listOf(">", ">").renderCppTokens())
    val prefix = "left >"
    assertEquals(
      "left >>value;",
      prefix + renderCppCompletionSuffix(prefix, listOf(">", "value", ";"))
    )
  }

  @Test
  fun userDefinedLiteralsKeepTheirLexerKindsWithoutFabricatingSuffixes() {
    val spellings = listOf(
      Triple("42", "42_tag", CppTokenKind.USER_DEFINED_INTEGER),
      Triple("1.5", "1.5_tag", CppTokenKind.USER_DEFINED_FLOATING),
      Triple("'x'", "'x'_tag", CppTokenKind.USER_DEFINED_CHARACTER),
      Triple("\"text\"", "\"text\"_tag", CppTokenKind.USER_DEFINED_STRING)
    )

    spellings.forEach { (ordinarySpelling, userDefinedSpelling, userDefinedKind) ->
      val userDefinedToken = cppLines(userDefinedSpelling).single().tokens.single()
      val ordinary = projectedSingleTerminal(ordinarySpelling)
      val projectedUserDefined = projectCppCompletionTokens(
        listOf(userDefinedToken), CppProjectionMode.SYNTAX
      ).single()

      assertEquals(userDefinedKind, userDefinedToken.kind)
      assertEquals(
        ordinary, projectedUserDefined,
        "A UDL suffix is a declaration-backed name, not a syntax-generated terminal"
      )
      assertTrue(
        cppSingleStatementSyntaxRecognizes(
          cppLines("return $userDefinedSpelling;").single().tokens
        ),
        "A real source UDL must remain valid statement syntax"
      )
      assertEquals(
        ordinary,
        projectedSingleTerminal(materializeCppTerminal(ordinary) { "freshId" }),
        "Ordinary literal materialization did not preserve its projected category"
      )
      assertFalse(
        '_' in materializeCppTerminal(projectedUserDefined) { "freshId" },
        "Syntax completion must not invent a user-defined literal operator suffix"
      )
    }
  }

  @Test
  fun generatedGrammarManifestAndIndependentStatementCorpusStayPinned() {
    assertEquals(
      "628062e9f75710ba1d1436ced8bd7d9d8f2f08c31a6e962c175e06b28994ff27",
      GeneratedCpp14StatementGrammar.parserSha256
    )
    assertEquals(
      "739a8782e05279318dccab76bf05af1ff5e3ff9e43f1b5b0d04e14d91d4fff47",
      GeneratedCpp14StatementGrammar.lexerSha256
    )
    assertEquals(4, GeneratedCpp14StatementGrammar.modernOverlayRevision)
    val statements = listOf(
      "for (int i = 0; i < 3; ++i) { continue; }",
      "switch (value) { case 1: break; default: return; }",
      "try { call(); } catch (const Error& error) { throw; }",
      "label: goto label;",
      "auto function = [value](int x) mutable -> int { return x + value; };",
      "co_return value;",
      "co_yield value;",
      "co_await task;",
      "char8_t codeUnit;",
      "auto [key, value] = pair;",
      "for (auto [key, value] : entries) { use(key); }",
      "left <=> right;",
      "consteval int answer();",
      "constinit int value = 0;",
      "using UnknownBound = int[];",
      "using FixedBound = int[4];",
      "using Matrix = int[][4];",
      "using PointerElements = int*[];",
      "using PointerToArray = int(*)[4];",
      "using AttributedArray = int[] [[maybe_unused]];",
      "std::unique_ptr<int[]> counters = std::make_unique<int[]>(3);",
      "Widget value{.x = 1, .y = 2};"
    )
    statements.forEach { statement ->
      assertTrue(
        cppSingleStatementSyntaxRecognizes(cppLines(statement).single().tokens),
        "Pinned parser + modern overlay rejected independent witness: `$statement`"
      )
    }
  }
}

private fun assertSyntaxCompletionAtEveryBoundary(statement: String) {
  val line = cppLines(statement).single()
  val identifiers = line.tokens.filter { it.kind == CppTokenKind.IDENTIFIER }
    .mapTo(linkedSetOf(), CppToken::text)
  assertTrue(
    cppSingleStatementSyntaxRecognizes(line.tokens),
    "The generated statement is outside the context-independent syntax grammar: `$statement`"
  )

  cppTruncations(line).dropLast(1).forEach { truncation ->
    val boundary = truncation.prefix.size
    val residual = cppSingleStatementSyntaxCompletion(
      truncation.prefix, identifierInventory = identifiers
    )
    assertNotNull(
      residual,
      "No syntax residual for `$statement` at token boundary $boundary after `${truncation.prefixText}`"
    )
    assertFalse(
      residual.isEmpty,
      "Empty syntax residual for `$statement` at token boundary $boundary"
    )
    // A witness may be longer than the globally shortest forest. recognizes() intentionally uses
    // the complete recursive syntax grammar for that check; non-emptiness and sampling above/below
    // independently exercise the bounded shortest-suffix forest.
    assertTrue(
      residual.recognizes(truncation.suffix),
      "Full-statement syntax check rejected witness `${truncation.suffixText()}` for `$statement` " +
        "at token boundary $boundary"
    )

    val completions = residual.shortestCompletions(
      prefixText = truncation.prefixText,
      identifiersInFile = truncation.prefixIdentifiers(),
      limit = 1,
      random = Random(statement.hashCode() * 31 + boundary)
    )
    assertTrue(
      completions.isNotEmpty(),
      "Syntax residual sampled no completion for `$statement` at token boundary $boundary"
    )
    assertTrue(
      completions.single().tokens.isNotEmpty(),
      "An incomplete statement must not be represented by an empty insertion"
    )
  }
}

/**
 * Produces a deterministic cross-section rather than locking completion to individual prompts.
 * Every type constructor is exercised with an input that is itself qualified or templated, and
 * [recursiveChain] places the result under another constructor at the next depth.
 */
private fun generatedSyntaxStatements(): List<String> {
  val roots = listOf(
    "int",
    "std::string",
    "project::model::Widget"
  )
  val constructors: List<(String) -> String> = listOf(
    { element -> "std::vector<$element>" },
    { element -> "std::optional<$element>" },
    { element -> "std::map<std::string, $element>" },
    { element -> "std::map<$element, std::string>" },
    { element -> "std::tuple<int, $element, double>" },
    { element -> "project::container::Box<$element>" }
  )
  val oneLevel = constructors.mapIndexed { index, constructor ->
    constructor(roots[index % roots.size])
  }
  val recursiveChain = buildList {
    var nested = "project::model::Widget"
    constructors.take(5).forEach { constructor ->
      nested = constructor(nested)
      add(nested)
    }
  }
  val generatedTypes = (roots + oneLevel + recursiveChain).distinct()

  return buildList {
    // This witness deliberately contains the exact partial spelling from the reported failure.
    add("using Record = std::tuple<int, std::string, std::string>;")
    generatedTypes.forEachIndexed { index, type ->
      if (index % 2 == 0) add("using GeneratedAlias$index = $type;")
      else add("$type generatedValue$index;")
    }
    add("using Rebound = typename traits::template rebind<std::vector<int>>::type;")
    add("const project::container::Box<std::vector<std::optional<int>>>& value = source;")
    add("std::array<std::vector<int>, 4> values;")
    add("return project::factory<std::vector<std::string>>(source);")
    add("registry.template lookup<project::model::Widget>(key);")
    add("static_cast<project::container::Box<std::vector<int>>*>(pointer);")
    add("new project::container::Box<std::tuple<int, std::string>>{};")
  }
}

private fun CppTruncation.prefixIdentifiers(): Set<String> =
  prefix.filter { it.kind == CppTokenKind.IDENTIFIER }.mapTo(linkedSetOf(), CppToken::text)

private fun CppTruncation.suffixText(): String = suffix.joinToString(" ", transform = CppToken::text)

private fun projectedSingleTerminal(spelling: String): String {
  val tokens = cppLines(spelling).single().tokens
  assertEquals(1, tokens.size, "`$spelling` must lex as one C++ token")
  return projectCppCompletionTokens(tokens, CppProjectionMode.SYNTAX).single()
}
