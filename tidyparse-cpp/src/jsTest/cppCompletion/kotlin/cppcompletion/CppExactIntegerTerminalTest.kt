package cppcompletion

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppExactIntegerTerminalTest {
  private fun tokens(source: String): List<CppToken> = cppLines(source).single().tokens

  @Test
  fun exactIntegerConditioningRecognitionAndSamplingPreserveTheCompilerSpelling() {
    val exactOne = cppExactIntegerTerminal("1")
    val prepared = PreparedCppCompletionGrammar(finiteAcyclicCnf(linkedSetOf(
      "START" to listOf(exactOne)
    )))

    assertEquals(listOf(CPP_INTEGER), projectCppTokens(tokens("1")))
    assertTrue(prepared.recognizes(tokens("1")))
    assertFalse(prepared.recognizes(tokens("0")))
    assertFalse(prepared.recognizes(tokens("2")))
    assertFalse(prepared.recognizes(tokens("1_km")), "an exact token never admits a UDL")

    val residual = prepared.generate(emptyList())
    assertTrue(residual.recognizes(tokens("1")))
    assertFalse(residual.recognizes(tokens("0")))
    assertEquals(
      listOf("1"),
      CppCompletionSampler(residual, emptySet(), Random(7)).sample(1).single().tokens
    )
    assertEquals("1", materializeCppTerminal(exactOne) { "unused" })
  }

  @Test
  fun abstractIntegerStillMatchesEveryIntegerBesideAnExactAlternative() {
    val prepared = PreparedCppCompletionGrammar(finiteAcyclicCnf(linkedSetOf(
      "START" to listOf("ABSTRACT"),
      "START" to listOf("EXACT"),
      "ABSTRACT" to listOf(CPP_INTEGER),
      "EXACT" to listOf(cppExactIntegerTerminal("1"))
    )))

    assertTrue(prepared.recognizes(tokens("0")))
    assertTrue(prepared.recognizes(tokens("1")))
    assertTrue(prepared.recognizes(tokens("42")))
    assertTrue(prepared.recognizes(tokens("0_km")), "the legacy abstract literal stays broad")
  }

  @Test
  fun freshMatchingRetainsExactIntegerRelations() {
    val prepared = PreparedCppCompletionGrammar(finiteAcyclicCnf(linkedSetOf(
      "START" to listOf(cppExactIntegerTerminal("1"), CPP_FRESH)
    )))
    val suffix = tokens("1 name")

    assertEquals(
      listOf(CppFreshMatch(listOf(listOf(1)))),
      prepared.generate(emptyList()).freshMatches(suffix)
    )
  }

  @Test
  fun exactIntegerParticipatesInPartialTokenCompletion() {
    val exact = cppExactIntegerTerminal("123")
    val prefix = CppToken(
      text = "12",
      start = 0,
      end = 2,
      kind = CppTokenKind.INTEGER,
      completeText = "123"
    )

    assertEquals(listOf("123"), cppCompletionTerminalSpellings(exact, prefix))
    assertEquals("123", cppCompletionTerminalSpelling(exact, prefix))
  }

  @Test
  fun everySyntheticLiteralCategoryRetainsExactRecognitionAndSampling() {
    data class Case(
      val kind: CppTokenKind,
      val spelling: String,
      val other: String,
      val userDefined: String? = null,
      val projected: String
    )
    val cases = listOf(
      Case(CppTokenKind.INTEGER, "0", "7", "0_km", CPP_INTEGER),
      Case(CppTokenKind.FLOATING, "0.0", "2.5", "0.0_km", "@floating"),
      Case(CppTokenKind.CHARACTER, "'\\0'", "'x'", "'\\0'_tag", "@character"),
      Case(CppTokenKind.STRING, "\"\"", "\"value\"", "\"\"_tag", "@string"),
      Case(CppTokenKind.BOOLEAN, "true", "false", projected = "@boolean")
    )

    cases.forEachIndexed { index, case ->
      val exact = cppExactLiteralTerminal(case.kind, case.spelling)
      val prepared = PreparedCppCompletionGrammar(finiteAcyclicCnf(linkedSetOf(
        "START" to listOf(exact)
      )))
      assertEquals(listOf(case.projected), projectCppTokens(tokens(case.spelling)))
      assertTrue(prepared.recognizes(tokens(case.spelling)), case.spelling)
      assertFalse(prepared.recognizes(tokens(case.other)), case.other)
      case.userDefined?.let { udl ->
        assertFalse(prepared.recognizes(tokens(udl)), "UDL $udl matched ${case.spelling}")
      }
      val residual = prepared.generate(emptyList())
      assertEquals(
        listOf(case.spelling),
        CppCompletionSampler(residual, emptySet(), Random(index + 11)).sample(1).single().tokens
      )
      assertEquals(case.spelling, materializeCppTerminal(exact) { "unused" })
    }
  }

  @Test
  fun nonIntegerExactLiteralsParticipateInPartialTokenCompletion() {
    val cases = listOf(
      Triple(CppTokenKind.FLOATING, "0.", "0.0"),
      Triple(CppTokenKind.CHARACTER, "'", "'\\0'"),
      Triple(CppTokenKind.STRING, "\"", "\"\""),
      Triple(CppTokenKind.BOOLEAN, "tr", "true")
    )
    cases.forEach { (kind, prefixText, complete) ->
      val prefix = CppToken(
        text = prefixText, start = 0, end = prefixText.length,
        kind = kind, completeText = complete
      )
      val exact = cppExactLiteralTerminal(
        if (kind == CppTokenKind.BOOLEAN) CppTokenKind.BOOLEAN else kind,
        complete
      )
      assertEquals(listOf(complete), cppCompletionTerminalSpellings(exact, prefix))
      assertEquals(complete, cppCompletionTerminalSpelling(exact, prefix))
    }
  }
}
