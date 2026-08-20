package cppcompletion

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppRequiredBinderGrammarTest {
  private val value = CppReference(
    name = "source", type = "int", canonicalType = "int",
    kind = "variable", isValue = true
  )

  @Test
  fun provenEmptyObligationRetainsOrdinaryStatements() {
    val language = language(CppRequiredBinderObligation(emptySet()))

    assertTrue(language.recognizes("0;"))
    assertTrue(
      language.freshMatches(cppLines("int freshBinding = source;").single().tokens).isNotEmpty(),
      "a proven-empty obligation must retain compiler-guarded fresh declarations"
    )
  }

  @Test
  fun singletonValueObligationRequiresThatExactValueBinder() {
    val language = language(CppRequiredBinderObligation(setOf("recovered")))

    assertTrue(language.recognizes("int recovered = source;"))
    assertFalse(language.recognizes("int other = source;"))
    assertFalse(
      language.recognizes("using recovered = int;"),
      "a type alias does not satisfy a downstream value use"
    )
    assertFalse(language.recognizes("source;"))
  }

  @Test
  fun multipleNecessaryBindersAreNotFlattenedIntoAlternatives() {
    val language = language(CppRequiredBinderObligation(setOf("left", "right")))

    assertFalse(language.recognizes("int left = source;"))
    assertFalse(language.recognizes("int right = source;"))
  }

  @Test
  fun singletonProfileEvidenceIsCorrelatedByDeclarationKindAndExactCv() {
    val gate = CppSingletonBindingGate(
      binder = "recovered",
      accepted = setOf(CppBindingProfile("int", declarationKind = "object")),
      probed = setOf(
        CppBindingProfile("int", declarationKind = "object"),
        CppBindingProfile("double", declarationKind = "object"),
        CppBindingProfile("int &", declarationKind = "lvalueReference")
      ),
      complete = false
    )
    val language = language(CppRequiredBinderObligation(setOf("recovered"), gate))

    assertTrue(language.recognizes("int recovered = source;"))
    assertFalse(language.recognizes("double recovered = source;"))
    assertFalse(
      language.recognizes("int & recovered = source;"),
      "a failed reference profile must remain separate from an accepted object profile"
    )
    assertFalse(
      language.recognizes("const int & recovered = source;"),
      "once a valid profile exists, an unprobed profile must not compete with compiler positives"
    )
  }

  private fun language(obligation: CppRequiredBinderObligation): CppSuffixGrammar =
    CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = setOf("source", "recovered", "other", "anyName", "left", "right"),
        sourceIdentifiers = setOf("source", "recovered", "other", "anyName", "left", "right"),
        values = listOf(value),
        requiredBinderObligation = obligation
      ),
      emptyList()
    )

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)
}
