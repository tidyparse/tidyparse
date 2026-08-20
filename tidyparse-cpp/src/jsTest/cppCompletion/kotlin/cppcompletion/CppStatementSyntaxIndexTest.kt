package cppcompletion

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import ai.hypergraph.kaliningraph.parsing.freeze
import ai.hypergraph.kaliningraph.parsing.tmLst
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class CppStatementSyntaxIndexTest {
  @Test
  fun minimumLengthsAndShortestForestsMatchExhaustiveBruteForce() {
    val grammar = smallRecursiveCnf()
    val index = CppStatementSyntaxIndex(grammar)
    val alphabet = listOf("a", "b", "(", ")", ";")
    val prefixes = terminalStrings(alphabet, maximumLength = 4)
    // Every live prefix in the hand-written language needs at most three more terminals. This
    // finite enumeration is therefore an exact oracle, not a sampling approximation.
    val candidateSuffixes = terminalStrings(alphabet, maximumLength = 3)

    prefixes.forEach { prefix ->
      val validCompletions = candidateSuffixes.filter { suffix ->
        smallRecursiveLanguageContains(prefix + suffix)
      }
      val expectedMinimum = validCompletions.minOfOrNull(List<String>::size)
      assertEquals(
        expectedMinimum,
        index.minimumSuffixLength(prefix),
        "Wrong minimum completion length for prefix `${prefix.joinToString(" ")}`"
      )

      if (expectedMinimum == null) {
        (0..3).forEach { suffixLength ->
          assertNull(
            index.completeShortestSuffix(prefix, suffixLength),
            "Dead prefix `${prefix.joinToString(" ")}` produced a $suffixLength-token forest"
          )
        }
        return@forEach
      }

      val forest = assertNotNull(
        index.completeShortestSuffix(prefix, expectedMinimum),
        "Missing shortest forest for prefix `${prefix.joinToString(" ")}`"
      )
      val bounded = forest.toAcyclicCfg(grammar.tmLst).boundedAcyclic(expectedMinimum)
      val expectedShortest = validCompletions
        .filter { it.size == expectedMinimum }
        .toSet()
      val recognized = candidateSuffixes.filter(bounded::recognizes).toSet()

      assertEquals(
        expectedShortest,
        recognized,
        "Shortest forest changed the completion language for prefix `${prefix.joinToString(" ")}`"
      )
    }
  }

  @Test
  fun firstTerminalConstraintsMatchExhaustiveBruteForce() {
    val grammar = smallRecursiveCnf()
    val index = CppStatementSyntaxIndex(grammar)
    val alphabet = listOf("a", "b", "(", ")", ";")
    val prefixes = terminalStrings(alphabet, maximumLength = 4)
    val candidateSuffixes = terminalStrings(alphabet, maximumLength = 3)
    val allowedSets = listOf(
      emptySet(), setOf("a"), setOf("b"), setOf(";"), setOf("(", "a"), setOf("a", "b")
    )

    for (prefix in prefixes) for (allowed in allowedSets) {
      val validCompletions = candidateSuffixes.filter { suffix ->
        suffix.firstOrNull() in allowed && smallRecursiveLanguageContains(prefix + suffix)
      }
      val expectedMinimum = validCompletions.minOfOrNull(List<String>::size)
      val actualMinimum = index.minimumSuffixLength(prefix, allowed)
      assertEquals(
        expectedMinimum,
        actualMinimum,
        "Wrong constrained minimum after `$prefix` with first terminal in $allowed"
      )
      if (actualMinimum == null) continue

      val forest = assertNotNull(index.completeShortestSuffix(prefix, actualMinimum, allowed))
      val bounded = forest.toAcyclicCfg(grammar.tmLst).boundedAcyclic(actualMinimum)
      val expectedShortest = validCompletions.filter { it.size == actualMinimum }.toSet()
      val recognized = candidateSuffixes.filter(bounded::recognizes).toSet()
      assertEquals(
        expectedShortest,
        recognized,
        "Constrained forest changed the language after `$prefix` with first terminal in $allowed"
      )
    }

    // `a a` already completes SEQUENCE, but `;` is not allowed first. The recurrence must retain
    // the distinct positive path that extends the same left-recursive variable with another `a`.
    assertEquals(2, index.minimumSuffixLength(listOf("a", "a"), setOf("a")))
    assertEquals(1, index.minimumSuffixLength(listOf("a", "a"), setOf(";")))
  }

  @Test
  fun concreteSpellingCallbackMapsAliasesToGrammarTerminals() {
    val grammar = linkedSetOf(
      "START" to listOf("OPERATOR", "SEMI"),
      "OPERATOR" to listOf("&&"),
      "OPERATOR" to listOf("||"),
      "SEMI" to listOf(";")
    ).freeze()
    val index = CppStatementSyntaxIndex(grammar)
    val aliases: (String) -> Iterable<String> = { terminal -> when (terminal) {
      "&&" -> listOf("&&", "and")
      "||" -> listOf("||", "or")
      else -> listOf(terminal)
    } }

    assertEquals(setOf("&&"), index.terminalsWithSourcePrefix("an", aliases))
    assertEquals(setOf("||"), index.terminalsWithSourcePrefix("o", aliases))
    assertEquals(2, index.minimumSuffixLength(emptyList(), setOf("&&")))
  }
}

/** Strict, epsilon-free CNF for `[ab]{2,};` or `();`. */
private fun smallRecursiveCnf(): CFG = linkedSetOf(
  "START" to listOf("SEQUENCE", "SEMI"),
  "START" to listOf("PARENS", "SEMI"),
  "SEQUENCE" to listOf("ATOM", "ATOM"),
  "SEQUENCE" to listOf("SEQUENCE", "ATOM"),
  "PARENS" to listOf("OPEN", "CLOSE"),
  "ATOM" to listOf("a"),
  "ATOM" to listOf("b"),
  "OPEN" to listOf("("),
  "CLOSE" to listOf(")"),
  "SEMI" to listOf(";")
).freeze()

private fun smallRecursiveLanguageContains(tokens: List<String>): Boolean {
  if (tokens.lastOrNull() != ";") return false
  val body = tokens.dropLast(1)
  return body == listOf("(", ")") || body.size >= 2 && body.all { it == "a" || it == "b" }
}

private fun terminalStrings(alphabet: List<String>, maximumLength: Int): List<List<String>> {
  val byLength = mutableListOf(listOf(emptyList<String>()))
  repeat(maximumLength) { length ->
    byLength += byLength[length].flatMap { prefix ->
      alphabet.map { terminal -> prefix + terminal }
    }
  }
  return byLength.flatten()
}
