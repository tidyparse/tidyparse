package cppcompletion

import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import ai.hypergraph.kaliningraph.parsing.freeze
import ai.hypergraph.kaliningraph.parsing.matches
import ai.hypergraph.kaliningraph.parsing.tmLst
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class CppStatementSyntaxAlgorithmTest {
  @Test
  fun minPlusRecurrenceAndExactForestAgreeWithExhaustiveLanguageQuotients() {
    // id (+ id)* ; -- deliberately left recursive, with no token horizon in the implementation.
    val grammar = setOf(
      "START" to listOf("EXPRESSION", "SEMICOLON"),
      "EXPRESSION" to listOf("id"),
      "EXPRESSION" to listOf("EXPRESSION", "TAIL"),
      "TAIL" to listOf("PLUS", "EXPRESSION"),
      "PLUS" to listOf("+"),
      "SEMICOLON" to listOf(";")
    ).freeze()
    val index = CppStatementSyntaxIndex(grammar)
    val alphabet = listOf("id", "+", ";")

    (0..4).flatMap { alphabet.words(it) }.forEach { prefix ->
      val witnessesByLength = (0..3).associateWith { length ->
        alphabet.words(length).filter { suffix -> (prefix + suffix).matches(grammar) }
      }
      val expectedMinimum = witnessesByLength.entries.firstOrNull { it.value.isNotEmpty() }?.key
      val actualMinimum = index.minimumSuffixLength(prefix)
      assertEquals(expectedMinimum, actualMinimum, "Wrong minimum continuation length after $prefix")

      if (actualMinimum != null && actualMinimum > 0) {
        val forest = assertNotNull(index.completeShortestSuffix(prefix, actualMinimum))
        val residual = forest.toAcyclicCfg(grammar.tmLst).boundedAcyclic(actualMinimum)
        alphabet.words(actualMinimum).forEach { suffix ->
          assertEquals(
            suffix in witnessesByLength.getValue(actualMinimum),
            residual.recognizes(suffix),
            "Exact forest disagrees after $prefix on suffix $suffix"
          )
        }
      }
    }
  }
}

private fun <T> List<T>.words(length: Int): List<List<T>> = when (length) {
  0 -> listOf(emptyList())
  else -> words(length - 1).flatMap { prefix -> map { terminal -> prefix + terminal } }
}
