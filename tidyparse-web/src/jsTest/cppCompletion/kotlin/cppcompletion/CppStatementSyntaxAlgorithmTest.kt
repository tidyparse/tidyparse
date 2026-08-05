package cppcompletion

import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import ai.hypergraph.kaliningraph.parsing.freeze
import ai.hypergraph.kaliningraph.parsing.matches
import ai.hypergraph.kaliningraph.parsing.tmLst
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

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
    val firstTerminalSets = listOf(emptySet(), setOf("id"), setOf("+"), setOf(";"))

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

      firstTerminalSets.forEach { allowed ->
        val constrainedByLength = witnessesByLength.mapValues { (_, suffixes) ->
          suffixes.filter { it.firstOrNull() in allowed }
        }
        val expectedConstrained = constrainedByLength.entries
          .firstOrNull { it.value.isNotEmpty() }?.key
        val actualConstrained = index.minimumSuffixLength(prefix, allowed)
        assertEquals(
          expectedConstrained,
          actualConstrained,
          "Wrong first-terminal-constrained continuation after $prefix for $allowed"
        )
        if (actualConstrained != null) {
          val forest = assertNotNull(
            index.completeShortestSuffix(prefix, actualConstrained, allowed)
          )
          val residual = forest.toAcyclicCfg(grammar.tmLst).boundedAcyclic(actualConstrained)
          alphabet.words(actualConstrained).forEach { suffix ->
            assertEquals(
              suffix in constrainedByLength.getValue(actualConstrained),
              residual.recognizes(suffix),
              "Constrained forest disagrees after $prefix on suffix $suffix for $allowed"
            )
          }
        }
      }
    }
  }

  @Test
  fun productionSyntaxResidualConstrainsTheFirstEmittedTerminal() {
    val returnResidual = assertNotNull(
      cppSingleStatementSyntaxCompletion(
        emptyList(), CppToken("ret", 0, 3, CppTokenKind.IDENTIFIER)
      )
    )
    assertTrue(returnResidual.bounded.recognizes(listOf("return", ";")))
    assertFalse(returnResidual.bounded.recognizes(listOf("throw", ";")))

    val prefix = cppLines("flag = left").single().tokens
    val aliasResidual = assertNotNull(cppSingleStatementSyntaxCompletion(
      prefix, CppToken("an", 0, 2, CppTokenKind.IDENTIFIER)
    ))
    assertTrue(
      aliasResidual.bounded.recognizes(listOf("&&", CPP_SYNTAX_IDENTIFIER, ";")),
      "The alias callback must constrain by the projected grammar terminal"
    )
  }
}

private fun <T> List<T>.words(length: Int): List<List<T>> = when (length) {
  0 -> listOf(emptyList())
  else -> words(length - 1).flatMap { prefix -> map { terminal -> prefix + terminal } }
}
