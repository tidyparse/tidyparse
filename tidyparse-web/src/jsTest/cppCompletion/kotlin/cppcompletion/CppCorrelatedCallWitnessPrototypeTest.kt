package cppcompletion

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

/**
 * Prototype of the transport/lowering contract for dependent callable templates.
 *
 * A witness is deliberately a flattened, ordered argument sequence. The compiler has already
 * performed deduction, overload resolution, constraints, and any required body instantiation;
 * Kotlin never reconstructs a parameter pack from independent argument choices.
 */
private data class PrototypeCallWitness(
  val callableId: String,
  val resultProfileId: String,
  val arguments: List<String>
)

private data class PrototypeProduction(val lhs: String, val rhs: List<String>)

private class PrototypeArgumentTrie {
  var accepting: Boolean = false
  val successors = linkedMapOf<String, PrototypeArgumentTrie>()
}

/** Lowers a finite relation of correlated argument profiles to an acyclic right-linear CFG. */
private fun lowerPrototypeWitnesses(
  witnesses: Collection<PrototypeCallWitness>
): Map<Pair<String, String>, Set<PrototypeProduction>> =
  witnesses.groupBy { it.callableId to it.resultProfileId }.mapValues { (_, alternatives) ->
    val root = PrototypeArgumentTrie()
    alternatives.map(PrototypeCallWitness::arguments).distinct().forEach { arguments ->
      var node = root
      arguments.forEach { profile ->
        node = node.successors.getOrPut(profile, ::PrototypeArgumentTrie)
      }
      node.accepting = true
    }

    val symbols = mutableMapOf<PrototypeArgumentTrie, String>()
    fun symbol(node: PrototypeArgumentTrie): String =
      symbols.getOrPut(node) { "@CALL_ARGUMENT_SUFFIX_${symbols.size}" }

    val productions = linkedSetOf<PrototypeProduction>()
    fun emit(node: PrototypeArgumentTrie) {
      val lhs = symbol(node)
      if (node.accepting) productions += PrototypeProduction(lhs, emptyList())
      node.successors.forEach { (profile, successor) ->
        if (successor.accepting)
          productions += PrototypeProduction(lhs, listOf(profile))
        if (successor.successors.isNotEmpty()) {
          productions += PrototypeProduction(lhs, listOf(profile, ",", symbol(successor)))
          emit(successor)
        }
      }
    }
    emit(root)
    productions
  }

private fun prototypeLanguage(productions: Set<PrototypeProduction>): Set<List<String>> {
  val byLeft = productions.groupBy(PrototypeProduction::lhs, PrototypeProduction::rhs)
  val memo = mutableMapOf<String, Set<List<String>>>()

  fun expand(symbol: String): Set<List<String>> = memo[symbol] ?: buildSet {
    byLeft.getValue(symbol).forEach { rhs ->
      var prefixes = setOf(emptyList<String>())
      rhs.forEach { item ->
        val suffixes = if (item.startsWith("@")) expand(item) else setOf(listOf(item))
        prefixes = prefixes.flatMapTo(linkedSetOf()) { prefix ->
          suffixes.map { suffix -> prefix + suffix }
        }
      }
      addAll(prefixes)
    }
  }.also { memo[symbol] = it }

  return expand("@CALL_ARGUMENT_SUFFIX_0")
}

class CppCorrelatedCallWitnessPrototypeTest {
  @Test
  fun trieLoweringPreservesWholePackAlternativesWithoutCartesianHybrids() {
    val expected = setOf(
      listOf("key", "mapped"),
      listOf("key", "integer", "text", "floating"),
      listOf("hint", "key", "integer", "text", "floating"),
      listOf("key", "integer", "name", "floating")
    )
    val grammar = lowerPrototypeWitnesses(expected.map { arguments ->
      PrototypeCallWitness("callable", "result", arguments)
    }).getValue("callable" to "result")
    val language = prototypeLanguage(grammar)

    assertEquals(expected.mapTo(linkedSetOf()) { sequence ->
      sequence.flatMapIndexed { index, profile ->
        if (index == 0) listOf(profile) else listOf(",", profile)
      }
    }, language)
    assertFalse(listOf("key", ",", "mapped", ",", "floating") in language)
    assertFalse(listOf("hint", ",", "key", ",", "mapped") in language)
    assertFalse(listOf("key", ",", "integer", ",", "text") in language)
  }

  @Test
  fun callableAndResultIdentityRemainSeparateWitnessRelations() {
    val lowered = lowerPrototypeWitnesses(listOf(
      PrototypeCallWitness("overload-a", "lvalue-result", listOf("text")),
      PrototypeCallWitness("overload-a", "rvalue-result", listOf("integer")),
      PrototypeCallWitness("overload-b", "lvalue-result", emptyList())
    ))

    assertEquals(setOf("text"), prototypeLanguage(lowered.getValue("overload-a" to "lvalue-result")).flatten().toSet())
    assertEquals(setOf("integer"), prototypeLanguage(lowered.getValue("overload-a" to "rvalue-result")).flatten().toSet())
    assertEquals(setOf(emptyList()), prototypeLanguage(lowered.getValue("overload-b" to "lvalue-result")))
  }
}
