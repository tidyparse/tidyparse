package cppcompletion

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.PreindexedAcyclicCFG
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import ai.hypergraph.kaliningraph.parsing.freeze
import com.ionspin.kotlin.bignum.integer.BigInteger
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Exhaustive finite-language checks for the prepared grammar's incremental left quotients. */
class CppFiniteConditionerTest {
  private class TestPreindexedCfg(
    private val productions: LinkedHashSet<Pair<String, List<String>>>,
    override val acyclicCountingOrder: List<String>
  ) : AbstractSet<Pair<String, List<String>>>(), PreindexedAcyclicCFG {
    private val byLhs = productions.groupBy { it.first }
    override val acyclicNonterminalIndex = acyclicCountingOrder.withIndex()
      .associate { (index, symbol) -> symbol to index }
    override val acyclicStructuralStats = "test-preindexed-cfg"
    override val size: Int get() = productions.size
    override fun iterator(): Iterator<Pair<String, List<String>>> = productions.iterator()
    override fun productionsFor(nonterminal: String): List<Pair<String, List<String>>> =
      byLhs[nonterminal].orEmpty()
  }

  private fun token(text: String, position: Int = 0) =
    CppToken(text, position, position + text.length, CppTokenKind.OTHER)

  private fun tokens(terminals: List<String>): List<CppToken> =
    terminals.mapIndexed { index, terminal -> token(terminal, index) }

  private fun derivations(grammar: CFG): Map<List<String>, Int> {
    val nonterminals = grammar.mapTo(linkedSetOf()) { it.first }
    val rules = grammar.groupBy { it.first }
    val memo = mutableMapOf<String, Map<List<String>, Int>>()

    fun concatenate(
      left: Map<List<String>, Int>,
      right: Map<List<String>, Int>
    ): Map<List<String>, Int> = buildMap {
      left.forEach { (leftYield, leftCount) ->
        right.forEach { (rightYield, rightCount) ->
          val yield = leftYield + rightYield
          put(yield, getOrElse(yield) { 0 } + leftCount * rightCount)
        }
      }
    }

    fun expand(nonterminal: String): Map<List<String>, Int> = memo.getOrPut(nonterminal) {
      buildMap {
        rules[nonterminal].orEmpty().forEach { (_, rhs) ->
          val expanded = rhs.fold(mapOf(emptyList<String>() to 1)) { prefix, symbol ->
            val child = if (symbol in nonterminals) expand(symbol)
            else mapOf(listOf(symbol) to 1)
            concatenate(prefix, child)
          }
          expanded.forEach { (yield, count) ->
            put(yield, getOrElse(yield) { 0 } + count)
          }
        }
      }
    }

    return expand("START")
  }

  private fun Map<List<String>, Int>.leftQuotient(prefix: List<String>): Map<List<String>, Int> =
    buildMap {
      this@leftQuotient.forEach { (yield, count) ->
        if (yield.size >= prefix.size && yield.take(prefix.size) == prefix) {
          val suffix = yield.drop(prefix.size)
          put(suffix, getOrElse(suffix) { 0 } + count)
        }
      }
    }

  private fun words(alphabet: List<String>, maxLength: Int): List<List<String>> = buildList {
    var frontier = listOf(emptyList<String>())
    addAll(frontier)
    repeat(maxLength) {
      frontier = frontier.flatMap { prefix -> alphabet.map { prefix + it } }
      addAll(frontier)
    }
  }

  private fun assertEveryResidualMatchesFiniteOracle(grammar: CFG) {
    val language = derivations(grammar)
    val alphabet = grammar.asSequence().flatMap { it.second.asSequence() }
      .filter { terminal -> grammar.none { it.first == terminal } }
      .distinct().sorted().toList()
    val maximumYield = language.keys.maxOfOrNull(List<String>::size) ?: 0
    val prefixes = buildList {
      language.keys.forEach { yield ->
        (0..yield.size).forEach { length -> add(yield.take(length)) }
      }
      add(listOf("not-in-language"))
      add(listOf(alphabet.first(), "not-in-language"))
    }.distinct()
    // This order exercises both the extending-prefix cache and its reset path.
    val prepared = PreparedCppCompletionGrammar(grammar)
    words(alphabet, maximumYield).forEach { statement ->
      assertEquals(
        statement in language,
        prepared.recognizes(tokens(statement)),
        "Wrong complete-language membership for $statement"
      )
    }
    prefixes.sortedWith(compareBy<List<String>> { it.firstOrNull().orEmpty() }.thenBy { it.size })
      .forEach { prefix ->
        val expected = language.leftQuotient(prefix)
        val residual = prepared.generate(tokens(prefix))
        val actualCount = expected.values.sum()
        assertEquals(
          BigInteger.fromInt(actualCount),
          residual.derivationCount,
          "Wrong residual derivation count after $prefix"
        )

        val sampled = buildMap<List<String>, Int> {
          repeat(actualCount) { rank ->
            val yield = residual.bounded.sample(BigInteger.fromInt(rank))
            put(yield, getOrElse(yield) { 0 } + 1)
          }
        }
        assertEquals(expected, sampled, "Wrong residual derivation multiset after $prefix")

        words(alphabet, maximumYield).forEach { suffix ->
          assertEquals(
            suffix in expected,
            residual.bounded.recognizes(suffix),
            "Wrong residual language membership after $prefix for suffix $suffix"
          )
        }

        (residual.syntax as? PreindexedAcyclicCFG)?.let { indexed ->
          indexed.acyclicCountingOrder.forEachIndexed { parentIndex, parent ->
            indexed.productionsFor(parent).forEach { (_, rhs) ->
              rhs.filter { it in indexed.acyclicNonterminalIndex }.forEach { child ->
                assertTrue(
                  indexed.acyclicNonterminalIndex.getValue(child) < parentIndex,
                  "Published order places $child after its parent $parent"
                )
              }
            }
          }
        }

        val frozen = residual.syntax.freeze().boundedAcyclic(
          maxLength = residual.templateTokens,
          startSymbol = residual.bounded.startSymbol
        )
        assertEquals(residual.derivationCount, frozen.derivationCount)
        words(alphabet, maximumYield).forEach { suffix ->
          assertEquals(residual.bounded.recognizes(suffix), frozen.recognizes(suffix))
        }
      }
  }

  @Test
  fun everyNullableAndUnitResidualMatchesItsFrozenBruteForceQuotient() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("S"),
      "S" to listOf("LEFT", "RIGHT"),
      "LEFT" to emptyList(),
      "LEFT" to listOf("A"),
      "A" to listOf("a"),
      "RIGHT" to emptyList(),
      "RIGHT" to listOf("B"),
      "RIGHT" to listOf("B", "C"),
      "B" to listOf("b"),
      "C" to listOf("c")
    )

    assertEveryResidualMatchesFiniteOracle(grammar)
  }

  @Test
  fun ambiguousBranchesRemainDistinctAcrossOneDerivative() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("LEFT"),
      "START" to listOf("RIGHT"),
      "LEFT" to listOf("a"),
      "RIGHT" to listOf("a")
    )

    assertEveryResidualMatchesFiniteOracle(grammar)
  }

  @Test
  fun nullableDerivativeMultiplicitySurvivesTheNextCommittedToken() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("S"),
      "S" to listOf("LEFT", "RIGHT"),
      "LEFT" to listOf("FIRST"),
      "LEFT" to listOf("SECOND"),
      "FIRST" to listOf("a"),
      "SECOND" to listOf("a"),
      "RIGHT" to listOf("b")
    )

    assertEveryResidualMatchesFiniteOracle(grammar)
  }

  @Test
  fun ambiguousSourceEpsilonMultiplicitySurvivesACommittedToken() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("LEFT", "RIGHT"),
      "LEFT" to listOf("FIRST_EMPTY"),
      "LEFT" to listOf("SECOND_EMPTY"),
      "FIRST_EMPTY" to emptyList(),
      "SECOND_EMPTY" to emptyList(),
      "RIGHT" to listOf("b")
    )

    assertEveryResidualMatchesFiniteOracle(grammar)
  }

  @Test
  fun incrementalAndNonPrefixCursorOrdersPreserveTheSameResidualMultiset() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("LEFT", "RIGHT"),
      "LEFT" to listOf("FIRST_EMPTY"),
      "LEFT" to listOf("SECOND_EMPTY"),
      "LEFT" to listOf("A"),
      "FIRST_EMPTY" to emptyList(),
      "SECOND_EMPTY" to emptyList(),
      "A" to listOf("a"),
      "RIGHT" to listOf("b")
    )
    val language = derivations(grammar)
    val cursorOrder = listOf(
      emptyList(), listOf("a"), listOf("a", "b"), // extending cache
      listOf("b"), emptyList(), listOf("b"),       // non-prefix resets
      listOf("a"), listOf("not-in-language"), listOf("a", "b")
    )
    val prepared = PreparedCppCompletionGrammar(grammar)

    cursorOrder.forEach { prefix ->
      val expected = language.leftQuotient(prefix)
      val residual = prepared.generate(tokens(prefix))
      val expectedCount = expected.values.sum()
      assertEquals(
        BigInteger.fromInt(expectedCount), residual.derivationCount,
        "Wrong derivation count after cursor order prefix $prefix"
      )
      val ranked = buildMap<List<String>, Int> {
        repeat(expectedCount) { rank ->
          val suffix = residual.bounded.sample(BigInteger.fromInt(rank))
          put(suffix, getOrElse(suffix) { 0 } + 1)
        }
      }
      assertEquals(expected, ranked, "Wrong derivation multiset after cursor order prefix $prefix")
      words(listOf("a", "b"), 2).forEach { suffix ->
        assertEquals(
          suffix in expected, residual.bounded.recognizes(suffix),
          "Wrong membership after cursor order prefix $prefix for $suffix"
        )
      }
    }
  }

  @Test
  fun preindexedSourceGrammarUsesTheSameExactResidualEquations() {
    val grammar = TestPreindexedCfg(
      linkedSetOf(
        "START" to listOf("EMPTY", "A"),
        "EMPTY" to emptyList(),
        "A" to listOf("a")
      ),
      acyclicCountingOrder = listOf("EMPTY", "A", "START")
    )

    assertEveryResidualMatchesFiniteOracle(grammar)
  }

  @Test
  fun indexedCnfPublicationPreservesFrozenSourceCountsRecognitionAndSamples() {
    val indexed = finiteAcyclicCnf(linkedSetOf(
      "START" to listOf("OPTIONAL", "+", "VALUE"),
      "START" to listOf("VALUE"),
      "START" to listOf("DEAD"),
      "OPTIONAL" to emptyList(),
      "OPTIONAL" to listOf("VALUE"),
      "VALUE" to listOf("x"),
      "VALUE" to listOf("y"),
      "DEAD" to listOf("TYPE_999_D0"),
      "UNREACHABLE" to listOf("z")
    ))
    assertTrue(indexed is PreindexedAcyclicCFG)
    val frozen = indexed.toSet().freeze()
    assertEquals(frozen, indexed)
    assertFalse(indexed.any { (lhs) -> lhs == "DEAD" || lhs == "UNREACHABLE" })

    val indexedBounded = indexed.boundedAcyclic(maxLength = 3)
    val frozenBounded = frozen.boundedAcyclic(maxLength = 3)
    assertEquals(frozenBounded.structuralStats(), indexedBounded.structuralStats())
    assertEquals(frozenBounded.derivationCount, indexedBounded.derivationCount)
    (0..3).forEach { length ->
      assertEquals(
        frozenBounded.derivationCount("START", length),
        indexedBounded.derivationCount("START", length)
      )
    }
    words(listOf("x", "y", "+"), 3).forEach { terminals ->
      assertEquals(frozenBounded.recognizes(terminals), indexedBounded.recognizes(terminals))
    }
    assertEquals(
      frozenBounded.samplesByIncreasingLength(Random(17), sampleLimit = 30).toList(),
      indexedBounded.samplesByIncreasingLength(Random(17), sampleLimit = 30).toList()
    )
  }

  @Test
  fun cnfFastPublicationRetainsSetSemanticsForDuplicateInputsAndReservedNames() {
    val duplicated = listOf(
      "START" to listOf("VALUE"),
      "START" to listOf("VALUE"),
      "VALUE" to listOf("x"),
      "VALUE" to listOf("x")
    )
    assertEquals(
      linkedSetOf("START" to listOf("VALUE"), "VALUE" to listOf("x")),
      finiteAcyclicCnf(duplicated)
    )

    val reservedCollision = linkedSetOf(
      "START" to listOf("A", "+", "B"),
      "START" to listOf("FINITE_TERMINAL_0"),
      "FINITE_TERMINAL_0" to listOf("+"),
      "A" to listOf("a"),
      "B" to listOf("b")
    )
    val indexed = finiteAcyclicCnf(reservedCollision)
    assertEquals(indexed.toSet().size, indexed.size)
    assertEquals(
      indexed.toSet().freeze().boundedAcyclic(3).derivationCount,
      indexed.boundedAcyclic(3).derivationCount
    )
  }

  @Test
  fun exactMembershipAccountsForNullableBinaryEdges() {
    val grammar: CFG = linkedSetOf(
      "START" to listOf("EMPTY", "A"),
      "EMPTY" to emptyList(),
      "A" to listOf("a")
    )
    val prepared = PreparedCppCompletionGrammar(grammar)

    assertTrue(prepared.generate(emptyList()).bounded.recognizes(listOf("a")))
    assertFalse(prepared.recognizes(tokens(listOf("not-in-language"))))
    // This assertion documents the exact-membership contract independently of residual CYK.
    assertTrue(prepared.recognizes(tokens(listOf("a"))))
  }
}
