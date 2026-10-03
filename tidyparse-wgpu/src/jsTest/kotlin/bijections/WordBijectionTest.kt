package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.readIndices
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.toGPUBuffer
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.test.*

class WordBijectionTest {
  private enum class Bijection { UNIFORM, GREEDY, HISTOGRAM }

  private data class Language(val grammar: String, val words: Set<String>) {
    val lengths = words.map { it.split(' ').size }.toSet()
  }

  private val mixed = Language("""
    START -> solo [1/10]
    START -> A B [8/10]
    START -> C D [1/10]
    A -> a [9/10]
    A -> aa [1/10]
    B -> b [1/2]
    B -> bb [1/3]
    B -> bbb [1/6]
    C -> c [1/1]
    D -> d [3/4]
    D -> dd [1/4]
  """, setOf("solo", "a b", "a bb", "a bbb", "aa b", "aa bb", "aa bbb", "c d", "c dd"))

  private fun gpuTest(block: suspend () -> Unit) = MainScope().promise {
    tryBootstrappingGPU()
    assertTrue(gpuAvailable, "WebGPU is required for word bijection tests")
    block()
  }

  @Test
  fun bijectionsEnumerateTheSameFiniteLanguageInDifferentOrders() = gpuTest {
    val orders = Bijection.entries.map { bijection ->
      sample(mixed, bijection).also { words ->
        assertEquals(mixed.words, words.toSet(), "$bijection must cover the whole language")
        assertEquals(mixed.words.size, words.size, "$bijection must emit each word once")
      }
    }
    assertEquals(Bijection.entries.size, orders.toSet().size,
      "The three bijections should give distinct orders for this weighted language")
  }

  @Test
  fun lexicalAndNestedLanguagesAreExhaustiveWithOrWithoutWeights() = gpuTest {
    val languages = listOf(
      Language("START -> only [1/1]", setOf("only")),
      Language("START -> a [3/5]\nSTART -> b [2/5]\nSTART -> c [0/5]", setOf("a", "b", "c")),
      Language("""
        START -> A X [1/1]
        X -> B C [1/1]
        A -> a [3/4]
        A -> aa [1/4]
        B -> b [1/1]
        C -> c [2/3]
        C -> cc [1/3]
      """, setOf("a b c", "a b cc", "aa b c", "aa b cc"))
    )
    for (language in languages) for (bijection in Bijection.entries) for (weighted in listOf(true, false)) {
      val words = sample(language, bijection, weighted = weighted)
      assertEquals(language.words, words.toSet(), "$bijection, weighted=$weighted")
      assertEquals(language.words.size, words.size, "Enumeration must be without replacement")
    }
  }

  @Test
  fun samplingIsRepeatableWithoutReplacementAndRespectsTheRequestedCount() = gpuTest {
    for (bijection in Bijection.entries) {
      val all = sample(mixed, bijection)
      assertEquals(all, sample(mixed, bijection), "$bijection must repeat the same seeded order")
      val otherSeed = sample(mixed, bijection, seed = 73)
      assertEquals(all.toSet(), otherSeed.toSet())
      assertNotEquals(all, otherSeed, "$bijection must respond to the seed")
      for (count in listOf(1, 3, mixed.words.size - 1)) {
        val words = sample(mixed, bijection, count = count)
        assertEquals(count, words.size)
        assertEquals(count, words.toSet().size, "$bijection must not repeat words")
        assertTrue(mixed.words.containsAll(words), "$bijection must stay in the language")
        assertEquals(words, sample(mixed, bijection, count = count))
        if (bijection == Bijection.UNIFORM)
          assertEquals(all.take(count), words, "Uniform samples use the same full-language permutation")
      }
    }
  }

  @Test
  fun allBijectionsRespectTheIntersectionMask() = gpuTest {
    val expected = setOf("a b", "a bb", "a bbb")
    for (bijection in Bijection.entries) {
      val words = sample(mixed, bijection, count = expected.size, mask = listOf("a", "_"))
      assertEquals(expected, words.toSet(), "$bijection must respect the fixed terminal")
      assertEquals(expected.size, words.size)
    }
  }

  // Use the production chart construction and enumeration path; compare only decoded words.
  private suspend fun sample(
    language: Language, bijection: Bijection, count: Int = language.words.size, seed: Int = 42,
    weighted: Boolean = true, mask: List<String> = List(language.lengths.max()) { "_" }
  ): List<String> = GPUBufferScope().use { owned ->
    val rules = language.grammar.trimIndent().trim()
    val cfg = rules.lines().joinToString("\n") { it.substringBeforeLast(" [") }.parseCNF()
    if (weighted) cfg.loadPCFG(rules)
    val fsa = makePorousFSA(mask, language.lengths)
    val states = fsa.numStates
    val nts = cfg.nonterminals.size
    val stride = mask.size + PKT_HDR_LEN + 1
    val meta = owned.own(Shader.packMetadata(cfg, fsa))
    val input = owned.own(porousToCodePoints(cfg, mask).toGPUBuffer())
    val dp = owned.newBuffer(states.toLong() * states * nts * 4)
    val active = owned.newBuffer(states.toLong() * states * ((nts + 31) / 32) * 4)
    init_line_chart(dp, active, input, meta, cfg.termBuf)
      .invoke(states, states, (nts + DENSE_NT_WORKGROUP_SIZE - 1) / DENSE_NT_WORKGROUP_SIZE)
    cfl_mul_upper.invokeCFLFixpoint(states, dp, active, meta)
    val (counts, offsets, children) = Shader.buildBackpointers(states, nts, dp, meta)
    owned.own(counts, offsets, children)
    val cdf = owned.newBuffer(maxOf(4L, children.size.toLong() / 2))
    Shader.buildLanguageSizeCDF(states, nts, dp, meta, cfg.termBuf, counts, offsets, children, cdf)

    val candidates = fsa.finalIdxs.map { it * nts + cfg.bindex[START_SYMBOL] }
    val reachable = dp.readIndices(candidates)
    val roots = candidates.filterIndexed { i, _ -> reachable[i] != 0 }.flatMap { listOf(it, 0) }
    assertTrue(roots.isNotEmpty(), "The fixture must have an accepting parse")
    val indices = owned.own(packStruct(listOf(seed, stride, nts, states, DISPATCH_GROUP_SIZE_X, count),
      roots.toGPUBuffer()))
    val index = owned.own(when (bijection) {
      Bijection.UNIFORM -> uniformIndex(states, nts, dp, counts, offsets, children, cdf, cfg.termBuf, indices)
      Bijection.GREEDY -> greedyPCFGIndex(states, nts, dp, counts, offsets, children, cdf, cfg.termBuf, indices, cfg.pcfgBuf)
      Bijection.HISTOGRAM -> histogramSemiringIndex(states, nts, dp, counts, offsets, children, cdf, cfg.termBuf, indices, cfg.pcfgBuf)
    })
    val output = owned.newBuffer(count * stride * 4)
    enum_words_wor(index, indices, output).dispatchFlat(count)
    val packets = output.readJSIntArray()
    // Preserve order and duplicates so the assertions can detect omissions or repeated words.
    IntersectionResults(List(count) { i ->
      assertNotNull(packets.decodePacket(i, cfg.tmLst.size, stride), "$bijection emitted an invalid word")
    }, cfg.tmLst).toList()
  }
}
