package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.wgpu.GPUBufferUsage.STCPSD
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.GPUBuffer
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.toGPUBuffer
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import web.gpu.GPUBuffer
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.test.*

class WGPUPCFGTest {
  private fun gpuTest(block: suspend () -> Unit) = MainScope().promise {
    configureWgpuRuntime(::println) { _, message -> println(message) }
    tryBootstrappingGPU()
    assertTrue(gpuAvailable, "WebGPU is required for PCFG sampling tests")
    block()
  }

  @Test
  fun unloadedGrammarStillUsesUniformEnumeration() = gpuTest {
    val charts = listOf(
      Chart("START -> a [9/10]\nSTART -> b [1/10]", weighted = false) to setOf("a", "b"),
      Chart("""
        START -> x [1/10]
        START -> A B [9/10]
        A -> a [1/2]
        A -> aa [1/2]
        B -> b [1/1]
      """, weighted = false) to setOf("x", "a b", "aa b")
    )
    try {
      for ((chart, expected) in charts) {
        val first = chart.sample(seed = 42, count = expected.size)
        assertEquals(first, chart.sample(seed = 42, count = expected.size))
        val orders = mutableSetOf<List<String>>()
        repeat(12) { seed ->
          val words = chart.sample(seed, expected.size)
          assertEquals(expected, words.toSet())
          assertEquals(expected.size, words.size)
          assertTrue(chart.sampleCosts(seed, expected.size).values.all { it == 0 })
          orders += words
        }
        assertTrue(orders.size > 1, "Changing the seed must affect the ordering")
      }
    } finally { charts.forEach { it.first.close() } }
  }

  @Test
  fun weightedEnumerationIsDeterministicAndExhaustive() = gpuTest {
    val chart = Chart("""
      START -> x [1/10]
      START -> y [1/10]
      START -> A B [7/10]
      START -> C D [1/10]
      A -> a [9/10]
      A -> aa [1/10]
      B -> b [3/10]
      B -> bb [7/10]
      C -> c [1/1]
      D -> d [1/1]
    """)
    try {
      val expected = setOf("x", "y", "a b", "a bb", "aa b", "aa bb", "c d")
      val first = chart.sample(seed = 42, count = expected.size)
      assertEquals(first, chart.sample(seed = 42, count = expected.size))
      val orders = mutableSetOf<List<String>>()
      repeat(12) { seed ->
        val words = chart.sample(seed, expected.size)
        assertEquals(expected, words.toSet())
        assertEquals(expected.size, words.size, "Every tree must appear exactly once")
        orders += words
      }
      assertTrue(orders.size > 1, "Changing the seed must affect the ordering")
    } finally { chart.close() }
  }

  @Test
  fun firstChoiceRenormalizesActiveBinaryAndLexicalWeights() = gpuTest {
    val charts = listOf(
      Chart("""
        START -> A B [9/10]
        START -> C D [1/10]
        A -> a [1/1]
        B -> b [1/1]
        C -> c [1/1]
        D -> d [1/1]
      """) to "a b",
      Chart("""
        START -> a [9/10]
        START -> b [1/10]
      """) to "a",
      Chart("""
        START -> A B [90/100]
        START -> C D [9/100]
        START -> E F [1/100]
        A -> a [1/1]
        B -> b [1/1]
        C -> c [1/1]
        D -> d [1/1]
        E -> e [1/1]
        F -> f [1/1]
      """, excluded = "START" to listOf("A", "B")) to "c d"
    )
    try {
      for ((chart, favored) in charts) {
        var hits = 0
        repeat(128) { seed -> if (chart.sample(seed, 1).single() == favored) hits++ }
        assertTrue(hits in 96..124, "Expected approximately 90% '$favored', got $hits/128")
      }
    } finally { charts.forEach { it.first.close() } }
  }

  @Test
  fun saturatedCountsStillEnumerateWithoutReplacement() = gpuTest {
    val grammar = buildString {
      appendLine("START -> N5 N5 [1/1]")
      for (level in 5 downTo 1) appendLine("N$level -> N${level - 1} N${level - 1} [1/1]")
      appendLine("N0 -> a [9/10]")
      appendLine("N0 -> b [1/10]")
    }
    for (weighted in listOf(true, false)) {
      val chart = Chart(grammar, weighted = weighted)
      try {
        assertEquals(0xffffffffL, chart.total)
        val words = chart.sample(seed = 71, count = 512)
        assertEquals(512, words.toSet().size, "Saturated counts must not introduce duplicate trees")
        assertTrue(words.all { it.split(' ').let { tokens -> tokens.size == 64 && tokens.all { it == "a" || it == "b" } } })
        assertEquals(words, chart.sample(seed = 71, count = 512))
      } finally { chart.close() }
    }
  }

  private fun cost(probability: Double) = (-ln(probability) * SCALE).roundToInt()

  @Test
  fun uniformPCFGStillContributesProductionCosts() = gpuTest {
    val chart = Chart("""
      START -> x [1/2]
      START -> A B [1/2]
      A -> a [1/2]
      A -> aa [1/2]
      B -> b [1/1]
    """)
    try {
      val expected = mapOf("x" to cost(0.5), "a b" to 2 * cost(0.5), "aa b" to 2 * cost(0.5))
      val first = chart.sample(42, expected.size)
      assertEquals(expected.keys, first.toSet())
      assertEquals(first, chart.sample(42, expected.size))
      repeat(4) { seed -> assertEquals(expected, chart.sampleCosts(seed, expected.size)) }
    } finally { chart.close() }
  }

  @Test
  fun cachedCostsSumNestedProductionsWithoutActiveRenormalization() = gpuTest {
    val chart = Chart("""
      START -> A B [1/4]
      START -> C D [3/4]
      A -> E F [1/2]
      A -> G H [1/2]
      B -> b [9/10]
      B -> bb [1/10]
      C -> c [1/1]
      D -> d [1/1]
      E -> e [1/1]
      F -> f [1/1]
      G -> g [1/1]
      H -> h [1/1]
    """, excluded = "START" to listOf("C", "D"))
    try {
      val prefixCost = cost(0.25) + cost(0.5)
      val expected = mapOf(
        "e f b" to prefixCost + cost(0.9), "e f bb" to prefixCost + cost(0.1),
        "g h b" to prefixCost + cost(0.9), "g h bb" to prefixCost + cost(0.1)
      )
      repeat(3) { seed -> assertEquals(expected, chart.sampleCosts(seed, 4)) }
    } finally { chart.close() }
  }

  @Test
  fun cachedCostsUseUnroundedProbabilitiesIncludingZero() = gpuTest {
    val chart = Chart("""
      START -> rare [1/1000000]
      START -> common [999999/1000000]
      START -> zero [0/1]
    """)
    try {
      assertEquals(mapOf("rare" to cost(0.000001), "common" to 0, "zero" to -1), chart.sampleCosts(42, 3))
    } finally { chart.close() }
  }

  @Test
  fun ngramTopKCombinesCachedPCFGAndDownstreamCosts() = gpuTest {
    // The downstream model favors b, but P(a)*P_ngram(a) > P(b)*P_ngram(b).
    val packets = listOf(0, cost(0.9), 4, 0, 0, cost(0.1), 5, 0, 0, -1, 4, 0).toGPUBuffer(STCPSD)
    val model = mapOf(4 to cost(0.25), 5 to cost(0.75)).map { (token, score) ->
      listOf(BOS_ID - 1, NEWLINE_ID - 1, token - 1, NEWLINE_ID - 1)
        .map { (it + FIRST_TID).toUInt() } to score.toUInt()
    }.toMap().loadToGPUBuffer()
    try {
      val top = scoreSelectGather(packets, model, markov_score, 3, 4, 1)
      assertEquals(4, top[PKT_HDR_LEN])
      assertEquals(10_000_000 + cost(0.9) + cost(0.25) + 1, top[1])
      assertEquals(-1, packets.readJSIntArray()[9], "A zero PCFG probability must remain invalid")
    } finally { packets.destroy(); model.destroy() }
  }

  @Test
  fun wdfaTopKCombinesCostsAtModelScaleAndPreservesInvalidPaths() = gpuTest {
    val inf = 0x3fffffff
    val packets = listOf(
      0, cost(0.9), 4, 0, 0, cost(0.1), 5, 0,
      0, 0, 6, 0, 0, 0, 7, 0, 0, -1, 4, 0
    ).toGPUBuffer(STCPSD)
    // Four states: a/b accept, c reaches a nonaccepting state, d has no edge.
    val model = listOf(
      0, 1, 1000, 4, 3, 0, 7, inf,
      0, 4, 4, 5, 9, 3, 12, 3, 15, 3,
      inf, 11, 13, inf,
      0, 3, 3, 3, 3,
      4, 5, 6,
      1, 2, 3,
      1386, 288, 0
    ).toGPUBuffer(STCPSD)
    try {
      val top = scoreSelectGather(packets, model, wdfa_score, 5, 4, 1)
      assertEquals(4, top[PKT_HDR_LEN])
      assertEquals(10_000_000 + 105 + 7 + 1386 + 11, top[1])
      val scored = packets.readJSIntArray()
      for (row in 2..4) assertEquals(-1, scored[row * 4 + 1], "Invalid sample $row must remain invalid")
    } finally { packets.destroy(); model.destroy() }
  }

  /** An acyclic chart with one cell per NT; inspect raw packets so deduplication cannot hide failures. */
  private class Chart(pcfg: String, excluded: Pair<String, List<String>>? = null, weighted: Boolean = true) {
    private val rules = pcfg.trimIndent().trim().lines()
    private val cfg = rules.joinToString("\n") { it.substringBeforeLast(" [") }.parseCNF()
    private val nts = cfg.nonterminals.toList()
    private val root = cfg.bindex[START_SYMBOL]
    private val literals = nts.map { nt -> cfg.count { it.first == nt && it.second.size == 1 } }
    private val branches = nts.map { nt -> cfg.filter { it.first == nt && it.second.size == 2 && it != excluded }
      .map { cfg.bindex[it.second[0]] to cfg.bindex[it.second[1]] } }
    private val sizes = mutableMapOf<Int, Long>()
    private val buffers = mutableListOf<GPUBuffer>()
    val total = size(root)

    private fun product(left: Int, right: Int): Long {
      val l = size(left)
      val r = size(right)
      return if (l > 0xffffffffL / r) 0xffffffffL else l * r
    }

    private fun size(nt: Int): Long = sizes.getOrPut(nt) {
      branches[nt].fold(literals[nt].toLong()) { sum, (left, right) ->
        (sum + product(left, right)).coerceAtMost(0xffffffffL)
      }
    }

    private fun buffer(values: List<Int>): GPUBuffer =
      values.ifEmpty { listOf(0) }.toGPUBuffer(STCPSD).also { buffers += it }

    private val dp = buffer(literals.map { if (it == 0) 1 else 0x47fffffe })
    private val counts = buffer(branches.map { it.size })
    private val offsets = buffer(branches.scan(0) { offset, row -> offset + row.size }.dropLast(1))
    private val storage = buffer(branches.flatten().flatMap { listOf(it.first, it.second) })
    private val cdf = buffer(branches.flatMapIndexed { nt, row ->
      row.scan(literals[nt].toLong()) { sum, (left, right) ->
        (sum + product(left, right)).coerceAtMost(0xffffffffL)
      }.drop(1).map { it.toInt() }
    })
    private val rootSizes = buffer(listOf(total.toInt()))
    private val rootCdf = buffer(listOf(0))

    init {
      if (weighted) cfg.loadPCFG(pcfg.trimIndent().trim())
      else assertNull(cfg.pcfgBuf, "An unloaded grammar must not allocate a uniform PCFG")
    }
    suspend fun sample(seed: Int, count: Int): List<String> = sampleWordsAndCosts(seed, count).map { it.first }

    suspend fun sampleCosts(seed: Int, count: Int): Map<String, Int> = sampleWordsAndCosts(seed, count).toMap()

    private suspend fun sampleWordsAndCosts(seed: Int, count: Int): List<Pair<String, Int>> {
      val stride = MAX_WORD_LEN + PKT_HDR_LEN + 1
      val indices = listOf(seed, stride, nts.size, 1, count, count, 0, 2, root, 0).toGPUBuffer(STCPSD)
      val output = GPUBuffer(count * stride * 4, STCPSD)
      val sampling = cfg.pcfgBuf?.let {
        preparePCFGSampling(dp, counts, offsets, storage, cdf, cfg.termBuf, indices, rootCdf, it)
      }
      try {
        enum_words_wor(dp, counts, offsets, storage, cdf, cfg.termBuf, indices, rootSizes, sampling ?: rootCdf, output)(count)
        val packets = output.readJSIntArray()
        return List(count) { row ->
          (PKT_HDR_LEN until stride).asSequence().map { packets[row * stride + it] }
            .takeWhile { it != 0 }.joinToString(" ") { cfg.tmLst[(it and PACKED_TOKEN_MASK) - 1] } to packets[row * stride + 1]
        }
      } finally { indices.destroy(); output.destroy(); sampling?.destroy() }
    }

    fun close() { buffers.forEach { it.destroy() } }
  }
}
