package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.wgpu.GPUBufferUsage.STCPSD
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.GPUBuffer
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.readIndices
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.toGPUBuffer
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import web.gpu.GPUBuffer
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.test.*

class WGPUWCFGTest {
  private fun gpuTest(block: suspend () -> Unit) = MainScope().promise {
    configureWgpuRuntime(::println) { _, message -> println(message) }
    tryBootstrappingGPU()
    assertTrue(gpuAvailable, "WebGPU is required for PCFG sampling tests")
    block()
  }

  @Test
  fun logarithmicCostTiersPreserveBoundariesAndTranslateWithTheBestCost() {
    val boundaries = mapOf(
      0u to 0, 1u to 1, 2u to 2, 3u to 2,
      4u to 3, 7u to 3, 8u to 4, 15u to 4,
      16u to 5, 31u to 5, 32u to 6,
      255u to 8, 256u to 9, 65535u to 16, 65536u to 17
    )
    for ((delta, tier) in boundaries) {
      assertEquals(tier, pcfgCostTier(delta, 0u), "Wrong tier for cost delta $delta")
      assertEquals(tier, pcfgCostTier(100_000u + delta, 100_000u), "Tiers must depend on excess cost")
    }
    assertEquals(31, pcfgCostTier(0x7fffffffu, 0u))
    assertEquals(32, pcfgCostTier(0x80000000u, 0u))
    assertEquals(32, pcfgCostTier(UInt.MAX_VALUE - 1u, 0u))
    assertEquals(0, pcfgCostTier(UInt.MAX_VALUE - 1u, UInt.MAX_VALUE - 1u))
    assertEquals(33, pcfgCostTier(UInt.MAX_VALUE, 0u))
    assertEquals(33, pcfgCostTier(UInt.MAX_VALUE, UInt.MAX_VALUE))
  }

  @Test
  fun boundedLogarithmicTiersAdaptToTheCostSpanAndReserveInfinity() {
    for (budget in listOf(2, 10, 100)) {
      val scale = PCFGCostTierScale(0u, 65_536u, true, budget)
      val shifted = PCFGCostTierScale(100_000u, 165_536u, true, budget)
      var previous = -1
      for (delta in 0..65_536) {
        val tier = scale.tier(delta.toUInt())
        assertTrue(tier >= previous, "Logarithmic tiers must be monotone")
        assertTrue(tier in 0 until budget - 1, "Finite costs must leave a tier for infinity")
        assertEquals(tier, shifted.tier(100_000u + delta.toUInt()))
        previous = tier
      }
      assertEquals(budget - 1, scale.tier(UInt.MAX_VALUE))
      assertEquals(0, scale.tier(0u))
    }
    val hundred = PCFGCostTierScale(0u, 65_536u, false, 100)
    assertEquals(99, hundred.tier(65_536u))
    assertTrue((0..65_536).map { hundred.tier(it.toUInt()) }.toSet().size > 34)
    assertEquals(0, PCFGCostTierScale(7u, 7u, false, 100).tier(7u))
    assertEquals(0, PCFGCostTierScale(UInt.MAX_VALUE, null, true, 100).tier(UInt.MAX_VALUE))
    val adjacent = PCFGCostTierScale(UInt.MAX_VALUE - 2u, UInt.MAX_VALUE - 1u, true, 100)
    assertTrue(adjacent.tier(UInt.MAX_VALUE - 2u) < adjacent.tier(UInt.MAX_VALUE - 1u))
    assertTrue(adjacent.tier(UInt.MAX_VALUE - 1u) < adjacent.tier(UInt.MAX_VALUE))
    assertFailsWith<IllegalArgumentException> { PCFGCostTierScale(0u, 1u, false, 1) }
  }

  @Test
  fun hundredBucketsPreserveTheDyadicEnumerationAndRequestedPrefixes() = gpuTest {
    val previous = pcfgHistogramBucketCount
    val grammar = buildString {
      for (numerator in 1..300) appendLine("START -> token$numerator [$numerator/45150]")
      appendLine("START -> zero [0/45150]")
    }
    val chart = Chart(grammar, inspectIndex = true)
    try {
      pcfgHistogramBucketCount = null
      val baseline = chart.sampleCosts(42, 301)
      val oldTierCount = chart.lastTierCount
      pcfgHistogramBucketCount = 100
      val actual = chart.sampleCosts(42, 301)
      assertEquals(baseline.toList(), actual.toList())
      assertTrue(chart.lastTierCount > oldTierCount, "A larger bucket budget must refine the histogram")
      assertTrue(chart.lastTierCount <= 100, "Infinity must fit within the per-constituent budget")
      assertEquals(301, actual.size)
      assertEquals("zero", actual.keys.last())
      for (requested in listOf(1, 99, 100, 250))
        assertEquals(actual.keys.take(requested), chart.sample(42, requested))
    } finally {
      pcfgHistogramBucketCount = previous
      chart.close()
    }
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
  fun weightedEnumerationIsDeterministicExhaustiveAndGloballyOrdered() = gpuTest {
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
      repeat(4) { seed ->
        val words = chart.sample(seed, expected.size)
        assertEquals(expected, words.toSet())
        assertEquals(expected.size, words.size, "Every tree must appear exactly once")
        val costs = chart.sampleCosts(seed, expected.size).values.toList()
        assertEquals(costs.sorted(), costs, "Enumeration must follow whole-tree PCFG cost")
        assertEquals(words.take(3), chart.sample(seed, 3), "Each prefix must contain the best-scoring trees")
      }
    } finally { chart.close() }
  }

  @Test
  fun firstChoiceAlwaysUsesTheHighestScoringActiveDerivation() = gpuTest {
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
        repeat(4) { seed ->
          assertEquals(favored, chart.sample(seed, 1).single(), "Seed must not change the best PCFG score")
        }
      }
    } finally { charts.forEach { it.first.close() } }
  }

  @Test
  fun globalCostsInterleaveBinaryAlternativesAndChildProducts() = gpuTest {
    val chart = Chart("""
      START -> A B [4/5]
      START -> C D [1/5]
      A -> a [1/10]
      A -> aa [9/10]
      B -> b [1/10]
      B -> bb [9/10]
      C -> c [1/1]
      D -> d [1/1]
    """)
    try {
      val expected = mapOf(
        "aa bb" to cost(0.8) + 2 * cost(0.9),
        "c d" to cost(0.2),
        "a bb" to cost(0.8) + cost(0.1) + cost(0.9),
        "aa b" to cost(0.8) + cost(0.9) + cost(0.1),
        "a b" to cost(0.8) + 2 * cost(0.1)
      )
      repeat(3) { seed ->
        val actual = chart.sampleCosts(seed, expected.size)
        assertEquals(expected, actual)
        assertEquals(expected.values.sorted(), actual.values.toList())
        assertEquals(listOf("aa bb", "c d"), chart.sample(seed, 2))
      }
    } finally { chart.close() }
  }

  @Test
  fun histogramBinsPreserveAllTiedAlternativeAndProductProvenance() = gpuTest {
    val chart = Chart("""
      START -> A B [1/2]
      START -> C D [1/2]
      A -> a0 [1/4]
      A -> a1 [1/4]
      A -> a2 [1/4]
      A -> a3 [1/4]
      B -> b0 [1/2]
      B -> b1 [1/2]
      C -> c0 [1/4]
      C -> c1 [1/4]
      C -> c2 [1/4]
      C -> c3 [1/4]
      D -> d0 [1/2]
      D -> d1 [1/2]
    """)
    try {
      val expected = (0..3).flatMap { left -> (0..1).flatMap { right ->
        listOf("a$left b$right", "c$left d$right")
      } }.toSet()
      val words = chart.sample(42, expected.size)
      assertEquals(expected, words.toSet())
      assertEquals(16, words.size, "Ten bins must not impose a limit of ten derivations")
      assertEquals(setOf(2 * cost(0.5) + cost(0.25)), chart.sampleCosts(42, expected.size).values.toSet())
      assertEquals(words, chart.sample(42, expected.size))
    } finally { chart.close() }
  }

  @Test
  fun histogramContinuesBeyondTenDistinctCostsIncludingTies() = gpuTest {
    val grammar = buildString {
      for (power in 0..11) appendLine("START -> token$power [${1 shl power}/4099]")
      appendLine("START -> cutoff_tie [4/4099]")
    }
    val chart = Chart(grammar)
    try {
      val expected = (0..11).map { "token$it" }.toSet() + "cutoff_tie"
      val words = chart.sample(42, 13)
      assertEquals(13, words.size, "Histogram capacity must grow until the requested words are indexed")
      assertEquals(expected, words.toSet())
      val costs = chart.sampleCosts(42, 13).values.toList()
      assertEquals(12, costs.toSet().size)
      assertEquals(costs.sorted(), costs)
      assertEquals((11 downTo 3).map { "token$it" }, words.take(9))
      assertEquals(setOf("token2", "cutoff_tie"), words.subList(9, 11).toSet())
      assertEquals(listOf("token1", "token0"), words.takeLast(2))
      for (requested in 10..12) assertEquals(words.take(requested), chart.sample(42, requested))
      assertEquals(words, chart.sample(42, 20), "An exhausted finite language must terminate below the request")
    } finally { chart.close() }
  }

  @Test
  fun multipleRootsAreMergedByGlobalCost() = gpuTest {
    val chart = Chart("""
      START -> A B [1/1]
      A -> a [1/10]
      A -> aa [9/10]
      B -> b [4/5]
      B -> bb [1/5]
    """, rootNames = listOf("A", "B"))
    try {
      repeat(3) { seed ->
        assertEquals(listOf("aa", "b", "bb", "a"), chart.sample(seed, 4))
        assertEquals(listOf("aa", "b"), chart.sample(seed, 2))
        assertEquals(listOf(cost(0.9), cost(0.8), cost(0.2), cost(0.1)), chart.sampleCosts(seed, 4).values.toList())
      }
    } finally { chart.close() }
  }

  @Test
  fun rootHistogramContinuesInGlobalOrderBeyondTheInitialBins() = gpuTest {
    val grammar = buildString {
      appendLine("START -> A B [1/1]")
      for (power in 0..11) appendLine("A -> token$power [${1 shl power}/4095]")
      appendLine("B -> expensive [1/1000000]")
    }
    val chart = Chart(grammar, rootNames = listOf("A", "B"))
    try {
      val expected = (11 downTo 0).map { "token$it" } + "expensive"
      assertEquals(expected, chart.sample(42, 13))
      assertEquals(expected.take(11), chart.sample(42, 11), "The eleventh low-cost word precedes the expensive root")
    } finally { chart.close() }
  }

  @Test
  fun histogramGrowthIncludesChildCostsBeyondTheInitialTenBins() = gpuTest {
    val grammar = buildString {
      appendLine("START -> A B [1/1]")
      for (power in 0..11) appendLine("A -> token$power [${1 shl power}/4095]")
      appendLine("B -> b [3/5]")
      appendLine("B -> bb [2/5]")
    }
    val chart = Chart(grammar)
    try {
      val expected = (0..11).flatMap { power ->
        val leftCost = cost((1 shl power).toDouble() / 4095)
        listOf("token$power b" to leftCost + cost(0.6), "token$power bb" to leftCost + cost(0.4))
      }.sortedBy { it.second }
      assertEquals(expected.map { it.first }, chart.sample(42, 24))
      assertEquals(expected.toMap(), chart.sampleCosts(42, 24))
      assertEquals(expected.take(21).map { it.first }, chart.sample(42, 21))
      assertEquals(expected.take(9).map { it.first }, chart.sample(42, 9))
    } finally { chart.close() }
  }

  @Test
  fun sharedLogarithmicTiersPreserveExactCostsAndProductRankOffsets() = gpuTest {
    val previous = pcfgHistogramBucketCount
    pcfgHistogramBucketCount = 3
    val numerators = listOf(199960, 199980, 200000, 200020, 200040)
    val grammar = buildString {
      appendLine("START -> A B [1/1]")
      numerators.forEachIndexed { index, numerator ->
        appendLine("A -> a$index [$numerator/1000000]")
        appendLine("B -> b$index [$numerator/1000000]")
      }
    }
    val chart = Chart(grammar, inspectIndex = true)
    try {
      // Each child has five consecutive costs, which must share a three-bucket budget.
      val childCosts = numerators.map { cost(it.toDouble() / 1_000_000) }
      assertEquals(4, childCosts.max() - childCosts.min())
      val expected = childCosts.indices.flatMap { left -> childCosts.indices.map { right ->
        "a$left b$right" to childCosts[left] + childCosts[right]
      } }.toMap()
      val words = chart.sample(42, 25)
      val actual = chart.sampleCosts(42, 25)
      assertEquals(expected.keys, words.toSet())
      assertEquals(25, words.size)
      assertEquals(expected, actual)
      assertEquals(expected.values.sorted(), actual.values.toList())
      assertTrue(chart.lastTierCount < chart.lastSliceCount, "Different exact costs must share logarithmic tiers")
      assertTrue(chart.lastHasTierOffset, "Shared tiers must retain nonzero exact-slice rank offsets")
      // These prefixes end inside cost slices and require offsets within their shared tiers.
      for (requested in listOf(3, 9, 17, 24))
        assertEquals(words.take(requested), chart.sample(42, requested))
    } finally {
      pcfgHistogramBucketCount = previous
      chart.close()
    }
  }

  @Test
  fun intersectionPipelineRanksAcrossEditRootsAndPreservesAnnotations() = gpuTest {
    val pcfg = """
      START -> A B [1/1]
      A -> a [1/1]
      B -> b [1/10]
      B -> bb [9/10]
    """.trimIndent()
    val cfg = pcfg.lines().joinToString("\n") { it.substringBeforeLast(" [") }.parseCNF()
    cfg.loadPCFG(pcfg)
    val input = listOf("a", "b")
    val results = intersectionPipeline(
      cfg, makeLevFSA(input, MAX_LEV_RAD), ledBuffer = 1,
      codePoints = input.map { cfg.tmMap.getValue(it) }.toIntArray()
    )
    assertEquals(listOf("a bb", "a b"), results)
    assertEquals(listOf(cost(0.9).toUInt(), cost(0.1).toUInt()), results.indices.map { results.scoreAt(it) })
    assertEquals(listOf(1, 0), results.indices.map { results.editDistanceAt(it) })
    assertEquals(listOf(LEV_EDIT_MATCH, LEV_EDIT_SUBSTITUTE), results.editScriptAt(0))
    assertEquals(listOf(LEV_EDIT_MATCH, LEV_EDIT_MATCH), results.editScriptAt(1))
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
        if (weighted) {
          val costs = chart.sampleCosts(seed = 71, count = 512).values.toList()
          assertEquals(costs.sorted(), costs, "Saturating bucket counts must preserve score order")
          assertEquals(List(64) { "a" }.joinToString(" "), words.first())
        }
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
  fun zeroProbabilityParentStillEnumeratesChildrenBeyondTenCostBins() = gpuTest {
    val grammar = buildString {
      appendLine("START -> A B [0/1]")
      for (power in 0..12) appendLine("A -> token$power [${1 shl power}/8191]")
      appendLine("B -> b [1/1]")
    }
    val chart = Chart(grammar)
    try {
      val expected = (0..12).map { "token$it b" }.toSet()
      val words = chart.sample(42, 13)
      assertEquals(13, words.size)
      assertEquals(expected, words.toSet())
      assertEquals(setOf(-1), chart.sampleCosts(42, 13).values.toSet())
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
      assertEquals(listOf("common", "rare", "zero"), chart.sample(42, 3))
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

  /** One active cell per NT, ordered by DAG depth; raw packets expose duplicate derivations. */
  private class Chart(
    pcfg: String, excluded: Pair<String, List<String>>? = null, weighted: Boolean = true,
    rootNames: List<String> = listOf(START_SYMBOL), private val inspectIndex: Boolean = false
  ) {
    private val rules = pcfg.trimIndent().trim().lines()
    private val cfg = rules.joinToString("\n") { it.substringBeforeLast(" [") }.parseCNF()
    private val nts = cfg.nonterminals.toList()
    private val roots = rootNames.map { cfg.bindex[it] }
    private val literals = nts.map { nt -> cfg.count { it.first == nt && it.second.size == 1 } }
    private val branches = nts.map { nt -> cfg.filter { it.first == nt && it.second.size == 2 && it != excluded }
      .map { cfg.bindex[it.second[0]] to cfg.bindex[it.second[1]] } }
    private val sizes = mutableMapOf<Int, Long>()
    private val buffers = mutableListOf<GPUBuffer>()
    val total = roots.fold(0L) { sum, root -> (sum + size(root)).coerceAtMost(0xffffffffL) }
    var lastTierCount = 0
      private set
    var lastSliceCount = 0
      private set
    var lastHasTierOffset = false
      private set

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

    private val spans = mutableMapOf<Int, Int>()
    private fun span(nt: Int): Int = spans.getOrPut(nt) {
      1 + (branches[nt].maxOfOrNull { (left, right) -> maxOf(span(left), span(right)) } ?: 0)
    }
    private val numStates = nts.indices.maxOf(::span) + 1
    private val cells = nts.indices.map { span(it) * nts.size + it }
    private val cellNts = IntArray(numStates * numStates * nts.size) { -1 }.apply {
      cells.forEachIndexed { nt, cell -> this[cell] = nt }
    }
    private val cellBranches = cellNts.map { nt -> if (nt < 0) emptyList() else branches[nt] }
    private val dp = buffer(cellNts.map { nt ->
      when { nt < 0 -> 0; literals[nt] == 0 -> 1; else -> 0x47fffffe }
    })
    private val counts = buffer(cellBranches.map { it.size })
    private val offsets = buffer(cellBranches.scan(0) { offset, row -> offset + row.size }.dropLast(1))
    private val storage = buffer(cellBranches.flatten().flatMap { listOf(cells[it.first], cells[it.second]) })
    private val cdf = buffer(cellBranches.flatMapIndexed { cell, row ->
      val nt = cellNts[cell]
      row.scan(if (nt < 0) 0L else literals[nt].toLong()) { sum, (left, right) ->
        (sum + product(left, right)).coerceAtMost(0xffffffffL)
      }.drop(1).map { it.toInt() }
    })
    private val rootSizes = buffer(roots.map { size(it).toInt() })
    private val rootCdf = buffer(roots.scan(0L) { sum, root ->
      (sum + size(root)).coerceAtMost(0xffffffffL)
    }.dropLast(1).map { it.toInt() })

    init {
      if (weighted) cfg.loadPCFG(pcfg.trimIndent().trim())
      else assertNull(cfg.pcfgBuf, "An unloaded grammar must not allocate a uniform PCFG")
    }
    suspend fun sample(seed: Int, count: Int): List<String> = sampleWordsAndCosts(seed, count).map { it.first }

    suspend fun sampleCosts(seed: Int, count: Int): Map<String, Int> = sampleWordsAndCosts(seed, count).toMap()

    private suspend fun sampleWordsAndCosts(seed: Int, count: Int): List<Pair<String, Int>> {
      val stride = MAX_WORD_LEN + PKT_HDR_LEN + 1
      val toDecode = minOf(count.toLong(), total).toInt()
      val indices = (listOf(seed, stride, nts.size, numStates, toDecode, toDecode, 0, roots.size * 2) +
        roots.flatMap { listOf(cells[it], 0) }).toGPUBuffer(STCPSD)
      val output = GPUBuffer(count * stride * 4, STCPSD)
      val sampling = cfg.pcfgBuf?.let {
        buildPCFGDecodeIndex(dp, counts, offsets, storage, cfg.termBuf, indices, it)
      }
      try {
        if (inspectIndex && sampling != null) {
          val header = sampling.readIndices((0..4).toList())
          lastTierCount = (header[3] - header[2]) / 2
          lastSliceCount = ((sampling.size / 4).toInt() - header[4]) / 3
          val offsets = sampling.readIndices((0 until lastSliceCount).map { header[4] + 3 * it + 1 })
          lastHasTierOffset = (0 until lastSliceCount).any { offsets[it] != 0 }
        }
        val retained = sampling?.readIndices(listOf(0))?.get(0)?.toUInt()?.toLong() ?: total
        assertTrue(retained >= toDecode, "Histogram indexed only $retained of $toDecode requested derivations")
        if (toDecode > 0)
          enum_words_wor(dp, counts, offsets, storage, cdf, cfg.termBuf, indices, rootSizes, sampling ?: rootCdf, output)(toDecode)
        val packets = output.readJSIntArray()
        return List(toDecode) { row ->
          (PKT_HDR_LEN until stride).asSequence().map { packets[row * stride + it] }
            .takeWhile { it != 0 }.joinToString(" ") { cfg.tmLst[(it and PACKED_TOKEN_MASK) - 1] } to packets[row * stride + 1]
        }
      } finally { indices.destroy(); output.destroy(); sampling?.destroy() }
    }

    fun close() { buffers.forEach { it.destroy() } }
  }
}
