import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.PyCodeSnippet
import ai.hypergraph.tidyparse.MAX_DISP_RESULTS
import ai.hypergraph.tidyparse.sampleGREUntilTimeout
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlinx.coroutines.withTimeout
import kotlin.test.*
import kotlin.time.Duration.Companion.minutes
import kotlin.time.TimeSource

/*
./gradlew jsTest
or
./gradlew replotMetrics
 */
class TestTidy {
  @BeforeTest
  fun before() { DEBUG_SUFFIX = "\n" }

  private val benchmarkTimeout = 30.minutes
  private fun browserTest(block: suspend () -> Unit) =
    MainScope().promise { withTimeout(benchmarkTimeout) { block() } }

  val cfg by lazy { vanillaS2PCFG }
  val pythonCfg by lazy { pythonStatementCNFAllProds }
  val TO_TEST = 50

  private val twoVariablesCompletionCfg by lazy {
    """
      START -> E& = E& | EP = EP | EQ = EQ | EV = EV

      EP -> p
      EP -> ( EP & EP ) | ( EP & EV ) | ( EV & EP )
      EP -> ( EP v EP ) | ( EP v E& ) | ( E& v EP )
      EQ -> q
      EQ -> ( EQ & EQ ) | ( EQ & EV ) | ( EV & EQ )
      EQ -> ( EQ v EQ ) | ( EQ v E& ) | ( E& v EQ )

      E& -> ( E& & E& ) | ( E& & EP ) | ( E& & EQ ) | ( E& & EV ) | ( EP & E& )
      E& -> ( EQ & E& ) | ( EV & E& ) | ( EP & EQ ) | ( EQ & EP ) | ( E& v E& )
      EV -> ( EV & EV ) | ( EV v EV ) | ( EV v EP ) | ( EV v EQ ) | ( EV v E& )
      EV -> ( EP v EV ) | ( EQ v EV ) | ( E& v EV ) | ( EP v EQ ) | ( EQ v EP )
    """.trimIndent().parseCFG(validate = true).noEpsilon.visibleCompletionCFG
  }

  private val twoVariablesPrefix =
    "( ( p & q ) & p ) = ( p & ( q &".tokenizeByWhitespace()

  val snippets by lazy {
    PYTHON_SNIPPETS.trim('\n', '\r').lines().chunked(4).also {
      require(it.all { snippet -> snippet.size == 4 }) {
        "PYTHON_SNIPPETS must contain complete 4-line records"
      }
    }
  }

  val repairs by lazy {
    snippets.asSequence().map { it.map { "$it NEWLINE".tokenizeByWhitespace() } }
      .map { it[0] to it[1] }.take(TO_TEST)
  }

  val rawPythonRepairs by lazy {
    snippets.asSequence().map { decodeBase64(it[2]) to decodeBase64(it[3]) }
      .filter { "\n" !in it.first && "\n" !in it.second }
      .take(TO_TEST).toList()
  }

  @Test
  fun testRepairCodeGPU() = browserTest {
    tryBootstrappingGPU()
    benchmarkRepair("GPU") { repairCode(cfg, code = it, LED_BUFFER) }
  }

  @Test
  fun testAnnotatedRepairAlignment() = browserTest {
    tryBootstrappingGPU()
    assertTrue(gpuAvailable, "WebGPU is required for the alignment test")

    suspend fun assertRepair(
      source: List<String>,
      grammar: String,
      expected: String,
      expectedAnnotated: String
    ) {
      val repairCfg = grammar.trimIndent().parseCFG()
      val results = repairCode(repairCfg, source, ledBuffer = 0)
      val index = results.indexOf(expected)
      assertTrue(index >= 0, "Missing '$expected' among plain repairs")

      val annotated = results.htmlAt(index)
      assertEquals(expectedAnnotated, annotated)
      assertEquals(
        levAndLenMetric(source)(expected.tokenizeByWhitespace()),
        results.editDistanceAt(index) * 7919 +
            kotlin.math.abs(source.sumOf { it.length } - expected.replace(" ", "").length),
        annotated
      )
    }

    assertRepair(listOf("a"), """
      START -> A B
      A -> a
      B -> b
    """, "a b", "a <ins>b</ins>")
    assertRepair(listOf("a"), """
      START -> B | A A
      A -> a
      B -> b
    """, "b", "<sub>b</sub>")
    assertRepair(listOf("a", "b", "c", "d"), """
      START -> A D | B B B B | C C C C
      A -> a
      B -> b
      C -> c
      D -> d
    """, "a d", "a <del></del> <del></del> d")
    assertRepair(listOf("a", "b", "c"), """
      START -> A | B B | C C
      A -> a
      B -> b
      C -> c
    """, "a", "a <del></del> <del></del>")

    val defaultWindowCfg = """
      START -> B | A A
      A -> a
      B -> b
    """.trimIndent().parseCFG()
    assertTrue(
      "b" in repairCode(defaultWindowCfg, listOf("a")),
      "The default edit-distance window must not overflow when LED is positive"
    )
  }

  @Test
  fun testBooleanRepairUsesCanonicalAlignment() = browserTest {
    tryBootstrappingGPU()
    assertTrue(gpuAvailable, "WebGPU is required for the alignment test")

    val source = "( ( p & q ) & p ) = p & ( q & p ) )".tokenizeByWhitespace()
    val expectedTokens = "( ( p & q ) & p ) = ( p & ( q & p ) )".tokenizeByWhitespace()
    val expected = expectedTokens.joinToString(" ")
    val repairCfg = """
      START -> E& = E& | EP = EP | EQ = EQ | EV = EV

      EP -> p
      EP -> ( EP & EP ) | ( EP & EV ) | ( EV & EP )
      EP -> ( EP v EP ) | ( EP v E& ) | ( E& v EP )
      EQ -> q
      EQ -> ( EQ & EQ ) | ( EQ & EV ) | ( EV & EQ )
      EQ -> ( EQ v EQ ) | ( EQ v E& ) | ( E& v EQ )

      E& -> ( E& & E& ) | ( E& & EP ) | ( E& & EQ ) | ( E& & EV ) | ( EP & E& )
      E& -> ( EQ & E& ) | ( EV & E& ) | ( EP & EQ ) | ( EQ & EP ) | ( E& v E& )
      EV -> ( EV & EV ) | ( EV v EV ) | ( EV v EP ) | ( EV v EQ ) | ( EV v E& )
      EV -> ( EP v EV ) | ( EQ v EV ) | ( E& v EV ) | ( EP v EQ ) | ( EQ v EP )
    """.trimIndent().parseCFG()

    val results = repairCode(repairCfg, source, ledBuffer = 2)
    val indices = results.indices.filter { results[it] == expected }
    assertEquals(1, indices.size, "Expected one canonical row per visible repair")
    val index = indices.single()

    val expectedScript = levenshteinAlign(source, expectedTokens).map { (old, new) ->
      when {
        old == null -> LEV_EDIT_INSERT
        new == null -> LEV_EDIT_DELETE
        old == new -> LEV_EDIT_MATCH
        else -> LEV_EDIT_SUBSTITUTE
      }
    }
    assertEquals(expectedScript, results.editScriptAt(index))
    assertEquals(1, results.editDistanceAt(index))
    assertEquals(
      "( ( p &amp; q ) &amp; p ) = <ins>(</ins> p &amp; ( q &amp; p ) )",
      results.htmlAt(index)
    )

    val originalLength = source.sumOf(String::length)
    val first = results.indices.minBy { candidate ->
      results.editDistanceAt(candidate) * 7919 +
        kotlin.math.abs(originalLength - results[candidate].count { !it.isWhitespace() })
    }
    assertEquals(expected, results[first])
  }

  @Test
  fun testCachedTerminalBufferSurvivesEmptyIntersection() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    val smallCfg = """
      START -> EQ
      EQ -> 1 + 1 = 2
    """.trimIndent().parseCFG()

    assertTrue(completeCode(smallCfg, listOf("_", "=", "=")).isEmpty())
    val completions = completeCode(smallCfg, listOf("_", "+", "_", "=", "_"))
    val completionIndex = completions.indexOf("1 + 1 = 2")
    assertTrue(
      completionIndex >= 0,
      "An empty intersection should not destroy the CFG's cached terminal buffer"
    )
    assertEquals("1 + 1 = 2", completions.htmlAt(completionIndex))
    assertTrue(completions.editScriptAt(completionIndex).all { it == LEV_EDIT_MATCH })
    assertEquals(0, completions.editDistanceAt(completionIndex))
  }

  @Test
  fun testGpuSuffixBatchBalancesNextTokensAndFillsFromRemainingWords() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    val suffixCfg = buildString {
      appendLine("P -> p")
      appendLine("A -> a")
      appendLine("B -> b")
      repeat(5) { index ->
        appendLine("START -> P A$index")
        appendLine("A$index -> A AX$index")
        appendLine("AX$index -> ax$index")
      }
      repeat(2) { index ->
        appendLine("START -> P B$index")
        appendLine("B$index -> B BT$index")
        appendLine("BT$index -> M BX$index")
        appendLine("BX$index -> bx$index")
      }
      appendLine("M -> mid")
    }.trimEnd().parseCNF()
    val batch = assertNotNull(suffixCfg.gpuSuffixBatch(listOf("p"), terminalCompletion = null, limit = 7))

    val words = assertNotNull(suffixCfg.gpuDiverseSuffixes(batch, limit = 7))

    assertEquals(7, words.size)
    assertEquals(listOf("a", "b", "a", "b"), words.take(4).map { it.tokenizeByWhitespace()[1] })
    assertEquals(5, words.count { it.startsWith("p a ") })
    assertEquals(2, words.count { it.startsWith("p b ") })
    assertTrue(words.all { it in suffixCfg.language })
  }

  @Test
  fun twoVariablesSuffixBatchStopsAfterThreeLengthsPerNextToken() {
    val batch = assertNotNull(twoVariablesCompletionCfg.gpuSuffixBatch(
      twoVariablesPrefix,
      terminalCompletion = null,
      limit = 35
    ))
    val lengthsByTerminal = batch.slices.groupBy({ it.terminal }, { it.length })

    assertEquals(16, batch.prefix.size)
    assertEquals(listOf(7, 11, 15), lengthsByTerminal["("])
    assertTrue(lengthsByTerminal.values.all { it.size <= 3 })
    assertEquals(15, batch.slices.maxOf { it.length })
    assertEquals(9, batch.slices.size)
    assertTrue(batch.prefix.size + batch.slices.maxOf { it.length } < MAX_WORD_LEN - 1)
  }

  @Test
  fun suffixBatchDropsSlicesAtTheGpuWordBoundary() {
    val prefix = List(124) { "p$it" }
    val terminalCompletion = TerminalCompletionPlan(
      originalPrefix = "a",
      expandedPrefix = "a",
      lexicalCandidateCount = 1,
      terminalCommitted = false,
      forcedContinuation = emptyList(),
      branches = listOf(
        TerminalCompletionBranch(
          terminal = "a",
          tokens = prefix + "a",
          suffixLengths = sequenceOf(0, 1, 2)
        )
      )
    )
    val batch = assertNotNull("START -> a".parseCFG().gpuSuffixBatch(
      tokens = prefix + "a",
      terminalCompletion = terminalCompletion,
      limit = 35
    ))

    assertEquals(listOf(SuffixSlice("a", 1), SuffixSlice("a", 2)), batch.slices)
    assertTrue(batch.slices.all { batch.prefix.size + it.length < MAX_WORD_LEN - 1 })

    val unsupported = terminalCompletion.copy(branches = listOf(
      TerminalCompletionBranch(
        terminal = "a",
        tokens = prefix + "a",
        suffixLengths = sequenceOf(2)
      )
    ))
    assertNull(
      "START -> a".parseCFG().gpuSuffixBatch(prefix + "a", unsupported, limit = 35),
      "A shortest continuation beyond the GPU bound must not be silently omitted"
    )
  }

  @Test
  fun testGpuTwoVariablesSuffixBatchFillsDisplayFromTheShortTemplate() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    val limit = 35
    val batch = assertNotNull(twoVariablesCompletionCfg.gpuSuffixBatch(
      twoVariablesPrefix,
      terminalCompletion = null,
      limit = limit
    ))
    val words = assertNotNull(
      twoVariablesCompletionCfg.gpuDiverseSuffixes(batch, limit),
      "The horizon-15 two_variables batch must remain on the GPU"
    )

    assertEquals(15, batch.slices.maxOf { it.length })
    assertEquals(limit, words.size)
    assertEquals(
      batch.slices.mapTo(linkedSetOf()) { it.terminal },
      words.mapTo(linkedSetOf()) { it.tokenizeByWhitespace()[twoVariablesPrefix.size] }
    )
    assertEquals(words.size, words.distinct().size)
    assertTrue(words.all { it in twoVariablesCompletionCfg.language })
  }

  @Test
  fun testGpuSuffixSamplingDenselyCombinesUnequalLengthRoots() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    val suffixCfg = """
      START -> P AS
      START -> P AL
      START -> P BS
      P -> p
      AS -> A X
      AL -> A T3
      BS -> B X
      T3 -> X T2
      T2 -> X X
      A -> a
      B -> b
      X -> x
      X -> y
    """.trimIndent().parseCNF()
    val prefix = listOf("p")
    val batch = SuffixBatch(
      prefix = prefix,
      slices = listOf(
        SuffixSlice("a", length = 2),
        SuffixSlice("a", length = 4),
        SuffixSlice("b", length = 2)
      )
    )

    val words = assertNotNull(suffixCfg.gpuDiverseSuffixes(batch, limit = 7))
    val tokenized = words.map { it.tokenizeByWhitespace() }

    assertEquals(7, words.size)
    assertEquals(5, tokenized.count { it[prefix.size] == "a" })
    assertEquals(2, tokenized.count { it[prefix.size] == "b" })
    assertEquals(setOf(2, 4), tokenized.filter { it[prefix.size] == "a" }.map { it.size - prefix.size }.toSet())
    assertEquals(words.size, words.distinct().size)
    assertTrue(words.all { it in suffixCfg.language })
  }

  @Test
  fun testGpuSuffixSamplingBalancesRareNextTokenGroupsWithoutFallback() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    // At suffix length 15, `while` has billions of parse-tree witnesses while
    // `for` and `if` have only three words apiece. Sampling the shared root
    // uniformly would almost certainly miss both rare next-token groups.
    val suffixCfg = buildString {
      appendLine("START -> P BODY")
      appendLine("P -> p")
      appendLine("BODY -> WHILE AMBIG")
      appendLine("BODY -> FOR RARE")
      appendLine("BODY -> IF RARE")
      appendLine("WHILE -> while")
      appendLine("FOR -> for")
      appendLine("IF -> if")
      appendLine("AMBIG -> AMBIG AMBIG")
      appendLine("AMBIG -> x")
      appendLine("AMBIG -> y")
      appendLine("RARE -> A T13")
      appendLine("RARE -> B T13")
      appendLine("RARE -> C T13")
      appendLine("A -> a")
      appendLine("B -> b")
      appendLine("C -> c")
      appendLine("X -> x")
      for (length in 13 downTo 3)
        appendLine("T$length -> X T${length - 1}")
      appendLine("T2 -> X X")
    }.trimEnd().parseCNF()
    val limit = 9
    val batch = SuffixBatch(
      prefix = listOf("p"),
      slices = listOf("while", "for", "if").map { SuffixSlice(it, length = 15) }
    )

    val words = assertNotNull(
      suffixCfg.gpuDiverseSuffixes(batch, limit),
      "Rare next-token groups must be sampled on the GPU without an incomplete-result fallback"
    )
    val nextTokens = words.map { it.tokenizeByWhitespace()[1] }

    assertEquals(limit, words.size)
    assertEquals(mapOf("while" to 3, "for" to 3, "if" to 3), nextTokens.groupingBy { it }.eachCount())
    assertEquals(words.size, words.distinct().size)
    assertTrue(words.all { it in suffixCfg.language })
  }

  @Test
  fun testGpuPlWhileSuffixSamplingCoversEveryPlannedNextTokenGroup() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    val completionCfg = """
      START -> STM+
      STM+ -> STM+ ; STM | STM
      STM -> ASGN | IFS | while ( BEXP ) { STM+ } | for ( ASGN ; BEXP ; ASGN ) { STM+ } | { STM+ } | { } | return EXP ; | break | continue | EXP | STM ;
      ASGN -> LHS = EXP | LHS += EXP | LHS -= EXP | LHS *= EXP | LHS /= EXP | LHS %= EXP
      LHS -> ID | ID . ID | ID [ EXP ] | ( LHS )
      IFS -> ID = if ( BEXP ) { EXP } else { EXP } | if ( BEXP ) { STM+ } else { STM+ } | if ( BEXP ) { STM+ }
      EXP -> ID | NUM | STR | LIT | EXP + EXP | EXP - EXP | EXP * EXP | EXP / EXP | EXP % EXP | ( EXP ) | ID ( ) | ID ( ARGS ) | EXP . ID | EXP [ EXP ] | BEXP
      ARGS -> EXP | EXP , ARGS
      BEXP -> EXP == EXP | EXP != EXP | EXP < EXP | BEXP && BEXP | BEXP or BEXP | ( BEXP ) | ! BEXP | LIT
      LIT -> true | false
    """.trimIndent().parseCFG(validate = true).noEpsilon.visibleCompletionCFG
    val prefix = "while ( ID == NUM ) { ID = ID + NUM ;".tokenizeByWhitespace()
    val limit = 35
    val batch = assertNotNull(completionCfg.gpuSuffixBatch(prefix, terminalCompletion = null, limit = limit))
    val expectedGroups = batch.slices.mapTo(linkedSetOf()) { it.terminal }

    assertEquals(23, expectedGroups.size)
    assertContains(batch.slices, SuffixSlice("while", 8))
    assertContains(batch.slices, SuffixSlice("if", 8))
    assertContains(batch.slices, SuffixSlice("for", 12))

    val words = assertNotNull(
      completionCfg.gpuDiverseSuffixes(batch, limit),
      "The GPU must cover all pl_while next-token groups without an incomplete-result fallback"
    )
    val counts = words
      .map { it.tokenizeByWhitespace()[prefix.size] }
      .groupingBy { it }.eachCount()

    assertEquals(limit, words.size)
    assertEquals(expectedGroups, counts.keys)
    assertTrue(
      counts.values.let { it.maxOrNull()!! - it.minOrNull()!! <= 1 },
      "Expected an even quota across pl_while next-token groups, received $counts"
    )
    assertEquals(words.size, words.distinct().size)
    assertTrue(words.all { it in completionCfg.language })
  }

  @Test
  fun testGpuPythonPartialTerminalSuffixBatch() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    val completionCfg = pythonCfg.visibleCompletionCFG
    val completion = assertNotNull(pythonCfg.terminalCompletionPlan(listOf("c")))
    val batch = assertNotNull(completionCfg.gpuSuffixBatch(listOf("c"), completion, limit = MAX_DISP_RESULTS))
    val started = TimeSource.Monotonic.markNow()
    var suffixTimings: Map<String, Int>? = null
    val words = assertNotNull(completionCfg.gpuDiverseSuffixes(
      batch, MAX_DISP_RESULTS, started
    ) { suffixTimings = it })

    log("Python `c` suffix batch produced ${words.size} rows in ${started.elapsedNow()}")
    assertTrue("preprocessing" in assertNotNull(suffixTimings))
    assertTrue("matrix closure" in assertNotNull(suffixTimings))
    assertTrue("decode" in assertNotNull(suffixTimings))
    assertEquals(MAX_DISP_RESULTS, words.size)
    assertEquals(setOf("class", "continue"), words.map { it.tokenizeByWhitespace().first() }.toSet())
    assertTrue(words.all { it in completionCfg.language })
  }

  @Test
  @OptIn(ExperimentalUnsignedTypes::class)
  fun testEndToEndRepairPipeline() = browserTest {
    tryBootstrappingGPU()
    val errHst = mutableMapOf<String, Int>()
    val filtered = JSTidyPyEditor.run {
      sequenceOf("x = 1", "x =").filterCompilerErrors(errHst, window = 2).toList()
    }

    assertEquals(listOf("x = 1"), filtered)
    assertTrue(errHst.isNotEmpty(), "Expected headless compiler filtering to reject invalid Python")

    benchmarkEndToEndRepairPipeline("end-to-end") { repairPythonLineRaw(pythonCfg, it) }
  }

  @Test
  fun testRepairCodeCPU() = browserTest {
    benchmarkRepair("CPU") { sampleGREUntilTimeout(it, cfg).distinct().toList() }
  }

  suspend fun benchmarkRepair(name: String, repair: suspend (List<String>) -> List<String>) {
    log("Testing $name repairs...")

    val startTime = TimeSource.Monotonic.markNow()
    var totalResults = 0; var totalRepairs = 0; var totalMatches = 0

    repairs.forEach { (line, fixed) ->
      totalRepairs++
      val t0 = TimeSource.Monotonic.markNow()
      val repairResults = repair(line)
      val results = listOf("Sample repairs:") + repairResults
      val elapsed = t0.elapsedNow()

      assertTrue(repairResults.isNotEmpty(), "No repairs generated for:\n '${line.joinToString(" ")}'")

      if (fixed in results.map { it.tokenizeByWhitespace() }) totalMatches++

      val numRepairs = repairResults.size.also { totalResults += it }
      log("Generated $numRepairs repairs in $elapsed")
      log(results.take(5).joinToString("\n\t\t\t"))
    }

    log("Total $name latency: ${startTime.elapsedNow().inWholeMilliseconds}")
    log("Total $name repairs: $totalResults\nTotal $name matches: $totalMatches")
  }

  suspend fun benchmarkEndToEndRepairPipeline(name: String, repair: suspend (String) -> String) {
    log("Testing $name Python repairs...")

    val startTime = TimeSource.Monotonic.markNow()
    var totalResults = 0; var totalRepairs = 0; var totalMatches = 0; var noResults = 0

    rawPythonRepairs.forEach { (line, fixed) ->
      totalRepairs++
      val t0 = TimeSource.Monotonic.markNow()
      val repairResults = repair(line).lines().filter { it.isNotBlank() }
      val elapsed = t0.elapsedNow()

      if (repairResults.isEmpty()) {
        noResults++
        log("No raw repairs generated in $elapsed for:\n\t\t\t$line")
        return@forEach
      }

      val fixedTokens = pythonTokens(fixed)
      if (repairResults.map { pythonTokens(it) }.contains(fixedTokens)) totalMatches++

      val numRepairs = repairResults.size.also { totalResults += it }
      log("Generated $numRepairs raw repairs in $elapsed")
      log((listOf("Sample raw repairs:") + repairResults).take(5).joinToString("\n\t\t\t"))
    }

    assertEquals(TO_TEST, totalRepairs)
    assertTrue(totalResults > 0, "Expected at least one raw repair result")

    log("Total $name raw latency: ${startTime.elapsedNow().inWholeMilliseconds}")
    log("Total $name raw repairs: $totalResults\nTotal $name raw matches: $totalMatches\nTotal $name raw no-results: $noResults")
  }

  private fun pythonTokens(code: String): List<String> =
    PyCodeSnippet(code).lexedTokens().tokenizeByWhitespace().map { if (it == "|") "OR" else it }

  private fun decodeBase64(s: String): String = js("atob")(s) as String

  // TODO: test parity between GPU- and CPU- versions
  // TODO: implement and test GPU-based hole completion
  // TODO: allow hole completion w/ Brozozowski decoding
}