import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.PyCodeSnippet
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

      val annotated = results.annotatedResults[index]
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
    assertEquals(expectedScript, results.editScript[index])
    assertEquals(1, results.editDistanceAt(index))
    assertEquals(
      "( ( p &amp; q ) &amp; p ) = <ins>(</ins> p &amp; ( q &amp; p ) )",
      results.annotatedResults[index]
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
    assertEquals("1 + 1 = 2", completions.annotatedResults[completionIndex])
    assertTrue(completions.editScript[completionIndex].all { it == LEV_EDIT_MATCH })
    assertEquals(0, completions.editDistanceAt(completionIndex))
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
