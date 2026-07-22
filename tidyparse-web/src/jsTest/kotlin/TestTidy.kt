import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.parsing.parseCFG
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
  fun testCachedTerminalBufferSurvivesEmptyIntersection() = browserTest {
    tryBootstrappingGPU()
    if (!gpuAvailable) return@browserTest

    val smallCfg = """
      START -> EQ
      EQ -> 1 + 1 = 2
    """.trimIndent().parseCFG()

    assertTrue(completeCode(smallCfg, listOf("_", "=", "=")).isEmpty())
    assertTrue(
      "1 + 1 = 2" in completeCode(smallCfg, listOf("_", "+", "_", "=", "_")),
      "An empty intersection should not destroy the CFG's cached terminal buffer"
    )
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
