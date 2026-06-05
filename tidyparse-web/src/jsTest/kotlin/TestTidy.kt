import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.sampleGREUntilTimeout
import kotlinx.coroutines.test.runTest
import kotlin.test.*
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

/*
./gradlew jsTest
or
./gradlew replotMetrics
 */
class TestTidy {
  companion object {
    private const val REPAIR_FIXTURE_COUNT = 12
    private const val GPU_FIXTURE_COUNT = 3
    private const val TEST_REPAIR_TIMEOUT_MS = 500
    private const val TEST_GPU_MAX_SAMPLES = 512
    private const val TEST_CPU_MAX_RESULTS = 64
  }

  @BeforeTest
  fun before() {
    DEBUG_SUFFIX = "\n"
    TIMEOUT_MS = TEST_REPAIR_TIMEOUT_MS
    LED_BUFFER = 0
  }

  val cfg by lazy { vanillaS2PCFG }
  val repairs by lazy {
    PYTHON_SNIPPETS.lineSequence()
      .filter { it.isNotBlank() }
      .chunked(4)
      .take(REPAIR_FIXTURE_COUNT)
      .map { (broken, fixed) -> "$broken NEWLINE".tokenizeByWhitespace() to "$fixed NEWLINE".tokenizeByWhitespace() }
      .toList()
  }

  @Test
  fun testRepairCodeGPU() = runTest(timeout = 60.seconds) {
    tryBootstrappingGPU()
    if (!gpuAvailable) {
      log("Skipping GPU repair test: WebGPU is not available.")
      return@runTest
    }

    benchmarkRepair("GPU", repairs.take(GPU_FIXTURE_COUNT)) {
      repairCode(cfg, code = it, ledBuffer = LED_BUFFER, ngrams = null, maxUniformSamples = TEST_GPU_MAX_SAMPLES)
    }
  }

  @Test
  fun testRepairCodeCPU() = runTest(timeout = 60.seconds) {
    benchmarkRepair("CPU") { sampleGREUntilTimeout(it, cfg).take(TEST_CPU_MAX_RESULTS).distinct().toList() }
  }

  suspend fun benchmarkRepair(
    name: String,
    fixtures: List<Pair<List<String>, List<String>>> = repairs,
    repair: suspend (List<String>) -> List<String>
  ) {
    log("Testing $name repairs...")

    val startTime = TimeSource.Monotonic.markNow()
    var totalResults = 0; var totalRepairs = 0; var totalMatches = 0

    fixtures.forEach { (line, fixed) ->
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
    log("Total $name repairs: $totalResults")
    log("Total $name matches: $totalMatches")
  }

  // TODO: test parity between GPU- and CPU- versions
  // TODO: implement and test GPU-based hole completion
  // TODO: allow hole completion w/ Brozozowski decoding
}