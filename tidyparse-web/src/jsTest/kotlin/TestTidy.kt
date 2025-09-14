import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.PlatformVars
import ai.hypergraph.tidyparse.sampleGREUntilTimeout
import kotlinx.browser.window
import kotlinx.coroutines.test.runTest
import kotlin.test.*
import kotlin.time.Duration.Companion.minutes
import kotlin.time.TimeSource

class TestTidy {
  @BeforeTest
  fun before() {
    if (window.navigator.userAgent.indexOf("hrome") != -1)
      PlatformVars.PLATFORM_CALLER_STACKTRACE_DEPTH = 4
    DEBUG_SUFFIX = "\n"
  }

  val cfg by lazy { vanillaS2PCFG }
  val repairs by lazy {
    PYTHON_SNIPPETS.lines().windowed(4)
      .map { it.map { "$it NEWLINE".tokenizeByWhitespace() } }
      .map { it[0] to it[1] }.take(50)
  }

  @Test
  fun testRepairCodeGPU() = runTest(timeout = 10.minutes) {
    tryBootstrappingGPU()
    benchmarkRepair("GPU") { repairCode(cfg, code = it, LED_BUFFER, null) }
  }

  @Test
  fun testRepairCodeCPU() = runTest(timeout = 10.minutes) {
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
    log("Total $name repairs: $totalResults")
    log("Total $name matches: $totalMatches")
  }
}