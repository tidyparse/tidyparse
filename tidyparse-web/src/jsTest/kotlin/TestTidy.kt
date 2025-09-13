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
    PYTHON_SNIPPETS.lines().windowed(4).map { it.first() }.take(50)
      .map { "$it NEWLINE".tokenizeByWhitespace() }
  }

  @Test
  fun testRepairCodeGPU() = runTest(timeout = 10.minutes) {
    log("Testing WGPU repairs...")
    tryBootstrappingGPU()

    val startTime = TimeSource.Monotonic.markNow()
    var totalRepairs = 0

    repairs.forEach { line ->
      val t0 = TimeSource.Monotonic.markNow()
      val results = listOf("Sample repairs:") + repairCode(cfg, code = line, LED_BUFFER, null)
      val elapsed = t0.elapsedNow()

      assertTrue(results.isNotEmpty(), "No repairs generated for:\n '${line.joinToString(" ")}'")

      val numRepairs = results.size.also { totalRepairs += it }
      log("Generated $numRepairs repairs in $elapsed")
      log(results.take(5).joinToString("\n\t\t\t"))
    }

    log("Total GPU time: ${startTime.elapsedNow().inWholeMilliseconds}ms")
    log("Total GPU repairs: $totalRepairs")

  }

  @Test
  fun testRepairCPU() = runTest(timeout = 10.minutes) {
    log("Testing CPU repairs...")

    val startTime = TimeSource.Monotonic.markNow()
    var totalRepairs = 0

    repairs.forEach { line ->
      val t0 = TimeSource.Monotonic.markNow()
      val results = listOf("Sample repairs:") + sampleGREUntilTimeout(line, cfg).distinct().toList()
      val elapsed = t0.elapsedNow()

      assertTrue(results.isNotEmpty(), "No repairs generated for:\n '${line.joinToString(" ")}'")

      val numRepairs = results.size.also { totalRepairs += it }
      log("Generated $numRepairs repairs in $elapsed")
      log(results.take(5).joinToString("\n\t\t\t"))
    }

    log("Total CPU time: ${startTime.elapsedNow().inWholeMilliseconds}ms")
    log("Total CPU repairs: $totalRepairs")
  }
}