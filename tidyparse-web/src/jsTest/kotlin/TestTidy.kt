import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.PlatformVars
import kotlinx.browser.window
import kotlinx.coroutines.test.runTest
import kotlin.test.*
import kotlin.time.TimeSource

class TestTidy {
  @BeforeTest
  fun before() {
    if (window.navigator.userAgent.indexOf("hrome") != -1)
      PlatformVars.PLATFORM_CALLER_STACKTRACE_DEPTH = 4
    DEBUG_SUFFIX = "\n"
  }

  @Test
  fun testRepairCode() = runTest {
    log("Test WGPU repair")
    tryBootstrappingGPU()
    val cfg = vanillaS2PCFG

    val startTime = TimeSource.Monotonic.markNow()
    val lines = PYTHON_SNIPPETS.lines().windowed(4).map { it.first() }.take(20)

    lines.forEachIndexed { index, line ->
      val t0 = TimeSource.Monotonic.markNow()
      val results = listOf("Sample repairs:") +
        repairCode(cfg, code="$line NEWLINE".tokenizeByWhitespace(), LED_BUFFER, null)
      val elapsed = t0.elapsedNow()

      assertTrue(results.isNotEmpty(), "No repairs generated for '$line'")

      log("Generated ${results.size} repairs in $elapsed")
      log(results.take(5).joinToString("\n\t\t\t"))
    }
    log("Total time: ${startTime.elapsedNow().inWholeMilliseconds}ms")
  }
}