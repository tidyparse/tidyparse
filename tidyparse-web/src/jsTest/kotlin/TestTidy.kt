import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.PlatformVars
import kotlinx.browser.window
import kotlinx.coroutines.test.runTest
import web.gpu.GPUBuffer
import kotlin.test.*
import kotlin.time.TimeSource

class TestTidy {
  @BeforeTest
  fun before() {
    if (window.navigator.userAgent.indexOf("hrome") != -1)
      PlatformVars.PLATFORM_CALLER_STACKTRACE_DEPTH = 4
  }

  @Test
  fun thingsShouldWork() {
    assertEquals(listOf(1,2,3).reversed(), listOf(3,2,1))
  }

  @Test
  fun thingsShouldFail() {
    assertEquals(listOf(1,2,3).reversed(), listOf(2,2,1))
  }

  @Test
  fun testRepairCode() = runTest {
    println("Test WGPU repair")
    tryBootstrappingGPU()
    val cfg = vanillaS2PCFG

    val content = """
      NAME NAME . NAME ( [ NUMBER , NUMBER , NUMBER , NUMBER , NUMBER ] ) [ NUMBER : NUMBER : NUMBER ]
      NAME [ NUMBER , NUMBER , NUMBER , NUMBER , NUMBER ] [ NUMBER : NUMBER : NUMBER ]
      cHJpbnQgbnAuYXJyYXkoWzAsIDEsIDIsIDMsIDRdKVswOjE6MV0=
      cHJpbnQgWzAsIDEsIDIsIDMsIDRdWzA6MToxXQ==
      NAME NAME . NAME ( [ NUMBER , NUMBER , NUMBER , NUMBER , NUMBER ] ) [ NUMBER : NUMBER : NUMBER ]
      NAME [ NUMBER , NUMBER , NUMBER , NUMBER , NUMBER ] [ NUMBER : NUMBER : NUMBER ]
      cHJpbnQgbnAuYXJyYXkoWzAsIDEsIDIsIDMsIDRdKVswOjA6MV0=
      cHJpbnQgWzAsIDEsIDIsIDMsIDRdWzA6MDoxXQ==
      NAME NEWLINE INDENT { STRING : [ NUMBER ] , STRING : [ NUMBER ] , STRING : [ NUMBER ] } NEWLINE DEDENT
      { STRING : [ NUMBER ] , STRING : [ NUMBER ] , STRING : [ NUMBER ] }
      c2V0cwogICAgeyd0aHJlZSc6IFsxXSwndHdvJzogWzFdLCAnb25lJzogWzFdfQ==
      eyd0aHJlZSc6IFsxXSwndHdvJzogWzFdLCAnb25lJzogWzFdfQ==
      NAME NEWLINE INDENT { STRING : [ ] , STRING : [ ] , STRING : [ ] , STRING : [ ] , STRING : [ NUMBER ] } NEWLINE DEDENT
      { STRING : [ ] , STRING : [ ] , STRING : [ ] , STRING : [ ] , STRING : [ NUMBER ] }
      c2V0cwogICAgeydmb3VyJzogW10sICd0aHJlZSc6IFtdLCAnZml2ZSc6IFtdLCAndHdvJzogW10sICdvbmUnOiBbMV19
      eydmb3VyJzogW10sICd0aHJlZSc6IFtdLCAnZml2ZSc6IFtdLCAndHdvJzogW10sICdvbmUnOiBbMV19
    """.trimIndent()

    val lines = content.lines().windowed(4).map { it.first() }
    println("Lines count: ${lines.size}")

    lines.forEachIndexed { index, line ->
      val t0 = TimeSource.Monotonic.markNow()
      val results = repairCode(cfg, code="$line NEWLINE".tokenizeByWhitespace(), LED_BUFFER, null)
      val elapsed = t0.elapsedNow()

      // Basic assertions
      assertTrue(results.isNotEmpty(), "No repairs generated for '$line'")

      // Print for inspection/perf
      println("Generated ${results.size} repairs in $elapsed")
      println("Sample repairs: ${results.take(5).joinToString("\n")}")
    }
  }
}