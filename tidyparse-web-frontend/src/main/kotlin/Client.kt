import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.types.*
import kotlinx.browser.*
import org.w3c.dom.*
import kotlin.time.TimeSource

/**
TODO (soon):
 - Extract platform-independent code from IntelliJ plugin
 - Get Myers diffs working properly
 - Clean up the Gradle build wreckage
 - Render the Chomsky-normalized CFG
 - Syntax highlighting for the snippets
 - Rank results by more sensible metric
 - Provide assistance for grammar editing
 - Allow richer HTML content in RHS panel
 *//*
TODO (maybe):
 - Add demo for Python and Java
 - Add Ctrl+Space code completion popup
 - Configurable settings, e.g., timeout, max repairs
 - Auto-alignment of the productions
 - Calculate finger-travel distance
 - Collect telemetry for a user study
 - Improve support for incrementalization
 - Look into ropes, zippers and lenses
*/
val parser = Parser(
  "whitespace" to "\\s+",
//  "red"        to "->|\\|",
  "blue"       to "---",
  "gray"       to "->|_|\\|",
  "green"      to "START",
  "other"      to "\\S"
  // Uncomment or add more rules as needed
  // "orange" to "orange",
  // "yellow" to "yellow",
  // "indigo" to "indigo",
  // "violet" to "violet",
)

// ./gradlew browserDevelopmentRun --continuous
fun main() {
  if (window.navigator.userAgent.indexOf("hrome") != -1) {
    PlatformVars.PLATFORM_CALLER_STACKTRACE_DEPTH = 4
  }
  TIMEOUT_MS = 5_000
  jsEditor.getLatestCFG()
  window.onload = { jsEditor.redecorateLines() }
  inputField.addEventListener("input", { jsEditor.run { continuation { handleInput() } } })
  inputField.addEventListener("input", { jsEditor.redecorateLines() })
}

val decorator by lazy { TextareaDecorator(inputField, parser) }
val jsEditor by lazy { JSTidyEditor(inputField, outputField) }
val inputField by lazy { document.getElementById("tidyparse-input") as HTMLTextAreaElement }
val outputField by lazy { document.getElementById("tidyparse-output") as Node }