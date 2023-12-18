import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.hasTimeLeft
import kotlinx.browser.*
import org.w3c.dom.*
import kotlin.math.absoluteValue
import kotlin.time.TimeSource

/**
TODO (soon):
 - Extract platform-independent code from IntelliJ plugin
 - Get Myers diffs working properly
 - Clean up the Gradle build wreckage
 - Render the Chomsky-normalized CFG
 - Syntax highlighting for the snippets
 - Rank results by more sensible metric
 - Add Ctrl+Space code completion popup
 - Provide assistance for grammar editing
 - Allow richer HTML content in RHS panel
 *//*
TODO (maybe):
 - Add demo for Python and Java
 - Configurable settings, e.g., timeout, max repairs
 - Auto-alignment of the productions
 - Calculate finger-travel distance
 - Collect telemetry for a user study
 - Improve support for incrementalization
 - Look into ropes, zippers and lenses
*/
val parser = Parser(
  "whitespace" to "\\s+",
  "red" to "->|\\|",
  "blue" to "---",
  "gray" to "_",
  "green" to "START",
  "other" to "\\S"
  // Uncomment or add more rules as needed
  // "orange" to "orange",
  // "yellow" to "yellow",
  // "indigo" to "indigo",
  // "violet" to "violet",
)

// ./gradlew browserDevelopmentRun --continuous
fun main() {
  TIMEOUT_MS = 5_000
  jsEditor.getLatestCFG()
  // Wait for the page to finish loading before accessing the DOM
  window.onload = { TextareaDecorator(inputField, parser) }
  inputField.addEventListener("input", { jsEditor.handleInput() })
}

val jsEditor by lazy { JSTidyEditor(inputField, outputField) }
val inputField by lazy { document.getElementById("tidyparse-input") as HTMLTextAreaElement }
val outputField by lazy { document.getElementById("tidyparse-output") as Node }