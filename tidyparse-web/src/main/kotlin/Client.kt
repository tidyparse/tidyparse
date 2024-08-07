import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import ai.hypergraph.kaliningraph.types.PlatformVars
import kotlinx.browser.*
import org.w3c.dom.*
import org.w3c.dom.events.Event

/**
TODO (soon):
 - Extract platform-independent code from IntelliJ plugin
 - Switch to an adaptive repair algorithm / more efficient variant of sampleWRGD
 - Look into constrained inference with llama.cpp / BERT
 - Render the Chomsky-normalized CFG
 - Rank results by more sensible metric
 - Provide assistance for grammar editing
 - Improve support for incrementalization
 - Configurable settings, e.g., timeout, max repairs, minimization
 - Add "real world" demo for Python/Java
 - Probabilistic Levenshtein-Bar-Hillel
 *//*
TODO (maybe):
 - Add Ctrl+Space code completion popup
 - Auto-alignment of the productions
 - Calculate finger-travel distance
 - Collect telemetry for a user study
 - Look into ropes, zippers and lenses
   - http://strictlypositive.org/diff.pdf
   - https://www.scs.stanford.edu/11au-cs240h/notes/zipper.html
   - https://www.st.cs.uni-saarland.de/edu/seminare/2005/advanced-fp/docs/huet-zipper.pdf
   - http://blog.ezyang.com/2010/04/you-could-have-invented-zippers/
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
  TIMEOUT_MS = 10_000
  jsEditor.getLatestCFG()
  window.onload = { jsEditor.redecorateLines() }
  inputField.addEventListener("input", { jsEditor.run { continuation { handleInput() } } })
  inputField.addEventListener("input", { jsEditor.redecorateLines() })
  mincheck.addEventListener("change", { jsEditor.minimize = mincheck.checked })
  ntscheck.addEventListener("change", {
    jsEditor.ntStubs = ntscheck.checked
    try {
      jsEditor.cfg = jsEditor.getGrammarText().parseCFG(validate = true)
        .let { if (ntscheck.checked) it else it.noNonterminalStubs }
    } catch (e: Exception) {}
    jsEditor.redecorateLines()
  })
}

val decorator by lazy { TextareaDecorator(inputField, parser) }
val jsEditor by lazy { JSTidyEditor(inputField, outputField) }
val inputField by lazy { document.getElementById("tidyparse-input") as HTMLTextAreaElement }
val outputField by lazy { document.getElementById("tidyparse-output") as Node }
val mincheck by lazy { document.getElementById("minimize-checkbox") as HTMLInputElement }
val ntscheck by lazy { document.getElementById("ntstubs-checkbox") as HTMLInputElement }