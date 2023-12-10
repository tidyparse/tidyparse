import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import kotlinx.browser.*
import org.w3c.dom.*

var cfg: CFG? = null
var cachedGrammar: String? = null

/**
TODO (soon):
 - Integrate with [ai.hypergraph.tidyparse.TidyEditor]
 - Clean up the Gradle build wreckage
 - Get Myers diffs working properly
 - Fix periodic lag during repairs
 - Render the Chomsky-normalized CFG
 - Rank results by more sensible metric
 - Add Ctrl+Space code completion popup
 - Provide assistance for grammar editing
 *//*
TODO (maybe):
 - Add demo for Python and Java
 - Use a proper editor instead of a TextArea
    - https://github.com/kueblc/LDT
 - Syntax highlighting for the grammar
 - Syntax highlighting for the snippets
 - Configurable settings, e.g., timeout, max repairs
 - Maybe possible to support ligatures?
 - Auto-alignment of the productions
 - Calculate finger-travel distance
 - Find a better pattern for timeslicing
 - Collect telemetry for a user study
 - Improve support for incrementalization
 - Look into ropes, zippers and lenses
*/

// ./gradlew browserDevelopmentRun --continuous
fun main() {
  preprocessGrammar()
  inputField.addEventListener("input", { processEditorContents() })
}

val inputField by lazy { document.getElementById("tidyparse-input") as HTMLTextAreaElement }
val outputField by lazy { document.getElementById("tidyparse-output") as Node }

fun preprocessGrammar() {
  val currentGrammar = inputField.grammar()
  if (cachedGrammar != currentGrammar) cfg = currentGrammar.parseCFG()
  cachedGrammar = currentGrammar
}

fun HTMLTextAreaElement.grammar(): String = value.substringBefore("---")
fun HTMLTextAreaElement.getEndOfLineIdx() = value.indexOf("\n", selectionStart!!)
fun HTMLTextAreaElement.getCurrentLine() = value.substring(0, getEndOfLineIdx()).split("\n").last()
fun HTMLTextAreaElement.isCursorInsideGrammar() = "---" in value.substring(0, inputField.selectionStart!!)

fun processEditorContents() {
  handleInput()
  preprocessGrammar()
}

var cache = mutableMapOf<Int, String>()
var workHash = 0
val toTake = 30

fun handleInput() {
  preprocessGrammar()

  if (!inputField.isCursorInsideGrammar()) return
  val line = inputField.getCurrentLine()
  workHash = line.tokenizeByWhitespace().hashCode()

  if ("_" !in line) {
    val tree = cfg!!.parse(line)?.prettyPrint()
    if (tree != null) outputField.textContent = "$line\n\n$tree"
    else cfg!!.repairSeq(line.tokenizeByWhitespace()).distinct()
        .take(toTake).iterator().extractAndRender(line)
  } else {
    outputField.textContent = "Solving: $line\n"
    println("Repairing $line")

    cfg!!.enumSeqSmart(line.tokenizeByWhitespace()).distinct()
      .take(toTake).iterator().extractAndRender(line)
  }
}

private fun Iterator<String>.extractAndRender(line: String) {
  val repairsToDisplay = mutableListOf<String>()
  val workHashWhenStarted = workHash

  fun flush(string: String) { (outputField as HTMLElement).innerHTML = string }

  fun updateDisplay(suffix: String = ""): String =
    repairsToDisplay.sortedBy { it.length }.joinToString("\n", "", suffix)
      .also { flush(it) }

  fun rerenderAndPause() {
    if (workHashWhenStarted == workHash) {
      if (hasNext()) {
        repairsToDisplay.add(line.diff(next()))
        updateDisplay()
        window.setTimeout({ rerenderAndPause() }, 10) // 10 ms pause
      } else { cache[workHashWhenStarted] = updateDisplay("\n...") }
    }
  }

  if (workHashWhenStarted !in cache) rerenderAndPause()
  else flush(cache[workHashWhenStarted]!!)
}

fun String.diff(other: String): String = other.escapeHTML()

//fun String.diff(other: String): String {
//  val output = tokenizeByWhitespace().toMutableList()
//  differenceOf(output, other.tokenizeByWhitespace())
//    .applyDiff(
//      remove = { index -> output.removeAt(index) },
//      insert = { item, index -> output.add(index, "<span style=\"background-color: green;\">${item.escapeHTML()}</span>") },
//      move = { old, new ->  }
//    )
//  return output.joinToString(" ") { if (it.startsWith("<span style=")) it else it.escapeHTML() }
//}