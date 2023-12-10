import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import kotlinx.browser.*
import kotlinx.coroutines.*
import org.w3c.dom.*

var cfg: CFG? = null
var cachedGrammar: String? = null

// ./gradlew browserDevelopmentRun --continuous
fun main() {
  preprocessGrammar()
  inputField.addEventListener("input", { processEditorContents() })
}

var ongoingWork: Job? = null
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
  ongoingWork?.cancel()
  ongoingWork = updateRecommendations()
  preprocessGrammar()
//  GlobalScope.async { workerPool() }
}

fun updateRecommendations() = GlobalScope.async { handleInput() }

fun updateProgress(query: String) {
  val sanitized = query.escapeHTML()
  outputField.textContent =
    outputField.textContent?.replace(
      "Solving:.*\n".toRegex(),
      "Solving: $sanitized\n"
    )
}

var cache = mutableMapOf<Int, String>()
var workHash = 0
val toTake = 30

fun CoroutineScope.handleInput() {
  preprocessGrammar()
  if (!inputField.isCursorInsideGrammar()) return
  val line = inputField.getCurrentLine()
  workHash = line.tokenizeByWhitespace().hashCode()
  if ("_" !in line) {
    val tree = cfg!!.parse(line)?.prettyPrint()
    if (tree != null) outputField.textContent = "$line\n\n$tree"
    else cfg!!.repairSeq(line.tokenizeByWhitespace()).distinct()
        .take(toTake).iterator().extractAndRender(workHash)
  } else {
    outputField.textContent = "Solving: $line\n"
    println("Repairing $line")

    cfg!!.enumSeqSmart(line.tokenizeByWhitespace()).distinct()
      .take(toTake).iterator().extractAndRender(workHash)
  }
}

private fun Iterator<String>.extractAndRender(line: Int) {
  val repairsToDisplay = mutableListOf<String>()
  val workHashWhenStarted = workHash

  fun updateDisplay(suffix: String = ""): String {
    val answerSet = repairsToDisplay.sortedBy { it.length }.joinToString("\n", "", suffix)
    outputField.textContent = answerSet
    return answerSet
  }

  fun rerenderAndPause() {
    if (workHashWhenStarted == workHash) {
      if (hasNext()) {
        repairsToDisplay.add(next())
        updateDisplay()
        window.setTimeout({ rerenderAndPause() }, 10) // 10 ms pause
      } else { cache[line] = updateDisplay("\n...") }
    }
  }

  if (line in cache) outputField.textContent = cache[line] else rerenderAndPause()
}