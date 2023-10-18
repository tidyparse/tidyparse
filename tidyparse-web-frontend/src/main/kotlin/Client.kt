import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import kotlinx.browser.document
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

fun CoroutineScope.handleInput() {
  preprocessGrammar()
  if (!inputField.isCursorInsideGrammar()) return
  val line = inputField.getCurrentLine()
  val tree= cfg!!.parse(line)?.prettyPrint()
  if (tree != null) outputField.textContent = "Parsing: $line\n\n$tree"
  else {
    outputField.textContent = "Solving: $line\n"
    println("Repairing $line")
    // Block names from repair since we don't have a good way to handle them yet
    cfg?.terminals?.filter { it.all { it.isLetterOrDigit() } }
      ?.let { cfg?.blocked?.addAll(it) }

    repair(
      prompt = line,
      cfg = cfg!!,
      synthesizer = { enumSeq(it) },
      updateProgress = { updateProgress(it) }
    )
      .also { println("Found ${it.size} repairs") }
      .let { outputField.textContent = it.joinToString("\n") }
  }
}