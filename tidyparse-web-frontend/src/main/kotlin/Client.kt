import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.hasTimeLeft
import kotlinx.browser.*
import org.w3c.dom.*
import kotlin.math.absoluteValue
import kotlin.time.TimeSource

var cfg: CFG? = null
var cachedGrammar: String? = null

/**
TODO (soon):
 - Integrate with [ai.hypergraph.tidyparse.TidyEditor]
 - Clean up the Gradle build wreckage
 - Get Myers diffs working properly
 - Render the Chomsky-normalized CFG
 - Rank results by more sensible metric
 - Add Ctrl+Space code completion popup
 - Provide assistance for grammar editing
 *//*
TODO (maybe):
 - Add demo for Python and Java
 - Use a proper editor instead of a TextArea
    - https://github.com/kueblc/LDT
 - Syntax highlighting for the snippets
 - Configurable settings, e.g., timeout, max repairs
 - Auto-alignment of the productions
 - Calculate finger-travel distance
 - Find a better pattern for timeslicing
 - Collect telemetry for a user study
 - Improve support for incrementalization
 - Look into ropes, zippers and lenses
*/

// ./gradlew browserDevelopmentRun --continuous
fun main() {
  TIMEOUT_MS = 5_000
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
fun HTMLTextAreaElement.isCaretInGrammar() = "---" !in value.substring(0, inputField.selectionStart!!)

fun processEditorContents() {
  handleInput()
  preprocessGrammar()
}

var cache = mutableMapOf<Int, String>()
var currentWorkHash = 0
val toTake = 30

fun handleInput() {
  preprocessGrammar()

  if (inputField.isCaretInGrammar()) { return }
  val line = inputField.getCurrentLine()
  val lineHash = line.tokenizeByWhitespace().hashCode()
  currentWorkHash = lineHash
  val startTime = TimeSource.Monotonic.markNow()

  val tokens = line.tokenizeByWhitespace()
  if ("_" !in line) {
    val tree = cfg!!.parse(line)?.prettyPrint()
    if (tree != null) outputField.textContent = "$line\n\n$tree"
    else if (lineHash in cache) flush(cache[lineHash]!!)
    else cfg!!.fastRepairSeq(tokens)
      .enumerateCompletionsInteractively(
        metric = { levenshtein(tokens, it) * 7919 + (tokens.sumOf { it.length } - it.sumOf { it.length }).absoluteValue },
        resultsToPost = toTake,
        shouldKeepGoing = { currentWorkHash == lineHash && startTime.hasTimeLeft() },
        postResults = { flush(it) },
        done = { cache[lineHash] = it; flush(it) },
        restThenResume = { window.setTimeout(it, 10) }
      )
  } else {
    outputField.textContent = "Solving: $line\n"
    println("Repairing $line")

    if (lineHash in cache) flush(cache[lineHash]!!)
    else cfg!!.enumSeqSmart(line.tokenizeByWhitespace()).distinct()
      .enumerateCompletionsInteractively(
        metric = { it.size * 7919 + it.sumOf { it.length } },
        resultsToPost = toTake,
        shouldKeepGoing = { currentWorkHash == lineHash && startTime.hasTimeLeft() },
        postResults = { flush(it) },
        done = { cache[lineHash] = it; flush(it) },
        restThenResume = { window.setTimeout(it, 50) }
      )
  }
}

fun flush(string: String) { outputField.textContent = string }

fun Sequence<String>.enumerateCompletionsInteractively(
  resultsToPost: Int,
  metric: (List<String>) -> Int,
  shouldKeepGoing: () -> Boolean,
  postResults: (String) -> Unit,
  done: (String) -> Unit = { postResults(it) },
  restThenResume: (() -> Unit) -> Unit
) {
  val results = mutableSetOf<String>()
  val topNResults = mutableListOf<Pair<String, Int>>()
  val iter = iterator()

  fun findNextRepair() {
    if (iter.hasNext() && shouldKeepGoing()) {
      val next = iter.next()
      val isNew = next !in results
      if (next.isNotEmpty() && isNew) {
        results.add(next)
        if (topNResults.size < resultsToPost || next.length < topNResults.last().second) {
          val score = metric(next.tokenizeByWhitespace())
          val loc = topNResults.binarySearch { it.second.compareTo(score) }
          val idx = if (loc < 0) { -loc - 1 } else loc
          topNResults.add(idx, next to score)
          if (topNResults.size > resultsToPost) topNResults.removeLast()
          postResults(topNResults.joinToString("\n") { it.first })
        }
      }

      restThenResume(::findNextRepair)
    } else done(topNResults.joinToString("\n", "", "\n...") { it.first })
  }

  findNextRepair()
}

fun String.diff(other: String): String = other

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