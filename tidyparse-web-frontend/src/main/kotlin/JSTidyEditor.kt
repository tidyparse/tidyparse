import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.*
import kotlinx.browser.window
import org.w3c.dom.*
import kotlin.math.absoluteValue
import kotlin.time.TimeSource

/** Compare with [ai.hypergraph.tidyparse.IJTidyEditor] */
class JSTidyEditor(val editor: HTMLTextAreaElement, val output: Node): TidyEditor {
  var cache = mutableMapOf<Int, String>()
  var currentWorkHash = 0
  val toTake = 30

  override fun readDisplayText(): Σᐩ = output.textContent ?: ""

  override fun readEditorText(): Σᐩ = editor.textContent ?: ""

  override fun getCaretPosition(): Int = editor.selectionStart!!

  companion object {
    private fun HTMLTextAreaElement.getEndOfLineIdx() = value.indexOf("\n", selectionStart!!)
    private fun HTMLTextAreaElement.getCurrentLine() = value.substring(0, getEndOfLineIdx()).split("\n").last()
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
  }

  override fun handleInput() {
    val cfg = getLatestCFG()
    if (caretInGrammar()) { return }
    val line = currentLine()
    val lineHash = line.tokenizeByWhitespace().hashCode()
    currentWorkHash = lineHash
    val startTime = TimeSource.Monotonic.markNow()

    val tokens = line.tokenizeByWhitespace()
    if ("_" !in line) {
      val tree = cfg.parse(line)?.prettyPrint()
      if (tree != null) outputField.textContent = "$line\n\n$tree"
      else if (lineHash in cache) writeDisplayText(cache[lineHash]!!)
      else cfg.fastRepairSeq(tokens)
        .enumerateCompletionsInteractively(
          metric = { levenshtein(tokens, it) * 7919 + (tokens.sumOf { it.length } - it.sumOf { it.length }).absoluteValue },
          resultsToPost = toTake,
          shouldKeepGoing = { currentWorkHash == lineHash && startTime.hasTimeLeft() },
          postResults = { writeDisplayText(it) },
          done = { cache[lineHash] = it; writeDisplayText(it) },
          restThenResume = { window.setTimeout(it, 10) }
        )
    } else {
      outputField.textContent = "Solving: $line\n"
      println("Repairing $line")

      if (lineHash in cache) writeDisplayText(cache[lineHash]!!)
      else cfg.enumSeqSmart(line.tokenizeByWhitespace()).distinct()
        .enumerateCompletionsInteractively(
          metric = { it.size * 7919 + it.sumOf { it.length } },
          resultsToPost = toTake,
          shouldKeepGoing = { currentWorkHash == lineHash && startTime.hasTimeLeft() },
          postResults = { writeDisplayText(it) },
          done = { cache[lineHash] = it; writeDisplayText(it) },
          restThenResume = { window.setTimeout(it, 50) }
        )
    }
  }
  override fun currentLine(): Σᐩ = editor.getCurrentLine()

  override fun writeDisplayText(s: Σᐩ) { outputField.textContent = s }

  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ> = TODO()
}
