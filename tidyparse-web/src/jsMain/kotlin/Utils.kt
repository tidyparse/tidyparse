import kotlinx.browser.window
import kotlin.time.TimeSource

fun IntArray.toLaTeX(numStates: Int, numNTs: Int): String {
  val tikzCommands = if (numStates == 0) "" else {
    (0 until numStates).flatMap { q1_rowIndex ->
      (0 until numStates).map { q2_colIndex ->
        val isActive = if (numNTs > 0) {
          val baseFlatIndex = q1_rowIndex * numStates * numNTs + q2_colIndex * numNTs
          (0 until numNTs).any { ntIdxInSlice ->
            val currentFlatIndex = baseFlatIndex + ntIdxInSlice
            (currentFlatIndex < size && this[currentFlatIndex] != 0)
          }
        } else { false }

        val tikzX = q2_colIndex
        val tikzY = numStates - 1 - q1_rowIndex

        val fillColor = if (isActive) "black" else "white"
        "  \\path[fill=${fillColor}] (${tikzX},${tikzY}) rectangle ++(1,1);"
      }
    }.joinToString("\n")
  }

  val squareUnitSize = "0.3cm"
  return """
  \begin{tikzpicture}[x=${squareUnitSize}, y=${squareUnitSize}, draw=gray, very thin]
  ${tikzCommands.ifBlank { "% Empty grid" }}
  \end{tikzpicture}
  """.trimIndent()
}

var lastTimeMeasurement: TimeSource.Monotonic.ValueTimeMark? = null
var DEBUG_SUFFIX = ""

fun log(s: String) {
  if (lastTimeMeasurement == null) lastTimeMeasurement = TimeSource.Monotonic.markNow()
  val prefix = "(Δ=${lastTimeMeasurement!!.elapsedNow().inWholeMilliseconds}ms):".padEnd(11)
  println("$prefix$s$DEBUG_SUFFIX")
  lastTimeMeasurement = TimeSource.Monotonic.markNow()
}

fun setCompletionsAndShow(items: List<String>) {
  window.asDynamic().COMPLETIONS = items.toTypedArray()

  val cm = window.asDynamic().cmEditor ?: return

  window.setTimeout({
    try { cm.state.completionActive?.close() } catch (_: dynamic) {}
    cm.focus()
    cm.showHint(js("({hint: window.fixedHtmlHint, completeSingle: false})"))
  }, 0)
}