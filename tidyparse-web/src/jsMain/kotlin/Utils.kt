import ai.hypergraph.kaliningraph.parsing.Σᐩ
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import js.buffer.ArrayBuffer
import kotlinx.browser.window
import kotlinx.coroutines.await
import kotlin.js.Promise
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

fun List<Σᐩ>.stripEpsilon() = asSequence().map { it.replace("ε", "").tokenizeByWhitespace().joinToString(" ") }

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

fun browserResourceUrl(path: String): String = js("new URL(path, document.baseURI).href") as String

private const val RAW_NGRAMS_GZIP_B64 = "raw_ngrams_gzip_b64"
private const val RAW_WDFA_GZIP_B64 = "raw_wdfa_gzip_b64"
private const val RAW_RERANKER_WEIGHTS_GZIP_B64 = "raw_reranker_weights_gzip_b64"

private fun browserGlobalString(name: String): String? = (window.asDynamic()[name] as? String)?.takeIf { it.isNotBlank() }

suspend fun gzipBase64ToArrayBuffer(b64: String): ArrayBuffer =
  js("""(function(s) {
      const binary = atob(s);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).arrayBuffer();
  })""")(b64).unsafeCast<Promise<ArrayBuffer>>().await()

suspend fun gzipBase64ToText(b64: String): String =
  js("""(function(s) {
      const binary = atob(s);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).text();
  })""")(b64).unsafeCast<Promise<String>>().await()

private suspend fun browserTextResource(file: String, inlineGzipB64: String): String? {
  browserGlobalString(inlineGzipB64)?.let { return gzipBase64ToText(it) }
  val response = window.fetch(browserResourceUrl(file)).await()
  return if (response.ok) response.text().await() else null
}

private suspend fun browserBinaryResource(file: String, inlineGzipB64: String): ArrayBuffer? {
  browserGlobalString(inlineGzipB64)?.let { return gzipBase64ToArrayBuffer(it) }
  val response = window.fetch(browserResourceUrl(file)).await()
  return if (response.ok) response.arrayBuffer().await().unsafeCast<ArrayBuffer>() else null
}

suspend fun browserNgrams(file: String): String? = browserTextResource(file, RAW_NGRAMS_GZIP_B64)
suspend fun browserWdfa(file: String): ArrayBuffer? = browserBinaryResource(file, RAW_WDFA_GZIP_B64)
suspend fun browserRerankerWeights(file: String): ArrayBuffer? = browserBinaryResource(file, RAW_RERANKER_WEIGHTS_GZIP_B64)