@file:OptIn(ExperimentalUnsignedTypes::class)

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.PlatformVars
import ai.hypergraph.tidyparse.MAX_DISP_RESULTS
import kotlinx.browser.*
import kotlinx.coroutines.*
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent
import org.w3c.fetch.RequestInit
import kotlin.js.Promise
import kotlin.time.TimeSource

/**
TODO (soon):
 - Look into constrained inference with llama.cpp / BERT
 - Rank results by more sensible metric
 - Improve support for incrementalization
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

// ./gradlew :tidyparse-web:jsBrowserDevelopmentRun --continuous
fun main() {
  if (window.navigator.userAgent.indexOf("hrome") != -1) {
    PlatformVars.PLATFORM_CALLER_STACKTRACE_DEPTH = 4
  }

  MainScope().async {
    if (window["REPAIR_MODE"] == "headless") headlessSetup()
    else if (window["PROGRAMMING_LANG"] == "cnf") cnfSetup()
    else if (window["PROGRAMMING_LANG"] == "python") pythonSetup()
    else defaultSetup()
  }
}

suspend fun headlessSetup() {
  log("Starting Tidyparse (headless)…")

  val cfg = vanillaS2PCFG
  tryBootstrappingGPU(needsExtraMemory = true)
  log("Bootstrapped GPU")

  var errors = 0
  val es = EventSource("/stream")
  es.onmessage = { ev ->
    MainScope().launch {
      errors = 0
      val prompt = (ev.data as String).also { log("Received prompt: $it") }.tokenizeByWhitespace()
//      val out = repairCode(cfg, prompt, LED_BUFFER, ngramTensor) // With reranking + truncation
      val out = repairCode(cfg, prompt, LED_BUFFER, null) // Without reranking + truncation
        .distinct().joinToString("\n")
      window.fetch("/result", RequestInit(method = "POST", body = out)).await()
    }
  }
  es.onerror = { if (errors++ > 20) es.close() }
}

suspend fun defaultSetup() {
  log("Starting Tidyparse/CFG")

  fetchSelectedExample()
  jsEditor.getLatestCFG()
  jsEditor.redecorateLines()
  LED_BUFFER = ledBuffSel.value.toInt()
  TIMEOUT_MS = timeout.value.toInt()
  jsEditor.minimize = mincheck.checked
  jsEditor.ntStubs = ntscheck.checked

  inputField.addEventListener("input", { jsEditor.run { continuation { handleInput() } } })
  inputField.addEventListener("input", { jsEditor.redecorateLines() })
  exSelector.addEventListener("change", { MainScope().async { fetchSelectedExample() } })

  inputField.addEventListener("keydown", { event -> jsEditor.navUpdate(event as KeyboardEvent) })
  mincheck.addEventListener("change", { jsEditor.minimize = mincheck.checked })
  ntscheck.addEventListener("change", {
    jsEditor.ntStubs = ntscheck.checked
    try {
      jsEditor.cfg = jsEditor.getGrammarText().parseCFG(validate = true)
        .let { if (ntscheck.checked) it else it.noNonterminalStubs }
    } catch (_: Exception) {}
    jsEditor.redecorateLines()
  })
  ledBuffSel.addEventListener("change", { LED_BUFFER = ledBuffSel.value.toInt() })
  timeout.addEventListener("change", { TIMEOUT_MS = timeout.value.toInt() })

  tryBootstrappingGPU()
}

suspend fun pythonSetup() {
  log("Starting TidyPython")

  jsPyEditor.getLatestCFG()
//    LED_BUFFER = maxEdits.value.toInt()

  loadNgrams()
  log("Loaded ngrams")
  MainScope().async {
    val t0 = TimeSource.Monotonic.markNow()
    tryBootstrappingGPU(true)
    if (gpuAvailable)
      log("Loaded n-grams into ${jsPyEditor.ngramTensor.size / 1000000}mb GPU buffer in ${t0.elapsedNow()}")
  }

  TIMEOUT_MS = 1000

  inputField.addEventListener("input", { jsPyEditor.run { continuation { handleInput() } } })
  inputField.addEventListener("input", { jsPyEditor.redecorateLines() })
  inputField.addEventListener("keydown", { event -> jsPyEditor.navUpdate(event as KeyboardEvent) })

//  jsPyEditor.minimize = mincheck.checked
//  mincheck.addEventListener("change", { jsPyEditor.minimize = mincheck.checked })
  LED_BUFFER = ledBuffSel.value.toInt()
  ledBuffSel.addEventListener("change", { LED_BUFFER = ledBuffSel.value.toInt() })

  initPyodide()
}

val exSelector by lazy { document.getElementById("ex-selector") as HTMLSelectElement }
val decorator by lazy { TextareaDecorator(inputField, parser) }
val pyDecorator by lazy { PyTextareaDecorator(inputField, parser = parser) }
val jsEditor by lazy { JSTidyEditor(inputField, outputField) }
val jsPyEditor by lazy { JSTidyPyEditor(inputField, outputField) }
val inputField by lazy { document.getElementById("tidyparse-input") as HTMLTextAreaElement }
val outputField by lazy { document.getElementById("tidyparse-output") as Node }
val mincheck by lazy { document.getElementById("minimize-checkbox") as HTMLInputElement }
val ntscheck by lazy { document.getElementById("ntstubs-checkbox") as HTMLInputElement }
val timeout by lazy { document.getElementById("timeout") as HTMLInputElement }
val ledBuffSel by lazy { document.getElementById("led-buffer") as HTMLInputElement }

suspend fun loadNgrams(file: String = "python_4grams.txt") {
  val t0 = TimeSource.Monotonic.markNow()
  val response = window.fetch(file).await()
  if (response.ok) {
    var numNgrams = 0
    var n = 0
    response.text().await().lines().filter { it.isNotBlank() }.forEach { line ->
      val (ngram, count) = line.split(" ::: ")
      jsPyEditor.ngrams[ngram.split(" ").also { n = it.size }] = count.toDouble()
      numNgrams++
    }

    log("Loaded ${jsPyEditor.ngrams.size} $n-grams from $file in ${t0.elapsedNow()}")
  } else log("Failed to load ngrams from $file")
}

suspend fun initPyodide() = try {
  val scriptTag = (document.querySelector("script[src*='pyodide.js']") as HTMLScriptElement)
    .getAttribute("src")!!.substringBefore("pyodide.js")

  val config = js("{}")
  config.indexURL = scriptTag
  jsPyEditor.pyodide = window.asDynamic().loadPyodide(config).unsafeCast<Promise<*>>().await()
  jsPyEditor.pyodide.loadPackage("micropip").unsafeCast<Promise<*>>().await()

  val micropip = jsPyEditor.pyodide.pyimport("micropip")
  micropip.install("black").unsafeCast<Promise<*>>().await()

  val testStr = "1+1"
  val beautified = jsPyEditor.formatCode(testStr)

  log("Black test => $beautified")
  log(jsPyEditor.getOutput("1+"))
} catch (e: Exception) { log("Error during Pyodide initialization: ${e.message}") }

suspend fun fetchSelectedExample() {
  if (exSelector.value.endsWith(".html")) {
    window.location.href = exSelector.value
    return
  }
  val response = window.fetch(exSelector.value).await()
  if (response.ok) {
    val text = response.text().await()
    inputField.apply {
      value = text
      window.setTimeout({scrollIntoView(js("{ behavior: 'instant', block: 'end' }"))}, 1)
    }
    jsEditor.redecorateLines()
  } else console.error("Failed to load file: ${response.status}")
}