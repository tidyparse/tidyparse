@file:OptIn(ExperimentalUnsignedTypes::class)

import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import js.buffer.ArrayBuffer
import js.typedarrays.Int32Array
import kotlinx.browser.*
import kotlinx.coroutines.*
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent
import org.w3c.fetch.RequestInit
import web.gpu.GPUBuffer
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
  MainScope().launch {
    try {
      if (window["REPAIR_MODE"] == "headless") headlessSetup()
      else if (window["REPAIR_MODE"] == "jcef") jcefSetup()
      else if (window["PROGRAMMING_LANG"] == "cnf") cnfSetup()
      else if (window["PROGRAMMING_LANG"] == "python") pythonSetup()
      else defaultSetup()
    } catch (t: Throwable) {
      if (window["REPAIR_MODE"] == "jcef") jcefSend("__TIDYPARSE_ERROR__${t.message ?: t.toString()}")
      else throw t
    }
  }
}

private val jcefScope = MainScope()
private val jcefNgrams: MutableMap<List<String>, Double> = mutableMapOf()
private var jcefNgramTensor: GPUBuffer? = null
private fun jcefSend(s: String) { window.asDynamic().__tidyparseJcefSend(s) }
suspend fun jcefSetup() {
  log("Starting TidyPython (JCEF)…")

  val cfg = pythonStatementCNFAllProds
  loadNgrams(target = jcefNgrams)
  LED_BUFFER = 2
  TIMEOUT_MS = 1000
  tryBootstrappingGPU(needsExtraMemory = true)
  if (gpuAvailable) jcefNgramTensor = jcefNgrams.toGpuHash(cfg = cfg).loadToGPUBuffer()
  if (gpuAvailable) loadWDFA()
  log("Bootstrapped GPU")

  window.asDynamic().__tidyparseRunString = { line: String ->
    jcefScope.launch {
      try { jcefSend(repairPythonLineRaw(cfg, line)) }
      catch (t: Throwable) { jcefSend("__TIDYPARSE_ERROR__${t.message ?: t.toString()}") }
    }
    Unit
  }

  jcefSend("READY")
  log("JCEF runtime ready")
}

private suspend fun repairPythonLineRaw(cfg: CFG, line: String, maxResults: Int = 50): String =
  JSTidyPyEditor.handleInput(line, cfg, jcefNgramTensor, maxResults).joinToString("\n")

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

var lastCaretStart = 0
var lastCaretEnd = 0

private val cmEditor: dynamic
  get() = window.asDynamic().cmEditor

private fun hasCmEditor(): Boolean = cmEditor != null && cmEditor != js("undefined")

suspend fun defaultSetup() {
  log("Starting Tidyparse/CFG")
  initTidyCodeMirror()
  initSplitLayout()
  inputField.scrollTop = inputField.scrollHeight.toDouble();

  if (!fetchUrlExample()) fetchSelectedExample()
  jsEditor.getLatestCFG()
  jsEditor.redecorateLines()
  LED_BUFFER = ledBuffSel.value.toInt()
  TIMEOUT_MS = timeout.value.toInt()
  jsEditor.epsilons = epscheck.checked
  jsEditor.ntStubs = ntscheck.checked
  lastCaretStart = inputField.asDynamic().selectionStart as Int
  lastCaretEnd = inputField.asDynamic().selectionEnd as Int

  inputField.addEventListener("selectionchange", {
    val start = inputField.asDynamic().selectionStart as Int
    val end = inputField.asDynamic().selectionEnd as Int
    val multi = start != end

    if (multi || start == lastCaretStart && end == lastCaretEnd) return@addEventListener

    lastCaretStart = start
    lastCaretEnd = end
    jsEditor.run { continuation { handleInput() } }
  })

  inputField.addEventListener("input", { jsEditor.run { continuation { handleInput() } } })
  inputField.addEventListener("input", { jsEditor.redecorateLines() })
  exSelector.addEventListener("change", { MainScope().launch {
    jsEditor.restoreInstructions()
    fetchSelectedExample()
    lastCaretStart = inputField.asDynamic().selectionStart as Int
    lastCaretEnd = inputField.asDynamic().selectionEnd as Int
    jsEditor.getLatestCFG()
    jsEditor.redecorateLines()
  } })
  window.addEventListener("hashchange", { MainScope().launch {
    if (fetchUrlExample()) {
      lastCaretStart = inputField.asDynamic().selectionStart as Int
      lastCaretEnd = inputField.asDynamic().selectionEnd as Int
      jsEditor.getLatestCFG()
      jsEditor.redecorateLines()
    }
  }})

  if (hasCmEditor()) cmEditor.on("keydown") { _: dynamic, event: dynamic -> jsEditor.navUpdate(event) }
  else inputField.addEventListener("keydown", { event -> jsEditor.navUpdate(event as KeyboardEvent) })
  epscheck.addEventListener("change", { log("Changed check"); jsEditor.epsilons = epscheck.checked; log("Checked: ${jsEditor.epsilons}") })
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
  jsPyEditor.redecorateLines()

  loadNgrams()
  MainScope().launch {
    val t0 = TimeSource.Monotonic.markNow()
    tryBootstrappingGPU(true)
    if (gpuAvailable) {
      log("Loaded n-grams into ${jsPyEditor.ngramTensor.size / 1000000}mb GPU buffer in ${t0.elapsedNow()}")
      loadWDFA()
      debugWDFATokenIndexing()
    }
  }

  TIMEOUT_MS = 1000

  inputField.addEventListener("input", { jsPyEditor.run { continuation { handleInput() } } })
  inputField.addEventListener("input", { jsPyEditor.redecorateLines() })
  jsPyEditor.cme.on("keydown") { _: dynamic, e: dynamic -> jsPyEditor.navUpdate(e) }

//  jsPyEditor.minimize = mincheck.checked
//  mincheck.addEventListener("change", { jsPyEditor.minimize = mincheck.checked })
  LED_BUFFER = ledBuffSel.value.toInt()
  ledBuffSel.addEventListener("change", { LED_BUFFER = ledBuffSel.value.toInt() })

  jsPyEditor.initPyodide()
}

val exSelector by lazy { document.getElementById("ex-selector") as HTMLSelectElement }
val decorator by lazy { TextareaDecorator(inputField, parser) }
val pyDecorator by lazy { PyTextareaDecorator(jsPyEditor.cme) }
val jsEditor by lazy { JSTidyEditor(inputField, outputField) }
val jsPyEditor by lazy { JSTidyPyEditor(inputField, outputField) }
val inputField by lazy { document.getElementById("tidyparse-input") as HTMLTextAreaElement }
val outputField by lazy { document.getElementById("tidyparse-output") as Node }
val epscheck by lazy { document.getElementById("epsilon-checkbox") as HTMLInputElement }
val ntscheck by lazy { document.getElementById("ntstubs-checkbox") as HTMLInputElement }
val timeout by lazy { document.getElementById("timeout") as HTMLInputElement }
val ledBuffSel by lazy { document.getElementById("led-buffer") as HTMLInputElement }

var wdfa: GPUBuffer? = null
var wdfaNumStates: Int = 0
var wdfaNumEdges: Int = 0

suspend fun loadWDFA(file: String = "wdfa.bin") {
  val t0 = TimeSource.Monotonic.markNow()
  val inlineWdfaB64 = window.asDynamic().raw_wdfa_b64 as? String
  if (!inlineWdfaB64.isNullOrBlank()) {
    loadWDFAFromArrayBuffer(base64ToArrayBuffer(inlineWdfaB64))
    log("Loaded WDFA(|Q|=$wdfaNumStates, |δ|=$wdfaNumEdges) inline in ${t0.elapsedNow()}")
    return
  }

  val response = window.fetch(file).await()
  if (!response.ok) { "Failed to load WDFA from $file".also { log(it); error(it) } }

  loadWDFAFromArrayBuffer(response.arrayBuffer().await().unsafeCast<ArrayBuffer>())
  log("Loaded WDFA(|Q|=$wdfaNumStates, |δ|=$wdfaNumEdges) from $file in ${t0.elapsedNow()}")
}

private fun loadWDFAFromArrayBuffer(ab: ArrayBuffer) {
  val ints = Int32Array(ab)
  val buf = GPUBuffer(byteSize = ints.byteLength, us = STCPSD, data = ints)

  wdfaNumStates = ints[3]
  wdfaNumEdges = ints[4]
  wdfa = buf
}

private fun base64ToArrayBuffer(b64: String): ArrayBuffer =
  js("""(function(s) {
      const binary = atob(s);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
  })""")(b64).unsafeCast<ArrayBuffer>()

suspend fun loadNgrams(
  file: String = "python_4grams.txt",
  target: MutableMap<List<String>, Double> = jsPyEditor.ngrams
) {
  val t0 = TimeSource.Monotonic.markNow()
  val inlineNgrams = window.asDynamic().raw_ngrams as? String
  if (!inlineNgrams.isNullOrBlank()) {
    parseNgrams(inlineNgrams, target)
    log("Loaded ${target.size} inline n-grams in ${t0.elapsedNow()}")
    return
  }

  val response = window.fetch(file).await()
  if (response.ok) {
    val n = parseNgrams(response.text().await(), target)
    log("Loaded ${target.size} $n-grams from $file in ${t0.elapsedNow()}")
  } else log("Failed to load ngrams from $file")
}

private fun parseNgrams(raw: String, target: MutableMap<List<String>, Double>): Int {
  var n = 0
  raw.lines().filter { it.isNotBlank() }.forEach { line ->
    val (ngram, count) = line.split(" ::: ")
    target[ngram.split(" ").also { n = it.size }] = count.toDouble()
  }
  return n
}

private fun Element.scrollIntoViewCompat() {
  val opts = js("({ block: 'end', inline: 'nearest', behavior: 'auto' })")
  try { this.asDynamic().scrollIntoView(opts) } catch (_: dynamic) { this.scrollIntoView(false) }
}

private fun decodeUrlComponent(value: String): String =
  try { js("decodeURIComponent(value)") as String } catch (_: dynamic) { value }

private fun directExampleRequest(): String? {
  val hash = window.location.hash.removePrefix("#")
  return normalizeExamplePath(hash) ?: normalizeExamplePath(window.location.pathname)
}

private fun normalizeExamplePath(rawPath: String): String? {
  var path = decodeUrlComponent(rawPath).trim()
    .substringBefore("?")
    .substringBefore("&")
    .removePrefix("example=")
    .removePrefix("/")

  if (path.startsWith("http://") || path.startsWith("https://"))
    path = path.substringAfter("://").substringAfter("/")

  path = path.replace('\\', '/')
    .removePrefix("/")
    .removePrefix("./")
    .replace(Regex("^examples/griebach/"), "examples/greibach/")
    .replace(Regex("^griebach/"), "greibach/")

  if (!path.endsWith(".tidy") || path.split('/').any { it.isBlank() || it == ".." }) return null
  return if (path.startsWith("examples/")) path else "examples/$path"
}

private fun syncExampleSelector(examplePath: String) {
  val options = exSelector.asDynamic().options
  for (i in 0 until (options.length as Int))
    if (options[i].value == examplePath) { exSelector.value = examplePath; return }
}

private fun updateExampleHash(examplePath: String) {
  val nextHash = "#$examplePath"
  if (window.location.hash != nextHash)
    window.history.asDynamic().replaceState(null, document.title, nextHash)
}

suspend fun fetchUrlExample(): Boolean =
  fetchExample(directExampleRequest() ?: return false, syncSelector = true, updateHash = false)

suspend fun fetchSelectedExample() {
  val selectedExample = exSelector.value
  if (selectedExample.endsWith(".html")) { window.location.href = selectedExample; return }
  fetchExample(selectedExample, syncSelector = false, updateHash = true)
}

suspend fun fetchExample(examplePath: String, syncSelector: Boolean, updateHash: Boolean): Boolean {
  val response = window.fetch(examplePath).await()
  if (response.ok) {
    val text = response.text().await()
    inputField.value = text
    if (hasCmEditor()) {
      cmEditor.setValue(text)
      cmEditor.save()
      cmEditor.scrollTo(null, cmEditor.getScrollInfo().height)
    } else window.requestAnimationFrame { (inputField as Element).scrollIntoViewCompat() }
    if (syncSelector) syncExampleSelector(examplePath)
    if (updateHash) updateExampleHash(examplePath)
    jsEditor.redecorateLines()
    return true
  } else console.error("Failed to load file '$examplePath': ${response.status}")
  return false
}
