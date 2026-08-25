@file:OptIn(ExperimentalUnsignedTypes::class)

import ai.hypergraph.tidyparse.wgpu.*
import ai.hypergraph.tidyparse.wgpu.GPUBufferUsage.STCPSD
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.GPUBuffer
import js.buffer.ArrayBuffer
import js.typedarrays.Int32Array
import kotlinx.coroutines.await
import kotlin.js.Promise
import kotlin.time.TimeSource

private const val WDFA_ASSET = "wdfa.bin"
private const val NGRAM_ASSET = "python_4grams.txt"
private const val DEFAULT_ASSET_DIRECTORY = "python3-repair/"

external val self: dynamic

private var configuredAssetBaseUrl: String? = null
private var runtimeStatusSink: ((String, String) -> Unit)? = null
private var lastTimeMeasurement: TimeSource.Monotonic.ValueTimeMark? = null

fun installRuntimeStatusSink(sink: (String, String) -> Unit) {
  runtimeStatusSink = sink
  configureWgpuRuntime(::log, ::reportRuntimeStatus)
}

fun reportRuntimeStatus(state: String, message: String) {
  log("[$state] $message")
  runtimeStatusSink?.invoke(state, message)
}

fun log(message: String) {
  if (lastTimeMeasurement == null) lastTimeMeasurement = TimeSource.Monotonic.markNow()
  val elapsed = lastTimeMeasurement!!.elapsedNow().inWholeMilliseconds
  println("(repair-worker Δ=${elapsed}ms): $message")
  lastTimeMeasurement = TimeSource.Monotonic.markNow()
}

fun configureRepairAssets(assetBaseUrl: String?): String {
  val requested = assetBaseUrl?.trim().orEmpty().ifBlank { DEFAULT_ASSET_DIRECTORY }
  val absolute = js("new URL(requested, self.location.href).href") as String
  return absolute.trimEnd('/')
    .plus('/')
    .also { configuredAssetBaseUrl = it }
}

fun repairAssetUrl(path: String): String {
  val base = configuredAssetBaseUrl ?: configureRepairAssets(null)
  return js("new URL(path, base).href") as String
}

private suspend fun browserBinaryResource(path: String): ArrayBuffer? {
  val url = repairAssetUrl(path)
  val response = self.fetch(url).unsafeCast<Promise<dynamic>>().await()
  val ok = response.ok as? Boolean ?: false
  if (!ok) {
    log("Asset fetch failed (${response.status ?: "unknown"}): $url")
    return null
  }
  return response.arrayBuffer().unsafeCast<Promise<ArrayBuffer>>().await()
}

private suspend fun browserTextResource(path: String): String? {
  val url = repairAssetUrl(path)
  val response = self.fetch(url).unsafeCast<Promise<dynamic>>().await()
  val ok = response.ok as? Boolean ?: false
  if (!ok) {
    log("Asset fetch failed (${response.status ?: "unknown"}): $url")
    return null
  }
  return response.text().unsafeCast<Promise<String>>().await()
}

suspend fun browserWdfa(file: String): ArrayBuffer? = browserBinaryResource(file)

suspend fun browserRerankerWeights(file: String): ArrayBuffer? = browserBinaryResource(file)

suspend fun loadNgrams(file: String = NGRAM_ASSET): Int {
  val started = TimeSource.Monotonic.markNow()
  val raw = browserTextResource(file) ?: error("Failed to load n-grams from ${repairAssetUrl(file)}")
  val parsed = linkedMapOf<List<String>, Double>()
  var order = 0
  raw.lineSequence().filter(String::isNotBlank).forEach { line ->
    val fields = line.split(" ::: ", limit = 2)
    require(fields.size == 2) { "Malformed n-gram row: $line" }
    val tokens = fields[0].split(' ')
    order = tokens.size
    parsed[tokens] = fields[1].toDouble()
  }
  require(parsed.isNotEmpty()) { "N-gram asset is empty" }

  val loadedBuffer = parsed.toGpuHash(cfg = pythonRepairGrammar).loadToGPUBuffer()
  ngrams?.destroy()
  ngrams = loadedBuffer
  log("Loaded ${parsed.size} ${order}-grams into ${loadedBuffer.size}B in ${started.elapsedNow()}")
  return parsed.size
}

suspend fun loadWDFA(file: String = WDFA_ASSET) {
  val started = TimeSource.Monotonic.markNow()
  val loaded = browserWdfa(file) ?: error("Failed to load WDFA from ${repairAssetUrl(file)}")
  val ints = Int32Array(loaded)
  require(ints.length >= 5) { "WDFA asset is shorter than its header" }

  val loadedBuffer = GPUBuffer(byteSize = ints.byteLength, us = STCPSD, data = ints)
  wdfa?.destroy()
  wdfa = loadedBuffer
  wdfaNumStates = ints[3]
  wdfaNumEdges = ints[4]
  log("Loaded WDFA(|Q|=$wdfaNumStates, |δ|=$wdfaNumEdges) in ${started.elapsedNow()}")
}
