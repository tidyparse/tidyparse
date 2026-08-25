import ai.hypergraph.tidyparse.wgpu.MAX_DISP_RESULTS
import ai.hypergraph.tidyparse.wgpu.RepairReranker
import ai.hypergraph.tidyparse.wgpu.configureRepairReranker
import ai.hypergraph.tidyparse.wgpu.gpuAvailable
import ai.hypergraph.tidyparse.wgpu.neuralRerankerEnabled
import ai.hypergraph.tidyparse.wgpu.tryBootstrappingGPU
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

private val workerScope = MainScope()
private val workerMutex = Mutex()
private var initialized = false
private var activeMode = "cpu"
private var readyMessage = "Syntax repair worker ready"
private var latestRepairGeneration = 0
private const val DEFAULT_REPAIR_RESULT_LIMIT = 100

fun main() {
  configureRepairReranker(::browserRerankerWeights)
  installRuntimeStatusSink(::postStatus)
  self.onmessage = { event: dynamic ->
    val request = event.data
    val repairGeneration = if (request?.type as? String == "repair") { ++latestRepairGeneration } else { 0 }
    workerScope.launch { workerMutex.withLock { handleWorkerRequest(request, repairGeneration) } }
  }
  postStatus("loading", "Syntax repair worker loaded")
}

private suspend fun handleWorkerRequest(request: dynamic, repairGeneration: Int) {
  val id = request?.id
  try {
    when (request?.type as? String) {
      "initialize" -> initializeWorker(id, request.assetBaseUrl as? String)
      "repair" -> {
        check(initialized) { "Syntax repair worker has not been initialized" }
        if (repairGeneration != latestRepairGeneration) {
          postRepairs(id, PythonLineRepairResult(repairMode = false, repairs = emptyList()))
          return
        }
        val line = request.line as? String ?: error("Repair request is missing line")
        val maxResults = (dynamicIntOrNull(request.maxResults) ?: DEFAULT_REPAIR_RESULT_LIMIT)
          .coerceIn(1, 1_000)
        postStatus("working", "Generating syntax repairs")
        val result = repairPythonLine(line, maxResults)
        postRepairs(id, result)
        postStatus("ready", readyMessage)
      }
      else -> error("Unknown syntax repair request type: ${request?.type}")
    }
  } catch (t: Throwable) {
    val message = t.message ?: t.toString()
    log("Request failed: $message")
    postError(id, message)
    postStatus("error", message)
  }
}

private suspend fun initializeWorker(id: dynamic, assetBaseUrl: String?) {
  val assetBase = configureRepairAssets(assetBaseUrl)
  postStatus("warming", "Initializing WebGPU repair runtime")

  tryBootstrappingGPU(needsExtraMemory = true)

  var wdfaReady = false
  var ngramsReady = false
  var neuralReady = false
  if (gpuAvailable) {
    try {
      postStatus("warming", "Loading Python repair language model")
      loadNgrams()
      ngramsReady = true
    } catch (t: Throwable) {
      log("N-gram initialization failed: ${t.message ?: t}")
    }

    try {
      postStatus("warming", "Loading weighted repair automaton")
      loadWDFA()
      wdfaReady = true
    } catch (t: Throwable) {
      log("WDFA initialization failed: ${t.message ?: t}")
    }

    postStatus("warming", "Loading neural repair reranker")
    neuralRerankerEnabled = true
    neuralReady = RepairReranker.preloadAvailable()
    if (!neuralReady) neuralRerankerEnabled = false
  } else {
    neuralRerankerEnabled = false
  }

  activeMode = if (gpuAvailable) "webgpu" else "cpu"
  initialized = true
  val capabilities = buildString {
    append("assets=").append(assetBase)
    append("; grammar=true")
    append("; ngrams=").append(ngramsReady)
    append("; wdfa=").append(wdfaReady)
    append("; neural=").append(neuralReady)
  }
  val readyLabel = when {
    gpuAvailable && wdfaReady && neuralReady -> "WebGPU + WDFA + neural syntax repair ready"
    gpuAvailable && wdfaReady -> "WebGPU + WDFA syntax repair ready"
    gpuAvailable -> "WebGPU syntax repair ready"
    else -> "CPU syntax repair fallback ready"
  }
  readyMessage = readyLabel
  postInitialized(
    id = id,
    mode = activeMode,
    gpuReady = gpuAvailable,
    neuralReady = neuralReady,
    candidateResultLimit = DEFAULT_REPAIR_RESULT_LIMIT,
    displayResultLimit = MAX_DISP_RESULTS,
    message = "$readyLabel; $capabilities"
  )
  postStatus("ready", readyLabel)
}

private fun dynamicIntOrNull(value: dynamic): Int? =
  if (value == null || jsTypeOf(value) != "number") null else (value as Number).toInt()

private fun postStatus(state: String, message: String) {
  val response = js("({})")
  response.type = "status"
  response.state = state
  response.message = message
  self.postMessage(response)
}

private fun postInitialized(
  id: dynamic,
  mode: String,
  gpuReady: Boolean,
  neuralReady: Boolean,
  candidateResultLimit: Int,
  displayResultLimit: Int,
  message: String
) {
  val response = js("({})")
  response.id = id
  response.type = "initialized"
  response.mode = mode
  response.gpuAvailable = gpuReady
  response.neuralReady = neuralReady
  response.candidateResultLimit = candidateResultLimit
  response.displayResultLimit = displayResultLimit
  response.message = message
  self.postMessage(response)
}

private fun postRepairs(id: dynamic, result: PythonLineRepairResult) {
  val response = js("({})")
  response.id = id
  response.type = "repairs"
  response.repairMode = result.repairMode
  response.repairs = result.repairs.toTypedArray()
  self.postMessage(response)
}

private fun postError(id: dynamic, error: String) {
  val response = js("({})")
  response.id = id
  response.type = "error"
  response.error = error
  self.postMessage(response)
}
