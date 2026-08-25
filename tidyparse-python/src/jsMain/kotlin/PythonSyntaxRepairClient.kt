import kotlinx.browser.window
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.async
import kotlinx.coroutines.await
import kotlin.js.Promise

private const val PYTHON_REPAIR_WORKER_FILE = "tidyparse-python-repair.js"
private const val PYTHON_REPAIR_ASSET_DIRECTORY = "python3-repair/"
private const val PYTHON_REPAIR_BOOT_TIMEOUT_MS = 90_000
private const val PYTHON_REPAIR_REQUEST_TIMEOUT_MS = 45_000

data class PythonRepairRuntime(
  val mode: String,
  val gpuAvailable: Boolean,
  val neuralReady: Boolean,
  val candidateResultLimit: Int,
  val displayResultLimit: Int,
  val message: String
)

data class PythonRepairResult(
  val repairMode: Boolean,
  val repairs: List<String>,
  val displayResultLimit: Int
)

/** Thin browser client for the separately bundled WebGPU syntax-repair worker. */
class PythonSyntaxRepairClient(
  private val onStatus: (state: String, message: String) -> Unit
) {
  private data class PendingRequest(
    val resolve: (dynamic) -> Unit,
    val reject: (Throwable) -> Unit,
    val timeout: Int
  )

  private val scope = MainScope()
  private val pending = mutableMapOf<Int, PendingRequest>()
  private var worker: dynamic = createWorker()
  private var initialization: Deferred<PythonRepairRuntime>? = null
  private var runtime: PythonRepairRuntime? = null
  private var nextId = 1
  private var disposed = false

  suspend fun initialize(): PythonRepairRuntime {
    check(!disposed) { "Syntax repair worker was disposed" }
    if (!repairDefined(worker)) worker = createWorker()
    runtime?.let { return it }
    val active = initialization ?: scope.async { initializeOnce() }.also { initialization = it }
    return active.await()
  }

  suspend fun repairLine(line: String, maxResults: Int? = null): PythonRepairResult {
    val activeRuntime = initialize()
    onStatus("working", "Generating ${activeRuntime.rankingLabel()} syntax repairs…")
    val request = repairObject()
    request.type = "repair"
    request.line = line
    request.maxResults = maxResults ?: activeRuntime.candidateResultLimit

    return try {
      val response = send(request, PYTHON_REPAIR_REQUEST_TIMEOUT_MS).await()
      check(response.type as? String == "repairs") { "Repair worker returned an unsupported response" }
      val repairMode = response.repairMode as? Boolean
        ?: error("Repair worker response is missing repairMode")
      val repairs = response.repairs
      val result = ArrayList<String>(repairArrayLength(repairs))
      for (index in 0 until repairArrayLength(repairs)) {
        (repairs[index] as? String)?.let(result::add)
      }
      val current = runtime
      onStatus("ready", current?.message ?: "Syntax repair is ready")
      PythonRepairResult(
        repairMode = repairMode,
        repairs = result,
        displayResultLimit = activeRuntime.displayResultLimit
      )
    } catch (failure: Throwable) {
      onStatus("error", failure.message ?: "Syntax repair failed")
      throw failure
    }
  }

  fun dispose() {
    if (disposed) return
    disposed = true
    terminateWorker()
    rejectPending("Syntax repair worker was disposed")
    runtime = null
    initialization = null
    onStatus("idle", "Syntax repair stopped")
  }

  fun completionDetail(): String =
    (runtime?.rankingLabel() ?: "local") + " repair · ty admissible"

  private suspend fun initializeOnce(): PythonRepairRuntime {
    onStatus("loading", "Loading WebGPU repair and neural ranking…")
    val request = repairObject()
    request.type = "initialize"
    request.assetBaseUrl = pythonBrowserResourceUrl(PYTHON_REPAIR_ASSET_DIRECTORY)

    return try {
      val response = send(request, PYTHON_REPAIR_BOOT_TIMEOUT_MS).await()
      check(response.type as? String == "initialized") {
        "Repair worker returned an unsupported initialization response"
      }
      val candidateResultLimit = (response.candidateResultLimit as? Number)?.toInt()
        ?: error("Repair worker response is missing candidateResultLimit")
      require(candidateResultLimit > 0) { "Repair worker returned an invalid candidateResultLimit" }
      val displayResultLimit = (response.displayResultLimit as? Number)?.toInt()
        ?: error("Repair worker response is missing displayResultLimit")
      require(displayResultLimit > 0) { "Repair worker returned an invalid displayResultLimit" }
      val initialized = PythonRepairRuntime(
        mode = response.mode as? String ?: "unknown",
        gpuAvailable = response.gpuAvailable as? Boolean ?: false,
        neuralReady = response.neuralReady as? Boolean ?: false,
        candidateResultLimit = candidateResultLimit,
        displayResultLimit = displayResultLimit,
        message = response.message as? String ?: "Syntax repair is ready"
      )
      runtime = initialized
      onStatus("ready", initialized.message)
      initialized
    } catch (failure: Throwable) {
      initialization = null
      onStatus("error", failure.message ?: "Unable to initialize syntax repair")
      throw failure
    }
  }

  private fun send(message: dynamic, timeoutMs: Int): Promise<dynamic> = Promise { resolve, reject ->
    if (disposed || !repairDefined(worker)) {
      reject(Throwable("Syntax repair worker is unavailable"))
      return@Promise
    }

    val id = nextId++
    message.id = id
    val timeout = window.setTimeout({
      if (id !in pending) return@setTimeout
      failWorker("Syntax repair worker timed out after ${timeoutMs / 1_000} seconds")
    }, timeoutMs)
    pending[id] = PendingRequest(resolve, reject, timeout)

    try {
      worker.postMessage(message)
    } catch (failure: Throwable) {
      pending.remove(id)
      window.clearTimeout(timeout)
      reject(failure)
    }
  }

  private fun createWorker(): dynamic {
    val created = js("(url) => new Worker(url, { name: 'tidyparse-python-repair' })")(
      pythonBrowserResourceUrl(PYTHON_REPAIR_WORKER_FILE)
    )
    created.onmessage = { event: dynamic ->
      if (sameRepairWorker(worker, created)) handleMessage(event.data)
    }
    created.onerror = { event: dynamic ->
      if (sameRepairWorker(worker, created)) {
        js("(event) => { if (event && typeof event.preventDefault === 'function') event.preventDefault(); }")(event)
        val detail = event?.message as? String ?: "Syntax repair worker failed"
        failWorker(detail)
      }
    }
    created.onmessageerror = { _: dynamic ->
      if (sameRepairWorker(worker, created)) {
        failWorker("Syntax repair worker returned an unreadable response")
      }
    }
    return created
  }

  private fun handleMessage(message: dynamic) {
    if (message.type as? String == "status" && (message.id as? Number) == null) {
      onStatus(
        message.state as? String ?: "loading",
        message.message as? String ?: "Preparing syntax repair…"
      )
      return
    }

    val id = (message.id as? Number)?.toInt() ?: return
    val request = pending.remove(id) ?: return
    window.clearTimeout(request.timeout)

    if (message.type as? String == "error") {
      request.reject(Throwable(message.error as? String ?: "Syntax repair worker failed"))
    } else {
      request.resolve(message)
    }
  }

  private fun terminateWorker() {
    if (repairDefined(worker)) worker.terminate()
    worker = null
  }

  private fun failWorker(detail: String) {
    terminateWorker()
    runtime = null
    initialization = null
    onStatus("error", detail)
    rejectPending(detail)
  }

  private fun rejectPending(detail: String) {
    val requests = pending.values.toList()
    pending.clear()
    requests.forEach { request ->
      window.clearTimeout(request.timeout)
      request.reject(Throwable(detail))
    }
  }
}

private fun PythonRepairRuntime.rankingLabel(): String = when {
  gpuAvailable && neuralReady -> "WebGPU + neural"
  gpuAvailable -> "WebGPU + WDFA"
  else -> "CPU grammar"
}

private fun repairObject(): dynamic = js("({})")

private fun repairArrayLength(value: dynamic): Int =
  if (repairDefined(value) && js("Array.isArray(value)") as Boolean) (value.length as Number).toInt()
  else 0

private fun repairDefined(value: dynamic): Boolean =
  js("(value) => value !== null && value !== undefined")(value) as Boolean

private fun sameRepairWorker(left: dynamic, right: dynamic): Boolean =
  js("(left, right) => left === right")(left, right) as Boolean
