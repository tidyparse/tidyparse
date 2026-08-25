import kotlinx.browser.window
import kotlin.js.Promise

private const val PYTHON_RUNNER_WORKER_FILE = "python-runner-worker.js"
private const val PYTHON_BOOT_TIMEOUT_MS = 90_000
private const val PYTHON_RUN_TIMEOUT_MS = 15_000

data class PythonExecutionResult(
  val exitCode: Int,
  val stdout: String,
  val stderr: String,
  val timedOut: Boolean = false
)

/** Client for the independently generated classic Pyodide worker. */
class PythonRunner(
  private val onStatus: (state: String, message: String) -> Unit
) {
  private data class PendingRun(
    val resolve: (PythonExecutionResult) -> Unit,
    val reject: (Throwable) -> Unit,
    val timeout: Int,
    val phase: String
  )

  private var worker: dynamic = null
  private val pending = mutableMapOf<Int, PendingRun>()
  private var nextId = 1
  private var disposed = false

  init {
    worker = createWorker()
    onStatus("ready", "Python runner is ready")
  }

  fun run(source: String, stdin: String): Promise<PythonExecutionResult> = Promise { resolve, reject ->
    if (disposed) {
      reject(Throwable("Python runner is disposed"))
      return@Promise
    }

    val id = nextId++
    val timeout = window.setTimeout({ handleTimeout(id) }, PYTHON_BOOT_TIMEOUT_MS)
    pending[id] = PendingRun(resolve, reject, timeout, "loading")

    val request = runnerObject()
    request.id = id
    request.type = "run"
    request.source = source
    request.stdin = stdin

    onStatus("loading", "Loading Python…")
    try {
      worker.postMessage(request)
    } catch (failure: Throwable) {
      pending.remove(id)?.also { window.clearTimeout(it.timeout) }
      restartAndRejectPending(failure.message ?: "Unable to contact the Python runner")
      reject(failure)
    }
  }

  fun dispose() {
    if (disposed) return
    disposed = true
    terminateWorker()
    rejectPending("Python runner was disposed")
    onStatus("idle", "Python runner stopped")
  }

  private fun createWorker(): dynamic {
    val created = js("(url) => new Worker(url)")(
      pythonBrowserResourceUrl(PYTHON_RUNNER_WORKER_FILE)
    )
    bindWorker(created)
    return created
  }

  private fun bindWorker(created: dynamic) {
    created.onmessage = { event: dynamic ->
      if (sameRunnerWorker(worker, created)) handleMessage(event.data)
    }
    created.onerror = { event: dynamic ->
      if (sameRunnerWorker(worker, created)) {
        js("(event) => { if (event && typeof event.preventDefault === 'function') event.preventDefault(); }")(event)
        val detail = event?.message as? String ?: "Python runner worker failed"
        restartAndRejectPending(detail)
      }
    }
    created.onmessageerror = { _: dynamic ->
      if (sameRunnerWorker(worker, created)) {
        restartAndRejectPending("Python runner returned an unreadable response")
      }
    }
  }

  private fun handleMessage(message: dynamic) {
    val id = (message?.id as? Number)?.toInt()
    if (id == null) {
      restartAndRejectPending("Python runner response did not include a request id")
      return
    }

    val request = pending[id] ?: return

    if (message.type as? String == "status") {
      handleStatus(id, request, message.state as? String)
      return
    }

    pending.remove(id)
    window.clearTimeout(request.timeout)

    when (message.type as? String) {
      "result" -> {
        val result = message.result
        if (!runnerDefined(result)) {
          val detail = "Python runner returned an empty result"
          restartAndRejectPending(detail)
          request.reject(Throwable(detail))
          return
        }

        try {
          val mapped = PythonExecutionResult(
            exitCode = (result.exitCode as? Number)?.toInt()
              ?: error("Python runner result is missing exitCode"),
            stdout = result.stdout as? String ?: "",
            stderr = result.stderr as? String ?: "",
            timedOut = result.timedOut as? Boolean ?: false
          )
          onStatus("ready", "Python runner is ready")
          request.resolve(mapped)
        } catch (failure: Throwable) {
          val detail = failure.message ?: "Python runner returned a malformed result"
          restartAndRejectPending(detail)
          request.reject(failure)
        }
      }

      "error" -> {
        val detail = message.error as? String ?: "Python runner failed"
        restartAndRejectPending(detail)
        request.reject(Throwable(detail))
      }

      else -> {
        val detail = "Python runner returned an unsupported response"
        restartAndRejectPending(detail)
        request.reject(Throwable(detail))
      }
    }
  }

  private fun handleStatus(id: Int, request: PendingRun, state: String?) {
    when (state) {
      "loading" -> onStatus("loading", "Loading Python…")
      "running" -> {
        window.clearTimeout(request.timeout)
        val timeout = window.setTimeout({ handleTimeout(id) }, PYTHON_RUN_TIMEOUT_MS)
        pending[id] = request.copy(timeout = timeout, phase = "running")
        onStatus("running", "Running Python…")
      }
      else -> restartAndRejectPending("Python runner returned an unsupported status")
    }
  }

  private fun handleTimeout(id: Int) {
    val request = pending.remove(id) ?: return
    val limit = if (request.phase == "loading") PYTHON_BOOT_TIMEOUT_MS else PYTHON_RUN_TIMEOUT_MS
    val detail = if (request.phase == "loading") {
      "Python runtime did not load within ${limit / 1_000} seconds"
    } else {
      "Python execution timed out after ${limit / 1_000} seconds"
    }
    restartAndRejectPending(detail)
    request.resolve(
      PythonExecutionResult(
        exitCode = 124,
        stdout = "",
        stderr = detail,
        timedOut = true
      )
    )
  }

  private fun restartAndRejectPending(detail: String) {
    terminateWorker()
    rejectPending(detail)
    if (!disposed) {
      try {
        worker = createWorker()
      } catch (failure: Throwable) {
        worker = null
        onStatus("error", failure.message ?: detail)
        return
      }
    }
    onStatus("error", detail)
  }

  private fun rejectPending(detail: String) {
    val requests = pending.values.toList()
    pending.clear()
    requests.forEach { request ->
      window.clearTimeout(request.timeout)
      request.reject(Throwable(detail))
    }
  }

  private fun terminateWorker() {
    val current = worker
    worker = null
    if (runnerDefined(current)) {
      try {
        current.terminate()
      } catch (_: Throwable) {
        // The worker is already unusable; there is nothing else to release.
      }
    }
  }
}

private fun runnerObject(): dynamic = js("({})")

private fun runnerDefined(value: dynamic): Boolean =
  js("(value) => value !== null && value !== undefined")(value) as Boolean

private fun sameRunnerWorker(left: dynamic, right: dynamic): Boolean =
  js("(left, right) => left === right")(left, right) as Boolean
