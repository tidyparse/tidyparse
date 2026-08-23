import kotlin.js.Promise

private const val RUST_GLANCER_WASM_FILE = "tidyparse-rust-glancer.wasm"

internal fun setupRustGlancerWorker() {
  val scope: dynamic = js("globalThis")
  val runtime: dynamic = loadRustGlancerRuntime()

  scope.onmessage = { event: dynamic ->
    val message = event.data
    if (message?.type as? String != "analyze") {
      Unit
    } else {
      val id = (message.id as Number).toInt()
      val source = message.source as? String ?: ""
      runtime.then(
        { wasm: dynamic ->
          try {
            postGlancerResult(scope, id, analyzeWithRustGlancer(wasm, source))
          } catch (failure: Throwable) {
            postGlancerFailure(scope, id, failure.message ?: "Rust Glancer analysis failed")
          }
        },
        { failure: dynamic ->
          postGlancerFailure(scope, id, jsErrorMessage(failure))
        }
      )
    }
  }
}

private fun loadRustGlancerRuntime(): dynamic {
  val wasmUrl = js("(fileName) => new URL(fileName, globalThis.location.href).href")(
    RUST_GLANCER_WASM_FILE
  )
  return js(
    """(url) => fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Unable to load Rust Glancer Wasm (" + response.status + ")");
        }
        return response.arrayBuffer();
      })
      .then((bytes) => WebAssembly.instantiate(bytes, {}))
      .then((instantiated) => {
        const wasm = instantiated.instance.exports;
        if (wasm.tidyparse_rust_abi_version() !== 1) {
          throw new Error("Unsupported Rust Glancer Wasm ABI");
        }
        return wasm;
      })"""
  )(wasmUrl)
}

private fun analyzeWithRustGlancer(wasm: dynamic, source: String): dynamic {
  val input: dynamic = js("(text) => new TextEncoder().encode(text)")(source)
  val length = (input.length as Number).toInt()
  val pointer = (wasm.tidyparse_rust_alloc(length) as Number).toInt()
  if (length > 0 && pointer == 0) error("Rust Glancer could not allocate source memory")

  try {
    js("(memory, pointer, length, input) => new Uint8Array(memory.buffer, pointer, length).set(input)")(
      wasm.memory,
      pointer,
      length,
      input
    )
    wasm.tidyparse_rust_analyze(pointer, length)
  } finally {
    wasm.tidyparse_rust_dealloc(pointer, length)
  }

  val outputPointer = (wasm.tidyparse_rust_output_ptr() as Number).toInt()
  val outputLength = (wasm.tidyparse_rust_output_len() as Number).toInt()
  val json = js(
    """(memory, pointer, length) =>
      new TextDecoder().decode(new Uint8Array(memory.buffer, pointer, length))"""
  )(wasm.memory, outputPointer, outputLength) as String
  return JSON.parse<dynamic>(json)
}

private fun postGlancerResult(scope: dynamic, id: Int, analysis: dynamic) {
  val response = js("({})")
  response.type = "result"
  response.id = id
  response.analysis = analysis
  scope.postMessage(response)
}

private fun postGlancerFailure(scope: dynamic, id: Int, message: String) {
  val response = js("({})")
  response.type = "error"
  response.id = id
  response.error = message
  scope.postMessage(response)
}

private fun jsErrorMessage(failure: dynamic): String =
  failure?.message as? String ?: failure?.toString() ?: "Rust Glancer failed to load"

internal class RustGlancerClient(
  private val onStatus: (state: String, message: String) -> Unit
) {
  private data class Pending(
    val resolve: (dynamic) -> Unit,
    val reject: (Throwable) -> Unit
  )

  private val worker: dynamic = createRustNamedWorker(RUST_GLANCER_WORKER_NAME)
  private val pending = mutableMapOf<Int, Pending>()
  private var nextId = 1
  private var ready = false

  init {
    onStatus("loading", "Loading Rust Glancer…")
    worker.onmessage = { event: dynamic ->
      val message = event.data
      val id = (message?.id as? Number)?.toInt()
      val request = if (id == null) null else pending.remove(id)
      if (request != null) {
        if (message.type as? String == "result") {
          if (!ready) {
            ready = true
            onStatus("ready", "Rust Glancer is ready")
          }
          request.resolve(message.analysis)
        } else {
          val detail = message.error as? String ?: "Rust Glancer worker failed"
          onStatus("error", detail)
          request.reject(Throwable(detail))
        }
      }
    }
    worker.onerror = { event: dynamic ->
      val detail = event?.message as? String ?: "Rust Glancer worker failed"
      onStatus("error", detail)
      rejectAll(detail)
    }
  }

  fun analyze(source: String): Promise<dynamic> = Promise { resolve, reject ->
    val id = nextId++
    pending[id] = Pending(resolve, reject)
    val request = js("({})")
    request.type = "analyze"
    request.id = id
    request.source = source
    worker.postMessage(request)
  }

  private fun rejectAll(message: String) {
    val requests = pending.values.toList()
    pending.clear()
    requests.forEach { it.reject(Throwable(message)) }
  }
}
