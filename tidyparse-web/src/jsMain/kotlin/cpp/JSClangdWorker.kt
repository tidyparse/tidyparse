import kotlinx.coroutines.MainScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import kotlin.js.Promise

private const val CPP_CLANGD_WORKER_NAME = "tidyparse-clangd"
internal const val CPP_CLANGD_ARTIFACT_VERSION = "llvm-21.1.0-emsdk-4.0.22-wasi-29.0-r2"
private const val CPP_CLANGD_JS_PATH = "wasm/clangd.js"
private const val CPP_CLANGD_WASM_PATH = "wasm/clangd.wasm"

private const val CPP_CLANGD_WORKSPACE_PATH = "/home/web_user"
private const val CPP_CLANGD_CPP_PATH = "$CPP_CLANGD_WORKSPACE_PATH/main.cpp"
private const val CPP_CLANGD_C_PATH = "$CPP_CLANGD_WORKSPACE_PATH/main.c"
private const val CPP_CLANGD_WORKSPACE_URI = "file://$CPP_CLANGD_WORKSPACE_PATH"
private const val CPP_CLANGD_CPP_URI = "file://$CPP_CLANGD_CPP_PATH"
private const val CPP_CLANGD_C_URI = "file://$CPP_CLANGD_C_PATH"
private const val CPP_CLANGD_MAX_VIRTUAL_FILE_SIZE = 4 * 1024 * 1024

private val cppClangdWorkerScope: dynamic
  get() = js("globalThis")

fun isCppClangdWorkerRuntime(): Boolean =
  js(
    """(name) => typeof document === "undefined" &&
       typeof globalThis.postMessage === "function" &&
       globalThis.name === name"""
  )(CPP_CLANGD_WORKER_NAME) as Boolean

fun setupCppClangdWorker() {
  val worker = cppClangdWorkerScope
  if (worker.__tidyparseClangdStarted == true) return
  worker.__tidyparseClangdStarted = true

  val input = ClangdLspInput()
  val output = ClangdLspOutput(::postClangdLspMessage, ::postClangdProtocolError)
  val stderr = ClangdStderrOutput()
  val virtualFiles = ClangdVirtualFiles()

  worker.onmessage = { event: dynamic ->
    handleClangdWorkerMessage(event.data, input, virtualFiles)
  }

  postClangdStatus("loading", "Loading clangd module")
  MainScope().launch {
    try {
      startClangd(input, output, stderr, virtualFiles)
    } catch (failure: Throwable) {
      postClangdFailure(failure)
    }
  }
}

private suspend fun startClangd(
  input: ClangdLspInput,
  output: ClangdLspOutput,
  stderr: ClangdStderrOutput,
  virtualFiles: ClangdVirtualFiles
) {
  val clangdJsUrl = clangdArtifactUrl(CPP_CLANGD_JS_PATH)
  val clangdWasmUrl = clangdArtifactUrl(CPP_CLANGD_WASM_PATH)

  // Keep the generated Emscripten module outside webpack's module graph.
  val importModule = js("Function('url', 'return import(url)')")
  val importedModule =
    (importModule(clangdJsUrl) as Promise<dynamic>).await()
  val clangdFactory = importedModule.default
  if (clangdFactory == null || clangdFactory == js("undefined")) {
    error("The clangd module has no default factory export")
  }

  postClangdStatus("loading", "Initializing clangd")
  val options = js("{}")
  options.mainScriptUrlOrBlob = clangdPthreadBootstrap(clangdJsUrl)
  options.thisProgram = "/usr/bin/clangd"
  options.locateFile = { path: String, prefix: String ->
    if (path.endsWith(".wasm")) clangdWasmUrl else prefix + path
  }
  options.stdinReady = { input.ready() }
  options.stdin = { input.readByte() }
  options.stdout = { byte: Int -> output.accept(byte) }
  options.stderr = { byte: Int -> stderr.accept(byte) }
  options.onAbort = { reason: dynamic ->
    postClangdError("clangd aborted: $reason", fatal = true)
  }
  options.onExit = { code: dynamic ->
    val message = clangdEnvelope("exit")
    message.code = code
    cppClangdWorkerScope.postMessage(message)
  }

  val clangd = awaitDynamic(clangdFactory(options))
  installClangdWorkspace(clangd)
  virtualFiles.attach(clangd)

  postClangdStatus("starting", "Starting clangd")
  clangd.callMain(
    arrayOf(
      "--compile-commands-dir=$CPP_CLANGD_WORKSPACE_PATH",
      "--background-index=0",
      "--clang-tidy=0",
      "-j=4",
      "--header-insertion=never",
      "--limit-results=100",
      "--log=error"
    )
  )

  val ready = clangdEnvelope("ready")
  ready.workspaceUri = CPP_CLANGD_WORKSPACE_URI
  ready.fileUri = CPP_CLANGD_CPP_URI
  ready.cppFileUri = CPP_CLANGD_CPP_URI
  ready.cFileUri = CPP_CLANGD_C_URI
  cppClangdWorkerScope.postMessage(ready)
}

private fun clangdArtifactUrl(path: String): String =
  js(
    """(base, path, version) => {
      const url = new URL(path, base);
      url.searchParams.set("v", version);
      return url.href;
    }"""
  )(
    cppClangdWorkerScope.location.href,
    path,
    CPP_CLANGD_ARTIFACT_VERSION
  ) as String

/**
 * A module worker can receive Emscripten's initial `load` message while its
 * imported clangd module is still evaluating. Queue those early messages and
 * replay them once clangd has installed the pthread handler.
 */
private fun clangdPthreadBootstrap(clangdJsUrl: String): dynamic =
  js(
    """(url) => {
      const source = [
        "const queuedMessages = [];",
        "self.onmessage = event => queuedMessages.push(event);",
        "await import(" + JSON.stringify(url) + ");",
        "const handler = self.onmessage;",
        "for (const event of queuedMessages) handler(event);"
      ].join("\n");
      return new Blob([source], { type: "text/javascript" });
    }"""
  )(clangdJsUrl)

private fun installClangdWorkspace(clangd: dynamic) {
  val fs = clangd.FS
  try {
    fs.mkdirTree(CPP_CLANGD_WORKSPACE_PATH)
  } catch (_: Throwable) {
    // Emscripten normally creates /home/web_user while initializing FS.
  }

  fs.writeFile(CPP_CLANGD_CPP_PATH, "")
  fs.writeFile(CPP_CLANGD_C_PATH, "")
  fs.writeFile(
    "$CPP_CLANGD_WORKSPACE_PATH/compile_commands.json",
    CPP_CLANGD_COMPILE_COMMANDS
  )
  fs.writeFile(
    "$CPP_CLANGD_WORKSPACE_PATH/.clangd",
    """{"CompileFlags":{"CompilationDatabase":"$CPP_CLANGD_WORKSPACE_PATH"}}"""
  )
}

private fun handleClangdWorkerMessage(
  data: dynamic,
  input: ClangdLspInput,
  virtualFiles: ClangdVirtualFiles
) {
  if (data == null || data == js("undefined")) return

  try {
    when (data.type as? String) {
      "lsp", "message" -> {
        val message =
          if (data.message != null && data.message != js("undefined")) data.message
          else data.payload
        if (message == null || message == js("undefined")) {
          postClangdProtocolError("Missing LSP message payload")
        } else {
          input.enqueue(message)
        }
      }

      "configure" -> {
        val language = if ((data.language as? String).equals("c", true)) "c" else "cpp"
        val configured = clangdEnvelope("configured")
        configured.language = language
        configured.fileUri = if (language == "c") CPP_CLANGD_C_URI else CPP_CLANGD_CPP_URI
        cppClangdWorkerScope.postMessage(configured)
      }

      "readFile" -> virtualFiles.read(data)

      "ping" -> cppClangdWorkerScope.postMessage(clangdEnvelope("pong"))

      null -> {
        // BrowserMessageWriter-style clients post JSON-RPC objects directly.
        if (data.jsonrpc == "2.0" ||
          data.method != null && data.method != js("undefined") ||
          data.id != null && data.id != js("undefined")
        ) {
          input.enqueue(data)
        }
      }
    }
  } catch (failure: Throwable) {
    postClangdProtocolError(failure.message ?: "Unable to queue LSP message")
  }
}

private class ClangdVirtualFiles {
  private var clangd: dynamic = null

  fun attach(module: dynamic) {
    clangd = module
  }

  fun read(request: dynamic) {
    val response = clangdEnvelope("file")
    response.id = request.id

    try {
      if (clangd == null || clangd == js("undefined")) {
        error("clangd virtual filesystem is not ready")
      }

      val requestedUri = request.uri as? String
      val requestedPath = request.path as? String
      val path = normalizeVirtualPath(
        when {
          !requestedPath.isNullOrBlank() -> requestedPath
          !requestedUri.isNullOrBlank() -> virtualPathFromUri(requestedUri)
          else -> error("readFile requires an absolute path or file URI")
        }
      )

      val fs = clangd.FS
      val stat = fs.stat(path)
      if (fs.isFile(stat.mode) != true) {
        error("$path is not a regular file")
      }
      val size = stat.size as Int
      if (size > CPP_CLANGD_MAX_VIRTUAL_FILE_SIZE) {
        error("$path is too large to open (${size} bytes)")
      }

      val options = js("{}")
      options.encoding = "utf8"
      val text = fs.readFile(path, options) as? String
        ?: error("Unable to decode $path as UTF-8")

      response.path = path
      response.uri = requestedUri ?: virtualFileUri(path)
      response.text = text
    } catch (failure: Throwable) {
      response.type = "fileError"
      response.path = request.path
      response.uri = request.uri
      response.message = failure.message ?: "Unable to read virtual file"
    }

    cppClangdWorkerScope.postMessage(response)
  }
}

private fun virtualPathFromUri(uri: String): String =
  js(
    """(uri) => {
      const parsed = new URL(uri);
      if (parsed.protocol !== "file:") {
        throw new Error("Only file: URIs can be read from clangd");
      }
      if (parsed.hostname !== "" && parsed.hostname !== "localhost") {
        throw new Error("Remote file URI authorities are not supported");
      }
      return decodeURIComponent(parsed.pathname);
    }"""
  )(uri) as String

private fun normalizeVirtualPath(path: String): String =
  js(
    """(path) => {
      if (!path.startsWith("/") || path.includes("\0")) {
        throw new Error("Virtual file paths must be absolute");
      }
      const normalized = [];
      for (const segment of path.split("/")) {
        if (segment === "" || segment === ".") continue;
        if (segment === "..") {
          normalized.pop();
        } else {
          normalized.push(segment);
        }
      }
      return "/" + normalized.join("/");
    }"""
  )(path) as String

private fun virtualFileUri(path: String): String =
  js(
    """(path) => {
      const uri = new URL("file:///");
      uri.pathname = path;
      return uri.href;
    }"""
  )(path) as String

private class ClangdLspInput {
  private val encoder: dynamic = js("new TextEncoder()")
  private val segments: dynamic = js("[]")
  private val waiters = mutableListOf<(Unit) -> Unit>()

  private var current: dynamic = null
  private var offset = 0
  private var boundaryPending = false

  fun enqueue(message: dynamic) {
    val json = JSON.stringify(message)
    val body = encoder.encode(json)
    val header = encoder.encode("Content-Length: ${body.length as Int}\r\n")
    val delimiter = encoder.encode("\r\n")

    // Returning null between these segments makes libc return a partial read.
    // clangd's Emscripten patch then awaits stdinReady before retrying.
    segments.push(header)
    segments.push(delimiter)
    segments.push(body)

    val readyWaiters = waiters.toList()
    waiters.clear()
    readyWaiters.forEach { it(Unit) }
  }

  fun ready(): Promise<Unit> = Promise { resolve, _ ->
    if (hasReadableInput()) resolve(Unit) else waiters += resolve
  }

  fun readByte(): Int? {
    if (boundaryPending) {
      boundaryPending = false
      return null
    }

    if (current == null || current == js("undefined")) {
      if ((segments.length as Int) == 0) return null
      current = segments.shift()
      offset = 0
    }

    val value = current[offset++] as Int
    if (offset >= (current.length as Int)) {
      current = null
      boundaryPending = true
    }
    return value
  }

  private fun hasReadableInput(): Boolean =
    boundaryPending ||
      current != null && current != js("undefined") ||
      (segments.length as Int) > 0
}

private class ClangdLspOutput(
  private val onMessage: (dynamic) -> Unit,
  private val onError: (String) -> Unit
) {
  private val decoder: dynamic = js("new TextDecoder('utf-8')")
  private var headerBytes: dynamic = js("[]")
  private var bodyBytes: dynamic = null
  private var bodyOffset = 0
  private var expectedBodyLength = -1

  fun accept(rawByte: Int) {
    val byte = rawByte and 0xff
    if (expectedBodyLength < 0) acceptHeaderByte(byte) else acceptBodyByte(byte)
  }

  private fun acceptHeaderByte(byte: Int) {
    headerBytes.push(byte)
    val size = headerBytes.length as Int
    if (size > 65_536) {
      reset()
      onError("clangd returned an oversized LSP header")
      return
    }

    if (size < 4 ||
      headerBytes[size - 4] != 13 ||
      headerBytes[size - 3] != 10 ||
      headerBytes[size - 2] != 13 ||
      headerBytes[size - 1] != 10
    ) return

    try {
      val bytes = js(
        "(values) => new Uint8Array(values.slice(0, values.length - 4))"
      )(headerBytes)
      val header = decoder.decode(bytes) as String
      val lengthLine = header.lineSequence().firstOrNull {
        it.substringBefore(':').trim().equals("Content-Length", ignoreCase = true)
      } ?: error("Missing Content-Length header")
      val length = lengthLine.substringAfter(':').trim().toInt()
      require(length >= 0) { "Negative Content-Length" }
      require(length <= 64 * 1024 * 1024) { "LSP response is too large" }

      expectedBodyLength = length
      bodyOffset = 0
      bodyBytes = js("(length) => new Uint8Array(length)")(length)
      headerBytes = js("[]")
      if (length == 0) emitBody()
    } catch (failure: Throwable) {
      reset()
      onError(failure.message ?: "Malformed clangd LSP header")
    }
  }

  private fun acceptBodyByte(byte: Int) {
    bodyBytes[bodyOffset++] = byte
    if (bodyOffset == expectedBodyLength) emitBody()
  }

  private fun emitBody() {
    try {
      val text = decoder.decode(bodyBytes) as String
      val message = js("(text) => JSON.parse(text)")(text)
      onMessage(message)
    } catch (failure: Throwable) {
      onError(failure.message ?: "Malformed clangd JSON response")
    } finally {
      reset()
    }
  }

  private fun reset() {
    headerBytes = js("[]")
    bodyBytes = null
    bodyOffset = 0
    expectedBodyLength = -1
  }
}

private class ClangdStderrOutput {
  private val decoder: dynamic = js("new TextDecoder('utf-8')")
  private var bytes: dynamic = js("[]")

  fun accept(rawByte: Int) {
    val byte = rawByte and 0xff
    if (byte == 10) {
      flush()
      return
    }
    if (byte == 13) return
    if ((bytes.length as Int) >= 8_192) flush()
    bytes.push(byte)
  }

  private fun flush() {
    if ((bytes.length as Int) == 0) return
    val line = decoder.decode(js("(values) => new Uint8Array(values)")(bytes)) as String
    bytes = js("[]")

    val log = clangdEnvelope("log")
    log.stream = "stderr"
    log.message = line
    cppClangdWorkerScope.postMessage(log)
  }
}

private suspend fun awaitDynamic(value: dynamic): dynamic =
  (js("(value) => Promise.resolve(value)")(value) as Promise<dynamic>).await()

private fun postClangdLspMessage(message: dynamic) {
  val envelope = clangdEnvelope("lsp")
  envelope.message = message
  cppClangdWorkerScope.postMessage(envelope)
}

private fun postClangdStatus(status: String, text: String) {
  val message = clangdEnvelope("status")
  message.status = status
  message.message = text
  cppClangdWorkerScope.postMessage(message)
}

private fun postClangdProtocolError(text: String) {
  postClangdError(text, fatal = false)
}

private fun postClangdError(text: String, fatal: Boolean) {
  val message = clangdEnvelope("error")
  message.message = text
  message.fatal = fatal
  cppClangdWorkerScope.postMessage(message)
}

private fun postClangdFailure(failure: Throwable) {
  val message = clangdEnvelope("error")
  message.message = failure.message ?: failure.toString()
  message.stack = failure.asDynamic().stack as? String
  message.fatal = true
  cppClangdWorkerScope.postMessage(message)
}

private fun clangdEnvelope(type: String): dynamic {
  val message = js("{}")
  message.type = type
  return message
}

private val CPP_CLANGD_COMPILE_COMMANDS = """
[
  {
    "directory": "$CPP_CLANGD_WORKSPACE_PATH",
    "file": "$CPP_CLANGD_CPP_PATH",
    "arguments": [
      "/usr/bin/clang++",
      "-xc++",
      "-std=c++23",
      "-pedantic-errors",
      "-Wall",
      "-Wextra",
      "--target=wasm32-wasi",
      "-isystem/usr/include/c++/v1",
      "-isystem/usr/include/wasm32-wasi/c++/v1",
      "-isystem/usr/include",
      "-isystem/usr/include/wasm32-wasi",
      "$CPP_CLANGD_CPP_PATH"
    ]
  },
  {
    "directory": "$CPP_CLANGD_WORKSPACE_PATH",
    "file": "$CPP_CLANGD_C_PATH",
    "arguments": [
      "/usr/bin/clang",
      "-xc",
      "-std=c23",
      "-pedantic-errors",
      "-Wall",
      "-Wextra",
      "--target=wasm32-wasi",
      "-isystem/usr/include",
      "-isystem/usr/include/wasm32-wasi",
      "$CPP_CLANGD_C_PATH"
    ]
  }
]
""".trimIndent()
