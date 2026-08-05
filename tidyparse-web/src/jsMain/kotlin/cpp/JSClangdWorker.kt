import kotlinx.coroutines.MainScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import kotlin.js.Promise

private const val CPP_CLANGD_WORKER_NAME = "tidyparse-clangd"
private const val CPP_CLANGD_JS_PATH = "clangd.js"
private const val CPP_CLANGD_WASM_PATH = "clangd.wasm"

private const val CPP_CLANGD_WORKSPACE_PATH = "/home/web_user"
private const val CPP_CLANGD_CPP_PATH = "$CPP_CLANGD_WORKSPACE_PATH/main.cpp"
private const val CPP_CLANGD_C_PATH = "$CPP_CLANGD_WORKSPACE_PATH/main.c"
private const val CPP_CLANGD_WORKSPACE_URI = "file://$CPP_CLANGD_WORKSPACE_PATH"
private const val CPP_CLANGD_CPP_URI = "file://$CPP_CLANGD_CPP_PATH"
private const val CPP_CLANGD_C_URI = "file://$CPP_CLANGD_C_PATH"

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
  val connection = ClangdMessagePort(input)
  val output = ClangdLspOutput(connection::send, ::postClangdProtocolError)
  val stderr = ClangdStderrOutput()

  worker.onmessage = { event: dynamic ->
    if (event.data?.type == "connect") connection.connect(event.data?.port)
  }

  postClangdStatus("loading", "Loading clangd module")
  MainScope().launch {
    try {
      startClangd(input, output, stderr, connection)
    } catch (failure: Throwable) {
      postClangdFailure(failure)
    }
  }
}

private suspend fun startClangd(
  input: ClangdLspInput,
  output: ClangdLspOutput,
  stderr: ClangdStderrOutput,
  connection: ClangdMessagePort
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
    connection.close()
  }
  options.onExit = { code: dynamic ->
    val exitCode = (code as? Number)?.toInt()
    if (exitCode != null && exitCode != 0) {
      postClangdError("clangd exited with code $exitCode", fatal = true)
    }
    connection.close()
  }

  val clangd = awaitDynamic(clangdFactory(options))
  installClangdWorkspace(clangd)

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

private class ClangdMessagePort(
  private val input: ClangdLspInput
) {
  private data class PendingAstContext(
    val source: String,
    val line: Int,
    val character: Int
  )

  private var port: dynamic = null
  private val pendingAstContexts = mutableMapOf<String, PendingAstContext>()

  fun connect(candidate: dynamic) {
    if (candidate == null || candidate == js("undefined")) {
      postClangdProtocolError("Missing clangd MessagePort")
      return
    }
    if (port != null && port != js("undefined")) {
      candidate.close()
      postClangdProtocolError("clangd already has an LSP connection")
      return
    }

    port = candidate
    candidate.onmessage = { event: dynamic ->
      try {
        input.enqueue(prepareRequest(event.data))
      } catch (failure: Throwable) {
        postClangdProtocolError(failure.message ?: "Unable to queue LSP message")
      }
    }
    candidate.onmessageerror = {
      postClangdProtocolError("The clangd LSP port received an unreadable message")
    }
    candidate.start()
  }

  fun send(message: dynamic) {
    val target = port
    if (target == null || target == js("undefined")) {
      error("clangd produced an LSP message before its port was connected")
    }
    target.postMessage(prepareResponse(message))
  }

  fun close() {
    val target = port
    port = null
    pendingAstContexts.clear()
    if (target != null && target != js("undefined")) target.close()
  }

  /**
   * Carries cursor metadata beside clangd's AST request, then removes it before serializing the
   * LSP message. The raw tree is reduced in this worker, so a large recovery AST can never block
   * Monaco's UI thread after the request itself has completed.
  */
  private fun prepareRequest(message: dynamic): dynamic {
    if (message == null || message == js("undefined")) return message
    val params: dynamic = message["params"]
    if (message["method"] as? String == "\$/cancelRequest") {
      cppClangdMessageId(if (cppClangdDefined(params)) params["id"] else null)
        ?.let(pendingAstContexts::remove)
      return message
    }
    if (message["method"] as? String != "textDocument/ast" || !cppClangdDefined(params)) return message
    val metadata: dynamic = params[CPP_AST_CONTEXT_REQUEST_FIELD]
    if (!cppClangdDefined(metadata)) return message
    val source = metadata["source"] as? String ?: return message
    val line = (metadata["line"] as? Number)?.toInt() ?: return message
    val character = (metadata["character"] as? Number)?.toInt() ?: return message
    val id = cppClangdMessageId(message["id"]) ?: return message
    pendingAstContexts[id] = PendingAstContext(source, line, character)
    return js(
      """(message, field) => {
        const forwarded = { ...message, params: { ...message.params } };
        delete forwarded.params[field];
        return forwarded;
      }"""
    )(message, CPP_AST_CONTEXT_REQUEST_FIELD)
  }

  private fun prepareResponse(message: dynamic): dynamic {
    if (message == null || message == js("undefined")) return message
    val id = cppClangdMessageId(message["id"]) ?: return message
    val context = pendingAstContexts.remove(id) ?: return message
    val result: dynamic = message["result"]
    if (!cppClangdDefined(result)) return message
    val normalized = cppClangdAstContextDto(
      rawAst = result,
      source = context.source,
      cursorLine = context.line,
      cursorCharacter = context.character
    )
    normalized[CPP_NORMALIZED_AST_CONTEXT_FIELD] = true
    return js("(message, result) => Object.assign({}, message, { result: result })")(
      message,
      normalized
    )
  }

  private fun cppClangdMessageId(value: dynamic): String? = when (value) {
    is String -> "s:$value"
    is Number -> "n:${value.toDouble()}"
    else -> null
  }

  private fun cppClangdDefined(value: dynamic): Boolean =
    value != null && value != js("undefined")
}

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
    cppClangdWorkerScope.console.error(line)
  }
}

private suspend fun awaitDynamic(value: dynamic): dynamic =
  (js("(value) => Promise.resolve(value)")(value) as Promise<dynamic>).await()

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
