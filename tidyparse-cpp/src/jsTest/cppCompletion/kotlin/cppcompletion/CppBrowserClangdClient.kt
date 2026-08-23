package cppcompletion

import CPP_SEMANTIC_GRAPH_DEPTH
import CPP_SEMANTIC_GRAPH_LIMIT
import CPP_SEMANTIC_CALL_WITNESS_LIMIT
import CPP_SEMANTIC_CALL_WITNESS_MAX_ARITY
import CPP_SEMANTIC_EXPRESSION_WITNESS_LIMIT
import CPP_SEMANTIC_OPERATION_DEPTH
import CPP_SEMANTIC_OPERATION_LIMIT
import CppEditorStatementSnapshot
import cppEditorStatementSnapshot
import cppSemanticCompletionCharacter
import cppSemanticCompletionContextDto
import cppCompletionContextFromDto
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import kotlin.time.Duration.Companion.milliseconds

private const val CPP_BROWSER_CLANGD_WORKER =
  "/__cpp_completion/browser-clangd/worker.js"
private const val CPP_BROWSER_CLANGD_ROOT = "file:///home/web_user"
private const val CPP_BROWSER_CLANGD_LIMIT = 128
private const val CPP_BROWSER_CLANGD_TIMEOUT_MILLIS = 30_000L

internal fun cppBrowserSemanticCompletionParams(
  snapshot: CppEditorStatementSnapshot,
  line: Int,
  character: Int,
  uri: String = "file:///home/web_user/main.cpp",
  graphLimit: Int = CPP_SEMANTIC_GRAPH_LIMIT,
  graphDepth: Int = CPP_SEMANTIC_GRAPH_DEPTH,
  operationLimit: Int = CPP_SEMANTIC_OPERATION_LIMIT,
  operationDepth: Int = CPP_SEMANTIC_OPERATION_DEPTH,
  callWitnessLimit: Int = CPP_SEMANTIC_CALL_WITNESS_LIMIT,
  callWitnessMaxArity: Int = CPP_SEMANTIC_CALL_WITNESS_MAX_ARITY,
  expressionWitnessLimit: Int = CPP_SEMANTIC_EXPRESSION_WITNESS_LIMIT
): dynamic {
  val semanticCharacter = cppSemanticCompletionCharacter(snapshot)
  val receiverOperator = receiverOperator(snapshot.semanticPrefixText)
  val params = js("({})")
  params.textDocument = js("({})")
  params.textDocument.uri = uri
  params.position = js("({})")
  params.position.line = line
  params.position.character = semanticCharacter
  params.context = js("({})")
  if (receiverOperator == null) {
    params.context.triggerKind = 1
  } else {
    params.context.triggerKind = 2
    params.context.triggerCharacter = receiverOperator.last().toString()
  }
  params.scopePosition = js("({})")
  params.scopePosition.line = line
  params.scopePosition.character = snapshot.statementStartCharacter +
    snapshot.prefixText.takeWhile { it == ' ' || it == '\t' }.length
  params.graphLimit = graphLimit
  params.graphDepth = graphDepth
  params.operationLimit = operationLimit
  params.operationDepth = operationDepth
  params.callWitnessLimit = callWitnessLimit
  params.callWitnessMaxArity = callWitnessMaxArity
  params.expressionWitnessLimit = expressionWitnessLimit
  params.limit = CPP_BROWSER_CLANGD_LIMIT
  params.allScopes = snapshot.activeFragment?.kind == CppTokenKind.IDENTIFIER ||
    semanticCharacter != character
  return params
}

private fun receiverOperator(prefix: String): String? = prefix.trimEnd().let {
  when {
    it.endsWith("->") -> "->"
    it.endsWith("::") -> "::"
    it.endsWith('.') -> "."
    else -> null
  }
}

/** Exact leading preprocessing region clangd may cache as this document's preamble. */
internal fun cppBrowserPreamblePrefix(source: String): String {
  var offset = 0
  var inBlockComment = false
  var continuedDirective = false
  source.lineSequence().forEach { line ->
    val trimmed = line.trimStart()
    val preambleLine = inBlockComment || continuedDirective || trimmed.isEmpty() ||
      trimmed.startsWith("//") || trimmed.startsWith("/*") || trimmed.startsWith('#')
    if (!preambleLine) return source.substring(0, offset)
    offset += line.length + if (offset + line.length < source.length) 1 else 0
    if (inBlockComment) inBlockComment = !line.contains("*/")
    else if (trimmed.startsWith("/*")) inBlockComment = !trimmed.contains("*/")
    continuedDirective = (continuedDirective || trimmed.startsWith('#')) &&
      line.trimEnd().endsWith('\\')
  }
  return source
}

private data class CppBrowserClangdDocument(
  val uri: String,
  var source: String? = null,
  var version: Int = 0
)

/**
 * Tiny JSON-RPC client for the same patched clangd worker used by the C++ editor.
 *
 * The benchmark deliberately bypasses monaco-languageclient: it needs one document, one private
 * Sema request, and no editor providers. Keeping the transport here also makes the benchmark fail
 * loudly when it is accidentally run against a clangd without the structured endpoint.
 */
internal class CppBrowserClangdClient {
  private val serial = Mutex()
  private val pending = mutableMapOf<Int, CompletableDeferred<dynamic>>()
  private val diagnosticWaiters = mutableMapOf<Pair<String, Int>, CompletableDeferred<dynamic>>()
  private var worker: dynamic = null
  private var port: dynamic = null
  private var ready: CompletableDeferred<Unit>? = null
  private var nextId = 1
  private var nextDocumentId = 0
  private var initialized = false
  private val documentsByPreamble = mutableMapOf<String, CppBrowserClangdDocument>()

  suspend fun context(
    source: String,
    line: Int,
    character: Int,
    graphLimit: Int = CPP_SEMANTIC_GRAPH_LIMIT,
    graphDepth: Int = CPP_SEMANTIC_GRAPH_DEPTH
  ): CppCompletionContext = serial.withLock {
    start()
    val document = updateDocument(source)
    queryContext(document.uri, source, line, character, graphLimit, graphDepth)
  }

  internal suspend fun publishedDiagnostics(source: String): dynamic = serial.withLock {
    start()
    val document = updateDocument(source)
    val response = CompletableDeferred<dynamic>()
    val key = document.uri to document.version
    diagnosticWaiters[key] = response
    try {
      withTimeout(CPP_BROWSER_CLANGD_TIMEOUT_MILLIS.milliseconds) { response.await() }
    } finally {
      diagnosticWaiters.remove(key)
    }
  }

  internal suspend fun semanticResponse(
    source: String,
    line: Int,
    character: Int,
    graphLimit: Int = CPP_SEMANTIC_GRAPH_LIMIT,
    graphDepth: Int = CPP_SEMANTIC_GRAPH_DEPTH,
    operationLimit: Int = CPP_SEMANTIC_OPERATION_LIMIT,
    operationDepth: Int = CPP_SEMANTIC_OPERATION_DEPTH,
    callWitnessLimit: Int = CPP_SEMANTIC_CALL_WITNESS_LIMIT,
    callWitnessMaxArity: Int = CPP_SEMANTIC_CALL_WITNESS_MAX_ARITY,
    expressionWitnessLimit: Int = CPP_SEMANTIC_EXPRESSION_WITNESS_LIMIT
  ): dynamic = serial.withLock {
    start()
    val document = updateDocument(source)
    querySemanticResponse(
      document.uri,
      source,
      line,
      character,
      graphLimit,
      graphDepth,
      operationLimit,
      operationDepth,
      callWitnessLimit,
      callWitnessMaxArity,
      expressionWitnessLimit
    )
  }

  private suspend fun queryContext(
    uri: String,
    source: String,
    line: Int,
    character: Int,
    graphLimit: Int,
    graphDepth: Int
  ): CppCompletionContext {
    val semantic = querySemanticResponse(
      uri,
      source,
      line,
      character,
      graphLimit,
      graphDepth,
      CPP_SEMANTIC_OPERATION_LIMIT,
      CPP_SEMANTIC_OPERATION_DEPTH,
      CPP_SEMANTIC_CALL_WITNESS_LIMIT,
      CPP_SEMANTIC_CALL_WITNESS_MAX_ARITY,
      CPP_SEMANTIC_EXPRESSION_WITNESS_LIMIT
    )
    return cppCompletionContextFromDto(
      cppSemanticCompletionContextDto(semantic, requireNotNull(
        cppEditorStatementSnapshot(source, line, character)
      ))
    )
  }

  private suspend fun querySemanticResponse(
    uri: String,
    source: String,
    line: Int,
    character: Int,
    graphLimit: Int,
    graphDepth: Int,
    operationLimit: Int,
    operationDepth: Int,
    callWitnessLimit: Int,
    callWitnessMaxArity: Int,
    expressionWitnessLimit: Int
  ): dynamic {
    val snapshot = requireNotNull(cppEditorStatementSnapshot(source, line, character)) {
      "The benchmark cursor is not a completable C++ statement location"
    }
    val params = cppBrowserSemanticCompletionParams(
      snapshot,
      line,
      character,
      uri,
      graphLimit,
      graphDepth,
      operationLimit,
      operationDepth,
      callWitnessLimit,
      callWitnessMaxArity,
      expressionWitnessLimit
    )

    val semantic = request("tidyparse/semanticCompletion", params)
    val schemaVersion = (semantic?.schemaVersion as? Number)?.toInt() ?: -1
    check(schemaVersion == 2) {
      "Bundled clangd did not provide tidyparse/semanticCompletion schemaVersion 2"
    }
    return semantic
  }

  private suspend fun start() {
    if (initialized) return
    check(js("globalThis.crossOriginIsolated === true") as Boolean) {
      "The benchmark page is not cross-origin isolated; browser clangd cannot use shared memory"
    }

    val started = CompletableDeferred<Unit>()
    ready = started
    val clangdWorker = js("(url) => new Worker(url, { name: 'tidyparse-clangd' })")(
      CPP_BROWSER_CLANGD_WORKER
    )
    worker = clangdWorker
    val channel = js("new MessageChannel()")
    port = channel.port1
    channel.port1.onmessage = { event: dynamic -> accept(event.data) }
    channel.port1.onmessageerror = {
      fail(IllegalStateException("Browser clangd returned an unreadable LSP message"))
    }
    channel.port1.start()
    clangdWorker.onmessage = { event: dynamic ->
      when (event.data?.type as? String) {
        "ready" -> started.complete(Unit)
        "error" -> if (event.data?.fatal as? Boolean == true) {
          fail(IllegalStateException(event.data?.message as? String ?: "Browser clangd failed"))
        }
      }
    }
    clangdWorker.onerror = { event: dynamic ->
      fail(IllegalStateException(event?.message as? String ?: "Browser clangd worker failed"))
    }
    val connect = js("({ type: 'connect' })")
    connect.port = channel.port2
    clangdWorker.postMessage(connect, arrayOf(channel.port2))

    withTimeout(CPP_BROWSER_CLANGD_TIMEOUT_MILLIS.milliseconds) { started.await() }
    val initialize = js("({})")
    initialize.processId = null
    initialize.rootUri = CPP_BROWSER_CLANGD_ROOT
    initialize.capabilities = js("({})")
    initialize.capabilities.workspace = js("({ configuration: false })")
    initialize.capabilities.textDocument = js("({})")
    initialize.capabilities.textDocument.completion = js("({})")
    initialize.capabilities.textDocument.completion.completionItem =
      js("({ snippetSupport: false, labelDetailsSupport: true })")
    initialize.initializationOptions = js("({ clangdFileStatus: true })")
    initialize.workspaceFolders = arrayOf(
      js("({ uri: 'file:///home/web_user', name: 'cpp-completion-benchmark' })")
    )
    request("initialize", initialize)
    notify("initialized", js("({})"))
    initialized = true
  }

  private fun updateDocument(source: String): CppBrowserClangdDocument {
    val document = documentsByPreamble.getOrPut(cppBrowserPreamblePrefix(source)) {
      CppBrowserClangdDocument(
        uri = "$CPP_BROWSER_CLANGD_ROOT/completion-${nextDocumentId++}.cpp"
      )
    }
    if (source == document.source) return document
    document.source = source
    document.version++
    val params = js("({})")
    if (document.version == 1) {
      params.textDocument = js("({})")
      params.textDocument.uri = document.uri
      params.textDocument.languageId = "cpp"
      params.textDocument.version = document.version
      params.textDocument.text = source
      notify("textDocument/didOpen", params)
    } else {
      params.textDocument = js("({})")
      params.textDocument.uri = document.uri
      params.textDocument.version = document.version
      val change = js("({})")
      change.text = source
      params.contentChanges = arrayOf(change)
      notify("textDocument/didChange", params)
    }
    return document
  }

  private suspend fun request(method: String, params: dynamic): dynamic {
    val id = nextId++
    val response = CompletableDeferred<dynamic>()
    pending[id] = response
    val message = js("({ jsonrpc: '2.0' })")
    message.id = id
    message.method = method
    message.params = params
    port.postMessage(message)
    return try {
      withTimeout(CPP_BROWSER_CLANGD_TIMEOUT_MILLIS.milliseconds) { response.await() }
    } finally {
      pending.remove(id)
    }
  }

  private fun notify(method: String, params: dynamic) {
    val message = js("({ jsonrpc: '2.0' })")
    message.method = method
    message.params = params
    port.postMessage(message)
  }

  private fun accept(message: dynamic) {
    val id = (message?.id as? Number)?.toInt()
    val method = message?.method as? String
    if (method == "textDocument/publishDiagnostics") {
      val uri = message.params?.uri as? String ?: return
      val version = (message.params?.version as? Number)?.toInt() ?: return
      diagnosticWaiters.remove(uri to version)?.complete(message.params)
      return
    }
    if (id != null && method != null) {
      val response = js("({ jsonrpc: '2.0' })")
      response.id = message.id
      response.result = if (method == "workspace/configuration") js("[]") else null
      port.postMessage(response)
      return
    }
    if (id == null) return
    val deferred = pending.remove(id) ?: return
    val error = message.error
    if (error != null && error != js("undefined")) {
      deferred.completeExceptionally(
        IllegalStateException(error.message as? String ?: "Browser clangd request failed")
      )
    } else deferred.complete(message.result)
  }

  private fun fail(error: Throwable) {
    ready?.completeExceptionally(error)
    pending.values.forEach { it.completeExceptionally(error) }
    pending.clear()
    diagnosticWaiters.values.forEach { it.completeExceptionally(error) }
    diagnosticWaiters.clear()
  }

}
