import kotlinx.browser.window

private const val CLANGD_WORKSPACE_URI = "file:///home/web_user"
private const val CLANGD_CHANGE_DEBOUNCE_MS = 180
private const val CLANGD_REQUEST_TIMEOUT_MS = 15_000
private const val CLANGD_DISPOSE_TIMEOUT_MS = 750
private const val CLANGD_FILE_READ_TIMEOUT_MS = 15_000

enum class ClangdClientState {
  STARTING,
  LOADING,
  INITIALIZING,
  READY,
  BUSY,
  ERROR,
  STOPPED
}

data class ClangdPosition(
  val line: Int,
  val character: Int
)

data class ClangdRange(
  val start: ClangdPosition,
  val end: ClangdPosition
)

data class ClangdDiagnostic(
  val range: ClangdRange,
  val severity: Int?,
  val message: String,
  val source: String?,
  val code: String?
)

data class ClangdTextEdit(
  val range: ClangdRange,
  val newText: String
)

data class ClangdCompletion(
  val label: String,
  val detail: String?,
  val documentation: String?,
  val kind: Int?,
  val sortText: String?,
  val filterText: String?,
  val insertText: String?,
  val insertTextFormat: Int?,
  val textEdit: ClangdTextEdit?,
  val additionalTextEdits: List<ClangdTextEdit>,
  val commitCharacters: List<String>,
  val deprecated: Boolean,
  val preselect: Boolean
)

data class ClangdMarkupContent(
  val value: String,
  val kind: String,
  val language: String? = null
)

data class ClangdHover(
  val contents: List<ClangdMarkupContent>,
  val range: ClangdRange?
)

/**
 * A normalized LSP Location or LocationLink.
 *
 * [range] is the full target range while [selectionRange] is the narrower
 * range Monaco should reveal and select.
 */
data class ClangdNavigationTarget(
  val uri: String,
  val range: ClangdRange,
  val selectionRange: ClangdRange,
  val originSelectionRange: ClangdRange?
)

data class ClangdDocumentHighlight(
  val range: ClangdRange,
  val kind: Int?
)

data class ClangdDocumentSymbol(
  val uri: String?,
  val name: String,
  val detail: String?,
  val kind: Int,
  val tags: List<Int>,
  val containerName: String?,
  val range: ClangdRange,
  val selectionRange: ClangdRange,
  val children: List<ClangdDocumentSymbol>
)

data class ClangdSignatureParameter(
  val label: String?,
  val labelStart: Int?,
  val labelEnd: Int?,
  val documentation: ClangdMarkupContent?
)

data class ClangdSignature(
  val label: String,
  val documentation: ClangdMarkupContent?,
  val parameters: List<ClangdSignatureParameter>,
  val activeParameter: Int?
)

data class ClangdSignatureHelp(
  val signatures: List<ClangdSignature>,
  val activeSignature: Int?,
  val activeParameter: Int?
)

data class ClangdSemanticTokensLegend(
  val tokenTypes: List<String>,
  val tokenModifiers: List<String>
)

data class ClangdSemanticTokens(
  val resultId: String?,
  val data: List<Int>
)

data class ClangdRequestFailure(
  val code: Int?,
  val message: String
)

data class ClangdVirtualFile(
  val uri: String,
  val path: String,
  val text: String
)

class ClangdRequestHandle internal constructor(
  private val cancelRequest: () -> Unit
) {
  private var cancelled = false

  fun cancel() {
    if (cancelled) return
    cancelled = true
    cancelRequest()
  }
}

/**
 * A deliberately small JSON-RPC/LSP client for the in-browser clangd worker.
 *
 * [requestCompletion] takes zero-based UTF-16 LSP coordinates. This matches
 * JavaScript strings and textarea selection offsets.
 */
class JSClangdClient(
  private val onStatus: (ClangdClientState, String) -> Unit = { _, _ -> },
  private val onProgress: (loaded: Int, total: Int) -> Unit = { _, _ -> },
  private val onDiagnostics: (uri: String, version: Int?, diagnostics: List<ClangdDiagnostic>) -> Unit =
    { _, _, _ -> },
  private val onSemanticTokensRefresh: () -> Unit = {}
) {
  private class PendingRequest(
    val timeout: Int,
    val callback: (result: dynamic, error: dynamic) -> Unit
  )

  private class PendingFileRead(
    val uri: String,
    val timeout: Int,
    val callback: (file: ClangdVirtualFile?, error: String?) -> Unit
  )

  private var worker: dynamic = null
  private var nextRequestId = 1
  private val pendingRequests = mutableMapOf<Int, PendingRequest>()
  private var nextFileReadId = 1
  private val pendingFileReads = mutableMapOf<Int, PendingFileRead>()

  private var workspaceUri = CLANGD_WORKSPACE_URI
  private var fileName = "main.cpp"
  private var languageId = "cpp"
  private var documentUri = "$CLANGD_WORKSPACE_URI/main.cpp"
  private var documentText = ""
  private var lastSentText = ""
  private var documentVersion = 1
  private var documentEpoch = 0

  private var initialized = false
  private var opened = false
  private var disposed = false
  private var changeTimer: Int? = null
  private var activeCompletionRequest: Int? = null
  private var completionTriggers = emptySet<String>()
  private var signatureHelpTriggers = emptyList<String>()
  private var signatureHelpRetriggers = emptyList<String>()
  private var semanticLegend: ClangdSemanticTokensLegend? = null

  fun completionTriggerCharacters(): List<String> = completionTriggers.sorted()

  fun signatureHelpTriggerCharacters(): List<String> = signatureHelpTriggers

  fun signatureHelpRetriggerCharacters(): List<String> = signatureHelpRetriggers

  fun semanticTokensLegend(): ClangdSemanticTokensLegend? =
    semanticLegend?.let {
      ClangdSemanticTokensLegend(
        tokenTypes = it.tokenTypes.toList(),
        tokenModifiers = it.tokenModifiers.toList()
      )
    }

  fun start(fileName: String, languageId: String, text: String) {
    if (disposed) {
      reportStatus(ClangdClientState.ERROR, "This clangd client has already been disposed")
      return
    }
    if (clangdDefined(worker)) {
      changeDocument(fileName, languageId, text)
      return
    }

    setDocument(fileName, languageId, text)
    semanticLegend = null
    reportStatus(ClangdClientState.STARTING, "Starting clangd…")

    try {
      val created = createClangdWorker()
      worker = created
      created.onmessage = { event: dynamic ->
        handleWorkerMessage(event.data)
      }
      created.onerror = { event: dynamic ->
        val message = event.message as? String ?: "The clangd worker failed"
        handleWorkerFailure(message)
      }
      created.onmessageerror = { _: dynamic ->
        handleWorkerFailure("The clangd worker returned an unreadable message")
      }
      sendConfigure()
    } catch (failure: Throwable) {
      handleWorkerFailure(failure.message ?: "Unable to start the clangd worker")
    }
  }

  fun changeDocument(fileName: String, languageId: String, text: String) {
    if (disposed) return

    cancelChangeTimer()
    cancelCompletion()
    val previousUri = documentUri
    if (opened) closeDocument(previousUri)

    setDocument(fileName, languageId, text)
    sendConfigure()
    emitDiagnostics(documentUri, null, emptyList())

    if (initialized) {
      openDocument()
      reportStatus(ClangdClientState.BUSY, "clangd is analyzing ${this.fileName}…")
    }
  }

  fun didChange(text: String) {
    if (disposed || text == documentText) return

    documentText = text
    documentEpoch++
    cancelCompletion()
    emitDiagnostics(documentUri, null, emptyList())

    if (!opened) return
    cancelChangeTimer()
    changeTimer = window.setTimeout({
      changeTimer = null
      flushChanges()
    }, CLANGD_CHANGE_DEBOUNCE_MS)
    reportStatus(ClangdClientState.BUSY, "clangd is analyzing…")
  }

  fun requestCompletion(
    line: Int,
    column: Int,
    callback: (List<ClangdCompletion>) -> Unit
  ): ClangdRequestHandle? {
    if (disposed || !initialized || !opened) {
      callbackSafely(callback, emptyList())
      return null
    }

    flushChanges()
    cancelCompletion()

    val requestedEpoch = documentEpoch
    val requestedUri = documentUri
    val position = clampedPosition(documentText, line, column)
    val textDocument = clangdObject()
    textDocument.uri = requestedUri
    val params = clangdObject()
    params.textDocument = textDocument
    params.position = lspPosition(position)

    val trigger = characterBefore(documentText, position)
      ?.takeIf { it in completionTriggers }
    val context = clangdObject()
    context.triggerKind = if (trigger == null) 1 else 2
    if (trigger != null) context.triggerCharacter = trigger
    params.context = context

    var requestId = 0
    requestId = sendRequest(
      method = "textDocument/completion",
      params = params,
      timeoutMs = CLANGD_REQUEST_TIMEOUT_MS
    ) { result, error ->
      if (activeCompletionRequest != requestId) return@sendRequest
      activeCompletionRequest = null
      if (
        disposed ||
        requestedEpoch != documentEpoch ||
        requestedUri != documentUri
      ) {
        callbackSafely(callback, emptyList())
        return@sendRequest
      }
      val completions =
        if (clangdDefined(error)) emptyList()
        else parseCompletions(result)
      callbackSafely(callback, completions)
    }
    activeCompletionRequest = requestId.takeIf { it > 0 }
    if (requestId <= 0) {
      callbackSafely(callback, emptyList())
      return null
    }
    return ClangdRequestHandle { cancelCompletion(requestId) }
  }

  fun requestHover(
    line: Int,
    column: Int,
    callback: (ClangdHover?) -> Unit
  ): ClangdRequestHandle? =
    requestAtPosition("textDocument/hover", line, column) { result, error ->
      deliver(callback, if (clangdDefined(error)) null else parseHover(result))
    }

  fun requestDefinition(
    line: Int,
    column: Int,
    callback: (List<ClangdNavigationTarget>) -> Unit
  ): ClangdRequestHandle? =
    requestNavigation("textDocument/definition", line, column, callback)

  fun requestDeclaration(
    line: Int,
    column: Int,
    callback: (List<ClangdNavigationTarget>) -> Unit
  ): ClangdRequestHandle? =
    requestNavigation("textDocument/declaration", line, column, callback)

  fun requestImplementation(
    line: Int,
    column: Int,
    callback: (List<ClangdNavigationTarget>) -> Unit
  ): ClangdRequestHandle? =
    requestNavigation("textDocument/implementation", line, column, callback)

  fun requestReferences(
    line: Int,
    column: Int,
    includeDeclaration: Boolean = true,
    callback: (List<ClangdNavigationTarget>) -> Unit
  ): ClangdRequestHandle? =
    requestAtPosition(
      method = "textDocument/references",
      line = line,
      column = column,
      configure = { params ->
        val context = clangdObject()
        context.includeDeclaration = includeDeclaration
        params.context = context
      }
    ) { result, error ->
      deliver(
        callback,
        if (clangdDefined(error)) emptyList()
        else parseNavigationTargets(result)
      )
    }

  fun requestDocumentHighlights(
    line: Int,
    column: Int,
    callback: (List<ClangdDocumentHighlight>) -> Unit
  ): ClangdRequestHandle? =
    requestAtPosition("textDocument/documentHighlight", line, column) { result, error ->
      deliver(
        callback,
        if (clangdDefined(error)) emptyList()
        else parseDocumentHighlights(result)
      )
    }

  fun requestDocumentSymbols(
    callback: (List<ClangdDocumentSymbol>) -> Unit
  ): ClangdRequestHandle? =
    requestForCurrentDocument("textDocument/documentSymbol") { result, error ->
      deliver(
        callback,
        if (clangdDefined(error)) emptyList()
        else parseDocumentSymbols(result)
      )
    }

  fun requestSignatureHelp(
    line: Int,
    column: Int,
    triggerKind: Int = 1,
    triggerCharacter: String? = null,
    isRetrigger: Boolean = false,
    callback: (ClangdSignatureHelp?) -> Unit
  ): ClangdRequestHandle? =
    requestAtPosition(
      method = "textDocument/signatureHelp",
      line = line,
      column = column,
      configure = { params ->
        val context = clangdObject()
        context.triggerKind = triggerKind.coerceIn(1, 3)
        if (!triggerCharacter.isNullOrEmpty()) context.triggerCharacter = triggerCharacter
        context.isRetrigger = isRetrigger
        params.context = context
      }
    ) { result, error ->
      deliver(
        callback,
        if (clangdDefined(error)) null
        else parseSignatureHelp(result)
      )
    }

  fun requestSemanticTokens(
    callback: (tokens: ClangdSemanticTokens?, failure: ClangdRequestFailure?) -> Unit
  ): ClangdRequestHandle? {
    if (semanticLegend == null) {
      deliverSemanticTokens(callback, null, null)
      return null
    }
    return requestForCurrentDocument("textDocument/semanticTokens/full") { result, error ->
      if (clangdDefined(error)) {
        deliverSemanticTokens(
          callback,
          null,
          ClangdRequestFailure(
            code = clangdInt(error.code),
            message = error.message as? String ?: "clangd semantic token request failed"
          )
        )
      } else {
        val tokens = parseSemanticTokens(result)
        val failure =
          if (clangdDefined(result) && tokens == null) {
            ClangdRequestFailure(null, "clangd returned malformed semantic tokens")
          } else {
            null
          }
        deliverSemanticTokens(callback, tokens, failure)
      }
    }
  }

  fun readVirtualFile(
    uri: String,
    callback: (file: ClangdVirtualFile?, error: String?) -> Unit
  ): ClangdRequestHandle? {
    if (disposed || !initialized || !clangdDefined(worker)) {
      deliverFileRead(callback, null, "clangd virtual filesystem is not ready")
      return null
    }
    if (uri.isBlank()) {
      deliverFileRead(callback, null, "A virtual file URI is required")
      return null
    }

    val requestId = nextFileReadId++
    val timeout = window.setTimeout({
      val pending = pendingFileReads.remove(requestId) ?: return@setTimeout
      deliverFileRead(
        pending.callback,
        null,
        "Reading ${pending.uri} timed out"
      )
    }, CLANGD_FILE_READ_TIMEOUT_MS)
    pendingFileReads[requestId] = PendingFileRead(uri, timeout, callback)

    val request = clangdObject()
    request.type = "readFile"
    request.id = requestId
    request.uri = uri
    return try {
      postToWorker(request)
      ClangdRequestHandle { cancelFileRead(requestId) }
    } catch (failure: Throwable) {
      pendingFileReads.remove(requestId)
      window.clearTimeout(timeout)
      deliverFileRead(
        callback,
        null,
        failure.message ?: "Unable to request $uri from clangd"
      )
      null
    }
  }

  fun dispose() {
    if (disposed) return
    disposed = true
    cancelChangeTimer()
    cancelCompletion()
    clearPendingRequests()
    clearPendingFileReads("clangd client was disposed")

    if (!clangdDefined(worker)) {
      reportStatus(ClangdClientState.STOPPED, "clangd stopped")
      return
    }

    if (opened) closeDocument(documentUri)
    if (!initialized) {
      terminateWorker()
      return
    }

    sendRequest(
      method = "shutdown",
      params = null,
      timeoutMs = CLANGD_DISPOSE_TIMEOUT_MS
    ) { _, _ ->
      if (clangdDefined(worker)) sendNotification("exit", null)
      terminateWorker()
    }
  }

  private fun setDocument(fileName: String, languageId: String, text: String) {
    this.fileName = safeFileName(fileName)
    this.languageId = normalizeLanguageId(languageId)
    documentUri = "${workspaceUri.trimEnd('/')}/${encodeUriSegment(this.fileName)}"
    documentText = text
    lastSentText = text
    documentVersion = 1
    documentEpoch++
  }

  private fun sendConfigure() {
    if (!clangdDefined(worker)) return
    val configure = clangdObject()
    configure.type = "configure"
    configure.language = if (languageId == "c") "c" else "cpp"
    postToWorker(configure)
  }

  private fun beginInitialize(message: dynamic) {
    if (disposed || initialized) return

    val announcedWorkspace = message.workspaceUri as? String
    if (!announcedWorkspace.isNullOrBlank() && announcedWorkspace != workspaceUri) {
      workspaceUri = announcedWorkspace.trimEnd('/')
      documentUri = "$workspaceUri/${encodeUriSegment(fileName)}"
    }

    reportStatus(ClangdClientState.INITIALIZING, "Initializing clangd…")
    val params = initializeParams()
    sendRequest(
      method = "initialize",
      params = params,
      timeoutMs = CLANGD_REQUEST_TIMEOUT_MS
    ) { result, error ->
      if (disposed) return@sendRequest
      if (clangdDefined(error)) {
        val messageText = error.message as? String ?: "clangd initialization failed"
        handleWorkerFailure(messageText)
        return@sendRequest
      }

      val capabilities = result?.capabilities
      completionTriggers = parseStringList(capabilities?.completionProvider?.triggerCharacters).toSet()
      signatureHelpTriggers =
        parseStringList(capabilities?.signatureHelpProvider?.triggerCharacters)
      signatureHelpRetriggers =
        parseStringList(capabilities?.signatureHelpProvider?.retriggerCharacters)
      semanticLegend =
        parseSemanticTokensLegend(capabilities?.semanticTokensProvider)
      val positionEncoding = capabilities?.positionEncoding as? String
      if (positionEncoding != null && !positionEncoding.equals("utf-16", ignoreCase = true)) {
        handleWorkerFailure("clangd selected unsupported position encoding $positionEncoding")
        return@sendRequest
      }

      initialized = true
      sendNotification("initialized", clangdObject())
      openDocument()
      reportStatus(ClangdClientState.READY, "clangd ready")
    }
  }

  private fun initializeParams(): dynamic {
    val workspaceFolder = clangdObject()
    workspaceFolder.uri = workspaceUri
    workspaceFolder.name = "workspace"

    val synchronization = clangdObject()
    synchronization.dynamicRegistration = false
    synchronization.willSave = false
    synchronization.willSaveWaitUntil = false
    synchronization.didSave = false

    val completionItem = clangdObject()
    completionItem.snippetSupport = true
    completionItem.commitCharactersSupport = true
    completionItem.documentationFormat = arrayOf("markdown", "plaintext")
    completionItem.deprecatedSupport = true
    completionItem.preselectSupport = true

    val completion = clangdObject()
    completion.dynamicRegistration = false
    completion.contextSupport = true
    completion.completionItem = completionItem

    val hover = clangdObject()
    hover.dynamicRegistration = false
    hover.contentFormat = arrayOf("markdown", "plaintext")

    val signatureInformation = clangdObject()
    signatureInformation.documentationFormat = arrayOf("markdown", "plaintext")
    val parameterInformation = clangdObject()
    parameterInformation.labelOffsetSupport = true
    signatureInformation.parameterInformation = parameterInformation
    val signatureHelp = clangdObject()
    signatureHelp.dynamicRegistration = false
    signatureHelp.contextSupport = true
    signatureHelp.signatureInformation = signatureInformation

    val declaration = clangdObject()
    declaration.dynamicRegistration = false
    declaration.linkSupport = true

    val definition = clangdObject()
    definition.dynamicRegistration = false
    definition.linkSupport = true

    val implementation = clangdObject()
    implementation.dynamicRegistration = false
    implementation.linkSupport = true

    val references = clangdObject()
    references.dynamicRegistration = false

    val documentHighlight = clangdObject()
    documentHighlight.dynamicRegistration = false

    val symbolKind = clangdObject()
    symbolKind.valueSet = (1..26).toList().toTypedArray()
    val documentSymbolTagSupport = clangdObject()
    documentSymbolTagSupport.valueSet = arrayOf(1)
    val documentSymbol = clangdObject()
    documentSymbol.dynamicRegistration = false
    documentSymbol.symbolKind = symbolKind
    documentSymbol.hierarchicalDocumentSymbolSupport = true
    documentSymbol.tagSupport = documentSymbolTagSupport
    documentSymbol.labelSupport = true

    val semanticTokenRequests = clangdObject()
    semanticTokenRequests.range = false
    semanticTokenRequests.full = true
    val semanticTokens = clangdObject()
    semanticTokens.dynamicRegistration = false
    semanticTokens.requests = semanticTokenRequests
    semanticTokens.tokenTypes = arrayOf(
      "namespace",
      "type",
      "class",
      "enum",
      "interface",
      "struct",
      "typeParameter",
      "parameter",
      "variable",
      "property",
      "enumMember",
      "event",
      "function",
      "method",
      "macro",
      "keyword",
      "modifier",
      "comment",
      "string",
      "number",
      "regexp",
      "operator",
      "decorator"
    )
    semanticTokens.tokenModifiers = arrayOf(
      "declaration",
      "definition",
      "readonly",
      "static",
      "deprecated",
      "abstract",
      "async",
      "modification",
      "documentation",
      "defaultLibrary"
    )
    semanticTokens.formats = arrayOf("relative")
    semanticTokens.overlappingTokenSupport = false
    semanticTokens.multilineTokenSupport = false
    semanticTokens.serverCancelSupport = false
    semanticTokens.augmentsSyntaxTokens = true

    val diagnostics = clangdObject()
    diagnostics.relatedInformation = true
    diagnostics.versionSupport = true
    diagnostics.codeDescriptionSupport = true
    val tagSupport = clangdObject()
    tagSupport.valueSet = arrayOf(1, 2)
    diagnostics.tagSupport = tagSupport

    val textDocument = clangdObject()
    textDocument.synchronization = synchronization
    textDocument.completion = completion
    textDocument.hover = hover
    textDocument.signatureHelp = signatureHelp
    textDocument.declaration = declaration
    textDocument.definition = definition
    textDocument.implementation = implementation
    textDocument.references = references
    textDocument.documentHighlight = documentHighlight
    textDocument.documentSymbol = documentSymbol
    textDocument.semanticTokens = semanticTokens
    textDocument.publishDiagnostics = diagnostics

    val workspace = clangdObject()
    workspace.configuration = true
    workspace.workspaceFolders = true
    val workspaceSemanticTokens = clangdObject()
    workspaceSemanticTokens.refreshSupport = true
    workspace.semanticTokens = workspaceSemanticTokens

    val windowCapabilities = clangdObject()
    windowCapabilities.workDoneProgress = true

    val general = clangdObject()
    general.positionEncodings = arrayOf("utf-16")

    val capabilities = clangdObject()
    capabilities.workspace = workspace
    capabilities.textDocument = textDocument
    capabilities.window = windowCapabilities
    capabilities.general = general

    val clientInfo = clangdObject()
    clientInfo.name = "tidyparse-web"
    clientInfo.version = "1"

    val initializationOptions = clangdObject()
    initializationOptions.clangdFileStatus = true

    val params = clangdObject()
    params.processId = null
    params.clientInfo = clientInfo
    params.locale = "en"
    params.rootUri = workspaceUri
    params.capabilities = capabilities
    params.initializationOptions = initializationOptions
    params.workspaceFolders = arrayOf(workspaceFolder)
    params.trace = "off"
    return params
  }

  private fun openDocument() {
    if (!initialized || disposed) return

    val textDocument = clangdObject()
    textDocument.uri = documentUri
    textDocument.languageId = languageId
    textDocument.version = documentVersion
    textDocument.text = documentText
    val params = clangdObject()
    params.textDocument = textDocument
    sendNotification("textDocument/didOpen", params)

    opened = true
    lastSentText = documentText
  }

  private fun closeDocument(uri: String) {
    if (!opened) return
    val textDocument = clangdObject()
    textDocument.uri = uri
    val params = clangdObject()
    params.textDocument = textDocument
    sendNotification("textDocument/didClose", params)
    opened = false
  }

  private fun flushChanges() {
    cancelChangeTimer()
    if (!opened || documentText == lastSentText || disposed) return

    documentVersion++
    val textDocument = clangdObject()
    textDocument.uri = documentUri
    textDocument.version = documentVersion
    val change = clangdObject()
    change.text = documentText
    val params = clangdObject()
    params.textDocument = textDocument
    params.contentChanges = arrayOf(change)
    sendNotification("textDocument/didChange", params)
    lastSentText = documentText
  }

  private fun requestNavigation(
    method: String,
    line: Int,
    column: Int,
    callback: (List<ClangdNavigationTarget>) -> Unit
  ): ClangdRequestHandle? =
    requestAtPosition(method, line, column) { result, error ->
      deliver(
        callback,
        if (clangdDefined(error)) emptyList()
        else parseNavigationTargets(result)
      )
    }

  private fun requestAtPosition(
    method: String,
    line: Int,
    column: Int,
    configure: (dynamic) -> Unit = {},
    callback: (result: dynamic, error: dynamic) -> Unit
  ): ClangdRequestHandle? {
    if (disposed || !initialized || !opened) {
      deliverRaw(callback, null, clangdRequestError(-32002, "clangd is not ready"))
      return null
    }

    flushChanges()
    val requestedEpoch = documentEpoch
    val requestedUri = documentUri
    val position = clampedPosition(documentText, line, column)
    val textDocument = clangdObject()
    textDocument.uri = requestedUri
    val params = clangdObject()
    params.textDocument = textDocument
    params.position = lspPosition(position)
    configure(params)

    val requestId = sendRequest(
      method = method,
      params = params,
      timeoutMs = CLANGD_REQUEST_TIMEOUT_MS
    ) { result, error ->
      if (
        disposed ||
        requestedEpoch != documentEpoch ||
        requestedUri != documentUri
      ) {
        deliverRaw(
          callback,
          null,
          clangdRequestError(-32801, "$method result is stale")
        )
      } else {
        deliverRaw(callback, result, error)
      }
    }
    if (requestId <= 0) {
      deliverRaw(callback, null, clangdRequestError(-32003, "Unable to send $method"))
      return null
    }
    return ClangdRequestHandle { cancelPendingRequest(requestId) }
  }

  private fun requestForCurrentDocument(
    method: String,
    callback: (result: dynamic, error: dynamic) -> Unit
  ): ClangdRequestHandle? {
    if (disposed || !initialized || !opened) {
      deliverRaw(callback, null, clangdRequestError(-32002, "clangd is not ready"))
      return null
    }

    flushChanges()
    val requestedEpoch = documentEpoch
    val requestedUri = documentUri
    val textDocument = clangdObject()
    textDocument.uri = requestedUri
    val params = clangdObject()
    params.textDocument = textDocument

    val requestId = sendRequest(
      method = method,
      params = params,
      timeoutMs = CLANGD_REQUEST_TIMEOUT_MS
    ) { result, error ->
      if (
        disposed ||
        requestedEpoch != documentEpoch ||
        requestedUri != documentUri
      ) {
        deliverRaw(
          callback,
          null,
          clangdRequestError(-32801, "$method result is stale")
        )
      } else {
        deliverRaw(callback, result, error)
      }
    }
    if (requestId <= 0) {
      deliverRaw(callback, null, clangdRequestError(-32003, "Unable to send $method"))
      return null
    }
    return ClangdRequestHandle { cancelPendingRequest(requestId) }
  }

  private fun cancelPendingRequest(id: Int) {
    val pending = pendingRequests.remove(id) ?: return
    window.clearTimeout(pending.timeout)
    if (clangdDefined(worker) && initialized) {
      val params = clangdObject()
      params.id = id
      sendNotification("$/cancelRequest", params)
    }
    deliverRaw(
      pending.callback,
      null,
      clangdRequestError(-32800, "Request cancelled")
    )
  }

  private fun handleWorkerMessage(data: dynamic) {
    if (!clangdDefined(data)) return
    val type = data.type as? String
    when (type) {
      "progress" -> {
        val loaded = clangdInt(data.loaded) ?: clangdInt(data.value) ?: 0
        val total = clangdInt(data.total) ?: clangdInt(data.max) ?: 0
        try {
          onProgress(loaded, total)
        } catch (_: Throwable) {
        }
        reportStatus(
          ClangdClientState.LOADING,
          if (total > 0) "Loading clangd… ${(loaded.toLong() * 100 / total).coerceIn(0, 100)}%"
          else "Loading clangd…"
        )
      }

      "status" -> {
        val status = data.status as? String
        val state = when (status) {
          "loading" -> ClangdClientState.LOADING
          "starting" -> ClangdClientState.STARTING
          else -> ClangdClientState.BUSY
        }
        reportStatus(state, data.message as? String ?: "clangd $status")
      }

      "ready" -> beginInitialize(data)
      "error" -> handleWorkerFailure(data.message as? String ?: "clangd failed")
      "file", "fileError" -> handleVirtualFileResponse(data)
      "lsp" -> handleLspMessage(
        when {
          clangdDefined(data.message) -> data.message
          clangdDefined(data.payload) -> data.payload
          else -> null
        }
      )

      else -> {
        if (data.jsonrpc as? String == "2.0") handleLspMessage(data)
      }
    }
  }

  private fun handleVirtualFileResponse(data: dynamic) {
    val requestId = clangdInt(data.id) ?: return
    val pending = pendingFileReads.remove(requestId) ?: return
    window.clearTimeout(pending.timeout)

    if ((data.type as? String) == "fileError") {
      deliverFileRead(
        pending.callback,
        null,
        data.message as? String ?: "Unable to read ${pending.uri}"
      )
      return
    }

    val text = data.text as? String
    if (text == null) {
      deliverFileRead(
        pending.callback,
        null,
        "clangd returned an unreadable virtual file"
      )
      return
    }

    deliverFileRead(
      pending.callback,
      ClangdVirtualFile(
        uri = data.uri as? String ?: pending.uri,
        path = data.path as? String ?: "",
        text = text
      ),
      null
    )
  }

  private fun handleLspMessage(message: dynamic) {
    if (!clangdDefined(message)) return
    val method = message.method as? String
    if (method != null) {
      if (clangdHasOwn(message, "id")) {
        handleServerRequest(message.id, method, message.params)
      } else {
        handleNotification(method, message.params)
      }
      return
    }

    val id = clangdInt(message.id) ?: return
    val pending = pendingRequests.remove(id) ?: return
    window.clearTimeout(pending.timeout)
    try {
      pending.callback(
        if (clangdHasOwn(message, "result")) message.result else null,
        if (clangdHasOwn(message, "error")) message.error else null
      )
    } catch (_: Throwable) {
    }
  }

  private fun handleServerRequest(id: dynamic, method: String, params: dynamic) {
    when (method) {
      "workspace/configuration" -> {
        val count = clangdInt(params?.items?.length) ?: 0
        respond(id, clangdNullArray(count))
      }

      "workspace/workspaceFolders" -> {
        val folder = clangdObject()
        folder.uri = workspaceUri
        folder.name = "workspace"
        respond(id, arrayOf(folder))
      }

      "workspace/applyEdit" -> {
        val result = clangdObject()
        result.applied = false
        result.failureReason = "Workspace edits are not supported by this playground"
        respond(id, result)
      }

      "window/showDocument" -> {
        val result = clangdObject()
        result.success = false
        respond(id, result)
      }

      "window/workDoneProgress/create",
      "window/showMessageRequest",
      "client/registerCapability",
      "client/unregisterCapability",
      "workspace/inlayHint/refresh",
      "workspace/inlineValue/refresh",
      "workspace/codeLens/refresh",
      "workspace/diagnostic/refresh",
      "workspace/foldingRange/refresh" -> respond(id, null)

      "workspace/semanticTokens/refresh" -> {
        try {
          onSemanticTokensRefresh()
        } catch (_: Throwable) {
        }
        respond(id, null)
      }

      else -> respondError(id, -32601, "Method not supported: $method")
    }
  }

  private fun handleNotification(method: String, params: dynamic) {
    when (method) {
      "textDocument/publishDiagnostics" -> publishDiagnostics(params)

      "textDocument/clangd.fileStatus" -> {
        if ((params?.uri as? String) != documentUri) return
        val state = params.state as? String ?: return
        if (state.equals("idle", ignoreCase = true)) {
          reportStatus(ClangdClientState.READY, "clangd ready")
        } else {
          reportStatus(ClangdClientState.BUSY, state)
        }
      }

      "$/progress" -> {
        val kind = params?.value?.kind as? String
        when (kind) {
          "begin" -> reportStatus(
            ClangdClientState.BUSY,
            params.value.title as? String ?: "clangd is working…"
          )
          "end" -> reportStatus(ClangdClientState.READY, "clangd ready")
        }
      }

      "window/showMessage" -> {
        val message = params?.message as? String ?: return
        if (clangdInt(params.type) == 1) {
          reportStatus(ClangdClientState.ERROR, message)
        }
      }

      "window/logMessage",
      "telemetry/event",
      "$/logTrace",
      "$/cancelRequest" -> Unit
    }
  }

  private fun publishDiagnostics(params: dynamic) {
    val uri = params?.uri as? String ?: return
    if (uri != documentUri || documentText != lastSentText) return

    val version = clangdInt(params.version)
    if (version != null && version != documentVersion) return

    val values = params.diagnostics
    val count = clangdInt(values?.length) ?: 0
    val diagnostics = ArrayList<ClangdDiagnostic>(count)
    for (index in 0 until count) {
      parseDiagnostic(values[index])?.let(diagnostics::add)
    }
    emitDiagnostics(uri, version, diagnostics)
  }

  private fun parseDiagnostic(value: dynamic): ClangdDiagnostic? {
    if (!clangdDefined(value)) return null
    val range = parseRange(value.range) ?: return null
    val message = value.message as? String ?: return null
    val code = when (val raw = value.code) {
      is String -> raw
      is Number -> raw.toString()
      else -> null
    }
    return ClangdDiagnostic(
      range = range,
      severity = clangdInt(value.severity),
      message = message,
      source = value.source as? String,
      code = code
    )
  }

  private fun parseCompletions(result: dynamic): List<ClangdCompletion> {
    if (!clangdDefined(result)) return emptyList()
    val items =
      if (clangdIsArray(result)) result
      else result.items
    val count = clangdInt(items?.length) ?: return emptyList()
    val completions = ArrayList<ClangdCompletion>(count)
    for (index in 0 until count) {
      parseCompletion(items[index])?.let(completions::add)
    }
    return completions
  }

  private fun parseCompletion(item: dynamic): ClangdCompletion? {
    if (!clangdDefined(item)) return null
    val label = item.label as? String ?: return null
    val documentation = when (val raw = item.documentation) {
      is String -> raw
      else -> raw?.value as? String
    }
    val additional = parseTextEdits(item.additionalTextEdits)
    return ClangdCompletion(
      label = label,
      detail = item.detail as? String,
      documentation = documentation,
      kind = clangdInt(item.kind),
      sortText = item.sortText as? String,
      filterText = item.filterText as? String,
      insertText = item.insertText as? String,
      insertTextFormat = clangdInt(item.insertTextFormat),
      textEdit = parseTextEdit(item.textEdit),
      additionalTextEdits = additional,
      commitCharacters = parseStringList(item.commitCharacters),
      deprecated = item.deprecated as? Boolean ?: false,
      preselect = item.preselect as? Boolean ?: false
    )
  }

  private fun parseHover(value: dynamic): ClangdHover? {
    if (!clangdDefined(value)) return null
    val contents = parseMarkupContents(value.contents)
    if (contents.isEmpty()) return null
    return ClangdHover(
      contents = contents,
      range = parseRange(value.range)
    )
  }

  private fun parseMarkupContents(value: dynamic): List<ClangdMarkupContent> {
    if (!clangdDefined(value)) return emptyList()
    if (!clangdIsArray(value)) {
      return listOfNotNull(parseMarkupContent(value))
    }
    val count = clangdInt(value.length) ?: return emptyList()
    return (0 until count).mapNotNull { parseMarkupContent(value[it]) }
  }

  private fun parseMarkupContent(value: dynamic): ClangdMarkupContent? {
    if (!clangdDefined(value)) return null
    if (value is String) {
      return ClangdMarkupContent(value = value, kind = "markdown")
    }

    val text = value.value as? String ?: return null
    val language = value.language as? String
    val kind = when {
      language != null -> "marked"
      (value.kind as? String).equals("plaintext", ignoreCase = true) -> "plaintext"
      else -> "markdown"
    }
    return ClangdMarkupContent(value = text, kind = kind, language = language)
  }

  private fun parseNavigationTargets(value: dynamic): List<ClangdNavigationTarget> {
    if (!clangdDefined(value)) return emptyList()
    if (!clangdIsArray(value)) {
      return listOfNotNull(parseNavigationTarget(value))
    }

    val count = clangdInt(value.length) ?: return emptyList()
    return (0 until count).mapNotNull { parseNavigationTarget(value[it]) }
  }

  private fun parseNavigationTarget(value: dynamic): ClangdNavigationTarget? {
    if (!clangdDefined(value)) return null
    val isLink = clangdDefined(value.targetUri)
    val uri =
      if (isLink) value.targetUri as? String
      else value.uri as? String
    val range =
      if (isLink) parseRange(value.targetRange)
      else parseRange(value.range)
    if (uri == null || range == null) return null

    return ClangdNavigationTarget(
      uri = uri,
      range = range,
      selectionRange =
        if (isLink) parseRange(value.targetSelectionRange) ?: range
        else range,
      originSelectionRange = parseRange(value.originSelectionRange)
    )
  }

  private fun parseDocumentHighlights(value: dynamic): List<ClangdDocumentHighlight> {
    val count = clangdInt(value?.length) ?: return emptyList()
    val highlights = ArrayList<ClangdDocumentHighlight>(count)
    for (index in 0 until count) {
      val item = value[index]
      val range = parseRange(item?.range) ?: continue
      highlights += ClangdDocumentHighlight(
        range = range,
        kind = clangdInt(item.kind)
      )
    }
    return highlights
  }

  private fun parseDocumentSymbols(value: dynamic): List<ClangdDocumentSymbol> {
    val count = clangdInt(value?.length) ?: return emptyList()
    val symbols = ArrayList<ClangdDocumentSymbol>(count)
    for (index in 0 until count) {
      parseDocumentSymbol(value[index], documentUri)?.let(symbols::add)
    }
    return symbols
  }

  private fun parseDocumentSymbol(
    value: dynamic,
    inheritedUri: String?
  ): ClangdDocumentSymbol? {
    if (!clangdDefined(value)) return null
    val name = value.name as? String ?: return null
    val kind = clangdInt(value.kind) ?: return null
    val location = value.location
    val uri =
      if (clangdDefined(location)) location.uri as? String ?: inheritedUri
      else inheritedUri
    val range =
      if (clangdDefined(location)) parseRange(location.range)
      else parseRange(value.range)
    if (range == null) return null
    val selectionRange =
      if (clangdDefined(location)) range
      else parseRange(value.selectionRange) ?: range

    val childrenValue = value.children
    val childCount = clangdInt(childrenValue?.length) ?: 0
    val children = ArrayList<ClangdDocumentSymbol>(childCount)
    for (index in 0 until childCount) {
      parseDocumentSymbol(childrenValue[index], uri)?.let(children::add)
    }

    val tags = parseIntList(value.tags).toMutableList()
    if (value.deprecated as? Boolean == true && 1 !in tags) tags += 1
    return ClangdDocumentSymbol(
      uri = uri,
      name = name,
      detail = value.detail as? String,
      kind = kind,
      tags = tags,
      containerName = value.containerName as? String,
      range = range,
      selectionRange = selectionRange,
      children = children
    )
  }

  private fun parseSignatureHelp(value: dynamic): ClangdSignatureHelp? {
    if (!clangdDefined(value)) return null
    val rawSignatures = value.signatures
    val count = clangdInt(rawSignatures?.length) ?: return null
    val signatures = ArrayList<ClangdSignature>(count)
    for (index in 0 until count) {
      parseSignature(rawSignatures[index])?.let(signatures::add)
    }
    return ClangdSignatureHelp(
      signatures = signatures,
      activeSignature = clangdInt(value.activeSignature),
      activeParameter = clangdInt(value.activeParameter)
    )
  }

  private fun parseSignature(value: dynamic): ClangdSignature? {
    if (!clangdDefined(value)) return null
    val label = value.label as? String ?: return null
    val rawParameters = value.parameters
    val count = clangdInt(rawParameters?.length) ?: 0
    val parameters = ArrayList<ClangdSignatureParameter>(count)
    for (index in 0 until count) {
      parseSignatureParameter(rawParameters[index])?.let(parameters::add)
    }
    return ClangdSignature(
      label = label,
      documentation = parseMarkupContent(value.documentation),
      parameters = parameters,
      activeParameter = clangdInt(value.activeParameter)
    )
  }

  private fun parseSignatureParameter(value: dynamic): ClangdSignatureParameter? {
    if (!clangdDefined(value)) return null
    val rawLabel = value.label
    val stringLabel = rawLabel as? String
    val offsetLabel = if (clangdIsArray(rawLabel)) rawLabel else null
    return ClangdSignatureParameter(
      label = stringLabel,
      labelStart = clangdInt(offsetLabel?.get(0)),
      labelEnd = clangdInt(offsetLabel?.get(1)),
      documentation = parseMarkupContent(value.documentation)
    )
  }

  private fun parseSemanticTokensLegend(
    provider: dynamic
  ): ClangdSemanticTokensLegend? {
    if (!clangdDefined(provider)) return null
    val full = provider.full
    val supportsFull =
      full as? Boolean == true ||
        clangdDefined(full) && full !is Boolean
    if (!supportsFull) return null

    val legend = provider.legend
    if (!clangdDefined(legend)) return null
    val tokenTypes = parseStringList(legend.tokenTypes)
    if (tokenTypes.isEmpty()) return null
    return ClangdSemanticTokensLegend(
      tokenTypes = tokenTypes.toList(),
      tokenModifiers = parseStringList(legend.tokenModifiers).toList()
    )
  }

  private fun parseSemanticTokens(value: dynamic): ClangdSemanticTokens? {
    if (!clangdDefined(value)) return null
    val rawData = value.data
    val count = clangdInt(rawData?.length) ?: return null
    if (count % 5 != 0) return null

    val data = ArrayList<Int>(count)
    for (index in 0 until count) {
      val item = clangdInt(rawData[index]) ?: return null
      if (item < 0) return null
      data += item
    }
    return ClangdSemanticTokens(
      resultId = value.resultId as? String,
      data = data.toList()
    )
  }

  private fun parseTextEdits(values: dynamic): List<ClangdTextEdit> {
    val count = clangdInt(values?.length) ?: return emptyList()
    val edits = ArrayList<ClangdTextEdit>(count)
    for (index in 0 until count) {
      parseTextEdit(values[index])?.let(edits::add)
    }
    return edits
  }

  private fun parseTextEdit(value: dynamic): ClangdTextEdit? {
    if (!clangdDefined(value)) return null
    val range = when {
      clangdDefined(value.range) -> parseRange(value.range)
      clangdDefined(value.replace) -> parseRange(value.replace)
      else -> parseRange(value.insert)
    } ?: return null
    return ClangdTextEdit(
      range = range,
      newText = value.newText as? String ?: return null
    )
  }

  private fun parseRange(value: dynamic): ClangdRange? {
    if (!clangdDefined(value)) return null
    val start = parsePosition(value.start) ?: return null
    val end = parsePosition(value.end) ?: return null
    return ClangdRange(start, end)
  }

  private fun parsePosition(value: dynamic): ClangdPosition? {
    if (!clangdDefined(value)) return null
    val line = clangdInt(value.line) ?: return null
    val character = clangdInt(value.character) ?: return null
    return ClangdPosition(line, character)
  }

  private fun sendRequest(
    method: String,
    params: dynamic,
    timeoutMs: Int,
    callback: (result: dynamic, error: dynamic) -> Unit
  ): Int {
    if (!clangdDefined(worker)) return -1
    val id = nextRequestId++
    val timeout = window.setTimeout({
      val pending = pendingRequests.remove(id) ?: return@setTimeout
      val error = clangdObject()
      error.code = -32001
      error.message = "$method timed out"
      try {
        pending.callback(null, error)
      } catch (_: Throwable) {
      }
    }, timeoutMs)
    pendingRequests[id] = PendingRequest(timeout, callback)

    val message = clangdObject()
    message.jsonrpc = "2.0"
    message.id = id
    message.method = method
    message.params = params
    return try {
      sendLsp(message)
      id
    } catch (failure: Throwable) {
      pendingRequests.remove(id)
      window.clearTimeout(timeout)
      handleWorkerFailure(failure.message ?: "Unable to send $method to clangd")
      -1
    }
  }

  private fun sendNotification(method: String, params: dynamic) {
    if (!clangdDefined(worker)) return
    val message = clangdObject()
    message.jsonrpc = "2.0"
    message.method = method
    message.params = params
    sendLsp(message)
  }

  private fun respond(id: dynamic, result: dynamic) {
    val message = clangdObject()
    message.jsonrpc = "2.0"
    message.id = id
    message.result = result
    sendLsp(message)
  }

  private fun respondError(id: dynamic, code: Int, text: String) {
    val error = clangdObject()
    error.code = code
    error.message = text
    val message = clangdObject()
    message.jsonrpc = "2.0"
    message.id = id
    message.error = error
    sendLsp(message)
  }

  private fun sendLsp(message: dynamic) {
    val envelope = clangdObject()
    envelope.type = "lsp"
    envelope.message = message
    postToWorker(envelope)
  }

  private fun postToWorker(message: dynamic) {
    val target = worker
    if (!clangdDefined(target)) return
    target.postMessage(message)
  }

  private fun cancelCompletion(requestId: Int? = null) {
    val id = activeCompletionRequest ?: return
    if (requestId != null && requestId != id) return
    val pending = pendingRequests.remove(id)
    pending?.let { window.clearTimeout(it.timeout) }
    if (clangdDefined(worker) && initialized) {
      val params = clangdObject()
      params.id = id
      sendNotification("$/cancelRequest", params)
    }
    if (pending != null) {
      deliverRaw(
        pending.callback,
        null,
        clangdRequestError(-32800, "Completion request cancelled")
      )
    }
    activeCompletionRequest = null
  }

  private fun cancelChangeTimer() {
    changeTimer?.let(window::clearTimeout)
    changeTimer = null
  }

  private fun clearPendingRequests(message: String = "clangd stopped") {
    if (pendingRequests.isEmpty()) return
    val requests = pendingRequests.values.toList()
    pendingRequests.clear()
    requests.forEach { pending ->
      window.clearTimeout(pending.timeout)
      deliverRaw(
        pending.callback,
        null,
        clangdRequestError(-32800, message)
      )
    }
  }

  private fun cancelFileRead(id: Int) {
    val pending = pendingFileReads.remove(id) ?: return
    window.clearTimeout(pending.timeout)
    deliverFileRead(pending.callback, null, "Virtual file read cancelled")
  }

  private fun clearPendingFileReads(message: String) {
    if (pendingFileReads.isEmpty()) return
    val reads = pendingFileReads.values.toList()
    pendingFileReads.clear()
    reads.forEach { pending ->
      window.clearTimeout(pending.timeout)
      deliverFileRead(pending.callback, null, message)
    }
  }

  private fun handleWorkerFailure(message: String) {
    if (disposed) {
      terminateWorker()
      return
    }
    cancelChangeTimer()
    cancelCompletion()
    clearPendingRequests(message)
    clearPendingFileReads(message)
    initialized = false
    opened = false
    semanticLegend = null
    reportStatus(ClangdClientState.ERROR, message)
    terminateWorker(reportStopped = false)
  }

  private fun terminateWorker(reportStopped: Boolean = true) {
    val target = worker
    worker = null
    if (clangdDefined(target)) {
      try {
        target.onmessage = null
        target.onerror = null
        target.onmessageerror = null
        target.terminate()
      } catch (_: Throwable) {
      }
    }
    cancelChangeTimer()
    clearPendingRequests("clangd stopped")
    clearPendingFileReads("clangd stopped")
    initialized = false
    opened = false
    semanticLegend = null
    if (reportStopped) reportStatus(ClangdClientState.STOPPED, "clangd stopped")
  }

  private fun emitDiagnostics(
    uri: String,
    version: Int?,
    diagnostics: List<ClangdDiagnostic>
  ) {
    try {
      onDiagnostics(uri, version, diagnostics)
    } catch (_: Throwable) {
    }
  }

  private fun reportStatus(state: ClangdClientState, message: String) {
    try {
      onStatus(state, message)
    } catch (_: Throwable) {
    }
  }

  private fun callbackSafely(
    callback: (List<ClangdCompletion>) -> Unit,
    completions: List<ClangdCompletion>
  ) {
    try {
      callback(completions)
    } catch (_: Throwable) {
    }
  }

  private fun deliverSemanticTokens(
    callback: (ClangdSemanticTokens?, ClangdRequestFailure?) -> Unit,
    tokens: ClangdSemanticTokens?,
    failure: ClangdRequestFailure?
  ) {
    try {
      callback(tokens, failure)
    } catch (_: Throwable) {
    }
  }

  private fun <T> deliver(callback: (T) -> Unit, value: T) {
    try {
      callback(value)
    } catch (_: Throwable) {
    }
  }

  private fun deliverFileRead(
    callback: (file: ClangdVirtualFile?, error: String?) -> Unit,
    file: ClangdVirtualFile?,
    error: String?
  ) {
    try {
      callback(file, error)
    } catch (_: Throwable) {
    }
  }

  private fun deliverRaw(
    callback: (result: dynamic, error: dynamic) -> Unit,
    result: dynamic,
    error: dynamic
  ) {
    try {
      callback(result, error)
    } catch (_: Throwable) {
    }
  }

  private fun createClangdWorker(): dynamic {
    val url = js("""(href) => {
      const result = new URL(href);
      result.search = "";
      result.hash = "";
      result.searchParams.set("cpp-worker", "clangd");
      return result.href;
    }""")(window.location.href) as String
    return js("(url) => new Worker(url, { name: 'tidyparse-clangd' })")(url)
  }
}

private fun safeFileName(value: String): String =
  value.substringAfterLast('/').substringAfterLast('\\').ifBlank { "main.cpp" }

private fun normalizeLanguageId(value: String): String =
  if (value.equals("c", ignoreCase = true)) "c" else "cpp"

private fun encodeUriSegment(value: String): String =
  js("(value) => encodeURIComponent(value)")(value) as String

private fun clampedPosition(text: String, requestedLine: Int, requestedColumn: Int): ClangdPosition {
  val targetLine = requestedLine.coerceAtLeast(0)
  var line = 0
  var lineStart = 0
  while (line < targetLine) {
    val newline = text.indexOf('\n', lineStart)
    if (newline < 0) {
      val actualLine = text.count { it == '\n' }
      val actualStart = text.lastIndexOf('\n').let { if (it < 0) 0 else it + 1 }
      return ClangdPosition(actualLine, text.length - actualStart)
    }
    lineStart = newline + 1
    line++
  }
  val lineEnd = text.indexOf('\n', lineStart).let { if (it < 0) text.length else it }
  return ClangdPosition(line, requestedColumn.coerceIn(0, lineEnd - lineStart))
}

private fun characterBefore(text: String, position: ClangdPosition): String? {
  var line = 0
  var offset = 0
  while (line < position.line) {
    val newline = text.indexOf('\n', offset)
    if (newline < 0) return null
    offset = newline + 1
    line++
  }
  val caret = (offset + position.character).coerceAtMost(text.length)
  return if (caret > offset) text.substring(caret - 1, caret) else null
}

private fun lspPosition(position: ClangdPosition): dynamic {
  val value = clangdObject()
  value.line = position.line
  value.character = position.character
  return value
}

private fun parseStringList(values: dynamic): List<String> {
  val count = clangdInt(values?.length) ?: return emptyList()
  return (0 until count).mapNotNull { values[it] as? String }
}

private fun parseIntList(values: dynamic): List<Int> {
  val count = clangdInt(values?.length) ?: return emptyList()
  return (0 until count).mapNotNull { clangdInt(values[it]) }
}

private fun clangdRequestError(code: Int, message: String): dynamic {
  val error = clangdObject()
  error.code = code
  error.message = message
  return error
}

private fun clangdObject(): dynamic = js("({})")

private fun clangdDefined(value: dynamic): Boolean =
  value != null && value != js("undefined")

private fun clangdHasOwn(value: dynamic, name: String): Boolean =
  clangdDefined(value) &&
    js("(value, name) => Object.prototype.hasOwnProperty.call(value, name)")(value, name) as Boolean

private fun clangdInt(value: dynamic): Int? =
  if (!clangdDefined(value)) null else (value as? Number)?.toInt()

private fun clangdIsArray(value: dynamic): Boolean =
  clangdDefined(value) && js("(value) => Array.isArray(value)")(value) as Boolean

private fun clangdNullArray(size: Int): dynamic =
  js("(size) => Array(size).fill(null)")(size)
