import kotlinx.browser.window
import kotlinx.coroutines.*
import kotlinx.coroutines.promise
import org.w3c.dom.HTMLElement
import kotlin.js.Promise
import kotlin.math.ceil

private const val CPP_WORKSPACE_PATH = "/home/web_user"
private const val CPP_WORKSPACE_URI = "file://$CPP_WORKSPACE_PATH"
private const val CPP_CLANGD_WORKER_NAME = "tidyparse-clangd"
private const val CPP_COMPLETION_CONTEXT_TIMEOUT_MS = 1_200L
private const val CPP_COMPLETION_AST_TIMEOUT_MS = 1_000L
private const val CPP_COMPLETION_WORKER_TIMEOUT_MS = 4_000L
private const val CPP_COMPLETION_FORMAT_TIMEOUT_MS = 1_200L
private const val CPP_COMPLETION_FORMAT_CLOSE_TIMEOUT_MS = 250L
private const val CPP_COMPLETION_TOTAL_TIMEOUT_MS = 7_000L
private const val CPP_COMPLETION_SHORTCUT_WINDOW_MS = 1_000.0
private const val CPP_COMPLETION_WIDGET_CHROME_PX = 80.0

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

private class CachedCppGrammarCompletion(
  val key: String,
  val reply: dynamic
)

private class CachedCppAstContext(
  val key: String,
  val context: dynamic
)

private class CppFormattingResponse(val edits: dynamic)

/** Links Monaco cancellation to one owned vscode-jsonrpc cancellation source. */
private class CppLspCancellation(
  private val source: dynamic,
  upstream: dynamic
) {
  private var registration: dynamic = null
  val token: dynamic = if (defined(source)) source.token else upstream

  init {
    try {
      if (defined(source) && completionTokenCancelled(upstream)) {
        source.cancel()
      } else if (defined(source) && defined(upstream?.onCancellationRequested)) {
        registration = upstream.onCancellationRequested { source.cancel() }
      }
    } catch (_: Throwable) {
      // The owned source still enforces the time budget even if a host token is not linkable.
    }
  }

  fun cancelAndDispose() {
    try {
      registration?.dispose()
    } catch (_: Throwable) {
    }
    registration = null
    if (!defined(source)) return
    try {
      source.cancel()
    } catch (_: Throwable) {
    }
    try {
      source.dispose()
    } catch (_: Throwable) {
    }
  }
}

private fun completionTokenCancelled(token: dynamic): Boolean =
  token?.isCancellationRequested as? Boolean == true

/**
 * The only C/C++ editor integration we own.
 *
 * MonacoVscodeApiWrapper installs the VS Code services, EditorApp owns Monaco's
 * models and editor lifecycle, and LanguageClientWrapper owns the complete LSP
 * client/provider bridge. This class only adapts those libraries to the page UI
 * and to our in-browser clangd worker.
 *
 * Imports remain lazy because tidyparse-web.js is also evaluated in the clangd
 * worker and service-worker globals, where Monaco's DOM modules cannot start.
 */
class JSMonacoEditor(
  private val container: HTMLElement,
  private var fileName: String,
  private val initialText: String,
  private var darkTheme: Boolean,
  private val onChange: (String) -> Unit,
  private val onPosition: (line: Int, column: Int) -> Unit,
  private val onOpenedFile: (String) -> Unit,
  private val onRun: () -> Unit
) {
  private var modules: dynamic = null
  private var apiWrapper: dynamic = null
  private var editorApp: dynamic = null
  private var languageClientWrapper: dynamic = null
  private var editor: dynamic = null
  private var worker: dynamic = null
  private var languageClientPort: dynamic = null
  private var diagnosticsSubscription: dynamic = null
  private val editorDisposables = mutableListOf<dynamic>()
  private val completionScope = MainScope()
  private var completionWorker: CppCompletionWorkerClient? = null
  private var explicitCompletionUntil = 0.0
  private var latestRawDiagnostics: dynamic = null
  private var cachedGrammarCompletion: CachedCppGrammarCompletion? = null
  private var cachedCppAstContext: CachedCppAstContext? = null
  private var cppCompletionContextEpoch = 0
  private var nextCppFormatDocumentId = 1
  private var requestedReadOnly = false
  private var disposed = false

  suspend fun start() {
    check(!disposed) { "The C/C++ editor has already been disposed" }
    if (editorApp != null) return

    val loaded = loadModules()
    val vscode = loaded.vscode

    val apiConfig = js("{}")
    apiConfig["\$type"] = "extended"
    apiConfig.viewsConfig = js("{}")
    apiConfig.viewsConfig["\$type"] = "EditorService"
    apiConfig.workspaceConfig = workspaceConfig(vscode)
    apiConfig.userConfiguration = js("{}")
    apiConfig.userConfiguration.json = userConfigurationJson()
    apiConfig.monacoWorkerFactory = { _: dynamic -> configureMonacoWorkers() }
    apiConfig.advanced = js("{}")
    apiConfig.advanced.enableExtHostWorker = false
    apiConfig.advanced.enforceSemanticHighlighting = true

    val ApiWrapper = loaded.api.MonacoVscodeApiWrapper
    apiWrapper = js("new ApiWrapper(apiConfig)")
    awaitPromise(apiWrapper.start())

    val appConfig = js("{}")
    appConfig.codeResources = codeResources(fileName, initialText)
    appConfig.useDiffEditor = false
    appConfig.editorOptions = editorOptions()

    val EditorApp = loaded.editorApp.EditorApp
    editorApp = js("new EditorApp(appConfig)")
    editorApp.registerOnTextChangedCallback { changes: dynamic ->
      val text = changes.modified as? String ?: return@registerOnTextChangedCallback
      cachedGrammarCompletion = null
      cachedCppAstContext = null
      cppCompletionContextEpoch++
      // VS Code diagnostics do not carry the model version. Never apply ranges published for the
      // previous text to a newly edited statement while clangd is still reparsing it.
      latestRawDiagnostics = null
      onChange(text)
      reportStatus(ClangdClientState.BUSY, "clangd analyzing…")
    }
    awaitPromise(editorApp.start(container))

    editor = editorApp.getEditor()
      ?: error("monaco-languageclient did not create an editor")
    bindEditor()
    installDiagnosticsListener(vscode)
    onOpenedFile(fileName)
    onPosition(1, 1)
  }

  suspend fun startClangd(
    onStatus: (ClangdClientState, String) -> Unit,
    onDiagnostics: (List<ClangdDiagnostic>) -> Unit
  ) {
    check(editorApp != null) { "The editor must be started before clangd" }
    if (languageClientWrapper != null || disposed) return

    statusListener = onStatus
    diagnosticsListener = onDiagnostics
    reportStatus(ClangdClientState.STARTING, "Starting clangd…")
    // Parse the shared completion bundle in parallel with clangd startup. The ready handshake
    // keeps its one-time script evaluation outside the per-request grammar deadline.
    ensureCppCompletionWorker()

    try {
      val clangdWorker = createClangdWorker()
      worker = clangdWorker
      val channel = js("new MessageChannel()")
      languageClientPort = channel.port1
      val connect = js("{}")
      connect.type = "connect"
      connect.port = channel.port2
      clangdWorker.postMessage(connect, arrayOf(channel.port2))

      awaitClangdReady(clangdWorker)
      reportStatus(ClangdClientState.INITIALIZING, "Initializing clangd…")

      val config = js("{}")
      config.languageId = "cpp"
      config.connection = js("{}")
      config.connection.options = js("{}")
      config.connection.options["\$type"] = "WorkerDirect"
      config.connection.options.worker = clangdWorker
      config.connection.options.messagePort = channel.port1
      config.clientOptions = js("{}")
      config.clientOptions.documentSelector = arrayOf("cpp", "c")
      config.clientOptions.middleware = clangdEditorMiddleware()
      config.clientOptions.initializationOptions = js("({ clangdFileStatus: true })")
      config.clientOptions.workspaceFolder = js("{}")
      config.clientOptions.workspaceFolder.index = 0
      config.clientOptions.workspaceFolder.name = "workspace"
      config.clientOptions.workspaceFolder.uri =
        modules.vscode.Uri.file(CPP_WORKSPACE_PATH)
      config.disposeWorker = true

      val LanguageClientWrapper = modules.languageClient.LanguageClientWrapper
      languageClientWrapper = js("new LanguageClientWrapper(config)")
      awaitPromise(languageClientWrapper.start())
      installClangdStatusNotification()
      emitDiagnostics()
      reportStatus(ClangdClientState.READY, "clangd is ready")
    } catch (failure: Throwable) {
      reportStatus(
        ClangdClientState.ERROR,
        failure.message ?: "Unable to initialize clangd"
      )
      disposeClangdRuntime()
      throw failure
    }
  }

  fun value(): String {
    val model = editorApp?.getTextModels()?.modified
    return model?.getValue() as? String ?: ""
  }

  fun focus() {
    editor?.focus()
  }

  fun setReadOnly(readOnly: Boolean) {
    requestedReadOnly = readOnly
    updateReadOnly()
  }

  fun setTheme(dark: Boolean) {
    darkTheme = dark
    val update = modules?.configuration?.updateUserConfiguration ?: return
    try {
      update(userConfigurationJson())
    } catch (_: Throwable) {
    }
  }

  suspend fun setDocument(nextFileName: String, text: String) {
    val app = editorApp ?: return
    fileName = nextFileName
    cachedGrammarCompletion = null
    cachedCppAstContext = null
    cppCompletionContextEpoch++
    latestRawDiagnostics = null
    if (nextFileName.endsWith(".c", ignoreCase = true)) {
      completionWorker?.dispose()
      completionWorker = null
    } else if (languageClientWrapper != null) {
      ensureCppCompletionWorker()
    }
    awaitPromise(app.updateCodeResources(codeResources(nextFileName, text)))
    editor = app.getEditor()
    editor?.setPosition(js("({ lineNumber: 1, column: 1 })"))
    editor?.revealPosition(js("({ lineNumber: 1, column: 1 })"))
    onOpenedFile(nextFileName)
    onPosition(1, 1)
    emitDiagnostics()
  }

  fun setValue(text: String) {
    val app = editorApp ?: return
    val activeEditor = editor ?: return
    val mainModel = app.getTextModels()?.modified
    if (!defined(mainModel)) return

    if (activeEditor.getModel() !== mainModel) {
      activeEditor.setModel(mainModel)
    }
    if (mainModel.getValue() != text) {
      mainModel.setValue(text)
    }
    activeEditor.setPosition(js("({ lineNumber: 1, column: 1 })"))
    activeEditor.revealPosition(js("({ lineNumber: 1, column: 1 })"))
    onOpenedFile(fileName)
    onPosition(1, 1)
  }

  fun dispose() {
    if (disposed) return
    disposed = true

    completionWorker?.dispose()
    completionWorker = null
    completionScope.cancel()

    editorDisposables.forEach(::disposeSafely)
    editorDisposables.clear()
    disposeSafely(diagnosticsSubscription)
    diagnosticsSubscription = null

    val app = editorApp
    editorApp = null
    editor = null

    val appDisposal = try {
      if (defined(app)) app.dispose() else null
    } catch (_: Throwable) {
      null
    }
    afterPromiseSettles(appDisposal) {
      disposeClangdRuntime(::finishDispose)
    }
  }

  private fun finishDispose() {
    try {
      apiWrapper?.dispose()
    } catch (_: Throwable) {
    }
    apiWrapper = null
    reportStatus(ClangdClientState.STOPPED, "clangd stopped")
  }

  private var statusListener: (ClangdClientState, String) -> Unit = { _, _ -> }
  private var diagnosticsListener: (List<ClangdDiagnostic>) -> Unit = {}

  private fun loadModules(): dynamic {
    if (defined(modules)) return modules

    // Register the default themes and C/C++ TextMate grammars before the VS
    // Code services initialize.
    js("require('@codingame/monaco-vscode-theme-defaults-default-extension')")
    js("require('@codingame/monaco-vscode-cpp-default-extension')")

    val result = js("{}")
    result.api = js("require('monaco-languageclient/vscodeApiWrapper')")
    result.editorApp = js("require('monaco-languageclient/editorApp')")
    result.languageClient = js("require('monaco-languageclient/lcwrapper')")
    result.configuration =
      js("require('@codingame/monaco-vscode-configuration-service-override')")
    result.monaco = js("require('monaco-editor')")
    result.vscode = js("require('vscode')")
    modules = result
    return result
  }

  private fun configureMonacoWorkers() {
    val global = js("globalThis")
    val environment =
      if (defined(global.MonacoEnvironment)) global.MonacoEnvironment
      else js("{}")
    environment.getWorker = { _: dynamic, label: dynamic ->
      when (label as? String) {
        "TextMateWorker" ->
          createMonacoWorker(CPP_TEXTMATE_WORKER_NAME)
        "editorWorkerService" ->
          createMonacoWorker(CPP_MONACO_EDITOR_WORKER_NAME)
        else -> js("undefined")
      }
    }
    global.MonacoEnvironment = environment
  }

  private fun createMonacoWorker(name: String): dynamic {
    val url = js(
      """(href) => {
        const controller = navigator.serviceWorker && navigator.serviceWorker.controller;
        const controllerUrl = controller && new URL(controller.scriptURL);
        const cppBootstrapControlsPage =
          controllerUrl &&
          controllerUrl.pathname.endsWith("/tidyparse-web.js") &&
          controllerUrl.searchParams.has("cpp-coi");
        const result = cppBootstrapControlsPage
          ? new URL(href)
          : new URL("tidyparse-web.js", href);
        result.search = "";
        result.hash = "";
        if (cppBootstrapControlsPage) {
          // The exact-page COI worker already maps this route to the shared
          // bundle. The worker name, rather than this compatibility key,
          // selects the Monaco or TextMate entry point.
          result.searchParams.set("cpp-worker", "clangd");
        }
        return result.href;
      }"""
    )(window.location.href) as String
    return js("(url, name) => new Worker(url, { name })")(url, name)
  }

  private fun workspaceConfig(vscode: dynamic): dynamic {
    val config = js("{}")
    config.workspaceProvider = js("{}")
    config.workspaceProvider.trusted = true
    config.workspaceProvider.workspace = js("{}")
    config.workspaceProvider.workspace.workspaceUri =
      vscode.Uri.file(CPP_WORKSPACE_PATH)
    config.workspaceProvider.open = { Promise.resolve(false) }
    return config
  }

  private fun codeResources(name: String, text: String): dynamic {
    val resources = js("{}")
    resources.modified = js("{}")
    resources.modified.text = text
    resources.modified.uri = "$CPP_WORKSPACE_URI/${encodePathSegment(name)}"
    resources.modified.enforceLanguageId =
      if (name.endsWith(".c", ignoreCase = true)) "c" else "cpp"
    return resources
  }

  private fun editorOptions(): dynamic = cppMonacoEditorOptions()

  private fun userConfigurationJson(): String {
    val configuration = js("{}")
    configuration["workbench.colorTheme"] =
      if (darkTheme) "Default Dark Modern" else "Default Light Modern"
    configuration["editor.wordBasedSuggestions"] = "off"
    configuration["editor.autoClosingBrackets"] = "never"
    configuration["editor.inlayHints.enabled"] = "offUnlessPressed"
    configuration["editor.quickSuggestionsDelay"] = 200
    configuration["editor.semanticHighlighting.enabled"] = true
    return JSON.stringify(configuration)
  }

  private fun bindEditor() {
    val activeEditor = editor ?: return
    val modelDisposable: dynamic = activeEditor.onDidChangeModel {
      updateReadOnly()
      val path = activeEditor.getModel()?.uri?.path as? String
      onOpenedFile(path?.substringAfterLast('/')?.takeIf(String::isNotBlank) ?: fileName)
    }
    editorDisposables.add(modelDisposable)
    val cursorDisposable: dynamic =
      activeEditor.onDidChangeCursorPosition { event: dynamic ->
        val cursor = event.position
        onPosition(number(cursor.lineNumber), number(cursor.column))
      }
    editorDisposables.add(cursorDisposable)

    val monaco = modules.monaco
    // Claim cross-model opens before VS Code tries to resolve a new file.
    // Returning false keeps Monaco's normal same-model navigation behavior.
    val editorOpener = js("{}")
    editorOpener.openCodeEditor =
      { source: dynamic, resource: dynamic, _: dynamic ->
        val currentEditor = editor
        val currentUri = currentEditor?.getModel()?.uri?.toString() as? String
        source === currentEditor && resource?.toString() != currentUri
      }
    editorDisposables.add(monaco.editor.registerEditorOpener(editorOpener))

    installCppGrammarCompletion(activeEditor, monaco)
    installCppCompletionWidgetSizing(activeEditor)

    val runKey = number(monaco.KeyMod.CtrlCmd) or number(monaco.KeyCode.Enter)
    activeEditor.addCommand(runKey, { onRun() })
    activeEditor.setPosition(js("({ lineNumber: 1, column: 1 })"))
    updateReadOnly()
  }

  /** Adds the full-statement grammar as the only user-visible C++ LSP completion provider. */
  private fun installCppGrammarCompletion(activeEditor: dynamic, monaco: dynamic) {
    editorDisposables.add(activeEditor.onKeyDown { event: dynamic ->
      if (isCppCompletionShortcut(event)) {
        explicitCompletionUntil = window.performance.now() + CPP_COMPLETION_SHORTCUT_WINDOW_MS
      }
    })

    val provider = js("({})")
    provider.provideCompletionItems =
      { model: dynamic, position: dynamic, _: dynamic, cancellation: dynamic ->
        val explicitlyRequested = consumeExplicitCppCompletion()
        val isCpp = model?.getLanguageId() as? String == "cpp"
        val worker = if (explicitlyRequested && isCpp) ensureCppCompletionWorker() else null
        if (!explicitlyRequested || !isCpp || worker == null) null
        else completionScope.promise {
          withTimeoutOrNull(CPP_COMPLETION_TOTAL_TIMEOUT_MS) {
            provideCppGrammarCompletions(model, position, cancellation, monaco)
          } ?: emptyCppCompletionResult()
        }
      }
    editorDisposables.add(monaco.languages.registerCompletionItemProvider("cpp", provider))
  }

  /**
   * Expands Monaco's completion list to the longest currently filtered label.
   *
   * Monaco calculates a preferred width from an 85th-percentile label length, but only applies it
   * after a manual sash reset. There is no public suggest-width option. The pinned widget's resize
   * routine is used here because it updates the virtualized list and recomputes overflow placement
   * together; changing CSS width alone leaves both geometries at the old 430-pixel default.
   */
  private fun installCppCompletionWidgetSizing(activeEditor: dynamic) {
    val controller = try {
      activeEditor.getContribution("editor.contrib.suggestController")
    } catch (_: Throwable) {
      null
    }
    if (!defined(controller)) return

    val subscription = try {
      controller.model.onDidSuggest { event: dynamic ->
        resizeCppCompletionWidget(controller, event?.completionModel)
      }
    } catch (_: Throwable) {
      null
    }
    if (defined(subscription)) editorDisposables.add(subscription)
  }

  private fun resizeCppCompletionWidget(controller: dynamic, completionModel: dynamic) {
    try {
      val items = completionModel?.items
      if (!defined(items) || js("Array.isArray(items)") as Boolean == false) return
      var longestLabelCharacters = 0
      for (index in 0 until number(items.length)) {
        val item = items[index]
        val rawLabel = item?.completion?.label
        val label = item?.textLabel as? String
          ?: rawLabel as? String
          ?: rawLabel?.label as? String
        if (label != null) longestLabelCharacters = maxOf(longestLabelCharacters, label.length)
      }
      if (longestLabelCharacters == 0) return

      val widget = controller.widget?.value
      val element = widget?.element
      val size = element?.size
      val layout = widget?.getLayoutInfo()
      val currentWidth = (size?.width as? Number)?.toDouble() ?: return
      val currentHeight = (size.height as? Number)?.toDouble() ?: return
      val maximumWidth = (element.maxSize?.width as? Number)?.toDouble() ?: return
      val halfwidth = (layout?.typicalHalfwidthCharacterWidth as? Number)?.toDouble() ?: return
      val targetWidth = cppCompletionWidgetTargetWidth(
        longestLabelCharacters = longestLabelCharacters,
        typicalHalfwidthCharacterWidth = halfwidth,
        currentWidth = currentWidth,
        maximumWidth = maximumWidth
      )
      if (targetWidth <= currentWidth) return

      val resize = widget["_resize"]
      if (jsTypeOf(resize) == "function") resize.call(widget, targetWidth, currentHeight)
    } catch (_: Throwable) {
      // Suggest sizing is cosmetic. A Monaco upgrade must not suppress completion results.
    }
  }

  private fun isCppCompletionShortcut(event: dynamic): Boolean =
    js(
      """(event) => {
        const browser = event && event.browserEvent || event || {};
        const modified = !!(event && (event.ctrlKey || event.metaKey) || browser.ctrlKey || browser.metaKey);
        const shifted = !!(event && event.shiftKey || browser.shiftKey);
        return modified && !shifted && !(event && event.altKey || browser.altKey) &&
          (browser.key === " " || browser.key === "Spacebar" || browser.code === "Space");
      }"""
    )(event) as Boolean

  private fun ensureCppCompletionWorker(): CppCompletionWorkerClient? {
    completionWorker?.let { return it }
    if (fileName.endsWith(".c", ignoreCase = true)) return null
    completionWorker = try {
      CppCompletionWorkerClient()
    } catch (failure: Throwable) {
      console.warn("The C++ grammar completion worker could not start.", failure)
      null
    }
    return completionWorker
  }

  private fun consumeExplicitCppCompletion(): Boolean {
    val requested = window.performance.now() <= explicitCompletionUntil
    explicitCompletionUntil = 0.0
    return requested
  }

  private suspend fun provideCppGrammarCompletions(
    model: dynamic,
    position: dynamic,
    cancellation: dynamic,
    monaco: dynamic
  ): dynamic {
    val empty = emptyCppCompletionResult()
    if (completionCancelled(cancellation)) return empty
    val source = model?.getValue() as? String ?: return empty
    val line = number(position?.lineNumber) - 1
    val character = number(position?.column) - 1
    val snapshot = cppEditorStatementSnapshot(source, line, character) ?: return empty
    val version = number(model.getVersionId())
    val modelUri = model.uri?.toString() as? String ?: documentUri()
    val contextEpoch = cppCompletionContextEpoch
    val resultKey = "$modelUri:$version:$contextEpoch:${snapshot.cacheKey}"
    val sourceKey = "$modelUri:$version"
    val astKey = "$sourceKey:${snapshot.line}:${snapshot.character}"

    val reply = cachedGrammarCompletion?.takeIf { it.key == resultKey }?.reply ?: try {
      val lspCancellation = cppLspCancellation(cancellation)
      val queriedFacts = try {
        withTimeoutOrNull(CPP_COMPLETION_CONTEXT_TIMEOUT_MS) {
          requestCppCompletionFacts(source, snapshot, astKey, lspCancellation.token)
        }
      } finally {
        // Cancelling after success is harmless; after a timeout it sends $/cancelRequest for any
        // unresolved work instead of leaving it queued in the single WASM-clangd process.
        lspCancellation.cancelAndDispose()
      }
      val facts = queriedFacts ?: CppCompletionSemanticFacts()
      if (
        contextEpoch != cppCompletionContextEpoch ||
        !cppCompletionStillCurrent(model, version, snapshot) ||
        completionCancelled(cancellation)
      )
        return empty

      val request = cppCompletionWorkerRequest(
        cacheKey = "$modelUri:$contextEpoch",
        source = source,
        snapshot = snapshot,
        facts = facts.copy(diagnostics = latestRawDiagnostics)
      )
      val activeWorker = requireNotNull(completionWorker)
      val completed = withTimeout(CPP_COMPLETION_WORKER_TIMEOUT_MS) {
        // [complete] owns the ready handshake as part of the same bounded operation. Keeping a
        // separate 15-second readiness wait made Monaco legitimately display "Loading..." long
        // after an interactive request was useful.
        activeWorker.complete(request)
      }
      if (
        contextEpoch != cppCompletionContextEpoch ||
        !cppCompletionStillCurrent(model, version, snapshot) ||
        completionCancelled(cancellation)
      )
        return empty
      val formatted = formatCppCompletionReply(completed, cancellation)
      if (
        contextEpoch != cppCompletionContextEpoch ||
        !cppCompletionStillCurrent(model, version, snapshot) ||
        completionCancelled(cancellation)
      )
        return empty
      // A cold AST timeout is intentionally retryable: return the degraded suggestions once, but
      // do not permanently suppress return/this/member facts at this unchanged cursor. A format
      // timeout is retryable for the same reason; the lexical suggestions remain available once.
      if (formatted && queriedFacts != null && defined(queriedFacts.ast))
        cachedGrammarCompletion = CachedCppGrammarCompletion(resultKey, completed)
      completed
    } catch (cancelled: CancellationException) {
      return empty
    } catch (failure: Throwable) {
      console.warn("C++ grammar completion failed.", failure)
      return empty
    }

    return monacoCompletionResult(reply, snapshot, position, monaco)
  }

  /** Formats the complete popup in one hidden clangd document and mutates the transport reply. */
  private suspend fun formatCppCompletionReply(reply: dynamic, upstreamCancellation: dynamic): Boolean {
    val suggestions = reply?.suggestions
    if (!defined(suggestions) || js("Array.isArray(suggestions)") as Boolean == false) return false
    val count = number(suggestions.length)
    if (count == 0) return false
    val candidates = (0 until count).map { index ->
      suggestions[index]?.candidateText as? String ?: return false
    }
    val client = try {
      languageClientWrapper?.getLanguageClient()
    } catch (_: Throwable) {
      null
    }
    if (!defined(client)) return false

    if (nextCppFormatDocumentId <= 0) nextCppFormatDocumentId = 1
    val nonce = nextCppFormatDocumentId++
    val batch = cppCompletionFormatBatch(candidates, nonce)
    val uri = "$CPP_WORKSPACE_URI/.tidyparse-completion-format-$nonce.cpp"
    val cancellation = cppLspCancellation(upstreamCancellation)
    var opened = false
    try {
      val response = withTimeoutOrNull(CPP_COMPLETION_FORMAT_TIMEOUT_MS) {
        val open = js("({})")
        open.textDocument = js("({})")
        open.textDocument.uri = uri
        open.textDocument.languageId = "cpp"
        open.textDocument.version = 1
        open.textDocument.text = batch.source
        awaitPromise(client.sendNotification("textDocument/didOpen", open))
        opened = true

        val params = js("({})")
        params.textDocument = js("({})")
        params.textDocument.uri = uri
        params.options = js("({})")
        params.options.tabSize = 4
        params.options.insertSpaces = true
        params.options.trimTrailingWhitespace = true
        params.options.insertFinalNewline = true
        val request = if (defined(cancellation.token))
          client.sendRequest("textDocument/formatting", params, cancellation.token)
        else client.sendRequest("textDocument/formatting", params)
        CppFormattingResponse(awaitPromise(request))
      } ?: return false
      val edits = cppFormatTextEdits(response.edits) ?: return false
      val formattedSource = applyCppFormatTextEdits(batch.source, edits) ?: return false
      val formatted = extractCppFormattedCompletions(batch, formattedSource) ?: return false
      if (formatted.size != count) return false
      val seen = mutableSetOf<String>()
      var kept = 0
      formatted.forEachIndexed { index, completion ->
        if (!seen.add(completion.replacementText)) return@forEachIndexed
        val suggestion: dynamic = suggestions[index]
        suggestion.candidateText = completion.replacementText
        suggestion.displayText = completion.displayText
        suggestions[kept++] = suggestion
      }
      suggestions.length = kept
      return kept > 0
    } catch (cancelled: CancellationException) {
      throw cancelled
    } catch (failure: Throwable) {
      console.warn("clang-format could not format C++ grammar completions; using lexical source.", failure)
      return false
    } finally {
      cancellation.cancelAndDispose()
      if (opened) withContext(NonCancellable) {
        withTimeoutOrNull(CPP_COMPLETION_FORMAT_CLOSE_TIMEOUT_MS) {
          try {
            val close = js("({})")
            close.textDocument = js("({})")
            close.textDocument.uri = uri
            awaitPromise(client.sendNotification("textDocument/didClose", close))
          } catch (_: Throwable) {
            // clangd may already be stopping; the unique scratch URI is never reused.
          }
        }
      }
    }
  }

  private fun cppFormatTextEdits(raw: dynamic): List<CppFormatTextEdit>? {
    if (!defined(raw)) return emptyList()
    if (js("Array.isArray(raw)") as Boolean == false) return null
    return (0 until number(raw.length)).map { index ->
      val edit = raw[index]
      val range = edit?.range ?: return null
      val start = range.start ?: return null
      val end = range.end ?: return null
      fun coordinate(value: dynamic): Int? =
        (value as? Number)?.toInt()?.takeIf { it >= 0 }
      CppFormatTextEdit(
        start = CppFormatPosition(
          coordinate(start.line) ?: return null,
          coordinate(start.character) ?: return null
        ),
        end = CppFormatPosition(
          coordinate(end.line) ?: return null,
          coordinate(end.character) ?: return null
        ),
        newText = edit.newText as? String ?: return null
      )
    }
  }

  private suspend fun requestCppCompletionFacts(
    source: String,
    snapshot: CppEditorStatementSnapshot,
    astKey: String,
    cancellation: dynamic
  ): CppCompletionSemanticFacts = coroutineScope {
    val client = try {
      languageClientWrapper?.getLanguageClient()
    } catch (_: Throwable) {
      null
    }
    if (!defined(client)) return@coroutineScope CppCompletionSemanticFacts()

    // clangd's AST extension is a whole-document request. The clangd worker reduces it at this
    // caret before returning anything to Monaco. Give a cold parse one full second, while keeping
    // a small outer-context margin in which the parallel completion requests can settle cleanly.
    val ast = async {
      cachedCppAstContext?.takeIf { it.key == astKey }?.let { return@async it.context }
      val astCancellation = cppLspCancellation(cancellation)
      val params = js("({})")
      params.textDocument = js("({})")
      params.textDocument.uri = documentUri()
      params[CPP_AST_CONTEXT_REQUEST_FIELD] = js("({})")
      params[CPP_AST_CONTEXT_REQUEST_FIELD].source = source
      params[CPP_AST_CONTEXT_REQUEST_FIELD].line = snapshot.line
      params[CPP_AST_CONTEXT_REQUEST_FIELD].character = snapshot.character
      val rawAst = try {
        withTimeoutOrNull(CPP_COMPLETION_AST_TIMEOUT_MS) {
          optionalClangdRequest(client, "textDocument/ast", params, astCancellation.token)
        }
      } finally {
        astCancellation.cancelAndDispose()
      }
      if (!defined(rawAst)) return@async null
      val normalized = if (rawAst[CPP_NORMALIZED_AST_CONTEXT_FIELD] as? Boolean == true) rawAst
      else cppClangdAstContextDto(rawAst, source, snapshot.line, snapshot.character)
      cachedCppAstContext = CachedCppAstContext(astKey, normalized)
      normalized
    }
    val semanticPrefix = snapshot.semanticPrefixText
    val semanticCharacter = snapshot.statementStartCharacter + semanticPrefix.length
    val receiverOperator = cppReceiverOperator(semanticPrefix)
    val base = async {
      optionalClangdRequest(
        client,
        "textDocument/completion",
        cppCompletionParams(snapshot.line, semanticCharacter, receiverOperator),
        cancellation
      )
    }
    val scopeCharacter = snapshot.statementStartCharacter +
      snapshot.prefixText.takeWhile { it == ' ' || it == '\t' }.length
    val scope = if (scopeCharacter == semanticCharacter) null else async {
      optionalClangdRequest(
        client,
        "textDocument/completion",
        cppCompletionParams(snapshot.line, scopeCharacter, null),
        cancellation
      )
    }
    val signatures = if (hasOpenCppCall(snapshot.prefixText)) async {
      optionalClangdRequest(
        client,
        "textDocument/signatureHelp",
        cppTextDocumentPositionParams(snapshot.line, snapshot.character),
        cancellation
      )
    } else null
    val hoverCharacter = cppReceiverHoverCharacter(semanticPrefix, receiverOperator)
      ?.let { snapshot.statementStartCharacter + it }
    val hover = hoverCharacter?.let { hoverAt -> async {
      optionalClangdRequest(
        client,
        "textDocument/hover",
        cppTextDocumentPositionParams(snapshot.line, hoverAt),
        cancellation
      )
    } }

    val groups = mutableListOf<CppClangdCompletionGroup>()
    val baseResult: dynamic = base.await()
    if (defined(baseResult))
      groups.add(CppClangdCompletionGroup(baseResult, receiverOperator != null, receiverOperator))
    val scopeResult: dynamic = scope?.await()
    if (defined(scopeResult)) groups.add(CppClangdCompletionGroup(scopeResult))
    CppCompletionSemanticFacts(groups, signatures?.await(), hover?.await(), ast = ast.await())
  }

  private suspend fun optionalClangdRequest(
    client: dynamic,
    method: String,
    params: dynamic,
    cancellation: dynamic
  ): dynamic = try {
    val request = if (defined(cancellation)) client.sendRequest(method, params, cancellation)
    else client.sendRequest(method, params)
    awaitPromise(request)
  } catch (cancelled: CancellationException) {
    throw cancelled
  } catch (_: Throwable) {
    null
  }

  private fun cppLspCancellation(upstream: dynamic): CppLspCancellation {
    val constructor = modules?.vscode?.CancellationTokenSource
    val source = try {
      if (defined(constructor)) js("(Ctor) => new Ctor()")(constructor) else null
    } catch (_: Throwable) {
      null
    }
    return CppLspCancellation(source, upstream)
  }

  private fun cppCompletionParams(line: Int, character: Int, receiverOperator: String?): dynamic {
    val params = cppTextDocumentPositionParams(line, character)
    params.context = js("({})")
    if (receiverOperator == null) {
      params.context.triggerKind = 1
    } else {
      params.context.triggerKind = 2
      params.context.triggerCharacter = receiverOperator.last().toString()
    }
    return params
  }

  private fun cppTextDocumentPositionParams(line: Int, character: Int): dynamic {
    val params = js("({})")
    params.textDocument = js("({})")
    params.textDocument.uri = documentUri()
    params.position = js("({})")
    params.position.line = line
    params.position.character = character
    return params
  }

  private fun cppReceiverOperator(prefixText: String): String? =
    prefixText.trimEnd().let { prefix ->
      when {
        prefix.endsWith("->") -> "->"
        prefix.endsWith("::") -> "::"
        prefix.endsWith('.') -> "."
        else -> null
      }
    }

  private fun cppReceiverHoverCharacter(prefixText: String, operator: String?): Int? {
    operator ?: return null
    val beforeOperator = prefixText.trimEnd().dropLast(operator.length)
    val character = beforeOperator.indexOfLast { it.isLetterOrDigit() || it == '_' }
    return character.takeIf { it >= 0 }
  }

  private fun hasOpenCppCall(prefixText: String): Boolean {
    var depth = 0
    prefixText.forEach { character -> when (character) {
      '(' -> depth++
      ')' -> if (depth > 0) depth--
    } }
    return depth > 0
  }

  private fun cppCompletionStillCurrent(
    model: dynamic,
    version: Int,
    snapshot: CppEditorStatementSnapshot
  ): Boolean {
    val activeEditor = editor ?: return false
    if (activeEditor.getModel() !== model || number(model.getVersionId()) != version) return false
    val current = activeEditor.getPosition() ?: return false
    return number(current.lineNumber) - 1 == snapshot.line &&
      number(current.column) - 1 == snapshot.character
  }

  private fun completionCancelled(cancellation: dynamic): Boolean =
    cancellation?.isCancellationRequested as? Boolean == true

  private fun monacoCompletionResult(
    reply: dynamic,
    snapshot: CppEditorStatementSnapshot,
    position: dynamic,
    monaco: dynamic
  ): dynamic {
    val suggestions = reply?.suggestions
    if (!defined(suggestions) ||
      js("(value) => Array.isArray(value)")(suggestions) as Boolean == false)
      return emptyCppCompletionResult()
    val lineNumber = number(position.lineNumber)
    val statementRange = js("(Range, line, start, end) => new Range(line, start, line, end)")(
      monaco.Range,
      lineNumber,
      snapshot.statementStartCharacter + 1,
      snapshot.replacementEndCharacter + 1
    )
    val items = (0 until number(suggestions.length)).mapNotNull { index ->
      val suggestion: dynamic = suggestions[index]
      if (!defined(suggestion)) return@mapNotNull null
      val candidate = suggestion["candidateText"] as? String ?: return@mapNotNull null
      if (candidate.isEmpty()) return@mapNotNull null
      val display = suggestion["displayText"] as? String ?: candidate.trim()
      val rawTokenLength: dynamic = suggestion["tokenLength"]
      val fallbackLength: dynamic = suggestion["length"]
      val tokenLength = cppCompletionInt(
        if (defined(rawTokenLength)) rawTokenLength else fallbackLength,
        0
      )
      val item = js("({})")
      item.label = display
      item.detail = "Tidyparse · shortest C++ statement completion"
      item.documentation = "Generated from the full C++ statement grammar (${tokenLength} tokens)."
      item.kind = monaco.languages.CompletionItemKind.Snippet
      item.insertText = candidate
      item.range = statementRange
      item.filterText = snapshot.prefixText
      if (defined(monaco.languages.CompletionItemInsertTextRule?.KeepWhitespace))
        item.insertTextRules = monaco.languages.CompletionItemInsertTextRule.KeepWhitespace
      item.sortText = "0000_${tokenLength.toString().padStart(2, '0')}_${index.toString().padStart(2, '0')}"
      item.preselect = index == 0
      item
    }
    val result = js("({})")
    result.suggestions = items.toTypedArray()
    result.incomplete = false
    return result
  }

  private fun emptyCppCompletionResult(): dynamic = js("({ suggestions: [], incomplete: false })")

  private fun updateReadOnly() {
    val activeEditor = editor ?: return
    val options = js("{}")
    options.readOnly = requestedReadOnly
    activeEditor.updateOptions(options)
  }

  private fun installDiagnosticsListener(vscode: dynamic) {
    diagnosticsSubscription = vscode.languages.onDidChangeDiagnostics { event: dynamic ->
      val current = documentUri()
      val uris = event.uris
      val containsCurrent =
        (0 until number(uris.length)).any { uris[it].toString() == current }
      if (containsCurrent) emitDiagnostics()
    }
  }

  private fun emitDiagnostics() {
    val vscode = modules?.vscode ?: return
    val uri = vscode.Uri.parse(documentUri())
    val diagnostics = vscode.languages.getDiagnostics(uri)
    cachedGrammarCompletion = null
    cppCompletionContextEpoch++
    latestRawDiagnostics = diagnostics
    val mapped = (0 until number(diagnostics.length)).map { index ->
      val diagnostic = diagnostics[index]
      val codeValue = diagnostic.code
      ClangdDiagnostic(
        range = ClangdRange(
          start = ClangdPosition(
            line = number(diagnostic.range.start.line),
            character = number(diagnostic.range.start.character)
          ),
          end = ClangdPosition(
            line = number(diagnostic.range.end.line),
            character = number(diagnostic.range.end.character)
          )
        ),
        severity = if (defined(diagnostic.severity)) number(diagnostic.severity) + 1 else null,
        message = diagnostic.message as? String ?: "",
        source = diagnostic.source as? String,
        code = when {
          !defined(codeValue) -> null
          defined(codeValue.value) -> codeValue.value.toString()
          else -> codeValue.toString()
        }
      )
    }
    diagnosticsListener(mapped)
    reportStatus(ClangdClientState.READY, "clangd is ready")
  }

  private fun installClangdStatusNotification() {
    val client = languageClientWrapper?.getLanguageClient() ?: return
    client.onNotification(
      "textDocument/clangd.fileStatus",
      { status: dynamic ->
        val uri = status?.uri as? String
        if (uri != null && uri != documentUri()) return@onNotification
        val state = status?.state as? String ?: return@onNotification
        if (state.equals("idle", ignoreCase = true)) {
          reportStatus(ClangdClientState.READY, "clangd is ready")
        } else {
          reportStatus(ClangdClientState.BUSY, "clangd $state")
        }
      }
    )
  }

  private suspend fun awaitClangdReady(clangdWorker: dynamic) {
    val ready = Promise<Unit> { resolve, reject ->
      clangdWorker.onmessage = { event: dynamic ->
        val message = event.data
        when (message?.type as? String) {
          "ready" -> resolve(Unit)
          "status" -> {
            val state =
              if (message.status == "loading") ClangdClientState.LOADING
              else ClangdClientState.INITIALIZING
            reportStatus(state, message.message as? String ?: "Loading clangd…")
          }
          "error" -> {
            val text = message.message as? String ?: "The clangd worker failed"
            reportStatus(ClangdClientState.ERROR, text)
            if (message.fatal == true) {
              disposeClangdRuntime()
              reject(Throwable(text))
            }
          }
        }
      }
      clangdWorker.onerror = { event: dynamic ->
        val text = event.message as? String ?: "The clangd worker failed"
        reportStatus(ClangdClientState.ERROR, text)
        disposeClangdRuntime()
        reject(Throwable(text))
      }
      clangdWorker.onmessageerror = {
        val text = "The clangd worker returned an unreadable message"
        reportStatus(ClangdClientState.ERROR, text)
        disposeClangdRuntime()
        reject(Throwable(text))
      }
    }
    ready.await()
  }

  private fun disposeClangdRuntime(onDisposed: () -> Unit = {}) {
    val wrapper = languageClientWrapper
    languageClientWrapper = null
    val port = languageClientPort
    languageClientPort = null
    val clangdWorker = worker
    worker = null

    val wrapperDisposal = try {
      if (defined(wrapper)) {
        wrapper.dispose()
      } else {
        clangdWorker?.terminate()
        null
      }
    } catch (_: Throwable) {
      try {
        clangdWorker?.terminate()
      } catch (_: Throwable) {
      }
      null
    }
    afterPromiseSettles(wrapperDisposal) {
      try {
        port?.close()
      } catch (_: Throwable) {
      }
      onDisposed()
    }
  }

  /**
   * Keep clangd's semantic services, but do not publish its completion provider to Monaco.
   *
   * Grammar completion still sends narrowly scoped `textDocument/completion` requests directly
   * through the language client to collect semantic names. Middleware only wraps requests made by
   * the language-client feature providers, so returning an empty list here cannot intercept those
   * private context requests or diagnostics, hover, signature help, and AST requests.
   */
  private fun clangdEditorMiddleware(): dynamic =
    js(
      """() => {
        const filterLocations = (document, result) => {
          if (result == null) return result;
          const documentUri = document.uri.toString();
          const belongsToDocument = location => {
            const uri = location && (location.targetUri || location.uri);
            return uri != null && uri.toString() === documentUri;
          };
          return Array.isArray(result)
            ? result.filter(belongsToDocument)
            : belongsToDocument(result) ? result : undefined;
        };
        const filter = (document, result) =>
          Promise.resolve(result).then(value => filterLocations(document, value));
        return {
          provideCompletionItem: () => [],
          provideDefinition: (document, position, token, next) =>
            filter(document, next(document, position, token)),
          provideDeclaration: (document, position, token, next) =>
            filter(document, next(document, position, token)),
          provideTypeDefinition: (document, position, token, next) =>
            filter(document, next(document, position, token)),
          provideImplementation: (document, position, token, next) =>
            filter(document, next(document, position, token)),
          provideReferences: (document, position, options, token, next) =>
            filter(document, next(document, position, options, token))
        };
      }"""
    )()

  private fun createClangdWorker(): dynamic {
    val url = js(
      """(href) => {
        const result = new URL(href);
        result.search = "";
        result.hash = "";
        result.searchParams.set("cpp-worker", "clangd");
        return result.href;
      }"""
    )(window.location.href) as String
    return js("(url, name) => new Worker(url, { name })")(url, CPP_CLANGD_WORKER_NAME)
  }

  private fun documentUri(): String =
    "$CPP_WORKSPACE_URI/${encodePathSegment(fileName)}"

  private fun reportStatus(state: ClangdClientState, message: String) {
    try {
      statusListener(state, message)
    } catch (_: Throwable) {
    }
  }
}

internal fun cppMonacoEditorOptions(): dynamic {
  val options = js("{}")
  options.automaticLayout = true
  options.fontFamily =
    "\"SFMono-Regular\", Consolas, \"Liberation Mono\", Menlo, monospace"
  options.fontSize = 15
  options.lineHeight = 25
  options.tabSize = 2
  options.insertSpaces = true
  options.detectIndentation = false
  options.autoClosingBrackets = "never"
  options.wordBasedSuggestions = "off"
  options.quickSuggestions = js("({ other: true, comments: false, strings: false })")
  options.quickSuggestionsDelay = 200
  options.suggestOnTriggerCharacters = true
  options.acceptSuggestionOnEnter = "on"
  options.parameterHints = js("({ enabled: true, cycle: true })")
  options.hover = js("({ enabled: true, delay: 300, sticky: true })")
  options.inlayHints = js("({ enabled: 'offUnlessPressed' })")
  options["semanticHighlighting.enabled"] = true
  options.bracketPairColorization =
    js("({ enabled: true, independentColorPoolPerBracketType: true })")
  options.guides =
    js("({ bracketPairs: true, bracketPairsHorizontal: 'active', highlightActiveBracketPair: true })")
  options.glyphMargin = true
  options.folding = true
  options.foldingHighlight = true
  options.showFoldingControls = "mouseover"
  options.lightbulb = js("({ enabled: 'on' })")
  options.renderValidationDecorations = "on"
  options.renderWhitespace = "selection"
  options.scrollBeyondLastLine = false
  options.smoothScrolling = true
  options.padding = js("({ top: 14, bottom: 28 })")
  options.minimap = js("({ enabled: false })")
  options.fixedOverflowWidgets = true
  options.occurrencesHighlight = "singleFile"
  options.selectionHighlight = true
  options.links = true
  return options
}

internal fun cppCompletionWidgetTargetWidth(
  longestLabelCharacters: Int,
  typicalHalfwidthCharacterWidth: Double,
  currentWidth: Double,
  maximumWidth: Double
): Double {
  require(longestLabelCharacters >= 0) { "Completion label length must be nonnegative" }
  require(typicalHalfwidthCharacterWidth >= 0.0) { "Completion character width must be nonnegative" }
  val contentWidth = ceil(
    longestLabelCharacters * typicalHalfwidthCharacterWidth + CPP_COMPLETION_WIDGET_CHROME_PX
  )
  return maxOf(currentWidth, contentWidth).coerceAtMost(maximumWidth)
}

private suspend fun awaitPromise(value: dynamic): dynamic =
  (js("(value) => Promise.resolve(value)")(value) as Promise<dynamic>).await()

private fun afterPromiseSettles(value: dynamic, callback: () -> Unit) {
  if (!defined(value)) {
    callback()
    return
  }
  try {
    js(
      """(value, callback) => {
        Promise.resolve(value).then(callback, callback);
      }"""
    )(value, callback)
  } catch (_: Throwable) {
    callback()
  }
}

private fun disposeSafely(disposable: dynamic) {
  if (!defined(disposable)) return
  try {
    disposable.dispose()
  } catch (_: Throwable) {
  }
}

private fun number(value: dynamic): Int = when (value) {
  is Int -> value
  is Number -> value.toInt()
  else -> 0
}

private fun defined(value: dynamic): Boolean =
  value != null && jsTypeOf(value) != "undefined"

private fun encodePathSegment(value: String): String =
  js("encodeURIComponent(value)") as String
