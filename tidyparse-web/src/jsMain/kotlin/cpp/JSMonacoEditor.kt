import kotlinx.browser.window
import kotlinx.coroutines.await
import org.w3c.dom.HTMLElement
import kotlin.js.Promise

private const val CPP_WORKSPACE_PATH = "/home/web_user"
private const val CPP_WORKSPACE_URI = "file://$CPP_WORKSPACE_PATH"
private const val CPP_CLANGD_WORKER_NAME = "tidyparse-clangd"

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
      config.clientOptions.middleware = sameDocumentNavigationMiddleware()
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

  private fun editorOptions(): dynamic {
    val options = js("{}")
    options.automaticLayout = true
    options.fontFamily =
      "\"SFMono-Regular\", Consolas, \"Liberation Mono\", Menlo, monospace"
    options.fontSize = 15
    options.lineHeight = 25
    options.tabSize = 2
    options.insertSpaces = true
    options.detectIndentation = false
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

  private fun userConfigurationJson(): String {
    val configuration = js("{}")
    configuration["workbench.colorTheme"] =
      if (darkTheme) "Default Dark Modern" else "Default Light Modern"
    configuration["editor.wordBasedSuggestions"] = "off"
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

    val runKey = number(monaco.KeyMod.CtrlCmd) or number(monaco.KeyCode.Enter)
    activeEditor.addCommand(runKey, { onRun() })
    activeEditor.setPosition(js("({ lineNumber: 1, column: 1 })"))
    updateReadOnly()
  }

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

  // Keep location-based features, including Peek, inside their source document.
  private fun sameDocumentNavigationMiddleware(): dynamic =
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
