import kotlinx.browser.document
import kotlinx.browser.window
import org.w3c.dom.HTMLButtonElement
import org.w3c.dom.HTMLElement
import org.w3c.dom.HTMLPreElement
import org.w3c.dom.HTMLTextAreaElement
import org.w3c.dom.events.KeyboardEvent
import kotlin.js.Promise
import kotlin.math.max
import kotlin.math.min

private const val RUST_SOURCE_KEY = "tidyparse-rust-source"
private const val RUST_THEME_KEY = "tidyparse-rust-theme"
private const val COMPILER_EXPLORER_API = "https://godbolt.org/api/compiler"
private const val RUST_COMPILER_ID = "r1910"
private const val COMPILER_REQUEST_TIMEOUT_MS = 20_000

private const val DEFAULT_RUST_SOURCE = """use std::fmt;

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}, {})", self.x, self.y)
    }
}

fn describe(point: Point) -> String {
    format!("{point} is {:.2} units from the origin", point.magnitude())
}

fn main() {
    let point = Point::new(3.0, 4.0);
    println!("{}", describe(point));
}
"""

private data class KeywordCompletion(
  val label: String,
  val insertText: String = label,
  val snippet: Boolean = false,
  val detail: String = "Rust keyword"
)

private data class RustExecutionResult(
  val didExecute: Boolean,
  val exitCode: Int,
  val timedOut: Boolean,
  val truncated: Boolean,
  val stdout: String,
  val stderr: String,
  val diagnostics: String
)

private val RUST_KEYWORD_COMPLETIONS = listOf(
  KeywordCompletion("fn", "fn \${1:name}(\${2}) {\n\t\${0}\n}", true, "Function declaration"),
  KeywordCompletion("struct", "struct \${1:Name} {\n\t\${0}\n}", true, "Struct declaration"),
  KeywordCompletion("enum", "enum \${1:Name} {\n\t\${0}\n}", true, "Enum declaration"),
  KeywordCompletion("trait", "trait \${1:Name} {\n\t\${0}\n}", true, "Trait declaration"),
  KeywordCompletion("impl", "impl \${1:Type} {\n\t\${0}\n}", true, "Implementation block"),
  KeywordCompletion("match", "match \${1:value} {\n\t\${2:pattern} => \${0},\n}", true),
  KeywordCompletion("if", "if \${1:condition} {\n\t\${0}\n}", true),
  KeywordCompletion("let", "let \${1:name} = \${0};", true),
  KeywordCompletion("for", "for \${1:item} in \${2:items} {\n\t\${0}\n}", true),
  KeywordCompletion("while", "while \${1:condition} {\n\t\${0}\n}", true),
  KeywordCompletion("loop", "loop {\n\t\${0}\n}", true),
  KeywordCompletion("mod"),
  KeywordCompletion("pub"),
  KeywordCompletion("use"),
  KeywordCompletion("const"),
  KeywordCompletion("static"),
  KeywordCompletion("type"),
  KeywordCompletion("where"),
  KeywordCompletion("async"),
  KeywordCompletion("await"),
  KeywordCompletion("move"),
  KeywordCompletion("return"),
  KeywordCompletion("Self"),
  KeywordCompletion("self")
)

fun rustSetup() {
  try {
    RustPlayground().start()
  } catch (failure: Throwable) {
    val status = document.getElementById("glancer-status") as? HTMLElement
    status?.setAttribute("data-state", "error")
    status?.querySelector(".status-text")?.textContent =
      failure.message ?: "Unable to start the Rust playground"
    throw failure
  }
}

private class RustPlayground {
  private lateinit var editorHost: HTMLElement
  private lateinit var status: HTMLElement
  private lateinit var analysisSummary: HTMLElement
  private lateinit var diagnosticCount: HTMLElement
  private lateinit var diagnostics: HTMLElement
  private lateinit var cursorPosition: HTMLElement
  private lateinit var engineName: HTMLElement
  private lateinit var engineRevision: HTMLElement
  private lateinit var engineCapabilities: HTMLElement
  private lateinit var modeBadge: HTMLElement
  private lateinit var runButton: HTMLButtonElement
  private lateinit var resetButton: HTMLButtonElement
  private lateinit var programInput: HTMLTextAreaElement
  private lateinit var executionStatus: HTMLElement
  private lateinit var executionOutput: HTMLPreElement
  private lateinit var executionOutputMeta: HTMLElement
  private lateinit var executionDiagnostics: HTMLPreElement
  private lateinit var executionDiagnosticsMeta: HTMLElement

  private var monaco: dynamic = null
  private var editor: dynamic = null
  private var model: dynamic = null
  private lateinit var client: RustGlancerClient

  private var darkTheme = false
  private var analysisTimer = 0
  private var latestVersion = -1
  private var latestAnalysis: dynamic = null
  private var pendingVersion = -1
  private var pendingAnalysis: dynamic = null

  fun start() {
    bindElements()
    applyInitialTheme()
    monaco = loadMonaco()
    client = RustGlancerClient(::setRuntimeStatus)
    createEditor()
    installProviders()
    bindEvents()
    requestAnalysisNow()
    editor.focus()
  }

  private fun bindElements() {
    editorHost = element("rust-editor")
    status = element("glancer-status")
    analysisSummary = element("analysis-summary")
    diagnosticCount = element("diagnostic-count")
    diagnostics = element("diagnostics")
    cursorPosition = element("cursor-position")
    engineName = element("engine-name")
    engineRevision = element("engine-revision")
    engineCapabilities = element("engine-capabilities")
    modeBadge = element("glancer-mode")
    runButton = element("run-source") as HTMLButtonElement
    resetButton = element("reset-source") as HTMLButtonElement
    programInput = element("program-input") as HTMLTextAreaElement
    executionStatus = element("execution-status")
    executionOutput = element("execution-output") as HTMLPreElement
    executionOutputMeta = element("execution-output-meta")
    executionDiagnostics = element("execution-diagnostics") as HTMLPreElement
    executionDiagnosticsMeta = element("execution-diagnostics-meta")
  }

  private fun loadMonaco(): dynamic {
    configureRustMonacoWorkers()
    // Import Monaco's API and only the editor contributions used by this demo. The all-in-one
    // entrypoint also bundles every built-in language worker and Monaco's optional LSP client.
    val loaded = js("require('vanilla-monaco-editor/esm/vs/editor/editor.api.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/bracketMatching/browser/bracketMatching.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/clipboard/browser/clipboard.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/comment/browser/comment.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/contextmenu/browser/contextmenu.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/documentSymbols/browser/documentSymbols.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/find/browser/findController.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/folding/browser/folding.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/gotoSymbol/browser/goToCommands.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/gotoSymbol/browser/link/goToDefinitionAtPosition.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/hover/browser/hoverContribution.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/indentation/browser/indentation.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/linesOperations/browser/linesOperations.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/multicursor/browser/multicursor.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/suggest/browser/suggestController.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/tokenization/browser/tokenization.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/wordOperations/browser/wordOperations.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/standalone/browser/quickAccess/standaloneGotoSymbolQuickAccess.js')")
    js("require('vanilla-monaco-editor/esm/vs/basic-languages/rust/rust.contribution.js')")
    return loaded
  }

  private fun createEditor() {
    val uri = monaco.Uri.parse("file:///workspace/src/main.rs")
    model = monaco.editor.createModel(initialSource(), "rust", uri)
    val options = js("({})")
    options.model = model
    options.theme = if (darkTheme) "vs-dark" else "vs"
    options.automaticLayout = true
    options.fontFamily = "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace"
    options.fontSize = 14
    options.lineHeight = 22
    options.minimap = js("({ enabled: false })")
    options.padding = js("({ top: 12, bottom: 12 })")
    options.renderWhitespace = "selection"
    options.scrollBeyondLastLine = false
    options.smoothScrolling = true
    options.tabSize = 4
    options.insertSpaces = true
    options.quickSuggestions = js("({ other: true, comments: false, strings: false })")
    options.suggest = js("({ showKeywords: true, showSnippets: true })")
    editor = monaco.editor.create(editorHost, options)
  }

  private fun bindEvents() {
    model.onDidChangeContent {
      saveSource(model.getValue() as String)
      latestVersion = -1
      analysisSummary.textContent = "Analyzing current source…"
      if (analysisTimer != 0) window.clearTimeout(analysisTimer)
      analysisTimer = window.setTimeout({ requestAnalysisNow() }, 180)
    }

    editor.onDidChangeCursorPosition { event: dynamic ->
      val position = event.position
      cursorPosition.textContent =
        "Ln ${(position.lineNumber as Number).toInt()}, Col ${(position.column as Number).toInt()}"
    }

    resetButton.onclick = {
      model.setValue(DEFAULT_RUST_SOURCE)
      editor.setPosition(js("({ lineNumber: 1, column: 1 })"))
      editor.focus()
      Unit
    }

    (element("toggle-theme") as HTMLButtonElement).onclick = {
      setTheme(!darkTheme)
      Unit
    }

    runButton.onclick = {
      compileAndRun()
      Unit
    }

    val runKey = (monaco.KeyMod.CtrlCmd as Number).toInt() or
      (monaco.KeyCode.Enter as Number).toInt()
    editor.addCommand(runKey, { compileAndRun() })
    bindExecutionTabs()
  }

  private fun bindExecutionTabs() {
    val tabs = document.querySelectorAll("#execution-tabs [data-tab]")
    for (index in 0 until tabs.length) {
      val tab = tabs.item(index) as HTMLButtonElement
      tab.onclick = {
        activateExecutionTab(tab.getAttribute("data-tab") ?: "input")
        Unit
      }
      tab.addEventListener("keydown", { rawEvent ->
        val event = rawEvent as KeyboardEvent
        val targetIndex = when (event.key) {
          "ArrowLeft" -> (index - 1 + tabs.length) % tabs.length
          "ArrowRight" -> (index + 1) % tabs.length
          "Home" -> 0
          "End" -> tabs.length - 1
          else -> -1
        }
        if (targetIndex >= 0) {
          event.preventDefault()
          (tabs.item(targetIndex) as HTMLButtonElement).also { next ->
            next.focus()
            next.click()
          }
        }
      })
    }
  }

  private fun installProviders() {
    val completionProvider = js("({})")
    completionProvider.triggerCharacters = arrayOf(".", ":")
    completionProvider.provideCompletionItems =
      { requestedModel: dynamic, position: dynamic, _: dynamic, _: dynamic ->
        val promise = analysisFor(requestedModel)
        promise.then { analysis: dynamic -> completionResult(requestedModel, position, analysis) }
      }
    monaco.languages.registerCompletionItemProvider("rust", completionProvider)

    val hoverProvider = js("({})")
    hoverProvider.provideHover = { requestedModel: dynamic, position: dynamic, _: dynamic ->
      val promise = analysisFor(requestedModel)
      promise.then { analysis: dynamic -> hoverResult(requestedModel, position, analysis) }
    }
    monaco.languages.registerHoverProvider("rust", hoverProvider)

    val definitionProvider = js("({})")
    definitionProvider.provideDefinition = { requestedModel: dynamic, position: dynamic, _: dynamic ->
      val promise = analysisFor(requestedModel)
      promise.then { analysis: dynamic -> definitionResult(requestedModel, position, analysis) }
    }
    monaco.languages.registerDefinitionProvider("rust", definitionProvider)

    val symbolProvider = js("({})")
    symbolProvider.provideDocumentSymbols = { requestedModel: dynamic, _: dynamic ->
      val promise = analysisFor(requestedModel)
      promise.then { analysis: dynamic -> documentSymbols(requestedModel, analysis.symbols) }
    }
    monaco.languages.registerDocumentSymbolProvider("rust", symbolProvider)
  }

  private fun analysisFor(requestedModel: dynamic): dynamic {
    val version = (requestedModel.getVersionId() as Number).toInt()
    if (latestVersion == version && latestAnalysis != null) {
      return js("(value) => Promise.resolve(value)")(latestAnalysis)
    }
    if (pendingVersion == version && pendingAnalysis != null) return pendingAnalysis

    pendingVersion = version
    val source = requestedModel.getValue() as String
    val requested = client.analyze(source)
    val handled = requested.then(
      { analysis: dynamic ->
        if (pendingVersion == version) {
          pendingVersion = -1
          pendingAnalysis = null
        }
        if ((requestedModel.getVersionId() as Number).toInt() == version) {
          publishAnalysis(analysis, version)
        }
        analysis
      },
      { failure: dynamic ->
        if (pendingVersion == version) {
          pendingVersion = -1
          pendingAnalysis = null
        }
        val message = failure?.message as? String ?: "Rust Glancer analysis failed"
        setRuntimeStatus("error", message)
        throw Throwable(message)
      }
    )
    pendingAnalysis = handled
    return handled
  }

  private fun requestAnalysisNow() {
    analysisTimer = 0
    val promise = analysisFor(model)
    promise.then({ _: dynamic -> Unit }, { _: dynamic -> Unit })
  }

  private fun publishAnalysis(analysis: dynamic, version: Int) {
    latestAnalysis = analysis
    latestVersion = version
    if (analysis?.ok as? Boolean != true) {
      val message = analysis?.error as? String ?: "Rust Glancer returned no analysis"
      setRuntimeStatus("error", message)
      return
    }

    applyDiagnostics(analysis.diagnostics)
    renderEngine(analysis.engine)
    val diagnosticTotal = arrayLength(analysis.diagnostics)
    val symbolTotal = arrayLength(analysis.symbols)
    analysisSummary.textContent =
      "$diagnosticTotal ${plural(diagnosticTotal, "diagnostic")} · " +
        "$symbolTotal top-level ${plural(symbolTotal, "symbol")}"
  }

  private fun applyDiagnostics(items: dynamic) {
    val markers = (0 until arrayLength(items)).map { index ->
      val item = items[index]
      val start = int(item.start)
      val rawEnd = int(item.end)
      val end = if (rawEnd > start) rawEnd else min(start + 1, int(model.getValueLength()))
      val range = rangeForOffsets(model, start, max(start, end))
      val marker = js("({})")
      marker.startLineNumber = range.startLineNumber
      marker.startColumn = range.startColumn
      marker.endLineNumber = range.endLineNumber
      marker.endColumn = range.endColumn
      marker.message = item.message as? String ?: "Rust syntax error"
      marker.severity = monaco.MarkerSeverity.Error
      marker.source = "Rust Glancer"
      marker
    }.toTypedArray()
    monaco.editor.setModelMarkers(model, "rust-glancer", markers)
    renderDiagnostics(items)
  }

  private fun renderDiagnostics(items: dynamic) {
    val count = arrayLength(items)
    diagnosticCount.textContent = count.toString()
    diagnostics.textContent = ""
    if (count == 0) {
      val empty = document.createElement("p") as HTMLElement
      empty.className = "empty-state"
      empty.textContent = "No Rust syntax errors. The document outline and local index are current."
      diagnostics.appendChild(empty)
      return
    }

    for (index in 0 until count) {
      val item = items[index]
      val start = int(item.start)
      val position = model.getPositionAt(start)
      val button = document.createElement("button") as HTMLButtonElement
      button.className = "diagnostic"
      val title = document.createElement("strong") as HTMLElement
      title.textContent = "Ln ${int(position.lineNumber)}, Col ${int(position.column)}"
      val message = document.createElement("span") as HTMLElement
      message.textContent = item.message as? String ?: "Rust syntax error"
      button.append(title, message)
      button.onclick = {
        editor.setPosition(position)
        editor.revealPositionInCenter(position)
        editor.focus()
        Unit
      }
      diagnostics.appendChild(button)
    }
  }

  private fun renderEngine(engine: dynamic) {
    engineName.textContent = engine?.name as? String ?: "Rust Glancer"
    val revision = engine?.revision as? String ?: "unknown"
    engineRevision.textContent = revision.take(8)
    engineRevision.title = revision
    modeBadge.textContent = engine?.mode as? String ?: "syntax prototype"
    engineCapabilities.textContent = ""
    val capabilities = engine?.capabilities
    for (index in 0 until arrayLength(capabilities)) {
      val item = document.createElement("span") as HTMLElement
      item.className = "capability"
      item.textContent = capabilities[index] as? String ?: continue
      engineCapabilities.appendChild(item)
    }
  }

  private fun compileAndRun() {
    if (runButton.disabled) return

    setRunning(true)
    setExecutionStatus("running", "Compiling…")
    executionOutputMeta.textContent = "Waiting for Rust 1.91…"
    executionDiagnosticsMeta.textContent = "Waiting for compiler diagnostics…"
    val startedAt = window.performance.now()

    try {
      val request = runCode(model.getValue() as String, programInput.value)
      request.then(
        { result: RustExecutionResult ->
          val elapsed = (window.performance.now() - startedAt).toInt()
          renderExecutionResult(result, elapsed)
          setRunning(false)
          Unit
        },
        { failure: dynamic ->
          renderExecutionFailure(failure?.message as? String ?: "The compiler request failed.")
          setRunning(false)
          Unit
        }
      )
    } catch (failure: Throwable) {
      renderExecutionFailure(failure.message ?: "The compiler request failed.")
      setRunning(false)
    }
  }

  private fun runCode(code: String, input: String): dynamic {
    val request = js("({})")
    request.source = code
    request.compiler = RUST_COMPILER_ID
    request.lang = "rust"
    request.options = js("({})")
    request.options.userArguments = "--edition=2024"
    request.options.executeParameters = js("({})")
    request.options.executeParameters.args = emptyArray<String>()
    request.options.executeParameters.stdin = input
    request.options.compilerOptions = js("({ executorRequest: true })")
    request.options.filters = js("({ execute: true })")

    val init = js("({})")
    init.method = "POST"
    init.headers = js("({ 'Content-Type': 'application/json', 'Accept': 'application/json' })")
    init.body = JSON.stringify(request)
    val controller = js("new AbortController()")
    init.signal = controller.signal
    val timeout = window.setTimeout({ controller.abort() }, COMPILER_REQUEST_TIMEOUT_MS)
    val fetched: dynamic = window.asDynamic().fetch(
      "$COMPILER_EXPLORER_API/$RUST_COMPILER_ID/compile",
      init
    )

    return fetched.then(
      { response: dynamic ->
        window.clearTimeout(timeout)
        if (!(response.ok as Boolean)) {
          throw IllegalStateException("Compiler Explorer returned HTTP ${response.status}.")
        }
        response.json()
      },
      { failure: dynamic ->
        window.clearTimeout(timeout)
        if (controller.signal.aborted as Boolean) {
          throw IllegalStateException(
            "Compiler request timed out after ${COMPILER_REQUEST_TIMEOUT_MS / 1_000} seconds."
          )
        }
        throw failure
      }
    ).then { payload: dynamic -> parseExecutionResult(payload) }
  }

  private fun parseExecutionResult(payload: dynamic): RustExecutionResult {
    val buildResult = payload.buildResult
    val didExecute = payload.didExecute as? Boolean ?: false
    val buildDiagnostics = stdioToString(buildResult?.stderr)
    val topLevelStderr = stdioToString(payload.stderr)
    val requestError = payload.error as? String
    if (!requestError.isNullOrBlank() && !defined(buildResult)) {
      throw IllegalStateException(requestError)
    }
    val diagnosticText = listOfNotNull(
      requestError?.takeIf { it.isNotBlank() },
      buildDiagnostics.takeIf { it.isNotBlank() },
      topLevelStderr.takeIf { !didExecute && buildDiagnostics.isBlank() && it.isNotBlank() }
    ).distinct().joinToString("\n")

    return RustExecutionResult(
      didExecute = didExecute,
      exitCode = (payload.code as? Number)?.toInt()
        ?: (buildResult?.code as? Number)?.toInt()
        ?: -1,
      timedOut = payload.timedOut as? Boolean ?: false,
      truncated = payload.truncated as? Boolean ?: false,
      stdout = stripAnsi(stdioToString(payload.stdout)),
      stderr = stripAnsi(topLevelStderr),
      diagnostics = stripAnsi(diagnosticText)
    )
  }

  private fun renderExecutionResult(result: RustExecutionResult, elapsed: Int) {
    executionOutput.textContent = formatProgramOutput(result)
    executionOutputMeta.textContent = when {
      !result.didExecute -> "The program was not executed."
      result.timedOut -> "Execution timed out after $elapsed ms."
      result.truncated -> "Process exited with code ${result.exitCode}; output was truncated."
      result.exitCode == 0 -> "Process exited successfully in $elapsed ms."
      else -> "Process exited with code ${result.exitCode} in $elapsed ms."
    }

    executionDiagnostics.textContent = when {
      result.diagnostics.isNotBlank() -> result.diagnostics
      result.didExecute -> "Compilation succeeded with no diagnostics."
      else -> "Compilation failed without diagnostics."
    }
    executionDiagnosticsMeta.textContent = when {
      result.diagnostics.isNotBlank() -> "rustc diagnostics from Compiler Explorer."
      result.didExecute -> "rustc reported no diagnostics."
      else -> "rustc did not return compiler diagnostics."
    }

    if (result.didExecute) {
      activateExecutionTab("output")
      setExecutionStatus(
        if (result.exitCode == 0 && !result.timedOut && !result.truncated) "success" else "error",
        when {
          result.timedOut -> "Timed out"
          result.truncated -> "Truncated"
          else -> "Exit ${result.exitCode}"
        }
      )
    } else {
      activateExecutionTab("diagnostics")
      setExecutionStatus("error", "Build failed")
    }
  }

  private fun renderExecutionFailure(message: String) {
    executionDiagnostics.textContent = message
    executionDiagnosticsMeta.textContent = "Unable to compile this program."
    activateExecutionTab("diagnostics")
    setExecutionStatus("error", "Request failed")
  }

  private fun formatProgramOutput(result: RustExecutionResult): String {
    if (result.stdout.isBlank() && result.stderr.isBlank()) {
      return if (result.didExecute) "Program produced no output." else "Program did not execute."
    }
    return buildString {
      if (result.stdout.isNotBlank()) append(result.stdout)
      if (result.stderr.isNotBlank()) {
        if (isNotEmpty()) append("\n")
        append(result.stderr)
      }
    }
  }

  private fun activateExecutionTab(name: String) {
    val tabs = document.querySelectorAll("#execution-tabs [data-tab]")
    for (index in 0 until tabs.length) {
      val tab = tabs.item(index) as HTMLElement
      val active = tab.getAttribute("data-tab") == name
      tab.setAttribute("aria-selected", active.toString())
      tab.tabIndex = if (active) 0 else -1
    }
    listOf("input", "output", "diagnostics").forEach { panelName ->
      element("execution-tab-$panelName").classList.toggle("is-hidden", panelName != name)
    }
  }

  private fun setRunning(running: Boolean) {
    runButton.disabled = running
    resetButton.disabled = running
    programInput.readOnly = running
    val options = js("({})")
    options.readOnly = running
    editor.updateOptions(options)
    element("rust-app").classList.toggle("is-running", running)
    runButton.querySelector(".run-label")?.textContent = if (running) "Running…" else "Run"
  }

  private fun setExecutionStatus(state: String, message: String) {
    executionStatus.setAttribute("data-state", state)
    executionStatus.textContent = message
  }

  private fun completionResult(requestedModel: dynamic, position: dynamic, analysis: dynamic): dynamic {
    val word = requestedModel.getWordUntilPosition(position)
    val range = js("({})")
    range.startLineNumber = position.lineNumber
    range.endLineNumber = position.lineNumber
    range.startColumn = word.startColumn
    range.endColumn = word.endColumn

    val suggestions = mutableListOf<dynamic>()
    val glancerItems = analysis?.completions
    for (index in 0 until arrayLength(glancerItems)) {
      val source = glancerItems[index]
      val label = source.label as? String ?: continue
      val item = js("({})")
      item.label = label
      item.insertText = label
      item.kind = completionKind(source.kind as? String)
      item.detail = "Rust Glancer · ${source.detail as? String ?: source.kind as? String ?: "local symbol"}"
      item.range = range
      item.sortText = "0_${label.lowercase()}"
      suggestions.add(item)
    }

    RUST_KEYWORD_COMPLETIONS.forEach { keyword ->
      val item = js("({})")
      item.label = keyword.label
      item.insertText = keyword.insertText
      item.kind = if (keyword.snippet) monaco.languages.CompletionItemKind.Snippet
      else monaco.languages.CompletionItemKind.Keyword
      item.detail = keyword.detail
      item.range = range
      item.sortText = "1_${keyword.label}"
      if (keyword.snippet) {
        item.insertTextRules = monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
      }
      suggestions.add(item)
    }

    val result = js("({})")
    result.suggestions = suggestions.toTypedArray()
    result.incomplete = false
    return result
  }

  private fun hoverResult(requestedModel: dynamic, position: dynamic, analysis: dynamic): dynamic {
    val offset = int(requestedModel.getOffsetAt(position))
    val occurrence = occurrenceAt(analysis, offset) ?: return null
    val detail = occurrence.detail as? String ?: occurrence.name as? String ?: "Rust symbol"
    val heading = js("({})")
    heading.value = "**${occurrence.name as? String ?: "Rust symbol"}** · `${occurrence.kind as? String ?: "symbol"}`"
    val code = js("({})")
    code.value = "```rust\n${detail.replace("```", "` ` `")}\n```"
    val provenance = js("({})")
    provenance.value = "_Rust Glancer browser syntax model_"
    val result = js("({})")
    result.range = rangeForOffsets(requestedModel, int(occurrence.start), int(occurrence.end))
    result.contents = arrayOf(heading, code, provenance)
    return result
  }

  private fun definitionResult(requestedModel: dynamic, position: dynamic, analysis: dynamic): dynamic {
    val offset = int(requestedModel.getOffsetAt(position))
    val occurrence = occurrenceAt(analysis, offset) ?: return null
    val startValue = occurrence.declarationStart
    val endValue = occurrence.declarationEnd
    if (!defined(startValue) || !defined(endValue)) return null
    val location = js("({})")
    location.uri = requestedModel.uri
    location.range = rangeForOffsets(requestedModel, int(startValue), int(endValue))
    return location
  }

  private fun occurrenceAt(analysis: dynamic, offset: Int): dynamic {
    val occurrences = analysis?.occurrences
    var best: dynamic = null
    var bestWidth = Int.MAX_VALUE
    for (index in 0 until arrayLength(occurrences)) {
      val occurrence = occurrences[index]
      val start = int(occurrence.start)
      val end = int(occurrence.end)
      if (offset < start || offset > end) continue
      val width = end - start
      if (width < bestWidth) {
        best = occurrence
        bestWidth = width
      }
    }
    return best
  }

  private fun documentSymbols(requestedModel: dynamic, items: dynamic): Array<dynamic> =
    (0 until arrayLength(items)).map { index ->
      val source = items[index]
      val item = js("({})")
      item.name = source.name as? String ?: "<anonymous>"
      item.detail = source.detail as? String ?: ""
      item.kind = symbolKind(source.kind as? String)
      item.range = rangeForOffsets(requestedModel, int(source.start), int(source.end))
      item.selectionRange = rangeForOffsets(
        requestedModel,
        int(source.selectionStart),
        int(source.selectionEnd)
      )
      item.children = documentSymbols(requestedModel, source.children)
      item
    }.toTypedArray()

  private fun symbolKind(kind: String?): dynamic = when (kind) {
    "const", "static" -> monaco.languages.SymbolKind.Constant
    "enum" -> monaco.languages.SymbolKind.Enum
    "variant" -> monaco.languages.SymbolKind.EnumMember
    "field" -> monaco.languages.SymbolKind.Field
    "fn" -> monaco.languages.SymbolKind.Function
    "method" -> monaco.languages.SymbolKind.Method
    "module" -> monaco.languages.SymbolKind.Module
    "struct", "union" -> monaco.languages.SymbolKind.Struct
    "trait" -> monaco.languages.SymbolKind.Interface
    "type_alias" -> monaco.languages.SymbolKind.TypeParameter
    "variable" -> monaco.languages.SymbolKind.Variable
    else -> monaco.languages.SymbolKind.Object
  }

  private fun completionKind(kind: String?): dynamic = when (kind) {
    "const", "static" -> monaco.languages.CompletionItemKind.Constant
    "enum" -> monaco.languages.CompletionItemKind.Enum
    "variant" -> monaco.languages.CompletionItemKind.EnumMember
    "field" -> monaco.languages.CompletionItemKind.Field
    "fn" -> monaco.languages.CompletionItemKind.Function
    "method" -> monaco.languages.CompletionItemKind.Method
    "module" -> monaco.languages.CompletionItemKind.Module
    "struct", "union" -> monaco.languages.CompletionItemKind.Struct
    "trait" -> monaco.languages.CompletionItemKind.Interface
    "type_alias" -> monaco.languages.CompletionItemKind.TypeParameter
    "variable" -> monaco.languages.CompletionItemKind.Variable
    else -> monaco.languages.CompletionItemKind.Text
  }

  private fun rangeForOffsets(requestedModel: dynamic, rawStart: Int, rawEnd: Int): dynamic {
    val length = int(requestedModel.getValueLength())
    val start = rawStart.coerceIn(0, length)
    val end = rawEnd.coerceIn(start, length)
    val startPosition = requestedModel.getPositionAt(start)
    val endPosition = requestedModel.getPositionAt(end)
    return monaco.Range.fromPositions(startPosition, endPosition)
  }

  private fun applyInitialTheme() {
    val saved = try {
      window.localStorage.getItem(RUST_THEME_KEY)
    } catch (_: Throwable) {
      null
    }
    darkTheme = when (saved) {
      "dark" -> true
      "light" -> false
      else -> window.matchMedia("(prefers-color-scheme: dark)").matches
    }
    document.documentElement?.setAttribute("data-theme", if (darkTheme) "dark" else "light")
  }

  private fun setTheme(dark: Boolean) {
    darkTheme = dark
    val name = if (dark) "dark" else "light"
    document.documentElement?.setAttribute("data-theme", name)
    monaco.editor.setTheme(if (dark) "vs-dark" else "vs")
    try {
      window.localStorage.setItem(RUST_THEME_KEY, name)
    } catch (_: Throwable) {
    }
  }

  private fun initialSource(): String = try {
    window.localStorage.getItem(RUST_SOURCE_KEY)?.takeIf { it.isNotBlank() } ?: DEFAULT_RUST_SOURCE
  } catch (_: Throwable) {
    DEFAULT_RUST_SOURCE
  }

  private fun saveSource(source: String) {
    try {
      window.localStorage.setItem(RUST_SOURCE_KEY, source)
    } catch (_: Throwable) {
    }
  }

  private fun setRuntimeStatus(state: String, message: String) {
    status.setAttribute("data-state", state)
    status.title = message
    status.querySelector(".status-text")?.textContent = message
  }

  private fun element(id: String): HTMLElement =
    document.getElementById(id) as? HTMLElement ?: error("Missing #$id")

  private fun arrayLength(value: dynamic): Int =
    if (!defined(value)) 0 else (value.length as? Number)?.toInt() ?: 0

  private fun int(value: dynamic): Int = (value as? Number)?.toInt() ?: 0

  private fun defined(value: dynamic): Boolean =
    js("(value) => value !== undefined && value !== null")(value) as Boolean

  private fun plural(count: Int, noun: String): String = if (count == 1) noun else "${noun}s"
}

private fun stdioToString(entries: dynamic): String {
  if (entries == null || entries == js("undefined")) return ""
  val length = (entries.length as? Number)?.toInt() ?: return ""
  return (0 until length)
    .mapNotNull { index -> entries[index]?.text as? String }
    .joinToString("\n")
}

private fun stripAnsi(text: String): String =
  text.replace(Regex("\u001B\\[[0-?]*[ -/]*[@-~]"), "")
