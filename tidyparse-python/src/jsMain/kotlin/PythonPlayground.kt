import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.promise
import org.w3c.dom.HTMLButtonElement
import org.w3c.dom.HTMLElement
import org.w3c.dom.HTMLPreElement
import org.w3c.dom.HTMLTextAreaElement
import org.w3c.dom.events.KeyboardEvent
import kotlin.math.max

private const val PYTHON_SOURCE_KEY = "tidyparse-python3-source"
private const val PYTHON_THEME_KEY = "tidyparse-python3-theme"

private data class RepairCompletionInvocation(
  val requestId: Int,
  val modelVersion: Int,
  val line: Int,
  val column: Int
)

private const val DEFAULT_PYTHON_SOURCE = """from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def magnitude(self) -> float:
        return hypot(self.x, self.y)


def average_distance(points: Iterable[Point]) -> float:
    distances = [point.magnitude for point in points]
    return sum(distances) / len(distances) if distances else 0.0


points = [Point(3.0, 4.0), Point(5.0, 12.0)]
print(f"Average distance: {average_distance(points):.2f}")
"""

fun pythonSetup() {
  MainScope().launch {
    try {
      PythonPlayground().start()
    } catch (failure: Throwable) {
      val status = document.getElementById("ty-status") as? HTMLElement
      status?.setAttribute("data-state", "error")
      status?.querySelector(".status-text")?.textContent =
        failure.message ?: "Unable to start the Python playground"
      console.error("Unable to start the Python playground", failure)
    }
  }
}

private class PythonPlayground {
  private val scope = MainScope()
  private lateinit var editorHost: HTMLElement
  private lateinit var status: HTMLElement
  private lateinit var repairStatus: HTMLElement
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
  private lateinit var engine: TyEngine
  private lateinit var runner: PythonRunner
  private lateinit var repairClient: PythonSyntaxRepairClient

  private var darkTheme = false
  private var analysisTimer = 0
  private var engineReady = false
  private var engineSourceVersion = -1
  private var repairRequestId = 0
  private var armedRepair: RepairCompletionInvocation? = null

  suspend fun start() {
    bindElements()
    applyInitialTheme()
    monaco = loadMonaco()
    createEditor()
    repairClient = PythonSyntaxRepairClient(::setRepairStatus)
    bindEvents()

    engine = TyEngine(::setRuntimeStatus)
    runner = PythonRunner { state, message ->
      if (state == "error") setExecutionStatus("error", message)
    }
    scope.launch {
      try {
        repairClient.initialize()
      } catch (failure: Throwable) {
        console.warn("Syntax repair will remain unavailable", failure)
      }
    }

    val initializedAtVersion = modelVersion()
    val initial = engine.initialize(model.getValue() as String)
    engineReady = true
    engineSourceVersion = initializedAtVersion
    installProviders()

    val snapshot = if (modelVersion() == initializedAtVersion) initial else synchronizeEngine()
    publishAnalysis(snapshot, modelVersion())
    renderEngine(initial)
    editor.focus()
  }

  private fun bindElements() {
    editorHost = element("python-editor")
    status = element("ty-status")
    repairStatus = element("repair-status")
    analysisSummary = element("analysis-summary")
    diagnosticCount = element("diagnostic-count")
    diagnostics = element("diagnostics")
    cursorPosition = element("cursor-position")
    engineName = element("engine-name")
    engineRevision = element("engine-revision")
    engineCapabilities = element("engine-capabilities")
    modeBadge = element("ty-mode")
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
    configurePythonMonacoWorkers()
    val loaded = js("require('vanilla-monaco-editor/esm/vs/editor/editor.api.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/bracketMatching/browser/bracketMatching.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/clipboard/browser/clipboard.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/comment/browser/comment.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/contextmenu/browser/contextmenu.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/find/browser/findController.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/folding/browser/folding.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/format/browser/formatActions.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/gotoSymbol/browser/goToCommands.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/gotoSymbol/browser/link/goToDefinitionAtPosition.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/hover/browser/hoverContribution.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/inlayHints/browser/inlayHintsContribution.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/indentation/browser/indentation.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/linesOperations/browser/linesOperations.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/multicursor/browser/multicursor.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/suggest/browser/suggestController.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/tokenization/browser/tokenization.js')")
    js("require('vanilla-monaco-editor/esm/vs/editor/contrib/wordOperations/browser/wordOperations.js')")
    js("require('vanilla-monaco-editor/esm/vs/basic-languages/python/python.contribution.js')")
    return loaded
  }

  private fun createEditor() {
    val uri = monaco.Uri.parse("file:///workspace/main.py")
    model = monaco.editor.createModel(initialSource(), "python", uri)

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
    options.detectIndentation = false
    options.autoClosingBrackets = "never"
    options.quickSuggestions = false
    options.suggestOnTriggerCharacters = false
    options.wordBasedSuggestions = "off"
    options.suggest = js("({ showKeywords: false, showSnippets: false })")
    options.inlayHints = js("({ enabled: 'on' })")
    editor = monaco.editor.create(editorHost, options)
  }

  private fun bindEvents() {
    model.onDidChangeContent {
      armedRepair = null
      repairRequestId++
      saveSource(model.getValue() as String)
      analysisSummary.textContent = if (engineReady) "Analyzing current source…" else "Loading ty…"
      if (analysisTimer != 0) window.clearTimeout(analysisTimer)
      if (engineReady) analysisTimer = window.setTimeout({ requestAnalysisNow() }, 160)
    }

    editor.onDidChangeCursorPosition { event: dynamic ->
      val position = event.position
      cursorPosition.textContent =
        "Ln ${int(position.lineNumber)}, Col ${int(position.column)}"
    }

    resetButton.onclick = {
      model.setValue(DEFAULT_PYTHON_SOURCE)
      editor.setPosition(js("({ lineNumber: 1, column: 1 })"))
      editor.focus()
      Unit
    }

    (element("toggle-theme") as HTMLButtonElement).onclick = {
      setTheme(!darkTheme)
      Unit
    }

    runButton.onclick = {
      runPython()
      Unit
    }

    val runKey = int(monaco.KeyMod.CtrlCmd) or int(monaco.KeyCode.Enter)
    editor.addCommand(runKey, { runPython() })

    val triggerRepairCompletion = {
      val position = editor.getPosition()
      if (defined(position)) {
        repairRequestId++
        armedRepair = RepairCompletionInvocation(
          requestId = repairRequestId,
          modelVersion = modelVersion(),
          line = int(position.lineNumber),
          column = int(position.column)
        )
      }
      editor.trigger("tidyparse.syntaxRepair", "editor.action.triggerSuggest", js("({})"))
      Unit
    }
    listOf(monaco.KeyMod.CtrlCmd, monaco.KeyMod.WinCtrl)
      .map { int(it) or int(monaco.KeyCode.Space) }
      .distinct()
      .forEach { keybinding -> editor.addCommand(keybinding, triggerRepairCompletion) }

    val repairAction = js("({})")
    repairAction.id = "tidyparse.triggerSyntaxRepair"
    repairAction.label = "Trigger TidyParse Syntax Repair"
    repairAction.contextMenuGroupId = "navigation"
    repairAction.contextMenuOrder = 1.5
    repairAction.run = { _: dynamic -> triggerRepairCompletion() }
    editor.addAction(repairAction)
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
    completionProvider.provideCompletionItems =
      { requestedModel: dynamic, position: dynamic, _: dynamic, cancellation: dynamic ->
        if (requestedModel != model || !engineReady) null
        else {
          val invocation = consumeRepairInvocation(position)
          if (invocation == null) {
            emptyCompletionResult()
          } else {
            synchronizeEngine()
            val source = requestedModel.getValue() as String
            val originalLine = requestedModel.getLineContent(invocation.line) as String
            scope.promise {
              val repairResult = try {
                repairClient.repairLine(originalLine)
              } catch (failure: Throwable) {
                console.warn("Unable to generate syntax repairs", failure)
                PythonRepairResult(
                  repairMode = true,
                  repairs = emptyList(),
                  displayResultLimit = 0
                )
              }

              if (!repairInvocationIsCurrent(invocation, originalLine, cancellation)) {
                emptyCompletionResult()
              } else if (!repairResult.repairMode) {
                emptyCompletionResult()
              } else {
                val admissible = admissibleRepairs(
                  invocation = invocation,
                  source = source,
                  originalLine = originalLine,
                  candidates = repairResult.repairs,
                  completionLimit = repairResult.displayResultLimit,
                  cancellation = cancellation
                )
                if (admissible.isEmpty()) emptyCompletionResult()
                else repairCompletionResult(requestedModel, position, admissible, originalLine)
              }
            }
          }
        }
      }
    monaco.languages.registerCompletionItemProvider("python", completionProvider)

    val hoverProvider = js("({})")
    hoverProvider.provideHover = { requestedModel: dynamic, position: dynamic, _: dynamic ->
      if (requestedModel != model || !engineReady) null
      else {
        synchronizeEngine()
        hoverResult(engine.hover(int(position.lineNumber), int(position.column)))
      }
    }
    monaco.languages.registerHoverProvider("python", hoverProvider)

    val definitionProvider = js("({})")
    definitionProvider.provideDefinition = { requestedModel: dynamic, position: dynamic, _: dynamic ->
      if (requestedModel != model || !engineReady) null
      else {
        synchronizeEngine()
        definitionResult(requestedModel, engine.definitions(
          int(position.lineNumber), int(position.column)
        ))
      }
    }
    monaco.languages.registerDefinitionProvider("python", definitionProvider)

    val inlayProvider = js("({})")
    inlayProvider.provideInlayHints = { requestedModel: dynamic, range: dynamic, _: dynamic ->
      if (requestedModel != model || !engineReady) null
      else {
        synchronizeEngine()
        inlayHintResult(engine.inlayHints(
          int(range.startLineNumber), int(range.startColumn),
          int(range.endLineNumber), int(range.endColumn)
        ))
      }
    }
    monaco.languages.registerInlayHintsProvider("python", inlayProvider)

    val formatProvider = js("({})")
    formatProvider.provideDocumentFormattingEdits =
      { requestedModel: dynamic, _: dynamic, _: dynamic ->
        if (requestedModel != model || !engineReady) emptyArray<dynamic>()
        else {
          synchronizeEngine()
          val formatted = engine.format()
          if (formatted == null || formatted == requestedModel.getValue()) emptyArray<dynamic>()
          else {
            val edit = js("({})")
            edit.range = requestedModel.getFullModelRange()
            edit.text = formatted
            arrayOf(edit)
          }
        }
      }
    monaco.languages.registerDocumentFormattingEditProvider("python", formatProvider)
  }

  private fun consumeRepairInvocation(position: dynamic): RepairCompletionInvocation? {
    val invocation = armedRepair
    armedRepair = null
    return invocation?.takeIf {
      it.requestId == repairRequestId &&
        it.modelVersion == modelVersion() &&
        it.line == int(position.lineNumber) &&
        it.column == int(position.column)
    }
  }

  private fun admissibleRepairs(
    invocation: RepairCompletionInvocation,
    source: String,
    originalLine: String,
    candidates: List<String>,
    completionLimit: Int,
    cancellation: dynamic
  ): List<String> = semanticallyAdmissibleRepairs(
    candidates = candidates,
    originalLine = originalLine,
    completionLimit = completionLimit,
    isCurrent = { repairInvocationIsCurrent(invocation, originalLine, cancellation) },
    sourceWithLine = { repairedLine -> replaceLine(source, invocation.line, repairedLine) },
    isSemanticallyAdmissible = engine::isSemanticallyAdmissible,
    formatCandidate = engine::formatRepairCandidate
  )

  private fun repairInvocationIsCurrent(
    invocation: RepairCompletionInvocation,
    originalLine: String,
    cancellation: dynamic
  ): Boolean =
    invocation.requestId == repairRequestId &&
      invocation.modelVersion == modelVersion() &&
      (model.getLineContent(invocation.line) as? String) == originalLine &&
      !(cancellation?.isCancellationRequested as? Boolean ?: false)

  private fun replaceLine(source: String, line: Int, replacement: String): String {
    val start = js("({})")
    start.lineNumber = line
    start.column = 1
    val end = js("({})")
    end.lineNumber = line
    end.column = model.getLineMaxColumn(line)
    val startOffset = int(model.getOffsetAt(start))
    val endOffset = int(model.getOffsetAt(end))
    return source.replaceRange(startOffset, endOffset, replacement)
  }

  private fun synchronizeEngine(): dynamic {
    if (!engineReady) return null
    val version = modelVersion()
    if (version == engineSourceVersion) return null
    val snapshot = engine.update(model.getValue() as String)
    engineSourceVersion = version
    return snapshot
  }

  private fun requestAnalysisNow() {
    analysisTimer = 0
    if (!engineReady) return
    val version = modelVersion()
    try {
      val snapshot = synchronizeEngine() ?: engine.update(model.getValue() as String)
      engineSourceVersion = version
      if (modelVersion() == version) publishAnalysis(snapshot, version)
    } catch (failure: Throwable) {
      setRuntimeStatus("error", failure.message ?: "ty analysis failed")
    }
  }

  private fun publishAnalysis(snapshot: dynamic, version: Int) {
    if (!defined(snapshot) || modelVersion() != version) return
    val items = snapshot.diagnostics
    applyDiagnostics(items)
    val count = arrayLength(items)
    analysisSummary.textContent =
      "$count ${plural(count, "diagnostic")} · type-aware incremental analysis"
    setRuntimeStatus("ready", "ty ${engine.version} is ready")
  }

  private fun applyDiagnostics(items: dynamic) {
    val markers = (0 until arrayLength(items)).map { index ->
      val item = items[index]
      val range = monacoRange(item.range)
      val marker = js("({})")
      marker.startLineNumber = range.startLineNumber
      marker.startColumn = range.startColumn
      marker.endLineNumber = range.endLineNumber
      marker.endColumn = range.endColumn
      marker.message = item.message as? String ?: "Python diagnostic"
      marker.severity = markerSeverity(int(item.severity))
      marker.source = "ty"
      marker.code = item.id as? String ?: "ty"
      marker
    }.toTypedArray()
    monaco.editor.setModelMarkers(model, "ty", markers)
    renderDiagnostics(items)
  }

  private fun renderDiagnostics(items: dynamic) {
    val count = arrayLength(items)
    diagnosticCount.textContent = count.toString()
    diagnostics.textContent = ""
    if (count == 0) {
      val empty = document.createElement("p") as HTMLElement
      empty.className = "empty-state"
      empty.textContent = "No ty diagnostics. Completions, hover, navigation, and inlay hints are current."
      diagnostics.appendChild(empty)
      return
    }

    for (index in 0 until count) {
      val item = items[index]
      val range = monacoRange(item.range)
      val button = document.createElement("button") as HTMLButtonElement
      button.className = "diagnostic"
      button.setAttribute("data-severity", severityName(int(item.severity)))
      val title = document.createElement("strong") as HTMLElement
      val id = item.id as? String
      title.textContent = buildString {
        append("Ln ${int(range.startLineNumber)}, Col ${int(range.startColumn)}")
        if (!id.isNullOrBlank()) append(" · $id")
      }
      val message = document.createElement("span") as HTMLElement
      message.textContent = item.message as? String ?: "Python diagnostic"
      button.append(title, message)
      button.onclick = {
        val position = range.getStartPosition()
        editor.setPosition(position)
        editor.revealPositionInCenter(position)
        editor.focus()
        Unit
      }
      diagnostics.appendChild(button)
    }
  }

  private fun renderEngine(metadata: dynamic) {
    engineName.textContent = "ty"
    val revision = (metadata?.version as? String) ?: engine.version
    engineRevision.textContent = revision.take(9)
    engineRevision.title = "Astral ruff revision $revision"
    modeBadge.textContent = "type-aware"
    engineCapabilities.textContent = ""
    listOf("diagnostics", "syntax repair", "hover", "definition", "inlay hints", "formatting")
      .forEach { capability ->
        val item = document.createElement("span") as HTMLElement
        item.className = "capability"
        item.textContent = capability
        engineCapabilities.appendChild(item)
      }
  }

  private fun repairCompletionResult(
    requestedModel: dynamic,
    position: dynamic,
    repairs: List<String>,
    originalLine: String
  ): dynamic {
    val repairRange = js("({})")
    repairRange.startLineNumber = position.lineNumber
    repairRange.endLineNumber = position.lineNumber
    repairRange.startColumn = 1
    repairRange.endColumn = requestedModel.getLineMaxColumn(position.lineNumber)
    val repairSuggestions = repairs.mapIndexed { index, repair ->
      val item = js("({})")
      val rendered = repair.trim().let { if (it.length <= 96) it else it.take(93) + "…" }
      val label = js("({})")
      label.label = rendered
      label.description = "TidyParse syntax repair"
      item.label = label
      item.insertText = repair
      item.insertTextRules = monaco.languages.CompletionItemInsertTextRule.KeepWhitespace
      item.filterText = originalLine
      item.kind = monaco.languages.CompletionItemKind.Text
      item.detail = repairClient.completionDetail()
      item.documentation = markdown(
        "Ranked locally by TidyParse, admitted only when the full repaired file has no ty diagnostics, and formatted with Ruff."
      )
      item.range = repairRange
      item.sortText = "0" + index.toString().padStart(6, '0')
      item.preselect = index == 0
      item
    }

    val result = js("({})")
    result.suggestions = repairSuggestions.toTypedArray()
    result.incomplete = false
    return result
  }

  private fun emptyCompletionResult(): dynamic {
    val result = js("({})")
    result.suggestions = emptyArray<dynamic>()
    result.incomplete = false
    return result
  }

  private fun hoverResult(source: dynamic): dynamic {
    if (!defined(source)) return null
    val text = source.markdown as? String ?: return null
    val result = js("({})")
    result.range = monacoRange(source.range)
    result.contents = arrayOf(markdown(text))
    return result
  }

  private fun definitionResult(requestedModel: dynamic, links: dynamic): dynamic {
    val locations = (0 until arrayLength(links)).map { index ->
      val source = links[index]
      val location = js("({})")
      location.uri = requestedModel.uri
      location.range = monacoRange(source.selectionRange ?: source.fullRange)
      location
    }.toTypedArray()
    return locations.takeIf { it.isNotEmpty() }
  }

  private fun inlayHintResult(items: dynamic): dynamic {
    val hints = (0 until arrayLength(items)).map { index ->
      val source = items[index]
      val hint = js("({})")
      val position = js("({})")
      position.lineNumber = max(1, int(source.position?.line))
      position.column = max(1, int(source.position?.column))
      hint.position = position
      hint.label = source.label as? String ?: ""
      hint.kind = when (int(source.kind, -1)) {
        0 -> monaco.languages.InlayHintKind.Type
        1 -> monaco.languages.InlayHintKind.Parameter
        else -> monaco.languages.InlayHintKind.Type
      }
      val edits = source.textEdits
      if (arrayLength(edits) > 0) {
        hint.textEdits = (0 until arrayLength(edits)).map { editIndex ->
          val sourceEdit = edits[editIndex]
          val edit = js("({})")
          edit.range = monacoRange(sourceEdit.range)
          edit.text = sourceEdit.text as? String ?: ""
          edit
        }.toTypedArray()
      }
      hint
    }.toTypedArray()
    val result = js("({})")
    result.hints = hints
    result.dispose = { Unit }
    return result
  }

  private fun runPython() {
    if (runButton.disabled) return
    setRunning(true)
    setExecutionStatus("running", "Starting Python…")
    executionOutputMeta.textContent = "Waiting for the browser Python runtime…"
    executionDiagnosticsMeta.textContent = "Runtime exceptions will appear here."
    val startedAt = window.performance.now()

    try {
      runner.run(model.getValue() as String, programInput.value).then(
        { result: PythonExecutionResult ->
          renderExecutionResult(result, (window.performance.now() - startedAt).toInt())
          setRunning(false)
          Unit
        },
        { failure: dynamic ->
          renderExecutionFailure(failure?.message as? String ?: "Python execution failed.")
          setRunning(false)
          Unit
        }
      )
    } catch (failure: Throwable) {
      renderExecutionFailure(failure.message ?: "Python execution failed.")
      setRunning(false)
    }
  }

  private fun renderExecutionResult(result: PythonExecutionResult, elapsed: Int) {
    executionOutput.textContent = result.stdout.ifBlank { "Program produced no output." }
    executionOutputMeta.textContent = when {
      result.timedOut -> "Execution timed out after $elapsed ms."
      result.exitCode == 0 -> "Python exited successfully in $elapsed ms."
      else -> "Python exited with code ${result.exitCode} in $elapsed ms."
    }
    executionDiagnostics.textContent = result.stderr.ifBlank { "No runtime exceptions or stderr output." }
    executionDiagnosticsMeta.textContent =
      if (result.stderr.isBlank()) "Python reported no runtime diagnostics."
      else "Python traceback and stderr output."

    if (result.exitCode != 0 || result.timedOut) {
      activateExecutionTab("diagnostics")
      setExecutionStatus("error", if (result.timedOut) "Timed out" else "Exit ${result.exitCode}")
    } else if (result.stderr.isNotBlank()) {
      activateExecutionTab("diagnostics")
      setExecutionStatus("warning", "Exit 0 · stderr")
    } else {
      activateExecutionTab("output")
      setExecutionStatus("success", "Exit 0")
    }
  }

  private fun renderExecutionFailure(message: String) {
    executionDiagnostics.textContent = message
    executionDiagnosticsMeta.textContent = "Unable to run this program."
    activateExecutionTab("diagnostics")
    setExecutionStatus("error", "Run failed")
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
    element("python-app").classList.toggle("is-running", running)
    runButton.querySelector(".run-label")?.textContent = if (running) "Running…" else "Run"
  }

  private fun setExecutionStatus(state: String, message: String) {
    executionStatus.setAttribute("data-state", state)
    executionStatus.textContent = message
  }

  private fun monacoRange(source: dynamic): dynamic {
    val startLine = max(1, int(source?.start?.line, 1))
    val startColumn = max(1, int(source?.start?.column, 1))
    val rawEndLine = max(1, int(source?.end?.line, startLine))
    val rawEndColumn = max(1, int(source?.end?.column, startColumn))
    val endLine = max(startLine, rawEndLine)
    val endColumn = if (endLine == startLine) max(startColumn, rawEndColumn) else rawEndColumn
    return newMonacoRange(monaco.Range, startLine, startColumn, endLine, endColumn)
  }

  private fun markerSeverity(severity: Int): dynamic = when (severity) {
    0 -> monaco.MarkerSeverity.Info
    1 -> monaco.MarkerSeverity.Warning
    2, 3 -> monaco.MarkerSeverity.Error
    else -> monaco.MarkerSeverity.Warning
  }

  private fun severityName(severity: Int): String = when (severity) {
    0 -> "info"
    1 -> "warning"
    else -> "error"
  }

  private fun markdown(value: String): dynamic {
    val markdown = js("({})")
    markdown.value = value
    markdown.isTrusted = false
    markdown.supportHtml = false
    return markdown
  }

  private fun applyInitialTheme() {
    val saved = try { window.localStorage.getItem(PYTHON_THEME_KEY) } catch (_: Throwable) { null }
    darkTheme = when (saved) {
      "dark" -> true
      "light" -> false
      else -> window.matchMedia("(prefers-color-scheme: dark)").matches
    }
    document.documentElement?.setAttribute("data-theme", if (darkTheme) "dark" else "light")
    updateThemeButton()
  }

  private fun setTheme(dark: Boolean) {
    darkTheme = dark
    val name = if (dark) "dark" else "light"
    document.documentElement?.setAttribute("data-theme", name)
    monaco.editor.setTheme(if (dark) "vs-dark" else "vs")
    updateThemeButton()
    try { window.localStorage.setItem(PYTHON_THEME_KEY, name) } catch (_: Throwable) {}
  }

  private fun updateThemeButton() {
    val button = element("toggle-theme") as HTMLButtonElement
    val nextTheme = if (darkTheme) "light" else "dark"
    button.setAttribute("aria-pressed", darkTheme.toString())
    button.setAttribute("aria-label", "Use $nextTheme theme")
    button.title = "Use $nextTheme theme"
  }

  private fun initialSource(): String = try {
    window.localStorage.getItem(PYTHON_SOURCE_KEY)?.takeIf { it.isNotBlank() } ?: DEFAULT_PYTHON_SOURCE
  } catch (_: Throwable) {
    DEFAULT_PYTHON_SOURCE
  }

  private fun saveSource(source: String) {
    try { window.localStorage.setItem(PYTHON_SOURCE_KEY, source) } catch (_: Throwable) {}
  }

  private fun setRuntimeStatus(state: String, message: String) {
    status.setAttribute("data-state", state)
    status.title = message
    status.querySelector(".status-text")?.textContent = message
  }

  private fun setRepairStatus(state: String, message: String) {
    repairStatus.setAttribute("data-state", state)
    repairStatus.title = message
    repairStatus.querySelector(".status-text")?.textContent = message
  }

  private fun modelVersion(): Int = int(model.getVersionId())

  private fun element(id: String): HTMLElement =
    document.getElementById(id) as? HTMLElement ?: error("Missing #$id")

  private fun arrayLength(value: dynamic): Int =
    if (!defined(value)) 0 else (value.length as? Number)?.toInt() ?: 0

  private fun int(value: dynamic, fallback: Int = 0): Int = (value as? Number)?.toInt() ?: fallback

  private fun defined(value: dynamic): Boolean =
    js("(value) => value !== undefined && value !== null")(value) as Boolean

  private fun plural(count: Int, noun: String): String = if (count == 1) noun else "${noun}s"
}

private fun newMonacoRange(
  constructor: dynamic,
  startLine: Int,
  startColumn: Int,
  endLine: Int,
  endColumn: Int
): dynamic = js("(Range, sl, sc, el, ec) => new Range(sl, sc, el, ec)")(
  constructor,
  startLine,
  startColumn,
  endLine,
  endColumn
)
