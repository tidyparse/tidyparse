import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import org.w3c.dom.HTMLButtonElement
import org.w3c.dom.HTMLElement
import org.w3c.dom.HTMLPreElement
import org.w3c.dom.HTMLSelectElement
import org.w3c.dom.HTMLTextAreaElement
import org.w3c.dom.events.KeyboardEvent
import kotlin.js.Promise

private const val COMPILER_EXPLORER_API = "https://godbolt.org/api/compiler"
private const val COMPILER_REQUEST_TIMEOUT_MS = 20_000
private const val CPP_THEME_KEY = "tidyparse-cpp-theme"
private const val CPP_SOURCE_KEY_PREFIX = "tidyparse-cpp-source-"
private const val CPP_STYLE_ID = "tidyparse-cpp-style"

// language=c++
private const val DEFAULT_CPP = """#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Animal {
    std::string name_;
public:
    explicit Animal(std::string name) : name_(std::move(name)) {}
    virtual ~Animal() = default;
    const std::string& name() const { return name_; }
    virtual std::string speak() const = 0;
};

class Dog final : public Animal {
    int age_;
public:
    Dog(std::string name, int age) : Animal(std::move(name)), age_(age) {}
    std::string speak() const override { return age_ < 2 ? "yip" : "woof"; }
};

class Cat final : public Animal {
public:
    using Animal::Animal;
    std::string speak() const override { return "meow"; }
};

void introduce(const Animal& animal, int times = 1) {
    std::cout << animal.name() << ": ";
    for (int i = 0; i < times; ++i)
        std::cout << animal.speak() << (i + 1 == times ? '\n' : ' ');
}

int main() {
    Dog dog{"Rex", 4};
    Cat cat{"Luna"};

    introduce(dog);
    introduce(cat, 3);

    std::vector<std::unique_ptr<Animal>> animals;
    animals.push_back(std::make_unique<Dog>("Pip", 1));
    animals.push_back(std::make_unique<Cat>("Milo"));

    for (const auto& animal : animals)
        introduce(*animal, 2);
}
"""

private const val DEFAULT_C = """#include <stdio.h>

int main(void) {
    puts("Hello, world!");
    return 0;
}
"""

private enum class NativeLanguage(
  val queryValue: String,
  val fileName: String,
  val compilerId: String,
  val apiLanguage: String,
  val arguments: List<String>,
  val sample: String
) {
  CPP(
    queryValue = "cpp",
    fileName = "main.cpp",
    compilerId = "clang2110",
    apiLanguage = "c++",
    arguments = listOf("-std=c++23", "-Wall", "-Wextra", "-pedantic-errors"),
    sample = DEFAULT_CPP
  ),
  C(
    queryValue = "c",
    fileName = "main.c",
    compilerId = "cclang2110",
    apiLanguage = "c",
    arguments = listOf("-std=c23", "-Wall", "-Wextra", "-pedantic-errors"),
    sample = DEFAULT_C
  );

  companion object {
    fun from(value: String?): NativeLanguage =
      if (value.equals("c", ignoreCase = true)) C else CPP
  }
}

private data class ExecutionResult(
  val didExecute: Boolean,
  val exitCode: Int,
  val timedOut: Boolean,
  val truncated: Boolean,
  val stdout: String,
  val stderr: String,
  val diagnostics: String
)

fun cppSetup() { CppPlayground().start() }

private class CppPlayground {
  private val scope = MainScope()
  private val params: dynamic = js("new URLSearchParams(window.location.search)")

  private lateinit var app: HTMLElement
  private lateinit var editorRoot: HTMLElement
  private lateinit var monacoHost: HTMLElement
  private lateinit var source: HTMLTextAreaElement
  private lateinit var lineNumbers: HTMLPreElement
  private lateinit var languageSelect: HTMLSelectElement
  private lateinit var fileName: HTMLElement
  private lateinit var position: HTMLElement
  private lateinit var status: HTMLElement
  private lateinit var lspStatus: HTMLElement
  private lateinit var runButton: HTMLButtonElement
  private lateinit var resetButton: HTMLButtonElement
  private lateinit var buildPanel: HTMLElement
  private lateinit var stdin: HTMLTextAreaElement
  private lateinit var output: HTMLPreElement
  private lateinit var diagnostics: HTMLPreElement
  private lateinit var outputMeta: HTMLElement
  private lateinit var diagnosticsMeta: HTMLElement
  private lateinit var problems: HTMLPreElement
  private lateinit var problemsMeta: HTMLElement
  private lateinit var problemCount: HTMLElement
  private lateinit var completionPopup: HTMLElement

  private var language = NativeLanguage.CPP
  private var monacoEditor: JSMonacoEditor? = null
  private var clangdClient: JSClangdClient? = null
  private var clangdDiagnostics: List<ClangdDiagnostic> = emptyList()
  private var clangdDiagnosticsPublished = false
  private var clangdFailure: String? = null
  private var completionRequest = 0
  private var completionItems: List<ClangdCompletion> = emptyList()
  private var selectedCompletion = 0

  fun start() {
    language = NativeLanguage.from(parameter("language") ?: parameter("lang"))
    installStyles()
    installMarkup()
    bindElements()
    applyInitialTheme()
    initializeEditor()
    bindEvents()
    updateEditorChrome()
    setStatus("ready", "Ready")
    scope.launch { bootstrapClangd() }

    when {
      isTrueParameter("run") -> scope.launch { compileAndRun() }
      parameter("run").equals("showonly", ignoreCase = true) -> showBuildPanel()
    }
  }

  private fun installStyles() {
    if (document.getElementById(CPP_STYLE_ID) != null) return
    val style = document.createElement("style") as HTMLElement
    style.id = CPP_STYLE_ID
    style.textContent = CPP_CSS
    document.head?.appendChild(style)
  }

  private fun installMarkup() {
    app = (document.getElementById("cpp-root") ?: document.body) as HTMLElement
    app.innerHTML = """
      <div id="cpp-app">
        <header id="cpp-toolbar">
          <div class="cpp-brand" aria-label="Tidyparse C and C++ Playground">
            <span class="cpp-brand-mark" aria-hidden="true">&lt;/&gt;</span>
            <span class="cpp-brand-title">C/C++ Playground</span>
            <span id="cpp-clangd-badge" class="cpp-badge">clangd</span>
          </div>
          <div class="cpp-toolbar-actions">
            <label class="cpp-language-control">
              <span class="sr-only">Language</span>
              <select id="cpp-language" aria-label="Language">
                <option value="cpp">C++23</option>
                <option value="c">C23</option>
              </select>
            </label>
            <button id="cpp-reset" class="cpp-button cpp-button-quiet" type="button" title="Reset the example">Reset</button>
            <button id="cpp-theme" class="cpp-icon-button" type="button" title="Toggle theme" aria-label="Toggle theme">
              <span aria-hidden="true">◐</span>
            </button>
            <button id="cpp-run" class="cpp-button cpp-button-run" type="button">
              <span class="cpp-run-icon" aria-hidden="true">▶</span>
              <span class="cpp-run-label">Run</span>
            </button>
          </div>
        </header>

        <main id="cpp-workspace">
          <section id="cpp-editor-pane" aria-label="Source editor">
            <div class="cpp-pane-header">
              <span id="cpp-file-name">main.cpp</span>
              <span class="cpp-pane-hint">Ctrl/⌘-click or F12 for definition · Shift+F12 for references · Ctrl/⌘+Enter to run</span>
            </div>
            <div id="cpp-editor">
              <pre id="cpp-line-numbers" aria-hidden="true">1</pre>
              <textarea id="cpp-source" aria-label="C or C++ source code" autocomplete="off"
                autocapitalize="off" spellcheck="false" wrap="off"></textarea>
              <div id="cpp-monaco" aria-label="C or C++ source code"></div>
              <div id="cpp-completions" class="is-hidden" role="listbox" aria-label="Code completions"></div>
            </div>
          </section>

          <section id="cpp-build-panel" class="is-hidden" aria-label="Build panel">
            <div id="cpp-build-resize" role="separator" tabindex="0" aria-label="Resize build panel"
              aria-orientation="horizontal" aria-valuemin="72" aria-valuenow="300" title="Drag or use arrow keys to resize"></div>
            <div class="cpp-build-header">
              <div id="cpp-build-tabs" role="tablist" aria-label="Build results">
                <button id="cpp-tab-button-input" type="button" role="tab" data-tab="input"
                  aria-controls="cpp-tab-input" aria-selected="true" tabindex="0">Input</button>
                <button id="cpp-tab-button-output" type="button" role="tab" data-tab="output"
                  aria-controls="cpp-tab-output" aria-selected="false" tabindex="-1">Output</button>
                <button id="cpp-tab-button-diagnostics" type="button" role="tab" data-tab="diagnostics"
                  aria-controls="cpp-tab-diagnostics" aria-selected="false" tabindex="-1">Diagnostics</button>
                <button id="cpp-tab-button-problems" type="button" role="tab" data-tab="problems"
                  aria-controls="cpp-tab-problems" aria-selected="false" tabindex="-1">
                  Problems <span id="cpp-problem-count" class="cpp-count">0</span>
                </button>
              </div>
              <button id="cpp-close-build" class="cpp-icon-button" type="button" title="Close build panel" aria-label="Close build panel">×</button>
            </div>
            <div class="cpp-build-content">
              <section id="cpp-tab-input" class="cpp-tab-panel" role="tabpanel" aria-labelledby="cpp-tab-button-input">
                <label for="cpp-stdin">Standard input</label>
                <textarea id="cpp-stdin" placeholder="Optional input for stdin…" spellcheck="false"></textarea>
              </section>
              <section id="cpp-tab-output" class="cpp-tab-panel is-hidden" role="tabpanel" aria-labelledby="cpp-tab-button-output">
                <div id="cpp-output-meta" class="cpp-result-meta">Run the program to see its output.</div>
                <pre id="cpp-output">No output yet.</pre>
              </section>
              <section id="cpp-tab-diagnostics" class="cpp-tab-panel is-hidden" role="tabpanel" aria-labelledby="cpp-tab-button-diagnostics">
                <div id="cpp-diagnostics-meta" class="cpp-result-meta">Compiler diagnostics will appear here.</div>
                <pre id="cpp-diagnostics">No diagnostics yet.</pre>
              </section>
              <section id="cpp-tab-problems" class="cpp-tab-panel is-hidden" role="tabpanel" aria-labelledby="cpp-tab-button-problems">
                <div id="cpp-problems-meta" class="cpp-result-meta">clangd diagnostics will appear here as you type.</div>
                <pre id="cpp-problems">clangd is starting…</pre>
              </section>
            </div>
            <div class="cpp-build-credit">
              Compilation and execution provided by
              <a href="https://godbolt.org/" target="_blank" rel="noreferrer">Compiler Explorer</a>.
            </div>
          </section>
        </main>

        <footer id="cpp-statusbar">
          <button id="cpp-show-build" type="button" class="cpp-status-button" title="Show build panel">
            <span aria-hidden="true">▰</span>
            Build
          </button>
          <div class="cpp-status-spacer"></div>
          <span id="cpp-position">Ln 1, Col 1</span>
          <span id="cpp-lsp-status" class="cpp-runtime-status" data-state="working" role="status"
            aria-live="polite" title="clangd is starting">
            <span class="cpp-status-dot" aria-hidden="true"></span>
            <span class="cpp-status-text">clangd…</span>
          </span>
          <span id="cpp-status" class="cpp-runtime-status" data-state="ready" role="status" aria-live="polite">
            <span class="cpp-status-dot" aria-hidden="true"></span>
            <span class="cpp-status-text">Ready</span>
          </span>
        </footer>
      </div>
    """.trimIndent()
  }

  private fun bindElements() {
    editorRoot = element("cpp-editor")
    monacoHost = element("cpp-monaco")
    source = element("cpp-source")
    lineNumbers = element("cpp-line-numbers")
    languageSelect = element("cpp-language")
    fileName = element("cpp-file-name")
    position = element("cpp-position")
    status = element("cpp-status")
    lspStatus = element("cpp-lsp-status")
    runButton = element("cpp-run")
    resetButton = element("cpp-reset")
    buildPanel = element("cpp-build-panel")
    stdin = element("cpp-stdin")
    output = element("cpp-output")
    diagnostics = element("cpp-diagnostics")
    outputMeta = element("cpp-output-meta")
    diagnosticsMeta = element("cpp-diagnostics-meta")
    problems = element("cpp-problems")
    problemsMeta = element("cpp-problems-meta")
    problemCount = element("cpp-problem-count")
    completionPopup = element("cpp-completions")
  }

  private fun initializeEditor() {
    languageSelect.value = language.queryValue
    val requestedCode = parameter("code")
    val savedCode = readLocalStorage(CPP_SOURCE_KEY_PREFIX + language.queryValue)
    source.value = requestedCode ?: savedCode ?: language.sample
    stdin.value = parameter("stdin") ?: ""
    editorRoot.classList.add("has-monaco")
    try {
      monacoEditor = JSMonacoEditor(
        container = monacoHost,
        fileName = language.fileName,
        text = source.value,
        darkTheme = document.body?.classList?.contains("cpp-dark") == true,
        onChange = { text -> handleMonacoChange(text) },
        onPosition = { line, column -> position.textContent = "Ln $line, Col $column" },
        onOpenedFile = { openedFile -> fileName.textContent = openedFile },
        onRun = { scope.launch { compileAndRun() } }
      )
      source.tabIndex = -1
      source.setAttribute("aria-hidden", "true")
      monacoEditor?.focus()
    } catch (failure: Throwable) {
      editorRoot.classList.remove("has-monaco")
      monacoHost.textContent = ""
      source.focus()
      source.setSelectionRange(0, 0)
      console.warn("Monaco could not start; using the textarea fallback.", failure)
    }
  }

  private fun handleMonacoChange(text: String) {
    source.value = text
    writeLocalStorage(CPP_SOURCE_KEY_PREFIX + language.queryValue, text)
    clangdDiagnosticsPublished = false
    clangdClient?.didChange(text)
  }

  private fun bindEvents() {
    source.addEventListener("input", {
      writeLocalStorage(CPP_SOURCE_KEY_PREFIX + language.queryValue, source.value)
      updateEditorChrome()
      clangdDiagnosticsPublished = false
      clangdClient?.didChange(source.value)
      val caret = source.selectionStart ?: 0
      val trigger = source.value.getOrNull(caret - 1)
      if (trigger == '.' || trigger == '>' || trigger == ':') {
        val request = ++completionRequest
        window.setTimeout({
          if (request == completionRequest) requestCompletions()
        }, 75)
      } else {
        hideCompletions()
      }
    })
    source.addEventListener("scroll", {
      lineNumbers.scrollTop = source.scrollTop
      hideCompletions()
    })
    source.addEventListener("click", { updateCaretPosition() })
    source.addEventListener("keyup", { updateCaretPosition() })
    source.addEventListener("select", { updateCaretPosition() })
    source.addEventListener("blur", {
      window.setTimeout({ hideCompletions() }, 120)
    })
    source.addEventListener("keydown", { event ->
      handleEditorKey(event as KeyboardEvent)
    })

    languageSelect.addEventListener("change", {
      switchLanguage(NativeLanguage.from(languageSelect.value))
    })

    runButton.addEventListener("click", { scope.launch { compileAndRun() } })
    resetButton.addEventListener("click", { resetSource() })
    element<HTMLButtonElement>("cpp-theme").addEventListener("click", { toggleTheme() })
    element<HTMLButtonElement>("cpp-show-build").addEventListener("click", { showBuildPanel() })
    element<HTMLButtonElement>("cpp-close-build").addEventListener("click", { hideBuildPanel() })

    val tabs = document.querySelectorAll("#cpp-build-tabs [data-tab]")
    for (index in 0 until tabs.length) {
      val tab = tabs.item(index) as HTMLButtonElement
      tab.addEventListener("click", { activateTab(tab.getAttribute("data-tab") ?: "input") })
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

    bindBuildPanelResize(element("cpp-build-resize"))

    window.addEventListener("pagehide", { event ->
      if (event.asDynamic().persisted as? Boolean == true) return@addEventListener
      monacoEditor?.dispose()
      clangdClient?.dispose()
    })
  }

  private fun handleEditorKey(event: KeyboardEvent) {
    if (handleCompletionKey(event)) return

    if (event.code == "Space" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      requestCompletions()
      return
    }

    if (event.key == "Escape") {
      runButton.focus()
      return
    }

    if (event.key == "Tab") {
      event.preventDefault()
      insertIndent(event.shiftKey)
      return
    }

    if (event.key == "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      scope.launch { compileAndRun() }
    }
  }

  private fun insertIndent(outdent: Boolean) {
    val start = source.selectionStart ?: 0
    val end = source.selectionEnd ?: start
    val value = source.value

    if (outdent) {
      val lineStart = value.lastIndexOf('\n', (start - 1).coerceAtLeast(0)).let { if (it < 0) 0 else it + 1 }
      val removable = when {
        value.startsWith("  ", lineStart) -> 2
        value.startsWith("\t", lineStart) -> 1
        else -> 0
      }
      if (removable > 0) {
        source.value = value.removeRange(lineStart, lineStart + removable)
        source.setSelectionRange((start - removable).coerceAtLeast(lineStart), (end - removable).coerceAtLeast(lineStart))
      }
    } else {
      source.value = value.substring(0, start) + "  " + value.substring(end)
      source.setSelectionRange(start + 2, start + 2)
    }

    source.dispatchEvent(js("new Event('input', { bubbles: true })"))
  }

  private fun switchLanguage(next: NativeLanguage) {
    if (next == language) return
    writeLocalStorage(CPP_SOURCE_KEY_PREFIX + language.queryValue, editorValue())
    language = next

    source.value = readLocalStorage(CPP_SOURCE_KEY_PREFIX + language.queryValue) ?: language.sample
    source.setSelectionRange(0, 0)
    monacoEditor?.setDocument(language.fileName, source.value)
    hideCompletions()
    clangdDiagnostics = emptyList()
    clangdDiagnosticsPublished = false
    renderClangdDiagnostics(analyzing = clangdClient != null)
    clangdClient?.changeDocument(language.fileName, language.queryValue, source.value)
    output.textContent = "No output yet."
    diagnostics.textContent = "No diagnostics yet."
    outputMeta.textContent = "Run the program to see its output."
    diagnosticsMeta.textContent = "Compiler diagnostics will appear here."
    setStatus("ready", "Ready")
    updateEditorChrome()
    monacoEditor?.focus() ?: source.focus()
  }

  private fun resetSource() {
    if (editorValue() != language.sample && !window.confirm("Reset ${language.fileName} to the example program?")) return
    val monaco = monacoEditor
    if (monaco != null) {
      monaco.setValue(language.sample)
      monaco.focus()
    } else {
      source.value = language.sample
      writeLocalStorage(CPP_SOURCE_KEY_PREFIX + language.queryValue, source.value)
      source.setSelectionRange(0, 0)
      source.focus()
      clangdClient?.didChange(source.value)
    }
    updateEditorChrome()
    hideCompletions()
    clangdDiagnosticsPublished = false
  }

  private suspend fun compileAndRun() {
    if (runButton.disabled) return
    showBuildPanel()
    setRunning(true)
    setStatus("working", "Compiling…")
    val startedAt = window.performance.now()

    try {
      val runLanguage = language
      val result = runCode(editorValue(), stdin.value, runLanguage)
      val elapsed = (window.performance.now() - startedAt).toInt()

      output.textContent = formatProgramOutput(result)
      outputMeta.textContent = when {
        !result.didExecute -> "The program was not executed."
        result.timedOut -> "Execution timed out after ${elapsed} ms."
        result.truncated -> "Process exited with code ${result.exitCode}; output was truncated."
        result.exitCode == 0 -> "Process exited successfully in ${elapsed} ms."
        else -> "Process exited with code ${result.exitCode} in ${elapsed} ms."
      }

      diagnostics.textContent = when {
        result.diagnostics.isNotBlank() -> result.diagnostics
        result.didExecute -> "Compilation succeeded with no diagnostics."
        else -> "Compilation failed without diagnostics."
      }
      diagnosticsMeta.textContent = when {
        result.diagnostics.isNotBlank() -> "Clang compiler diagnostics."
        result.didExecute -> "Clang reported no diagnostics."
        else -> "Clang did not return compiler diagnostics."
      }

      if (result.didExecute) {
        activateTab("output")
        when {
          result.timedOut -> setStatus("warning", "Timed out")
          result.truncated -> setStatus("warning", "Truncated")
          else -> setStatus(if (result.exitCode == 0) "ready" else "warning", "Exit ${result.exitCode}")
        }
      } else {
        activateTab("diagnostics")
        setStatus("error", "Build failed")
      }
    } catch (failure: Throwable) {
      diagnostics.textContent = failure.message ?: "The compiler request failed."
      diagnosticsMeta.textContent = "Unable to compile this program."
      activateTab("diagnostics")
      setStatus("error", "Request failed")
    } finally {
      setRunning(false)
    }
  }

  private suspend fun runCode(code: String, input: String, runLanguage: NativeLanguage): ExecutionResult {
    val request = js("{}")
    request.source = code
    request.compiler = runLanguage.compilerId
    request.lang = runLanguage.apiLanguage
    request.options = js("{}")
    request.options.userArguments = runLanguage.arguments.joinToString(" ")
    request.options.executeParameters = js("{}")
    request.options.executeParameters.args = emptyArray<String>()
    request.options.executeParameters.stdin = input
    request.options.compilerOptions = js("({ executorRequest: true })")
    request.options.filters = js("({ execute: true })")

    val init = js("{}")
    init.method = "POST"
    init.headers = js("({ 'Content-Type': 'application/json', 'Accept': 'application/json' })")
    init.body = JSON.stringify(request)
    val controller = js("new AbortController()")
    init.signal = controller.signal
    val timeout = window.setTimeout({ controller.abort() }, COMPILER_REQUEST_TIMEOUT_MS)
    val response = try {
      (window.asDynamic().fetch(
        "$COMPILER_EXPLORER_API/${runLanguage.compilerId}/compile",
        init
      ) as Promise<dynamic>).await()
    } catch (failure: Throwable) {
      if (controller.signal.aborted as Boolean) {
        throw IllegalStateException("Compiler request timed out after ${COMPILER_REQUEST_TIMEOUT_MS / 1_000} seconds.")
      }
      throw failure
    } finally {
      window.clearTimeout(timeout)
    }

    if (!(response.ok as Boolean)) {
      throw IllegalStateException("Compiler Explorer returned HTTP ${response.status}.")
    }

    val payload = (response.json() as Promise<dynamic>).await()
    val buildResult = payload.buildResult
    val didExecute = payload.didExecute as? Boolean ?: false
    val buildDiagnostics = stdioToString(buildResult?.stderr)
    val topLevelStderr = stdioToString(payload.stderr)
    val requestError = payload.error as? String
    if (!requestError.isNullOrBlank() && buildResult == null) {
      throw IllegalStateException(requestError)
    }
    val diagnosticText = listOfNotNull(
      requestError?.takeIf { it.isNotBlank() },
      buildDiagnostics.takeIf { it.isNotBlank() },
      topLevelStderr.takeIf { !didExecute && buildDiagnostics.isBlank() && it.isNotBlank() }
    ).distinct().joinToString("\n")

    return ExecutionResult(
      didExecute = didExecute,
      exitCode = payload.code as? Int ?: buildResult?.code as? Int ?: -1,
      timedOut = payload.timedOut as? Boolean ?: false,
      truncated = payload.truncated as? Boolean ?: false,
      stdout = stdioToString(payload.stdout),
      stderr = stdioToString(payload.stderr),
      diagnostics = stripAnsi(diagnosticText)
    )
  }

  private fun formatProgramOutput(result: ExecutionResult): String {
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

  private fun updateEditorChrome() {
    if (monacoEditor != null) return
    val count = source.value.count { it == '\n' } + 1
    lineNumbers.innerHTML = ""
    val byLine = clangdDiagnostics.groupBy { it.range.start.line }
    repeat(count) { index ->
      val marker = document.createElement("span") as HTMLElement
      marker.className = "cpp-line-number"
      marker.textContent = (index + 1).toString()
      val lineDiagnostics = byLine[index].orEmpty()
      val severity = lineDiagnostics.mapNotNull { it.severity }.minOrNull()
      if (severity != null) marker.setAttribute("data-severity", severity.toString())
      if (lineDiagnostics.isNotEmpty()) {
        marker.title = lineDiagnostics.joinToString("\n") { it.message }
      }
      lineNumbers.appendChild(marker)
    }
    fileName.textContent = language.fileName
    updateCaretPosition()
  }

  private fun updateCaretPosition() {
    if (monacoEditor != null) return
    val caret = source.selectionStart ?: 0
    val prefix = source.value.substring(0, caret.coerceIn(0, source.value.length))
    val line = prefix.count { it == '\n' } + 1
    val column = prefix.length - prefix.lastIndexOf('\n')
    position.textContent = "Ln $line, Col $column"
  }

  private fun requestCompletions() {
    val client = clangdClient ?: return
    val caret = source.selectionStart ?: return
    val prefix = source.value.substring(0, caret.coerceIn(0, source.value.length))
    val line = prefix.count { it == '\n' }
    val lineStart = prefix.lastIndexOf('\n') + 1
    val column = prefix.length - lineStart
    val request = ++completionRequest

    client.requestCompletion(line, column) { items ->
      if (request != completionRequest || source.selectionStart != caret) return@requestCompletion
      completionItems = items
        .sortedWith(compareBy<ClangdCompletion>({ it.sortText ?: it.label }, { it.label }))
        .take(80)
      selectedCompletion = 0
      if (completionItems.isEmpty()) hideCompletions() else showCompletions(line, column)
    }
  }

  private fun showCompletions(line: Int, column: Int) {
    completionPopup.innerHTML = ""
    completionItems.forEachIndexed { index, item ->
      val button = document.createElement("button") as HTMLButtonElement
      button.type = "button"
      button.className = "cpp-completion-item"
      button.setAttribute("role", "option")
      button.setAttribute("aria-selected", (index == selectedCompletion).toString())
      button.setAttribute("data-index", index.toString())

      val label = document.createElement("span") as HTMLElement
      label.className = "cpp-completion-label"
      label.textContent = item.label.trim()
      button.appendChild(label)

      val detailText = item.detail
      if (!detailText.isNullOrBlank()) {
        val detail = document.createElement("span") as HTMLElement
        detail.className = "cpp-completion-detail"
        detail.textContent = detailText
        button.appendChild(detail)
      }

      button.addEventListener("mouseenter", {
        selectedCompletion = index
        updateCompletionSelection()
      })
      button.addEventListener("mousedown", { rawEvent ->
        rawEvent.preventDefault()
        selectedCompletion = index
        applySelectedCompletion()
      })
      completionPopup.appendChild(button)
    }

    val style = window.getComputedStyle(source)
    val fontSize = style.fontSize.removeSuffix("px").toDoubleOrNull() ?: 15.0
    val lineHeight = style.lineHeight.removeSuffix("px").toDoubleOrNull() ?: fontSize * 1.65
    val gutterWidth = lineNumbers.offsetWidth.toDouble()
    val visibleLine = line - (source.scrollTop / lineHeight).toInt()
    val left = gutterWidth + 16 + column * fontSize * 0.6 - source.scrollLeft
    val top = 18 + (visibleLine + 1) * lineHeight
    completionPopup.style.left = "${left.coerceIn(gutterWidth + 4, (app.clientWidth - 120).coerceAtLeast(gutterWidth.toInt() + 4).toDouble()).toInt()}px"
    completionPopup.style.top = "${top.coerceIn(4.0, (source.clientHeight - 48).coerceAtLeast(4).toDouble()).toInt()}px"
    completionPopup.classList.remove("is-hidden")
  }

  private fun handleCompletionKey(event: KeyboardEvent): Boolean {
    if (completionPopup.classList.contains("is-hidden")) return false
    when (event.key) {
      "ArrowDown" -> selectedCompletion = (selectedCompletion + 1) % completionItems.size
      "ArrowUp" -> selectedCompletion = (selectedCompletion - 1 + completionItems.size) % completionItems.size
      "Enter", "Tab" -> {
        event.preventDefault()
        applySelectedCompletion()
        return true
      }
      "Escape" -> {
        event.preventDefault()
        hideCompletions()
        return true
      }
      else -> return false
    }
    event.preventDefault()
    updateCompletionSelection()
    return true
  }

  private fun updateCompletionSelection() {
    val children = completionPopup.children
    for (index in 0 until children.length) {
      val item = children.item(index) as HTMLElement
      val selected = index == selectedCompletion
      item.setAttribute("aria-selected", selected.toString())
      if (selected) item.asDynamic().scrollIntoView(js("({ block: 'nearest' })"))
    }
  }

  private fun applySelectedCompletion() {
    val completion = completionItems.getOrNull(selectedCompletion) ?: return
    val edit = completion.textEdit
    val value = source.value
    val caret = source.selectionStart ?: value.length
    val completionText =
      edit?.newText
        ?: completion.insertText
        ?: completion.label
    val replacement =
      if (completion.insertTextFormat == 2) sanitizeSnippet(completionText)
      else completionText

    val editRange = edit?.range
    val start = if (editRange != null) {
      offsetForPosition(value, editRange.start.line, editRange.start.character)
    } else {
      var prefixStart = caret
      while (prefixStart > 0 && value[prefixStart - 1].let { it == '_' || it.isLetterOrDigit() }) prefixStart--
      prefixStart
    }
    val end = if (editRange != null) {
      offsetForPosition(value, editRange.end.line, editRange.end.character)
    } else {
      caret
    }

    source.value = value.substring(0, start) + replacement + value.substring(end)
    val nextCaret = start + replacement.length
    source.setSelectionRange(nextCaret, nextCaret)
    hideCompletions()
    source.dispatchEvent(js("new Event('input', { bubbles: true })"))
    source.focus()
  }

  private fun hideCompletions() {
    completionRequest++
    completionItems = emptyList()
    completionPopup.classList.add("is-hidden")
    completionPopup.innerHTML = ""
  }

  private fun offsetForPosition(text: String, line: Int, character: Int): Int {
    var offset = 0
    repeat(line.coerceAtLeast(0)) {
      val newline = text.indexOf('\n', offset)
      if (newline < 0) return text.length
      offset = newline + 1
    }
    val end = text.indexOf('\n', offset).let { if (it < 0) text.length else it }
    return (offset + character.coerceAtLeast(0)).coerceAtMost(end)
  }

  private fun sanitizeSnippet(text: String): String =
    text
      .replace(Regex("""\$\{\d+:([^}]*)\}"""), "$1")
      .replace(Regex("""\$\{\d+\}"""), "")
      .replace(Regex("""\$\d+"""), "")

  private fun activateTab(name: String) {
    val tabs = document.querySelectorAll("#cpp-build-tabs [data-tab]")
    for (index in 0 until tabs.length) {
      val tab = tabs.item(index) as HTMLElement
      val active = tab.getAttribute("data-tab") == name
      tab.setAttribute("aria-selected", active.toString())
      tab.tabIndex = if (active) 0 else -1
    }

    listOf("input", "output", "diagnostics", "problems").forEach { panelName ->
      element<HTMLElement>("cpp-tab-$panelName").classList.toggle("is-hidden", panelName != name)
    }
  }

  private fun showBuildPanel() {
    buildPanel.classList.remove("is-hidden")
  }

  private fun hideBuildPanel() {
    buildPanel.classList.add("is-hidden")
  }

  private fun bindBuildPanelResize(handle: HTMLElement) {
    fun setHeight(requestedHeight: Double) {
      val maximum = (window.innerHeight - 124).coerceAtLeast(72)
      val nextHeight = requestedHeight.coerceIn(72.0, maximum.toDouble())
      app.style.setProperty("--cpp-build-height", "${nextHeight.toInt()}px")
      handle.setAttribute("aria-valuenow", nextHeight.toInt().toString())
      handle.setAttribute("aria-valuemax", maximum.toString())
    }

    handle.addEventListener("pointerdown", { rawEvent ->
      val event = rawEvent.asDynamic()
      event.preventDefault()
      val startY = event.clientY as Double
      val startHeight = buildPanel.clientHeight.toDouble()
      handle.asDynamic().setPointerCapture(event.pointerId)

      handle.asDynamic().onpointermove = { moveEvent: dynamic ->
        setHeight(startHeight + startY - (moveEvent.clientY as Double))
      }
      handle.asDynamic().onpointerup = { _: dynamic ->
        handle.asDynamic().onpointermove = null
        handle.asDynamic().onpointerup = null
      }
    })
    handle.addEventListener("keydown", { rawEvent ->
      val event = rawEvent as KeyboardEvent
      val delta = when (event.key) {
        "ArrowUp" -> 32.0
        "ArrowDown" -> -32.0
        "Home" -> 72.0 - buildPanel.clientHeight
        "End" -> window.innerHeight.toDouble()
        else -> 0.0
      }
      if (delta != 0.0) {
        event.preventDefault()
        setHeight(buildPanel.clientHeight + delta)
      }
    })
  }

  private fun setRunning(running: Boolean) {
    runButton.disabled = running
    resetButton.disabled = running
    languageSelect.disabled = running
    source.readOnly = running
    monacoEditor?.setReadOnly(running)
    stdin.readOnly = running
    element<HTMLElement>("cpp-app").classList.toggle("is-running", running)
    element<HTMLElement>("cpp-run").querySelector(".cpp-run-label")?.textContent =
      if (running) "Running…" else "Run"
  }

  private fun setStatus(state: String, text: String) {
    status.setAttribute("data-state", state)
    status.querySelector(".cpp-status-text")?.textContent = text
  }

  private suspend fun bootstrapClangd() {
    if (isExplicitFalseParameter("lsp")) {
      setLspStatus("disabled", "clangd off", "clangd disabled by URL parameter")
      problems.textContent = "clangd is disabled by the lsp=false URL parameter."
      problemsMeta.textContent = "Live language diagnostics are disabled."
      return
    }

    val wasIsolated = cppCrossOriginIsolated()
    if (!wasIsolated) {
      setLspStatus("working", "Isolating…", "Enabling shared memory for clangd")
    }
    val isolationError = ensureCppCrossOriginIsolation()
    if (isolationError != null) {
      setLspStatus("error", "clangd unavailable", isolationError)
      problems.textContent = isolationError
      problemsMeta.textContent = "clangd could not start."
      return
    }
    if (!cppCrossOriginIsolated()) {
      return
    }

    setLspStatus("working", "clangd loading…", "Loading locally built clangd WebAssembly")
    val client = JSClangdClient(
      onStatus = { state, message -> handleClangdStatus(state, message) },
      onProgress = { loaded, total ->
        val percent = if (total > 0) (loaded.toDouble() * 100.0 / total).toInt().coerceIn(0, 100) else 0
        setLspStatus(
          "working",
          if (total > 0) "clangd $percent%" else "clangd loading…",
          if (total > 0) "Loading clangd WebAssembly ($percent%)" else "Loading clangd WebAssembly"
        )
      },
      onDiagnostics = { _, version, items ->
        clangdDiagnostics = items
        monacoEditor?.setDiagnostics(items)
        if (version != null || items.isNotEmpty()) clangdDiagnosticsPublished = true
        renderClangdDiagnostics(analyzing = !clangdDiagnosticsPublished)
        updateEditorChrome()
      },
      onSemanticTokensRefresh = { monacoEditor?.refreshSemanticTokens() }
    )
    clangdClient = client
    monacoEditor?.bindClangd(client)
    client.start(language.fileName, language.queryValue, editorValue())
  }

  private fun setLspStatus(state: String, text: String, title: String = text) {
    lspStatus.setAttribute("data-state", state)
    lspStatus.setAttribute("title", title)
    lspStatus.querySelector(".cpp-status-text")?.textContent = text
  }

  private fun handleClangdStatus(state: ClangdClientState, message: String) {
    when (state) {
      ClangdClientState.STARTING,
      ClangdClientState.LOADING,
      ClangdClientState.INITIALIZING,
      ClangdClientState.BUSY -> {
        clangdFailure = null
        setLspStatus("working", message.ifBlank { "clangd…" }, message)
        if (state == ClangdClientState.BUSY && clangdDiagnostics.isEmpty() && !clangdDiagnosticsPublished) {
          renderClangdDiagnostics(analyzing = true)
        }
      }
      ClangdClientState.READY -> {
        clangdFailure = null
        setLspStatus("ready", "clangd ready", message.ifBlank { "clangd is ready" })
        monacoEditor?.enableClangdSemanticTokens()
        if (clangdDiagnostics.isEmpty() && clangdDiagnosticsPublished) renderClangdDiagnostics()
      }
      ClangdClientState.ERROR -> {
        clangdFailure = message
        setLspStatus("error", "clangd unavailable", message)
        if (clangdDiagnostics.isEmpty()) {
          renderClangdDiagnostics()
        }
      }
      ClangdClientState.STOPPED -> setLspStatus("disabled", "clangd stopped", message)
    }
  }

  private fun renderClangdDiagnostics(analyzing: Boolean = false) {
    problemCount.textContent = clangdDiagnostics.size.toString()
    if (clangdDiagnostics.isEmpty()) {
      clangdFailure?.let { failure ->
        problems.textContent = failure
        problemsMeta.textContent = "clangd stopped before publishing diagnostics."
        return
      }
      if (analyzing) {
        problems.textContent = "Analyzing ${language.fileName}…"
        problemsMeta.textContent = "Waiting for fresh clangd diagnostics."
        return
      }
      problems.textContent = "No problems detected."
      problemsMeta.textContent = "clangd reports a clean translation unit."
      return
    }

    problemsMeta.textContent =
      "${clangdDiagnostics.size} ${if (clangdDiagnostics.size == 1) "problem" else "problems"} reported by clangd."
    problems.textContent = clangdDiagnostics.joinToString("\n\n") { diagnostic ->
      val line = diagnostic.range.start.line + 1
      val column = diagnostic.range.start.character + 1
      val severity = when (diagnostic.severity) {
        1 -> "error"
        2 -> "warning"
        3 -> "information"
        4 -> "hint"
        else -> "problem"
      }
      buildString {
        append(language.fileName)
        append(':')
        append(line)
        append(':')
        append(column)
        append(": ")
        append(severity)
        diagnostic.source?.takeIf { it.isNotBlank() }?.let {
          append(" [")
          append(it)
          diagnostic.code?.takeIf { code -> code.isNotBlank() }?.let { code ->
            append(' ')
            append(code)
          }
          append(']')
        }
        append('\n')
        append(diagnostic.message)
      }
    }
  }

  private fun applyInitialTheme() {
    val requested = parameter("theme")?.lowercase()
    val saved = readLocalStorage(CPP_THEME_KEY)
    val systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches
    val dark = when (requested) {
      "dark" -> true
      "light" -> false
      else -> saved == "dark" || (saved != "light" && systemDark)
    }
    document.body?.classList?.toggle("cpp-dark", dark)
  }

  private fun toggleTheme() {
    val dark = document.body?.classList?.toggle("cpp-dark") ?: false
    writeLocalStorage(CPP_THEME_KEY, if (dark) "dark" else "light")
    monacoEditor?.setTheme(dark)
  }

  private fun editorValue(): String = monacoEditor?.value() ?: source.value

  private fun parameter(name: String): String? = params.get(name) as? String

  private fun isTrueParameter(name: String): Boolean {
    if (!(params.has(name) as Boolean)) return false
    return (parameter(name) ?: "").lowercase() in setOf("", "true", "1", "yes", "y", "on")
  }

  private fun isExplicitFalseParameter(name: String): Boolean =
    (parameter(name) ?: "").lowercase() in setOf("false", "0", "no", "n", "off")

  private fun readLocalStorage(key: String): String? =
    try {
      window.localStorage.getItem(key)
    } catch (_: Throwable) {
      null
    }

  private fun writeLocalStorage(key: String, value: String) {
    try {
      window.localStorage.setItem(key, value)
    } catch (_: Throwable) {
      // Storage can be unavailable in sandboxed or private contexts.
    }
  }

  @Suppress("UNCHECKED_CAST")
  private fun <T : HTMLElement> element(id: String): T =
    (document.getElementById(id) ?: error("Missing #$id")) as T
}

private fun stdioToString(entries: dynamic): String {
  if (entries == null || entries == js("undefined")) return ""
  val length = entries.length as? Int ?: return ""
  return (0 until length)
    .mapNotNull { index -> entries[index]?.text as? String }
    .joinToString("\n")
}

private fun stripAnsi(text: String): String =
  text.replace(Regex("\u001B\\[[0-?]*[ -/]*[@-~]"), "")

private val CPP_CSS = """
  :root {
    color-scheme: light;
    --cpp-bg: #f6f8fb;
    --cpp-panel: #ffffff;
    --cpp-panel-muted: #f0f3f8;
    --cpp-editor: #fbfcfe;
    --cpp-border: #d7dee9;
    --cpp-border-strong: #bbc6d6;
    --cpp-text: #172033;
    --cpp-muted: #667085;
    --cpp-accent: #2563eb;
    --cpp-accent-hover: #1d4ed8;
    --cpp-accent-soft: #dbeafe;
    --cpp-success: #15803d;
    --cpp-warning: #b45309;
    --cpp-danger: #dc2626;
    --cpp-gutter: #f3f5f9;
    --cpp-shadow: 0 8px 30px rgba(31, 41, 55, 0.08);
    --cpp-build-height: min(38vh, 390px);
    --cpp-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    --cpp-sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  body.cpp-dark {
    color-scheme: dark;
    --cpp-bg: #0f141d;
    --cpp-panel: #151b26;
    --cpp-panel-muted: #1b2330;
    --cpp-editor: #111721;
    --cpp-border: #293344;
    --cpp-border-strong: #3c495d;
    --cpp-text: #d8e0ed;
    --cpp-muted: #8f9caf;
    --cpp-accent: #60a5fa;
    --cpp-accent-hover: #93c5fd;
    --cpp-accent-soft: #172d4d;
    --cpp-success: #4ade80;
    --cpp-warning: #fbbf24;
    --cpp-danger: #fb7185;
    --cpp-gutter: #141a24;
    --cpp-shadow: 0 10px 35px rgba(0, 0, 0, 0.28);
  }

  *,
  *::before,
  *::after {
    box-sizing: border-box;
  }

  html,
  body {
    width: 100%;
    height: 100%;
    margin: 0;
    overflow: hidden;
  }

  body {
    background: var(--cpp-bg);
    color: var(--cpp-text);
    font-family: var(--cpp-sans);
  }

  button,
  select,
  textarea {
    font: inherit;
  }

  button,
  select {
    color: inherit;
  }

  button {
    cursor: pointer;
  }

  button:focus-visible,
  select:focus-visible,
  textarea:focus-visible {
    outline: 2px solid var(--cpp-accent);
    outline-offset: -2px;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .is-hidden {
    display: none !important;
  }

  #cpp-root,
  #cpp-app {
    width: 100%;
    height: 100%;
  }

  #cpp-app {
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: var(--cpp-bg);
  }

  #cpp-toolbar {
    min-height: 58px;
    padding: 9px 14px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 1px solid var(--cpp-border);
    background: var(--cpp-panel);
    box-shadow: var(--cpp-shadow);
    z-index: 2;
  }

  .cpp-brand,
  .cpp-toolbar-actions {
    display: flex;
    align-items: center;
  }

  .cpp-brand {
    gap: 10px;
    min-width: 0;
  }

  .cpp-brand-mark {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: var(--cpp-accent);
    color: white;
    font: 800 13px/1 var(--cpp-mono);
    box-shadow: 0 4px 12px color-mix(in srgb, var(--cpp-accent) 28%, transparent);
  }

  .cpp-brand-title {
    font-size: 15px;
    font-weight: 720;
    white-space: nowrap;
  }

  .cpp-badge {
    padding: 3px 7px;
    border: 1px solid var(--cpp-border);
    border-radius: 999px;
    color: var(--cpp-muted);
    background: var(--cpp-panel-muted);
    font: 600 11px/1.2 var(--cpp-mono);
    white-space: nowrap;
  }

  .cpp-toolbar-actions {
    margin-left: auto;
    gap: 8px;
  }

  #cpp-language {
    min-width: 86px;
    height: 36px;
    padding: 0 28px 0 10px;
    border: 1px solid var(--cpp-border);
    border-radius: 8px;
    background: var(--cpp-panel-muted);
  }

  .cpp-button,
  .cpp-icon-button {
    height: 36px;
    border: 1px solid var(--cpp-border);
    border-radius: 8px;
    background: var(--cpp-panel);
  }

  .cpp-button {
    padding: 0 13px;
    font-weight: 650;
  }

  .cpp-icon-button {
    min-width: 36px;
    padding: 0 9px;
    display: inline-grid;
    place-items: center;
    font-size: 20px;
  }

  .cpp-button:hover,
  .cpp-icon-button:hover,
  #cpp-language:hover {
    border-color: var(--cpp-border-strong);
    background: var(--cpp-panel-muted);
  }

  .cpp-button-run {
    min-width: 92px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    border-color: var(--cpp-accent);
    background: var(--cpp-accent);
    color: white;
  }

  .cpp-button-run:hover {
    border-color: var(--cpp-accent-hover);
    background: var(--cpp-accent-hover);
  }

  .cpp-button-run:disabled {
    cursor: wait;
    opacity: 0.72;
  }

  .cpp-button:disabled,
  #cpp-language:disabled {
    cursor: wait;
    opacity: 0.68;
  }

  .cpp-run-icon {
    font-size: 11px;
  }

  .is-running .cpp-run-icon {
    width: 13px;
    height: 13px;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    color: transparent;
    animation: cpp-spin 0.75s linear infinite;
  }

  @keyframes cpp-spin {
    to { transform: rotate(360deg); }
  }

  #cpp-workspace {
    min-height: 0;
    display: flex;
    flex: 1;
    flex-direction: column;
  }

  #cpp-editor-pane {
    min-height: 100px;
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    background: var(--cpp-editor);
  }

  .cpp-pane-header {
    min-height: 35px;
    padding: 0 14px 0 54px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid var(--cpp-border);
    background: var(--cpp-panel);
    color: var(--cpp-muted);
    font: 12px/1 var(--cpp-mono);
  }

  #cpp-file-name {
    color: var(--cpp-text);
  }

  .cpp-pane-hint {
    margin-left: auto;
  }

  #cpp-editor {
    min-height: 0;
    display: flex;
    flex: 1;
    position: relative;
    background: var(--cpp-editor);
  }

  #cpp-monaco {
    position: absolute;
    inset: 0;
    display: none;
    overflow: hidden;
    background: var(--cpp-editor);
  }

  #cpp-editor.has-monaco #cpp-monaco {
    display: block;
  }

  #cpp-editor.has-monaco #cpp-line-numbers,
  #cpp-editor.has-monaco #cpp-source,
  #cpp-editor.has-monaco #cpp-completions {
    display: none !important;
  }

  #cpp-monaco .monaco-editor,
  #cpp-monaco .monaco-editor-background,
  #cpp-monaco .monaco-editor .margin {
    background-color: var(--cpp-editor);
  }

  #cpp-line-numbers,
  #cpp-source {
    margin: 0;
    padding-top: 18px;
    padding-bottom: 32px;
    border: 0;
    border-radius: 0;
    font: 15px/1.65 var(--cpp-mono);
    tab-size: 2;
  }

  #cpp-line-numbers {
    width: 54px;
    padding-left: 8px;
    padding-right: 13px;
    overflow: hidden;
    flex: 0 0 54px;
    border-right: 1px solid var(--cpp-border);
    background: var(--cpp-gutter);
    color: var(--cpp-muted);
    text-align: right;
    user-select: none;
  }

  #cpp-source {
    min-width: 0;
    min-height: 0;
    padding-left: 16px;
    padding-right: 22px;
    flex: 1;
    resize: none;
    overflow: auto;
    outline: none;
    background: transparent;
    color: var(--cpp-text);
    caret-color: var(--cpp-accent);
    white-space: pre;
  }

  #cpp-source::selection,
  #cpp-stdin::selection {
    background: var(--cpp-accent-soft);
  }

  #cpp-completions {
    position: absolute;
    z-index: 5;
    width: min(430px, calc(100% - 76px));
    max-height: 260px;
    overflow: auto;
    border: 1px solid var(--cpp-border-strong);
    border-radius: 9px;
    background: var(--cpp-panel);
    box-shadow: 0 14px 38px rgba(15, 23, 42, 0.22);
    font: 12px/1.35 var(--cpp-mono);
  }

  .cpp-completion-item {
    width: 100%;
    min-height: 29px;
    padding: 5px 9px;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 10px;
    border: 0;
    border-radius: 0;
    background: transparent;
    color: var(--cpp-text);
    text-align: left;
  }

  .cpp-completion-item:hover,
  .cpp-completion-item[aria-selected="true"] {
    background: var(--cpp-accent-soft);
    color: var(--cpp-accent);
  }

  .cpp-completion-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cpp-completion-detail {
    max-width: 170px;
    overflow: hidden;
    color: var(--cpp-muted);
    font-size: 10px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cpp-line-number {
    display: block;
    min-height: 1.65em;
    border-right: 3px solid transparent;
  }

  .cpp-line-number[data-severity="1"] {
    border-right-color: var(--cpp-danger);
    color: var(--cpp-danger);
  }

  .cpp-line-number[data-severity="2"] {
    border-right-color: var(--cpp-warning);
    color: var(--cpp-warning);
  }

  .cpp-line-number[data-severity="3"],
  .cpp-line-number[data-severity="4"] {
    border-right-color: var(--cpp-accent);
  }

  #cpp-build-panel {
    height: var(--cpp-build-height);
    min-height: 72px;
    max-height: calc(100vh - 124px);
    display: flex;
    flex: 0 0 var(--cpp-build-height);
    flex-direction: column;
    border-top: 1px solid var(--cpp-border-strong);
    background: var(--cpp-panel);
  }

  #cpp-build-resize {
    height: 6px;
    margin-top: -3px;
    flex: 0 0 6px;
    cursor: row-resize;
    touch-action: none;
    z-index: 1;
  }

  #cpp-build-resize::after {
    content: "";
    display: block;
    width: 44px;
    height: 3px;
    margin: 1px auto 0;
    border-radius: 999px;
    background: var(--cpp-border-strong);
  }

  .cpp-build-header {
    min-height: 43px;
    padding: 3px 10px 4px 14px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid var(--cpp-border);
  }

  #cpp-build-tabs {
    display: flex;
    align-self: stretch;
    gap: 5px;
  }

  #cpp-build-tabs button {
    padding: 0 11px;
    border: 0;
    border-bottom: 2px solid transparent;
    background: transparent;
    color: var(--cpp-muted);
    font-size: 12px;
    font-weight: 650;
  }

  #cpp-build-tabs button:hover {
    color: var(--cpp-text);
  }

  #cpp-build-tabs button[aria-selected="true"] {
    border-bottom-color: var(--cpp-accent);
    color: var(--cpp-accent);
  }

  .cpp-count {
    min-width: 16px;
    padding: 1px 5px;
    border-radius: 999px;
    background: var(--cpp-panel-muted);
    color: var(--cpp-muted);
    font: 700 10px/1.4 var(--cpp-mono);
  }

  #cpp-close-build {
    width: 32px;
    height: 32px;
    margin-left: auto;
    border: 0;
    background: transparent;
  }

  .cpp-build-content {
    min-height: 0;
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .cpp-tab-panel {
    width: 100%;
    min-height: 0;
    display: flex;
    flex: 1;
    flex-direction: column;
  }

  #cpp-tab-input label,
  .cpp-result-meta {
    padding: 8px 15px 6px;
    color: var(--cpp-muted);
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.025em;
  }

  #cpp-stdin,
  #cpp-output,
  #cpp-diagnostics,
  #cpp-problems {
    width: auto;
    min-height: 0;
    margin: 0 12px 9px;
    padding: 12px 14px;
    flex: 1;
    overflow: auto;
    border: 1px solid var(--cpp-border);
    border-radius: 8px;
    outline: none;
    background: var(--cpp-panel-muted);
    color: var(--cpp-text);
    font: 13px/1.55 var(--cpp-mono);
    white-space: pre-wrap;
    word-break: break-word;
  }

  #cpp-stdin {
    resize: none;
  }

  #cpp-diagnostics,
  #cpp-problems {
    color: var(--cpp-danger);
  }

  .cpp-build-credit {
    padding: 0 14px 8px;
    color: var(--cpp-muted);
    font-size: 10px;
    text-align: right;
  }

  .cpp-build-credit a {
    color: var(--cpp-accent);
    text-decoration: none;
  }

  #cpp-statusbar {
    min-height: 30px;
    padding: 0 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    border-top: 1px solid var(--cpp-border);
    background: var(--cpp-panel);
    color: var(--cpp-muted);
    font-size: 11px;
  }

  .cpp-status-spacer {
    flex: 1;
  }

  .cpp-status-button {
    height: 24px;
    padding: 0 6px;
    border: 0;
    border-radius: 5px;
    background: transparent;
    color: var(--cpp-muted);
    font-size: 11px;
  }

  .cpp-status-button:hover {
    background: var(--cpp-panel-muted);
    color: var(--cpp-text);
  }

  .cpp-runtime-status {
    min-width: 82px;
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: 6px;
  }

  .cpp-runtime-status .cpp-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--cpp-success);
  }

  .cpp-runtime-status[data-state="working"] .cpp-status-dot {
    background: var(--cpp-accent);
    animation: cpp-pulse 1s ease-in-out infinite;
  }

  .cpp-runtime-status[data-state="warning"] .cpp-status-dot {
    background: var(--cpp-warning);
  }

  .cpp-runtime-status[data-state="error"] .cpp-status-dot {
    background: var(--cpp-danger);
  }

  .cpp-runtime-status[data-state="disabled"] .cpp-status-dot {
    background: var(--cpp-muted);
  }

  @keyframes cpp-pulse {
    50% { opacity: 0.35; transform: scale(0.78); }
  }

  @media (max-width: 650px) {
    #cpp-toolbar {
      min-height: 52px;
      padding: 7px 9px;
    }

    .cpp-brand-title,
    .cpp-badge,
    .cpp-button-quiet,
    .cpp-pane-hint {
      display: none;
    }

    .cpp-brand-mark {
      width: 32px;
      height: 32px;
    }

    .cpp-toolbar-actions {
      gap: 6px;
    }

    #cpp-language {
      min-width: 78px;
      height: 34px;
    }

    .cpp-button,
    .cpp-icon-button {
      height: 34px;
    }

    .cpp-button-run {
      min-width: 76px;
    }

    .cpp-pane-header {
      padding-left: 46px;
    }

    #cpp-line-numbers {
      width: 46px;
      flex-basis: 46px;
    }

    #cpp-line-numbers,
    #cpp-source {
      font-size: 13px;
    }

    #cpp-build-tabs button {
      padding: 0 7px;
      font-size: 11px;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
    }
  }
""".trimIndent()
