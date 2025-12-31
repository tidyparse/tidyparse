@file:Suppress("UnsafeCastFromDynamic")

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.tidyparse.*
import kotlinx.browser.*
import kotlinx.coroutines.*
import org.w3c.dom.*
import org.w3c.dom.events.*
import kotlin.js.Promise
import kotlin.time.Duration.Companion.nanoseconds
import kotlin.time.TimeSource

// Access your existing editor elements (or keep your earlier lazy vals)
val cnfInputField by lazy { document.getElementById("tidyparse-input") as HTMLTextAreaElement }
val cnfOutputField by lazy { document.getElementById("tidyparse-output") as Node }

// ---- Minimal CNF-only editor ----
class JSTidyCNFEditor(
  override val editor: HTMLTextAreaElement,
  override val output: Node
) : JSTidyEditor(editor, output) {

  /** Load a CNF from text and refresh highlighting */
  fun loadCNFFromText(text: String) {
    cfg = text.trimIndent().lines().map { it.split(" -> ").let { Pair(it[0], it[1].split(" ")) } }.toSet().freeze()
  }

  override fun getLatestCFG(): CFG = cfg

  override fun redecorateLines(cfg: CFG) { decorator.fullDecorate(cfg) }

  override fun handleInput() {
    val t0 = TimeSource.Monotonic.markNow()
    val context = getApplicableContext()
    if (context.isEmpty()) return
    log("Applicable context:\n$context")

    val tokens = context.tokenizeByWhitespace()

    // Always use the in-memory CFG (loaded from the CNF file)
    val cfg = getLatestCFG()
    if (cfg.isEmpty()) return

    var containsUnkTok = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnkTok = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, minimize, ntStubs).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash.hashCode()
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)
    else writeDisplayText("")

    runningJob?.cancel()

    val scenario = when {
      tokens.size == 1 && stubMatcher.matches(tokens[0]) -> Scenario.STUB
      HOLE_MARKER in tokens -> Scenario.COMPLETION
      !containsUnkTok && tokens in cfg.language -> Scenario.PARSEABLE
      else -> Scenario.REPAIR
    }

    when(scenario) {
      Scenario.REPAIR -> writeDisplayText("Searching for repairs... (please be patient)")
      Scenario.COMPLETION -> writeDisplayText("Generating completions... (please be patient)")
      Scenario.STUB -> writeDisplayText("Stub completion...")
      else -> {}
    }

    var i = 0; suspend fun pause(freq: Int = 3) { if (i++ % freq == 0) { delay(50.nanoseconds) }}
    runningJob = MainScope().launch {
      when (scenario) {
        Scenario.STUB -> cfg.enumNTSmall(tokens[0].stripStub()).take(100)
        Scenario.COMPLETION -> cfg.enumSeqSmartSuspendable(tokens, suspender = { pause() })
          .take(100)
          .enumerateInteractively(
            workHash = workHash,
            origTks = tokens,
            reason = scenario.reason,
            postCompletionSummary = { ", ${t0.elapsedNow()} latency." }
          )
        Scenario.PARSEABLE -> writeDisplayText(parsedPrefix.dropLast(8).also { cache[workHash] = it })
        Scenario.REPAIR ->
          (if (gpuAvailable)
            repairCode(cfg, tokens, if (minimize) 0 else LED_BUFFER).asSequence()
              .map { it.replace("ε", "").tokenizeByWhitespace().joinToString(" ") }
          else sampleGREUntilTimeout(tokens, cfg)).enumerateInteractively(
            workHash = workHash,
            metric = { _ -> 1 },
            origTks = tokens,
            reason = scenario.reason,
            postCompletionSummary = { ", ${t0.elapsedNow()} latency." }
          )
      }
    }
  }
}

lateinit var jsCnfEditor: JSTidyCNFEditor

// ---- Modal + wiring for CNF ----
fun cnfSetup() {
  jsCnfEditor = JSTidyCNFEditor(cnfInputField, cnfOutputField)
  val overlay = buildCnfModal()
  document.body?.appendChild(overlay)

  // Wire keystrokes to the CNF editor (parse/complete/repair)
  cnfInputField.addEventListener("input", { jsCnfEditor.run { continuation { handleInput() } } })
  cnfInputField.addEventListener("input", { jsCnfEditor.redecorateLines() })
  cnfInputField.addEventListener("keydown", { e -> jsCnfEditor.navUpdate(e as KeyboardEvent) })

// Optional: read initial settings and keep LED buffer in sync
  val ledBuffSel = document.getElementById("led-buffer") as? HTMLInputElement
  LED_BUFFER = ledBuffSel?.value?.toIntOrNull() ?: LED_BUFFER
  ledBuffSel?.addEventListener("change", { LED_BUFFER = ledBuffSel.value.toInt() })

  fun show() {
    overlay.style.display = "flex"
    document.documentElement?.classList?.add("cnf-lock-scroll")
    window.setTimeout({
      (document.getElementById("cnfFileInput") as? HTMLInputElement)?.focus()
    }, 50)
  }
  fun hide() {
    overlay.style.display = "none"
    document.documentElement?.classList?.remove("cnf-lock-scroll")
  }

  // Open on DOM ready; if we're already past loading, show immediately.
  document.addEventListener("DOMContentLoaded", { show() })
  if (document.readyState != DocumentReadyState.LOADING) show()

  var lastUrl: String? = null
  val input = document.getElementById("cnfFileInput") as HTMLInputElement
  val fileName = document.getElementById("cnfFileName") as HTMLSpanElement

  input.addEventListener("change", { _ ->
    MainScope().launch {
      try {
        val f = input.files?.item(0) ?: return@launch
        fileName.textContent = f.name

        // Update/revoke blob URL
        lastUrl?.let { js("URL.revokeObjectURL")(it) }
        val url = js("URL.createObjectURL")(f) as String
        lastUrl = url

        // Read text (+ bytes optional)
        val text = try { f.asDynamic().text().unsafeCast<Promise<String>>().await() } catch (_: Throwable) { null }
        val ab = try { f.asDynamic().arrayBuffer().unsafeCast<Promise<org.khronos.webgl.ArrayBuffer>>().await() } catch (_: Throwable) { null }
        val bytes = ab?.let { org.khronos.webgl.Uint8Array(it) }

        // Publish to window
        val obj = js("{}")
        obj.file = f; obj.name = f.name; obj.size = f.size
        obj.type = f.type ?: "text/plain"; obj.url = url
        obj.text = text; obj.bytes = bytes
        window.asDynamic().tidySelectedFile = obj

        // Notify listeners
        val init = js("{}"); init.detail = obj
        window.dispatchEvent(js("new CustomEvent('tidy:cnf-loaded', init)") as Event)

        // Parse + decorate (if you do this here)
        text?.let {
          try {
            jsCnfEditor.loadCNFFromText(it)
            // Immediately re-run with the active caret context
            jsCnfEditor.run { continuation { handleInput() } }
            jsCnfEditor.redecorateLines()
            log("Loaded ${it.length} bytes from ${obj.name}")
          } catch (e: dynamic) {
            console.error("Failed to parse CNF:", e)
          }
        }
      } catch (e: dynamic) {
        console.error("CNF load error:", e)
      } finally {
        hide()
        document.documentElement?.classList?.remove("cnf-lock-scroll")
      }
    }
  })

  // Optional: only allow Esc to close after something was chosen
  document.addEventListener("keydown", { e ->
    if ((e as KeyboardEvent).key == "Escape" && window.asDynamic().tidySelectedFile != undefined) hide()
  })
}

private fun buildCnfModal(): HTMLDivElement {
  val overlay = (document.createElement("div") as HTMLDivElement).apply {
    id = "cnfModal"; className = "cnf-modal__overlay"
    setAttribute("role","dialog"); setAttribute("aria-modal","true")
    setAttribute("aria-labelledby","cnfModalTitle")
  }
  val dialog = (document.createElement("div") as HTMLDivElement).apply { className = "cnf-modal__dialog" }
  val header = (document.createElement("div") as HTMLDivElement).apply {
    className = "cnf-modal__header"; id = "cnfModalTitle"; textContent = "Load a CNF grammar"
  }
  val body = (document.createElement("div") as HTMLDivElement).apply {
    className = "cnf-modal__body"
    innerHTML = """
    <p><em>CNF file requirements:</em></p>
    <ul>
      <li>The start symbol must be named <code>START</code>.</li>
      <li>Grammar must be in <em>Chomsky Normal Form</em>:
        <ul>
          <li>Binary: <code>W -&gt; X Z</code> (with <code>W,X,Z ∈ V</code> nonterminals)</li>
          <li>Unary (lexical): <code>W -&gt; a</code> (with <code>a ∈ Σ</code> terminal)</li>
        </ul>
      </li>
      <li>One production per line. No alternation (<code>|</code>).</li>
      <li>Tokens are space-separated; use the literal arrow <code>-&gt;</code>.</li>
    </ul>
    <p><em>Example:</em><br><br>
      <code>START -&gt; EXPR EXPR</code><br>
      <code>EXPR -&gt; LPAREN EXPR</code><br>
      <code>LPAREN -&gt; (</code>
    </p>
  """.trimIndent()
  }
  val footer = (document.createElement("div") as HTMLDivElement).apply { className = "cnf-modal__footer" }
  val btn = (document.createElement("label") as HTMLLabelElement).apply {
    className = "cnf-btn"; setAttribute("for","cnfFileInput"); title = "Choose a CNF file"; textContent = "Choose CNF…"
  }
  val input = (document.createElement("input") as HTMLInputElement).apply {
    id = "cnfFileInput"; className = "cnf-input"; type = "file"; accept = ".cnf"
  }
  val fileName = (document.createElement("span") as HTMLSpanElement).apply {
    id = "cnfFileName"; className = "cnf-file-name"; setAttribute("aria-live","polite")
    textContent = "No file selected"
  }
  btn.appendChild(input)
  footer.appendChild(btn)
  footer.appendChild(fileName)
  dialog.appendChild(header)
  dialog.appendChild(body)
  dialog.appendChild(footer)
  overlay.appendChild(dialog)
  return overlay
}
