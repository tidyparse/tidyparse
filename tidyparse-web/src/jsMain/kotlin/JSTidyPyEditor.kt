import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.*
import kotlinx.browser.window
import kotlinx.coroutines.*
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent
import web.gpu.GPUBuffer
import kotlin.math.ln
import kotlin.time.TimeSource


@ExperimentalUnsignedTypes
class JSTidyPyEditor(override val editor: HTMLTextAreaElement, override val output: Node) : JSTidyEditor(editor, output) {
  val ngrams: MutableMap<List<String>, Double> = mutableMapOf()

  val order: Int by lazy { ngrams.keys.firstOrNull()!!.size }
  val normalizingConst by lazy { ngrams.values.sum() }
  var allowCompilerErrors = false

  val ngramTensor: GPUBuffer by lazy { ngrams.toGpuHash(cfg = cfg).loadToGPUBuffer() }

  val PLACEHOLDERS = listOf("STRING", "NAME", "NUMBER")
  override val stubMatcher: Regex = Regex(PLACEHOLDERS.joinToString("|") { Regex.escape(it) })

  override fun getLatestCFG(): CFG = pythonStatementCNFAllProds.apply { cfg = this }

  private val lineParseCache = HashMap<String, Boolean>()

  override fun redecorateLines(cfg: CFG) {
    val currentHash = ++hashIter

    fun decorate() {
      if (currentHash != hashIter) return
      val decCFG = getLatestCFG()

      val text = readEditorText()
      val invalid = LinkedHashMap<Int, String>()

      val lines = text.lines()
      for ((ln, rawLine) in lines.withIndex()) {
        val trimmed = rawLine.trim()
        if (trimmed.isEmpty() || trimmed.startsWith("#")) continue

        val ok = lineParseCache.getOrPut(trimmed) {
          PyCodeSnippet(trimmed).lexedTokens().replace("|", "OR") in decCFG.language
        }

        if (!ok) invalid[ln] = "Unparseable (line ${ln + 1})"
      }

      pyDecorator.setInvalidLines(invalid)

      if (currentHash != hashIter) return
      pyDecorator.fullDecorate()
    }

    continuation { decorate() }
  }

  companion object {
    val prefix = listOf("BOS", "NEWLINE")
    val suffix = listOf("NEWLINE", "EOS")
  }

  override fun writeDisplayText(s: Σᐩ) {
    setCompletionsAndShow(s.split("\n")
      .map { it.substringAfter("</span>") }
      .drop(2).dropLast(2))
  }

  fun score(text: List<String>): Double =
    -(prefix + text + suffix).windowed(order, 1)
      .sumOf { ngram -> ln((ngrams[ngram] ?: 1.0) / normalizingConst) }

  var pyodide: dynamic? = null

  fun getOutput(code: String): String = try {
    if (pyodide == null) throw Exception("Pyodide not initialized")
    val src = code.replace("NUMBER",  "1").replace("STRING", "\"\"")

    val encoded: String = js("btoa")(src) as String

    val pyCode = """
        import sys, traceback, io, base64, textwrap
        _out = io.StringIO()
        sys.stdout = sys.stderr = _out
        try:
            _src = base64.b64decode("$encoded").decode("utf-8")
            _src = textwrap.dedent(_src)
            compile(_src, "test_compile.py", "exec")
        except Exception:
            traceback.print_exc()
        _result = _out.getvalue()
    """.trimIndent()

    jsPyEditor.pyodide.runPython(pyCode)
    jsPyEditor.pyodide.globals.get("_result") as String
  } catch (e: dynamic) {""}//{ "Error during compilation: $e".also { log(it) }; "" }

  private fun String.getErrorType(): String =
    if (isEmpty()) "" else lines().dropLast(1).lastOrNull()?.substringBeforeLast(":")?.substringAfterLast(":1: ") ?: this

  private fun String.getErrorMessage(): String = substringAfterLast(": ").substringBefore('.').trim()

  override fun formatCode(code: String): String = try {
    if (pyodide == null) throw Exception("Pyodide not initialized")
    jsPyEditor.pyodide.runPython("""
      from black import format_str, FileMode
      pretty_code = format_str("${code.replace("\\", "\\\\").replace("\"", "\\\"")}", mode=FileMode(string_normalization=False))
    """.trimIndent())
    jsPyEditor.pyodide.globals.get("pretty_code").trim().replace("\n", " ")
  } catch (error: dynamic) {
    code.also { log("Error formatting Python code: $error") }
  }

  fun String.replacePythonKeywords() =
    replace("OR", "|").replace("not_in", "not in").replace("is_not", "is not")

  override fun navUpdate(event: KeyboardEvent) {
    val key = event.keyCode.toSelectorAction() ?: return
    val hasStub = stubMatcher.find(currentLine(), 0) != null
    if (key == SelectorAction.TAB && hasStub) { event.preventDefault(); handleTab(); return }
  }

  val cme by lazy { js("window.cmEditor") }
  override fun setCaretPosition(range: IntRange) =
    cme.setSelection(cme.posFromIndex(range.first), cme.posFromIndex(range.last))

  override fun handleInput() {
    window.asDynamic().COMPLETIONS = arrayOf<String>()
    val t0 = TimeSource.Monotonic.markNow()
    val currentLine = currentLine().also { log("Current line is: $it") }
    if (currentLine.isBlank() || currentLine.trimStart().startsWith("#")) return
    val pcs = PyCodeSnippet(currentLine)
    val tokens = pcs.lexedTokens().tokenizeByWhitespace().map { if (it == "|") "OR" else it }

    log("Repairing: " + tokens.dropLast(1).joinToString(" "))

    var containsUnk = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnk = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, epsilons).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    runningJob?.cancel()

    val errHst = mutableMapOf<String, Int>()
    if (!containsUnk && tokens in cfg.language) {
//      val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
      val compilerFeedback = getOutput(pcs.rawCode)
        .let { tcm -> if (tcm.getErrorType().isEmpty()) "" else "\n\n⚠\uFE0F ${tcm.getErrorMessage()}" }
      writeDisplayText("✅ ${tokens.dropLast(1).joinToString(" ")}$compilerFeedback".also { cache[workHash] = it })
    } else /* Repair */ Unit.also {
      runningJob = MainScope().launch {
        var (rejected, total) = 0 to 0
//      var metric: (List<String>) -> Int = { (score(it) * 1_000.0).toInt() } // TODO: Is reordering really necessary if we are decoding GREs by ngram score?
        var metric: (List<String>) -> Int = { (levenshtein(tokens.dropLast(1), it) * 10_000 + score(it) * 1_000.0).toInt() }
//        var metric: (List<String>) -> Int = { -1 }

        (if (gpuAvailable) {
          log("Repairing on GPU...")
          repairCode(cfg, tokens, LED_BUFFER, ngramTensor).asSequence()
        } else {
          log("Repairing on CPU...")
          metric = { (levenshtein(tokens.dropLast(1), it) * 10_000 + score(it) * 1_000.0).toInt() }
          sampleGREUntilTimeout(tokens, cfg)
        })
          // Drop NEWLINE (added by default to PyCodeSnippets)
          .map { it.dropLast(8).replacePythonKeywords() }
          .distinct().let {
            if (allowCompilerErrors) it.onEach { total++ }
            else it.filter { s ->
//              if (total - rejected)
              val output = getOutput(s)
              val errorType = output.getErrorType()
              when (errorType) {
                "" -> true
                else -> {
                  "$errorType: ${output.getErrorMessage()}"
                    .also { errHst[it] = 1 + errHst.getOrElse(it) {
//                    log("REPAIR: $s\nERROR: $it")
                    0 }; }
                   false
                }
              }.also { if (!it) rejected++; total++ }
            }
          }.enumerateInteractively(
            workHash = workHash,
            origTks = tokens.dropLast(1),
            recognizer = { "$it NEWLINE".replace("|", "OR") in cfg.language },
            metric = metric,
            customDiff = {
              val levAlign = levenshteinAlign(tokens.dropLast(1), it.tokenizeByWhitespace())
              pcs.paintDiff(levAlign) { formatCode(it) }
            },
            postCompletionSummary = {
              if (errHst.isNotEmpty()) {
                val pad = (errHst.values.maxOrNull()?.toString()?.length ?: 1) + 1
                val summary = errHst.toMap().entries.sortedBy { -it.component2() }
                    .joinToString("\n") { "${it.value.toString().padEnd(pad)}| ${it.key}" }
                log("Rejection histogram:\n$summary")
              }
              ", discarded $rejected/$total, ${t0.elapsedNow()} latency."
            },
            reason = invalidPrefix
          )
      }
    }
  }
}