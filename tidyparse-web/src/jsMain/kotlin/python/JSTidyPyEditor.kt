import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.*
import ai.hypergraph.tidyparse.wgpu.*
import kotlinx.browser.*
import kotlinx.coroutines.*
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent
import web.gpu.GPUBuffer
import kotlin.js.Promise
import kotlin.math.ln
import kotlin.time.TimeMark
import kotlin.time.TimeSource

@ExperimentalUnsignedTypes
class JSTidyPyEditor(editor: HTMLTextAreaElement, output: Node) : JSTidyEditor(editor, output) {
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

    suspend fun Sequence<String>.filterCompilerErrors(errHst: MutableMap<String, Int> = mutableMapOf(), window: Int = 16): Sequence<String> = coroutineScope {
      val kept = ArrayList<String>()
      val iterator = iterator()
      val inFlight = ArrayDeque<Deferred<Pair<String, String>>>()

      fun launchOne() {
        if (!iterator.hasNext()) return
        val s = iterator.next()
        inFlight.addLast(async { s to pythonCompilerOutput(s) })
      }

      repeat(window) { launchOne() }

      while (inFlight.isNotEmpty()) {
        val (s, output) = inFlight.removeFirst().await()
        val ok = when (val errorType = output.getErrorType()) {
          "" -> true
          else -> {
            "$errorType: ${output.getErrorMessage()}".also { err -> errHst[err] = 1 + errHst.getOrElse(err) { 0 } }
            false
          }
        }

        if (ok) kept.add(s)

        launchOne()
      }

      kept.asSequence()
    }

    suspend fun handleInput(line: String, cfg: CFG, maxResults: Int = 50): List<String> {
      val currentLine = line.trim()
      if (currentLine.isBlank() || currentLine.startsWith("#")) return emptyList()

      val pcs = PyCodeSnippet(currentLine)
      val tokens = pcs.lexedTokens().tokenizeByWhitespace().map { if (it == "|") "OR" else it }
      val errHst = mutableMapOf<String, Int>()

      return repairCode(
        cfg,
        tokens,
        LED_BUFFER,
        rerankerQuery = neuralRerankerQuery(tokens),
        reranker = RepairReranker::rerankOrOriginal
      )
          .pythonRepairs().map {
            pcs.restitch(levenshteinAlign(tokens.dropLast(1), it.tokenizeByWhitespace()))
          }.distinct().filterCompilerErrors(errHst).take(maxResults).toList().also { errHst.logRejections() }
    }
  }

  override fun writeDisplayText(s: Σᐩ) =
    setCompletionsAndShow(s.split("\n")
      .map { it.substringAfter("</span>") }
      .drop(2).dropLast(2))

  fun score(text: List<String>): Double =
    -(prefix + text + suffix).windowed(order, 1)
      .sumOf { ngram -> ln((ngrams[ngram] ?: 1.0) / normalizingConst) }

  var pyodide: dynamic? = null
  var blackFormatFn: dynamic = null

  fun String.sanitizeForPyodideCompiler() = replace("NUMBER",  "1").replace("STRING", "\"\"")
  fun getOutput(code: String): String = try {
    if (pyodide == null) throw Exception("Pyodide not initialized")
    val src = code.sanitizeForPyodideCompiler()

    val encoded: String = js("btoa")(src) as String

    //language=python
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
  } catch (e: dynamic) { "" } //{ "Error during compilation: $e".also { log(it) }; "" }

  suspend fun formatCodeAsync(code: String): String =
    try { ensurePyCompileWorkers().format(code).also { if (it.startsWith("__BLACK_ERROR__")) log(it) } }
    catch (t: Throwable) { log("Worker formatting failed: ${t.message ?: t}"); formatCode(code) }

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

  override fun navUpdate(event: KeyboardEvent) {
    val key = event.keyCode.toSelectorAction() ?: return
    val hasStub = stubMatcher.find(currentLine(), 0) != null
    if (key == SelectorAction.TAB && hasStub) { event.preventDefault(); handleTab(); return }
  }

  val cme by lazy { js("window.cmEditor") }
  override fun setCaretPosition(range: IntRange) =
    cme.setSelection(cme.posFromIndex(range.first), cme.posFromIndex(range.last))

  override fun handleInputNow(started: TimeMark) {
    window.asDynamic().COMPLETIONS = arrayOf<String>()
    val t0 = started
    val currentLine = currentLine().also { log("Current line is: $it") }
    if (currentLine.isBlank() || currentLine.trimStart().startsWith("#")) return
    val pcs = PyCodeSnippet(currentLine)
    val tokens = pcs.lexedTokens().tokenizeByWhitespace().map { if (it == "|") "OR" else it }

    log("Repairing: " + tokens.dropLast(1).joinToString(" "))

    var containsUnk = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnk = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, epsilons, neuralRerankerEnabled).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash
    if (!inputWorkInvalidated && workHash == currentWorkHash) return
    inputWorkInvalidated = false
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
        var total = 0
        val gpuRepairs = if (gpuAvailable) {
          log("Repairing on GPU...")
          repairCode(
            cfg, tokens, LED_BUFFER,
            rerankerQuery = neuralRerankerQuery(tokens),
            reranker = RepairReranker::rerankOrOriginal,
            requestStarted = started
          )
        } else { log("Repairing on CPU..."); null }

        val repairs = gpuRepairs?.pythonRepairs()
          ?: sampleGREUntilTimeout(tokens, cfg).map { it.toPythonRepair() }.distinct()
        val repairMetric: ((List<String>) -> Int)? = if (gpuRepairs != null) null
          else { repair -> (levenshtein(tokens.dropLast(1), repair) * 10_000 + score(repair) * 1_000.0).toInt() }

        val postProcTimer = TimeSource.Monotonic.markNow()
        val compilerFilteredRepairs = repairs.onEach { total++ }
          .let { if (allowCompilerErrors) it else it.filterCompilerErrors(errHst) }

        compilerFilteredRepairs.withIndex().enumerateInteractively(
          workHash = workHash,
          keyOf = { it.value },
          resultsToPost = MAX_DISP_RESULTS,
          metric = { repair ->
            repairMetric?.invoke(repair.value.tokenizeByWhitespace()) ?: repair.index
          },
          customDiff = { repair ->
            val repairTks = repair.value.tokenizeByWhitespace()
            pcs.paintDiffAsync(levenshteinAlign(tokens.dropLast(1), repairTks)) { formatCodeAsync(it) }
          },
          postCompletionSummary = {
            errHst.logRejections()

            if (gpuAvailable) {
              mark("postprocessing", postProcTimer)
              timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
              log("Results rendered in ${timings["total"]}ms")
              timings.logTimesheet()
            }

            ", discarded ${errHst.values.sum()}/$total, ${t0.elapsedNow()} latency."
          },
          reason = invalidPrefix
        )
      }
    }
  }

  suspend fun initPyodide() {
    var loading = "Python runtime"

    fun statusLoading(id: String, label: String) {
      loading = label
      setPythonRuntimeStatus(id, "pending", "$label loading")
    }

    fun statusReady(id: String, label: String) =
      setPythonRuntimeStatus(id, "ready", "$label ready")

    try {
      val scriptTag = (document.querySelector("script[src*='pyodide.js']") as HTMLScriptElement)
        .getAttribute("src")!!.substringBefore("pyodide.js")

      val workerPoolReady = startPyodideWorkers(scriptTag)

      val config = js("{}")
      config.indexURL = scriptTag
      jsPyEditor.pyodide = window.asDynamic().loadPyodide(config).unsafeCast<Promise<*>>().await()
      setPythonRuntimeStatus(PY_STATUS_WORKERS, "warming", "Python runtime ready; workers loading")

      statusLoading(PY_STATUS_BLACK, "Python formatter")
      installVendoredBlack(jsPyEditor.pyodide)

      val testStr = "1+1"
      val beautified = jsPyEditor.formatCode(testStr)
      log("Main-thread Black test => $beautified")
      if (beautified != "1 + 1") throw RuntimeException("Black sanity check returned '$beautified'")
      setPythonRuntimeStatus(PY_STATUS_BLACK, "warming", "Python formatter ready; worker formatter loading")

      log(jsPyEditor.getOutput("1+"))

      loading = "Python workers"
      val pool = workerPoolReady.await()

      loading = "Python workers"
      val workerCompileReady = pool.compile("1+").output.getErrorType() == "SyntaxError"
      log("Worker compile ready => $workerCompileReady")
      if (!workerCompileReady) throw RuntimeException("Worker compile sanity check failed")
      statusReady(PY_STATUS_WORKERS, "Python workers")

      loading = "Python formatter"
      val workerFormat = pool.format("x=1")
      val workerFormatReady = workerFormat == "x = 1"
      log("Worker format ready => $workerFormatReady")
      if (!workerFormatReady) throw RuntimeException("Worker Black sanity check returned '$workerFormat'")
      statusReady(PY_STATUS_BLACK, "Python formatter")
    } catch (e: Throwable) {
      markPythonRuntimeStatusesNotReadyError("$loading failed: ${e.message ?: e.toString()}")
      log("Error during Pyodide initialization: ${e.message ?: e.toString()}")
    }
  }
}

private fun IntersectionResults.pythonRepairs(): Sequence<String> =
  mapTerminals { it.toPythonTerminal() }.asSequence().map { it.withoutFinalNewline() }.distinct()

private fun String.toPythonRepair() =
  withoutFinalNewline().tokenizeByWhitespace().joinToString(" ") { it.toPythonTerminal() }

private fun String.withoutFinalNewline() = if (this == "NEWLINE") "" else removeSuffix(" NEWLINE")

private fun String.toPythonTerminal() = when (this) {
  "OR" -> "|"
  "not_in" -> "not in"
  "is_not" -> "is not"
  else -> this
}

private fun Map<String, Int>.logRejections() {
  if (isEmpty()) return
  val pad = (values.maxOrNull()?.toString()?.length ?: 1) + 1
  val summary = entries.sortedByDescending { it.value }
    .joinToString("\n") { "${it.value.toString().padEnd(pad)}| ${it.key}" }
  log("Rejection histogram:\n$summary")
}

private fun String.getErrorType(): String =
  if (isEmpty()) "" else lines().dropLast(1).lastOrNull()?.substringBeforeLast(":")?.substringAfterLast(":1: ") ?: this

private fun String.getErrorMessage(): String = substringAfterLast(": ").substringBefore('.').trim()

private const val PYODIDE_CDN_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/"
private const val WEBWORKERS = 16
private const val PY_STATUS_BLACK = "blackStatus"
private const val PY_STATUS_WORKERS = "pyWorkerStatus"

private val PY_RUNTIME_STATUS_IDS = listOf(
  PY_STATUS_BLACK,
  PY_STATUS_WORKERS
)

private fun setPythonRuntimeStatus(id: String, state: String, label: String) {
  val node = document.getElementById(id) as? HTMLElement ?: return
  node.classList.remove("pending")
  node.classList.remove("warming")
  node.classList.remove("ready")
  node.classList.remove("error")
  node.classList.add(state)
  node.setAttribute("aria-label", label)
  node.setAttribute("title", label)
}

private fun markPythonRuntimeStatusesNotReadyError(label: String) =
  PY_RUNTIME_STATUS_IDS.forEach { id ->
    val node = document.getElementById(id) as? HTMLElement ?: return@forEach
    if (!node.classList.contains("ready")) setPythonRuntimeStatus(id, "error", label)
  }

private fun pyodideIndexURLFromDocument(): String? =
  try {
    (document.querySelector("script[src*='pyodide.js']") as? HTMLScriptElement)
      ?.getAttribute("src")
      ?.substringBefore("pyodide.js")
  } catch (_: Throwable) { null }

private fun pyodideIndexURL(): String = pyodideIndexURLFromDocument() ?: PYODIDE_CDN_INDEX_URL

private suspend fun installVendoredBlack(pyodide: dynamic) {
  val alreadyInstalled = pyodide.runPython(
    "import importlib.util; importlib.util.find_spec('black') is not None"
  ) as Boolean

  if (alreadyInstalled) return

  val response = window.asDynamic().fetch(PYODIDE_BLACK_VENDOR_ARCHIVE).unsafeCast<Promise<dynamic>>().await()
  val ok = response.ok as? Boolean ?: false
  if (!ok) throw RuntimeException("Failed to load vendored Black archive: ${response.status} ${response.statusText}")

  val buffer = response.arrayBuffer().unsafeCast<Promise<dynamic>>().await()
  val sitePackages = pyodide.runPython("import site; site.getsitepackages()[0]") as String
  val options = js("{}")
  options.extractDir = sitePackages
  pyodide.unpackArchive(buffer, "zip", options)
}

private var webWorkerPoolReady: Deferred<WebWorkerPool>? = null
private fun startPyodideWorkers(indexURL: String = pyodideIndexURL()): Deferred<WebWorkerPool> = webWorkerPoolReady ?: MainScope().async {
  try {
    WebWorkerPool(indexURL = indexURL, size = WEBWORKERS)
      .also {
        log("Started $WEBWORKERS Python Web Workers")
        it.init()
        log("Initialized $WEBWORKERS Python Web Workers")
      }
  } catch (t: Throwable) {
    webWorkerPoolReady = null
    log("Failed to initialize Python Web Workers: ${t.message ?: t}")
    throw t
  }
}.also { webWorkerPoolReady = it }

private suspend fun ensurePyCompileWorkers(): WebWorkerPool = startPyodideWorkers().await()

private suspend fun pythonCompilerOutput(code: String): String =
  try { ensurePyCompileWorkers().compile(code).output }
  catch (t: Throwable) {
    log("Python compiler unavailable: ${t.message ?: t}")
    "__TIDYPARSE_COMPILER_INFRA_ERROR__: ${t.message ?: t}\n"
  }
