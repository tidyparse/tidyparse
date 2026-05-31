import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.*
import kotlinx.browser.*
import kotlinx.coroutines.*
import org.kosat.round
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent
import web.gpu.GPUBuffer
import kotlin.js.Promise
import kotlin.math.ln
import kotlin.time.DurationUnit
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

  suspend fun Sequence<String>.filterCompilerErrors(
    errHst: MutableMap<String, Int> = mutableMapOf(),
    window: Int = 16,
    onSeen: (ok: Boolean) -> Unit = {},
  ): Sequence<String> = coroutineScope {
    val pool = try { ensurePyCompileWorkers() }
    catch (t: Throwable) { log("Pyodide worker pool unavailable: ${t.message ?: t}"); null }

    val kept = ArrayList<String>()
    val iterator = iterator()
    val inFlight = ArrayDeque<Deferred<Pair<String, String>>>()

    fun launchOne() {
      if (!iterator.hasNext()) return
      val s = iterator.next()
      inFlight.addLast(async { s to (pool?.compile(s)?.output ?: getOutput(s)) })
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
      onSeen(ok)

      launchOne()
    }

    kept.asSequence()
  }

  private fun String.getErrorType(): String =
    if (isEmpty()) "" else lines().dropLast(1).lastOrNull()?.substringBeforeLast(":")?.substringAfterLast(":1: ") ?: this

  private fun String.getErrorMessage(): String = substringAfterLast(": ").substringBefore('.').trim()

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

        val repairs =
          (if (gpuAvailable) {
            log("Repairing on GPU...")
            repairCode(cfg, tokens, LED_BUFFER, ngramTensor).asSequence()
          } else {
            log("Repairing on CPU...")
            metric = { (levenshtein(tokens.dropLast(1), it) * 10_000 + score(it) * 1_000.0).toInt() }
            sampleGREUntilTimeout(tokens, cfg)
          }).map { it.dropLast(8).replacePythonKeywords() }.distinct()

        val postProcTimer = TimeSource.Monotonic.markNow()
        val compilerFilteredRepairs =
          if (allowCompilerErrors) { repairs.onEach { total++ } }
          else { repairs.filterCompilerErrors(errHst = errHst) { ok -> if (!ok) rejected++; total++ } }

        compilerFilteredRepairs.enumerateRankedRepairsInteractively(
          workHash = workHash,
          metric = metric,
          render = { _, repairTks ->
            pcs.paintDiffAsync(levenshteinAlign(tokens.dropLast(1), repairTks)) { formatCodeAsync(it) }
          },
          postCompletionSummary = {
            if (errHst.isNotEmpty()) {
              val pad = (errHst.values.maxOrNull()?.toString()?.length ?: 1) + 1
              val summary = errHst.toMap().entries.sortedBy { -it.component2() }
                .joinToString("\n") { "${it.value.toString().padEnd(pad)}| ${it.key}" }
              log("Rejection histogram:\n$summary")
            }

            if (gpuAvailable) {
              mark("postprocessing", postProcTimer)
              timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
              log("Results rendered in ${timings["total"]}ms")
              timings.logTimesheet()
            }

            ", discarded $rejected/$total, ${t0.elapsedNow()} latency."
          },
          reason = invalidPrefix
        )
      }
    }
  }

  suspend fun initPyodide() = try {
    val scriptTag = (document.querySelector("script[src*='pyodide.js']") as HTMLScriptElement)
      .getAttribute("src")!!.substringBefore("pyodide.js")

    val workerPoolReady = startPyodideWorkers(8)
    val config = js("{}")
    config.indexURL = scriptTag
    jsPyEditor.pyodide = window.asDynamic().loadPyodide(config).unsafeCast<Promise<*>>().await()
    jsPyEditor.pyodide.loadPackage("micropip").unsafeCast<Promise<*>>().await()

    val micropip = jsPyEditor.pyodide.pyimport("micropip")
    micropip.install("black").unsafeCast<Promise<*>>().await()

    val testStr = "1+1"
    val beautified = jsPyEditor.formatCode(testStr)

    log("Main-thread Black test => $beautified")
    log(jsPyEditor.getOutput("1+"))

    val pool = workerPoolReady.await()
    log("Worker compile ready => ${pool.compile("1+").output.getErrorType() == "SyntaxError"}")
    log("Worker format ready => ${pool.format("x=1") == "x = 1"}")
  } catch (e: Throwable) { log("Error during Pyodide initialization: ${e.message ?: e.toString()}") }

  private val pyodideIndexURL: String by lazy {
    (document.querySelector("script[src*='pyodide.js']") as HTMLScriptElement)
      .getAttribute("src")!!.substringBefore("pyodide.js")
  }

  private var webWorkerPoolReady: Deferred<WebWorkerPool>? = null
  fun startPyodideWorkers(nWorkers: Int = 16): Deferred<WebWorkerPool> {
    webWorkerPoolReady?.let { return it }
    return MainScope().async {
      try {
        WebWorkerPool(indexURL = pyodideIndexURL, size = nWorkers)
          .also { it.init(); log("Initialized $nWorkers Web Workers") }
      } catch (t: Throwable) {
        webWorkerPoolReady = null
        log("Failed to initialize Web Workers: ${t.message ?: t}")
        throw t
      }
    }.also { webWorkerPoolReady = it }
  }

  suspend fun ensurePyCompileWorkers(): WebWorkerPool = startPyodideWorkers().await()

  private data class RankedCompletion(val raw: String, val tokens: List<String>, val score: Int)

  private suspend fun Sequence<String>.enumerateRankedRepairsInteractively(
    workHash: Int,
    metric: (List<String>) -> Int,
    render: suspend (raw: String, tokens: List<String>) -> String,
    postCompletionSummary: () -> String = { "." },
    reason: String = "Generic completions:\n\n",
    resultsToPost: Int = MAX_DISP_RESULTS,
    timer: TimeSource.Monotonic.ValueTimeMark = TimeSource.Monotonic.markNow(),
    shouldContinue: () -> Boolean = { currentWorkHash == workHash && timer.hasTimeLeft() },
  ) {
    val seen = HashSet<String>()
    val top = ArrayList<RankedCompletion>(resultsToPost + 1)
    val iter = iterator()
    val startTime = TimeSource.Monotonic.markNow()

    while (iter.hasNext() && shouldContinue()) {
      pause()

      val raw = iter.next()
      if (raw.isEmpty() || !seen.add(raw)) continue

      val tks = raw.tokenizeByWhitespace()
      val score = metric(tks)

      if (top.size < resultsToPost || score < top.last().score) {
        val loc = top.binarySearch { it.score.compareTo(score) }
        val idx = if (loc < 0) -loc - 1 else loc

        top.add(idx, RankedCompletion(raw, tks, score))
        if (top.size > resultsToPost) top.removeLast()
      }
    }

    val throughput = (seen.size / (startTime.elapsedNow().toDouble(DurationUnit.SECONDS) + 0.001)).round(3)

    val moreResults = (seen.size - top.size).let { if (it == 0) "\n\n" else "\n\n...$it more, " }

    val renderedItems = coroutineScope {
      top.mapIndexed { i, candidate ->
        async {
          val result =
            "<span style=\"color: gray\" class=\"noselect\">" +
                "${i.toString().padStart(2)}.) </span>" +
                render(candidate.raw, candidate.tokens)

          if (i == 0) "<mark>$result</mark>" else result
        }
      }.awaitAll()
    }

    val rendered = renderedItems.joinToString("\n")
    if (currentWorkHash == workHash) writeDisplayText("$reason$rendered".also { cache[workHash] = it })
    log("$moreResults~$throughput res/s${postCompletionSummary()}")
  }
}
