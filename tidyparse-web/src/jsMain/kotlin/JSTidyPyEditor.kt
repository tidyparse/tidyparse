import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.*
import kotlinx.coroutines.*
import org.w3c.dom.*
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

  override fun redecorateLines(cfg: CFG) {
    val currentHash = ++hashIter

    fun decorate() {
      if (currentHash != hashIter) return
      val decCFG = getLatestCFG()
      preparseParseableLines(decCFG, readEditorText()) {
        PyCodeSnippet(it).lexedTokens().replace("|", "OR") in decCFG.language
      }
      if (currentHash == hashIter) decorator.fullDecorate(decCFG)
    }

    continuation { decorate() }
//    println("Redecorated in ${timer.elapsedNow()}")
  }

  companion object {
    val prefix = listOf("BOS", "NEWLINE")
    val suffix = listOf("NEWLINE", "EOS")
  }

  fun score(text: List<String>): Double =
    -(prefix + text + suffix).windowed(order, 1)
      .sumOf { ngram -> ln((ngrams[ngram] ?: 1.0) / normalizingConst) }

  var pyodide: dynamic? = null

  fun getOutput(code: String): String = try {
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
  } catch (e: dynamic) { "Error during compilation: $e".also { println(it) } }

  private fun String.getErrorType(): String =
    if (isEmpty()) "" else lines().dropLast(1).lastOrNull()?.substringBeforeLast(":") ?: this

  private fun String.getErrorMessage(): String = substringAfter(": ")

  override fun formatCode(code: String): String = try {
    jsPyEditor.pyodide.runPython("""
      from black import format_str, FileMode
      pretty_code = format_str("${code.replace("\\", "\\\\").replace("\"", "\\\"")}", mode=FileMode())
    """.trimIndent())
    jsPyEditor.pyodide.globals.get("pretty_code").trim().replace("\n", " ")
  } catch (error: dynamic) {
    // If there's any issue, log the error and return the original
    println("Error formatting Python code: $error")
    code
  }

  fun String.replacePythonKeywords() =
    replace("OR", "|").replace("not_in", "not in").replace("is_not", "is not")

  override fun handleInput() {
    val t0 = TimeSource.Monotonic.markNow()
    val currentLine = currentLine().also { println("Current line is: $it") }
    if (currentLine.isBlank()) return
    val pcs = PyCodeSnippet(currentLine)
    val tokens = pcs.lexedTokens().tokenizeByWhitespace().map { if (it == "|") "OR" else it }

    println("Repairing: " + tokens.dropLast(1).joinToString(" "))

    var containsUnk = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnk = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, minimize).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    runningJob?.cancel()

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
          println("Repairing on GPU...")
          repairCode(cfg, tokens, LED_BUFFER, ngramTensor).asSequence()
        } else {
          println("Repairing on CPU...")
          metric = { (levenshtein(tokens.dropLast(1), it) * 10_000 + score(it) * 1_000.0).toInt() }
          sampleGREUntilTimeout(tokens, cfg)
        })
          // Drop NEWLINE (added by default to PyCodeSnippets)
          .map { it.dropLast(8).replacePythonKeywords() }
          .distinct().let {
            if (allowCompilerErrors) it.onEach { total++ }
            else it.filter { s ->
              val errorType = getOutput(s).getErrorType()
              when (errorType) {
                "SyntaxError", "TypeError" -> false
                "" -> true
                else -> false
              }.also { if (!it) rejected++; total++ }
            }
          }.enumerateInteractively(
            workHash = workHash,
            origTks = tokens.dropLast(1),
            recognizer = { "$it NEWLINE".replace("|", "OR") in cfg.language },
            metric = metric,
            customDiff = {
              val levAlign = levenshteinAlign(tokens.dropLast(1), it.tokenizeByWhitespace())
              pcs.paintDiff(levAlign)
            },
            postCompletionSummary = { ", discarded $rejected/$total, ${t0.elapsedNow()} latency." },
            reason = invalidPrefix
          )
      }
    }
  }
}