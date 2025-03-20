import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.initiateSuspendableRepair
import kotlinx.coroutines.*
import org.w3c.dom.*
import kotlin.math.ln

class JSTidyPyEditor(override val editor: HTMLTextAreaElement, override val output: Node) : JSTidyEditor(editor, output) {
  val ngrams: MutableMap<List<String>, Double> = mutableMapOf<List<String>, Double>()
  val order: Int by lazy { ngrams.keys.firstOrNull()!!.size }
  val normalizingConst by lazy { ngrams.values.sum() }

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
    -(prefix + text + suffix).windowed(order, 1).sumOf { ngram -> ln((ngrams[ngram] ?: 1.0) / normalizingConst) }

  var pyodide: dynamic? = null

  fun getOutput(code: String): String {
    val types = code.replace("NUMBER", "1").replace("STRING", "\"\"")
    val pyCode = """
        import sys
        from io import StringIO
        _output = StringIO()
        sys.stdout = sys.stderr = _output
        try:
            compile(""${'"'}${types.trimIndent()}${'"'}"", 'test_compile.py', 'exec')
        except Exception:
            import traceback
            traceback.print_exc()
        _result = _output.getvalue()
    """.trimIndent()

    // Run the Python code synchronously.
    jsPyEditor.pyodide.runPython(pyCode)
    // Retrieve _result from the Pyodide globals.
    return jsPyEditor.pyodide.globals.get("_result") as String
  }

  private fun String.getErrorType(): String =
    if (isEmpty()) "" else lines().dropLast(1).lastOrNull()?.substringBeforeLast(":") ?: this

  override fun formatCode(pythonCode: String): String = try {
    jsPyEditor.pyodide.runPython("""
      from black import format_str, FileMode
      pretty_code = format_str("${pythonCode.replace("\\", "\\\\").replace("\"", "\\\"")}", mode=FileMode())
    """.trimIndent())
    jsPyEditor.pyodide.globals.get("pretty_code").trim()
  } catch (error: dynamic) {
    // If there's any issue, log the error and return the original
    println("Error formatting Python code: $error")
    pythonCode
  }

  fun String.replacePythonKeywords() =
    replace("OR", "|").replace("not_in", "not in").replace("is_not", "is not")

  override fun handleInput() {
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

    if (!containsUnk && tokens in cfg.language)
//      val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
      writeDisplayText("✅ ${tokens.dropLast(1).joinToString(" ")}".also { cache[workHash] = it })
    else /* Repair */ Unit.also {
      runningJob = MainScope().launch {
        initiateSuspendableRepair(tokens, cfg, ngrams)
          // Drop NEWLINE (added by default to PyCodeSnippets)
          .map { it.substring(0, it.length - 8).replacePythonKeywords() }
          .filter { s ->
            val errorType = getOutput(s).getErrorType()
            when (errorType) {
              "SyntaxError", "TypeError" -> false
              "" -> true
              else -> false
            }
          }.enumerateInteractively(
            workHash = workHash,
            origTks = tokens.dropLast(1),
            recognizer = { "$it NEWLINE".replace("|", "OR") in cfg.language },
//            metric = { (score(it) * 1_000.0).toInt() }, // TODO: Is reordering really necessary if we are decoding GREs by ngram score?
            metric = { (levenshtein(tokens.dropLast(1), it) * 10000 + score(it) * 1_000.0).toInt() },
            customDiff = {
              val levAlign = levenshteinAlign(tokens.dropLast(1), it.tokenizeByWhitespace())
              pcs.paintDiff(levAlign)
            }
          )
      }
    }
  }
}