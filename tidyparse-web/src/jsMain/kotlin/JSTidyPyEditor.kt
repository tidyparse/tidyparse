import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.initiateSuspendableRepair
import kotlinx.coroutines.*
import org.w3c.dom.*
import kotlin.math.ln

class JSTidyPyEditor(override val editor: HTMLTextAreaElement, override val output: Node) : JSTidyEditor(editor, output) {
  val ngrams: MutableMap<List<String>, Double> = mutableMapOf<List<String>, Double>()
  val order: Int by lazy { ngrams.keys.firstOrNull()!!.size }
  val normalizingConst by lazy { ngrams.keys.map { it.last() }.distinct().size }

  override var cfg = pythonStatementCNFAllProds

  override fun getLatestCFG(): CFG = cfg

  override fun redecorateLines(cfg: CFG) {
    val currentHash = ++hashIter

    fun decorate() {
      if (currentHash != hashIter) return
      val decCFG = getLatestCFG()
      preparseParseableLines(pythonStatementCNFAllProds, readEditorText()) {
        PyCodeSnippet(it).lexedTokens() in pythonStatementCNFAllProds.language
      }
      if (currentHash == hashIter) decorator.fullDecorate(decCFG)
    }

    continuation { decorate() }
//    println("Redecorated in ${timer.elapsedNow()}")
  }

  fun score(text: List<String>): Double = if (text.size < order) 0.0
    else -(listOf("BOS") + text + listOf("EOS")).windowed(order, 1)
      .sumOf { ngram -> ln((ngrams[ngram] ?: 1.0) / normalizingConst) }

  override fun handleInput() {
    val currentLine = currentLine().also { println("Current line is: $it") }
    if (currentLine.isBlank()) return
    val pcs = PyCodeSnippet(currentLine)
    val tokens = pcs.lexedTokens().tokenizeByWhitespace()

    println("Repairing: " + tokens.dropLast(1).joinToString(" "))

    var containsUnk = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnk = true; "_" } }

    val workHash = abstractUnk.hashCode() + cfg.hashCode()
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    runningJob?.cancel()

    if (!containsUnk && tokens in cfg.language) {
//      val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
      writeDisplayText("✅ ${tokens.joinToString(" ")}".also { cache[workHash] = it })
    } else /* Repair */ Unit.also {
      runningJob = MainScope().launch {
        initiateSuspendableRepair(tokens, cfg)
          // Drop NEWLINE (added by default to PyCodeSnippets)
          .map { it.substring(0, it.length - 8).replace("OR", "|") }
          .enumerateInteractively(
            workHash,
            tokens.dropLast(1),
            metric = { levenshtein(tokens, it) * 7919 + (score(it) * 1_000.0).toInt() },
            customDiff = {
              val levAlign = levenshteinAlign(tokens.dropLast(1), it.tokenizeByWhitespace())
              pcs.paintDiff(levAlign)
            }
          )
      }
    }
  }
}