import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.*
import kotlinx.browser.window
import org.w3c.dom.*
import kotlin.time.TimeSource

/** Compare with [ai.hypergraph.tidyparse.IJTidyEditor] */
class JSTidyEditor(val editor: HTMLTextAreaElement, val output: Node): TidyEditor() {
  override fun readDisplayText(): Σᐩ = output.textContent ?: ""

  override fun readEditorText(): Σᐩ = editor.value

  override fun getCaretPosition(): Int = editor.selectionStart!!

  companion object {
    private fun HTMLTextAreaElement.getEndOfLineIdx() = value.indexOf("\n", selectionStart!!)
    private fun HTMLTextAreaElement.getCurrentLine() = value.substring(0, getEndOfLineIdx()).split("\n").last()
    fun String.diff(other: String): String = other

    //fun String.diff(other: String): String {
    //  val output = tokenizeByWhitespace().toMutableList()
    //  differenceOf(output, other.tokenizeByWhitespace())
    //    .applyDiff(
    //      remove = { index -> output.removeAt(index) },
    //      insert = { item, index -> output.add(index, "<span style=\"background-color: green;\">${item.escapeHTML()}</span>") },
    //      move = { old, new ->  }
    //    )
    //  return output.joinToString(" ") { if (it.startsWith("<span style=")) it else it.escapeHTML() }
    //}
  }

  override fun continuation(f: () -> Unit): Any = window.setTimeout(f, 0)

  override fun currentLine(): Σᐩ = editor.getCurrentLine()

  override fun writeDisplayText(s: Σᐩ) {
    outputField.textContent = s
//    (outputField as HTMLTextAreaElement).outerHTML.also { println(it) }
  }

  var hashIter = 0

  override fun redecorateLines(cfg: CFG) {
    val currentHash = ++hashIter
//    val timer = TimeSource.Monotonic.markNow()
    if (caretInGrammar()) decorator.quickDecorate()
    val decCFG = getLatestCFG()

    fun decorate() {
      if (currentHash != hashIter) return
      jsEditor.apply { preparseParseableLines(decCFG, getExampleText()) }
      if (currentHash == hashIter) decorator.fullDecorate(decCFG)
    }

    if (!caretInGrammar()) continuation { decorate() }
    else if (currentLine().isValidProd()) window.setTimeout({ decorate() }, 100)
//    println("Redecorated in ${timer.elapsedNow()}")
  }

  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ> = TODO()
}