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
    private fun HTMLTextAreaElement.getEndOfLineIdx() =
      // Gets the end of the line or the end of the string, whichever comes first
      value.indexOf("\n", selectionStart!!).takeIf { it != -1 } ?: value.length
    private fun HTMLTextAreaElement.getCurrentLine() =
      value.substring(0, getEndOfLineIdx()).substringAfterLast("\n")
  }

  override fun continuation(f: () -> Unit): Any = window.setTimeout(f, 0)

  override fun currentLine(): Σᐩ = editor.getCurrentLine()

  override fun writeDisplayText(s: Σᐩ) { (outputField as HTMLDivElement).innerHTML = s }

  var hashIter = 0

  override fun redecorateLines(cfg: CFG) {
    val currentHash = ++hashIter
//    val timer = TimeSource.Monotonic.markNow()
    if (caretInGrammar()) decorator.quickDecorate()

    fun decorate() {
      if (currentHash != hashIter) return
      val decCFG = getLatestCFG()
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