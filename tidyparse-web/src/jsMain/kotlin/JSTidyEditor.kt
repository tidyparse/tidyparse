import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.*
import kotlinx.browser.window
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent

/** Compare with [ai.hypergraph.tidyparse.IJTidyEditor] */
open class JSTidyEditor(open val editor: HTMLTextAreaElement, open val output: Node): TidyEditor() {
  companion object {
    private fun HTMLTextAreaElement.getEndOfLineIdx() =
      // Gets the end of the line or the end of the string, whichever comes first
      value.indexOf("\n", selectionStart!!).takeIf { it != -1 } ?: value.length
    private fun HTMLTextAreaElement.getCurrentLine() =
      value.substring(0, getEndOfLineIdx()).substringAfterLast("\n")

    fun HTMLTextAreaElement.overwriteCurrentLineWith(text: String) {
      val lineStartIdx = value.lastIndexOf('\n', selectionStart!! - 1) .takeIf { it != -1 } ?.plus(1) ?: 0
      val lineEndIdx = getEndOfLineIdx()

      value = buildString {
        append(value.substring(0, lineStartIdx))
        append(text)
        append(value.substring(lineEndIdx))
      }

      val newSelectionStart = lineStartIdx + text.length
      selectionStart = newSelectionStart
      selectionEnd = newSelectionStart
    }
  }

  override fun continuation(f: () -> Unit): Any = window.setTimeout(f, 0)

  override fun currentLine(): Σᐩ = editor.getCurrentLine()
  fun overwriteCurrentLine(s: Σᐩ) { editor.overwriteCurrentLineWith(s) }
  override fun readEditorText(): Σᐩ = editor.value
  override fun getCaretPosition(): Int = editor.selectionStart!!
  private fun rawDisplayHTML() = (outputField as HTMLDivElement).innerHTML
  override fun readDisplayText(): Σᐩ = output.textContent ?: ""
  override fun writeDisplayText(s: Σᐩ) { (outputField as HTMLDivElement).innerHTML = s }
  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  var hashIter = 0

  class ModInt(val v: Int, val j: Int) { operator fun plus(i: Int) = ModInt(((v + i) % j + j) % j, j) }

  var selIdx: ModInt = ModInt(0, toTake)

  enum class SelectorAction { ENTER, ARROW_DOWN, ARROW_UP }

  private fun Int.toSelectorAction(): SelectorAction? = when(this) {
    13 -> SelectorAction.ENTER
    40 -> SelectorAction.ARROW_DOWN
    38 -> SelectorAction.ARROW_UP
    else -> null
  }

  fun navUpdate(event: KeyboardEvent) {
    val key = event.keyCode.toSelectorAction() ?: return
    val currentText = rawDisplayHTML()
    val lines = currentText.lines()
    val htmlIndex = lines.indexOfFirst { it.startsWith("<mark>") }
    if (htmlIndex == -1) return
    event.preventDefault()
    val currentIdx = lines[htmlIndex].substringBefore(".)").substringAfterLast('>').trim().toInt()
    selIdx = ModInt(currentIdx, minOf(toTake, lines.size - 4)) +
      when (key) {
        SelectorAction.ENTER -> {
          val selection = readDisplayText().lines()[selIdx.v + 2].substringAfter(".) ")
          println("SEL: " + selection)
          overwriteCurrentLine(selection.tokenizeByWhitespace().joinToString(" "))
          redecorateLines()
          continuation { handleInput() }
          return
        }
        SelectorAction.ARROW_DOWN -> 1
        SelectorAction.ARROW_UP -> -1
      }
    writeDisplayText(lines.mapIndexed { i, line ->
      if (i == htmlIndex) line.drop(6).dropLast(7)
      else if (i == selIdx.v + 2) "<mark>$line</mark>"
      else line
    }.joinToString("\n"))
  }

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


  override fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ> = TODO()
}