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
    private fun HTMLTextAreaElement.getLineStartIdx() =
      value.lastIndexOf('\n', selectionStart!! - 1).takeIf { it != -1 } ?.plus(1) ?: 0
    private fun HTMLTextAreaElement.lineBounds() = getLineStartIdx()..getEndOfLineIdx()
    private fun HTMLTextAreaElement.getCurrentLine() =
      value.substring(0, getEndOfLineIdx()).substringAfterLast("\n")

    fun HTMLTextAreaElement.overwriteCurrentLineWith(region: IntRange, text: String) {
      value = buildString {
        append(value.substring(0, region.first))
        append(text)
        append(value.substring(region.last))
      }

      val newSelectionStart = region.first + text.length
      selectionStart = newSelectionStart
      selectionEnd = newSelectionStart
    }
  }

  override fun continuation(f: () -> Unit): Any = window.setTimeout(f, 0)

  override fun getLineBounds(): IntRange = editor.lineBounds()
  override fun currentLine(): Σᐩ = editor.getCurrentLine()
  override fun overwriteRegion(region: IntRange, s: Σᐩ) { editor.overwriteCurrentLineWith(region, s) }
  override fun readEditorText(): Σᐩ = editor.value
  override fun getCaretPosition(): IntRange = editor.selectionStart!!..editor.selectionEnd!!
  override fun setCaretPosition(range: IntRange) = editor.setSelectionRange(range.first, range.last)
  private fun rawDisplayHTML() = (outputField as HTMLDivElement).innerHTML
  override fun readDisplayText(): Σᐩ = output.textContent ?: ""
  override fun writeDisplayText(s: Σᐩ) { (outputField as HTMLDivElement).innerHTML = s }
  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  var hashIter = 0

  class ModInt(val v: Int, val j: Int) { operator fun plus(i: Int) = ModInt(((v + i) % j + j) % j, j) }

  var selIdx: ModInt = ModInt(2, MAX_DISP_RESULTS)

  enum class SelectorAction { ENTER, ARROW_DOWN, ARROW_UP, TAB }

  private fun Int.toSelectorAction(): SelectorAction? = when (this) {
    13 -> SelectorAction.ENTER
    40 -> SelectorAction.ARROW_DOWN
    38 -> SelectorAction.ARROW_UP
    9 -> SelectorAction.TAB
    else -> null
  }

  fun navUpdate(event: KeyboardEvent) {
    val key = event.keyCode.toSelectorAction() ?: return
    if (key == SelectorAction.TAB) { event.preventDefault(); handleTab(); return }
    val currentText = rawDisplayHTML()
    val lines = currentText.lines()
    val htmlIndex = lines.indexOfFirst { it.startsWith("<mark>") }
    if (htmlIndex == -1) return
    event.preventDefault()
    val currentIdx = lines[htmlIndex].substringBefore(".)").substringAfterLast('>').trim().toInt()
    when (key) {
      SelectorAction.ENTER -> {
        val selection = readDisplayText().lines()[currentIdx + 2].substringAfter(".) ")
        overwriteRegion(getCaretPosition().takeIf { it.last - it.first > 0 } ?: getLineBounds(), selection.tokenizeByWhitespace().joinToString(" "))
        redecorateLines()
        continuation { handleInput() }
        continuation { handleTab() }
        return
      }
      SelectorAction.ARROW_DOWN -> selIdx = ModInt(currentIdx, minOf(MAX_DISP_RESULTS, lines.size - 4)) + 1
      SelectorAction.ARROW_UP -> selIdx = ModInt(currentIdx, minOf(MAX_DISP_RESULTS, lines.size - 4)) + -1
      SelectorAction.TAB -> {}
    }
    writeDisplayText(lines.mapIndexed { i, line ->
      if (i == htmlIndex) line.substring(6, line.length - 7)
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
}