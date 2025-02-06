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
    private fun HTMLTextAreaElement.lineIdx() =
      value.substring(0, selectionStart!!).lastIndexOf("\n")

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
  override fun setCaretPosition(range: IntRange) = editor.setSelectionRange(range.first, range.last)
  private fun rawDisplayHTML() = (outputField as HTMLDivElement).innerHTML
  override fun readDisplayText(): Σᐩ = output.textContent ?: ""
  override fun writeDisplayText(s: Σᐩ) { (outputField as HTMLDivElement).innerHTML = s }
  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  var hashIter = 0

  class ModInt(val v: Int, val j: Int) { operator fun plus(i: Int) = ModInt(((v + i) % j + j) % j, j) }

  var selIdx: ModInt = ModInt(2, toTake)

  enum class SelectorAction { ENTER, ARROW_DOWN, ARROW_UP, TAB }

  private fun Int.toSelectorAction(): SelectorAction? = when(this) {
    13 -> SelectorAction.ENTER
    40 -> SelectorAction.ARROW_DOWN
    38 -> SelectorAction.ARROW_UP
    9 -> SelectorAction.TAB
    else -> null
  }

  fun handleTab() {
    val PLACEHOLDERS = listOf("STRING", "NAME", "NUMBER")
    val lineIdx = editor.lineIdx() + 1
    val line = currentLine()
    val regex = Regex(PLACEHOLDERS.joinToString("|") { Regex.escape(it) } + "|<\\S+>")
    var firstPlaceholder = regex.find(line, (getCaretPosition() + 1 - lineIdx).coerceAtMost(line.length))
    if (firstPlaceholder == null) firstPlaceholder = regex.find(line, 0)
    if (firstPlaceholder == null) return

    setCaretPosition((lineIdx + firstPlaceholder.range.first)..(lineIdx + firstPlaceholder.range.last + 1))
  }

  fun navUpdate(event: KeyboardEvent) {
    val key = event.keyCode.toSelectorAction() ?: return
    event.preventDefault()
    if (key == SelectorAction.TAB) { handleTab(); return }
    val currentText = rawDisplayHTML()
    val lines = currentText.lines()
    val htmlIndex = lines.indexOfFirst { it.startsWith("<mark>") }
    if (htmlIndex == -1) return
    val currentIdx = lines[htmlIndex].substringBefore(".)").substringAfterLast('>').trim().toInt()
    when (key) {
      SelectorAction.ENTER -> {
        val selection = readDisplayText().lines()[currentIdx + 2].substringAfter(".) ")
        overwriteCurrentLine(selection.tokenizeByWhitespace().joinToString(" "))
        redecorateLines()
        continuation { handleInput() }
        continuation { handleTab() }
        return
      }
      SelectorAction.ARROW_DOWN -> selIdx = ModInt(currentIdx, minOf(toTake, lines.size - 4)) + 1
      SelectorAction.ARROW_UP -> selIdx = ModInt(currentIdx, minOf(toTake, lines.size - 4)) + -1
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


  override fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ> = TODO()
}