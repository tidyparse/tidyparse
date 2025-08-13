import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.tidyparse.*
import kotlinx.browser.window
import kotlinx.coroutines.*
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent
import kotlin.time.TimeSource

/** Compare with [ai.hypergraph.tidyparse.IJTidyEditor] */
open class JSTidyEditor(open val editor: dynamic, open val output: Node): TidyEditor() {

  private fun pos(line: Int, ch: Int) = js("({line: line, ch: ch})")

  private fun lineBounds(): IntRange {
    val cursor = editor.getCursor()
    val line = cursor.line as Int
    val start = editor.indexFromPos(pos(line, 0)) as Int
    val end = editor.indexFromPos(pos(line, (editor.getLine(line) as String).length)) as Int
    return start..end
  }

  private fun currentLineText(): String {
    val cursor = editor.getCursor()
    return editor.getLine(cursor.line) as String
  }

  private fun overwriteCurrentLineWith(region: IntRange, text: String) {
    val from = editor.posFromIndex(region.first)
    val to = editor.posFromIndex(region.last)
    editor.replaceRange(text, from, to)
    val newPos = editor.posFromIndex(region.first + text.length)
    editor.setCursor(newPos)
  }

  override fun continuation(f: () -> Unit): Any = window.setTimeout(f, 0)

  override fun getLineBounds(): IntRange = lineBounds()
  override fun currentLine(): Σᐩ = currentLineText()
  override fun overwriteRegion(region: IntRange, s: Σᐩ) { overwriteCurrentLineWith(region, s) }
  override fun readEditorText(): Σᐩ = editor.getValue() as String
  override fun getCaretPosition(): IntRange =
    (editor.indexFromPos(editor.getCursor("from")) as Int)..(editor.indexFromPos(editor.getCursor("to")) as Int)
  override fun setCaretPosition(range: IntRange) = editor.setSelection(editor.posFromIndex(range.first), editor.posFromIndex(range.last))
  private fun rawDisplayHTML() = (output as HTMLDivElement).innerHTML
  override fun readDisplayText(): Σᐩ = output.textContent ?: ""
  override fun writeDisplayText(s: Σᐩ) { (output as HTMLDivElement).innerHTML = s }
  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun handleInput() {
    val t0 = TimeSource.Monotonic.markNow()
    val caretInGrammar = caretInGrammar()
    val context = getApplicableContext()
    if (context.isEmpty()) return
    log("Applicable context:\n$context")
    val tokens = context.tokenizeByWhitespace()

    val cfg =
      if (caretInGrammar)
        CFGCFG(names = tokens.filter { it !in setOf("->", "|") }.toSet())
      else getLatestCFG()

    if (cfg.isEmpty()) return

    var containsUnkTok = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnkTok = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, minimize, ntStubs).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash.hashCode()
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    runningJob?.cancel()

    val scenario = when {
      tokens.size == 1 && stubMatcher.matches(tokens[0]) -> Scenario.STUB
      HOLE_MARKER in tokens -> Scenario.COMPLETION
      !containsUnkTok && tokens in cfg.language -> Scenario.PARSEABLE
      else -> Scenario.REPAIR
    }

    runningJob = MainScope().launch {
      when (scenario) {
        Scenario.STUB -> cfg.enumNTSmall(tokens[0].stripStub()).take(100)
        Scenario.COMPLETION -> cfg.enumSeqSmart(tokens)
        Scenario.PARSEABLE -> {
          val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
          writeDisplayText("$parsedPrefix$parseTree".also { cache[workHash] = it }); null
        }
        Scenario.REPAIR ->
          if (gpuAvailable)
            repairCode(cfg, tokens, if (minimize) 0 else LED_BUFFER).asSequence()
              .map { it.replace("ε", "").tokenizeByWhitespace().joinToString(" ") }
          else sampleGREUntilTimeout(tokens, cfg)
      }?.enumerateInteractively(workHash, tokens,
        reason = scenario.reason, postCompletionSummary = { ", ${t0.elapsedNow()} latency." })
    }
  }

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

  open fun formatCode(code: String): String = code

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

        overwriteRegion(
          getCaretPosition().takeIf { it.last - it.first > 0 } ?: getLineBounds(),
          formatCode(selection.tokenizeByWhitespace().joinToString(" ").replace("STRING", "\"STRING\""))
        )
        redecorateLines()
        continuation { handleInput() }
        continuation { handleTab() }

        return
      }
      SelectorAction.ARROW_DOWN -> selIdx = ModInt(currentIdx, lines.size - 4) + 1
      SelectorAction.ARROW_UP -> selIdx = ModInt(currentIdx, lines.size - 4) + -1
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
    if (caretInGrammar()) Unit

    fun decorate() {
      if (currentHash != hashIter) return
      val decCFG = getLatestCFG()
      jsEditor.apply { preparseParseableLines(decCFG, getExampleText()) }
    }

    if (!caretInGrammar()) continuation { decorate() }
    else if (currentLine().isValidProd()) window.setTimeout({ decorate() }, 100)
//    log("Redecorated in ${timer.elapsedNow()}")
  }
}