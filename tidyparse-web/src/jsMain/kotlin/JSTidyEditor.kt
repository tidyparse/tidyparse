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
        val selection = readDisplayText().lines()[currentIdx + 2]
          .substringAfter(".) ").replace("\\s+".toRegex(), " ").trim()
        overwriteRegion(getCaretPosition().takeIf { it.last - it.first > 0 } ?: getLineBounds(), selection)
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
    if (caretInGrammar()) decorator.quickDecorate()

    fun decorate() {
      if (currentHash != hashIter) return
      val decCFG = getLatestCFG()
      jsEditor.apply { preparseParseableLines(decCFG, getExampleText()) }
      if (currentHash == hashIter) decorator.fullDecorate(decCFG)
    }

    if (!caretInGrammar()) continuation { decorate() }
    else if (currentLine().isValidProd()) window.setTimeout({ decorate() }, 100)
//    log("Redecorated in ${timer.elapsedNow()}")
  }
}