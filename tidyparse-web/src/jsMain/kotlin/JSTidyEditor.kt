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
    private fun HTMLTextAreaElement.getText(range: IntRange) = value.substring(range)

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
  fun caretInMiddle() = editor.getText(getCaretPosition().first..getLineBounds().last).trim().isNotEmpty()
  private fun rawDisplayHTML() = (outputField as HTMLDivElement).innerHTML
  override fun readDisplayText(): Σᐩ = output.textContent ?: ""
  override fun writeDisplayText(s: Σᐩ) { (outputField as HTMLDivElement).innerHTML = s }
  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  // TODO: define coalgebraically using prefix closure //prefix == tokens.dropLast(1) && tokens.last() in nextTerms
  data class SuffixCompletion(val prefix: List<Σᐩ>, val nextTerms: Set<Σᐩ>)
//  fun sampleForward(tokens: List<Σᐩ>, cfg: CFG): Sequence<Σᐩ> = cfg.enumSuffixes(tokens)
//  fun isValidContinuation(tokens: List<Σᐩ>, cfg: CFG): Boolean = tokens in cfg.admitsPrefix(tokens)
//  fun ForwardCompletion?.seed(tokens: List<Σᐩ>) =
//    if (this == null) ForwardCompletion(tokens, emptySet())
//    else if (cfg.isEmpty() || nextTerms.isEmpty()) ForwardCompletion(tokens, emptySet())
//    else TODO()
//  var forwardCompletion: ForwardCompletion? = null

  override fun handleInput() {
    val t0 = TimeSource.Monotonic.markNow()
    val caretInGrammar = caretInGrammar()
    val context = getApplicableContext()
    log("Applicable context:\n$context")

    val tokens = context.tokenizeByWhitespace()
    if (tokens.isEmpty()) return

    val cfg = if (caretInGrammar) CFGCFG(names = tokens.filter { it !in setOf("->", "|") }.toSet()) else getLatestCFG()

    if (cfg.isEmpty()) return

    var containsUnkTok = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnkTok = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, epsilons, ntStubs).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash.hashCode()
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    if (workHash in cache) return writeDisplayText(cache[workHash]!!)

    runningJob = MainScope().also { runningJob?.cancel() }.launch {
      val scenario = when {
        tokens.size == 1 && stubMatcher.matches(tokens[0]) -> Scenario.STUB
        HOLE_MARKER in tokens -> Scenario.COMPLETION
//        !containsUnkTok && forwardCompletion?.isValidContinuation(tokens) == true -> Scenario.FORWARD_COMPLETION
        // This scenario can be handled much more elegantly using coalegbra and incremental decoding
        !containsUnkTok -> handleSuffixCheck(cfg, tokens)
        else -> Scenario.REPAIR
      }

      when (scenario) {
        Scenario.STUB -> cfg.enumNTSmall(tokens[0].stripStub()).take(100)
        Scenario.COMPLETION ->
          if (!gpuAvailable) cfg.enumSeqSmart(tokens)
          else completeCode(cfg, tokens).stripEpsilon()
        Scenario.SUFFIX_COMPLETION -> handleSuffix(cfg, tokens)
        Scenario.PARSEABLE -> {
          val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
          writeDisplayText("$parsedPrefix$parseTree".also { cache[workHash] = it }); null
        }
        Scenario.REPAIR ->
          if (!gpuAvailable) sampleGREUntilTimeout(tokens, cfg)
          else repairCode(cfg, tokens, LED_BUFFER).stripEpsilon()
      }?.enumerateInteractively(workHash, tokens,
        metric = when (scenario) {
          Scenario.REPAIR -> levAndLenMetric(tokens)
          Scenario.SUFFIX_COMPLETION -> ({ it.size })
          else -> ({ 0 })
        },
        reason = scenario.reason, postCompletionSummary = { ", ${t0.elapsedNow()} latency." }
      )
    }
  }

  var suffixLenCache: List<Int> = emptyList()
  suspend fun handleSuffixCheck(cfg: CFG, tokens: List<Σᐩ>): Scenario =
    if (caretInMiddle()) { // Skip suffix completion if the caret is within line
      if (gpuAvailable) { if (checkSuffix(cfg, tokens, 0).let { it.isNotEmpty() && it[0] == 0 }) Scenario.PARSEABLE else Scenario.REPAIR }
      else if (tokens in cfg.language) Scenario.PARSEABLE else Scenario.REPAIR
    } else if (gpuAvailable) {
      val suffixLens = checkSuffix(cfg, tokens).also { suffixLenCache = it }
      println("Using GPU suffix lens: $suffixLenCache")
      if (suffixLens.isEmpty()) Scenario.REPAIR
      else if (suffixLens[0] == 0) Scenario.PARSEABLE
      else Scenario.SUFFIX_COMPLETION
    } else {
      val suffixLens = cfg.admitsPrefix(tokens).toList().also { suffixLenCache = it }
      println("Using CPU suffix lens: $suffixLenCache")
      if (suffixLens[0] == 0) Scenario.PARSEABLE
      else if (suffixLens[0] > 0) Scenario.SUFFIX_COMPLETION
      else Scenario.REPAIR
    }

  fun handleSuffix(cfg: CFG, tokens: List<Σᐩ>): Sequence<Σᐩ> =
    cfg.enumSuffixes(tokens, MAX_DISP_RESULTS * 10, suffixLenCache)

  var hashIter = 0

  class ModInt(val v: Int, val j: Int) { operator fun plus(i: Int) = ModInt(((v + i) % j + j) % j, j) }

  var selIdx: ModInt = ModInt(2, MAX_DISP_RESULTS)

  enum class SelectorAction { ENTER, ARROW_DOWN, ARROW_UP, TAB }

  fun Int.toSelectorAction(): SelectorAction? = when (this) {
    13 -> SelectorAction.ENTER
    40 -> SelectorAction.ARROW_DOWN
    38 -> SelectorAction.ARROW_UP
    9 -> SelectorAction.TAB
    else -> null
  }

  open fun formatCode(code: String): String = code

  open fun navUpdate(event: KeyboardEvent) {
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
        continuation { handleTab() }
        continuation { handleInput() }

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