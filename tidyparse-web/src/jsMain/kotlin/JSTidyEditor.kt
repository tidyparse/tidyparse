import JSTidyEditor.SelectorAction.*
import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.tidyparse.*
import ai.hypergraph.tidyparse.TidyEditor.Scenario.*
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

  protected val codeMirror: dynamic
    get() = window.asDynamic().cmEditor

  protected fun hasCodeMirror(): Boolean = codeMirror != null && codeMirror != js("undefined")

  private fun cmPos(index: Int): dynamic = codeMirror.posFromIndex(index)

  private fun cmIndex(which: String): Int = codeMirror.indexFromPos(codeMirror.getCursor(which)) as Int

  override fun continuation(f: () -> Unit): Any = window.setTimeout(f, 0)

  val instructions = (outputField as HTMLDivElement).innerHTML
  val alphabetHist: MutableMap<String, Int> =
    readEditorText().tokenizeByWhitespace().groupBy { it }.mapValues { it.value.size }.toMutableMap()

  override fun getLineBounds(): IntRange =
    if (hasCodeMirror()) {
      val value = readEditorText()
      val start = getCaretPosition().first
      val lineStart = value.lastIndexOf('\n', start - 1).takeIf { it != -1 }?.plus(1) ?: 0
      val lineEnd = value.indexOf("\n", start).takeIf { it != -1 } ?: value.length
      lineStart..lineEnd
    } else editor.lineBounds()

  override fun currentLine(): Σᐩ =
    if (hasCodeMirror()) {
      val cursor = codeMirror.getCursor()
      codeMirror.getLine(cursor.line) as String
    } else editor.getCurrentLine()

  override fun overwriteRegion(region: IntRange, s: Σᐩ) {
    if (hasCodeMirror()) {
      codeMirror.replaceRange(s, cmPos(region.first), cmPos(region.last))
      setCaretPosition((region.first + s.length).let { it..it })
      codeMirror.save()
    } else editor.overwriteCurrentLineWith(region, s)
  }

  override fun readEditorText(): Σᐩ =
    if (hasCodeMirror()) codeMirror.getValue() as String else editor.value

  override fun getCaretPosition(): IntRange =
    if (hasCodeMirror()) cmIndex("from")..cmIndex("to")
    else editor.selectionStart!!..editor.selectionEnd!!

  override fun setCaretPosition(range: IntRange) {
    if (hasCodeMirror()) codeMirror.setSelection(cmPos(range.first), cmPos(range.last))
    else editor.setSelectionRange(range.first, range.last)
  }

  fun caretInMiddle() = readEditorText().substring(getCaretPosition().first, getLineBounds().last).trim().isNotEmpty()
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

  fun restoreInstructions() = writeDisplayText(instructions)

  override fun handleInput() {
    val t0 = TimeSource.Monotonic.markNow()
    val caretInGrammar = caretInGrammar()
    val context = getApplicableContext()
    log("Applicable context:\n$context")
    val suffixEligible = context.endsWith(" ") && !caretInMiddle() && !caretInGrammar

    var tokens = context.tokenizeByWhitespace()
    if (tokens.isEmpty()) { restoreInstructions(); return }

    val cfg = if (caretInGrammar) {
      tokens = tokens.map { if (it == "START") "[START]" else it }
      CFGCFG(names = tokens.filter { it !in setOf("->", "|") }.toSet() + "[START]")
    } else getLatestCFG()

    if (cfg.isEmpty()) return

    var containsUnkTok = false
    val abstractUnk = tokens.map { if (it in cfg.terminals) it else { containsUnkTok = true; "_" } }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, epsilons, ntStubs).hashCode()
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash.hashCode()
    if (workHash == currentWorkHash && !suffixEligible) return
    currentWorkHash = workHash

    val cached = cache[workHash]
    if (cached != null && (!suffixEligible || cached.startsWith("->"))) return writeDisplayText(cache[workHash]!!)

    runningJob = MainScope().also { runningJob?.cancel() }.launch {
      val scenario = when {
        tokens.size == 1 && stubMatcher.matches(tokens[0]) -> STUB
        HOLE_MARKER in tokens -> COMPLETION
//        !containsUnkTok && forwardCompletion?.isValidContinuation(tokens) == true -> FORWARD_COMPLETION
        // This scenario can be handled much more elegantly using coalegbra and incremental decoding
        tokens in cfg.language && !suffixEligible -> PARSEABLE
        !containsUnkTok -> handleSuffixCheck(cfg.language, tokens)
        else -> REPAIR
      }

      var postProcTimer = TimeSource.Monotonic.markNow()
      when (scenario) {
        STUB -> cfg.enumNTSmall(tokens[0].stripStub()).take(100)
        COMPLETION -> if (!gpuAvailable) cfg.enumSeqSmart(tokens) else completeCode(cfg, tokens).stripEpsilon()
        SUFFIX_COMPLETION -> cfg.enumSuffixes(tokens, scenario.data).distinct()
        PARSEABLE -> {
          val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
          writeDisplayText("$parsedPrefix$parseTree".also { cache[workHash] = it }); null
        }
        REPAIR ->
          if (!gpuAvailable) { log("Repairing on CPU..."); sampleGREUntilTimeout(tokens, cfg) }
          else repairCode(cfg, tokens, LED_BUFFER).stripEpsilon()
      }?.let { if (scenario != REPAIR) it.take(MAX_DISP_RESULTS) else it }
      ?.let { if (caretInGrammar) it.map { it.replace("[START]", "START") } else it }
      ?.enumerateInteractively(workHash, tokens,
        metric = when (scenario) {
          REPAIR -> levAndLenMetric(tokens)
          SUFFIX_COMPLETION -> ({ it.size })
          else -> ({ 0 })
        },
        reason = scenario.reason, postCompletionSummary = {
          if (gpuAvailable) {
            mark("postprocessing", postProcTimer);
            timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
            log("Results rendered in ${timings["total"]}ms")
            timings.logTimesheet()
          }
          ", ${t0.elapsedNow()} latency."
        }.also { postProcTimer = TimeSource.Monotonic.markNow() }
      )
    }
  }

  suspend fun handleSuffixCheck(cfl: CFL, tokens: List<Σᐩ>): Scenario =
    if (caretInMiddle()) { // Skip suffix completion if the caret is within line
      if (gpuAvailable) { if (cfl.cfg.checkSuffix(tokens, 0).let { it.isNotEmpty() && it[0] == 0 }) PARSEABLE else REPAIR }
      else if (tokens in cfl) PARSEABLE else REPAIR
    } else if (gpuAvailable) {
      val suffixLens = cfl.cfg.checkSuffix(tokens)
      println("Read GPU suffix lens: $suffixLens")
      if (suffixLens.isEmpty()) REPAIR
      else SUFFIX_COMPLETION(suffixLens)
    } else {
      val suffixLens = cfl.admitsPrefix(tokens).toList()
      println("Read CPU suffix lens: $suffixLens")
      if (suffixLens[0] > 0) SUFFIX_COMPLETION(suffixLens)
      else REPAIR
    }

  var hashIter = 0

  class ModInt(val v: Int, val j: Int) { operator fun plus(i: Int) = ModInt(((v + i) % j + j) % j, j) }

  var selIdx: ModInt = ModInt(2, MAX_DISP_RESULTS)

  enum class SelectorAction { ENTER, ARROW_DOWN, ARROW_UP, ARROW_RIGHT, TAB, ESCAPE }

  fun Int.toSelectorAction(): SelectorAction? = when (this) {
    13 -> ENTER
    40 -> ARROW_DOWN
    38 -> ARROW_UP
    39 -> ARROW_RIGHT
//    32 -> SPACE
    9 -> TAB
    27 -> ESCAPE
    else -> null
  }

  open fun formatCode(code: String): String = code

  open fun navUpdate(event: KeyboardEvent) {
    val key = event.keyCode.toSelectorAction() ?: return
    if (key == ENTER && event.shiftKey) return
    if (key == TAB) { event.preventDefault(); handleTab(); return }
    if (key == ARROW_RIGHT) {
      val dispText = readDisplayText()
      val mode = dispText.substringBefore("\n")
      if (!mode.startsWith("-> Forward completion")) return
    }
    if (key == ESCAPE) { restoreInstructions(); return }
    val currentText = rawDisplayHTML()
    val lines = currentText.lines()
    val htmlIndex = lines.indexOfFirst { it.startsWith("<mark>") }
    if (htmlIndex == -1) return
    event.preventDefault()
    val currentIdx = lines[htmlIndex].substringBefore(".)").substringAfterLast('>').trim().toInt()
    when (key) {
      ENTER -> {
        val selection = readDisplayText().lines()[currentIdx + 2]
          .substringAfter(".) ").replace("\\s+".toRegex(), " ").trim()
        log("Selected: $selection / ${selection in cfg.language}")
        overwriteRegion(getCaretPosition().takeIf { it.last - it.first > 0 } ?: getLineBounds(), selection)
        redecorateLines()
        continuation { handleTab() }
        continuation { handleInput() }

        return
      }
      ARROW_DOWN -> selIdx = ModInt(currentIdx, lines.size - 4) + 1
      ARROW_UP -> selIdx = ModInt(currentIdx, lines.size - 4) + -1
      ARROW_RIGHT -> {
        val selection = readDisplayText().lines()[currentIdx + 2]
          .substringAfter(".) ").replace("\\s+".toRegex(), " ").trim()

        val toksToTake = currentLine().tokenizeByWhitespace().size + 1
        val continuation = selection.tokenizeByWhitespace().take(toksToTake).joinToString(" ")
        overwriteRegion(getCaretPosition().takeIf { it.last - it.first > 0 } ?: getLineBounds(), continuation)
        redecorateLines()
        continuation { handleTab() }
        continuation { handleInput() }

        return
      }
      TAB -> {}
      ESCAPE -> {}
    }
    writeDisplayText(lines.mapIndexed { i, line -> when (i) {
        htmlIndex -> line.substring(6, line.length - 7)
        selIdx.v + 2 -> "<mark>$line</mark>"
        else -> line
    } }.joinToString("\n"))
  }

  override fun redecorateLines(cfg: CFG) {
    val currentHash = ++hashIter
//    val timer = TimeSource.Monotonic.markNow()
    if (caretInGrammar()) decorator.quickDecorate()

    fun decorate() {
      if (currentHash != hashIter) return
      val decCFG = getLatestCFG()
      jsEditor.apply { preparseParseableLines(decCFG, getExampleText())  }
      if (currentHash == hashIter) decorator.fullDecorate(decCFG)
    }

    if (!caretInGrammar()) continuation { decorate() }
    else if (currentLine().isValidProd()) window.setTimeout({ decorate() }, 100)
//    log("Redecorated in ${timer.elapsedNow()}")
  }
}
