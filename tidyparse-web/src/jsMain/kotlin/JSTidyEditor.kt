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
import kotlin.math.absoluteValue
import kotlin.time.Duration
import kotlin.time.TimeMark
import kotlin.time.TimeSource

private const val COMPLETION_DATA_ATTRIBUTE = "data-completion"

private data class SuffixCpuTiming(val step: String, val elapsed: Duration)

private fun <T> MutableList<SuffixCpuTiming>.measure(step: String, block: () -> T): T {
  val started = TimeSource.Monotonic.markNow()
  return block().also { add(SuffixCpuTiming(step, started.elapsedNow())) }
}

private fun List<SuffixCpuTiming>.logSuffixTimings(stage: String) {
  if (isEmpty()) return
  log(joinToString(
    separator = "\n",
    prefix = "Suffix CPU preprocessing ($stage):\n"
  ) { "  ${it.step}: ${it.elapsed}" })
}

private fun encodeCompletion(value: String): String =
  js("encodeURIComponent(value)") as String

private fun decodeCompletion(value: String): String =
  try { js("decodeURIComponent(value)") as String } catch (_: dynamic) { value }

private fun List<List<String>>.commonTokenPrefix(): List<String> =
  firstOrNull()?.let { first ->
    first.take(first.indices.firstOrNull { i -> any { it.getOrNull(i) != first[i] } } ?: first.size)
  }.orEmpty()

private fun abbreviatedSuffixHTML(
  original: String,
  commonPrefix: List<String>,
  completion: String
): String {
  val aligned = levenshteinAlign(original, completion).filter { it.second != null }
  if (aligned.isEmpty()) return ""
  val prefixSize = commonPrefix.size.takeIf {
    completion.tokenizeByWhitespace().take(it) == commonPrefix
  } ?: 0
  val prefix = when (prefixSize) {
    0 -> ""
    in 1..3 -> aligned.take(prefixSize).paintDiffs()
    else -> "${listOf(aligned.first()).paintDiffs()} ... ${aligned.take(prefixSize).takeLast(2).paintDiffs()}"
  }
  val suffix = aligned.drop(prefixSize).paintDiffs()
  val label = listOf(prefix, suffix).filter(String::isNotEmpty).joinToString(" ")
  return """<span $COMPLETION_DATA_ATTRIBUTE="${encodeCompletion(completion)}">$label</span>"""
}

/** Compare with [ai.hypergraph.tidyparse.IJTidyEditor] */
open class JSTidyEditor(val editor: HTMLTextAreaElement, val output: Node): TidyEditor() {
  private val handleInputDebouncer = BackspaceDebouncer
  protected var inputWorkInvalidated = false

  private val softCompletionManager = SoftCompletionManager(
    editor = editor,
    codeMirror = { codeMirror },
    hasCodeMirror = { hasCodeMirror() },
    codeMirrorPosition = { cmPos(it) },
    readEditorText = { readEditorText() },
    getCaretPosition = { getCaretPosition() },
    setCaretPosition = { setCaretPosition(it) },
    getLatestCFG = { getLatestCFG() },
    completionContextHash = { cfg -> listOf(cfg.hashCode(), epsilons, ntStubs).hashCode() },
    previewRenderer = { insertion, offset -> renderSoftTerminalInsertionPreview(insertion, offset) },
    afterCommit = { redecorateLines(); continuation { handleInput() } }
  )

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

  protected open fun renderSoftTerminalInsertionPreview(insertion: Σᐩ, offset: Int): Boolean =
    softCompletionManager.renderPreview(insertion, offset)

  internal val pendingTerminalCompletionInsertion: Σᐩ? get() = softCompletionManager.pendingInsertion

  internal fun discardSoftTerminalCompletion() = softCompletionManager.reset()

  internal fun commitSoftTerminalInsertion(): Boolean = softCompletionManager.commit()

  internal fun recordFreshUserInsertion() = softCompletionManager.recordFreshUserInsertion()

  override fun readEditorText(): Σᐩ = if (hasCodeMirror()) codeMirror.getValue() as String else editor.value

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
//  fun ForwardCompletion?.seed(tokens: List<Σᐩ>) =
//    if (this == null) ForwardCompletion(tokens, emptySet())
//    else if (cfg.isEmpty() || nextTerms.isEmpty()) ForwardCompletion(tokens, emptySet())
//    else TODO()
//  var forwardCompletion: ForwardCompletion? = null

  fun restoreInstructions() = writeDisplayText(instructions)

  protected open fun caretIsInGrammar(): Boolean = caretInGrammar()

  protected open suspend fun holeCompletionCandidates(cfg: CFG, tokens: List<Σᐩ>): Sequence<Σᐩ> =
    if (!gpuAvailable) cfg.enumSeqSmart(tokens) else completeCode(cfg, tokens).asSequence()

  internal open suspend fun suffixCompletionCandidates(
    cfg: CFG,
    tokens: List<Σᐩ>,
    terminalCompletion: TerminalCompletionPlan?,
    limit: Int,
    requestStarted: TimeMark,
    recordGpuTimings: (Map<String, Int>) -> Unit = {}
  ): Sequence<Σᐩ> {
    val cpuTimings = mutableListOf<SuffixCpuTiming>()
    val completionCFG = cpuTimings.measure("prepare visible completion grammar") {
      cfg.visibleCompletionCFG
    }
    fun cpuCandidates(): Sequence<Σᐩ> =
      terminalCompletion?.enumerationBranches(limit)
        ?.map { completionCFG.enumTerminalSuffixes(it, limit) }
        ?.let(::fairMerge)?.distinct()
        ?: completionCFG.enumDiverseSuffixes(tokens, limit)

    return if (!gpuAvailable || START_SYMBOL !in completionCFG.nonterminals) cpuCandidates()
    else try {
      val batch = cpuTimings.measure("$GPU_SUFFIX_LENGTHS_PER_TERMINAL-slice next-token frontier") {
        completionCFG.gpuSuffixBatch(tokens, terminalCompletion, limit)
      } ?: return cpuCandidates().also {
        cpuTimings.logSuffixTimings("batch planning; GPU word limit exceeded")
      }
      cpuTimings.logSuffixTimings("batch planning")
      log(
        "Suffix GPU template: ${batch.slices.size} slices " +
          "(up to $GPU_SUFFIX_LENGTHS_PER_TERMINAL per next token), " +
          "horizon=${batch.slices.maxOfOrNull { it.length } ?: 0}, " +
          "complete=${batch.completeWords.size}"
      )
      var gpuTimings: Map<String, Int>? = null
      completionCFG.gpuDiverseSuffixes(batch, limit, requestStarted) { gpuTimings = it }
        ?.also { gpuTimings?.let(recordGpuTimings) }
        ?.asSequence()
        ?: cpuCandidates().also { gpuTimings?.let(recordGpuTimings) }
    } catch (cancelled: CancellationException) {
      throw cancelled
    } catch (failure: Throwable) {
      log("GPU suffix completion failed; falling back to CPU: ${failure.message}")
      cpuCandidates()
    }
  }

  protected open fun currentDisplayResultLimit(): Int =
    (output as? HTMLElement)?.let(::displayResultLimit) ?: MAX_DISP_RESULTS

  final override fun handleInput() {
    val started = TimeSource.Monotonic.markNow()
    if (handleInputDebouncer.submit(this) { handleInputNow(started) }) {
      runningJob?.cancel()
      runningJob = null
      inputWorkInvalidated = true
    }
  }

  protected open fun handleInputNow(started: TimeMark) {
    val cpuTimings = mutableListOf<SuffixCpuTiming>()
    val contextStarted = TimeSource.Monotonic.markNow()
    softCompletionManager.invalidateIfChanged()
    val t0 = started
    val freshUserInsertion = !softCompletionManager.compositionActive() && softCompletionManager.consumeFreshUserInsertion()
    val caretInGrammar = caretIsInGrammar()
    val context = getApplicableContext()
    val resultLimit = currentDisplayResultLimit()
    val suffixEligible = context.endsWith(" ") && !caretInMiddle() && !caretInGrammar

    val displayTokens = context.tokenizeByWhitespace()
    if (displayTokens.isEmpty()) { restoreInstructions(); return }
    var tokens = displayTokens
    cpuTimings += SuffixCpuTiming("editor state, context, and tokenization", contextStarted.elapsedNow())
    val contextLogStarted = TimeSource.Monotonic.markNow()
    log("Applicable context:\n$context")
    cpuTimings += SuffixCpuTiming("applicable-context console log", contextLogStarted.elapsedNow())

    val cfg = cpuTimings.measure("grammar lookup / normalization") {
      if (caretInGrammar) {
        tokens = tokens.map { if (it == "START") "[START]" else it }
        CFGCFG(names = tokens.filter { it !in setOf("->", "|") }.toSet() + "[START]")
      } else getLatestCFG()
    }

    if (cfg.isEmpty()) {
      softCompletionManager.clear()
      return
    }

    val exactStubContext = tokens.size == 1 && stubMatcher.matches(tokens.single())

    val caretAtLastTokenEnd =
      getCaretPosition().let { caret ->
        caret.first == caret.last &&
          caret.first == getLineBounds().first +
            context.indexOfLast { !it.isWhitespace() } + 1
      }

    val activeTerminalPlan = softCompletionManager.activePlan(cfg)
    val hasActiveSoftInsertion = softCompletionManager.hasActiveInsertion(cfg)

    val terminalResolutionEligible =
      !caretInGrammar &&
        !exactStubContext &&
        !caretInMiddle() &&
        !softCompletionManager.compositionActive() &&
        !softCompletionManager.undoOrRedoInProgress() &&
        getCaretPosition().let { it.first == it.last } &&
        HOLE_MARKER !in tokens

    val terminalCompletion = cpuTimings.measure("terminal completion plan") {
      activeTerminalPlan
        ?: if (terminalResolutionEligible) tokens.lastOrNull()
          ?.takeIf { it !in cfg.terminals || freshUserInsertion && caretAtLastTokenEnd }
          ?.let { softCompletionManager.cachedPlan(cfg, tokens) }
        else null
    }

    if (
      activeTerminalPlan == null &&
      freshUserInsertion &&
      terminalCompletion != null
    ) softCompletionManager.remember(cfg, terminalCompletion)

    val unambiguousIncompleteExactTerminal = cpuTimings.measure("exact-terminal eligibility") {
      terminalCompletion == null &&
          terminalResolutionEligible &&
          freshUserInsertion &&
          caretAtLastTokenEnd &&
          tokens.lastOrNull()?.let(cfg::isUnambiguousExactTerminal) == true &&
          // Token separation is lexical, so even an invalid prefix such as
          // "... ID }" receives a space without pretending it has a CFG suffix.
          // The prefix before that token must still have been in suffix mode.
          tokens !in cfg.language &&
          cfg.visibleCompletionCFG.let {
            START_SYMBOL in it.nonterminals &&
              it.minimumNonemptySuffixLength(tokens.dropLast(1)) != null
          }
    }

    if (
      !hasActiveSoftInsertion &&
      activeTerminalPlan == null &&
      freshUserInsertion
    ) {
      val terminalCommitted =
        terminalCompletion?.terminalCommitted == true || unambiguousIncompleteExactTerminal
      val terminalRemainder =
        terminalCompletion?.expandedPrefix?.removePrefix(terminalCompletion.originalPrefix).orEmpty()
      val continuationTokens =
        terminalCompletion?.forcedContinuation.orEmpty()
      val continuation =
        if (continuationTokens.isEmpty()) "" else continuationTokens.joinToString(separator = " ", prefix = " ")
      val trailingSpace =
        if ( terminalCommitted && context.lastOrNull()?.isWhitespace() == false ) " " else ""

      (terminalRemainder + continuation + trailingSpace)
        .takeIf { it.isNotEmpty() }
        ?.let { insertion ->
          val caret = getCaretPosition().first
          val lineStart = getLineBounds().first
          val tokenEnd = lineStart +
            context.indexOfLast { !it.isWhitespace() } + 1
          val caretMovesPastExistingSeparator =
            terminalCommitted &&
              caret == tokenEnd &&
              tokenEnd - lineStart < context.length
          softCompletionManager.show(
            cfg = cfg,
            offset = tokenEnd,
            insertion = insertion,
            caretAfterCommit = caret + insertion.length +
              if (caretMovesPastExistingSeparator) 1 else 0
          )
        }
    }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, epsilons, ntStubs, resultLimit).hashCode()

    val hasHoleMarker = HOLE_MARKER in tokens
    if (hasHoleMarker) {
      val unknownToken = tokens.firstOrNull { it != HOLE_MARKER && cfg.tmMap[it] == null }
      if (unknownToken != null) {
        val workHash = tokens.hashCode() + cfg.hashCode() + settingsHash.hashCode()
        if (!inputWorkInvalidated && workHash == currentWorkHash) return
        runningJob?.cancel()
        inputWorkInvalidated = false
        currentWorkHash = workHash
        writeDisplayText(unknownTokenHtml(unknownToken))
        return
      }
    }

    var containsUnkTok = false
    val abstractUnk = tokens.mapIndexed { index, token ->
      if (token in cfg.terminals) token
      else {
        containsUnkTok = true
        if (terminalCompletion != null && index == tokens.lastIndex) token else HOLE_MARKER
      }
    }

    val requestCompletionCFG = cpuTimings.measure("prepare visible completion grammar") {
      cfg.visibleCompletionCFG
    }
    val minimumSuffixLength = cpuTimings.measure("minimum suffix query") {
      if (!containsUnkTok && START_SYMBOL in requestCompletionCFG.nonterminals)
        requestCompletionCFG.minimumSuffixLength(tokens)
      else null
    }
    val completeInput =
      if (START_SYMBOL in requestCompletionCFG.nonterminals) minimumSuffixLength == 0
      else !containsUnkTok && tokens in cfg.language
    val completeSuffixInput = suffixEligible && completeInput
    val continuationRequested = cpuTimings.measure("nonempty continuation query") {
      completeSuffixInput && requestCompletionCFG.minimumNonemptySuffixLength(tokens) != null
    }

    val routingStarted = TimeSource.Monotonic.markNow()
    val terminalCompletionHash = terminalCompletion?.let {
      listOf(
        TERMINAL_COMPLETION_WORK_SALT,
        it.originalPrefix,
        it.expandedPrefix,
        it.forcedContinuation,
        it.branches.map { branch -> branch.terminal to branch.suffixLengths.firstOrNull() }
      ).hashCode()
    } ?: 0
    val workHash = listOf(abstractUnk, continuationRequested).hashCode() +
      cfg.hashCode() + settingsHash.hashCode() + terminalCompletionHash
    if (!inputWorkInvalidated && workHash == currentWorkHash) return
    inputWorkInvalidated = false
    currentWorkHash = workHash

    val cached = cache[workHash]
    if (cached != null && (!continuationRequested || cached.startsWith("->"))) {
      runningJob?.cancel()
      runningJob = null
      return writeDisplayText(cached)
    }

    runningJob = MainScope().also { runningJob?.cancel() }.launch {
      val scenario = when {
        exactStubContext -> STUB
        hasHoleMarker -> COMPLETION
        terminalCompletion != null -> SUFFIX_COMPLETION
//        !containsUnkTok && forwardCompletion?.isValidContinuation(tokens) == true -> FORWARD_COMPLETION
        // This scenario can be handled much more elegantly using coalegbra and incremental decoding
        continuationRequested -> SUFFIX_COMPLETION
        completeInput -> PARSEABLE
        !containsUnkTok && !caretInMiddle() && minimumSuffixLength != null -> SUFFIX_COMPLETION
        else -> REPAIR
      }
      val suffixTimingTrace =
        if (scenario == SUFFIX_COMPLETION) linkedMapOf<String, Int>() else null
      if (suffixTimingTrace != null) {
        cpuTimings += SuffixCpuTiming("work hashing, cache lookup, and dispatch", routingStarted.elapsedNow())
        cpuTimings.logSuffixTimings("editor routing")
      }

      var gpuRepairResults: IntersectionResults? = null
      val candidates: Sequence<String>? = when (scenario) {
        STUB -> cfg.enumNTSmall(tokens[0].stripStub()).take(resultLimit)
        COMPLETION -> holeCompletionCandidates(cfg, tokens)
        SUFFIX_COMPLETION ->
          suffixCompletionCandidates(cfg, tokens, terminalCompletion, resultLimit, started) {
            suffixTimingTrace?.putAll(it)
          }
        INFIX_COMPLETION -> TODO("Infix mode not implemented")
        PARSEABLE -> {
          val parseTree = cfg.parse(tokens.joinToString(" "))?.prettyPrint()
          writeDisplayText("$parsedPrefix$parseTree".also { cache[workHash] = it }); null
        }
        REPAIR ->
          if (!gpuAvailable) { log("Repairing on CPU..."); sampleGREUntilTimeout(tokens, cfg) }
          else repairCode(cfg, tokens, LED_BUFFER, requestStarted = started)
            .let { results ->
              if (!caretInGrammar) results
              else results.mapTerminals { if (it == "[START]") "START" else it }
            }.also { gpuRepairResults = it }.asSequence()
      }

      val displayCandidates = candidates
        ?.let { if (epsilons) it.map(String::removeEpsilon).distinct() else it }
        ?.let { if (scenario != REPAIR) it.take(resultLimit) else it }
        ?.let { if (caretInGrammar && gpuRepairResults == null) it.map { it.replace("[START]", "START") } else it }
      if (suffixTimingTrace != null && "preprocessing" !in suffixTimingTrace)
        suffixTimingTrace["preprocessing"] = t0.elapsedNow().inWholeMilliseconds.toInt()
      val postCompletionSummary = TimeSource.Monotonic.markNow().let { postProcTimer -> {
        val report = when {
          gpuRepairResults != null -> timings
          suffixTimingTrace != null -> suffixTimingTrace.toMutableMap()
          else -> null
        }
        report?.let {
          it["postprocessing"] = postProcTimer.elapsedNow().inWholeMilliseconds.toInt()
          it["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
          log("Results rendered in ${it["total"]}ms")
          it.logTimesheet()
        }
        ", ${t0.elapsedNow()} latency."
      }}

      gpuRepairResults?.let { results ->
        val originalLength = displayTokens.sumOf(String::length)
        results.indices.asSequence().enumerateInteractively(
          workHash = workHash,
          keyOf = { it },
          metric = { index ->
            results.editDistanceAt(index) * 7919 + (originalLength - results.characterLengthAt(index)).absoluteValue
          },
          customDiff = results::htmlAt,
          resultsToPost = resultLimit,
          reason = scenario.reason,
          postCompletionSummary = postCompletionSummary
        )
      } ?: displayCandidates?.let { candidates ->
        val metric = when (scenario) {
          REPAIR -> levAndLenMetric(displayTokens)
          else -> ({ tokens: List<String> -> tokens.size })
        }
        val originalText = displayTokens.joinToString(" ")
        val commonPrefix = (terminalCompletion?.branches
          ?.map { it.tokens }?.commonTokenPrefix() ?: tokens)
          .let { prefix ->
            if (caretInGrammar) prefix.map { it.replace("[START]", "START") }
            else prefix
          }
        candidates.enumerateInteractively(
          workHash = workHash,
          keyOf = { it },
          metric = { metric(it.tokenizeByWhitespace()) },
          customDiff = { completion ->
            if (scenario == SUFFIX_COMPLETION) abbreviatedSuffixHTML(originalText, commonPrefix, completion)
            else levenshteinAlign(originalText, completion).paintDiffs()
          },
          resultsToPost = resultLimit,
          reason = scenario.reason,
          postCompletionSummary = postCompletionSummary
        )
      }
    }
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

  private fun selectedCompletion(index: Int): String =
    (output as? HTMLElement)
      ?.querySelector("mark [$COMPLETION_DATA_ATTRIBUTE]")
      ?.getAttribute(COMPLETION_DATA_ATTRIBUTE)
      ?.let(::decodeCompletion)
      ?: readDisplayText().lines()[index + 2]
        .substringAfter(".) ").replace("\\s+".toRegex(), " ").trim()

  open fun navUpdate(event: KeyboardEvent) {
    val key = event.keyCode.toSelectorAction() ?: return
    if (key == ENTER && event.shiftKey) return
    if (key == TAB) {
      event.preventDefault()
      if (!commitSoftTerminalInsertion()) handleTab()
      return
    }
    if (key == ARROW_RIGHT) {
      val dispText = readDisplayText()
      val mode = dispText.substringBefore("\n")
      if (!mode.startsWith("-> Forward completion")) return
    }
    if (key == ESCAPE) {
      softCompletionManager.clear()
      restoreInstructions()
      return
    }
    val currentText = rawDisplayHTML()
    val lines = currentText.lines()
    val htmlIndex = lines.indexOfFirst { it.startsWith("<mark>") }
    if (htmlIndex == -1) return
    event.preventDefault()
    val currentIdx = lines[htmlIndex].substringBefore(".)").substringAfterLast('>').trim().toInt()
    when (key) {
      ENTER -> {
        val selection = selectedCompletion(currentIdx)
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
        val selection = selectedCompletion(currentIdx)

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