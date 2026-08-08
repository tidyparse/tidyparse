import JSTidyEditor.SelectorAction.*
import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import ai.hypergraph.kaliningraph.tensor.UTMatrix
import ai.hypergraph.tidyparse.*
import ai.hypergraph.tidyparse.TidyEditor.Scenario.*
import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.*
import org.w3c.dom.*
import org.w3c.dom.events.KeyboardEvent
import kotlin.time.TimeSource

internal const val MAX_TERMINAL_COMPLETION_BRANCHES = 3
private const val TERMINAL_COMPLETION_WORK_SALT = "terminal-completion"
private const val SOFT_COMPLETION_COMMIT_ORIGIN = "+tidyparse-soft-completion"

internal data class TerminalCompletionBranch(
  val terminal: Σᐩ,
  val tokens: List<Σᐩ>,
  val suffixLengths: List<Int>
)

internal data class TerminalCompletionPlan(
  val originalPrefix: Σᐩ,
  val expandedPrefix: Σᐩ,
  val lexicalCandidateCount: Int,
  val terminalCommitted: Boolean,
  val forcedContinuation: List<Σᐩ>,
  val branches: List<TerminalCompletionBranch>
)

private fun Iterable<Σᐩ>.longestCommonPrefix(): Σᐩ {
  val strings = toList()
  if (strings.isEmpty()) return ""

  val shortest = strings.minBy { it.length }
  return shortest.take(shortest.indices
    .firstOrNull { index -> strings.any { it[index] != shortest[index] } }
    ?: shortest.length)
}

private fun CFG.hasTerminalCompletion(template: List<Σᐩ>): Boolean =
  UTMatrix(
    ts = template.map { token ->
      toBooleanArray(
        if (token == HOLE_MARKER) unitNonterminals
        else bimap[listOf(token)].toSet()
      )
    }.toTypedArray(),
    algebra = bitwiseAlgebra
  ).seekFixpoint().diagonals.last()[0][bindex[START_SYMBOL]]

private fun CFG.validContinuationSuffixLengths(tokens: List<Σᐩ>): List<Int> =
  (1..MAX_SUFF_LEN).filter { suffixLength ->
    hasTerminalCompletion(tokens + List(suffixLength) { HOLE_MARKER })
  }

private fun CFG.validCompletionSuffixLengths(tokens: List<Σᐩ>, includeCompleteInput: Boolean): List<Int> =
  buildList {
    if (includeCompleteInput && hasTerminalCompletion(tokens)) add(0)
    addAll(validContinuationSuffixLengths(tokens))
  }

private typealias TokenSequenceBounds = Pair<List<Σᐩ>, List<Σᐩ>>

// The lexicographic extrema determine the common prefix of every yield at
// this visible length without enumerating an exponentially large forest.
private data class PTreeYieldBounds(val byLength: Map<Int, TokenSequenceBounds>)

private fun compareTokenSequences(left: List<Σᐩ>, right: List<Σᐩ>): Int {
  for (index in 0..<minOf(left.size, right.size)) {
    val comparison = left[index].compareTo(right[index])
    if (comparison != 0) return comparison
  }
  return left.size.compareTo(right.size)
}

private fun MutableMap<Int, TokenSequenceBounds>.mergeBounds(minimum: List<Σᐩ>, maximum: List<Σᐩ>) {
  val existing = this[minimum.size]
  this[minimum.size] =
    if (existing == null) minimum to maximum
    else {
      val mergedMinimum =
        if (compareTokenSequences(minimum, existing.first) < 0) minimum
        else existing.first
      val mergedMaximum =
        if (compareTokenSequences(maximum, existing.second) > 0) maximum
        else existing.second
      mergedMinimum to mergedMaximum
    }
}

private fun PTree.yieldBounds(cache: MutableMap<PTree, PTreeYieldBounds>): PTreeYieldBounds {
  cache[this]?.let { return it }

  val result =
    if (branches.isEmpty()) {
      val sequence = if (epsStr.isEmpty()) emptyList() else listOf(root)
      PTreeYieldBounds(mapOf(sequence.size to (sequence to sequence)))
    } else {
      val bounds = mutableMapOf<Int, TokenSequenceBounds>()
      branches.forEach { (left, right) ->
        val leftBounds = left.yieldBounds(cache).byLength
        val rightBounds = right.yieldBounds(cache).byLength
        leftBounds.values.forEach { (leftMinimum, leftMaximum) ->
          rightBounds.values.forEach { (rightMinimum, rightMaximum) ->
            bounds.mergeBounds(
              minimum = leftMinimum + rightMinimum,
              maximum = leftMaximum + rightMaximum
            )
          }
        }
      }
      PTreeYieldBounds(bounds)
    }

  cache[this] = result
  return result
}

private fun PTreeYieldBounds.extrema(): TokenSequenceBounds? {
  var minimum: List<Σᐩ>? = null
  var maximum: List<Σᐩ>? = null

  byLength.values.forEach { (candidateMinimum, candidateMaximum) ->
    if (
      minimum == null ||
      compareTokenSequences(candidateMinimum, minimum!!) < 0
    ) minimum = candidateMinimum
    if (
      maximum == null ||
      compareTokenSequences(candidateMaximum, maximum!!) > 0
    ) maximum = candidateMaximum
  }

  return if (minimum == null || maximum == null) null else minimum to maximum
}

private fun CFG.commonForcedContinuation(branch: TerminalCompletionBranch): List<Σᐩ> {
  // Keep separate bounds by visible length because ε-enabled forests can
  // represent several visible widths for the same number of suffix holes.
  val boundsCache = mutableMapOf<PTree, PTreeYieldBounds>()
  val suffixExtrema = branch.suffixLengths.flatMap { suffixLength ->
    val template = branch.tokens + List(suffixLength) { HOLE_MARKER }
    val root = initPTreeListMat(template).seekFixpoint()
      .diagonals.last()[0][bindex[START_SYMBOL]]
      ?: return emptyList()
    val extrema = root.yieldBounds(boundsCache).extrema()
      ?: return emptyList()
    listOf(extrema.first, extrema.second)
  }
  if (suffixExtrema.any {
      it.size < branch.tokens.size ||
        it.take(branch.tokens.size) != branch.tokens
     }) return emptyList()

  val suffixes = suffixExtrema.map { it.drop(branch.tokens.size) }
  val commonLength = suffixes.minOfOrNull { it.size } ?: return emptyList()
  val continuation = mutableListOf<Σᐩ>()

  for (offset in 0..<commonLength) {
    val terminals = suffixes.map { it[offset] }.toSet()
    if (terminals.size != 1) break
    continuation += terminals.single()
  }

  return continuation
}

private fun TerminalCompletionBranch.advanceBy(forcedContinuation: List<Σᐩ>): TerminalCompletionBranch =
  if (forcedContinuation.isEmpty()) this
  else copy(
    tokens = tokens + forcedContinuation,
    suffixLengths = suffixLengths.map {
      it - forcedContinuation.size
    }
  )

internal fun CFG.terminalCompletionPlan(tokens: List<Σᐩ>): TerminalCompletionPlan? {
  val partial = tokens.lastOrNull() ?: return null
  val lexicalCandidates = terminals.filter { it.startsWith(partial) }.sorted()
  if (lexicalCandidates.isEmpty()) return null
  val exactInputComplete = partial in terminals && hasTerminalCompletion(tokens)

  // Resolve lexical ambiguity using the grammar: a spelling that cannot lead
  // to a completion is not a viable interpretation of the last token. Length
  // zero matters when completing the partial token itself finishes the input,
  // as with a generated nonterminal stub such as "<" -> "<EXP>".
  val prefixTokens = tokens.dropLast(1)
  val viableBranches = lexicalCandidates.mapNotNull { terminal ->
    val candidateTokens = prefixTokens + terminal
    val suffixLengths = validCompletionSuffixLengths(
      tokens = candidateTokens,
      // A complete exact spelling preserves the already-typed terminal
      // interpretation. When that spelling is also a prefix of a stub, the
      // complete stub is the competing interpretation we need to retain.
      // Other partial tokens still require a positive continuation.
      includeCompleteInput =
        terminal == partial || (partial in terminals && terminal.isNonterminalStubIn(this))
    )
    suffixLengths.takeIf { it.isNotEmpty() }
      ?.let { TerminalCompletionBranch(terminal, candidateTokens, it) }
  }
  if (viableBranches.isEmpty()) return null
  if (
    exactInputComplete && viableBranches.none {
      it.terminal != partial &&
        it.terminal.isNonterminalStubIn(this)
    }
  ) return null

  val terminalCommitted = viableBranches.size == 1
  val expandedPrefix = when {
    terminalCommitted -> viableBranches.single().terminal
    viableBranches.size > 1 ->
      viableBranches.map { it.terminal }.longestCommonPrefix()
        .takeIf { it.length > partial.length } ?: partial
    else -> partial
  }
  val forcedContinuation =
    if (terminalCommitted) commonForcedContinuation(viableBranches.single())
    else emptyList()
  val advancedBranches = viableBranches.map { it.advanceBy(forcedContinuation) }

  return TerminalCompletionPlan(
    originalPrefix = partial,
    expandedPrefix = expandedPrefix,
    lexicalCandidateCount = lexicalCandidates.size,
    terminalCommitted = terminalCommitted,
    forcedContinuation = forcedContinuation,
    branches = advancedBranches
  )
}

private fun TerminalCompletionPlan.enumerationBranches(): List<TerminalCompletionBranch> =
  branches.sortedWith(
    compareBy<TerminalCompletionBranch>(
      // A partial token can already be a complete terminal while also
      // prefixing generated nonterminal stubs. Preserve that interpretation,
      // then spend the remaining bounded work on the branches that can finish
      // soonest instead of whichever terminal happens to sort first.
      { if (it.terminal == originalPrefix) 0 else 1 },
      { it.suffixLengths.minOrNull() ?: Int.MAX_VALUE },
      { it.terminal }
    )
  ).take(MAX_TERMINAL_COMPLETION_BRANCHES)

private fun CFG.isUnambiguousExactTerminal(token: Σᐩ): Boolean =
  terminals.count { it.startsWith(token) } == 1 && token in terminals

internal fun <T> fairMerge(sequences: List<Sequence<T>>): Sequence<T> = sequence {
  var active = sequences.map { it.iterator() }

  while (active.isNotEmpty()) {
    val nextRound = mutableListOf<Iterator<T>>()
    for (iterator in active) {
      if (iterator.hasNext()) {
        yield(iterator.next())
        nextRound += iterator
      }
    }
    active = nextRound
  }
}

private fun CFG.enumTerminalSuffixes(
  branch: TerminalCompletionBranch
): Sequence<Σᐩ> = sequence {
  if (0 in branch.suffixLengths) yield(branch.tokens.joinToString(" "))
  yieldAll(enumSuffixes(branch.tokens, branch.suffixLengths))
}.distinct()

/** Compare with [ai.hypergraph.tidyparse.IJTidyEditor] */
open class JSTidyEditor(val editor: HTMLTextAreaElement, val output: Node): TidyEditor() {
  private data class SoftTerminalInsertion(
    val editorText: Σᐩ,
    val caret: IntRange,
    val contextHash: Int,
    val offset: Int,
    val insertion: Σᐩ,
    val caretAfterCommit: Int
  )

  private data class TerminalPrefixResolution(
    val editorText: Σᐩ,
    val caret: IntRange,
    val contextHash: Int,
    val completion: TerminalCompletionPlan
  )

  private data class FreshUserInsertion(
    val editorText: Σᐩ,
    val caret: IntRange
  )

  private data class CachedTerminalCompletion(
    val cfgHash: Int,
    val tokens: List<Σᐩ>,
    val plan: TerminalCompletionPlan?
  )

  private var softTerminalInsertion: SoftTerminalInsertion? = null
  private var softTerminalInsertionMarker: HTMLSpanElement? = null
  private var terminalPrefixResolution: TerminalPrefixResolution? = null
  private var cachedTerminalCompletion: CachedTerminalCompletion? = null
  private var nativeFreshUserInsertion: FreshUserInsertion? = null
  private var nativeCompositionActive = false

  init {
    editor.addEventListener("compositionstart", {
      nativeCompositionActive = true
      clearTerminalCompletionState()
      clearFreshUserInsertion()
    })
    editor.addEventListener("compositionend", { nativeCompositionActive = false })
    editor.addEventListener("blur", { clearTerminalCompletionState() })
    editor.addEventListener("input", { event ->
      if (hasCodeMirror()) return@addEventListener
      val inputEvent = event.asDynamic()
      val isFreshInsertion =
        inputEvent.isTrusted == true &&
          inputEvent.isComposing != true &&
          inputEvent.data is String &&
          (inputEvent.data as String).isNotEmpty() &&
          inputEvent.inputType in arrayOf(
            "insertText",
            "insertCompositionText",
            "insertFromComposition"
          )
      if (isFreshInsertion) recordFreshUserInsertion()
      else {
        clearTerminalCompletionState()
        nativeFreshUserInsertion = null
      }
    })
    editor.addEventListener("selectionchange", {
      invalidateTerminalCompletionStateIfChanged()
      nativeFreshUserInsertion?.let { snapshot ->
        if (!snapshot.matchesCurrentEditorState())
          nativeFreshUserInsertion = null
      }
    })
  }

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

  private fun insertTerminalCompletion(offset: Int, insertion: Σᐩ, caret: Int) {
    if (hasCodeMirror()) {
      codeMirror.tidyparseCompletionCommitActive = true
      try {
        codeMirror.replaceRange(
          insertion,
          cmPos(offset),
          cmPos(offset),
          SOFT_COMPLETION_COMMIT_ORIGIN
        )
      } finally {
        codeMirror.tidyparseCompletionCommitActive = false
      }
      setCaretPosition(caret.let { it..it })
      codeMirror.save()
    } else {
      editor.overwriteCurrentLineWith(offset..offset, insertion)
      setCaretPosition(caret.let { it..it })
    }
  }

  private fun currentCompletionContextHash(cfg: CFG): Int =
    listOf(cfg.hashCode(), epsilons, ntStubs).hashCode()

  private fun SoftTerminalInsertion.matchesCurrentEditorState(): Boolean =
    editorText == readEditorText() && caret == getCaretPosition() && caret.first == caret.last

  private fun SoftTerminalInsertion.matchesCurrentContext(cfg: CFG): Boolean =
    matchesCurrentEditorState() && contextHash == currentCompletionContextHash(cfg)

  private fun TerminalPrefixResolution.matchesCurrentEditorState(): Boolean =
    editorText == readEditorText() && caret == getCaretPosition() && caret.first == caret.last

  private fun TerminalPrefixResolution.matchesCurrentContext(cfg: CFG): Boolean =
    matchesCurrentEditorState() && contextHash == currentCompletionContextHash(cfg)

  private fun invalidateTerminalCompletionStateIfChanged() {
    if (
      softTerminalInsertion?.let { !it.matchesCurrentEditorState() } == true ||
      terminalPrefixResolution?.let { !it.matchesCurrentEditorState() } == true
    ) clearTerminalCompletionState()
  }

  private fun terminalCompletionStateMatchesCurrentEditor(): Boolean =
    softTerminalInsertion?.matchesCurrentEditorState() != false &&
      terminalPrefixResolution?.matchesCurrentEditorState() != false

  private fun installTerminalCompletionCallbacks() {
    if (!hasCodeMirror()) return
    codeMirror.tidyparseClearSoftInsertion = { clearTerminalCompletionState() }
    codeMirror.tidyparseInvalidateSoftInsertion = { invalidateTerminalCompletionStateIfChanged() }
    codeMirror.tidyparseReconcileSoftInsertion =
      { insertedText: String, offset: Int ->
        reconcileSoftTerminalInsertion(
          insertedText = insertedText,
          insertionOffset = offset,
          requireCurrentCaret = false
        )
      }
  }

  private fun clearSoftTerminalInsertionPreview() {
    softTerminalInsertion = null
    softTerminalInsertionMarker?.remove()
    softTerminalInsertionMarker = null
    if (hasCodeMirror()) {
      codeMirror.tidyparsePositionSoftInsertion = null
      if (terminalPrefixResolution == null) {
        codeMirror.tidyparseClearSoftInsertion = null
        codeMirror.tidyparseInvalidateSoftInsertion = null
        codeMirror.tidyparseReconcileSoftInsertion = null
      }
    }
  }

  private fun clearTerminalCompletionState() {
    terminalPrefixResolution = null
    clearSoftTerminalInsertionPreview()
  }

  private fun rememberTerminalPrefixResolution(cfg: CFG, completion: TerminalCompletionPlan) {
    terminalPrefixResolution = TerminalPrefixResolution(
      editorText = readEditorText(),
      caret = getCaretPosition(),
      contextHash = currentCompletionContextHash(cfg),
      completion = completion
    )
    installTerminalCompletionCallbacks()
  }

  private fun updateSoftTerminalInsertionMarker() {
    val insertion = softTerminalInsertion ?: return
    val marker = softTerminalInsertionMarker ?: return
    marker.textContent = insertion.insertion.ifBlank { "\u00a0" }
    if (insertion.insertion.isBlank()) marker.classList.add("tidyparse-soft-completion--whitespace-only")
    else marker.classList.remove("tidyparse-soft-completion--whitespace-only")
    marker.setAttribute("data-soft-completion", insertion.insertion)
    positionSoftTerminalInsertion()
  }

  private fun reconcileSoftTerminalInsertion(
    insertedText: String,
    insertionOffset: Int,
    requireCurrentCaret: Boolean
  ): Boolean {
    val insertion = softTerminalInsertion ?: return false
    val resolution = terminalPrefixResolution
    if (
      insertedText.isEmpty() ||
      insertionOffset != insertion.offset ||
      insertion.caret.first != insertion.caret.last ||
      insertion.caret.first != insertion.offset ||
      !insertion.insertion.startsWith(insertedText) ||
      resolution?.let {
        it.editorText != insertion.editorText ||
          it.caret != insertion.caret ||
          it.contextHash != insertion.contextHash
      } == true
    ) return false

    val nextEditorText = buildString {
      append(insertion.editorText.substring(0, insertionOffset))
      append(insertedText)
      append(insertion.editorText.substring(insertionOffset))
    }
    val nextOffset = insertionOffset + insertedText.length
    val nextCaret = nextOffset.let { it..it }
    if (
      readEditorText() != nextEditorText ||
      requireCurrentCaret && getCaretPosition() != nextCaret
    ) return false

    terminalPrefixResolution = resolution?.copy(
      editorText = nextEditorText,
      caret = nextCaret
    )
    val remainingInsertion =
      insertion.insertion.removePrefix(insertedText)
    if (remainingInsertion.isEmpty()) {
      clearSoftTerminalInsertionPreview()
      return true
    }

    softTerminalInsertion = insertion.copy(
      editorText = nextEditorText,
      caret = nextCaret,
      offset = nextOffset,
      insertion = remainingInsertion
    )
    updateSoftTerminalInsertionMarker()
    return true
  }

  private fun reconcileSoftTerminalInsertionFromCurrentEditor(): Boolean {
    val insertion = softTerminalInsertion ?: return false
    val currentText = readEditorText()
    val insertedLength = currentText.length - insertion.editorText.length
    if (insertedLength <= 0 || insertion.offset + insertedLength > currentText.length ) return false

    return reconcileSoftTerminalInsertion(
      insertedText = currentText.substring(insertion.offset, insertion.offset + insertedLength),
      insertionOffset = insertion.offset,
      requireCurrentCaret = true
    )
  }

  private fun positionSoftTerminalInsertion() {
    val insertion = softTerminalInsertion ?: return
    val marker = softTerminalInsertionMarker ?: return
    val coordinates = codeMirror.cursorCoords(cmPos(insertion.offset), "local")
    marker.style.left = "${coordinates.left + 1}px"
    marker.style.top = "${coordinates.top}px"
  }

  protected open fun renderSoftTerminalInsertionPreview(insertion: Σᐩ, offset: Int): Boolean {
    if (!hasCodeMirror()) return false
    val marker = document.createElement("span") as HTMLSpanElement
    marker.className = "tidyparse-soft-completion"
    marker.setAttribute("aria-hidden", "true")
    codeMirror.addWidget(cmPos(offset), marker, false)
    softTerminalInsertionMarker = marker
    // Keep the overlay outside CodeMirror's editable line DOM, but align it
    // with the text baseline so contenteditable input retains a stable caret.
    updateSoftTerminalInsertionMarker()
    installTerminalCompletionCallbacks()
    codeMirror.tidyparsePositionSoftInsertion = {
      positionSoftTerminalInsertion()
    }
    return true
  }

  private fun showSoftTerminalInsertion(insertion: SoftTerminalInsertion) {
    clearSoftTerminalInsertionPreview()
    softTerminalInsertion = insertion
    if (!renderSoftTerminalInsertionPreview(
        insertion = insertion.insertion,
        offset = insertion.offset
      )) clearSoftTerminalInsertionPreview()
  }

  internal val pendingTerminalCompletionInsertion: Σᐩ? get() = softTerminalInsertion?.insertion

  internal fun discardSoftTerminalCompletion() = clearTerminalCompletionState()

  internal fun commitSoftTerminalInsertion(): Boolean {
    val insertion = softTerminalInsertion ?: return false
    val currentCfg = getLatestCFG()
    val renderedInsertionAvailable = !hasCodeMirror() || softTerminalInsertionMarker?.isConnected == true
    if (
      compositionActive() ||
      currentCfg.isEmpty() ||
      !renderedInsertionAvailable ||
      !insertion.matchesCurrentContext(currentCfg)
    ) {
      clearTerminalCompletionState()
      return false
    }

    clearTerminalCompletionState()
    insertTerminalCompletion(
      offset = insertion.offset,
      insertion = insertion.insertion,
      caret = insertion.caretAfterCommit
    )
    redecorateLines()
    continuation { handleInput() }
    return true
  }

  private fun compositionActive(): Boolean {
    if (nativeCompositionActive) return true
    if (!hasCodeMirror()) return false
    val composing = codeMirror.display.input.composing
    return composing != null && composing != js("undefined")
  }

  private fun undoOrRedoInProgress(): Boolean {
    if (!hasCodeMirror()) return false
    val origin = codeMirror.tidyparseLastChangeOrigin
    return origin == "undo" || origin == "redo"
  }

  private fun freshUserInsertionSnapshot() =
    FreshUserInsertion(readEditorText(), getCaretPosition())

  private fun FreshUserInsertion.matchesCurrentEditorState(): Boolean =
    editorText == readEditorText() &&
      caret == getCaretPosition() &&
      caret.first == caret.last

  internal fun recordFreshUserInsertion() {
    if (
      (softTerminalInsertion != null ||
        terminalPrefixResolution != null) &&
      !terminalCompletionStateMatchesCurrentEditor() &&
      !reconcileSoftTerminalInsertionFromCurrentEditor()
    ) clearTerminalCompletionState()
    nativeFreshUserInsertion = freshUserInsertionSnapshot()
  }

  private fun clearFreshUserInsertion() {
    nativeFreshUserInsertion = null
    if (hasCodeMirror()) {
      codeMirror.tidyparseFreshInsertionText = null
      codeMirror.tidyparseFreshInsertionStart = null
      codeMirror.tidyparseFreshInsertionEnd = null
    }
  }

  private fun consumeFreshUserInsertion(): Boolean {
    val snapshot =
      if (hasCodeMirror()) {
        val text = codeMirror.tidyparseFreshInsertionText
        val start = codeMirror.tidyparseFreshInsertionStart
        val end = codeMirror.tidyparseFreshInsertionEnd
        clearFreshUserInsertion()
        if (
          text == null || text == js("undefined") ||
          start == null || start == js("undefined") ||
          end == null || end == js("undefined")
        ) null
        else FreshUserInsertion(text as String, (start as Int)..(end as Int))
      } else nativeFreshUserInsertion.also {
        nativeFreshUserInsertion = null
      }

    return snapshot?.matchesCurrentEditorState() == true
  }

  private fun CFG.cachedTerminalCompletionPlan(
    tokens: List<Σᐩ>
  ): TerminalCompletionPlan? {
    val cfgHash = hashCode()
    cachedTerminalCompletion
      ?.takeIf { it.cfgHash == cfgHash && it.tokens == tokens }
      ?.let { return it.plan }

    return terminalCompletionPlan(tokens).also { plan ->
      cachedTerminalCompletion = CachedTerminalCompletion(
        cfgHash = cfgHash,
        tokens = tokens.toList(),
        plan = plan
      )
    }
  }

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
//  fun isValidContinuation(tokens: List<Σᐩ>, cfg: CFG): Boolean = tokens in cfg.admitsPrefix(tokens)
//  fun ForwardCompletion?.seed(tokens: List<Σᐩ>) =
//    if (this == null) ForwardCompletion(tokens, emptySet())
//    else if (cfg.isEmpty() || nextTerms.isEmpty()) ForwardCompletion(tokens, emptySet())
//    else TODO()
//  var forwardCompletion: ForwardCompletion? = null

  fun restoreInstructions() = writeDisplayText(instructions)

  override fun handleInput() {
    invalidateTerminalCompletionStateIfChanged()
    val t0 = TimeSource.Monotonic.markNow()
    val freshUserInsertion = !compositionActive() && consumeFreshUserInsertion()
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

    if (cfg.isEmpty()) {
      clearTerminalCompletionState()
      return
    }

    val exactStubContext = tokens.size == 1 && stubMatcher.matches(tokens.single())

    val caretAtLastTokenEnd =
      getCaretPosition().let { caret ->
        caret.first == caret.last &&
          caret.first == getLineBounds().first +
            context.indexOfLast { !it.isWhitespace() } + 1
      }

    val activeTerminalResolution =
      terminalPrefixResolution?.takeIf { it.matchesCurrentContext(cfg) }
      .also { if (it == null && terminalPrefixResolution != null) clearTerminalCompletionState() }

    val activeSoftInsertion =
      softTerminalInsertion ?.takeIf { it.matchesCurrentContext(cfg) }
      .also { if (it == null && softTerminalInsertion != null) clearTerminalCompletionState() }

    val terminalResolutionEligible =
      !caretInGrammar &&
        !exactStubContext &&
        !caretInMiddle() &&
        !compositionActive() &&
        !undoOrRedoInProgress() &&
        getCaretPosition().let { it.first == it.last } &&
        HOLE_MARKER !in tokens

    val terminalCompletion = activeTerminalResolution?.completion
      ?: if (terminalResolutionEligible) tokens.lastOrNull()
        ?.takeIf { it !in cfg.terminals || freshUserInsertion && caretAtLastTokenEnd }
        ?.let { cfg.cachedTerminalCompletionPlan(tokens) }
      else null

    if (
      activeTerminalResolution == null &&
      freshUserInsertion &&
      terminalCompletion != null
    ) rememberTerminalPrefixResolution(cfg, terminalCompletion)

    val unambiguousIncompleteExactTerminal =
      terminalCompletion == null &&
        terminalResolutionEligible &&
        freshUserInsertion &&
        caretAtLastTokenEnd &&
        tokens.lastOrNull()?.let(cfg::isUnambiguousExactTerminal) == true &&
        // Token separation is lexical, so even an invalid prefix such as
        // "... ID }" receives a space without pretending it has a CFG suffix.
        // The prefix before that token must still have been in suffix mode.
        !cfg.hasTerminalCompletion(tokens) &&
        cfg.validContinuationSuffixLengths(tokens.dropLast(1)).isNotEmpty()

    if (
      activeSoftInsertion == null &&
      activeTerminalResolution == null &&
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
          showSoftTerminalInsertion(SoftTerminalInsertion(
            editorText = readEditorText(),
            caret = getCaretPosition(),
            contextHash = currentCompletionContextHash(cfg),
            offset = tokenEnd,
            insertion = insertion,
            caretAfterCommit = caret + insertion.length +
              if (caretMovesPastExistingSeparator) 1 else 0,
          ))
        }
    }

    val settingsHash = listOf(LED_BUFFER, TIMEOUT_MS, epsilons, ntStubs).hashCode()

    val hasHoleMarker = HOLE_MARKER in tokens
    if (hasHoleMarker) {
      val unknownToken = tokens.firstOrNull { it != HOLE_MARKER && cfg.tmMap[it] == null }
      if (unknownToken != null) {
        val workHash = tokens.hashCode() + cfg.hashCode() + settingsHash.hashCode()
        if (workHash == currentWorkHash) return
        runningJob?.cancel()
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

    val terminalCompletionHash = terminalCompletion?.let {
      listOf(
        TERMINAL_COMPLETION_WORK_SALT,
        it.originalPrefix,
        it.expandedPrefix,
        it.forcedContinuation,
        it.branches.map { branch ->
          branch.terminal to branch.suffixLengths
        }
      ).hashCode()
    } ?: 0
    val workHash = abstractUnk.hashCode() + cfg.hashCode() + settingsHash.hashCode() + terminalCompletionHash
    if (workHash == currentWorkHash) return
    currentWorkHash = workHash

    val cached = cache[workHash]
    if (cached != null && (!suffixEligible || cached.startsWith("->"))) {
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
        tokens in cfg.language && !suffixEligible -> PARSEABLE
        !containsUnkTok -> handleSuffixCheck(cfg.language, tokens)
        else -> REPAIR
      }

      when (scenario) {
        STUB -> cfg.enumNTSmall(tokens[0].stripStub()).take(100)
        COMPLETION -> if (!gpuAvailable) cfg.enumSeqSmart(tokens) else completeCode(cfg, tokens).stripEpsilon()
        SUFFIX_COMPLETION ->
          terminalCompletion?.enumerationBranches()
            ?.map(cfg::enumTerminalSuffixes)
            ?.let(::fairMerge)?.distinct()
            ?: cfg.enumSuffixes(tokens, scenario.data).distinct()
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
        customDiff = { completion ->
          levenshteinAlign(tokens.joinToString(" "), completion).paintDiffs()
        },
        reason = scenario.reason,
        postCompletionSummary = TimeSource.Monotonic.markNow().let { postProcTimer -> {
          if (gpuAvailable) {
            mark("postprocessing", postProcTimer);
            timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
            log("Results rendered in ${timings["total"]}ms")
            timings.logTimesheet()
          }
          ", ${t0.elapsedNow()} latency."
        }}
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
      clearTerminalCompletionState()
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
