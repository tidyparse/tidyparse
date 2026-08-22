import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tensor.UTMatrix
import kotlinx.browser.document
import org.w3c.dom.HTMLSpanElement
import org.w3c.dom.HTMLTextAreaElement

internal const val MAX_TERMINAL_COMPLETION_BRANCHES = 3
internal const val TERMINAL_COMPLETION_WORK_SALT = "terminal-completion"
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

internal fun CFG.hasTerminalCompletion(template: List<Σᐩ>): Boolean =
  UTMatrix(
    ts = template.map { token ->
      toBooleanArray( if (token == HOLE_MARKER) unitNonterminals else bimap[listOf(token)].toSet() )
    }.toTypedArray(),
    algebra = bitwiseAlgebra
  ).seekFixpoint().diagonals.last()[0][bindex[START_SYMBOL]]

internal fun CFG.validContinuationSuffixLengths(tokens: List<Σᐩ>): List<Int> =
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
        if (compareTokenSequences(minimum, existing.first) < 0) minimum else existing.first
      val mergedMaximum =
        if (compareTokenSequences(maximum, existing.second) > 0) maximum else existing.second
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
      compareTokenSequences(candidateMinimum, minimum) < 0
    ) minimum = candidateMinimum
    if (
      maximum == null ||
      compareTokenSequences(candidateMaximum, maximum) > 0
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
      .diagonals.last()[0][bindex[START_SYMBOL]] ?: return emptyList()
    val extrema = root.yieldBounds(boundsCache).extrema() ?: return emptyList()
    listOf(extrema.first, extrema.second)
  }
  if (suffixExtrema.any { it.size < branch.tokens.size || it.take(branch.tokens.size) != branch.tokens })
    return emptyList()

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
      includeCompleteInput = terminal == partial || (partial in terminals && terminal.isNonterminalStubIn(this))
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

internal fun TerminalCompletionPlan.enumerationBranches(): List<TerminalCompletionBranch> =
  branches.sortedWith(
    compareBy(
      // A partial token can already be a complete terminal while also
      // prefixing generated nonterminal stubs. Preserve that interpretation,
      // then spend the remaining bounded work on the branches that can finish
      // soonest instead of whichever terminal happens to sort first.
      { if (it.terminal == originalPrefix) 0 else 1 },
      { it.suffixLengths.minOrNull() ?: Int.MAX_VALUE },
      { it.terminal }
    )
  ).take(MAX_TERMINAL_COMPLETION_BRANCHES)

internal fun CFG.isUnambiguousExactTerminal(token: Σᐩ): Boolean =
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

internal fun CFG.enumTerminalSuffixes(branch: TerminalCompletionBranch): Sequence<Σᐩ> =
  sequence {
    if (0 in branch.suffixLengths) yield(branch.tokens.joinToString(" "))
    yieldAll(enumSuffixes(branch.tokens, branch.suffixLengths))
  }.distinct()

internal class SoftCompletionManager(
  private val editor: HTMLTextAreaElement,
  private val codeMirror: () -> dynamic,
  private val hasCodeMirror: () -> Boolean,
  private val codeMirrorPosition: (Int) -> dynamic,
  private val readEditorText: () -> Σᐩ,
  private val getCaretPosition: () -> IntRange,
  private val setCaretPosition: (IntRange) -> Unit,
  private val getLatestCFG: () -> CFG,
  private val completionContextHash: (CFG) -> Int,
  private val previewRenderer: (Σᐩ, Int) -> Boolean,
  private val afterCommit: () -> Unit
) {
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

  private data class FreshUserInsertion(val editorText: Σᐩ, val caret: IntRange)

  private data class CachedTerminalCompletion(val cfgHash: Int, val tokens: List<Σᐩ>, val plan: TerminalCompletionPlan?)

  private var softTerminalInsertion: SoftTerminalInsertion? = null
  private var softTerminalInsertionMarker: HTMLSpanElement? = null
  private var terminalPrefixResolution: TerminalPrefixResolution? = null
  private var cachedTerminalCompletion: CachedTerminalCompletion? = null
  private var nativeFreshUserInsertion: FreshUserInsertion? = null
  private var nativeCompositionActive = false

  init {
    editor.addEventListener("compositionstart", {
      nativeCompositionActive = true
      clear()
      clearFreshUserInsertion()
    })
    editor.addEventListener("compositionend", { nativeCompositionActive = false })
    editor.addEventListener("blur", { clear() })
    editor.addEventListener("input", input@{ event ->
      if (hasCodeMirror()) return@input
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
        clear()
        nativeFreshUserInsertion = null
      }
    })
    editor.addEventListener("selectionchange", {
      invalidateIfChanged()
      nativeFreshUserInsertion?.let { snapshot ->
        if (!snapshot.matchesCurrentEditorState())
          nativeFreshUserInsertion = null
      }
    })
  }

  internal val pendingInsertion: Σᐩ? get() = softTerminalInsertion?.insertion

  private fun SoftTerminalInsertion.matchesCurrentEditorState(): Boolean =
    editorText == readEditorText() && caret == getCaretPosition() && caret.first == caret.last

  private fun SoftTerminalInsertion.matchesCurrentContext(cfg: CFG): Boolean =
    matchesCurrentEditorState() && contextHash == completionContextHash(cfg)

  private fun TerminalPrefixResolution.matchesCurrentEditorState(): Boolean =
    editorText == readEditorText() && caret == getCaretPosition() && caret.first == caret.last

  private fun TerminalPrefixResolution.matchesCurrentContext(cfg: CFG): Boolean =
    matchesCurrentEditorState() && contextHash == completionContextHash(cfg)

  internal fun invalidateIfChanged() {
    if (
      softTerminalInsertion?.let { !it.matchesCurrentEditorState() } == true ||
      terminalPrefixResolution?.let { !it.matchesCurrentEditorState() } == true
    ) clear()
  }

  private fun stateMatchesCurrentEditor(): Boolean =
    softTerminalInsertion?.matchesCurrentEditorState() != false &&
      terminalPrefixResolution?.matchesCurrentEditorState() != false

  private fun installCodeMirrorCallbacks() {
    if (!hasCodeMirror()) return
    val cm = codeMirror()
    cm.tidyparseClearSoftInsertion = { clear() }
    cm.tidyparseInvalidateSoftInsertion = { invalidateIfChanged() }
    cm.tidyparseReconcileSoftInsertion =
      { insertedText: String, offset: Int ->
        reconcileInsertion(
          insertedText = insertedText,
          insertionOffset = offset,
          requireCurrentCaret = false
        )
      }
  }

  private fun clearInsertionPreview() {
    softTerminalInsertion = null
    softTerminalInsertionMarker?.remove()
    softTerminalInsertionMarker = null
    if (hasCodeMirror()) {
      val cm = codeMirror()
      cm.tidyparsePositionSoftInsertion = null
      if (terminalPrefixResolution == null) {
        cm.tidyparseClearSoftInsertion = null
        cm.tidyparseInvalidateSoftInsertion = null
        cm.tidyparseReconcileSoftInsertion = null
      }
    }
  }

  internal fun clear() {
    terminalPrefixResolution = null
    clearInsertionPreview()
  }

  internal fun remember(cfg: CFG, completion: TerminalCompletionPlan) {
    terminalPrefixResolution = TerminalPrefixResolution(
      editorText = readEditorText(),
      caret = getCaretPosition(),
      contextHash = completionContextHash(cfg),
      completion = completion
    )
    installCodeMirrorCallbacks()
  }

  internal fun activePlan(cfg: CFG): TerminalCompletionPlan? =
    terminalPrefixResolution?.takeIf { it.matchesCurrentContext(cfg) }?.completion
      .also { if (it == null && terminalPrefixResolution != null) clear() }

  internal fun hasActiveInsertion(cfg: CFG): Boolean =
    (softTerminalInsertion?.takeIf { it.matchesCurrentContext(cfg) } != null)
      .also { isActive -> if (!isActive && softTerminalInsertion != null) clear() }

  private fun updateInsertionMarker() {
    val insertion = softTerminalInsertion ?: return
    val marker = softTerminalInsertionMarker ?: return
    marker.textContent = insertion.insertion.ifBlank { "\u00a0" }
    if (insertion.insertion.isBlank()) marker.classList.add("tidyparse-soft-completion--whitespace-only")
    else marker.classList.remove("tidyparse-soft-completion--whitespace-only")
    marker.setAttribute("data-soft-completion", insertion.insertion)
    positionInsertion()
  }

  private fun reconcileInsertion(insertedText: String, insertionOffset: Int, requireCurrentCaret: Boolean): Boolean {
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

    terminalPrefixResolution = resolution?.copy(editorText = nextEditorText, caret = nextCaret)
    val remainingInsertion = insertion.insertion.removePrefix(insertedText)
    if (remainingInsertion.isEmpty()) {
      clearInsertionPreview()
      return true
    }

    softTerminalInsertion = insertion.copy(
      editorText = nextEditorText,
      caret = nextCaret,
      offset = nextOffset,
      insertion = remainingInsertion
    )
    updateInsertionMarker()
    return true
  }

  private fun reconcileInsertionFromCurrentEditor(): Boolean {
    val insertion = softTerminalInsertion ?: return false
    val currentText = readEditorText()
    val insertedLength = currentText.length - insertion.editorText.length
    if (
      insertedLength <= 0 ||
      insertion.offset + insertedLength > currentText.length
    ) return false

    return reconcileInsertion(
      insertedText = currentText.substring(insertion.offset, insertion.offset + insertedLength),
      insertionOffset = insertion.offset,
      requireCurrentCaret = true
    )
  }

  private fun positionInsertion() {
    val insertion = softTerminalInsertion ?: return
    val marker = softTerminalInsertionMarker ?: return
    val coordinates = codeMirror().cursorCoords(codeMirrorPosition(insertion.offset), "local")
    marker.style.left = "${coordinates.left + 1}px"
    marker.style.top = "${coordinates.top}px"
  }

  internal fun renderPreview(insertion: Σᐩ, offset: Int): Boolean {
    if (!hasCodeMirror()) return false
    val marker = document.createElement("span") as HTMLSpanElement
    marker.className = "tidyparse-soft-completion"
    marker.setAttribute("aria-hidden", "true")
    codeMirror().addWidget(codeMirrorPosition(offset), marker, false)
    softTerminalInsertionMarker = marker
    // Keep the overlay outside CodeMirror's editable line DOM, but align it
    // with the text baseline so contenteditable input retains a stable caret.
    updateInsertionMarker()
    installCodeMirrorCallbacks()
    codeMirror().tidyparsePositionSoftInsertion = { positionInsertion() }
    return true
  }

  internal fun show(cfg: CFG, offset: Int, insertion: Σᐩ, caretAfterCommit: Int) {
    clearInsertionPreview()
    softTerminalInsertion = SoftTerminalInsertion(
      editorText = readEditorText(),
      caret = getCaretPosition(),
      contextHash = completionContextHash(cfg),
      offset = offset,
      insertion = insertion,
      caretAfterCommit = caretAfterCommit
    )
    if (!previewRenderer(insertion, offset)) clearInsertionPreview()
  }

  internal fun commit(): Boolean {
    val insertion = softTerminalInsertion ?: return false
    val currentCfg = getLatestCFG()
    val renderedInsertionAvailable =
      !hasCodeMirror() || softTerminalInsertionMarker?.isConnected == true
    if (
      compositionActive() ||
      currentCfg.isEmpty() ||
      !renderedInsertionAvailable ||
      !insertion.matchesCurrentContext(currentCfg)
    ) {
      clear()
      return false
    }

    clear()
    insert(
      offset = insertion.offset,
      insertion = insertion.insertion,
      caret = insertion.caretAfterCommit
    )
    afterCommit()
    return true
  }

  private fun insert(offset: Int, insertion: Σᐩ, caret: Int) {
    if (hasCodeMirror()) {
      val cm = codeMirror()
      cm.tidyparseCompletionCommitActive = true
      try {
        cm.replaceRange(
          insertion,
          codeMirrorPosition(offset),
          codeMirrorPosition(offset),
          SOFT_COMPLETION_COMMIT_ORIGIN
        )
      } finally { cm.tidyparseCompletionCommitActive = false }
      setCaretPosition(caret.let { it..it })
      cm.save()
    } else {
      editor.value = buildString {
        append(editor.value.substring(0, offset))
        append(insertion)
        append(editor.value.substring(offset))
      }
      val selectionAfterInsertion = offset + insertion.length
      editor.setSelectionRange(selectionAfterInsertion, selectionAfterInsertion)
      setCaretPosition(caret.let { it..it })
    }
  }

  internal fun compositionActive(): Boolean {
    if (nativeCompositionActive) return true
    if (!hasCodeMirror()) return false
    val composing = codeMirror().display.input.composing
    return composing != null && composing != js("undefined")
  }

  internal fun undoOrRedoInProgress(): Boolean {
    if (!hasCodeMirror()) return false
    val origin = codeMirror().tidyparseLastChangeOrigin
    return origin == "undo" || origin == "redo"
  }

  private fun freshUserInsertionSnapshot() =
    FreshUserInsertion(readEditorText(), getCaretPosition())

  private fun FreshUserInsertion.matchesCurrentEditorState(): Boolean =
    editorText == readEditorText() && caret == getCaretPosition() && caret.first == caret.last

  internal fun recordFreshUserInsertion() {
    if (
      (softTerminalInsertion != null || terminalPrefixResolution != null) &&
      !stateMatchesCurrentEditor() &&
      !reconcileInsertionFromCurrentEditor()
    ) clear()
    nativeFreshUserInsertion = freshUserInsertionSnapshot()
  }

  private fun clearFreshUserInsertion() {
    nativeFreshUserInsertion = null
    if (hasCodeMirror()) {
      val cm = codeMirror()
      cm.tidyparseFreshInsertionText = null
      cm.tidyparseFreshInsertionStart = null
      cm.tidyparseFreshInsertionEnd = null
    }
  }

  internal fun consumeFreshUserInsertion(): Boolean {
    val snapshot =
      if (hasCodeMirror()) {
        val cm = codeMirror()
        val text = cm.tidyparseFreshInsertionText
        val start = cm.tidyparseFreshInsertionStart
        val end = cm.tidyparseFreshInsertionEnd
        clearFreshUserInsertion()
        if (
          text == null || text == js("undefined") ||
          start == null || start == js("undefined") ||
          end == null || end == js("undefined")
        ) null
        else FreshUserInsertion(text as String, (start as Int)..(end as Int))
      } else nativeFreshUserInsertion.also { nativeFreshUserInsertion = null }

    return snapshot?.matchesCurrentEditorState() == true
  }

  internal fun cachedPlan(cfg: CFG, tokens: List<Σᐩ>): TerminalCompletionPlan? {
    val cfgHash = cfg.hashCode()
    cachedTerminalCompletion
      ?.takeIf { it.cfgHash == cfgHash && it.tokens == tokens }
      ?.let { return it.plan }

    return cfg.terminalCompletionPlan(tokens).also { plan ->
      cachedTerminalCompletion = CachedTerminalCompletion(
        cfgHash = cfgHash,
        tokens = tokens.toList(),
        plan = plan
      )
    }
  }
}