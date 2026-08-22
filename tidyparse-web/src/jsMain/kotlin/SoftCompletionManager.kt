import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.types.cache
import kotlinx.browser.document
import org.w3c.dom.HTMLSpanElement
import org.w3c.dom.HTMLTextAreaElement

internal const val MAX_TERMINAL_COMPLETION_BRANCHES = 3
internal const val TERMINAL_COMPLETION_WORK_SALT = "terminal-completion"
private const val SOFT_COMPLETION_COMMIT_ORIGIN = "+tidyparse-soft-completion"

internal data class TerminalCompletionBranch(
  val terminal: Σᐩ,
  val tokens: List<Σᐩ>,
  val suffixLengths: Sequence<Int>
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

// Normalization materializes nullable alternatives, so its literal ε padding
// can be pruned before counting editor-visible tokens and enumerating rows.
internal val CFG.visibleCompletionCFG: CFG by cache {
  if ("ε" !in terminals) this
  else {
    val originalNonterminals = nonterminals
    val rules = filter { (lhs, rhs) -> "ε" !in lhs && rhs.none { "ε" in it } }
    val productive = (terminals - "ε").toMutableSet()
    while (productive.addAll(rules.filter { (_, rhs) -> rhs.all(productive::contains) }.map { it.LHS })) {}
    rules.filter { (lhs, rhs) ->
      lhs in productive && rhs.all { it !in originalNonterminals || it in productive }
    }.toSet().freeze()
  }
}

private fun CFG.validCompletionSuffixLengths(
  tokens: List<Σᐩ>,
  includeCompleteInput: Boolean,
  prefixCFG: CFG
): Sequence<Int> =
  if (tokens !in prefixCFG.language) emptySequence()
  else visibleCompletionCFG.let {
    if (START_SYMBOL in it.nonterminals) it.completionSuffixLengths(tokens, includeCompleteInput)
    else emptySequence()
  }

private fun CFG.commonForcedContinuation(branch: TerminalCompletionBranch): List<Σᐩ> {
  val includesEmpty = branch.suffixLengths.firstOrNull() == 0
  return visibleCompletionCFG.let {
    if (START_SYMBOL in it.nonterminals)
      it.completionIndex.forcedContinuation(branch.tokens, includesEmpty)
    else emptyList()
  }
}

private fun TerminalCompletionBranch.advanceBy(forcedContinuation: List<Σᐩ>): TerminalCompletionBranch =
  if (forcedContinuation.isEmpty()) this
  else copy(
    tokens = tokens + forcedContinuation,
    suffixLengths = suffixLengths.map {
      it - forcedContinuation.size
    }
  )

internal fun CFG.terminalCompletionPlan(tokens: List<Σᐩ>, prefixCFG: CFG = prefixClosure): TerminalCompletionPlan? {
  val partial = tokens.lastOrNull() ?: return null
  val lexicalCandidates = terminals.filter { it.startsWith(partial) }.sorted()
  if (lexicalCandidates.isEmpty()) return null
  val exactInputComplete = partial in terminals && tokens in language

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
      includeCompleteInput = terminal == partial || (partial in terminals && terminal.isNonterminalStubIn(this)),
      prefixCFG = prefixCFG
    )
    suffixLengths.firstOrNull()
      ?.let { TerminalCompletionBranch(terminal, candidateTokens, suffixLengths) }
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
      // then spend the remaining enumeration work on the branches that can finish
      // soonest instead of whichever terminal happens to sort first.
      { if (it.terminal == originalPrefix) 0 else 1 },
      { it.suffixLengths.firstOrNull() ?: Int.MAX_VALUE },
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
    if (branch.suffixLengths.firstOrNull() == 0) yield(branch.tokens.joinToString(" "))
    yieldAll(visibleCompletionCFG.enumSuffixes(branch.tokens, branch.suffixLengths))
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

  private data class CachedTerminalCompletion(
    val cfgHash: Int,
    val prefixCFGHash: Int,
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

  internal fun cachedPlan(cfg: CFG, prefixCFG: CFG, tokens: List<Σᐩ>): TerminalCompletionPlan? {
    val cfgHash = cfg.hashCode()
    val prefixCFGHash = prefixCFG.hashCode()
    cachedTerminalCompletion
      ?.takeIf { it.cfgHash == cfgHash && it.prefixCFGHash == prefixCFGHash && it.tokens == tokens }
      ?.let { return it.plan }

    return cfg.terminalCompletionPlan(tokens, prefixCFG).also { plan ->
      cachedTerminalCompletion = CachedTerminalCompletion(cfgHash, prefixCFGHash, tokens.toList(), plan)
    }
  }
}