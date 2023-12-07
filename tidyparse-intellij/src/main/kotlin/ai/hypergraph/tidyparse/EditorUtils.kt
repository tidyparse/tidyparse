package ai.hypergraph.tidyparse

import java.lang.System.currentTimeMillis
import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import bijectiveRepair
import com.github.difflib.text.DiffRow.Tag.*
import com.github.difflib.text.DiffRowGenerator
import com.intellij.openapi.application.*
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.markup.*
import com.intellij.openapi.editor.markup.HighlighterTargetArea.EXACT_RANGE
import com.intellij.openapi.util.*
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import com.intellij.ui.JBColor
import com.intellij.util.concurrency.AppExecutorUtil
import com.jetbrains.rd.util.concurrentMapOf
import java.awt.Color
import java.lang.management.ManagementFactory
import java.util.concurrent.*
import kotlin.math.abs
import kotlin.time.*

var mostRecentQuery: String = ""
var promise: Future<*>? = null

class IJTidyEditor(val editor: Editor, val psiFile: PsiFile): TidyEditor {
  override fun readDisplayText(): Σᐩ = TidyToolWindow.text

  override fun writeDisplayText(s: Σᐩ) { TidyToolWindow.text = s }

  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun readEditorText(): Σᐩ = runReadAction { editor.document.text }

  override fun getCaretPosition(): Int = runReadAction { editor.caretModel.offset }

  override fun currentLine(): Σᐩ =
    runReadAction { editor.currentLine() }.tokenizeByWhitespace().joinToString(" ")

  override fun getLatestCFG(): CFG = psiFile.recomputeGrammar()

  @OptIn(ExperimentalTime::class)
  override fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ> {
    val tokens: List<String> = str.tokenizeByWhitespace()
    val sanitized: String = tokens.joinToString(" ")
    println("Sanitized: $sanitized")
    MAX_SAMPLE = 50
    TIMEOUT_MS = 5_000

    val startTime = currentTimeMillis()
    val takeMoreWhile: () -> Boolean =
      TimeSource.Monotonic.markNow().run { { hasTimeLeft() && hasMemoryLeft() && !Thread.interrupted() } }

    val renderedStubs = if (str.containsHole()) null
    else cfg.parseWithStubs(sanitized).second.renderStubs()
    val reason = if (str.containsHole()) null else no
    writeDisplayText(render(cfg, emptyList(), this, stubs = renderedStubs, reason = reason))

    updateProgress(sanitized, this)
    val renderFrequencyMillis = 20
    var lastRenderTime = currentTimeMillis() - renderFrequencyMillis

    val runningRepairs = concurrentMapOf<Σᐩ, Int>()
    var sortedRepairsP = listOf<Σᐩ>()

    fun topRunningRepairs(toTake: Int = MAX_SAMPLE) =
      runningRepairs.entries.sortedWith(
        compareBy<MutableMap.MutableEntry<Σᐩ, Int>> { it.value }
          .thenBy { it.key.tokenizeByWhitespace().size }
      ).take(toTake).map { it.key }

    fun renderUpdates() {
      val sortedRepairs = topRunningRepairs()
      if (currentTimeMillis() - lastRenderTime > renderFrequencyMillis && sortedRepairs != sortedRepairsP) {
        lastRenderTime = currentTimeMillis()
        sortedRepairsP = sortedRepairs
        if (takeMoreWhile()) invokeLater {
          val htmlSolutions = sortedRepairs.renderToHTML(tokens, calculateDiffs = true)
          writeDisplayText(renderLite(htmlSolutions, this, stubs = renderedStubs, reason = reason))
        }
      }
    }

    if ("_" !in tokens)
      (2..maxOf(5, tokens.size))
        .asSequence().takeWhile { takeMoreWhile() }
        .firstNotNullOfOrNull { numEdits ->
          bijectiveRepair(
            promptTokens = tokens.intersperse(),
            deck = cfg.terminals.toList(),
            maxEdits = numEdits.also { println("Using bijective sampler with $it edits") },
            parallelize = true,
            admissibilityFilter = { this in cfg.language },
            takeMoreWhile = { takeMoreWhile() },
            diagnostic = { rep ->
              runningRepairs[rep.result.joinToString(" ")] = levenshtein(tokens, rep.result)
              renderUpdates()
            }
          ).map { it.result.joinToString(" ") }.distinct().toList().ifEmpty { null }
        } ?: emptyList()
    else
      cfg.enumSeqSmart(sanitized.tokenizeByWhitespace())
        .retainOnlySamplesWithDistinctEditSignature(sanitized)
        .takeWhile { takeMoreWhile() }
        .onEach { result ->
          runningRepairs[result] = levenshtein(tokens, result.tokenizeByWhitespace())
          renderUpdates()
        }.toList()

    println("Found ${runningRepairs.size} total repairs in ${currentTimeMillis() - startTime}ms")

    return topRunningRepairs(runningRepairs.size).renderToHTML(tokens, calculateDiffs = true)
  }

  private fun List<Σᐩ>.renderToHTML(tokens: List<String>, calculateDiffs: Boolean = false) =
    sortedWith(displayComparator(tokens)).let { solutions ->
      if (calculateDiffs)
        solutions
        //        .also { it.map { println(diffAsLatex(tokens, it.tokenizeByWhitespace())) }; println() }
        .map { diffAsHtml(tokens, it.tokenizeByWhitespace()) }
      else solutions.map { it.escapeHTML() }
    }

  // Cache value every 10 seconds
  private var lastMemCheck = currentTimeMillis()
  private fun hasMemoryLeft() =
    !(currentTimeMillis().let { it - lastMemCheck > 10000 && true.apply { lastMemCheck = it } } &&
        0.7 < ManagementFactory.getMemoryMXBean().heapMemoryUsage.let { it.used.toDouble() / it.max })

  override fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ =
    htmlDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") {
      when (it.tag) {
        INSERT -> it.newLine.replace("<span>", "<span style=\"background-color: $insertColor\">")
        CHANGE -> it.newLine.replace("<span>", "<span style=\"background-color: $changeColor\">")
        DELETE -> "<span style=\"background-color: $deleteColor\">${List(it.oldLine.length) { " " }.joinToString("")}</span>"
        else -> it.newLine.replace("<span>", "<span style=\"background-color: #FFFF66\">")
      }
    }

  private val segmentationCache = mutableMapOf<Int, Segmentation>()

  fun getOrComputeSegmentations(cfg: CFG, editorText: Σᐩ): List<Segmentation> {
    val lines = editorText.lines()
    val lastGrammarLine = lines.map { it.trim() }.indexOfFirst { it.trim() == "---" }

    fun String.isEmptyOrGrammarDelim(i: Int) = trim().isEmpty() || i in 0..lastGrammarLine

    return lines.mapIndexed { lineNo, line ->
      val key = editor.hashCode() + cfg.hashCode() + line.hashCode()
      if (key in segmentationCache) Segmentation() // This means it was previously highlighted
      else if (line.isEmptyOrGrammarDelim(lineNo)) Segmentation()
      else segmentationCache.computeIfAbsent(key) { Segmentation.build(cfg, editorText) }
    }
  }

  override fun redecorateLines(cfg: CFG) = invokeLater {
    val document = editor.document
    val editorText = document.text
    val highlightManager = editor.markupModel

//    highlightManager.removeAllHighlighters()
//
//    getOrComputeSegmentations(cfg, editorText).forEachIndexed { i, seg ->
//      seg.highlightLine(document.getLineStartOffset(i), highlightManager)
//    }
  }

  fun Segmentation.highlightLine(lineStart: Int, highlightManager: MarkupModel) {
    if ((valid + invalid).isNotEmpty()) {
      parseableRegions.forEach {
        val range = TextRange(lineStart + it.first, lineStart + it.last)
        highlightManager.addRangeHighlighter(range.startOffset, range.endOffset, 0, greenUnderline, EXACT_RANGE)
      }

      unparseableRegions.forEach {
        val range = TextRange(lineStart + it.first, lineStart + it.last)
        highlightManager.addRangeHighlighter(
          range.startOffset, range.endOffset, 0, orangeUnderline, EXACT_RANGE
        ).apply {
          errorStripeMarkColor = JBColor.RED
          errorStripeTooltip = "Line not in grammar"
        }
      }

      illegalRegions.forEach {
        val range = TextRange(lineStart + it.first, lineStart + it.last)
        highlightManager.addRangeHighlighter(
          range.startOffset, range.endOffset, 0, redUnderline, EXACT_RANGE
        )
      }
    }
  }
}

val greenUnderline = TextAttributes().apply {
  effectType = EffectType.WAVE_UNDERSCORE
  effectColor = JBColor.BLUE
}
val orangeUnderline = TextAttributes().apply {
  effectType = EffectType.WAVE_UNDERSCORE
  effectColor = JBColor.ORANGE
}
val redUnderline = TextAttributes().apply {
  effectType = EffectType.WAVE_UNDERSCORE
  effectColor = JBColor.RED
}

// TODO: Do not re-compute all work on each keystroke, cache prior results
fun IJTidyEditor.handle(): Future<*>? {
  val currentLine = currentLine()
  if (psiFile.name.endsWith(".tidy") && currentLine != mostRecentQuery) {
    mostRecentQuery = currentLine
    promise?.cancel(true)
    TidyToolWindow.text = ""
    promise = AppExecutorUtil.getAppExecutorService().submit { tryToReconcile() }

    ToolWindowManager.getInstance(editor.project!!).getToolWindow("Tidyparse")
      ?.let { runReadAction { if (!it.isVisible) it.show() } }
  }
  return promise
}

fun Editor.currentLine(): String =
  caretModel.let { it.visualLineStart to it.visualLineEnd }
    .let { (lineStart, lineEnd) ->
      document.getText(TextRange.create(lineStart, lineEnd))
    }

fun generateColors(n: Int): List<Color> =
  (0 until n).map { i ->
    Color.getHSBColor(i.toFloat() / n.toFloat(), 0.85f, 1.0f)
  }

private val htmlDiffGenerator: DiffRowGenerator by lazy {
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean -> "<${if (f) "" else "/"}span>" }
    .oldTag { _ -> "" }
    .build()
}

private val plaintextDiffGenerator: DiffRowGenerator by lazy {
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { _ -> "" }
    .oldTag { _ -> "" }
    .build()
}

// TODO: maybe add a dedicated SAT constraint instead of filtering out NT invariants after?
private fun CFG.overrideInvariance(l1: List<String>, l2: List<String>): String =
  plaintextDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") { dr ->
  val (old,new ) = dr.oldLine to dr.newLine
  // If repair substitutes a terminal with the same NT that it belongs to, do not treat as a diff
  if (new.isNonterminalStub() && preservesNTInvariance(new.treatAsNonterminal(), old)) old else new
}

fun List<String>.dittoSummarize(): List<String> =
  listOf("", *toTypedArray())
    .windowed(2, 1).map { it[0] to it[1] }
    .map { (a, b) -> ditto(a, b) }

fun ditto(s1: String, s2: String): String =
  plaintextDiffGenerator.generateDiffRows(
    s1.tokenizeByWhitespace(),
    s2.tokenizeByWhitespace()
  ).joinToString("") {
    when (it.tag) {
      EQUAL -> ""
      else -> it.newLine + " "
    }
  }

fun PsiFile.recomputeGrammar(): CFG {
  val grammar: String = runReadAction { text.substringBefore("---") }
  return if (grammar != grammarFileCache || cfg.isNotEmpty()) {
    grammarFileCache = grammar
    grammarFileCache.parseCFG().also { cfg = it }
  } else cfg
}

// Takes a sequence of whitespace-delimited strings and filters for unique edit fingerprints
// from the original string. For example, if the original string is "a b c d" and the sequence
// emits "a b Q d" and "a b F d" then only the first is retained and the second is discarded.
fun Sequence<String>.retainOnlySamplesWithDistinctEditSignature(originalString: String) =
  distinctBy { computeEditSignature(originalString, it) }

fun computeEditSignature(s1: String, s2: String): String {
  return plaintextDiffGenerator.generateDiffRows(s1.tokenizeByWhitespace(), s2.tokenizeByWhitespace())
    .joinToString(" ") {
      when (it.tag) {
        INSERT -> "I.${it.newLine.cfgType()}"
        DELETE -> ""
        CHANGE -> "C.${it.newLine.cfgType()}"
        else -> "E"
      }
    }
}

/**
 * Timer for triggering events with a designated delay.
 * May be invoked multiple times inside the delay, but
 * doing so will only prolong the event from firing.
 */

object Trigger : () -> Unit {
  private var delay = 0L
  private var timer = currentTimeMillis()
  private var isRunning = false
  private var invokable: () -> Unit = {}

  override fun invoke() {
    timer = currentTimeMillis()
    if (isRunning) return
    synchronized(this) {
      isRunning = true

      while (currentTimeMillis() - timer <= delay)
        Thread.sleep(abs(delay - (currentTimeMillis() - timer)))

      try {
        invokable()
      } catch (e: Exception) {
        e.printStackTrace()
      }

      isRunning = false
    }
  }

  operator fun invoke(withDelay: Long = 100, event: () -> Unit = {}) {
    delay = withDelay
    invokable = event
    invoke()
  }
}