package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import com.github.difflib.text.DiffRow.Tag.*
import com.github.difflib.text.DiffRowGenerator
import com.intellij.openapi.application.*
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.markup.*
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.*
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import com.intellij.ui.JBColor
import com.intellij.util.concurrency.AppExecutorUtil
import java.awt.Color
import java.lang.management.ManagementFactory
import java.util.concurrent.Future
import kotlin.time.*

var mostRecentQuery: String = ""
var promise: Future<*>? = null

class IJTidyEditor(val editor: Editor, val psiFile: PsiFile): TidyEditor {
  override fun readDisplayText(): Σᐩ = TidyToolWindow.text

  override fun writeDisplayText(s: Σᐩ) { TidyToolWindow.text = s }

  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun readEditorText(): Σᐩ = runReadAction { editor.document.text }

  override fun getCaretPosition(): Int = runReadAction { editor.caretModel.offset }

  override fun getLatestCFG(): CFG = psiFile.recomputeGrammar()

  @OptIn(ExperimentalTime::class)
  override fun getOptimalSynthesizer(cfg: CFG, sanitized: Σᐩ, variations: List<Mutator>): Sequence<Σᐩ> =
    sanitized.synthesizeIncrementally(
      cfg = cfg,
      variations = variations,
      takeMoreWhile = TimeSource.Monotonic.markNow().run { { hasTimeLeft() && hasMemoryLeft() } },
      updateProgress = { query ->
        if ("Solving:" in readDisplayText()) updateProgress(query, this)
      }
    ).retainOnlySamplesWithDistinctEditSignature(sanitized)

  @OptIn(ExperimentalTime::class)
  private fun TimeSource.Monotonic.ValueTimeMark.hasTimeLeft() =
    elapsedNow().inWholeMilliseconds < TIMEOUT_MS

  // Cache value every 10 seconds
  private var lastMemCheck = System.currentTimeMillis()
  private fun hasMemoryLeft() =
    !(System.currentTimeMillis().let { it - lastMemCheck > 10000 && true.apply { lastMemCheck = it } } &&
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

  data class Segmentation(val valid: List<Int>, val invalid: List<Int>, val illegal: List<Int>, val line: String)
  fun Σᐩ.illegalWordIndices(cfg: CFG) =
    tokenizeByWhitespace().mapIndexedNotNull { idx: Int, s: Σᐩ -> if (s !in cfg.terminals) idx else null }

  override fun redecorateLines() = invokeLater {
    val editorText = readEditorText()
    val cfgText = editorText.substringBefore("---")
    val cfg = cfgText.parseCFG()
    if (cfg.isEmpty()) return@invokeLater

    val grammarLines = 0..cfgText.lines().size
    val lines = editorText.lines()
    fun String.isEmptyOrGrammarDelim(i: Int) =
      trim().isEmpty() || "---" in trim() || i in grammarLines
    val validAndInvalidTokens =
      lines.mapIndexed { i, line ->
        val tokens = line.tokenizeByWhitespace()
        when {
          line.isEmptyOrGrammarDelim(i) -> emptyList<Int>() to emptyList()
          tokens.any { it.isHoleTokenIn(cfg) } -> emptyList<Int>() to emptyList()
          line in cfg.language -> emptyList<Int>() to emptyList()
          tokens.size < 4 -> emptyList<Int>() to tokens.indices.toList()
          else -> cfg.parseInvalidWithMaximalFragments(line).map { it.span }
            .filter { 2 < (it.last - it.first) }.flatten()
            .let { it to tokens.indices.filterNot { i -> i in it } }
        }.let {
          Segmentation(
            valid=it.first,
            invalid=it.second,
            illegal=line.illegalWordIndices(cfg),
            line=line
          )
        }
      }

    // Add squiggly underlines to lines that are not in the grammar
    val document = editor.document
    val highlightManager = editor.markupModel

    highlightManager.removeAllHighlighters()

    validAndInvalidTokens.forEachIndexed { lineNo, (parseableSubregion, unparseableSubregion, illegalSubregion, line) ->
      if ((unparseableSubregion + parseableSubregion).isNotEmpty()) {
        val lineStart = document.getLineStartOffset(lineNo)
        val lineEnd = document.getLineEndOffset(lineNo)

        parseableSubregion.map { it..it }.mergeContiguousRanges()
          .map { it.charIndicesOfWordsInString(line) }.forEach {
            val range = TextRange(lineStart + it.start, lineStart + it.endInclusive)
            highlightManager.addRangeHighlighter(
              range.startOffset, range.endOffset, 0, greenUnderline, HighlighterTargetArea.EXACT_RANGE
            )
          }

        unparseableSubregion.filter { it !in illegalSubregion }
          .map { it..it }.mergeContiguousRanges()
          .map { it.charIndicesOfWordsInString(line) }.forEach {
            val range = TextRange(lineStart + it.start, lineStart + it.endInclusive)
            highlightManager.addRangeHighlighter(
              range.startOffset, range.endOffset, 0, orangeUnderline, HighlighterTargetArea.EXACT_RANGE
            ).apply {
              errorStripeMarkColor = JBColor.RED
              errorStripeTooltip = "Line not in grammar"
            }
          }

        illegalSubregion.map { it..it }
          .map { it.charIndicesOfWordsInString(line) }.forEach {
            val range = TextRange(lineStart + it.start, lineStart + it.endInclusive)
            highlightManager.addRangeHighlighter(
              range.startOffset, range.endOffset, 0, redUnderline, HighlighterTargetArea.EXACT_RANGE
            )
          }
      }
    }
  }
}

val greenUnderline = TextAttributes().apply {
  effectType = EffectType.WAVE_UNDERSCORE
  effectColor = JBColor.GREEN
}
val orangeUnderline = TextAttributes().apply {
  effectType = EffectType.WAVE_UNDERSCORE
  effectColor = JBColor.ORANGE
}
val redUnderline = TextAttributes().apply {
  effectType = EffectType.WAVE_UNDERSCORE
  effectColor = JBColor.RED
}
operator fun List<IntRange>.contains(i: Int) = any { i in it }

// Takes an IntRange of word indices and a String of words delimited by one or more whitespaces,
// and returns the corresponding IntRange of character indices in the original string.
// For example, if the input is (1..2, "a__bb___ca d e f"), the output is 3..10
fun IntRange.charIndicesOfWordsInString(str: String): IntRange {
  // All tokens, including whitespaces
  val wordTokens = str.split("\\s+".toRegex()).filter { it.isNotEmpty() }
  val whitespaceTokens = str.split("\\S+".toRegex())

  val allTokens = wordTokens.zip(whitespaceTokens)
  val polarity = str.startsWith(wordTokens.first())
  val interwoven = allTokens.flatMap {
    if (polarity) listOf(it.first, it.second)
    else listOf(it.second, it.first)
  }

  val s = start * 2
  val l = last * 2
  val (startIdx, endIdx) = (s) to (l + 1)

  val adjust = if (startIdx == 0) 0 else 1

  val startOffset = interwoven.subList(0, startIdx).sumOf { it.length } + adjust
  val endOffset = interwoven.subList(0, endIdx + 1).sumOf { it.length }
  return startOffset..endOffset
}

fun List<IntRange>.mergeContiguousRanges(): List<IntRange> =
  sortedBy { it.first }.fold(mutableListOf<IntRange>()) { acc, range ->
    if (acc.isEmpty()) acc.add(range)
    else if (acc.last().last + 1 >= range.first) acc[acc.lastIndex] = acc.last().first..range.last
    else acc.add(range)
    acc
  }

// TODO: Do not re-compute all work on each keystroke, cache prior results
fun handle(currentLine: String, project: Project, editor: Editor, file: PsiFile): Future<*>? {
    val tidyEditor = IJTidyEditor(editor, file)

    val sanitized = currentLine.trim().tokenizeByWhitespace().joinToString(" ")
    if (file.name.endsWith(".tidy") && sanitized != mostRecentQuery) {
      mostRecentQuery = sanitized
      promise?.cancel(true)
      TidyToolWindow.text = ""
      promise = AppExecutorUtil.getAppExecutorService().submit {
         val (caretPos, isInGrammar) = runReadAction {
           editor.caretModel.logicalPosition.column to
               editor.document.text.lastIndexOf("---")
                 .let { sepIdx -> if (sepIdx == -1) true else editor.caretModel.offset < sepIdx }
         }
         tidyEditor.tryToReconcile(sanitized, isInGrammar, caretPos)
       }

      ToolWindowManager.getInstance(project).getToolWindow("Tidyparse")
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
  if (new.isNonterminal() && preservesNTInvariance(new.treatAsNonterminal(), old)) old else new
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
  fun String.type() = when {
    isNonterminal() -> "NT/$this"
    all { it.isJavaIdentifierPart() } -> "ID"
    any { it in BRACKETS } -> "BK/$this"
    else -> "OT"
  }

  return plaintextDiffGenerator.generateDiffRows(s1.tokenizeByWhitespace(), s2.tokenizeByWhitespace())
    .joinToString(" ") {
      when (it.tag) {
        INSERT -> "I.${it.newLine.type()}"
        DELETE -> ""
        CHANGE -> "C.${it.newLine.type()}"
        else -> "E"
      }
    }
}