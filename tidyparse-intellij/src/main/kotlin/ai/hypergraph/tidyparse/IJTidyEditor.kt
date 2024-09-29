package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import com.intellij.openapi.application.*
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.markup.*
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import com.intellij.ui.JBColor
import com.jetbrains.rd.util.concurrentMapOf
import java.lang.management.ManagementFactory
import kotlin.time.*

/** Compare with [JSTidyEditor] */
class IJTidyEditor(val editor: Editor, val psiFile: PsiFile): TidyEditor() {
  override fun readDisplayText(): Σᐩ = TidyToolWindow.text

  override fun writeDisplayText(s: Σᐩ) { TidyToolWindow.text = s }

  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun readEditorText(): Σᐩ = runReadAction { editor.document.text }

  override fun getCaretPosition(): Int = runReadAction { editor.caretModel.offset }

  override fun currentLine(): Σᐩ =
    runReadAction { editor.currentLine() }.tokenizeByWhitespace().joinToString(" ")

  override fun handleInput() {
    val currentLine = currentLine()
    if (currentLine.isBlank()) return
    val caretInGrammar = caretInGrammar()
    val cfg =
      (if (caretInGrammar)
        CFGCFG(
          names = currentLine.tokenizeByWhitespace()
            .filter { it !in setOf("->", "|") }.toSet()
        )
      else getLatestCFG()).freeze()

    if (cfg.isEmpty()) return

    if (!caretInGrammar) redecorateLines(cfg)

    var debugText: String
    if ("_" in currentLine.tokenizeByWhitespace()) {
      currentLine.synthesizeCachingAndDisplayProgress(this, cfg).take(MAX_SAMPLE).let {
        debugText = "<pre><b>🔍 Found ${it.size}${if(it.size == MAX_SAMPLE)"+" else ""} admissible solutions!</b>\n\n" +
            it.joinToString("\n", "\n", "\n") + "</pre>"
      }
    } else {
      println("Parsing `$currentLine` with stubs!")
      val (parseForest, stubs) = cfg.parseWithStubs(currentLine)
      debugText = if (parseForest.isNotEmpty()) {
        if (parseForest.size == 1) "<pre>$ok\n🌳" + parseForest.first().prettyPrint() + "</pre>"
        else "<pre>$ambig\n🌳" + parseForest.joinToString("\n\n") { it.prettyPrint() } + "</pre>"
      } else {
        val repairs = currentLine.synthesizeCachingAndDisplayProgress(this, cfg)
          .take(MAX_SAMPLE).joinToString("\n", "\n", "\n")
        "<pre>$no" + repairs + "\n$legend</pre>" + stubs.renderStubs()
      }
    }

    // Append the CFG only if parse succeeds
    debugText += cfg.renderedHTML//cfg.renderCFGToHTML(currentLine.tokenizeByWhitespace().toSet())

//  println(cfg.original.graph.toString())
//  println(cfg.original.graph.toDot())
//  println(cfg.graph.toDot().alsoCopy())
//  cfg.graph.A.show()

    writeDisplayText("""
      <html>
      <body>
      $debugText
      </body>
      </html>
    """.trimIndent())
//    .also { it.show() }
  }

  @OptIn(ExperimentalTime::class)
  override fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ> {
    val tokens: List<String> = str.tokenizeByWhitespace()
    val sanitized: String = tokens.joinToString(" ")
    println("Sanitized: $sanitized")
    MAX_SAMPLE = 50
    TIMEOUT_MS = 5_000

    val startTime = System.currentTimeMillis()
    val takeMoreWhile: () -> Boolean =
      TimeSource.Monotonic.markNow().run { { hasTimeLeft() && hasMemoryLeft() && !Thread.interrupted() } }

    val renderedStubs = if (str.containsHole()) null
    else cfg.parseWithStubs(sanitized).second.renderStubs()
    val reason = if (str.containsHole()) null else no
    writeDisplayText(render(cfg, emptyList(), this, stubs = renderedStubs, reason = reason))

    updateProgress(sanitized, this)
    val renderFrequencyMillis = 20
    var lastRenderTime = System.currentTimeMillis() - renderFrequencyMillis

    val runningRepairs = concurrentMapOf<Σᐩ, Int>()
    var sortedRepairsP = listOf<Σᐩ>()

    fun topRunningRepairs(toTake: Int = MAX_SAMPLE) =
      runningRepairs.entries.sortedWith(
        compareBy<MutableMap.MutableEntry<Σᐩ, Int>> { it.value }
          .thenBy { it.key.tokenizeByWhitespace().size }
      ).take(toTake).map { it.key }

    fun renderUpdates() {
      val sortedRepairs = topRunningRepairs()
      if (System.currentTimeMillis() - lastRenderTime > renderFrequencyMillis && sortedRepairs != sortedRepairsP) {
        lastRenderTime = System.currentTimeMillis()
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
        .takeWhile { takeMoreWhile() }
        .filter { it.isNotEmpty() }
        .retainOnlySamplesWithDistinctEditSignature(sanitized) { "${cfg.bimap[listOf(it)].hashCode()}" }
        .onEach { result ->
          runningRepairs[result] = levenshtein(tokens, result.tokenizeByWhitespace())
          renderUpdates()
        }.toList()

    println("Found ${runningRepairs.size} total repairs in ${System.currentTimeMillis() - startTime}ms")

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
  private var lastMemCheck = System.currentTimeMillis()
  private fun hasMemoryLeft() =
    !(System.currentTimeMillis().let { it - lastMemCheck > 10000 && true.apply { lastMemCheck = it } } &&
        0.7 < ManagementFactory.getMemoryMXBean().heapMemoryUsage.let { it.used.toDouble() / it.max })

  override fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ =
    levenshteinAlign(l1, l2).joinToString(" ") { (a, b) ->
      val bs = b.toString().escapeHTML()
      when {
        a == null -> "<span style=\"background-color: $insertColor\">$bs</span>"
        b == null -> "<span style=\"background-color: $deleteColor\">${List(a.length) { " " }.joinToString("")}</span>"
        a != b -> "<span style=\"background-color: $changeColor\">$bs</span>"
        else -> bs
      }
    }

  fun getOrComputeSegmentations(cfg: CFG, editorText: Σᐩ): List<Segmentation> {
    val lines = editorText.lines()
    val lastGrammarLine = lines.map { it.trim() }.indexOfFirst { it.trim() == "---" }

    fun String.isEmptyOrGrammarDelim(i: Int) = trim().isEmpty() || i in 0..lastGrammarLine

    return lines.mapIndexed { lineNo, line ->
      val key = editor.hashCode() + cfg.hashCode() + line.hashCode()
      if (key in segmentationCache) Segmentation() // This means it was previously highlighted
      else if (line.isEmptyOrGrammarDelim(lineNo)) Segmentation()
      else segmentationCache.getOrPut(key) { Segmentation.build(cfg, editorText) }
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
    val greenUnderline = TextAttributes()
      .apply { effectType = EffectType.WAVE_UNDERSCORE; effectColor = JBColor.BLUE }
    val orangeUnderline = TextAttributes()
      .apply { effectType = EffectType.WAVE_UNDERSCORE; effectColor = JBColor.ORANGE }
    val redUnderline = TextAttributes()
      .apply { effectType = EffectType.WAVE_UNDERSCORE; effectColor = JBColor.RED }

    if ((valid + invalid).isNotEmpty()) {
      parseableRegions.forEach {
        val range = TextRange(lineStart + it.first, lineStart + it.last)
        highlightManager.addRangeHighlighter(range.startOffset, range.endOffset, 0, greenUnderline,
          HighlighterTargetArea.EXACT_RANGE
        )
      }

      unparseableRegions.forEach {
        val range = TextRange(lineStart + it.first, lineStart + it.last)
        highlightManager.addRangeHighlighter(
          range.startOffset, range.endOffset, 0, orangeUnderline, HighlighterTargetArea.EXACT_RANGE
        ).apply {
//          errorStripeMarkColor = JBColor.RED
          setErrorStripeMarkColor(JBColor.RED)
          errorStripeTooltip = "Line not in grammar"
        }
      }

      illegalRegions.forEach {
        val range = TextRange(lineStart + it.first, lineStart + it.last)
        highlightManager.addRangeHighlighter(
          range.startOffset, range.endOffset, 0, redUnderline, HighlighterTargetArea.EXACT_RANGE
        )
      }
    }
  }
}