package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.automata.GRE
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
import kotlin.collections.plusAssign
import kotlin.math.absoluteValue
import kotlin.time.*

/** Compare with [JSTidyEditor] */
class IJTidyEditor(val editor: Editor, val psiFile: PsiFile): TidyEditor() {
  override fun readDisplayText(): Σᐩ = TidyToolWindow.text

  override fun writeDisplayText(s: Σᐩ) { TidyToolWindow.text = s }

  override fun writeDisplayText(s: (Σᐩ) -> Σᐩ) = writeDisplayText(s(readDisplayText()))

  override fun readEditorText(): Σᐩ = runReadAction { editor.document.text }

  override fun getCaretPosition() = runReadAction { editor.caretModel.offset.let { it..it } }

  override fun currentLine(): Σᐩ =
    runReadAction { editor.currentLine() }.tokenizeByWhitespace().joinToString(" ")

  fun Σᐩ.synthesizeCachingAndDisplayProgress(tidyEditor: TidyEditor, cfg: CFG): List<Σᐩ> {
    val sanitized: Σᐩ = tokenizeByWhitespace().joinToString(" ") { if (it in cfg.terminals) it else "_" }

    val cacheResultOn: Pair<Σᐩ, CFG> = sanitized to cfg

    val cached = synthCache[cacheResultOn]

    return if (cached?.isNotEmpty() == true) cached
    // Cache miss could be due to prior timeout or cold cache. Either way, we need to recompute
    else tidyEditor.repair(cfg, this).also { synthCache.put(cacheResultOn, it) }
  }

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

    var debugText: String =
      if ("_" in currentLine.tokenizeByWhitespace()) {
        currentLine.synthesizeCachingAndDisplayProgress(this, cfg).take(MAX_SAMPLE).let {
          "<pre><b>🔍 Found ${it.size}${if(it.size == MAX_SAMPLE)"+" else ""} admissible solutions!</b>\n\n" +
              it.joinToString("\n", "\n", "\n") + "</pre>"
        }
      } else {
        println("Parsing `$currentLine` with stubs!")
        val (parseForest, stubs) = cfg.parseWithStubs(currentLine)
        if (parseForest.isNotEmpty()) {
          if (parseForest.size == 1) "<pre>$ok\n🌳" + parseForest.first().prettyPrint() + "</pre>"
          else "<pre>$ambig\n🌳" + parseForest.joinToString("\n\n") { it.prettyPrint() } + "</pre>"
        } else currentLine.synthesizeCachingAndDisplayProgress(this, cfg).take(MAX_SAMPLE).let {
          "<pre><b>🔍 Found ${it.size}${if(it.size == MAX_SAMPLE)"+" else ""} admissible solutions!</b>\n\n" +
            it.joinToString("\n", "\n", "\n") + "</pre>"
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
    TIMEOUT_MS = 1_000

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

    (if ("_" !in tokens) initiateSuspendableRepair(tokens, cfg)
    else cfg.enumSeqSmart(sanitized.tokenizeByWhitespace()))
        .takeWhile { takeMoreWhile() }
        .filter { it.isNotEmpty() }
//        .retainOnlySamplesWithDistinctEditSignature(sanitized) { "${cfg.bimap[listOf(it)].hashCode()}" }
        .onEach { result ->
          runningRepairs[result] = levenshtein(tokens, result.tokenizeByWhitespace())
          renderUpdates()
        }.toList()

    println("Found ${runningRepairs.size} total repairs in ${System.currentTimeMillis() - startTime}ms")

    return topRunningRepairs(runningRepairs.size).renderToHTML(tokens, calculateDiffs = true)
  }

  fun initiateSuspendableRepair(brokenStr: List<Σᐩ>, cfg: CFG): Sequence<Σᐩ> {
    var i = 0
    val upperBound = MAX_RADIUS * 3
//  val monoEditBounds = cfg.maxParsableFragmentB(brokenStr, pad = upperBound)
    val timer = TimeSource.Monotonic.markNow()
    val bindex = cfg.bindex
    val width = cfg.nonterminals.size
    val vindex = cfg.vindex
    val ups = cfg.unitProductions
    val t2vs = cfg.tmToVidx
    val maxBranch = vindex.maxOf { it.size }
    val startIdx = bindex[START_SYMBOL]

    fun nonemptyLevInt(levFSA: FSA): Int? {
      val ap: List<List<List<Int>?>> = levFSA.allPairs
      val dp = Array(levFSA.numStates) { Array(levFSA.numStates) { BooleanArray(width) { false } } }

      levFSA.allIndexedTxs0(ups, bindex).forEach { (q0, nt, q1) -> dp[q0][q1][nt] = true }
      var minRad: Int = Int.MAX_VALUE

      // For pairs (p,q) in topological order
      for (dist: Int in 1 until dp.size) {
        for (iP: Int in 0 until dp.size - dist) {
          val p = iP
          val q = iP + dist
          if (ap[p][q] == null) continue
          val appq = ap[p][q]!!
          for ((A: Int, indexArray: IntArray) in vindex.withIndex()) {
            outerloop@for(j: Int in 0..<indexArray.size step 2) {
              val B = indexArray[j]
              val C = indexArray[j + 1]
              for (r in appq)
                if (dp[p][r][B] && dp[r][q][C]) {
                  dp[p][q][A] = true
                  break@outerloop
                }
            }

            if (p == 0 && A == startIdx && q in levFSA.finalIdxs && dp[p][q][A]) {
              val (x, y) = levFSA.idsToCoords[q]!!
              /** See final state conditions for [makeExactLevCFL] */
              // The minimum radius such that this final state is included in the L-FSA
              minRad = minOf(minRad, (brokenStr.size - x + y).absoluteValue)
            }
          }
        }
      }

      return if (minRad == Int.MAX_VALUE) null else minRad
    }

    val led = (3 until upperBound)
      .firstNotNullOfOrNull { nonemptyLevInt(makeLevFSA(brokenStr, it)) } ?:
    upperBound.also { println("Hit upper bound") }
    val radius = led + LED_BUFFER

    println("Identified LED=$led, radius=$radius in ${timer.elapsedNow()}")

    val levFSA = makeLevFSA(brokenStr, radius)

    val nStates = levFSA.numStates
    val tml = cfg.tmLst
    val tms = tml.size
    val tmm = cfg.tmMap

    // 1) Create dp array of parse trees
    val dp: Array<Array<Array<GRE?>>> = Array(nStates) { Array(nStates) { Array(width) { null } } }

    // 2) Initialize terminal productions A -> a
    val aitx = levFSA.allIndexedTxs1(ups)
    for ((p, σ, q) in aitx) for (Aidx in t2vs[tmm[σ]!!])
      dp[p][q][Aidx] = ((dp[p][q][Aidx] as? GRE.SET) ?: GRE.SET(tms))
        .apply { s.set(tmm[σ]!!)/*; dq[p][q].set(Aidx)*/ }

    var maxChildren = 0
    var location = -1 to -1

    // 3) CYK + Floyd Warshall parsing
    for (dist in 1 until nStates) {
      for (p in 0 until (nStates - dist)) {
        val q = p + dist
        if (levFSA.allPairs[p][q] == null) continue
        val appq = levFSA.allPairs[p][q]!!

        for ((Aidx, indexArray) in vindex.withIndex()) {
          //      println("${cfg.bindex[Aidx]}(${pm!!.ntLengthBounds[Aidx]}):${levFSA.stateLst[p]}-${levFSA.stateLst[q]}(${levFSA.SPLP(p, q)})")
          val rhsPairs = dp[p][q][Aidx]?.let { mutableListOf(it) } ?: mutableListOf()
          outerLoop@for (j in 0..<indexArray.size step 2) {
            val Bidx = indexArray[j]
            val Cidx = indexArray[j + 1]
            for (r in appq) {
              val left = dp[p][r][Bidx]
              if (left == null) continue
              val right = dp[r][q][Cidx]
              if (right == null) continue
              // Found a parse for A
              rhsPairs += left * right
              //            if (rhsPairs.size > 10) break@outerLoop
            }
          }

          val list = rhsPairs.toTypedArray()
          if (rhsPairs.isNotEmpty()) {
            if (list.size > maxChildren) {
              maxChildren = list.size
              location = p to q
            }
            dp[p][q][Aidx] = GRE.CUP(*list)
          }
        }
      }
    }

    println("Completed parse matrix in: ${timer.elapsedNow()}")

    // 4) Gather final parse trees from dp[0][f][startIdx], for all final states f
    val allParses = levFSA.finalIdxs.mapNotNull { q -> dp[0][q][startIdx] }

    fun TimeSource.Monotonic.ValueTimeMark.hasTimeLeft() =
      elapsedNow().inWholeMilliseconds < TIMEOUT_MS

    val clock = TimeSource.Monotonic.markNow()
    // 5) Combine them under a single GRE
    return (
      if (allParses.isEmpty()) sequenceOf()
      else GRE.CUP(*allParses.toTypedArray()).let {
        it.words(tml) { clock.hasTimeLeft() }
//    if (ngrams == null) it.words(tml) { clock.hasTimeLeft() }
//    else it.wordsOrdered(tml, ngrams) { clock.hasTimeLeft() }
      }
    ).also { println("Parsing took ${timer.elapsedNow()} with |σ|=${brokenStr.size}, " +
      "|Q|=$nStates, |G|=${cfg.size}, maxBranch=$maxBranch, |V|=$width, |Σ|=$tms, maxChildren=$maxChildren@$location")
    }
  }

  private fun List<Σᐩ>.renderToHTML(tokens: List<String>, calculateDiffs: Boolean = false) =
    sortedWith(displayComparator(tokens)).let { solutions ->
      if (calculateDiffs)
        solutions
          //        .also { it.map { println(diffAsLatex(tokens, it.tokenizeByWhitespace())) }; println() }
          .map { diffAsHtml(tokens, it.tokenizeByWhitespace()) }
      else solutions.map { it.escapeHTML().also { println("HTML: $it") } }
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