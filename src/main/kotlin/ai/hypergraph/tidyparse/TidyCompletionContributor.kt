package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.sat.everySingleHoleConfig
import ai.hypergraph.kaliningraph.sat.increasingLengthChunks
import ai.hypergraph.kaliningraph.sat.synthesizeFrom
import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.patterns.PlatformPatterns
import com.intellij.psi.PlainTextTokenTypes
import com.intellij.util.ProcessingContext
import io.ktor.util.*

class TidyCompletionContributor : CompletionContributor() {
  init {
    extend(
      CompletionType.BASIC,
      PlatformPatterns.psiElement(PlainTextTokenTypes.PLAIN_TEXT),
      TidyCompletionProvider()
    )
  }
}

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

fun synth(str: String, cfg: CFG, trim: String = str.trim(), maxResults: Int = 20) =
  synthCache.getOrPut(trim to cfg) {
    trim.synthesizeFrom(cfg, " ",
      variations = listOf(
        { sequenceOf(trim) },
        String::everySingleHoleConfig,
        String::increasingLengthChunks)
    ).runningFold(listOf<String>()){ a, s -> a + s }
      .map {
      TidyToolWindow.textArea.text = """
        <html>
        <body style=\"font-family: JetBrains Mono\">
        <pre>${it.joinToString("\n").escapeHTML()}</pre>
        </body>
        </html>
      """.trimIndent()
      it
    }.take(maxResults).toList().last()
  }

class TidyCompletionProvider : CompletionProvider<CompletionParameters>() {
  override fun addCompletions(
    parameters: CompletionParameters,
    context: ProcessingContext,
    result: CompletionResultSet
  ) {
    cfg = parameters.originalFile.recomputeGrammar()
    var currentLine = parameters.editor.currentLine()
    currentLine = if ("_" in currentLine) currentLine else {
      val colIdx = parameters.editor.caretModel.currentCaret.visualPosition.column
      currentLine.substring(0, colIdx) + "_" + currentLine.substring(colIdx, currentLine.length)
    }

    synchronized(cfg) {
      try {
        synth(currentLine, cfg).forEach { result.addElement(LookupElementBuilder.create(it)) }
      } catch (e: Exception) { e.printStackTrace() }
    }
  }
}