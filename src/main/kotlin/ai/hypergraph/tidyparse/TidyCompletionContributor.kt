package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.parsing.everySingleHoleConfig
import ai.hypergraph.kaliningraph.parsing.increasingLengthChunks
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.patterns.PlatformPatterns
import com.intellij.psi.PlainTextTokenTypes
import com.intellij.util.ProcessingContext
import java.util.TreeSet

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

fun String.synthesizeCachingAndDisplayProgress(
  cfg: CFG,
  tokens: List<String> = tokenizeByWhitespace(),
  sanitized: String = tokens.joinToString(" "),
  maxResults: Int = 20,
  variations: List<(String) -> Sequence<String>> =
    listOf(
      String::everySingleHoleConfig,
      String::increasingLengthChunks
    ),
  allowNTs: Boolean = true
) =
  synthCache.getOrPut(sanitized to cfg) {
    val solutions = TreeSet(compareBy<String> {
      levenshtein(tokens.filterNot { "_" in it }, it.tokenizeByWhitespace())
    }.thenBy { it.length })
    sanitized.synthesizeIncrementally(
      cfg = cfg,
      join = " ",
      variations = variations,
      allowNTs = allowNTs,
      cfgFilter = { true },
      progress = {
        if("Progress:" in TidyToolWindow.textArea.text)
          TidyToolWindow.textArea.text = updateProgress(it)
        else
          TidyToolWindow.textArea.text = render(solutions)
      }
    ).runningFold(listOf<String>()) { a, s -> a + s }.map {
      if (it.isNotEmpty()) {
        solutions.add(it.last())
        TidyToolWindow.textArea.text = render(solutions)
      }
      it
    }.take(maxResults).toList().last()

    solutions.toList()
  }

private fun updateProgress(it: String) =
  TidyToolWindow.textArea.text.replace("Progress:.*\n".toRegex(), "Progress: ${it.escapeHTML()}\n")

private fun render(solutions: TreeSet<String>) =
    """
        <html>
        <body style=\"font-family: JetBrains Mono\">
        <pre>Synthesizing...
    """.trimIndent() +
    solutions.joinToString("\n", "\n\n", "\n\n").escapeHTML() +
    """🔍 Progress:
        $delim
        </pre>
        </body>
        </html>
    """.trimIndent()

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
        currentLine.synthesizeCachingAndDisplayProgress(cfg)
          .forEach { result.addElement(LookupElementBuilder.create("\n" + it)) }
      } catch (e: Exception) { e.printStackTrace() }
    }
  }
}