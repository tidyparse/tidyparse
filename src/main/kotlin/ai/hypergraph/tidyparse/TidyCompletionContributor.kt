package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.allTokensExceptHoles
import ai.hypergraph.kaliningraph.parsing.everySingleHoleConfig
import ai.hypergraph.kaliningraph.parsing.increasingLengthChunks
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.patterns.PlatformPatterns
import com.intellij.psi.PlainTextTokenTypes
import com.intellij.util.ProcessingContext

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
  tokens: List<String> = allTokensExceptHoles(),
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

    sanitized.synthesizeIncrementally(
      cfg = cfg,
      join = " ",
      variations = variations,
      allowNTs = allowNTs,
      cfgFilter = { TODO(); true },
      progress = {
        TidyToolWindow.textArea.text =
          TidyToolWindow.textArea.text.replace("Progress:.*\n".toRegex(), "Progress: ${it.escapeHTML()}\n")
      }
    )
      .runningFold(listOf<String>()) { a, s -> a + s }
      .map {
        TidyToolWindow.textArea.text = """
          <html>
          <body style=\"font-family: JetBrains Mono\">
          <pre>Synthesizing...
          
${it.joinToString("\n").escapeHTML()}
🔍 Progress:
$delim
          </pre>
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
        currentLine.synthesizeCachingAndDisplayProgress(cfg)
          .forEach { result.addElement(LookupElementBuilder.create("\n" + it)) }
      } catch (e: Exception) { e.printStackTrace() }
    }
  }
}