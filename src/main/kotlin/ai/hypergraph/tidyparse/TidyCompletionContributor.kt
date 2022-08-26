package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.CFGCFG
import ai.hypergraph.kaliningraph.parsing.bimap
import ai.hypergraph.kaliningraph.parsing.tokenizeByWhitespace
import com.intellij.codeInsight.completion.CompletionContributor
import com.intellij.codeInsight.completion.CompletionParameters
import com.intellij.codeInsight.completion.CompletionProvider
import com.intellij.codeInsight.completion.CompletionResultSet
import com.intellij.codeInsight.completion.CompletionType.BASIC
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.openapi.application.runReadAction
import com.intellij.patterns.PlatformPatterns
import com.intellij.psi.PlainTextTokenTypes.PLAIN_TEXT
import com.intellij.util.ProcessingContext

class TidyCompletionContributor : CompletionContributor() {
  init {
    extend(
      BASIC,
      PlatformPatterns.psiElement(PLAIN_TEXT),
      TidyCompletionProvider()
    )
  }
}

class TidyCompletionProvider : CompletionProvider<CompletionParameters>() {
  override fun addCompletions(
    parameters: CompletionParameters,
    context: ProcessingContext,
    result: CompletionResultSet
  ) {
    parameters.apply {
      val (currentLine, isInGrammar) = runReadAction {
        editor.currentLine() to
          (editor.caretModel.offset < editor.document.text.lastIndexOf("---"))
      }
      handle(
        currentLine,
        editor.project!!,
        editor,
        originalFile
      )

      val selectedText = editor.caretModel.currentCaret.selectedText

      if (selectedText != null && selectedText.matches(Regex("<[^\\s>]*>"))) {
        val cfg = originalFile.recomputeGrammar()

        val terminals =
          cfg.bimap[selectedText.drop(1).dropLast(1)].filter {
            it.joinToString().let { "." !in it && "ε" !in it && it != selectedText }
          }.map { it.joinToString(" ") }
        result.addAllElements(terminals.map { LookupElementBuilder.create(it) })
      } else {
        synchronized(cfg) {
          try {
            synthCache.get(currentLine.tokenizeByWhitespace().joinToString(" ") to cfg)
              ?.forEach { result.addElement(LookupElementBuilder.create("\n" + it)) }
          } catch (e: Exception) {
            e.printStackTrace()
          }
        }
      }
    }
  }
}