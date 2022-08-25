package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.containsHole
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
    parameters.run {
      val currentLine = runReadAction { editor.currentLine() }
      if (!currentLine.containsHole()) return@run
      handle(
        currentLine,
        editor.project!!,
        editor,
        originalFile
      )

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