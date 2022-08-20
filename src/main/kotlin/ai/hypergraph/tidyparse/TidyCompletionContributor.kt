package ai.hypergraph.tidyparse

import com.intellij.codeInsight.completion.CompletionContributor
import com.intellij.codeInsight.completion.CompletionParameters
import com.intellij.codeInsight.completion.CompletionProvider
import com.intellij.codeInsight.completion.CompletionResultSet
import com.intellij.codeInsight.completion.CompletionType.BASIC
import com.intellij.codeInsight.lookup.LookupElementBuilder
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