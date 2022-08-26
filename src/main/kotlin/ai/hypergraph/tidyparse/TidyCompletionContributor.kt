package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*
import com.intellij.codeInsight.completion.*
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

      val ntRegex = Regex("<[^\\s>]*>")
      if (selectedText != null && selectedText.matches(ntRegex)) {
        val cfg = originalFile.recomputeGrammar()

        val completions =
          cfg.original.bimap[selectedText.drop(1).dropLast(1)]
            .map { it.joinToString(" ") { if (it in cfg.original.terminals) it else "<$it>" } }
            .sortedWith(compareBy<String> { !it.matches(ntRegex) }.thenBy { it.count { ' ' == it } })

        completions.forEachIndexed { i, it ->
          result.addElement(
            PrioritizedLookupElement.withPriority(
              LookupElementBuilder.create(it),
              -i.toDouble()
            )
          )
        }
      } else {
        synchronized(cfg) {
          try {
            synthCache.get(currentLine.tokenizeByWhitespace().joinToString(" ") to cfg)
              ?.map { it.dehtmlify() }
              ?.forEachIndexed { i, it ->
                result.addElement(
                  PrioritizedLookupElement.withPriority(
                    LookupElementBuilder.create("\n" + it),
                    -i.toDouble()
                  )
                )
              }
          } catch (e: Exception) {
            e.printStackTrace()
          }
        }
      }
    }
  }
}