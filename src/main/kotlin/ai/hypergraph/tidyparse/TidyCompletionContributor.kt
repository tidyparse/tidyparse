package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*
import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.completion.CompletionType.BASIC
import com.intellij.codeInsight.lookup.LookupElement
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.codeInsight.lookup.LookupElementDecorator
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

      originalFile.recomputeGrammar()
      val selection =
        currentLine.getSurroundingNonterminal(editor.caretModel.logicalPosition.column)?.value

      val ntRegex = Regex("<[^\\s>]*>")
      if (selection != null)
        cfg.originalForm.bimap[selection.drop(1).dropLast(1)]
          .map { it.joinToString(" ") { if (it in cfg.originalForm.terminals) it else "<$it>" } }
          .sortedWith(compareBy<String> { !it.matches(ntRegex) }.thenBy { it.count { ' ' == it } })
          .forEachIndexed { i, it -> result.addElement(createLookupElement(it, i)) }
      else synchronized(cfg) {
        try {
          synthCache.get(currentLine.tokenizeByWhitespace().joinToString(" ") to cfg)?.map { it.dehtmlify() }
            ?.forEachIndexed { i, it -> result.addElement(createLookupElement(it, i)) }
        } catch (e: Exception) {
          e.printStackTrace()
        }
      }
    }
  }

  private fun String.getSurroundingNonterminal(i: Int): MatchResult? =
    Regex("<[^\\s>]*>").findAll(this).firstOrNull { i in it.range }

  private fun createLookupElement(it: String, i: Int): LookupElementDecorator<LookupElement> =
    LookupElementDecorator.withDelegateInsertHandler(
      PrioritizedLookupElement.withPriority(LookupElementBuilder.create(it), -i.toDouble())
    ) { context, item ->
      val selection = context.editor.currentLine()
        .getSurroundingNonterminal(context.editor.caretModel.logicalPosition.column)

      val (startReplace, endReplace) =
        context.editor.caretModel.run {
          selection?.range?.run { first to last } ?: (visualLineStart to visualLineEnd - 1)
        }

      context.document.replaceString(startReplace, endReplace, item.lookupString)
    }
}