package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.completion.CompletionType.BASIC
import com.intellij.codeInsight.lookup.LookupElement
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.codeInsight.lookup.LookupElementDecorator
import com.intellij.openapi.application.runReadAction
import com.intellij.patterns.PlatformPatterns
import com.intellij.psi.PlainTextTokenTypes.PLAIN_TEXT
import com.intellij.util.ProcessingContext
import com.jetbrains.rd.util.printlnError

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
    if (parameters.originalFile.fileType.defaultExtension != "tidy") return

    val ijEditor = parameters.editor
    var currentLine = runReadAction { ijEditor.currentLine() }.trim()

    val tidyEditor = IJTidyEditor(ijEditor, parameters.originalFile)
    tidyEditor.handle()?.get()

    val column = ijEditor.caretModel.logicalPosition.column

    currentLine = if (!currentLine.containsHole() &&
      tidyEditor.cfg.parse(currentLine) != null &&
      currentLine.getSurroundingNonterminal(column) == null)
      currentLine.getSurroundingToken(column)
        ?.let { currentLine.replaceRange(it.range, "_") } ?: currentLine
    else currentLine

    try { tidyEditor.getLatestCFG() } catch (e: Exception) { return }

    val surroundingNonterminal: MatchResult? = currentLine.getSurroundingNonterminal(column)

    tidyEditor.run {
      if (surroundingNonterminal != null) {
        cfg.originalForm.bimap[surroundingNonterminal.value.drop(1).dropLast(1)].take(50)
          .map { it.joinToString(" ") { if (it in cfg.originalForm.terminals) it else "<$it>" } }
          .sortedWith(compareBy<String> { !it.isNonterminalStub() }.thenBy { it.count { ' ' == it } })
          .forEachIndexed { i, it -> result.addElement(createLookupElement(it, i, surroundingNonterminal)) }
      } else synchronized(cfg) {
        try {
          synthCache[currentLine.sanitized(cfg.terminals) to cfg]?.take(50)
            ?.map { it.dehtmlify().tokenizeByWhitespace().joinToString(" ") }
            ?.forEachIndexed { i, it -> result.addElement(createLookupElement(it, i, null)) }
        } catch (e: Exception) {
          e.printStackTrace()
        }
      }
    }
  }

  private fun String.getSurroundingToken(i: Int): MatchResult? =
    Regex("\\S+").findAll(this).firstOrNull { i in it.range.let { it.first..it.last + 1 } }

  private fun String.getSurroundingNonterminal(i: Int): MatchResult? =
    Regex("<[^\\s>]+>").findAll(this).firstOrNull { i in it.range.let { it.first..it.last + 1 } }

  private fun createLookupElement(it: String, i: Int, selection: MatchResult?): LookupElementDecorator<LookupElement> =
    LookupElementDecorator.withInsertHandler(
      PrioritizedLookupElement.withPriority(LookupElementBuilder.create(it), -i.toDouble())
    ) { context, item ->
      context.editor.caretModel.run {
        if (selection == null)
          context.document.replaceString(visualLineStart, visualLineEnd - 1, item.lookupString)
        else {
          val preDelete = (visualLineStart + selection.range.first)..context.startOffset
          val preDeleteLen= preDelete.last - preDelete.first
          val postDelete =
            context.selectionEndOffset.let {
              (it - preDeleteLen)..(it + visualLineStart + selection.range.last - context.startOffset - preDeleteLen + 1)
            }
          context.document.deleteString(preDelete.first, preDelete.last)
          context.document.deleteString(postDelete.first, postDelete.last)
        }
      }
    }
}