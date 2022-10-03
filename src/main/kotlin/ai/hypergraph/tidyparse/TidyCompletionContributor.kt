package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.containsHole
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tensor.seekFixpoint
import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.completion.CompletionType.BASIC
import com.intellij.codeInsight.lookup.LookupElement
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.codeInsight.lookup.LookupElementDecorator
import com.intellij.openapi.application.invokeAndWaitIfNeeded
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.application.runWriteAction
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
      if (originalFile.fileType.defaultExtension != "tidy") return

      var currentLine = runReadAction { editor.currentLine() }.trim()

      handle(
        currentLine,
        editor.project!!,
        editor,
        originalFile
      )?.get()

      currentLine = if (!currentLine.containsHole() && cfg.parse(currentLine) != null &&
        currentLine.getSurroundingNonterminal(editor.caretModel.logicalPosition.column) == null)
        currentLine.getSurroundingToken(editor.caretModel.logicalPosition.column)
          .let { currentLine.replaceRange(it.range, "_") } else currentLine

      try { originalFile.recomputeGrammar() } catch (e: Exception) { return }

      val selection: MatchResult? =
        currentLine.getSurroundingNonterminal(editor.caretModel.logicalPosition.column)

      val ntRegex = Regex("<[^\\s>]*>")
      if (selection != null) {
        cfg.originalForm.bimap[selection.value.drop(1).dropLast(1)]
          .map { it.joinToString(" ") { if (it in cfg.originalForm.terminals) it else "<$it>" } }
          .sortedWith(compareBy<String> { !it.matches(ntRegex) }.thenBy { it.count { ' ' == it } })
          .forEachIndexed { i, it -> result.addElement(createLookupElement(it, i, selection)) }
      } else synchronized(cfg) {
        try {
          synthCache.get(currentLine.sanitized() to cfg)?.map { it.dehtmlify() }
            ?.forEachIndexed { i, it -> result.addElement(createLookupElement(it, i, null)) }
        } catch (e: Exception) {
          e.printStackTrace()
        }
      }
    }
  }

  private fun String.getSurroundingToken(i: Int): MatchResult =
    Regex("\\S+").findAll(this).first { i in it.range.let { it.first..it.last + 1 } }

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
              (it-preDeleteLen)..(it + (visualLineStart + selection.range.last - context.startOffset - preDeleteLen + 1))
            }
          context.document.deleteString(preDelete.first, preDelete.last)
          context.document.deleteString(postDelete.first, postDelete.last)
        }
      }
    }
}