package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*

interface TidyEditor {
  fun readDisplayText(): Σᐩ
  fun readEditorText(): Σᐩ
  fun getCaretPosition(): Int
  fun currentLine(): Σᐩ
  fun writeDisplayText(s: Σᐩ)
  fun writeDisplayText(s: (Σᐩ) -> Σᐩ)
  fun getLatestCFG(): CFG
  fun caretInGrammar(): Boolean =
    readEditorText().indexOf("---").let { it == -1 || getCaretPosition() < it }
  fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ = l2.joinToString(" ")
  fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ>
  fun redecorateLines(cfg: CFG) {}
}

fun TidyEditor.getGrammarText(): Σᐩ = readEditorText().substringBefore("---")

fun TidyEditor.currentGrammar(): CFG =
  try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

fun TidyEditor.currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()