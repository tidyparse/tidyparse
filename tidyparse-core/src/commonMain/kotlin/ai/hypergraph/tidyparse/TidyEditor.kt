package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*

interface TidyEditor {
  fun readDisplayText(): Σᐩ
  fun readEditorText(): Σᐩ
  fun getCaretPosition(): Int
  fun writeDisplayText(s: Σᐩ)
  fun writeDisplayText(s: (Σᐩ) -> Σᐩ)
  fun getLatestCFG(): CFG
  fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ = l2.joinToString(" ")
  fun getOptimalSynthesizer(cfg: CFG, sanitized: Σᐩ, variations: List<Mutator>): Sequence<Σᐩ>
  fun redecorateLines() {}
}

fun TidyEditor.getGrammarText(): Σᐩ = readEditorText().substringBefore("---")

fun TidyEditor.currentGrammar(): CFG =
  try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

fun TidyEditor.currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()