package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.parsing.*

// TODO: eliminate this completely
var cfg: CFG = setOf()
var grammarFileCache: String = ""
val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

interface TidyEditor {
  fun readDisplayText(): Σᐩ
  fun readEditorText(): Σᐩ
  fun getCaretPosition(): Int
  fun currentLine(): Σᐩ
  fun writeDisplayText(s: Σᐩ)
  fun writeDisplayText(s: (Σᐩ) -> Σᐩ)
  fun getLatestCFG(): CFG {
    val grammar: String = readEditorText().substringBefore("---")
    return if (grammar != grammarFileCache || cfg.isNotEmpty()) {
      grammarFileCache = grammar
      grammarFileCache.parseCFG().also { cfg = it }
    } else cfg
  }

  fun handleInput() { }
  fun caretInGrammar(): Boolean =
    readEditorText().indexOf("---")
      .let { it == -1 || getCaretPosition() < it }
  fun diffAsHtml(l1: List<Σᐩ>, l2: List<Σᐩ>): Σᐩ = l2.joinToString(" ")
  fun repair(cfg: CFG, str: Σᐩ): List<Σᐩ>
  fun redecorateLines(cfg: CFG) {}
}

fun TidyEditor.getGrammarText(): Σᐩ = readEditorText().substringBefore("---")

fun TidyEditor.currentGrammar(): CFG =
  try { readEditorText().parseCFG() } catch (e: Exception) { setOf() }

fun TidyEditor.currentGrammarIsValid(): Boolean = currentGrammar().isNotEmpty()