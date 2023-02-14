package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.* // TODO: Why is this not available?
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.cache
import ai.hypergraph.kaliningraph.parsing.prettyPrint

fun helper() = "helper"

fun CFG.renderCFGToHTML(): String =
  (listOf(originalForm.summarize("Original form")) +
      (if (originalForm == nonparametricForm) listOf()
      else listOf(nonparametricForm.summarize("Nonparametric form"))) +
      listOf(summarize("Normal form"))).let { rewriteSummary ->
      val maxLen = rewriteSummary.joinToString("\n").lines().maxOf { it.length }
      rewriteSummary.joinToString(delim(maxLen), "<pre>${delim(maxLen)}", "</pre>")
    }

fun CFG.summarize(name: String): String = "<b>$name</b> (" +
    "${nonterminals.size} nonterminal${if (1 < nonterminals.size) "s" else ""} / " +
    "${terminals.size} terminal${if (1 < terminals.size) "s" else ""} / " +
    "$size production${if (1 < size) "s" else ""})\n${prettyHTML}"

fun delim(len: Int = 120) = List(len) { "─" }.joinToString("", "\n", "\n")

val CFG.prettyHTML by cache { prettyPrint().carveSeams().escapeHTML() }

// Determines whether a substitution is invariant w.r.t. NT membership
fun CFG.preservesNTInvariance(newNT: String, oldTerminal: String) =
  newNT in bimap[listOf(oldTerminal)]

val la = "<".escapeHTML()
val ra = ">".escapeHTML()
fun String.treatAsNonterminal() = drop(la.length).dropLast(ra.length)
fun String.isNonterminal() = startsWith(la) && endsWith(ra)

fun String.dehtmlify(): String =
  replace("&lt;", "<").replace("&gt;", ">")