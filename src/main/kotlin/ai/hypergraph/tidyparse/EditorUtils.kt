package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.image.toHtmlTable
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.isSubsetOf
import com.github.difflib.text.DiffRow
import com.github.difflib.text.DiffRow.Tag.CHANGE
import com.github.difflib.text.DiffRow.Tag.INSERT
import com.github.difflib.text.DiffRowGenerator
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import com.intellij.util.concurrency.AppExecutorUtil
import java.awt.Color
import java.util.*
import java.util.concurrent.Future

fun Editor.currentLine(): String =
  caretModel.let { it.visualLineStart to it.visualLineEnd }
    .let { (lineStart, lineEnd) ->
      document.getText(TextRange.create(lineStart, lineEnd))
    }

val AnActionEvent.editor: Editor get() =
  CommonDataKeys.EDITOR.getData(dataContext)!!

fun generateColors(n: Int): List<Color> =
  (0 until n).map { i ->
    Color.getHSBColor(i.toFloat() / n.toFloat(), 0.85f, 1.0f)
  }

val generator =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean? -> "</span>" }
    .build()

fun diffAsHtml(l1: List<String>, l2: List<String>): String =
  generator.generateDiffRows(l1, l2).joinToString(" ") {
    when(it.tag) {
      INSERT -> it.newLine.replaceFirst("</span>", "<span style=\"background-color: #85FF7A\">")
      CHANGE -> it.newLine.replaceFirst("</span>", "<span style=\"background-color: #FFC100\">")
      else -> it.newLine.replaceFirst("</span>", "<span style=\"background-color: #FFFF66\">")
    }
  }

val generator2 =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean? -> "" }
    .build()

// TODO: maybe add a dedicated SAT constraint instead of filtering out NT invariants after?
fun CFG.overrideInvariance(l1: List<String>, l2: List<String>): String =
  generator2.generateDiffRows(l1, l2).joinToString(" ") { dr ->
  val (old,new ) = dr.oldLine to dr.newLine
  // If repair substitutes a terminal with the same NT that it belongs to, do not treat as a diff
  if (new.isNonterminal() && preservesNTInvariance(new.treatAsNonterminal(), old)) old
  else new
}

// Determines whether a substitution is invariant w.r.t. NT membership
private fun CFG.preservesNTInvariance(newNT: String, oldTerminal: String) =
  newNT in bimap[listOf(oldTerminal)]

fun String.treatAsNonterminal() = drop(1).dropLast(1)
fun String.isNonterminal() = startsWith('<') && endsWith('>')

fun render(solutions: List<String>, prompt: String? = null): String {
  val cnf = "<pre>$delim<b>Chomsky normal form:</b></pre>\n${cfg.pretty.map { it.escapeHTML() }.toHtmlTable()}"
  return """
    <html>
    <body style=\"font-family: JetBrains Mono\">
    <pre>Synthesizing...
    """.trimIndent() +
    solutions.joinToString("\n", "\n\n", "\n\n") +
    """🔍 Solving: ${
      prompt ?: TidyToolWindow.text.substringAfter("Solving: ").substringBefore("\n")
    }
        </pre>
        $cnf
        </body>
        </html>
    """.trimIndent()
}

val synthCache = LRUCache<Pair<String, CFG>, List<String>>()

fun String.synthesizeCachingAndDisplayProgress(
  cfg: CFG,
  tokens: List<String> = tokenizeByWhitespace(),
  sanitized: String = tokens.joinToString(" "),
  maxResults: Int = 20,
  variations: List<(String) -> Sequence<String>> =
    listOf(
      String::everySingleHoleConfig,
      String::increasingLengthChunks
    ),
  allowNTs: Boolean = true
) =
  synthCache.getOrPut(sanitized to cfg) {
    val solutions = TreeSet(compareBy<String> {
      levenshtein(tokens.filterNot { "_" in it }, it.tokenizeByWhitespace())
    }.thenBy { it.length })
    sanitized.synthesizeIncrementally(
      cfg = cfg,
      join = " ",
      variations = variations,
      allowNTs = allowNTs,
      cfgFilter = { true },
      progress = {
        if ("Solving:" in TidyToolWindow.text) updateProgress(it)
        else {
          val htmlEscaped =
            solutions.map { diffAsHtml(tokens, it.tokenizeByWhitespace()) }
          TidyToolWindow.text = render(htmlEscaped, it.escapeHTML())
        }
      }
    ).map {
      updateSolutions(solutions, cfg, tokens, it)
      val htmlSolutions = if("_" in this) solutions.map { it.escapeHTML() }
      else solutions.map { diffAsHtml(tokens, it.tokenizeByWhitespace()) }
      TidyToolWindow.text = render(htmlSolutions)
    }.takeWhile { solutions.size <= maxResults }.toList()

    solutions.toList()
  }

private fun String.updateSolutions(
  solutions: TreeSet<String>,
  cfg: CFG,
  tokens: List<String>,
  it: String
) =
  if ("_" in this) solutions.add(it)
  else solutions.add(cfg.overrideInvariance(tokens, it.tokenizeByWhitespace()))

fun updateProgress(it: String) {
  TidyToolWindow.text =
    TidyToolWindow.text.replace(
      "Solving:.*\n".toRegex(),
      "Solving: ${it.escapeHTML()}\n"
    )
}

fun Sequence<Tree>.allIndicesInsideParseableRegions(): Set<Int> =
  map { it.span }.filter { 3 < it.last - it.first }
    .flatMap { (it.first + 1) until it.last }.toSet()

var promise: Future<*>? = null

fun PsiFile.reconcile(currentLine: String, isInGrammar: Boolean) =
  promise?.cancel(true).also {
    AppExecutorUtil.getAppExecutorService().submit {
      if (currentLine.isBlank()) return@submit
      val cfg =
        if (isInGrammar)
          CFGCFG(
            names = currentLine.split(Regex("\\s+"))
              .filter { it.isNotBlank() && it !in setOf("->", "|") }.toSet()
          )
        else recomputeGrammar()

      var debugText = ""
      if (currentLine.containsHole()) {
        synchronized(cfg) {
          currentLine.synthesizeCachingAndDisplayProgress(cfg).let {
            debugText = "<pre><b>🔍 Found ${it.size} admissible solutions!</b>\n\n" +
              it.joinToString("\n") { it.escapeHTML() } + "</pre>"
          }
        }
      } else {
        val (parse, stubs) = cfg.parseWithStubs(currentLine)
        debugText = if (parse != null) ok + "<pre>" + parse.prettyPrint() + "</pre>"
        else no + currentLine.findRepairs(cfg, stubs.allIndicesInsideParseableRegions()) + stubs.renderStubs()
      }

      // Append the CFG only if parse succeeds
      val cnf = "<pre>$delim<b>Chomsky normal form:</b></pre>\n${cfg.pretty.map { it.escapeHTML() }.toHtmlTable()}"
      debugText += cnf

      TidyToolWindow.text = """
        <html>
        <body style=\"font-family: JetBrains Mono\">
        $debugText
        </body>
        </html>
      """.trimIndent()
    }.also { promise = it }
  }

fun String.findRepairs(cfg: CFG, exclusions: Set<Int>): String =
  synthesizeCachingAndDisplayProgress(
    cfg = cfg,
    variations = listOf { it.multiTokenSubstitutionsAndInsertions(numberOfEdits = 3, exclusions = exclusions) },
    allowNTs = true
  ).let {
    if (it.isNotEmpty())
      it.joinToString("\n", "<pre>", "\n") {
        diffAsHtml(tokenizeByWhitespace(), it.tokenizeByWhitespace())
      } + "${delim}Partial AST branches:</pre>"
    else ""
  }

fun Sequence<Tree>.renderStubs(): String =
  runningFold(setOf<Tree>()) { acc, t -> if (acc.any { t.span isSubsetOf it.span }) acc else acc + t }
    .last().map { it.prettyPrint() }.partition { it.contains('─') }
    .let { (trees, stubs) ->
      stubs.distinct().joinToString("  ", "<pre>", "</pre>\n") { it.trim() } +
          trees.let { asts -> if (asts.size % 2 == 1) asts + listOf("") else asts }
            .let { asts -> FreeMatrix(asts.size / 2, 2) { r, c -> asts[r * 2 + c] } }
            .toHtmlTable()
    }

fun String.containsHole(): Boolean =
  "_" in this || Regex("<[^\\s>]*>") in this