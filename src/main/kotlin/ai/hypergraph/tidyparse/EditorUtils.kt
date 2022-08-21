package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.image.toHtmlTable
import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import ai.hypergraph.kaliningraph.tensor.FreeMatrix
import ai.hypergraph.kaliningraph.types.cache
import ai.hypergraph.kaliningraph.types.isSubsetOf
import com.github.difflib.text.DiffRow.Tag.*
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

private val htmlDiffGenerator: DiffRowGenerator =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean -> "<${if(f) "" else "/"}span>" }
    .build()

fun diffAsHtml(l1: List<String>, l2: List<String>): String =
  htmlDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") {
    when(it.tag) {
      INSERT -> it.newLine.replace("<span>", "<span style=\"background-color: #85FF7A\">")
      CHANGE -> it.newLine.replace("<span>", "<span style=\"background-color: #FFC100\">")
      else -> it.newLine.replace("<span>", "<span style=\"background-color: #FFFF66\">")
    }
  }

private val plaintextDiffGenerator: DiffRowGenerator =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { _ -> "" }
    .oldTag { _ -> "" }
    .build()

// TODO: maybe add a dedicated SAT constraint instead of filtering out NT invariants after?
fun CFG.overrideInvariance(l1: List<String>, l2: List<String>): String =
  plaintextDiffGenerator.generateDiffRows(l1, l2).joinToString(" ") { dr ->
  val (old,new ) = dr.oldLine to dr.newLine
  // If repair substitutes a terminal with the same NT that it belongs to, do not treat as a diff
  if (new.isNonterminal() && preservesNTInvariance(new.treatAsNonterminal(), old)) old else new
}

// Determines whether a substitution is invariant w.r.t. NT membership
private fun CFG.preservesNTInvariance(newNT: String, oldTerminal: String) =
  newNT in bimap[listOf(oldTerminal)]

private val la = "<".escapeHTML()
private val ra = ">".escapeHTML()
private fun String.treatAsNonterminal() = drop(la.length).dropLast(ra.length)
private fun String.isNonterminal() = startsWith(la) && endsWith(ra)

val CFG.prettyHTML by cache { pretty.toString().escapeHTML() }

fun render(solutions: List<String>, prompt: String? = null): String {
  val cnf = "<pre>$delim<b>Chomsky normal form:</b>\n${cfg.prettyHTML}</pre>"
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
): List<String> =
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
      updateProgress = { query ->
        if ("Solving:" in TidyToolWindow.text) updateProgress(query)
        else {
          val htmlEscaped =
            solutions.map { diffAsHtml(tokens, it.tokenizeByWhitespace()) }
          TidyToolWindow.text = render(htmlEscaped, query.escapeHTML())
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

fun List<String>.dittoSummarize() =
  listOf("", *toTypedArray())
    .windowed(2, 1).map { it[0] to it[1] }
    .map { (a, b) -> ditto(a, b) }

fun ditto(s1: String, s2: String): String =
  plaintextDiffGenerator.generateDiffRows(
    s1.tokenizeByWhitespace(),
    s2.tokenizeByWhitespace()
  ).joinToString("") {
    when (it.tag) {
      EQUAL -> ""
      else -> it.newLine + " "
    }
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


fun PsiFile.reconcile(currentLine: String, isInGrammar: Boolean) {
  if (currentLine.isBlank()) return
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
  val cnf = "<pre>$delim<b>Chomsky normal form:</b>\n${cfg.prettyHTML}</pre>"
  debugText += cnf

  TidyToolWindow.text = """
        <html>
        <body style=\"font-family: JetBrains Mono\">
        $debugText
        </body>
        </html>
      """.trimIndent()
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
    .last().sortedBy { it.span.first }.map { it.prettyPrint() }
    .partition { it.contains('─') }
    .let { (trees, stubs) ->
      stubs.distinct().joinToString("  ", "<pre>", "</pre>\n") { it.trim() } +
        trees.let { asts -> if (asts.size % 2 == 1) asts + listOf("") else asts }
          .let { asts -> FreeMatrix(asts.size / 2, 2) { r, c -> asts[r * 2 + c] } }
          .toHtmlTable()
    }

fun String.containsHole(): Boolean =
  "_" in this || Regex("<[^\\s>]*>") in this