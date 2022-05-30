package org.jetbrains.plugins.template

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.parse
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.parsing.prettyPrint
import ai.hypergraph.kaliningraph.sat.synthesizeFrom
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFile
import org.jetbrains.concurrency.runAsync

var grammarFileCache: String? = ""
var grammarLastModified = 0L
lateinit var cfg: CFG

fun PsiFile.recomputeGrammar(): Set<Pair<String, List<String>>> =
  if (
    text != grammarFileCache ||
    modificationStamp != grammarLastModified ||
    grammarLastModified == 0L
  ) {
    grammarFileCache = text
    grammarLastModified = modificationStamp
    ReadAction.compute<String, Exception> { grammarFileCache }.parseCFG()
  } else cfg

class TidyKeyHandler : TypedHandlerDelegate() {
  val ok = "✅ Current line parses!\n"
  val no = "❌ Current line invalid\n"

  override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile) =
    CONTINUE.also {
      if (file.name.endsWith(".tidy"))
        reconcile(
          grammarFile = file.getGrammarFile() ?: return@also,
          currentLine = editor.currentLine()
        )
    }


  private fun reconcile(grammarFile: PsiFile, currentLine: String) =
    runAsync {
      cfg = grammarFile.recomputeGrammar()

      var debugText = ""
      if ("_" !in currentLine && Regex("<[^\\s>]*>") !in currentLine) {
        val parse = cfg.parse(currentLine)
        debugText = if (parse != null) ok + parse.prettyPrint()
        else no + TidyToolWindow.textArea.text.drop(ok.length)
      } else synchronized(cfg) {
        currentLine.synthesizeFrom(cfg, " ").take(20).toList().shuffled()
          .let { if (it.isNotEmpty()) debugText = it.joinToString("\n") }
      }

      val delim = List(50) { "─" }.joinToString("", "\n", "\n")
      debugText += delim + "Chomsky normal form:\n${cfg.prettyPrint(3)}"
      TidyToolWindow.textArea.text = debugText
    }
}