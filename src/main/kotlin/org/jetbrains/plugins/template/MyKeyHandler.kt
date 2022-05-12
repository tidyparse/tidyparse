package org.jetbrains.plugins.template

import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.parse
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.sat.synthesizeFromFPSolving
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate
import com.intellij.codeInsight.editorActions.TypedHandlerDelegate.Result.CONTINUE
import com.intellij.openapi.application.ReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import org.jetbrains.concurrency.runAsync

var grammarLastModified = 0L
lateinit var cfg: CFG

fun recomputeGrammar(grammarFile: PsiFile) =
    if (grammarFile.modificationStamp != grammarLastModified || grammarLastModified == 0L) {
        grammarLastModified = grammarFile.modificationStamp
        ReadAction.compute<String, Exception> { grammarFile.text }.parseCFG()
    } else cfg

class MyKeyHandler : TypedHandlerDelegate() {
    val ok = "✅ Current line parses!\n"
    val no = "❌ Current line invalid\n"

    override fun charTyped(c: Char, project: Project, editor: Editor, file: PsiFile): Result {
        if (!file.name.endsWith(".tidy")) return CONTINUE
        val grammarFile = file.getGrammarFile() ?: return CONTINUE
        runAsync {
            cfg = recomputeGrammar(grammarFile)
            val currentLine = editor.currentLine()

            if ("_" !in currentLine) {
                val parse = cfg.parse(currentLine)
                if (parse != null) MyToolWindow.textArea.text = ok + parse.prettyPrint()
                else MyToolWindow.textArea.text = no + MyToolWindow.textArea.text.drop(ok.length)
            } else currentLine.synthesizeFromFPSolving(cfg, " ").take(20).toList().shuffled()
                .let { if (it.isNotEmpty()) MyToolWindow.textArea.text = it.joinToString("\n") }
        }

        return CONTINUE
    }
}