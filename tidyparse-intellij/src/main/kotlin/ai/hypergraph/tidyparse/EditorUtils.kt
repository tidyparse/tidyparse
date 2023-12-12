package ai.hypergraph.tidyparse

import java.lang.System.currentTimeMillis
import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import bijectiveRepair
import com.github.difflib.text.DiffRow.Tag.*
import com.github.difflib.text.DiffRowGenerator
import com.intellij.openapi.application.*
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.markup.*
import com.intellij.openapi.editor.markup.HighlighterTargetArea.EXACT_RANGE
import com.intellij.openapi.util.*
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiFile
import com.intellij.ui.JBColor
import com.intellij.util.concurrency.AppExecutorUtil
import com.jetbrains.rd.util.concurrentMapOf
import java.awt.Color
import java.lang.management.ManagementFactory
import java.util.concurrent.*
import kotlin.math.abs
import kotlin.time.*

var mostRecentQuery: String = ""
var promise: Future<*>? = null

// TODO: Do not re-compute all work on each keystroke, cache prior results
fun IJTidyEditor.handle(): Future<*>? {
  val currentLine = currentLine()
  if (psiFile.name.endsWith(".tidy") && currentLine != mostRecentQuery) {
    mostRecentQuery = currentLine
    promise?.cancel(true)
    TidyToolWindow.text = ""
    promise = AppExecutorUtil.getAppExecutorService().submit {
      try { handleInput() } catch (e: Exception) { e.printStackTrace() }
    }

    ToolWindowManager.getInstance(editor.project!!).getToolWindow("Tidyparse")
      ?.let { runReadAction { if (!it.isVisible) it.show() } }
  }
  return promise
}

fun Editor.currentLine(): String =
  caretModel.let { it.visualLineStart to it.visualLineEnd }
    .let { (lineStart, lineEnd) ->
      document.getText(TextRange.create(lineStart, lineEnd))
    }

val htmlDiffGenerator: DiffRowGenerator by lazy {
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean -> "<${if (f) "" else "/"}span>" }
    .oldTag { _ -> "" }
    .build()
}

val plaintextDiffGenerator: DiffRowGenerator by lazy {
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { _ -> "" }
    .oldTag { _ -> "" }
    .build()
}

// Takes a sequence of whitespace-delimited strings and filters for unique edit fingerprints
// from the original string. For example, if the original string is "a b c d" and the sequence
// emits "a b Q d" and "a b F d" then only the first is retained and the second is discarded.
fun Sequence<String>.retainOnlySamplesWithDistinctEditSignature(
  originalString: String, abstractTerminal: (String) -> String = { it }
) = distinctBy { computeAbstractEditSignature(originalString, it, abstractTerminal) }

fun computeAbstractEditSignature(s1: String, s2: String, abstraction: (String) -> String = { it }): String =
  plaintextDiffGenerator.generateDiffRows(s1.tokenizeByWhitespace(), s2.tokenizeByWhitespace())
    .joinToString(" ") {
      when (it.tag) {
        INSERT -> "I.${abstraction(it.newLine.dehtmlify())}"
        DELETE -> ""
        CHANGE -> "C.${abstraction(it.newLine.dehtmlify())}"
        else -> "E"
      }
    }

fun computeEditSignature(s1: String, s2: String): String =
  plaintextDiffGenerator.generateDiffRows(s1.tokenizeByWhitespace(), s2.tokenizeByWhitespace())
    .joinToString(" ") {
      when (it.tag) {
        INSERT -> "I.${it.newLine.cfgType()}"
        DELETE -> ""
        CHANGE -> "C.${it.newLine.cfgType()}"
        else -> "E"
      }
    }

/**
 * Timer for triggering events with a designated delay.
 * May be invoked multiple times inside the delay, but
 * doing so will only prolong the event from firing.
 */

object Trigger : () -> Unit {
  private var delay = 0L
  private var timer = currentTimeMillis()
  private var isRunning = false
  private var invokable: () -> Unit = {}

  override fun invoke() {
    timer = currentTimeMillis()
    if (isRunning) return
    synchronized(this) {
      isRunning = true

      while (currentTimeMillis() - timer <= delay)
        Thread.sleep(abs(delay - (currentTimeMillis() - timer)))

      try {
        invokable()
      } catch (e: Exception) {
        e.printStackTrace()
      }

      isRunning = false
    }
  }

  operator fun invoke(withDelay: Long = 100, event: () -> Unit = {}) {
    delay = withDelay
    invokable = event
    invoke()
  }
}