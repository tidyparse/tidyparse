package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.parsing.*
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.util.TextRange
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.util.concurrency.AppExecutorUtil
import java.lang.System.currentTimeMillis
import java.util.concurrent.Future
import kotlin.math.abs

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

// Takes a sequence of whitespace-delimited strings and filters for unique edit fingerprints
// from the original string. For example, if the original string is "a b c d" and the sequence
// emits "a b Q d" and "a b F d" then only the first is retained and the second is discarded.
fun Sequence<String>.retainOnlySamplesWithDistinctEditSignature(
  originalString: String, abstractTerminal: (String) -> String = { it }
) = distinctBy { computeAbstractEditSignature(originalString, it, abstractTerminal) }

fun computeAbstractEditSignature(s1: String, s2: String, abstraction: (String) -> String = { it }): String =
  levenshteinAlign(s1.tokenizeByWhitespace(), s2.tokenizeByWhitespace())
    .joinToString(" ") { (a, b) ->
      when {
        b == null -> ""
        a == null -> "I.${abstraction(b.dehtmlify())}"
        a != b -> "C.${abstraction(b.dehtmlify())}"
        else -> "E"
      }
    }

fun computeEditSignature(s1: String, s2: String): String =
  levenshteinAlign(s1.tokenizeByWhitespace(), s2.tokenizeByWhitespace())
    .joinToString(" ") { (a, b) ->
      when {
        b == null -> ""
        a == null -> "I.${b.cfgType()}"
        a != b -> "C.${b.cfgType()}"
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