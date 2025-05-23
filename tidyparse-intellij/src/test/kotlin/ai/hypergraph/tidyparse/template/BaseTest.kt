package ai.hypergraph.tidyparse.template

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.sat.synthesizeIncrementally
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.π2
import ai.hypergraph.tidyparse.*
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.IdeActions
import com.intellij.psi.PsiFile
import com.intellij.testFramework.FileEditorManagerTestCase
import com.intellij.util.ui.UIUtil
import com.jetbrains.rd.util.measureTimeMillis

abstract class BaseTest : FileEditorManagerTestCase() {
  companion object {
    inline fun averageTimeWithWarmup(warmupRuns: Int, timedRuns: Int, action: () -> Long): Long {
      repeat(warmupRuns) { action() }
      var time = 0L
      repeat(timedRuns) { time += action() }
      return time / timedRuns
    }
  }

  override fun tearDown() {
    resetEditor()
    super.tearDown()
  }

  fun takeAction(action: String) = myFixture.performEditorAction(action)
  fun takeAction(action: AnAction) = myFixture.testAction(action)

  fun makeEditor(contents: String): PsiFile =
    myFixture.configureByText(TidyFileType(), contents)

  fun resetEditor() {
    myFixture.editor?.let {
      takeAction(IdeActions.ACTION_EDITOR_ESCAPE)
      UIUtil.dispatchAllInvocationEvents()
    }
    manager?.closeAllFiles()
  }

  fun getCodeCompletionResults(): List<String> {
    myFixture.editor?.let {
      takeAction(IdeActions.ACTION_CODE_COMPLETION)
      UIUtil.dispatchAllInvocationEvents()
    }
    return myFixture.lookupElementStrings ?: emptyList()
  }

  fun typeAndAwaitResults(string: String) {
    myFixture.type(string)
    UIUtil.dispatchAllInvocationEvents()
    promise?.get()
  }

  fun String.simulateKeystroke() = myFixture.run {
    makeEditor(this@simulateKeystroke + "<caret>")
    typeAndAwaitResults(" ")
    if ("---\n" in this@simulateKeystroke) checkCachedResultParses()
//    typeAndAwaitResults("p")
//    checkCachedResultParses()
  }

  private fun String.checkCachedResultParses() {
    val cfg = ijTidyEditor.getLatestCFG().freeze()
    val results = getCodeCompletionResults()
    val lastLine = lines().last()
    if (results.isEmpty() && lastLine.containsHole() &&
      lastLine.tokenizeByWhitespace().all { it in cfg.terminals }
    ) {
      val satResult = lastLine.synthesizeIncrementally(cfg).firstOrNull()
      assertNull("No result produced for: $this\nBut a solution exists: $satResult", satResult)
    }
    results.forEach {
      assertNotNull(
        "Unrecognized: \"$it\" for CFG:\n${cfg.prettyPrint()} in $this",
        cfg.parse(it)
      )
    }
  }

  fun String.invokeOnAllLines() {
    measureTimeMillis {
      lines().fold("") { acc, s ->
        "$acc\n$s".also {
          try {
            it.simulateKeystroke()
          } catch (exception: Exception) {
            println("Repair error ${exception.message} on line: $s")
            throw exception
          }
        }
      }
    }.also { println("Round trip latency: ${it}ms") }
  }
}