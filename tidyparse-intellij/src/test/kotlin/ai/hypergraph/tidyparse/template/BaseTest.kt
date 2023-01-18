package ai.hypergraph.tidyparse.template

import ai.hypergraph.kaliningraph.parsing.parse
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.types.π2
import ai.hypergraph.tidyparse.*
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.IdeActions
import com.intellij.psi.PsiFile
import com.intellij.testFramework.FileEditorManagerTestCase
import com.intellij.util.ui.UIUtil
import com.jetbrains.rd.util.measureTimeMillis

abstract class BaseTest: FileEditorManagerTestCase() {
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
    myFixture.configureByText(TidyFileType, contents)

  fun resetEditor() {
    myFixture.editor?.let {
      takeAction(IdeActions.ACTION_EDITOR_ESCAPE)
      UIUtil.dispatchAllInvocationEvents()
    }
    manager?.closeAllFiles()
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
    val key = lines().last().sanitized() to substringBefore("---").parseCFG()
    synthCache[key]?.forEach {
//      println("Checking: ${it} (${synthCache[key]?.joinToString(",")})")
      assertNotNull(it.dehtmlify(), key.π2.parse(it.dehtmlify()))
    }
  }

  fun String.testAllLines() {
    measureTimeMillis {
      lines().fold("") { acc, s -> "$acc\n$s".also { it.simulateKeystroke() } }
    }.also { println("Round trip latency: $it") }
  }
}
