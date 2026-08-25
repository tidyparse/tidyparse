import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.Σᐩ
import ai.hypergraph.kaliningraph.parsing.pretty
import ai.hypergraph.kaliningraph.parsing.parseCNF
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.tidyparse.wgpu.MAX_DISP_RESULTS
import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.test.runTest
import org.w3c.dom.HTMLDivElement
import org.w3c.dom.HTMLTextAreaElement
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.TimeMark

class JSTidyCNFEditorTest {
  private val cnf = """
    START -> A B
    A -> begin
    B -> end
  """.trimIndent()

  private class RecordingCNFEditor(
    editor: HTMLTextAreaElement,
    output: HTMLDivElement
  ) : JSTidyCNFEditor(editor, output) {
    var preview: Pair<String, Int>? = null
    var injectedSuffixCandidates: Sequence<Σᐩ>? = null
    val suffixRequests = mutableListOf<Triple<List<Σᐩ>, TerminalCompletionPlan?, Int>>()

    override fun continuation(f: () -> Unit): Any = Unit
    override fun currentDisplayResultLimit(): Int = MAX_DISP_RESULTS

    override fun renderSoftTerminalInsertionPreview(
      insertion: String,
      offset: Int
    ): Boolean {
      preview = insertion to offset
      return true
    }

    override suspend fun suffixCompletionCandidates(
      cfg: CFG,
      tokens: List<Σᐩ>,
      terminalCompletion: TerminalCompletionPlan?,
      limit: Int,
      requestStarted: TimeMark,
      recordGpuTimings: (Map<String, Int>) -> Unit
    ): Sequence<Σᐩ> {
      suffixRequests += Triple(tokens, terminalCompletion, limit)
      return injectedSuffixCandidates ?:
        super.suffixCompletionCandidates(
          cfg, tokens, terminalCompletion, limit, requestStarted, recordGpuTimings
        )
    }
  }

  private fun editorFor(line: String, grammar: String = cnf): RecordingCNFEditor {
    window.asDynamic().cmEditor = null

    val input = (document.getElementById("tidyparse-input") as? HTMLTextAreaElement)
      ?: (document.createElement("textarea") as HTMLTextAreaElement).also {
        it.id = "tidyparse-input"
        document.body?.appendChild(it)
      }
    val output = (document.getElementById("tidyparse-output") as? HTMLDivElement)
      ?: (document.createElement("div") as HTMLDivElement).also {
        it.id = "tidyparse-output"
        document.body?.appendChild(it)
      }

    input.value = line
    input.setSelectionRange(line.length, line.length)
    output.innerHTML = "instructions"

    return RecordingCNFEditor(input, output).also { it.loadCNFFromText(grammar) }
  }

  private fun diverseForwardCNF(aCount: Int, bCount: Int): String = buildString {
    appendLine("P -> p")
    appendLine("A -> a")
    appendLine("B -> b")
    appendLine("M -> mid")
    repeat(aCount) { index ->
      appendLine("START -> P A$index")
      appendLine("A$index -> A AX$index")
      appendLine("AX$index -> ax$index")
    }
    repeat(bCount) { index ->
      appendLine("START -> P B$index")
      appendLine("B$index -> B BT$index")
      appendLine("BT$index -> M BX$index")
      appendLine("BX$index -> bx$index")
    }
  }.trimEnd()

  private fun RecordingCNFEditor.forwardRows(): List<String> =
    readDisplayText().lineSequence()
      .map { it.substringAfter(".) ", missingDelimiterValue = "") }
      .filter(String::isNotEmpty)
      .toList()

  private fun RecordingCNFEditor.handleFreshUserInsertion() {
    recordFreshUserInsertion()
    handleInput()
  }

  private fun RecordingCNFEditor.pressTab() {
    val event = js("""({
      keyCode: 9,
      prevented: false,
      preventDefault: function() { this.prevented = true; }
    })""")
    navUpdate(event)
    assertTrue(event.prevented as Boolean)
  }

  private fun RecordingCNFEditor.pressEnter() {
    val event = js("""({
      keyCode: 13,
      prevented: false,
      preventDefault: function() { this.prevented = true; }
    })""")
    navUpdate(event)
    assertTrue(event.prevented as Boolean)
  }

  @Test
  fun properPrefixProducesForwardCompletionRows() = runTest {
    val editor = editorFor("begin")

    editor.handleInput()
    assertNotNull(editor.runningJob).join()

    val display = editor.readDisplayText()
    assertTrue(display.startsWith("-> Forward completion"), display)
    assertContains(display, "0.) begin end")

    editor.pressEnter()
    assertEquals("begin end", editor.editor.value)
  }

  @Test
  fun completeInputRemainsParseable() = runTest {
    val editor = editorFor("begin end")

    editor.handleInput()
    assertNotNull(editor.runningJob).join()

    val display = editor.readDisplayText()
    assertTrue(display.startsWith("✅ Current line parses"), display)
  }

  @Test
  fun trailingSpaceTransitionsACompleteInputToSuffixCompletion() = runTest {
    val grammar = """
      START -> A B
      START -> a
      A -> a
      B -> b
    """.trimIndent()
    val editor = editorFor("a", grammar)

    editor.handleInput()
    assertNotNull(editor.runningJob).join()
    assertTrue(editor.readDisplayText().startsWith("✅ Current line parses"))

    editor.editor.value = "a "
    editor.editor.setSelectionRange(2, 2)
    editor.handleInput()
    assertNotNull(editor.runningJob).join()

    val display = editor.readDisplayText()
    assertTrue(display.startsWith("-> Forward completion"), display)
    assertContains(display, "a b")
  }

  @Test
  fun invalidInputStillUsesRepairMode() = runTest {
    val editor = editorFor("bogus")

    editor.handleInput()
    assertNotNull(editor.runningJob).join()

    val display = editor.readDisplayText()
    assertTrue(display.startsWith("❌ Current line invalid"), display)
  }

  @Test
  fun partialTerminalPreviewsWithoutMutationAndTabCommits() = runTest {
    val grammar = """
      START -> A R1
      START -> A R2
      A -> begin
      R1 -> B C
      R2 -> B D
      B -> Table
      C -> x
      D -> y
    """.trimIndent()
    val editor = editorFor("begin Tab", grammar)
    val typedText = editor.editor.value
    val typedCaret = editor.editor.selectionStart

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, editor.editor.value)
    assertEquals(typedCaret, editor.editor.selectionStart)
    assertEquals("le " to typedText.length, editor.preview)
    assertEquals("le ", editor.pendingTerminalCompletionInsertion)
    assertContains(editor.readDisplayText(), "begin Table x")
    assertContains(editor.readDisplayText(), "begin Table y")

    editor.pressTab()

    assertEquals("begin Table ", editor.editor.value)
    assertEquals(editor.editor.value.length, editor.editor.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun ambiguousPartialExcludesLexicallyMatchingButNonviableTerminal() = runTest {
    val grammar = """
      START -> A R1
      START -> A R2
      START -> A R3
      START -> O R4
      A -> begin
      O -> other
      R1 -> B W
      R2 -> C X
      R3 -> D Y
      R4 -> E Z
      B -> Table
      C -> Target
      D -> Task
      E -> Taco
      W -> w
      X -> x
      Y -> y
      Z -> z
    """.trimIndent()
    val editor = editorFor("begin T", grammar)
    val typedText = editor.editor.value
    val typedCaret = editor.editor.selectionStart

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, editor.editor.value)
    assertEquals(typedCaret, editor.editor.selectionStart)
    assertEquals("a" to typedText.length, editor.preview)
    assertEquals("a", editor.pendingTerminalCompletionInsertion)
    val display = editor.readDisplayText()
    assertContains(display, "begin Table w")
    assertContains(display, "begin Target x")
    assertContains(display, "begin Task y")
    assertFalse("begin Taco z" in display, display)

    editor.pressTab()

    assertEquals("begin Ta", editor.editor.value)
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun oneSuffixCandidateRequestCoversEveryViableLexicalBranch() = runTest {
    val grammar = """
      START -> Class ClassTail
      START -> Continue Newline
      Class -> class
      ClassTail -> Name Newline
      Name -> NAME
      Continue -> continue
      Newline -> NEWLINE
    """.trimIndent()
    val editor = editorFor("c", grammar).apply {
      injectedSuffixCandidates = sequenceOf(
        "class NAME NEWLINE",
        "continue NEWLINE"
      )
    }

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    val (tokens, completion, limit) = editor.suffixRequests.single()
    assertEquals(listOf("c"), tokens)
    assertEquals(setOf("class", "continue"), assertNotNull(completion).branches.map { it.terminal }.toSet())
    assertEquals(MAX_DISP_RESULTS, limit)
    assertEquals(
      setOf("class NAME NEWLINE", "continue NEWLINE"),
      editor.forwardRows().toSet()
    )
  }

  @Test
  fun grammarReloadDoesNotConsumeAStaleFreshInsertion() = runTest {
    val editor = editorFor("begin Tab")
    editor.recordFreshUserInsertion()

    editor.loadCNFFromText("""
      START -> A R
      A -> begin
      R -> B C
      B -> Table
      C -> end
    """.trimIndent())
    editor.handleInput()
    assertNotNull(editor.runningJob).join()

    assertEquals("begin Tab", editor.editor.value)
    assertNull(editor.preview)
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun pythonFinalTerminalCanBeSoftCompleted() = runTest {
    val editor = editorFor(
      line = "class NAME : ... ; NEW",
      grammar = pythonStatementCNFAllProds.joinToString("\n") { it.pretty() }
    )

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals("LINE " to editor.editor.value.length, editor.preview)
    assertEquals("LINE ", editor.pendingTerminalCompletionInsertion)
    assertContains(editor.readDisplayText(), "class ... ; NEWLINE")

    editor.pressTab()

    assertEquals("class NAME : ... ; NEWLINE ", editor.editor.value)
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun pythonStatementWithoutFinalNewlineHasAForwardCompletion() = runTest {
    val prefix = "def NAME ( ) : + True @ NAME ( )"
    val editor = editorFor(
      line = "$prefix NEWLINE",
      grammar = pythonStatementCNFAllProds.joinToString("\n") { it.pretty() }
    )

    editor.handleInput()
    assertNotNull(editor.runningJob).join()
    assertTrue(editor.readDisplayText().startsWith("✅ Current line parses"))

    editor.editor.value = "$prefix "
    editor.editor.setSelectionRange(editor.editor.value.length, editor.editor.value.length)
    editor.handleInput()
    assertNotNull(editor.runningJob).join()

    val display = editor.readDisplayText()
    assertTrue(display.startsWith("-> Forward completion"), display)
    assertContains(display, "def ... ( ) NEWLINE")
  }

  @Test
  fun forwardRowsAreBalancedAcrossNextTokens() = runTest {
    val editor = editorFor("p ", diverseForwardCNF(aCount = 20, bCount = 20))

    editor.handleInput()
    assertNotNull(editor.runningJob).join()

    val rows = editor.forwardRows()
    assertEquals(MAX_DISP_RESULTS, rows.size)
    assertEquals(
      listOf(14, 15),
      listOf(
        rows.count { it.substringAfterLast(' ').startsWith("ax") },
        rows.count { it.substringAfterLast(' ').startsWith("bx") }
      ).sorted()
    )
    assertTrue(rows.all { it.startsWith("p ") && " ... " !in it })
  }

  @Test
  fun diverseSuffixesRedistributeAndStopAtTheLanguageSize() {
    val zeroLengthBranch = """
      START -> p
      START -> P A
      P -> p
      A -> a
    """.trimIndent().parseCNF().enumDiverseSuffixes(emptyList()).toList()
    assertEquals(listOf("p", "p a"), zeroLengthBranch)

    val redistributed = diverseForwardCNF(aCount = 1, bCount = 40)
      .parseCNF().enumDiverseSuffixes(listOf("p")).toList()
    assertEquals(MAX_DISP_RESULTS, redistributed.size)
    assertEquals(1, redistributed.count { it.startsWith("p a ") })
    assertEquals(MAX_DISP_RESULTS - 1, redistributed.count { it.startsWith("p b ") })

    val finite = diverseForwardCNF(aCount = 2, bCount = 3)
      .parseCNF().enumDiverseSuffixes(listOf("p")).toList()
    assertEquals(5, finite.size)
    assertEquals(5, finite.toSet().size)
    assertEquals(2, finite.count { it.startsWith("p a ") })
    assertEquals(3, finite.count { it.startsWith("p b ") })
  }
}
