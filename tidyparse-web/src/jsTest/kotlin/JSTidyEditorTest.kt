import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.noEpsilon
import ai.hypergraph.kaliningraph.parsing.noEpsilonOrNonterminalStubs
import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.parsing.terminals
import ai.hypergraph.tidyparse.MAX_DISP_RESULTS
import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.test.runTest
import org.w3c.dom.HTMLDivElement
import org.w3c.dom.HTMLTextAreaElement
import org.w3c.dom.events.Event
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNotSame
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class JSTidyEditorTest {
  private val cfg = """
    START -> S
    S -> a
  """.trimIndent().parseCFG()

  private val plWhileGrammar = """
    START -> STM+
    STM+ -> STM+ ; STM | STM
    STM -> ASGN | IFS | while ( BEXP ) { STM+ } | for ( ASGN ; BEXP ; ASGN ) { STM+ } | { STM+ } | { } | return EXP ; | break | continue | STM ; | EXP ;
    ASGN -> LHS = EXP | LHS += EXP | LHS -= EXP | LHS *= EXP | LHS /= EXP | LHS %= EXP
    LHS -> ID | ID . ID | ID [ EXP ] | ( LHS )
    IFS -> ID = if ( BEXP ) { EXP } else { EXP } | if ( BEXP ) { STM+ } else { STM+ } | if ( BEXP ) { STM+ }
    EXP -> ID | NUM | STR | LIT | EXP + EXP | EXP - EXP | EXP * EXP | EXP / EXP | EXP % EXP | ( EXP ) | ID ( ) | ID ( ARGS ) | EXP . ID | EXP [ EXP ] | BEXP
    ARGS -> EXP | EXP , ARGS
    BEXP -> EXP == EXP | EXP != EXP | EXP < EXP | BEXP && BEXP | BEXP or BEXP | ( BEXP ) | ! BEXP | LIT
    LIT -> true | false
  """.trimIndent()

  private val plWhileCfg =
    plWhileGrammar.parseCFG(validate = true).noEpsilon

  private class RecordingEditor(
    editor: HTMLTextAreaElement,
    output: HTMLDivElement,
    override var cfg: CFG,
    private val softPreviewAvailable: Boolean
  ) : JSTidyEditor(editor, output) {
    var writes = 0

    override fun getLatestCFG(): CFG = cfg
    override fun continuation(f: () -> Unit): Any = Unit
    override fun renderSoftTerminalInsertionPreview(
      insertion: String,
      offset: Int
    ): Boolean = softPreviewAvailable
    override fun readDisplayText(): String = output.textContent ?: ""
    override fun writeDisplayText(s: String) {
      writes++
      (output as HTMLDivElement).innerHTML = s
    }
  }

  private fun editorFor(
    line: String,
    softPreviewAvailable: Boolean = true
  ): Pair<RecordingEditor, HTMLTextAreaElement> {
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

    output.innerHTML = "instructions"
    input.value = "START -> S\nS -> a\n---\n$line"
    input.setSelectionRange(input.value.length, input.value.length)
    return RecordingEditor(input, output, cfg, softPreviewAvailable) to input
  }

  private fun RecordingEditor.handleFreshUserInsertion() {
    recordFreshUserInsertion()
    handleInput()
  }

  private fun RecordingEditor.pressTab() {
    val event = js("""({
        keyCode: 9,
        prevented: false,
        preventDefault: function() { this.prevented = true; }
    })""")
    navUpdate(event)
    assertTrue(event.prevented as Boolean)
  }

  @Test
  fun trailingWhitespaceDoesNotRestartTokenWork() {
    val (editor, input) = editorFor("a")

    editor.handleInput()
    val originalJob = assertNotNull(editor.runningJob)

    input.value += " "
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleInput()

    assertSame(originalJob, editor.runningJob)
    originalJob.cancel()
  }

  @Test
  fun trailingWhitespaceDoesNotRewriteUnknownTokenError() {
    val (editor, input) = editorFor("_ unknown")

    editor.handleInput()
    assertEquals(1, editor.writes)

    input.value += "   "
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleInput()

    assertEquals(1, editor.writes)
  }

  @Test
  fun uniqueViableTerminalExpandsCompletely() {
    val cfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("begin", "Tab")))

    assertEquals("Tab", completion.originalPrefix)
    assertEquals("Table", completion.expandedPrefix)
    assertTrue(completion.terminalCommitted)
    assertTrue(completion.forcedContinuation.isEmpty())
    assertEquals(listOf("Table"), completion.branches.map { it.terminal })
    assertEquals(listOf(1), completion.branches.single().suffixLengths)
  }

  @Test
  fun uniqueTerminalAdvancesThroughTheCommonContinuation() {
    val cfg = """
      START -> if ( x ) | if ( y )
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("i")))

    assertEquals("if", completion.expandedPrefix)
    assertEquals(listOf("("), completion.forcedContinuation)
    assertEquals(listOf("if", "("), completion.branches.single().tokens)
    assertEquals(listOf(2), completion.branches.single().suffixLengths)
  }

  @Test
  fun plWhileIfAdvancesThroughItsOpeningParenthesis() {
    listOf(
      plWhileGrammar.parseCFG(validate = true),
      plWhileCfg
    ).forEach { cfg ->
      val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("i")))

      assertEquals("if", completion.expandedPrefix)
      assertEquals(listOf("("), completion.forcedContinuation)
    }
  }

  @Test
  fun plWhileStubPrefixKeepsItsUnchangedLcpAndOnlyViableBranches() {
    val completion = assertNotNull(
      plWhileCfg.terminalCompletionPlan(listOf("while", "(", "<"))
    )
    val lexicalStubs =
      plWhileCfg.terminals.filter { it.startsWith("<") }.sorted()

    assertEquals(lexicalStubs.size, completion.lexicalCandidateCount)
    assertEquals("<", completion.originalPrefix)
    assertEquals("<", completion.expandedPrefix)
    assertFalse(completion.terminalCommitted)
    assertTrue(completion.forcedContinuation.isEmpty())
    assertEquals(
      listOf("<BEXP>", "<EXP>", "<LIT>"),
      completion.branches.map { it.terminal }
    )
    assertTrue(completion.branches.all { it.suffixLengths.isNotEmpty() })
    assertEquals(
      MAX_TERMINAL_COMPLETION_BRANCHES,
      completion.branches.size
    )
  }

  @Test
  fun exactTerminalAndCompleteStubPrefixBothReachForwardCompletion() = runTest {
    val cfg = """
      START -> EXP
      EXP -> ID | ID ( ) | ID < ID | ID . EXP | ID EXP
    """.trimIndent().parseCFG(validate = true).noEpsilon

    val completion = assertNotNull(
      cfg.terminalCompletionPlan(listOf("ID", "<"))
    )
    assertEquals(2, completion.lexicalCandidateCount)
    assertEquals("<", completion.originalPrefix)
    assertEquals("<", completion.expandedPrefix)
    assertFalse(completion.terminalCommitted)
    assertTrue(completion.forcedContinuation.isEmpty())
    assertEquals(
      mapOf("<" to listOf(1), "<EXP>" to listOf(0)),
      completion.branches.associate {
        it.terminal to it.suffixLengths
      }
    )

    val (editor, input) = editorFor("ID <")
    editor.cfg = cfg
    val typedText = input.value
    val typedCaret = input.selectionStart

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals(typedCaret, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    val display = editor.output.textContent ?: ""
    assertTrue(display.startsWith("-> Forward completion"))
    assertContains(display, "ID < ID")
    assertContains(display, "ID <EXP>")
  }

  @Test
  fun completeExactTerminalRetainsACompleteStubPrefixInterpretation() {
    val cfg = """
      START -> ID < | ID EXP
      EXP -> ID
    """.trimIndent().parseCFG(validate = true).noEpsilon

    val completion = assertNotNull(
      cfg.terminalCompletionPlan(listOf("ID", "<"))
    )

    assertFalse(completion.terminalCommitted)
    assertEquals(
      mapOf("<" to listOf(0), "<EXP>" to listOf(0)),
      completion.branches.associate {
        it.terminal to it.suffixLengths
      }
    )
  }

  @Test
  fun ordinaryPartialTerminalStillRequiresAContinuation() {
    val cfg = "START -> done".parseCFG().noEpsilon

    assertNull(cfg.terminalCompletionPlan(listOf("do")))
  }

  @Test
  fun partialStubStillRequiresAContinuationWithoutAnExactTerminalConflict() {
    val cfg = """
      START -> EXP
      EXP -> ID
    """.trimIndent().parseCFG().noEpsilon

    assertNull(cfg.terminalCompletionPlan(listOf("<E")))
  }

  @Test
  fun disabledNonterminalStubsLeaveOnlyTheExactTerminalInterpretation() {
    val cfg = """
      START -> EXP
      EXP -> ID | ID < ID | ID EXP
    """.trimIndent().parseCFG(validate = true)
      .noEpsilonOrNonterminalStubs

    assertFalse("<EXP>" in cfg.terminals)
    val completion = assertNotNull(
      cfg.terminalCompletionPlan(listOf("ID", "<"))
    )
    assertEquals(listOf("<"), completion.branches.map { it.terminal })
    assertTrue(completion.terminalCommitted)
    assertEquals(listOf("ID"), completion.forcedContinuation)
  }

  @Test
  fun branchLimitRetainsExactTerminalAndShortestStubInterpretation() = runTest {
    val cfg = """
      START -> S ;
      S -> ID :: ID < FLOATONE
      S -> ID :: ID ARITHMETIC_OPERATOR ID
      S -> ID :: ID ASSIGNMENT_OPERATOR ID
      S -> ID :: ID POSTFIX_OPERATOR
      ARITHMETIC_OPERATOR -> +
      ASSIGNMENT_OPERATOR -> =
      POSTFIX_OPERATOR -> ++ | --
    """.trimIndent().parseCFG(validate = true).noEpsilon
    val (editor, input) = editorFor("ID :: ID <")
    editor.cfg = cfg
    val typedText = input.value
    val typedCaret = input.selectionStart
    val completion = assertNotNull(
      cfg.terminalCompletionPlan(listOf("ID", "::", "ID", "<"))
    )

    assertEquals(
      listOf(
        "<",
        "<ARITHMETIC_OPERATOR>",
        "<ASSIGNMENT_OPERATOR>",
        "<POSTFIX_OPERATOR>"
      ),
      completion.branches.map { it.terminal }
    )
    assertTrue(
      completion.branches.indexOfFirst {
        it.terminal == "<POSTFIX_OPERATOR>"
      } >= MAX_TERMINAL_COMPLETION_BRANCHES
    )
    assertEquals(
      listOf(2),
      completion.branches
        .first { it.terminal == "<" }
        .suffixLengths
    )
    assertEquals(
      listOf(1),
      completion.branches
        .first { it.terminal == "<POSTFIX_OPERATOR>" }
        .suffixLengths
    )

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals(typedCaret, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    val display = editor.output.textContent ?: ""
    assertContains(display, "ID :: ID < FLOATONE ;")
    assertContains(display, "ID :: ID <POSTFIX_OPERATOR> ;")
  }

  @Test
  fun plWhileStubPrefixImmediatelyEnumeratesBranchesFairlyWithoutAGhost() = runTest {
    val (editor, input) = editorFor("while ( <")
    editor.cfg = plWhileCfg
    val typedText = input.value
    val typedCaret = input.selectionStart

    editor.handleFreshUserInsertion()
    val completionJob = assertNotNull(editor.runningJob)

    // CodeMirror can dispatch more than one post-key notification. Replaying
    // the unchanged state must retain the viable prefix resolution instead of
    // reinterpreting literal "<" as an invalid exact terminal.
    editor.handleInput()
    assertSame(completionJob, editor.runningJob)

    assertEquals(typedText, input.value)
    assertEquals(typedCaret, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)

    completionJob.join()

    assertEquals(typedText, input.value)
    assertEquals(typedCaret, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)

    val displayText = editor.output.textContent ?: ""
    assertTrue(displayText.startsWith("-> Forward completion"))
    val results = displayText.lineSequence()
      .map {
        it.substringAfter(
          delimiter = ".) ",
          missingDelimiterValue = ""
        )
      }
      .filter { it.isNotEmpty() }
      .toList()
    assertEquals(MAX_DISP_RESULTS, results.size)

    val viableStubs = listOf("<BEXP>", "<EXP>", "<LIT>")
    val resultStubs = results.map { result ->
      assertNotNull(
        viableStubs.singleOrNull {
          result.startsWith("while ( $it ")
        },
        "Unexpected terminal-completion branch: $result"
      )
    }
    val counts = viableStubs.associateWith { stub ->
      resultStubs.count { it == stub }
    }

    assertEquals(viableStubs.toSet(), resultStubs.toSet())
    assertEquals(MAX_DISP_RESULTS, counts.values.sum())
    assertEquals(listOf(9, 10, 10), counts.values.sorted())
  }

  @Test
  fun plWhileFreshInsertionOffersIfAndOpeningParenthesisUntilTab() = runTest {
    val (editor, input) = editorFor("i")
    editor.cfg = plWhileGrammar.parseCFG(validate = true)
    val typedText = input.value
    val typedCaret = input.selectionStart

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals(typedCaret, input.selectionStart)
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)
    assertContains(editor.output.textContent ?: "", "if (")

    editor.handleInput()
    assertEquals(typedText, input.value)
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)

    editor.pressTab()
    assertTrue(input.value.endsWith("\nif ( "))
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun plWhileExactClosingParenthesisForcesOpeningBrace() {
    listOf(
      plWhileGrammar.parseCFG(validate = true),
      plWhileCfg
    ).forEach { cfg ->
      val completion = assertNotNull(
        cfg.terminalCompletionPlan(listOf("if", "(", "true", ")"))
      )

      assertEquals(listOf("{"), completion.forcedContinuation)
      assertEquals(
        listOf("if", "(", "true", ")", "{"),
        completion.branches.single().tokens
      )
      assertEquals((2..9).toList(), completion.branches.single().suffixLengths)
    }
  }

  @Test
  fun plWhileFreshClosingParenthesisOffersOpeningBraceUntilTab() = runTest {
    val (editor, input) = editorFor("if ( true ")
    editor.cfg = plWhileGrammar.parseCFG(validate = true)

    input.value += ")"
    input.setSelectionRange(input.value.length, input.value.length)
    val typedText = input.value
    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals(" { ", editor.pendingTerminalCompletionInsertion)
    assertContains(editor.output.textContent ?: "", "if ( true ) {")

    editor.pressTab()
    assertTrue(input.value.endsWith("\nif ( true ) { "))
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun matchingTypedCharactersConsumeTheSoftInsertionPrefix() {
    val (editor, input) = editorFor("while ( true ")
    editor.cfg = plWhileCfg

    input.value += ")"
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertTrue(input.value.endsWith("\nwhile ( true )"))
    assertEquals(" { ", editor.pendingTerminalCompletionInsertion)

    input.value += " "
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertTrue(input.value.endsWith("\nwhile ( true ) "))
    assertEquals("{ ", editor.pendingTerminalCompletionInsertion)

    input.value += "{"
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertTrue(input.value.endsWith("\nwhile ( true ) {"))
    assertEquals(" ", editor.pendingTerminalCompletionInsertion)

    input.value += " "
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    val fullyTypedText = input.value
    assertTrue(fullyTypedText.endsWith("\nwhile ( true ) { "))
    assertNull(editor.pendingTerminalCompletionInsertion)

    editor.pressTab()
    assertEquals(fullyTypedText, input.value)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun unexpectedTypedCharacterClearsTheRemainingSoftInsertion() {
    val (editor, input) = editorFor("while ( true ")
    editor.cfg = plWhileCfg

    input.value += ")"
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertEquals(" { ", editor.pendingTerminalCompletionInsertion)

    input.value += " "
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertEquals("{ ", editor.pendingTerminalCompletionInsertion)

    input.value += "x"
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertTrue(input.value.endsWith("\nwhile ( true ) x"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun plWhileFreshClosingBraceOffersTrailingSpaceUntilTab() {
    val (editor, input) = editorFor("if ( true ) { if ( true ) { ID ")
    editor.cfg = plWhileGrammar.parseCFG(validate = true)
    val tokens = listOf(
      "if", "(", "true", ")", "{",
      "if", "(", "true", ")", "{", "ID", "}"
    )

    input.value += "}"
    input.setSelectionRange(input.value.length, input.value.length)
    assertNull(editor.cfg.terminalCompletionPlan(tokens))
    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nif ( true ) { if ( true ) { ID }"))
    assertEquals(" ", editor.pendingTerminalCompletionInsertion)

    editor.handleInput()
    assertTrue(input.value.endsWith("\nif ( true ) { if ( true ) { ID }"))
    assertEquals(" ", editor.pendingTerminalCompletionInsertion)

    editor.pressTab()
    assertTrue(input.value.endsWith("\nif ( true ) { if ( true ) { ID } "))
    assertEquals(input.value.length, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun exactTerminalAfterANonSuffixPrefixDoesNotReceiveASeparator() {
    val cfg = "START -> begin end".parseCFG().noEpsilon
    val (editor, input) = editorFor("unknown ")
    editor.cfg = cfg

    input.value += "end"
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nunknown end"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun plWhileChoosesTheSuffixViableTerminalForAnExactPrefix() = runTest {
    listOf(
      plWhileGrammar.parseCFG(validate = true),
      plWhileCfg
    ).forEach { cfg ->
      val completion = assertNotNull(
        cfg.terminalCompletionPlan(listOf("while", "(", "true", "="))
      )

      assertEquals(2, completion.lexicalCandidateCount)
      assertEquals(listOf("=="), completion.branches.map { it.terminal })
      assertEquals("==", completion.expandedPrefix)
      assertTrue(completion.terminalCommitted)
      assertTrue(completion.forcedContinuation.isEmpty())
      assertEquals((5..10).toList(), completion.branches.single().suffixLengths)
    }

    val (editor, input) = editorFor("while ( true ")
    editor.cfg = plWhileGrammar.parseCFG(validate = true)

    input.value += "="
    input.setSelectionRange(input.value.length, input.value.length)
    val typedText = input.value
    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals("= ", editor.pendingTerminalCompletionInsertion)
    assertContains(editor.output.textContent ?: "", "while ( true ==")

    editor.pressTab()
    assertTrue(input.value.endsWith("\nwhile ( true == "))
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun committedExactTerminalGetsASeparatorBeforeAmbiguousContinuation() {
    val cfg = """
      START -> go left | go right
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("go")
    editor.cfg = cfg

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("go")))
    assertTrue(completion.terminalCommitted)
    assertTrue(completion.forcedContinuation.isEmpty())
    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\ngo"))
    assertEquals(" ", editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\ngo "))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun completeSentenceDoesNotRequestAnExactTerminalContinuation() {
    val cfg = "START -> done".parseCFG().noEpsilon
    val (editor, input) = editorFor("done")
    editor.cfg = cfg

    assertNull(cfg.terminalCompletionPlan(listOf("done")))
    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\ndone"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun multipleViableTerminalInterpretationsRemainAmbiguous() {
    val cfg = """
      START -> if ( x | if ( y | ifdef z
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("if")
    editor.cfg = cfg

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("if")))
    assertEquals(listOf("if", "ifdef"), completion.branches.map { it.terminal })
    assertEquals("if", completion.expandedPrefix)
    assertFalse(completion.terminalCommitted)
    assertTrue(completion.forcedContinuation.isEmpty())

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nif"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun freshSeparatorAfterAnExactTerminalDoesNotTriggerContinuation() {
    val (editor, input) = editorFor("if ( true ) ")
    editor.cfg = plWhileCfg

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nif ( true ) "))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun staleInsertionAuthorizationCannotTriggerExactTerminalContinuation() {
    val (editor, input) = editorFor("if ( true )x")
    editor.cfg = plWhileCfg
    editor.recordFreshUserInsertion()
    input.value = input.value.dropLast(1)
    input.setSelectionRange(input.value.length, input.value.length)

    editor.handleInput()

    assertTrue(input.value.endsWith("\nif ( true )"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun uniqueTerminalAdvancesThroughMultipleCommonContinuationTokens() {
    val cfg = """
      START -> if ( value x | if ( value y
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("i")))

    assertEquals(listOf("(", "value"), completion.forcedContinuation)
    assertEquals(listOf("if", "(", "value"), completion.branches.single().tokens)
    assertEquals(listOf(1), completion.branches.single().suffixLengths)
  }

  @Test
  fun commonContinuationStopsWhenTheShortestSuffixEnds() {
    val cfg = """
      START -> if ( | if ( x
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("i")))

    assertEquals(listOf("("), completion.forcedContinuation)
    assertEquals(listOf(0, 1), completion.branches.single().suffixLengths)
  }

  @Test
  fun commonContinuationRequiresExactWholeTokenEquality() {
    val cfg = """
      START -> if ( x | if (( y
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("i")))

    assertTrue(completion.forcedContinuation.isEmpty())
  }

  @Test
  fun lexicalAmbiguityDoesNotCommitACompletedTerminalOrSpace() {
    val cfg = """
      START -> if ( x | ifdef ( y
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("i")
    editor.cfg = cfg

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("i")))
    assertEquals("if", completion.expandedPrefix)
    assertFalse(completion.terminalCommitted)
    assertTrue(completion.forcedContinuation.isEmpty())

    editor.handleFreshUserInsertion()
    assertTrue(input.value.endsWith("\ni"))
    assertEquals("f", editor.pendingTerminalCompletionInsertion)

    editor.pressTab()
    assertTrue(input.value.endsWith("\nif"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun nonContinuableTerminalPrefixDoesNotComplete() {
    val cfg = """
      START -> begin Table z | other Taco q
    """.trimIndent().parseCFG().noEpsilon

    assertNull(cfg.terminalCompletionPlan(listOf("begin", "Tac")))
  }

  @Test
  fun viableCandidatesAloneDetermineInsertedCommonPrefix() {
    val cfg = """
      START -> begin Table z | begin Target z | begin Task z | other Taco q
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("begin", "T")))

    assertEquals(4, completion.lexicalCandidateCount)
    assertEquals("Ta", completion.expandedPrefix)
    assertEquals(listOf("Table", "Target", "Task"), completion.branches.map { it.terminal })
    assertTrue(completion.branches.all { it.suffixLengths == listOf(1) })
  }

  @Test
  fun eachTerminalBranchKeepsItsOwnValidSuffixLengths() {
    val cfg = """
      START -> begin Table x | begin Target y z
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("begin", "T")))
    val suffixLengthsByTerminal = completion.branches.associate {
      it.terminal to it.suffixLengths
    }

    assertEquals(listOf(1), suffixLengthsByTerminal["Table"])
    assertEquals(listOf(2), suffixLengthsByTerminal["Target"])
  }

  @Test
  fun singleViableCandidateWinsOverNonviableLexicalCandidates() {
    val cfg = """
      START -> begin Table z | other Target q
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("begin", "T")))

    assertEquals(2, completion.lexicalCandidateCount)
    assertEquals(listOf("Table"), completion.branches.map { it.terminal })
    assertEquals("Table", completion.expandedPrefix)
    assertTrue(completion.terminalCommitted)
  }

  @Test
  fun viableCandidatesNeedAnotherSharedCharacterBeforeInsertion() {
    val cfg = """
      START -> begin Table x | begin Tree y
    """.trimIndent().parseCFG().noEpsilon

    val completion = assertNotNull(cfg.terminalCompletionPlan(listOf("begin", "T")))

    assertEquals(listOf("Table", "Tree"), completion.branches.map { it.terminal })
    assertEquals("T", completion.expandedPrefix)
  }

  @Test
  fun fairMergeBalancesAndFillsDisplayCapacity() {
    val merged = fairMerge(listOf(
      generateSequence(0) { it + 1 }.map { "a$it" },
      generateSequence(0) { it + 1 }.map { "b$it" },
      generateSequence(0) { it + 1 }.map { "c$it" }
    )).take(29).toList()

    assertEquals(10, merged.count { it.startsWith("a") })
    assertEquals(10, merged.count { it.startsWith("b") })
    assertEquals(9, merged.count { it.startsWith("c") })

    val withExhaustedBranch = fairMerge(listOf(
      sequenceOf("a0"),
      generateSequence(0) { it + 1 }.map { "b$it" }
    )).take(6).toList()
    assertEquals(listOf("a0", "b0", "b1", "b2", "b3", "b4"), withExhaustedBranch)
  }

  @Test
  fun tabCommitsTheUnambiguousTerminalSuffixAtCaret() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab")
    editor.cfg = uniqueCfg

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nbegin Tab"))
    assertEquals("le ", editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\nbegin Table "))
    assertEquals(input.value.length, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun tabCompletesTheLastTokenBeforeTrailingWhitespace() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab   ")
    editor.cfg = uniqueCfg
    val originalCaret = input.selectionStart!!

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nbegin Tab   "))
    assertEquals(originalCaret, input.selectionStart)
    assertEquals("le", editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\nbegin Table   "))
    assertEquals(originalCaret + 2, input.selectionStart)
    assertEquals(input.selectionStart, input.selectionEnd)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun completionReusesAnExistingTrailingTab() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab\t")
    editor.cfg = uniqueCfg

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nbegin Tab\t"))
    assertEquals("le", editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\nbegin Table\t"))
    assertEquals(input.value.length, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun completionMovesCaretPastAnExistingSeparatorAtTokenEnd() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab ")
    editor.cfg = uniqueCfg
    input.setSelectionRange(input.value.length - 1, input.value.length - 1)

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nbegin Tab "))
    assertEquals(input.value.length - 1, input.selectionStart)
    assertEquals("le", editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\nbegin Table "))
    assertEquals(input.value.length, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun ambiguousPrefixInsertionLeavesCaretBeforeExistingSeparator() {
    val ambiguousCfg = """
      START -> if ( x | ifdef ( y
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("i ")
    editor.cfg = ambiguousCfg
    input.setSelectionRange(input.value.length - 1, input.value.length - 1)

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\ni "))
    assertEquals(input.value.length - 1, input.selectionStart)
    assertEquals("f", editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\nif "))
    assertEquals(input.value.length - 1, input.selectionStart)
    assertEquals(input.selectionStart, input.selectionEnd)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun backspaceToPartialTerminalDoesNotAutoInsert() {
    val (editor, input) = editorFor("if")
    editor.cfg = plWhileCfg

    input.value = input.value.dropLast(1)
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleInput()

    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun caretMovementInvalidatesFreshInsertion() {
    val (editor, input) = editorFor("i ")
    editor.cfg = plWhileCfg
    editor.recordFreshUserInsertion()
    input.setSelectionRange(input.value.length - 1, input.value.length - 1)

    editor.handleInput()

    assertTrue(input.value.endsWith("\ni "))
    assertEquals(input.value.length - 1, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun staleInsertionAuthorizationIsInvalidatedByBackspace() {
    val (editor, input) = editorFor("if")
    editor.cfg = plWhileCfg
    editor.recordFreshUserInsertion()
    input.value = input.value.dropLast(1)
    input.setSelectionRange(input.value.length, input.value.length)

    editor.handleInput()

    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun backspaceCannotReinterpretExactEqualsAsDoubleEquals() {
    val (editor, input) = editorFor("while ( true ==")
    editor.cfg = plWhileCfg
    editor.recordFreshUserInsertion()
    input.value = input.value.dropLast(1)
    input.setSelectionRange(input.value.length, input.value.length)

    editor.handleInput()

    assertTrue(input.value.endsWith("\nwhile ( true ="))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun caretMovementCannotReinterpretExactEqualsAsDoubleEquals() {
    val (editor, input) = editorFor("while ( true =")
    editor.cfg = plWhileCfg
    editor.recordFreshUserInsertion()
    input.setSelectionRange(input.value.length - 1, input.value.length - 1)

    editor.handleInput()

    assertTrue(input.value.endsWith("\nwhile ( true ="))
    assertEquals(input.value.length - 1, input.selectionStart)
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun backspaceToAnEmptyLineInvalidatesAnExistingSoftInsertion() {
    val (editor, input) = editorFor("i")
    editor.cfg = plWhileCfg
    editor.handleFreshUserInsertion()
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)

    input.value = input.value.dropLast(1)
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleInput()

    assertTrue(input.value.endsWith("\n"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\n"))
    editor.runningJob?.cancel()
  }

  @Test
  fun caretMovementInvalidatesAnExistingSoftInsertion() {
    val (editor, input) = editorFor("i")
    editor.cfg = plWhileCfg
    editor.handleFreshUserInsertion()
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)

    input.setSelectionRange(input.value.length - 1, input.value.length - 1)
    editor.handleInput()

    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\ni"))
    editor.runningJob?.cancel()
  }

  @Test
  fun grammarChangeInvalidatesSoftInsertionAtCommitTime() {
    val (editor, input) = editorFor("i")
    editor.cfg = plWhileCfg
    editor.handleFreshUserInsertion()
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)

    editor.cfg = "START -> other".parseCFG().noEpsilon
    editor.pressTab()

    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun completionModeChangeInvalidatesSoftInsertionAtCommitTime() {
    val (editor, input) = editorFor("i")
    editor.cfg = plWhileCfg
    editor.handleFreshUserInsertion()
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)

    editor.epsilons = !editor.epsilons
    editor.pressTab()

    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun tabWithoutSoftInsertionRetainsNonterminalStubNavigation() {
    val (editor, input) = editorFor("left <EXP> right")
    val stubStart = input.value.lastIndexOf("<EXP>")

    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.pressTab()

    assertTrue(input.value.endsWith("\nleft <EXP> right"))
    assertEquals(stubStart, input.selectionStart)
    assertEquals(stubStart + "<EXP>".length, input.selectionEnd)
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun freshExactStubRetainsStubGenerationAndTabNavigation() = runTest {
    val stub = "<BEXP>"
    val (editor, input) = editorFor(stub)
    editor.cfg = plWhileCfg
    val typedText = input.value
    val stubStart = input.value.lastIndexOf(stub)

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertNull(editor.pendingTerminalCompletionInsertion)
    assertTrue(
      (editor.output.textContent ?: "")
        .startsWith("</> Stub generation, possible completions:")
    )

    editor.pressTab()

    assertEquals(typedText, input.value)
    assertEquals(stubStart, input.selectionStart)
    assertEquals(stubStart + stub.length, input.selectionEnd)
    assertNull(editor.pendingTerminalCompletionInsertion)
  }

  @Test
  fun textareaFallbackDoesNotArmAnInvisibleSoftInsertion() {
    val (editor, input) = editorFor("i", softPreviewAvailable = false)
    editor.cfg = plWhileCfg

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.pressTab()
    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun compositionStartInvalidatesAnEarlierInsertion() {
    val (editor, input) = editorFor("i")
    editor.cfg = plWhileCfg
    editor.recordFreshUserInsertion()

    input.dispatchEvent(Event("compositionstart"))
    input.dispatchEvent(Event("compositionend"))
    editor.handleInput()

    assertTrue(input.value.endsWith("\ni"))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun partialTerminalInsertionWaitsForCompositionToEnd() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab")
    editor.cfg = uniqueCfg

    input.dispatchEvent(Event("compositionstart"))
    editor.handleFreshUserInsertion()
    assertTrue(input.value.endsWith("\nbegin Tab"))
    assertNull(editor.pendingTerminalCompletionInsertion)

    input.dispatchEvent(Event("compositionend"))
    editor.handleInput()
    assertTrue(input.value.endsWith("\nbegin Tab"))
    assertEquals("le ", editor.pendingTerminalCompletionInsertion)

    editor.pressTab()
    assertTrue(input.value.endsWith("\nbegin Table "))
    assertNull(editor.pendingTerminalCompletionInsertion)
    editor.runningJob?.cancel()
  }

  @Test
  fun partialCompletionDoesNotReuseTheExactTerminalWorkHash() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Table")
    editor.cfg = uniqueCfg

    editor.handleInput()
    val exactTerminalJob = assertNotNull(editor.runningJob)

    input.value = input.value.removeSuffix("Table") + "Tab"
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nbegin Tab"))
    assertEquals("le ", editor.pendingTerminalCompletionInsertion)
    assertNotSame(exactTerminalJob, editor.runningJob)

    editor.pressTab()
    assertTrue(input.value.endsWith("\nbegin Table "))
    assertNull(editor.pendingTerminalCompletionInsertion)
    exactTerminalJob.cancel()
    editor.runningJob?.cancel()
  }

  @Test
  fun partialSuffixResultsEnumerateWithoutMutatingTheEditor() = runTest {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab")
    editor.cfg = uniqueCfg
    val typedText = input.value

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals("le ", editor.pendingTerminalCompletionInsertion)
    val outputText = editor.output.textContent ?: ""
    assertContains(outputText, "begin Table x")
    assertContains(outputText, "begin Table y")
  }

  @Test
  fun handleInputOffersTheLastCommonContinuationWithoutMutating() = runTest {
    val cfg = """
      START -> if ( x ) | if ( y )
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("i")
    editor.cfg = cfg
    val typedText = input.value

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)
    val outputText = editor.output.textContent ?: ""
    assertContains(outputText, "if ( x )")
    assertContains(outputText, "if ( y )")
  }

  @Test
  fun fullyForcedContinuationRemainsAVisibleCompletion() = runTest {
    val cfg = "START -> if (".parseCFG().noEpsilon
    val (editor, input) = editorFor("i")
    editor.cfg = cfg
    val typedText = input.value

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals("f ( ", editor.pendingTerminalCompletionInsertion)
    assertContains(editor.output.textContent ?: "", "if (")
  }

  @Test
  fun ambiguousPartialFansOutOnlyAcrossViableTerminalBranches() = runTest {
    val cfg = """
      START -> begin Table x | begin Target y | begin Task z | other Taco q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin T")
    editor.cfg = cfg
    val typedText = input.value

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals("a", editor.pendingTerminalCompletionInsertion)
    val text = editor.output.textContent ?: ""
    assertContains(text, "begin Table x")
    assertContains(text, "begin Target y")
    assertContains(text, "begin Task z")
    assertFalse(text.contains("begin Taco q"))
  }

  @Test
  fun ambiguousPartialLimitsFanoutToThreeTerminalBranches() = runTest {
    val cfg = """
      START -> begin Table w | begin Tangent x | begin Target y | begin Task z
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin T")
    editor.cfg = cfg
    val typedText = input.value

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertEquals(typedText, input.value)
    assertEquals("a", editor.pendingTerminalCompletionInsertion)
    val text = editor.output.textContent ?: ""
    assertContains(text, "begin Table w")
    assertContains(text, "begin Tangent x")
    assertContains(text, "begin Target y")
    assertFalse(text.contains("begin Task z"))
  }
}
