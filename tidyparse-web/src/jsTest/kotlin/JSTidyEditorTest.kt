import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.noEpsilon
import ai.hypergraph.kaliningraph.parsing.parseCFG
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
    override var cfg: CFG
  ) : JSTidyEditor(editor, output) {
    var writes = 0

    override fun getLatestCFG(): CFG = cfg
    override fun readDisplayText(): String = output.textContent ?: ""
    override fun writeDisplayText(s: String) {
      writes++
      (output as HTMLDivElement).innerHTML = s
    }
  }

  private fun editorFor(line: String): Pair<RecordingEditor, HTMLTextAreaElement> {
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
    return RecordingEditor(input, output, cfg) to input
  }

  private fun RecordingEditor.handleFreshUserInsertion() {
    recordFreshUserInsertion()
    handleInput()
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
  fun plWhileFreshInsertionCompletesIfAndOpeningParenthesis() = runTest {
    val (editor, input) = editorFor("i")
    editor.cfg = plWhileGrammar.parseCFG(validate = true)

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertTrue(input.value.endsWith("\nif ( "))
    assertContains(editor.output.textContent ?: "", "if (")
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

      assertFalse(completion.isPartialMatch)
      assertEquals(listOf("{"), completion.forcedContinuation)
      assertEquals(
        listOf("if", "(", "true", ")", "{"),
        completion.branches.single().tokens
      )
      assertEquals((2..9).toList(), completion.branches.single().suffixLengths)
    }
  }

  @Test
  fun plWhileFreshClosingParenthesisAdvancesThroughOpeningBrace() = runTest {
    val (editor, input) = editorFor("if ( true ")
    editor.cfg = plWhileGrammar.parseCFG(validate = true)

    input.value += ")"
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertTrue(input.value.endsWith("\nif ( true ) { "))
    val output = editor.output as HTMLDivElement
    assertContains(output.textContent ?: "", "if ( true ) {")
    assertContains(output.innerHTML, "<span style=\"color: green\">{</span>")
    assertFalse(output.innerHTML.contains("partial-terminal-match"))
    assertFalse(output.innerHTML.contains("color: orange"))
  }

  @Test
  fun plWhileFreshClosingBraceReceivesTrailingSpace() {
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

    assertTrue(input.value.endsWith("\nif ( true ) { if ( true ) { ID } "))
    assertEquals(input.value.length, input.selectionStart)

    editor.handleInput()
    assertTrue(input.value.endsWith("\nif ( true ) { if ( true ) { ID } "))

    input.value = input.value.dropLast(1)
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleInput()
    assertTrue(input.value.endsWith("\nif ( true ) { if ( true ) { ID }"))
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
      assertTrue(completion.isPartialMatch)
      assertTrue(completion.forcedContinuation.isEmpty())
      assertEquals((5..10).toList(), completion.branches.single().suffixLengths)
    }

    val (editor, input) = editorFor("while ( true ")
    editor.cfg = plWhileGrammar.parseCFG(validate = true)

    input.value += "="
    input.setSelectionRange(input.value.length, input.value.length)
    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertTrue(input.value.endsWith("\nwhile ( true == "))
    assertContains(editor.output.textContent ?: "", "while ( true ==")
    val output = editor.output as HTMLDivElement
    assertContains(output.innerHTML, "<span class=\"partial-terminal-match\">=</span>")
    assertContains(output.innerHTML, "<span style=\"color: orange\">=</span>")
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

    assertTrue(input.value.endsWith("\ngo "))
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
    editor.runningJob?.cancel()
  }

  @Test
  fun freshSeparatorAfterAnExactTerminalDoesNotTriggerContinuation() {
    val (editor, input) = editorFor("if ( true ) ")
    editor.cfg = plWhileCfg

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nif ( true ) "))
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
    assertTrue(input.value.endsWith("\nif"))
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
  fun partialTerminalDiffHighlightsOnlyMatchedTokenCharacters() {
    val html = terminalCompletionDiff(
      inputTokens = listOf("Ta", "lead", "Ta"),
      matchedPrefix = "Ta",
      completion = "Ta lead Table z"
    )

    assertEquals(1, "partial-terminal-match".toRegex().findAll(html).count())
    assertContains(html, "Ta lead <span class=\"partial-terminal-match\">Ta</span>")
    assertContains(html, "<span style=\"color: orange\">ble</span>")
    assertContains(html, "<span style=\"color: green\">z</span>")

    val div = document.createElement("div") as HTMLDivElement
    div.innerHTML = html
    assertEquals("Ta lead Table z", div.textContent)
  }

  @Test
  fun partialTerminalDiffEscapesHighlightedTerminal() {
    val html = terminalCompletionDiff(
      inputTokens = listOf("lead", "<&"),
      matchedPrefix = "<&",
      completion = "lead <&rest tail"
    )

    assertContains(html, "<span class=\"partial-terminal-match\">&lt;&amp;</span>")
    assertFalse(html.contains("<&rest"))
  }

  @Test
  fun handleInputInsertsTheUnambiguousTerminalSuffixAtCaret() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab")
    editor.cfg = uniqueCfg

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nbegin Table "))
    assertEquals(input.value.length, input.selectionStart)
    editor.runningJob?.cancel()
  }

  @Test
  fun handleInputCompletesTheLastTokenBeforeTrailingWhitespace() {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin Tab   ")
    editor.cfg = uniqueCfg
    val originalCaret = input.selectionStart!!

    editor.handleFreshUserInsertion()

    assertTrue(input.value.endsWith("\nbegin Table   "))
    assertEquals(originalCaret + 2, input.selectionStart)
    assertEquals(input.selectionStart, input.selectionEnd)
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

    assertTrue(input.value.endsWith("\nbegin Table\t"))
    assertEquals(input.value.length, input.selectionStart)
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

    assertTrue(input.value.endsWith("\nbegin Table "))
    assertEquals(input.value.length, input.selectionStart)
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

    assertTrue(input.value.endsWith("\nif "))
    assertEquals(input.value.length - 1, input.selectionStart)
    assertEquals(input.selectionStart, input.selectionEnd)
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

    input.dispatchEvent(Event("compositionend"))
    editor.handleInput()
    assertTrue(input.value.endsWith("\nbegin Table "))
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

    assertTrue(input.value.endsWith("\nbegin Table "))
    assertNotSame(exactTerminalJob, editor.runningJob)
    exactTerminalJob.cancel()
    editor.runningJob?.cancel()
  }

  @Test
  fun partialSuffixResultsRenderWithYellowMatchedPrefix() = runTest {
    val uniqueCfg = """
      START -> begin Table x | begin Table y | other Target q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, _) = editorFor("begin Tab")
    editor.cfg = uniqueCfg

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    val output = editor.output as HTMLDivElement
    assertContains(output.innerHTML, "<span class=\"partial-terminal-match\">Tab</span>")
    assertContains(output.innerHTML, "<span style=\"color: orange\">le</span>")
    assertContains(output.textContent ?: "", "begin Table x")
    assertContains(output.textContent ?: "", "begin Table y")
  }

  @Test
  fun handleInputAdvancesToTheLastCommonContinuationToken() = runTest {
    val cfg = """
      START -> if ( x ) | if ( y )
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("i")
    editor.cfg = cfg

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertTrue(input.value.endsWith("\nif ( "))
    val output = editor.output as HTMLDivElement
    assertContains(output.textContent ?: "", "if ( x )")
    assertContains(output.textContent ?: "", "if ( y )")
    assertContains(output.innerHTML, "<span class=\"partial-terminal-match\">i</span>")
    assertContains(output.innerHTML, "<span style=\"color: orange\">f</span>")
    assertContains(output.innerHTML, "<span style=\"color: green\">(</span>")
  }

  @Test
  fun fullyForcedContinuationRemainsAVisibleCompletion() = runTest {
    val cfg = "START -> if (".parseCFG().noEpsilon
    val (editor, input) = editorFor("i")
    editor.cfg = cfg

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertTrue(input.value.endsWith("\nif ( "))
    assertContains(editor.output.textContent ?: "", "if (")
  }

  @Test
  fun ambiguousPartialFansOutOnlyAcrossViableTerminalBranches() = runTest {
    val cfg = """
      START -> begin Table x | begin Target y | begin Task z | other Taco q
    """.trimIndent().parseCFG().noEpsilon
    val (editor, input) = editorFor("begin T")
    editor.cfg = cfg

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    assertTrue(input.value.endsWith("\nbegin Ta"))
    val output = editor.output as HTMLDivElement
    val text = output.textContent ?: ""
    assertContains(text, "begin Table x")
    assertContains(text, "begin Target y")
    assertContains(text, "begin Task z")
    assertFalse(text.contains("begin Taco q"))
    assertEquals(3, "partial-terminal-match".toRegex().findAll(output.innerHTML).count())
    assertEquals(3, "<span class=\"partial-terminal-match\">T</span>".toRegex()
      .findAll(output.innerHTML).count())
  }

  @Test
  fun ambiguousPartialLimitsFanoutToThreeTerminalBranches() = runTest {
    val cfg = """
      START -> begin Table w | begin Tangent x | begin Target y | begin Task z
    """.trimIndent().parseCFG().noEpsilon
    val (editor, _) = editorFor("begin T")
    editor.cfg = cfg

    editor.handleFreshUserInsertion()
    assertNotNull(editor.runningJob).join()

    val text = editor.output.textContent ?: ""
    assertContains(text, "begin Table w")
    assertContains(text, "begin Tangent x")
    assertContains(text, "begin Target y")
    assertFalse(text.contains("begin Task z"))
  }
}
