import cppcompletion.CppTokenKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CppCompletionBrowserAdapterTest {
  @Test
  fun snapshotPreservesRawPrefixAndUsesExactLineSuffixCoordinates() {
    val source = "int before;\r\n  value = next /* keep this comment */\r\n"
    val prefix = "  value ="
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 1, prefix.length))

    assertEquals(1, snapshot.line)
    assertEquals(prefix.length, snapshot.character)
    assertEquals(prefix, snapshot.prefixText)
    assertEquals(listOf("value", "="), snapshot.tokens.map { it.text })
    assertEquals(listOf(CppTokenKind.IDENTIFIER, CppTokenKind.OTHER), snapshot.tokens.map { it.kind })
    assertEquals("=", snapshot.activeFragment?.text)
    assertEquals(listOf("value"), snapshot.stableTokens.map { it.text })
    assertEquals("  value ", snapshot.stablePrefixText)
    assertEquals(prefix, snapshot.semanticPrefixText)
    val physicalLine = source.lines()[1]
    assertEquals(physicalLine.indexOf("/*") - 1, snapshot.replacementEndCharacter)
    assertTrue(physicalLine.substring(snapshot.replacementEndCharacter).startsWith(" /* keep"))
    assertEquals(
      "  value = 0; /* keep this comment */",
      physicalLine.replaceRange(
        snapshot.statementStartCharacter,
        snapshot.replacementEndCharacter,
        "  value = 0;"
      )
    )
    assertTrue(snapshot.cacheKey.isNotBlank())
  }

  @Test
  fun snapshotRejectsDirectivesAndCommentsWithoutCappingStatements() {
    assertNull(cppEditorStatementSnapshot("#include <vector>", 0, 4))
    val continuedDirective = """
      #define VALUE \
        7
    """.trimIndent()
    assertNull(cppEditorStatementSnapshot(continuedDirective, 1, 2))
    assertNull(cppEditorStatementSnapshot("value // explanation", 0, 10))
    assertNull(cppEditorStatementSnapshot("/* open\nstill open", 1, "still open".length))
    assertNull(cppEditorStatementSnapshot("value /* explanation */", 0, 12))
    listOf("123" to 1, "\"text\"" to 3, "'x'" to 2, "42_km" to 2).forEach { (text, caret) ->
      val literal = assertNotNull(cppEditorStatementSnapshot(text, 0, caret))
      assertEquals(text, literal.activeFragment?.completeText)
      assertEquals(text.substring(0, caret), literal.activeFragment?.text)
    }

    val identifier = assertNotNull(cppEditorStatementSnapshot("identifier", 0, 4))
    assertEquals("iden", identifier.activeFragment?.text)
    assertEquals(emptyList(), identifier.stableTokens)
    assertEquals("", identifier.stablePrefixText)

    val quoted = "const char* text = R\"tag(// not a comment)tag\";"
    assertNotNull(cppEditorStatementSnapshot(quoted, 0, quoted.length))

    val longStatement = List(96) { "name" }.joinToString(" ")
    assertEquals(
      96,
      assertNotNull(
        cppEditorStatementSnapshot(longStatement, 0, longStatement.length)
      ).tokens.size
    )
  }

  @Test
  fun snapshotPartitionsSafeWordAndOperatorFragmentsAtTheCaret() {
    data class Case(
      val source: String,
      val character: Int,
      val fragment: String,
      val stablePrefix: String,
      val semanticPrefix: String
    )

    listOf(
      Case("std::string", "std::str".length, "str", "std::", "std::"),
      Case("records.try_emplace", "records.try_emp".length,
        "try_emp", "records.", "records."),
      Case("true", 3, "tru", "", ""),
      Case("nullptr", 5, "nullp", "", ""),
      Case("!=", 1, "!", "", "!="),
      Case("ptr ->field", "ptr -".length, "-", "ptr ", "ptr ->"),
      Case("std::string", "std:".length, ":", "std", "std::"),
      Case("value >>= rhs", "value >>".length, ">>", "value ", "value >>="),
      Case("u8\"text\"", "u8\"text\"".length, "u8\"text\"", "", "u8\"text\""),
      Case("std::", "std::".length, "::", "std", "std::"),
      Case("visit(", "visit(".length, "(", "visit", "visit(")
    ).forEach { case ->
      val snapshot = assertNotNull(
        cppEditorStatementSnapshot(case.source, 0, case.character),
        "Expected a snapshot for `${case.source}` at ${case.character}"
      )
      assertEquals(case.fragment, snapshot.activeFragment?.text, case.source)
      assertEquals(case.stablePrefix, snapshot.stablePrefixText, case.source)
      assertEquals(case.semanticPrefix, snapshot.semanticPrefixText, case.source)
      assertEquals(case.fragment, snapshot.prefixText.substring(snapshot.activeFragment!!.start))
    }
  }

  @Test
  fun snapshotIsolatesTheCurrentSameLineStatementAndPreservesFollowingCode() {
    val inlineFunction = "int main() { return ; }"
    val returnCaret = inlineFunction.indexOf(';')
    val returnSnapshot = assertNotNull(
      cppEditorStatementSnapshot(inlineFunction, 0, returnCaret)
    )
    assertEquals(inlineFunction.indexOf('{') + 1, returnSnapshot.statementStartCharacter)
    assertEquals(" return ", returnSnapshot.prefixText)
    assertEquals(listOf("return"), returnSnapshot.tokens.map { it.text })
    assertEquals(1, returnSnapshot.tokens.single().start)
    assertEquals(inlineFunction.indexOf(';') + 1, returnSnapshot.replacementEndCharacter)
    assertEquals(" }", inlineFunction.substring(returnSnapshot.replacementEndCharacter))

    val multiple = "int x = 0; value; later();"
    val valueCaret = multiple.indexOf("value") + "value".length
    val valueSnapshot = assertNotNull(cppEditorStatementSnapshot(multiple, 0, valueCaret))
    assertEquals(multiple.indexOf(';') + 1, valueSnapshot.statementStartCharacter)
    assertEquals(" value", valueSnapshot.prefixText)
    assertEquals(1, valueSnapshot.tokens.single().start)
    assertEquals(multiple.indexOf(';', valueCaret) + 1, valueSnapshot.replacementEndCharacter)
    assertEquals(" later();", multiple.substring(valueSnapshot.replacementEndCharacter))

    val afterBlock = "if (ready) { work(); } next;"
    val nextCaret = afterBlock.indexOf("next") + "next".length
    val nextSnapshot = assertNotNull(cppEditorStatementSnapshot(afterBlock, 0, nextCaret))
    assertEquals(afterBlock.indexOf('}') + 1, nextSnapshot.statementStartCharacter)
    assertEquals(" next", nextSnapshot.prefixText)

    val qualifiedFunction = "auto f() const -> int requires Ready { return ; }"
    val qualifiedCaret = qualifiedFunction.indexOf(';')
    val qualifiedSnapshot = assertNotNull(
      cppEditorStatementSnapshot(qualifiedFunction, 0, qualifiedCaret)
    )
    assertEquals(qualifiedFunction.indexOf('{') + 1, qualifiedSnapshot.statementStartCharacter)
    assertEquals(" return ", qualifiedSnapshot.prefixText)

    val emptyPrefix = "int f() { return value; }"
    val emptyCaret = emptyPrefix.indexOf("return")
    val emptySnapshot = assertNotNull(cppEditorStatementSnapshot(emptyPrefix, 0, emptyCaret))
    assertTrue(emptySnapshot.tokens.isEmpty())
    assertEquals(emptyPrefix.indexOf(';') + 1, emptySnapshot.replacementEndCharacter)
    assertEquals(" }", emptyPrefix.substring(emptySnapshot.replacementEndCharacter))

    val loop = "for (int i = 0; i < n; ++i) { use; } later;"
    val useCaret = loop.indexOf("use") + "use".length
    val loopSnapshot = assertNotNull(cppEditorStatementSnapshot(loop, 0, useCaret))
    assertEquals(loop.indexOf('{') + 1, loopSnapshot.statementStartCharacter)
    assertEquals(" use", loopSnapshot.prefixText)
    assertEquals(loop.indexOf(';', useCaret) + 1, loopSnapshot.replacementEndCharacter)
    assertEquals(" } later;", loop.substring(loopSnapshot.replacementEndCharacter))

    val splitToken = "return; later(); // keep"
    val partial = assertNotNull(cppEditorStatementSnapshot(splitToken, 0, "ret".length))
    assertEquals(
      splitToken,
      splitToken.replaceRange(
        partial.statementStartCharacter,
        partial.replacementEndCharacter,
        "return;"
      )
    )
  }

  @Test
  fun completionFactsPreferSemanticLabelsAndNeverExposeSnippetPayloadsAsNames() {
    val source = "int foo(int); int bar(int); int baz(int);"
    val result = js(
      """({items: [
        {label: 'foo', filterText: 'foo', insertText: '', insertTextFormat: 2,
          textEdit: {newText: ''}, kind: 3, detail: 'int',
          labelDetails: {detail: '(int arg)'}},
        {label: '', filterText: 'bar', insertText: '', insertTextFormat: 2,
          textEdit: {newText: ''}, kind: 3, detail: 'int'},
        {label: '', insertText: '', insertTextFormat: 2,
          textEdit: {newText: ''}, kind: 3, detail: 'int'}
      ]})"""
    )
    result.items[0].insertText = "foo(\${1:arg})"
    result.items[0].textEdit.newText = "foo(\${1:arg})"
    result.items[1].insertText = "bar(\${1:arg})"
    result.items[1].textEdit.newText = "wrong(\${1:arg})"
    result.items[2].insertText = "baz(\${1:arg})"
    result.items[2].textEdit.newText = "baz(\${1:arg})"

    val context = cppCompletionContextFromDto(cppCompletionContextDto(
      source = source,
      completionGroups = listOf(CppClangdCompletionGroup(result))
    ))

    assertEquals(listOf("foo", "bar", "baz"), context.completions.map { it.name })
    assertTrue(context.completions.none { '$' in it.name || '(' in it.name || ')' in it.name })
    assertTrue(context.functions.any { it.name == "foo" && it.parameters.single().name == "arg" })
  }

  @Test
  fun nonReceiverMethodCompletionsAreNeverPromotedToFreeFunctions() {
    val source = """
      #include <map>
      #include <string>
      #include <tuple>
      using Record = std::tuple<int, std::string, double>;
      std::map<int, Record> records;
      records.try_emplace
    """.trimIndent()
    val collapsedMethod = js(
      """({items: [{label: 'try_emplace', insertText: 'try_emplace(${'$'}0)', kind: 2,
        detail: '[4 overloads]', labelDetails: {detail: '(…)'}}]})"""
    )
    val ast = js(
      """({values: [{name: 'records',
        type: 'std::map<int, std::tuple<int, std::string, double>>',
        kind: 'variable', source: 'ast'}]})"""
    )
    val line = source.lines().lastIndex
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, line, source.lines().last().length))
    val context = cppCompletionContextFromDto(cppCompletionContextDto(
      source = source,
      completionGroups = listOf(CppClangdCompletionGroup(collapsedMethod)),
      ast = ast,
      snapshot = snapshot
    ))

    assertTrue(context.completions.none { it.name == "try_emplace" })
    assertTrue(context.functions.none { it.name == "try_emplace" })
    assertTrue(context.values.any { it.name == "records" })
  }

  @Test
  fun diagnosticsFromOtherStatementsDoNotConstrainThisCursor() {
    val source = "missing(); current = ;\nfuture;"
    val caret = source.indexOf('=') + 1
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 0, caret))
    val diagnostics = js(
      """([
        {code: 'undeclared_var_use', message: "use of undeclared identifier 'missing'",
          range: {start: {line: 0, character: 0}, end: {line: 0, character: 7}}},
        {code: 'undeclared_var_use', message: "use of undeclared identifier 'current'",
          range: {start: {line: 0, character: 11}, end: {line: 0, character: 18}}},
        {code: 'undeclared_var_use', message: "use of undeclared identifier 'future'",
          range: {start: {line: 1, character: 0}, end: {line: 1, character: 6}}}
      ])"""
    )

    val context = cppCompletionContextFromDto(cppCompletionContextDto(
      source = source,
      diagnostics = diagnostics,
      snapshot = snapshot
    ))

    assertEquals(setOf("current"), context.unresolvedIdentifiers)
    assertEquals("current", context.requiredIdentifier)
  }

  @Test
  fun contextDtoRoundTripsRichClangdAndLexicalFacts() {
    val source = """
      #include <vector>
      struct Widget {};
      using Count = unsigned long;
      bool helper(int count);
      int main() {
        std::vector<Widget> items;
        missing = items.push_ba
      }
    """.trimIndent()
    val line = source.lines().indexOfFirst { "missing" in it }
    val character = source.lines()[line].length
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, line, character))

    val scopeResult = js(
      """({items: [
        {label: 'items', insertText: 'items', kind: 6, detail: 'std::vector<Widget>'},
        {label: 'helper', insertText: 'helper', kind: 3, detail: 'bool',
          labelDetails: {detail: '(int count = 1)'}},
        {label: 'Widget', insertText: 'Widget', kind: 22, detail: 'Widget'}
      ]})"""
    )
    val receiverResult = js(
      """({items: [
        {label: 'push_back', insertText: 'push_back', kind: 2, detail: 'void',
          labelDetails: {detail: '(const Widget &value)'}}
      ]})"""
    )
    val signatureHelp = js(
      """({activeSignature: 0, activeParameter: 0, signatures: [{
        label: 'helper(int count) -> bool', parameters: [{label: [7, 16]}]
      }]})"""
    )
    val hover = js("({contents: {kind: 'markdown', value: 'Type: std::vector<Widget>'}})")
    val diagnostics = js(
      """([{code: 'undeclared_var_use', message: "use of undeclared identifier 'missing'",
        range: {start: {line: 0, character: 2}, end: {line: 0, character: 9}}}])"""
    )
    diagnostics[0].range.start.line = line
    diagnostics[0].range.end.line = line
    val ast = js(
      """({values: [{name: 'count', type: 'Count', kind: 'variable', source: 'ast'}],
        conversions: [{from: 'Count', to: 'unsigned long'}]})"""
    )

    val dto = cppCompletionContextDto(
      source = source,
      completionGroups = listOf(
        CppClangdCompletionGroup(receiverResult, receiverMember = true, receiverOperator = "."),
        CppClangdCompletionGroup(scopeResult)
      ),
      signatures = signatureHelp,
      hover = hover,
      diagnostics = diagnostics,
      ast = ast,
      snapshot = snapshot
    )
    val clone = JSON.parse<dynamic>(JSON.stringify(dto))
    val context = cppCompletionContextFromDto(clone)

    assertEquals(setOf("vector"), context.headers)
    assertTrue(setOf("Widget", "Count", "int", "unsigned", "long").all { it in context.typeNames })
    assertTrue(setOf("items", "helper", "missing", "Widget", "Count").all { it in context.sourceIdentifiers })
    assertTrue(context.values.any { it.name == "items" && it.type == "std::vector<Widget>" })
    assertTrue(context.values.any { it.name == "count" && it.type == "Count" })
    assertTrue(context.functions.any { it.name == "helper" && it.parameters.single().defaultValue == "1" })
    assertTrue(context.types.any { it.name == "Count" && it.type == "unsigned long" })
    assertTrue(context.types.any { it.name == "Widget" && it.kind == "struct" && it.source == "source" })
    assertEquals(setOf("int"), context.expectedTypes)
    assertEquals(".", context.receiver?.operator)
    assertEquals("items", context.receiver?.expression)
    assertEquals("std::vector<Widget>", context.receiver?.type)
    assertEquals("push_back", context.receiver?.members?.single()?.name)
    assertEquals("std::vector<Widget>", context.receiver?.members?.single()?.ownerType)
    assertEquals("missing", context.requiredIdentifier)
    assertEquals(setOf("missing"), context.unresolvedIdentifiers)
    assertEquals(listOf("Count" to "unsigned long"), context.conversions.map { it.from to it.to })
  }
}
