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
    assertEquals("int before;\r\n".length, snapshot.lineStartOffset)
    assertEquals(snapshot.lineStartOffset + prefix.length, snapshot.caretOffset)
    assertEquals(prefix, snapshot.prefixText)
    assertEquals(listOf("value", "="), snapshot.tokens.map { it.text })
    assertEquals(listOf(CppTokenKind.IDENTIFIER, CppTokenKind.OTHER), snapshot.tokens.map { it.kind })
    assertEquals(prefix.length, snapshot.replacementStartCharacter)
    assertEquals(source.substring(snapshot.lineStartOffset, source.indexOf("/*")),
      source.substring(snapshot.lineStartOffset, snapshot.replacementEndOffset))
    assertEquals(source.indexOf("/*") - snapshot.lineStartOffset, snapshot.replacementEndCharacter)
    assertTrue(snapshot.cacheKey.isNotBlank())
  }

  @Test
  fun snapshotRejectsUnsafeLocationsWithoutRejectingLongStatements() {
    assertNull(cppEditorStatementSnapshot("#include <vector>", 0, 4))
    val continuedDirective = """
      #define VALUE \
        7
    """.trimIndent()
    assertNull(cppEditorStatementSnapshot(continuedDirective, 1, 2))
    assertNull(cppEditorStatementSnapshot("identifier", 0, 4))
    assertNull(cppEditorStatementSnapshot("value // explanation", 0, 10))
    assertNull(cppEditorStatementSnapshot("/* open\nstill open", 1, "still open".length))
    assertNull(cppEditorStatementSnapshot("value /* explanation */", 0, 12))

    val quoted = "const char* text = R\"tag(// not a comment)tag\";"
    assertNotNull(cppEditorStatementSnapshot(quoted, 0, quoted.length))

    val longStatement = List(96) { "name" }.joinToString(" ")
    assertEquals(
      96,
      assertNotNull(
        cppEditorStatementSnapshot(longStatement, 0, longStatement.length)
      ).projectedTokens.size
    )
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
        missing = items.
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
