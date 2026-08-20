package cppcompletion

import cppEditorStatementSnapshot
import completionQuery
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticIdentifierProvenanceTest {
  @Test
  fun dependentAndInternalTypesNeverBecomeSourceTypeSpellings() {
    val internalType = "(lambda at source.cc:4:7)"
    val context = CppCompletionContext(
      identifiers = setOf("closure", "dependentValue"),
      sourceIdentifiers = setOf("closure", "dependentValue"),
      values = listOf(
        CppReference(
          "closure", type = internalType, isValue = true,
          typeInfo = CppTypeInfo(
            canonicalId = "type:closure", isSourceSpellable = false
          )
        ),
        CppReference(
          "dependentValue", type = "T", isValue = true,
          typeInfo = CppTypeInfo(
            canonicalId = "type:T", isDependent = true, isSourceSpellable = false
          )
        )
      )
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    val terminals = language.sourceSyntax.flatMapTo(linkedSetOf()) { it.second }

    assertTrue(language.recognizes(cppLines("closure;").single().tokens))
    assertFalse(language.recognizes(cppLines("dependentValue;").single().tokens))
    assertFalse(terminals.any { "lambda" in it || "source" in it || it == encodeIdentifier("T") })
  }

  @Test
  fun structuredReferenceCategoriesGateArgumentsAndRefQualifiedMethods() {
    val objectType = CppTypeInfo(
      canonicalId = "type:Widget", valueCanonicalId = "type:Widget",
      kind = "record", isSourceSpellable = true
    )
    fun returnInfo(kind: String = "record", isConst: Boolean = false) = objectType.copy(
      canonicalId = "type:$kind:${if (isConst) "const" else "mutable"}",
      kind = kind,
      isConst = isConst
    )
    val intInfo = CppTypeInfo(
      canonicalId = "type:int", valueCanonicalId = "type:int",
      kind = "builtin", isSourceSpellable = true
    )
    val members = listOf(
      CppReference(
        "onlyLvalue", returnType = "int", kind = "method", ownerType = "Widget",
        isCallable = true, isMember = true, isConstMethod = true, refQualifier = "&",
        ownerTypeInfo = objectType, returnTypeInfo = intInfo
      ),
      CppReference(
        "onlyRvalue", returnType = "int", kind = "method", ownerType = "Widget",
        isCallable = true, isMember = true, isConstMethod = true, refQualifier = "&&",
        ownerTypeInfo = objectType, returnTypeInfo = intInfo
      )
    )
    val context = CppCompletionContext(
      identifiers = setOf("Widget", "widget", "makeWidget", "makeConstWidget", "takeWidget") +
        members.map(CppReference::name),
      sourceIdentifiers = setOf("Widget", "widget", "makeWidget", "makeConstWidget", "takeWidget") +
        members.map(CppReference::name),
      values = listOf(CppReference(
        "widget", type = "Widget", isValue = true, typeInfo = objectType
      )),
      types = listOf(CppReference(
        "Widget", type = "Widget", isType = true, typeInfo = objectType
      )),
      functions = listOf(
        CppReference(
          "makeWidget", returnType = "Widget", isCallable = true,
          returnTypeInfo = returnInfo()
        ),
        CppReference(
          "makeConstWidget", returnType = "const Widget &&", isCallable = true,
          returnTypeInfo = returnInfo("rvalueReference", isConst = true)
        ),
        CppReference(
          "takeWidget", returnType = "int", isCallable = true,
          returnTypeInfo = intInfo,
          parameters = listOf(CppParameter(
            type = "Widget &&",
            typeInfo = returnInfo("rvalueReference")
          ))
        )
      ),
      membersByType = listOf(CppTypeMembers("Widget", members))
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    fun recognizes(statement: String) = language.recognizes(cppLines(statement).single().tokens)

    assertTrue(recognizes("widget.onlyLvalue();"))
    assertFalse(recognizes("widget.onlyRvalue();"))
    assertTrue(recognizes("makeWidget().onlyRvalue();"))
    assertFalse(recognizes("makeWidget().onlyLvalue();"))
    assertTrue(recognizes("takeWidget(makeWidget());"))
    assertFalse(recognizes("takeWidget(makeConstWidget());"))
  }

  @Test
  fun semanticGrammarContainsOnlyDeclarationBackedIdentifierTerminals() {
    val identifiers = setOf(
      "astral", "Beacon", "beacon", "illuminate", "intensity", "status"
    )
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      typeNames = setOf("astral::Beacon"),
      values = listOf(
        CppReference("beacon", type = "astral::Beacon", kind = "variable", source = "sema"),
        CppReference("intensity", type = "int", kind = "variable", source = "sema")
      ),
      types = listOf(
        CppReference("astral::Beacon", type = "astral::Beacon", kind = "class", source = "sema")
      ),
      functions = listOf(
        CppReference(
          name = "astral::illuminate",
          returnType = "int",
          parameters = listOf(CppParameter(name = "level", type = "int")),
          kind = "function",
          source = "sema"
        )
      ),
      membersByType = listOf(
        CppTypeMembers(
          "astral::Beacon",
          listOf(
            CppReference(
              name = "status",
              type = "int",
              kind = "field",
              ownerType = "astral::Beacon",
              source = "sema"
            )
          )
        )
      )
    )

    val identifierTerminals = CppCompletionGrammar().generate(context, emptyList()).sourceSyntax
      .flatMapTo(linkedSetOf()) { (_, rhs) ->
        rhs.filter { it.startsWith("@id:") }.map { it.removePrefix("@id:") }
      }

    assertTrue(identifiers.all(identifierTerminals::contains))
    assertEquals(
      emptySet(),
      identifierTerminals - identifiers,
      "Every reference terminal must originate in clang/Sema facts"
    )
  }

  @Test
  fun expressionAndQualifiedCompletionsNeverMaterializeInventedNames() {
    val identifiers = setOf("cosmos", "ignite", "telemetry")
    val context = CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      values = listOf(
        CppReference("telemetry", type = "int", kind = "variable", source = "sema")
      ),
      functions = listOf(
        CppReference(
          name = "cosmos::ignite",
          returnType = "int",
          parameters = listOf(CppParameter(name = "reading", type = "int")),
          kind = "function",
          source = "sema"
        )
      ),
      enclosingReturnType = "int"
    )

    listOf("cosmos::", "return ", "telemetry + ").forEach { prefixText ->
      val snapshot = assertNotNull(
        cppEditorStatementSnapshot(prefixText, 0, prefixText.length),
        prefixText
      )
      val suggestions = CppCompletionGrammar().completeCppStatement(
        context,
        snapshot.completionQuery(context.identifiers, seed = prefixText.hashCode())
      ).suggestions

      assertTrue(suggestions.isNotEmpty(), "Expected a semantic completion for `$prefixText`")
      suggestions.forEach { suggestion ->
        val emittedIdentifiers = cppLines(suggestion.candidateText).single().tokens
          .filter { it.kind == CppTokenKind.IDENTIFIER }
          .mapTo(linkedSetOf(), CppToken::text)
        assertEquals(
          emptySet(),
          emittedIdentifiers - identifiers,
          "Completion `${suggestion.candidateText}` introduced a non-Sema reference"
        )
        assertTrue(
          suggestion.freshNames.isEmpty(),
          "An expression position cannot introduce a fresh declaration binder"
        )
      }
    }
  }

  @Test
  fun declarationCompletionMayIntroduceOnlyItsRecordedFreshBinder() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("int ", 0, 4))
    val suggestions = CppCompletionGrammar().completeCppStatement(
      CppCompletionContext(emptySet()),
      snapshot.completionQuery(emptySet(), seed = 47)
    ).suggestions
    val declaration = assertNotNull(suggestions.firstOrNull { it.freshNames.isNotEmpty() })
    val tokens = cppLines(declaration.candidateText).single().tokens
    val identifiers = tokens.filter { it.kind == CppTokenKind.IDENTIFIER }.map(CppToken::text)

    assertEquals(declaration.freshNames, identifiers.toSet())
    declaration.freshNames.forEach { binder ->
      val binderIndex = tokens.indexOfFirst { it.text == binder }
      assertTrue(tokens.take(binderIndex).any { it.text == "int" })
      assertEquals(";", tokens.last().text)
    }
  }

  @Test
  fun arbitrarySemaNamesAreAlphaInvariant() {
    fun completionShape(scope: String, callable: String, argument: String): List<String> {
      val identifiers = setOf(scope, callable, argument)
      val context = CppCompletionContext(
        identifiers = identifiers,
        sourceIdentifiers = identifiers,
        values = listOf(
          CppReference(argument, type = "int", kind = "variable", source = "sema")
        ),
        functions = listOf(
          CppReference(
            name = "$scope::$callable",
            returnType = "int",
            parameters = listOf(CppParameter(name = "input", type = "int")),
            kind = "function",
            source = "sema"
          )
        )
      )
      val prefixText = "$scope::${callable.take(4)}"
      val snapshot = assertNotNull(
        cppEditorStatementSnapshot(prefixText, 0, prefixText.length),
        prefixText
      )
      val suggestion = assertNotNull(
        CppCompletionGrammar().completeCppStatement(
          context,
          snapshot.completionQuery(identifiers, seed = 91)
        ).suggestions.firstOrNull { callable in it.tokens },
        "No completion used the arbitrary Sema callable `$scope::$callable`"
      )
      return suggestion.tokens.map { token -> when (token) {
        scope -> "<scope>"
        callable -> "<callable>"
        argument -> "<argument>"
        else -> token
      } }
    }

    assertEquals(
      completionShape("cosmos", "illuminate", "telemetry"),
      completionShape("nebula", "transfigure", "quasar")
    )
  }

  @Test
  fun arbitrarySemaMemberNamesAreAlphaInvariant() {
    fun completionShape(type: String, receiver: String, member: String): List<String> {
      val identifiers = setOf(type, receiver, member)
      val context = CppCompletionContext(
        identifiers = identifiers,
        sourceIdentifiers = identifiers,
        values = listOf(
          CppReference(receiver, type = type, kind = "variable", source = "sema")
        ),
        types = listOf(
          CppReference(type, type = type, kind = "class", source = "sema")
        ),
        membersByType = listOf(
          CppTypeMembers(
            type,
            listOf(
              CppReference(
                name = member,
                returnType = "int",
                parameters = listOf(CppParameter(name = "level", type = "int")),
                kind = "method",
                ownerType = type,
                source = "sema"
              )
            )
          )
        )
      )
      val prefixText = "$receiver.${member.take(4)}"
      val snapshot = assertNotNull(
        cppEditorStatementSnapshot(prefixText, 0, prefixText.length),
        prefixText
      )
      val suggestion = assertNotNull(
        CppCompletionGrammar().completeCppStatement(
          context,
          snapshot.completionQuery(identifiers, seed = 103)
        ).suggestions.firstOrNull { member in it.tokens },
        "No completion used the arbitrary Sema member `$type::$member`"
      )
      return suggestion.tokens.map { token -> when (token) {
        receiver -> "<receiver>"
        member -> "<member>"
        else -> token
      } }
    }

    assertEquals(
      completionShape("Probe", "probe", "calibrate"),
      completionShape("Glyph", "glyph", "resonate")
    )
  }

  @Test
  fun editorSamplingNeverWidensTheSemanticGrammarWithUntypedSyntaxNames() {
    val identifiers = setOf("scopeObject", "knownMember", "knapsack")
    val context = CppCompletionContext(
      identifiers = identifiers,
      values = listOf(
        CppReference("scopeObject", type = "int", kind = "variable", source = "sema")
      )
    )
    val snapshot = assertNotNull(cppEditorStatementSnapshot("scopeObject.kn", 0, 14))
    val query = snapshot.completionQuery(identifiers, seed = 211)
    val prepared = CppCompletionGrammar().prepare(context, query.prefix)
    val residual = prepared.generate(query.prefix)
    val completions = residual.shortestCompletions(
      query.prefixText, identifiers, query.limit, Random(query.seed), query.tokenPrefix
    )

    assertTrue(completions.none { completion ->
      completion.tokens.any { it == "knownMember" || it == "knapsack" }
    }, "An untyped syntax-only identifier escaped into the editor completion batch")
  }

}
