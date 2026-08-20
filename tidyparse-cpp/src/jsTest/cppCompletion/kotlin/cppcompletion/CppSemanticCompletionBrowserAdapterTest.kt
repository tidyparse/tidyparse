import cppcompletion.CppCompletionGrammar
import cppcompletion.completeCppStatement
import cppcompletion.cppLines
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CppSemanticCompletionBrowserAdapterTest {
  @Test
  fun templateDefaultsAndTypeRolesFlowFromSemaIntoTheExactTemplateBoundaryGrammar() {
    val source = "work::Box<"
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 0, source.length))
    val result = js(
      """({schemaVersion: 1, context: {kind: 'Symbol'}, items: [{
        name: 'Box', requiredQualifier: 'work::', insertText: 'work::Box', kind: 7,
        symbols: [{
          id: 'box-template', qualifiedName: 'work::Box', kind: 'ClassTemplate',
          provenance: {sema: true, index: true}, isType: true,
          type: 'Box<T, Policy>', canonicalType: 'Box<T, Policy>',
          typeInfo: {canonicalId: 'template:box', isDependent: true,
            isInstantiationDependent: true, isSourceSpellable: false},
          templateParameters: [
            {name: 'T', kind: 'type', hasDefault: false, isPack: false},
            {name: 'Policy', kind: 'type', hasDefault: true, isPack: false}
          ]
        }]
      }], scopeItems: [{
        name: 'Visible', insertText: 'Visible', kind: 7,
        symbols: [{
          id: 'visible-type', qualifiedName: 'Visible', kind: 'CXXRecord',
          provenance: {sema: true, index: false}, isType: true,
          type: 'Visible', canonicalType: 'Visible',
          typeInfo: {canonicalId: 'type:visible', valueCanonicalId: 'type:visible',
            kind: 'record', isSourceSpellable: true}
        }]
      }, {
        name: 'from_range_t', requiredQualifier: 'std::', insertText: 'std::from_range_t', kind: 7,
        symbols: [{
          id: 'from-range-type', qualifiedName: 'std::from_range_t', kind: 'TypeAlias',
          provenance: {sema: true, index: false}, isType: true,
          type: 'from_range_t', canonicalType: 'std::from_range_t',
          typeInfo: {canonicalId: 'type:from-range', valueCanonicalId: 'type:from-range',
            kind: 'record', isSourceSpellable: true}
        }]
      }]})"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val template = context.types.single { it.qualifiedName == "work::Box" }
    assertEquals(listOf(false, true), template.templateParameters.map { it.hasDefault })
    val suggestions = CppCompletionGrammar().completeCppStatement(
      context,
      snapshot.completionQuery(context.identifiers, seed = 19)
    ).suggestions

    assertTrue(suggestions.isNotEmpty())
    assertTrue(suggestions.all { it.candidateText.startsWith("work::Box<") })
    assertTrue(suggestions.all { "<%" !in it.candidateText && "<:" !in it.candidateText })
    assertTrue(suggestions.any { "work::Box<Visible>" in it.candidateText })
    val language = CppCompletionGrammar().generate(context.copy(requiredIdentifier = "fresh"), emptyList())
    assertTrue(language.recognizes(cppLines("work::Box<Visible>* fresh;").single().tokens))
    assertTrue(language.recognizes(cppLines("work::Box<std::from_range_t>* fresh;").single().tokens))
    assertFalse(language.recognizes(cppLines("work::Box<from_range_t>* fresh;").single().tokens))
  }

  @Test
  fun dependentActiveTemplateUsesSemaValuesPacksAndEmptyAggregates() {
    val source = "Result rendered = apply("
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 0, source.length))
    val result = js(
      """({schemaVersion: 1, context: {kind: 'Expression'}, items: [
        {name: 'Result', insertText: 'Result', kind: 7, symbols: [{
          id: 'result-type', qualifiedName: 'nova::Result', kind: 'CXXRecord',
          provenance: {sema: true, index: false}, isType: true,
          type: 'nova::Result', canonicalType: 'nova::Result',
          typeInfo: {canonicalId: 'type:result', isSourceSpellable: true}
        }]},
        {name: 'Describe', insertText: 'Describe', kind: 7, symbols: [{
          id: 'visitor-type', qualifiedName: 'nova::Describe', kind: 'CXXRecord',
          provenance: {sema: true, index: false}, isType: true, isEmptyAggregate: true,
          type: 'nova::Describe', canonicalType: 'nova::Describe',
          typeInfo: {canonicalId: 'type:visitor', isSourceSpellable: true}
        }]},
        {name: 'payload', insertText: 'payload', kind: 6, symbols: [{
          id: 'payload', qualifiedName: 'payload', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          type: 'nova::Payload', canonicalType: 'nova::Payload',
          typeInfo: {canonicalId: 'type:payload', isSourceSpellable: true}
        }]},
        {name: 'textual', insertText: 'textual', kind: 6, symbols: [{
          provenance: {sema: true, index: false}, isValue: true,
          type: 'bool', canonicalType: 'bool',
          typeInfo: {canonicalId: 'type:bool', isSourceSpellable: true}
        }]},
        {name: 'text', insertText: 'text', kind: 6, symbols: [{
          provenance: {sema: true, index: false}, isValue: true,
          type: 'const nova::Text*', canonicalType: 'const nova::Text*',
          typeInfo: {canonicalId: 'type:text-pointer', isSourceSpellable: true}
        }]},
        {name: 'display', insertText: 'display', kind: 6, symbols: [{
          provenance: {sema: true, index: false}, isValue: true,
          type: 'nova::Text', canonicalType: 'nova::Text',
          typeInfo: {canonicalId: 'type:text', isSourceSpellable: true}
        }]},
        {name: 'nickname', insertText: 'nickname', kind: 6, symbols: [{
          provenance: {sema: true, index: false}, isValue: true,
          type: 'nova::OptionalText', canonicalType: 'nova::OptionalText',
          typeInfo: {canonicalId: 'type:optional-text', isSourceSpellable: true}
        }]}
      ], activeArgument: 0, activeCallIsBraced: false, activeCallables: [{
        id: 'apply-template', qualifiedName: 'nova::apply', kind: 'FunctionTemplate',
        provenance: {sema: true, index: false}, isCallable: true,
        returnType: 'decltype(auto)', canonicalReturnType: 'decltype(auto)',
        returnTypeInfo: {isDependent: true, isInstantiationDependent: true},
        parameters: [
          {name: 'operation', type: 'Visitor&&', canonicalType: 'Visitor&&',
           typeInfo: {kind: 'rvalueReference', isDependent: true, isInstantiationDependent: true}},
          {name: 'states', type: 'States&&...', canonicalType: 'States&&...', isPack: true,
           typeInfo: {kind: 'rvalueReference', isDependent: true, isInstantiationDependent: true}}
        ], templateParameters: [
          {name: 'Visitor', kind: 'type', isPack: false},
          {name: 'States', kind: 'type', isPack: true}
        ]
      }]})"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    assertTrue(context.types.single { it.name == "Describe" }.emptyAggregate)
    val callable = context.functions.single { it.activeCallable }
    assertTrue(callable.parameters[0].typeInfo?.isDependent == true)
    assertTrue(callable.parameters[1].isPack)

    listOf(source, "apply(", "auto outcome = apply(").forEach { prefix ->
      val cursor = assertNotNull(cppEditorStatementSnapshot(prefix, 0, prefix.length))
      val facts = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, cursor))
      val insertions = CppCompletionGrammar().completeCppStatement(
        facts,
        cursor.completionQuery(facts.identifiers)
      ).suggestions.map { it.candidateText.removePrefix(prefix) }
      assertTrue("Describe{},payload);" in insertions, "$prefix -> $insertions")
      assertTrue(insertions.none { "true" in it || "0" in it }, "$prefix -> $insertions")
    }
  }

  @Test
  fun activeSemaOverloadsCompleteArgumentsWithoutSignatureHelp() {
    // The source uses a using-declaration; the Sema identity remains fully qualified.
    val source = "fold("
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 0, source.length))
    val result = js(
      """({schemaVersion: 1, context: {kind: 'Expression'}, items: [
        {name: 'visitor', insertText: 'visitor', kind: 6, symbols: [{
          id: 'visitor', qualifiedName: 'visitor', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          type: 'nova::Visitor', canonicalType: 'nova::Visitor',
          typeInfo: {canonicalId: 'type:visitor', isSourceSpellable: true}
        }]},
        {name: 'payload', insertText: 'payload', kind: 6, symbols: [{
          id: 'payload', qualifiedName: 'payload', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          type: 'nova::Payload', canonicalType: 'nova::Payload',
          typeInfo: {canonicalId: 'type:payload', isSourceSpellable: true}
        }]}
      ], activeArgument: 0, activeCallIsBraced: false, activeCallables: [{
        id: 'fold', qualifiedName: 'nova::fold', kind: 'Function',
        provenance: {sema: true, index: false}, isCallable: true,
        returnType: 'int', canonicalReturnType: 'int',
        returnTypeInfo: {canonicalId: 'type:int', isSourceSpellable: true},
        parameters: [
          {name: 'operation', type: 'nova::Visitor', canonicalType: 'nova::Visitor',
           typeInfo: {canonicalId: 'type:visitor', isSourceSpellable: true}},
          {name: 'state', type: 'nova::Payload', canonicalType: 'nova::Payload',
           typeInfo: {canonicalId: 'type:payload', isSourceSpellable: true}}
        ]
      }]})"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val callable = context.functions.single { it.qualifiedName == "nova::fold" }
    assertEquals(listOf("type:visitor", "type:payload"),
      callable.parameters.map { it.typeInfo?.canonicalId })
    assertEquals("type:int", callable.returnTypeInfo?.canonicalId)

    val insertions = CppCompletionGrammar().completeCppStatement(
      context,
      snapshot.completionQuery(context.identifiers)
    ).suggestions.map { it.candidateText.removePrefix(source) }
    assertTrue("visitor,payload);" in insertions, insertions.toString())
  }

  @Test
  fun semanticCompletionPreservesSemaTypesOverloadsAndIndexProvenance() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("neb", 0, 3))
    val result = js(
      """({
        schemaVersion: 1,
        context: {
          kind: 'Expression',
          preferredType: 'cosmos::Reading',
          canonicalPreferredType: 'const cosmos::Reading'
        },
        items: [
          {
            name: 'ambientFlux', requiredQualifier: 'cosmos::',
            insertText: 'cosmos::ambientFlux', kind: 6,
            symbols: [{
              id: 'sema-variable', qualifiedName: 'cosmos::ambientFlux', kind: 'Var',
              provenance: {sema: true, index: false},
              isType: false, isValue: true, isCallable: false, isMember: false,
              type: 'const cosmos::Reading &',
              canonicalType: 'const cosmos::Reading &',
              typeInfo: {canonicalId: 'type:reading-ref', isSourceSpellable: true}
            }]
          },
          {
            name: 'Nebula', requiredQualifier: 'cosmos::',
            insertText: 'cosmos::Nebula', kind: 7,
            symbols: [{
              id: 'sema-type', qualifiedName: 'cosmos::Nebula', kind: 'CXXRecord',
              provenance: {sema: true, index: true},
              isType: true, isValue: false, isCallable: false, isMember: false,
              type: 'cosmos::Nebula', canonicalType: 'cosmos::Nebula',
              typeInfo: {canonicalId: 'type:nebula', isSourceSpellable: true}
            }]
          },
          {
            name: 'transmute', requiredQualifier: 'cosmos::',
            insertText: 'cosmos::transmute', signature: '(cosmos::Reading, int)',
            returnType: 'cosmos::Nebula', kind: 3,
            symbols: [
              {
                id: 'sema-overload-one', qualifiedName: 'cosmos::transmute', kind: 'Function',
                provenance: {sema: true, index: true},
                isType: false, isValue: true, isCallable: true, isMember: false,
                returnType: 'cosmos::Nebula', canonicalReturnType: 'cosmos::Nebula',
                returnTypeInfo: {canonicalId: 'type:nebula', isSourceSpellable: true},
                parameters: [
                  {name: 'reading', type: 'cosmos::Reading',
                   canonicalType: 'cosmos::Reading', hasDefault: false, isPack: false,
                   typeInfo: {canonicalId: 'type:reading', isSourceSpellable: true}},
                  {name: 'retries', type: 'int', canonicalType: 'int',
                   hasDefault: true, isPack: false,
                   typeInfo: {canonicalId: 'type:int', isSourceSpellable: true}}
                ]
              },
              {
                id: 'sema-overload-two', qualifiedName: 'cosmos::transmute', kind: 'Function',
                provenance: {sema: true, index: false},
                isType: false, isValue: true, isCallable: true, isMember: false,
                returnType: 'bool', canonicalReturnType: 'bool',
                returnTypeInfo: {canonicalId: 'type:bool', isSourceSpellable: true},
                parameters: [{name: 'nebula', type: 'const cosmos::Nebula &',
                  canonicalType: 'const cosmos::Nebula &', hasDefault: false, isPack: false,
                  typeInfo: {canonicalId: 'type:nebula-ref', isSourceSpellable: true}}]
              }
            ]
          },
          {
            name: 'remotePulse', requiredQualifier: 'catalog::',
            insertText: 'catalog::remotePulse', signature: '(catalog::Key)',
            returnType: 'catalog::Pulse', kind: 3,
            symbols: [{
              id: 'index-function', qualifiedName: 'catalog::remotePulse', kind: 'Function',
              provenance: {sema: false, index: true},
              isType: false, isValue: true, isCallable: true, isMember: false
            }]
          },
          {
            name: 'fabricatedLexeme', insertText: 'fabricatedLexeme', kind: 6,
            symbols: []
          }
        ]
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))

    assertEquals(setOf(
      "Nebula", "Reading", "ambientFlux", "bool", "catalog", "const",
      "cosmos", "int", "remotePulse", "transmute"
    ),
      context.identifiers)
    assertFalse("neb" in context.identifiers)
    assertFalse("fabricatedLexeme" in context.identifiers)
    assertEquals(setOf("cosmos", "ambientFlux", "Nebula", "transmute"), context.sourceIdentifiers)
    assertEquals(setOf("cosmos::Nebula"), context.typeNames)
    assertEquals(setOf("cosmos::Reading", "const cosmos::Reading"), context.expectedTypes)

    val value = context.values.single()
    assertEquals("cosmos::ambientFlux", value.name)
    assertEquals("const cosmos::Reading &", value.type)
    assertEquals("const cosmos::Reading &", value.canonicalType)
    assertEquals("sema", value.source)
    assertEquals("sema", value.provenance)
    assertEquals("sema-variable", value.id)
    assertEquals("cosmos::ambientFlux", value.qualifiedName)
    assertEquals(true, value.isValue)
    assertEquals(false, value.isCallable)
    assertNull(value.ownerType)

    val type = context.types.single()
    assertEquals("cosmos::Nebula", type.name)
    assertEquals("cosmos::Nebula", type.canonicalType)
    assertEquals(true, type.isType)
    assertEquals("sema+index", type.source)

    val overloads = context.functions.filter { it.name == "cosmos::transmute" }
    assertEquals(2, overloads.size)
    assertEquals(listOf("cosmos::Reading", "int"), overloads[0].parameters.map { it.type })
    assertEquals(listOf("reading", "retries"), overloads[0].parameters.map { it.name })
    assertEquals(listOf("cosmos::Reading", "int"),
      overloads[0].parameters.map { it.canonicalType })
    assertEquals(listOf(false, true), overloads[0].parameters.map { it.hasDefault })
    assertEquals("", overloads[0].parameters[1].defaultValue)
    assertEquals("cosmos::Nebula", overloads[0].canonicalReturnType)
    assertEquals("bool", overloads[1].returnType)
    assertTrue(overloads.none { it.ownerType != null || it.receiverMember })

    val indexed = context.completions.single { it.name == "catalog::remotePulse" }
    assertEquals("index", indexed.source)
    assertEquals("index", indexed.provenance)
    assertNull(indexed.returnType)
    assertTrue(indexed.parameters.isEmpty())
    assertEquals(false, indexed.isCallable)
    assertFalse(indexed in context.functions)
    assertFalse(indexed in context.values)
  }

  @Test
  fun semanticCompletionBuildsReceiverMembersFromTheirDeclaringTypes() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("probe.scan", 0, "probe.scan".length))
    val result = js(
      """({
        schemaVersion: 1,
        context: {
          kind: 'DotMemberAccess',
          baseType: 'const galaxy::Probe',
          canonicalBaseType: 'const galaxy::Probe',
          preferredType: 'galaxy::Spectrum',
          canonicalPreferredType: 'galaxy::Spectrum',
          queryScopes: ['galaxy::detail::', 'galaxy::'],
          accessibleScopes: ['', 'galaxy::']
        },
        items: [{
          name: 'scanBands', insertText: 'scanBands', signature: '(galaxy::Band, unsigned)',
          returnType: 'galaxy::Spectrum', kind: 2,
          symbols: [
            {
              id: 'member-overload-one', qualifiedName: 'galaxy::Probe::scanBands',
              kind: 'CXXMethod', provenance: {sema: true, index: false},
              isType: false, isValue: true, isCallable: true, isMember: true,
              isStatic: false, isVariadic: false,
              ownerType: 'galaxy::Probe', canonicalOwnerType: 'galaxy::Probe',
              ownerTypeInfo: {canonicalId: 'type:probe', isSourceSpellable: true},
              returnType: 'galaxy::Spectrum', canonicalReturnType: 'galaxy::Spectrum',
              returnTypeInfo: {canonicalId: 'type:spectrum', isSourceSpellable: true},
              parameters: [
                {name: 'band', type: 'galaxy::Band', canonicalType: 'galaxy::Band',
                 hasDefault: false, isPack: false,
                 typeInfo: {canonicalId: 'type:band', isSourceSpellable: true}},
                {name: 'passes', type: 'unsigned int', canonicalType: 'unsigned int',
                 hasDefault: true, isPack: false,
                 typeInfo: {canonicalId: 'type:uint', isSourceSpellable: true}}
              ]
            },
            {
              id: 'member-overload-two', qualifiedName: 'galaxy::Probe::scanBands',
              kind: 'CXXMethod', provenance: {sema: true, index: false},
              isType: false, isValue: true, isCallable: true, isMember: true,
              isStatic: false, isVariadic: true,
              ownerType: 'galaxy::Probe', canonicalOwnerType: 'galaxy::Probe',
              ownerTypeInfo: {canonicalId: 'type:probe', isSourceSpellable: true},
              returnType: 'galaxy::Spectrum', canonicalReturnType: 'galaxy::Spectrum',
              returnTypeInfo: {canonicalId: 'type:spectrum', isSourceSpellable: true},
              parameters: [{name: 'bands', type: 'galaxy::Band', canonicalType: 'galaxy::Band',
                hasDefault: false, isPack: true,
                typeInfo: {canonicalId: 'type:band', isSourceSpellable: true}}]
            }
          ]
        }]
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val receiver = assertNotNull(context.receiver)

    assertEquals(".", receiver.operator)
    assertEquals("probe", receiver.expression)
    assertEquals("const galaxy::Probe", receiver.type)
    assertEquals(2, receiver.members.size)
    assertEquals(setOf("Band", "Probe", "Spectrum", "galaxy", "int", "scanBands", "unsigned"),
      context.identifiers)
    assertFalse("probe" in context.identifiers)
    assertTrue(context.values.isEmpty())
    assertEquals(2, context.functions.size)
    assertEquals(2, context.membersByType.single { it.type == "galaxy::Probe" }.members.size)
    receiver.members.forEach { member ->
      assertEquals("galaxy::Probe", member.ownerType)
      assertTrue(member.receiverMember)
      assertEquals("sema", member.source)
      assertEquals(true, member.isMember)
    }
    assertEquals(true, receiver.members.single { it.id == "member-overload-two" }.isVariadic)
    assertEquals(true,
      receiver.members.single { it.id == "member-overload-two" }.parameters.single().isPack)
    assertEquals("DotMemberAccess", context.completionKind)
    assertEquals("galaxy::Spectrum", context.preferredType)
    assertEquals("galaxy::Spectrum", context.canonicalPreferredType)
    assertEquals("const galaxy::Probe", context.baseType)
    assertEquals("const galaxy::Probe", context.canonicalBaseType)
    assertEquals(listOf("galaxy::detail::", "galaxy::"), context.queryScopes)
    assertEquals(listOf("", "galaxy::"), context.accessibleScopes)
  }

  @Test
  fun schemaTwoReceiverMembersRequireTheExactCanonicalOwnerId() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("probe.", 0, 6))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'DotMemberAccess',
          baseType: 'const Probe', canonicalBaseType: 'const Probe',
          baseTypeInfo: {canonicalId: 'type:const-probe', valueCanonicalId: 'type:probe',
            kind: 'record', isConst: true, isSourceSpellable: true}}, items: [{
        name: 'inspect', insertText: 'inspect', kind: 2, symbols: [{
          id: 'probe-inspect', qualifiedName: 'Probe::inspect', kind: 'CXXMethod',
          provenance: {sema: true, index: false}, isCallable: true, isMember: true,
          ownerType: 'Probe', canonicalOwnerType: 'Probe',
          ownerTypeInfo: {canonicalId: 'type:probe', valueCanonicalId: 'type:probe',
            kind: 'record', isSourceSpellable: true},
          returnType: 'void', canonicalReturnType: 'void',
          returnTypeInfo: {canonicalId: 'type:void', valueCanonicalId: 'type:void',
            kind: 'builtin', isSourceSpellable: true}, parameters: []
        }]
      }, {
        name: 'foreign', insertText: 'foreign', kind: 2, symbols: [{
          id: 'other-foreign', qualifiedName: 'Other::foreign', kind: 'CXXMethod',
          provenance: {sema: true, index: false}, isCallable: true, isMember: true,
          ownerType: 'Other', canonicalOwnerType: 'Other',
          ownerTypeInfo: {canonicalId: 'type:other', valueCanonicalId: 'type:other',
            kind: 'record', isSourceSpellable: true},
          returnType: 'void', canonicalReturnType: 'void',
          returnTypeInfo: {canonicalId: 'type:void', valueCanonicalId: 'type:void',
            kind: 'builtin', isSourceSpellable: true}, parameters: []
        }]
      }, {
        name: 'ownerless', insertText: 'ownerless', kind: 2, symbols: [{
          id: 'ownerless-method', qualifiedName: 'ownerless', kind: 'CXXMethod',
          provenance: {sema: true, index: false}, isCallable: true, isMember: true,
          returnType: 'void', canonicalReturnType: 'void',
          returnTypeInfo: {canonicalId: 'type:void', valueCanonicalId: 'type:void',
            kind: 'builtin', isSourceSpellable: true}, parameters: []
        }]
      }], graph: {limit: 8, depth: 1, isIncomplete: false, nodes: []},
        operations: {limit: 8, depth: 2, isIncomplete: false, nodes: [], templates: [],
          conversions: [], expressionWitnesses: [], callWitnesses: []}})"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val receiver = assertNotNull(context.receiver)
    assertEquals(listOf("probe-inspect"), receiver.members.mapNotNull { it.id })
    assertEquals(setOf("probe-inspect", "other-foreign"),
      context.functions.mapNotNull { it.id }.toSet())
    assertFalse(context.completions.any { it.id == "ownerless-method" })
  }

  @Test
  fun semanticCompletionUsesOnlyDeclarationBackedNamesForPartialTokens() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("st", 0, 2))
    val result = js(
      """({schemaVersion: 1, context: {kind: 'Expression'}, items: [
        {name: 'starlight', insertText: 'starlight', kind: 6, symbols: [{
          qualifiedName: 'starlight', kind: 'Var', provenance: {sema: true, index: false},
          isValue: true, type: 'astral::Luminosity',
          typeInfo: {canonicalId: 'type:luminosity', isSourceSpellable: true}
        }]},
        {name: 'hiddenCallable', insertText: 'hiddenCallable', kind: 6, symbols: [{
          qualifiedName: 'hiddenCallable', kind: 'Var', provenance: {sema: true, index: false},
          isValue: true, type: 'lambda::ClosureArtifact',
          typeInfo: {canonicalId: 'type:closure', isSourceSpellable: false}
        }]},
        {name: 'dependentValue', insertText: 'dependentValue', kind: 6, symbols: [{
          qualifiedName: 'dependentValue', kind: 'Var', provenance: {sema: true, index: false},
          isValue: true, type: 'meta::DependentToken',
          typeInfo: {canonicalId: 'type:dependent', isSourceSpellable: true,
                     isDependent: true, isInstantiationDependent: true}
        }]},
        {name: 'stowaway', insertText: 'stowaway', kind: 6, symbols: []}
      ], scopeItems: [
        {name: 'stream', requiredQualifier: 'stellar::', insertText: 'stellar::stream',
         kind: 3, returnType: 'void', signature: '(catalog::SignatureGhost)', symbols: [{
          id: 'index-stream', qualifiedName: 'stellar::stream', kind: 'Function',
          provenance: {sema: false, index: true},
          returnType: 'catalog::IndexOnlyResult',
          canonicalReturnType: 'catalog::IndexOnlyResult'
        }]}
      ]})"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val query = snapshot.completionQuery(context.identifiers)

    assertEquals("st", query.tokenPrefix?.text)
    assertEquals(setOf(
      "Luminosity", "astral", "dependentValue", "hiddenCallable", "starlight", "stellar", "stream"
    ),
      query.identifiersInFile)
    assertEquals(query.identifiersInFile, context.identifiers)
    assertTrue(setOf("hiddenCallable", "dependentValue", "stellar", "stream")
      .all { it in context.identifiers })
    assertTrue(setOf(
      "lambda", "ClosureArtifact", "meta", "DependentToken", "catalog", "IndexOnlyResult",
      "SignatureGhost", "void"
    ).none { it in context.identifiers })
    assertFalse("st" in context.identifiers)
    assertFalse("stowaway" in context.identifiers)
  }

  @Test
  fun schemaTwoGraphUsesTheExactSemaReferencePathAndPreservesCompleteness() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("nova::", 0, 6))
    val result = js(
      """({
        schemaVersion: 2,
        context: {kind: 'Expression'},
        items: [{name: 'Orbit', insertText: 'nova::Orbit', kind: 7, symbols: [{
          id: 'orbit', qualifiedName: 'nova::Orbit', kind: 'CXXRecord',
          provenance: {sema: true, index: true}, isType: true,
          type: 'nova::Orbit', canonicalType: 'nova::Orbit',
          typeInfo: {canonicalId: 'type:orbit', isSourceSpellable: true}
        }]}],
        graph: {limit: 8, depth: 1, isIncomplete: true, nodes: [{
          name: 'nova::Orbit', id: 'orbit', qualifiedName: 'nova::Orbit', kind: 'CXXRecord',
          provenance: {sema: true, index: true}, isType: true,
          type: 'nova::Orbit', canonicalType: 'nova::Orbit',
          typeInfo: {canonicalId: 'type:orbit', isSourceSpellable: true}
        }, {
          name: 'nova::launch', id: 'launch', qualifiedName: 'nova::launch', kind: 'Function',
          provenance: {sema: true, index: false}, isValue: true, isCallable: true,
          returnType: 'nova::Orbit', canonicalReturnType: 'nova::Orbit',
          returnTypeInfo: {canonicalId: 'type:orbit', isSourceSpellable: true},
          parameters: [{name: 'altitude', type: 'double', canonicalType: 'double',
            typeInfo: {canonicalId: 'type:double', isSourceSpellable: true}}]
        }]}
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    assertEquals(2, context.semanticGraphNodeCount)
    assertTrue(context.semanticGraphIsIncomplete)
    assertEquals(1, context.completions.count { it.id == "orbit" })
    val launch = context.functions.single { it.id == "launch" }
    assertEquals("nova::launch", launch.name)
    assertEquals("nova::Orbit", launch.returnType)
    assertEquals("double", launch.parameters.single().type)
    assertEquals("type:double", launch.parameters.single().typeInfo?.canonicalId)
    assertEquals("sema", launch.provenance)
    assertTrue(setOf("nova", "Orbit", "launch", "double").all(context.identifiers::contains))
  }

  @Test
  fun graphClassSpecializationsAppendQualTypeArgumentsToTheAuthenticatedRoute() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("using Local = ", 0, 14))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Type'}, items: [],
        graph: {limit: 8, depth: 1, isIncomplete: false, nodes: [{
          name: 'facade::Facet', id: 'facet-char', qualifiedName: 'physical::Facet',
          kind: 'ClassTemplateSpecialization', provenance: {sema: true, index: false},
          isType: true, type: 'physical::Facet<char>',
          canonicalType: 'physical::Facet<char>',
          typeInfo: {canonicalId: 'type:facet-char', valueCanonicalId: 'type:facet-char',
            kind: 'record', isDependent: false, isInstantiationDependent: false,
            isSourceSpellable: true}, templateParameters: []
        }]}
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val specialization = context.types.single()
    assertEquals("facade::Facet<char>", specialization.name)
    assertEquals("physical::Facet", specialization.qualifiedName)
  }

  @Test
  fun unqualifiedGraphPathsRequireExactItemSpellingAndTemplatesRequireItemEvidence() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("", 0, 0))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Expression'}, items: [{
          name: 'Thing', insertText: 'nova::Thing', kind: 7, symbols: [{
            id: 'thing', qualifiedName: 'nova::Thing', kind: 'CXXRecord',
            provenance: {sema: true, index: true}, isType: true,
            type: 'nova::Thing', canonicalType: 'nova::Thing',
            typeInfo: {canonicalId: 'type:thing', isSourceSpellable: true}
          }]
        }], graph: {limit: 8, depth: 1, isIncomplete: false, nodes: [{
          name: 'Thing', id: 'thing', qualifiedName: 'nova::Thing', kind: 'CXXRecord',
          provenance: {sema: true, index: true}, isType: true,
          type: 'nova::Thing', canonicalType: 'nova::Thing',
          typeInfo: {canonicalId: 'type:thing', isSourceSpellable: true}
        }, {
          name: 'nova::HiddenTemplate', id: 'hidden-template',
          qualifiedName: 'nova::HiddenTemplate', kind: 'ClassTemplate',
          provenance: {sema: true, index: false}, isType: true,
          type: 'nova::HiddenTemplate<T>', canonicalType: 'nova::HiddenTemplate<T>',
          typeInfo: {canonicalId: 'template:hidden', isDependent: true,
            isInstantiationDependent: true, isSourceSpellable: false},
          templateParameters: [{name: 'T', kind: 'type', hasDefault: false, isPack: false}]
        }]}
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    assertEquals(listOf("nova::Thing"), context.types.map { it.name })
  }

  @Test
  fun operationTypeIdentityPromotesOnlyTheMatchingQualifiedGraphAlias() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("", 0, 0))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Statement'}, items: [],
        graph: {limit: 8, depth: 1, isIncomplete: false, nodes: [{
          name: 'nova::VisibleText', id: 'visible-text-alias',
          qualifiedName: 'nova::VisibleText', kind: 'TypeAlias',
          provenance: {sema: true, index: true}, isType: true,
          type: 'nova::VisibleText', canonicalType: 'nova::Text<char>',
          typeInfo: {canonicalId: 'type:text-char', valueCanonicalId: 'type:text-char',
            kind: 'record', isDependent: false, isInstantiationDependent: false,
            isSourceSpellable: true, isComplete: true}
        }, {
          name: 'nova::DormantText', id: 'dormant-text-alias',
          qualifiedName: 'nova::DormantText', kind: 'TypeAlias',
          provenance: {sema: true, index: true}, isType: true,
          type: 'nova::DormantText', canonicalType: 'nova::Text<long>',
          typeInfo: {canonicalId: 'type:text-long', valueCanonicalId: 'type:text-long',
            kind: 'record', isDependent: false, isInstantiationDependent: false,
            isSourceSpellable: true, isComplete: true}
        }]}, operations: {limit: 8, depth: 2, isIncomplete: false, nodes: [{
          name: 'nova::Text<char>', role: 'type', id: 'type:text-char',
          qualifiedName: 'nova::Text<char>', kind: 'Type',
          provenance: {sema: true, index: false}, isType: true,
          type: 'nova::Text<char>', canonicalType: 'nova::Text<char>',
          typeInfo: {canonicalId: 'type:text-char', valueCanonicalId: 'type:text-char',
            kind: 'record', isDependent: false, isInstantiationDependent: false,
            isSourceSpellable: true, isComplete: true}
        }], conversions: []}}
      )"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    assertTrue(context.types.single { it.name == "nova::VisibleText" }.completionVisible)
    assertFalse(context.types.single { it.name == "nova::DormantText" }.completionVisible)
  }

  @Test
  fun staticOperationMembersUseTheirExactSourceOwnerInsteadOfPhysicalIdentity() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("", 0, 0))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Statement'}, items: [{
          name: 'text', insertText: 'text', kind: 6, symbols: [{
            id: 'text-value', qualifiedName: 'text', kind: 'Var',
            provenance: {sema: true, index: false}, isValue: true,
            type: 'public_api::Text', canonicalType: 'physical::Text',
            typeInfo: {canonicalId: 'type:text', valueCanonicalId: 'type:text',
              kind: 'record', isSourceSpellable: true, isComplete: true}
          }]
        }], graph: {limit: 8, depth: 1, isIncomplete: false, nodes: []},
        operations: {limit: 8, depth: 2, isIncomplete: false, nodes: [{
          name: 'public_api::Text', role: 'type', id: 'type:text',
          qualifiedName: 'physical::Text', kind: 'Type',
          provenance: {sema: true, index: false}, isType: true,
          type: 'public_api::Text', canonicalType: 'physical::Text',
          typeInfo: {canonicalId: 'type:text', valueCanonicalId: 'type:text',
            kind: 'record', isSourceSpellable: true, isComplete: true}
        }, {
          name: 'limit', role: 'member', id: 'text-limit',
          qualifiedName: 'physical::Text::limit', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          isMember: true, isStatic: true,
          type: 'const unsigned long', canonicalType: 'const unsigned long',
          typeInfo: {canonicalId: 'type:const-size', valueCanonicalId: 'type:size',
            kind: 'builtin', isConst: true, isSourceSpellable: true, isComplete: true},
          ownerType: 'public_api::Text', canonicalOwnerType: 'physical::Text',
          ownerTypeInfo: {canonicalId: 'type:text', valueCanonicalId: 'type:text',
            kind: 'record', isSourceSpellable: true, isComplete: true}
        }, {
          name: 'hiddenLimit', role: 'member', id: 'hidden-text-limit',
          qualifiedName: 'physical::HiddenText::hiddenLimit', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          isMember: true, isStatic: true,
          type: 'unsigned long', canonicalType: 'unsigned long',
          typeInfo: {canonicalId: 'type:size', valueCanonicalId: 'type:size',
            kind: 'builtin', isSourceSpellable: true, isComplete: true},
          ownerType: 'physical::HiddenText', canonicalOwnerType: 'physical::HiddenText',
          ownerTypeInfo: {canonicalId: 'type:hidden-text',
            valueCanonicalId: 'type:hidden-text', kind: 'record',
            isSourceSpellable: false, isComplete: true}
        }], conversions: []}}
      )"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val staticValue = context.values.single { it.id == "text-limit" }
    val staticMember = context.membersByType.flatMap { it.members }
      .single { it.id == "text-limit" }
    val unroutableStatic = context.values.single { it.id == "hidden-text-limit" }
    assertEquals("public_api::Text::limit", staticValue.name)
    assertEquals("public_api::Text::limit", staticMember.name)
    assertEquals("physical::Text::limit", staticValue.qualifiedName)
    assertTrue(staticValue.completionVisible)
    assertEquals("hiddenLimit", unroutableStatic.name)
    assertFalse(unroutableStatic.completionVisible)

    val language = CppCompletionGrammar().generate(context, emptyList())
    fun recognizes(statement: String) =
      language.recognizes(cppLines(statement).single().tokens)
    assertTrue(recognizes("public_api::Text::limit;"))
    assertTrue(recognizes("text.limit;"))
    assertFalse(recognizes("physical::Text::limit;"))
    assertFalse(recognizes("limit;"))
    assertFalse(recognizes("hiddenLimit;"))
  }

  @Test
  fun graphMembersKeepTheirAuthenticatedRouteAndRequireExactOwnerIdentity() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("", 0, 0))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Statement'}, items: [],
        graph: {limit: 8, depth: 2, isIncomplete: false, nodes: [{
          name: 'facade::Widget::quota', id: 'widget-quota',
          qualifiedName: 'physical::Widget::quota', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          isMember: true, isStatic: true,
          type: 'unsigned long', canonicalType: 'unsigned long',
          typeInfo: {canonicalId: 'type:size', valueCanonicalId: 'type:size',
            kind: 'builtin', isSourceSpellable: true, isComplete: true},
          ownerType: 'public_api::Widget', canonicalOwnerType: 'physical::Widget',
          ownerTypeInfo: {canonicalId: 'type:widget', valueCanonicalId: 'type:widget',
            kind: 'record', isSourceSpellable: true, isComplete: true}
        }, {
          name: 'facade::Widget::ownerless', id: 'ownerless-member',
          qualifiedName: 'physical::Widget::ownerless', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          isMember: true, isStatic: true,
          type: 'int', canonicalType: 'int',
          typeInfo: {canonicalId: 'type:int', valueCanonicalId: 'type:int',
            kind: 'builtin', isSourceSpellable: true, isComplete: true}
        }, {
          name: 'facade::Widget::contradictory', id: 'contradictory-owner',
          qualifiedName: 'physical::Widget::contradictory', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          isMember: false, isStatic: false,
          type: 'int', canonicalType: 'int',
          typeInfo: {canonicalId: 'type:int', valueCanonicalId: 'type:int',
            kind: 'builtin', isSourceSpellable: true, isComplete: true},
          ownerType: 'public_api::Widget', canonicalOwnerType: 'physical::Widget',
          ownerTypeInfo: {canonicalId: 'type:widget', valueCanonicalId: 'type:widget',
            kind: 'record', isSourceSpellable: true, isComplete: true}
        }, {
          name: 'facade::Widget::unidentified', id: 'unidentified-owner',
          qualifiedName: 'physical::Widget::unidentified', kind: 'Var',
          provenance: {sema: true, index: false}, isValue: true,
          isMember: true, isStatic: true,
          type: 'int', canonicalType: 'int',
          typeInfo: {canonicalId: 'type:int', valueCanonicalId: 'type:int',
            kind: 'builtin', isSourceSpellable: true, isComplete: true},
          ownerType: 'public_api::Widget', canonicalOwnerType: 'physical::Widget',
          ownerTypeInfo: {kind: 'record', isSourceSpellable: true, isComplete: true}
        }]}, operations: {limit: 8, depth: 2, isIncomplete: false,
          nodes: [], templates: [], conversions: [], expressionWitnesses: [], callWitnesses: []}}
      )"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val member = context.values.single()
    assertEquals("widget-quota", member.id)
    assertEquals("facade::Widget::quota", member.name)
    assertEquals("physical::Widget::quota", member.qualifiedName)
    assertEquals("type:widget", member.ownerTypeInfo?.valueCanonicalId)
    assertTrue(member.completionVisible)
    assertEquals(listOf("widget-quota"), context.completions.mapNotNull { it.id })
  }

  @Test
  fun fieldObjectKindSurvivesEndpointAndWorkerBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("", 0, 0))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Statement'}, items: [],
        graph: {limit: 8, depth: 1, isIncomplete: false, nodes: []},
        operations: {limit: 8, depth: 2, isIncomplete: false, nodes: [{
          name: 'bits', role: 'member', id: 'state-bits',
          qualifiedName: 'State::bits', kind: 'FieldDecl',
          provenance: {sema: true, index: false}, isValue: true, isMember: true,
          isBitField: true, type: 'unsigned int', canonicalType: 'unsigned int',
          typeInfo: {canonicalId: 'type:unsigned', valueCanonicalId: 'type:unsigned',
            kind: 'builtin', isSourceSpellable: true, isComplete: true},
          ownerType: 'State', canonicalOwnerType: 'State',
          ownerTypeInfo: {canonicalId: 'type:state', valueCanonicalId: 'type:state',
            kind: 'record', isSourceSpellable: true, isComplete: true}
        }, {
          name: 'value', role: 'member', id: 'state-value',
          qualifiedName: 'State::value', kind: 'FieldDecl',
          provenance: {sema: true, index: false}, isValue: true, isMember: true,
          isBitField: false, type: 'long', canonicalType: 'long',
          typeInfo: {canonicalId: 'type:long', valueCanonicalId: 'type:long',
            kind: 'builtin', isSourceSpellable: true, isComplete: true},
          ownerType: 'State', canonicalOwnerType: 'State',
          ownerTypeInfo: {canonicalId: 'type:state', valueCanonicalId: 'type:state',
            kind: 'record', isSourceSpellable: true, isComplete: true}
        }], templates: [], conversions: [], expressionWitnesses: [], callWitnesses: []}}
      )"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val fields = context.membersByType.single().members.associateBy { it.name }
    assertEquals(true, fields.getValue("bits").isBitField)
    assertEquals(false, fields.getValue("value").isBitField)
  }

  @Test
  fun schemaTwoOperationsRemainTypedReceiverFactsAndExactConversions() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("vec", 0, 3))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Expression'}, items: [],
        graph: {limit: 8, depth: 1, isIncomplete: false, nodes: []},
        operations: {limit: 8, depth: 2, isIncomplete: false, nodes: [{
          name: 'push_back', role: 'member', id: 'push-int',
          qualifiedName: 'std::vector<int>::push_back', kind: 'CXXMethod',
          provenance: {sema: true, index: false}, isCallable: true, isMember: true,
          ownerType: 'std::vector<int>', canonicalOwnerType: 'std::vector<int>',
          ownerTypeInfo: {canonicalId: 'type:vector-int', valueCanonicalId: 'type:vector-int',
            kind: 'record', isSourceSpellable: true},
          returnType: 'void', canonicalReturnType: 'void',
          returnTypeInfo: {canonicalId: 'type:void', isSourceSpellable: true},
          parameters: [{name: 'value', type: 'const int&', canonicalType: 'const int&',
            typeInfo: {canonicalId: 'type:const-int-ref', isSourceSpellable: true}}]
        }, {
          name: 'make_box<int>', role: 'specialization', id: 'make-box-int',
          qualifiedName: 'nova::make_box<int>', kind: 'Function',
          provenance: {sema: true, index: false}, isCallable: true,
          returnType: 'nova::Box<int>', canonicalReturnType: 'nova::Box<int>',
          returnTypeInfo: {canonicalId: 'type:box-int', isSourceSpellable: true},
          parameters: []
        }, {
          name: 'nova::Derived', role: 'type', id: 'type:derived',
          qualifiedName: 'nova::Derived', kind: 'Type',
          provenance: {sema: true, index: false}, isType: true,
          type: 'nova::Derived', canonicalType: 'nova::Derived',
          typeInfo: {canonicalId: 'type:derived', valueCanonicalId: 'type:derived',
            kind: 'record', isSourceSpellable: true}
        }, {
          name: 'nova::Base', role: 'type', id: 'type:base',
          qualifiedName: 'nova::Base', kind: 'Type',
          provenance: {sema: true, index: false}, isType: true,
          type: 'nova::Base', canonicalType: 'nova::Base',
          typeInfo: {canonicalId: 'type:base', valueCanonicalId: 'type:base',
            kind: 'record', isSourceSpellable: true}
        }], conversions: [{kind: 'base', fromType: 'nova::Derived',
          canonicalFromType: 'nova::Derived',
          fromTypeInfo: {canonicalId: 'type:derived', valueCanonicalId: 'type:derived',
            kind: 'record', isConst: false, isVolatile: false, isDependent: false,
            isInstantiationDependent: false, isSourceSpellable: true},
          toType: 'nova::Base', canonicalToType: 'nova::Base',
          toTypeInfo: {canonicalId: 'type:base', valueCanonicalId: 'type:base',
            kind: 'record', isConst: false, isVolatile: false, isDependent: false,
            isInstantiationDependent: false, isSourceSpellable: true}},
          {kind: 'constructor', fromType: 'const char *', canonicalFromType: 'const char *',
            fromTypeInfo: {canonicalId: 'type:c-string', valueCanonicalId: 'type:c-string',
              kind: 'pointer', isConst: false, isVolatile: false, isDependent: false,
              isInstantiationDependent: false, isSourceSpellable: true,
              pointeeCanonicalId: 'type:char', pointeeIsConst: true},
            toType: 'nova::Base', canonicalToType: 'nova::Base',
            toTypeInfo: {canonicalId: 'type:base', valueCanonicalId: 'type:base',
              kind: 'record', isConst: false, isVolatile: false, isDependent: false,
              isInstantiationDependent: false, isSourceSpellable: true}}]}
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val member = context.membersByType.single().members.single()
    assertEquals("push_back", member.name)
    assertEquals("std::vector<int>", member.ownerType)
    assertEquals("type:vector-int", member.ownerTypeInfo?.canonicalId)
    assertTrue(member.receiverMember)
    assertEquals("const int&", member.parameters.single().type)
    assertTrue(context.functions.none { it.id == "make-box-int" })
    assertEquals(
      listOf(cppcompletion.CppConversion(
        from = "nova::Derived",
        to = "nova::Base",
        kind = "base",
        canonicalFromType = "nova::Derived",
        canonicalToType = "nova::Base",
        fromTypeInfo = cppcompletion.CppTypeInfo(
          canonicalId = "type:derived",
          valueCanonicalId = "type:derived",
          kind = "record",
          isSourceSpellable = true
        ),
        toTypeInfo = cppcompletion.CppTypeInfo(
          canonicalId = "type:base",
          valueCanonicalId = "type:base",
          kind = "record",
          isSourceSpellable = true
        )
      )),
      context.conversions
    )
    assertEquals(setOf("type:derived", "type:base"),
      context.types.filter { it.name in setOf("nova::Derived", "nova::Base") }
        .mapNotNullTo(linkedSetOf()) { it.typeInfo?.canonicalId })
  }

  @Test
  fun operationRolesAndComponentCompletenessFailClosed() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("", 0, 0))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Statement'}, items: [],
        graph: {limit: 8, depth: 1, isIncomplete: false, nodes: []},
        operations: {limit: 8, depth: 1, isIncomplete: false,
          nodesIncomplete: false, templatesIncomplete: false, conversionsIncomplete: true,
          nodes: [{name: 'FutureThing', role: 'future-role', id: 'future',
            qualifiedName: 'FutureThing', kind: 'Type',
            provenance: {sema: true, index: false}, isType: true,
            type: 'FutureThing', canonicalType: 'FutureThing',
            typeInfo: {canonicalId: 'type:future', valueCanonicalId: 'type:future',
              kind: 'record', isSourceSpellable: true}}],
          templates: [], conversions: [], expressionWitnesses: [], callWitnesses: []}}
      )"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    assertTrue(context.completions.none { it.id == "future" })
    assertTrue(context.semanticOperationsAreIncomplete)
  }

  @Test
  fun conversionDtoKeepsLegacyEdgesButRejectsCorruptPresentPointerMetadata() {
    val context = cppCompletionContextFromDto(js(
      """({identifiers: [], conversions: [
        {from: 'LegacySource', to: 'LegacyTarget', kind: 'conversion'},
        {from: 'const char *', to: 'Text', kind: 'constructor',
          canonicalFromType: 'const char *', canonicalToType: 'Text',
          fromTypeInfo: {canonicalId: 'type:c-string', valueCanonicalId: 'type:c-string',
            kind: 'pointer', isConst: false, isVolatile: false, isDependent: false,
            isInstantiationDependent: false, isSourceSpellable: true,
            pointeeCanonicalId: 'type:char', pointeeIsConst: true},
          toTypeInfo: {canonicalId: 'type:text', valueCanonicalId: 'type:text',
            kind: 'record', isConst: false, isVolatile: false, isDependent: false,
            isInstantiationDependent: false, isSourceSpellable: true}}
      ]})"""
    ))

    assertEquals(
      listOf(cppcompletion.CppConversion(
        from = "LegacySource",
        to = "LegacyTarget",
        kind = "conversion"
      )),
      context.conversions
    )
  }

  @Test
  fun primaryOperationTemplatesRemainAdvisoryWhileNewSemaTraitsRoundTrip() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("", 0, 0))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Statement'}, items: [{
          name: 'Pipeline', insertText: 'Pipeline', kind: 7, symbols: [{
            id: 'pipeline-type', qualifiedName: 'Pipeline', kind: 'CXXRecord',
            provenance: {sema: true, index: false}, isType: true,
            type: 'Pipeline', canonicalType: 'Pipeline',
            typeInfo: {canonicalId: 'type:pipeline', valueCanonicalId: 'type:pipeline',
              kind: 'record', isSourceSpellable: true, isComplete: true,
              isDefaultConstructible: true}
          }]
        }], graph: {limit: 8, depth: 1, isIncomplete: false, nodes: []},
        operations: {limit: 8, depth: 2, isIncomplete: false, nodes: [{
          name: 'Pipeline', role: 'constructor', id: 'pipeline-int-constructor',
          qualifiedName: 'Pipeline::Pipeline', kind: 'CXXConstructor',
          provenance: {sema: true, index: false}, isCallable: true, isMember: true,
          isExplicit: true,
          ownerType: 'Pipeline', canonicalOwnerType: 'Pipeline',
          ownerTypeInfo: {canonicalId: 'type:pipeline', valueCanonicalId: 'type:pipeline',
            kind: 'record', isSourceSpellable: true, isComplete: true,
            isDefaultConstructible: true},
          returnType: 'Pipeline', canonicalReturnType: 'Pipeline',
          returnTypeInfo: {canonicalId: 'type:pipeline', valueCanonicalId: 'type:pipeline',
            kind: 'record', isSourceSpellable: true, isComplete: true,
            isDefaultConstructible: true},
          parameters: [{name: 'count', type: 'int', canonicalType: 'int',
            typeInfo: {canonicalId: 'type:int', valueCanonicalId: 'type:int',
              kind: 'builtin', isSourceSpellable: true, isComplete: true}}]
        }], templates: [{
          name: 'append', role: 'member', requiresCompilerSubstitution: true,
          minExplicitArguments: 0, hasFunctionParameterPack: true,
          pattern: {
            id: 'pipeline-append-primary', qualifiedName: 'Pipeline::append',
            kind: 'FunctionTemplate', provenance: {sema: true, index: false},
            isCallable: true, isMember: true,
            ownerType: 'Pipeline', canonicalOwnerType: 'Pipeline',
            ownerTypeInfo: {canonicalId: 'type:pipeline', valueCanonicalId: 'type:pipeline',
              kind: 'record', isSourceSpellable: true, isComplete: true,
              isDefaultConstructible: true},
            returnType: 'Pipeline &', canonicalReturnType: 'Pipeline &',
            returnTypeInfo: {canonicalId: 'type:pipeline-ref',
              valueCanonicalId: 'type:pipeline', kind: 'lvalueReference',
              isSourceSpellable: true, isComplete: true},
            parameters: [{name: 'items', type: 'Items&&...', canonicalType: 'Items&&...',
              isPack: true, typeInfo: {canonicalId: 'dependent:items',
                kind: 'rvalueReference', isDependent: true,
                isInstantiationDependent: true, isSourceSpellable: false}}],
            templateParameters: [{name: 'Items', kind: 'type', isPack: true}]
          }
        }], conversions: []}}
      )"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val pipeline = context.types.single { it.id == "pipeline-type" }
    assertEquals(true, pipeline.typeInfo?.isComplete)
    assertEquals(true, pipeline.typeInfo?.isDefaultConstructible)
    assertEquals(true,
      context.completions.single { it.id == "pipeline-int-constructor" }.isExplicit)

    val advisory = context.completions.single { it.id == "pipeline-append-primary" }
    assertEquals("primaryTemplateAdvisory", advisory.kind)
    assertEquals("member", advisory.detail)
    assertEquals("Pipeline", advisory.ownerType)
    assertEquals("Pipeline &", advisory.returnType)
    assertTrue(advisory.parameters.single().isPack)
    assertTrue(advisory.templateParameters.single().isPack)
    assertEquals(false, advisory.isCallable)
    assertEquals(false, advisory.isValue)
    assertEquals(false, advisory.isMember)
    assertFalse(context.functions.any { it.id == advisory.id })
    assertFalse(context.values.any { it.id == advisory.id })
    assertFalse(context.membersByType.flatMap { it.members }.any { it.id == advisory.id })
  }
}
