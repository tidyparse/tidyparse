import cppcompletion.cppLines
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticCallWitnessAdapterTest {
  @Test
  fun authoritativeArgumentVectorsSurviveTheWorkerDtoBoundaryWithoutFlattening() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
        graph: {nodes: [], isIncomplete: false}, operations: {
          nodes: [], templates: [], conversions: [], callWitnesses: [{
            name: 'insert', syntax: 'memberCall',
            validation: 'recursiveDefinitionInstantiation', authoritative: true,
            targetId: 'template:insert',
            primaryTemplateId: 'template:insert',
            receiver: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'Box', canonicalType: 'Box',
              valueCategory: 'lvalue', typeInfo: {canonicalId: 'type:box',
                valueCanonicalId: 'type:box', kind: 'record', isConst: false,
                isVolatile: false, isDependent: false, isInstantiationDependent: false,
                isSourceSpellable: true}},
            arguments: [
              {kind: 'integerZero', spelling: '0', objectKind: 'ordinary',
                type: 'int', canonicalType: 'int',
                valueCategory: 'prvalue', typeInfo: {canonicalId: 'type:int',
                  valueCanonicalId: 'type:int', kind: 'builtin', isConst: false,
                  isVolatile: false, isDependent: false, isInstantiationDependent: false,
                  isSourceSpellable: true}},
              {kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'Text', canonicalType: 'Text',
                valueCategory: 'lvalue', typeInfo: {canonicalId: 'type:text',
                  valueCanonicalId: 'type:text', kind: 'record', isConst: false,
                  isVolatile: false, isDependent: false, isInstantiationDependent: false,
                  isSourceSpellable: true}}
            ],
            callable: {id: 'specialization:insert', primaryTemplateId: 'template:insert',
              qualifiedName: 'Box::insert',
              kind: 'CXXMethod', provenance: {sema: true, index: false},
              isCallable: true, isMember: true, returnType: 'Box &',
              canonicalReturnType: 'Box &', ownerType: 'Box', canonicalOwnerType: 'Box',
              ownerTypeInfo: {canonicalId: 'type:box', valueCanonicalId: 'type:box',
                kind: 'record', isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false, isSourceSpellable: true},
              returnTypeInfo: {canonicalId: 'type:box-ref', valueCanonicalId: 'type:box',
                kind: 'lvalueReference', isConst: false, isVolatile: false,
                isDependent: false, isInstantiationDependent: false,
                isSourceSpellable: true}, parameters: []},
            result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'Box', canonicalType: 'Box',
              valueCategory: 'lvalue', typeInfo: {canonicalId: 'type:box-ref',
                valueCanonicalId: 'type:box', kind: 'lvalueReference',
                isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false,
                isSourceSpellable: true}}
          }]}
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val witness = context.callWitnesses.single()

    assertEquals(0, context.semanticOperationNodeCount)
    assertEquals(0, context.semanticOperationTemplateCount)
    assertEquals(false, context.semanticOperationsAreIncomplete)
    assertEquals(false, context.semanticCallWitnessesAreIncomplete)
    assertTrue(witness.authoritative)
    assertEquals("recursiveDefinitionInstantiation", witness.validation)
    assertEquals("template:insert", witness.targetId)
    assertEquals("template:insert", witness.primaryTemplateId)
    assertEquals("lvalue", witness.receiver?.valueCategory)
    assertEquals(listOf("integerZero", "opaque"), witness.arguments.map { it.kind })
    assertEquals(listOf("0", ""), witness.arguments.map { it.spelling })
    assertEquals(listOf("ordinary", "ordinary"), witness.arguments.map { it.objectKind })
    assertEquals(listOf("prvalue", "lvalue"), witness.arguments.map { it.valueCategory })
    assertEquals("type:text", witness.arguments[1].typeInfo?.valueCanonicalId)
    assertEquals("template:insert", witness.callable.primaryTemplateId)
    assertEquals("Box::insert", witness.callable.qualifiedName)
    assertEquals("lvalue", witness.result.valueCategory)
    // Keep a lexer use here so the test source remains linked to the same JS completion module.
    assertEquals(listOf("box", ".", "insert", "(", "0", ",", "text", ")", ";"),
      cppLines("box.insert(0, text);").single().tokens.map { it.text })
  }
}
