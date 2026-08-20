import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticOperatorWitnessAdapterTest {
  @Test
  fun binaryRelationsArePartitionedFromCallsAndPreserveSourceOperandOrder() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val result = js(
      """(() => {
        function typeInfo(id, kind = 'record') {
          return {
            canonicalId: id, valueCanonicalId: id, kind,
            isConst: false, isVolatile: false, isDependent: false,
            isInstantiationDependent: false, isSourceSpellable: true
          };
        }
        function profile(type, category) {
          return {
            kind: 'opaque', spelling: '', objectKind: 'ordinary', type,
            canonicalType: type, valueCategory: category,
            typeInfo: typeInfo('type:' + type, type === 'bool' ? 'builtin' : 'record')
          };
        }
        function callable(id, primaryTemplateId = '') {
          return {
            id, primaryTemplateId, qualifiedName: 'compare::operator<=>',
            kind: 'Function', provenance: {sema: true, index: false},
            isCallable: true, isMember: false,
            returnType: 'Ordering', canonicalReturnType: 'Ordering',
            returnTypeInfo: typeInfo('type:Ordering'),
            // Deliberately opposite to the source-side receiver/argument order.
            parameters: [
              {type: 'RightSurface', canonicalType: 'RightSurface',
               typeInfo: typeInfo('type:RightSurface')},
              {type: 'LeftSurface', canonicalType: 'LeftSurface',
               typeInfo: typeInfo('type:LeftSurface')}
            ]
          };
        }
        function witness(validation, targetId, selected, primaryTemplateId = '') {
          return {
            name: 'operator<', syntax: 'binaryOperator', operatorSpelling: '<',
            validation, authoritative: true, targetId, primaryTemplateId,
            explicitTemplateArguments: [],
            receiver: profile('LeftSurface', 'lvalue'),
            arguments: [profile('RightSurface', 'xvalue')],
            callable: selected,
            result: profile('bool', 'prvalue')
          };
        }
        return {
          schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
          graph: {nodes: [], isIncomplete: false},
          operations: {
            nodes: [], templates: [], conversions: [],
            binaryOperatorWitnessesIncomplete: true,
            callWitnesses: [
              witness('semaBinaryOperatorExpression', 'function:cmp',
                callable('function:cmp')),
              witness('semaDefaultedDefinition', 'function:cmp',
                callable('function:cmp')),
              witness('recursiveDefinitionInstantiation', 'template:cmp',
                callable('specialization:cmp', 'template:cmp'), 'template:cmp'),
              witness('overloadResolution', 'function:cmp', callable('function:cmp')),
              witness('semaBinaryOperatorExpression', 'function:other',
                callable('function:cmp'))
            ]
          }
        };
      })()"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))

    assertTrue(context.callWitnesses.isEmpty())
    assertEquals(3, context.binaryOperatorWitnesses.size)
    assertEquals(
      setOf(
        "semaBinaryOperatorExpression",
        "semaDefaultedDefinition",
        "recursiveDefinitionInstantiation"
      ),
      context.binaryOperatorWitnesses.mapTo(linkedSetOf()) { it.validation }
    )
    context.binaryOperatorWitnesses.forEach { witness ->
      assertEquals("<", witness.operatorSpelling)
      assertEquals("LeftSurface", witness.left.type)
      assertEquals("lvalue", witness.left.valueCategory)
      assertEquals("RightSurface", witness.right.type)
      assertEquals("xvalue", witness.right.valueCategory)
      assertEquals(listOf("RightSurface", "LeftSurface"),
        witness.callable.parameters.map { it.type })
      assertEquals("bool", witness.result.type)
      assertEquals("compare::operator<=>", witness.callable.qualifiedName)
    }
    val recursive = context.binaryOperatorWitnesses.single {
      it.validation == "recursiveDefinitionInstantiation"
    }
    assertEquals("template:cmp", recursive.targetId)
    assertEquals("template:cmp", recursive.primaryTemplateId)
    assertEquals("specialization:cmp", recursive.callable.id)
    assertEquals("template:cmp", recursive.callable.primaryTemplateId)
    context.binaryOperatorWitnesses
      .filterNot { it.validation == "recursiveDefinitionInstantiation" }
      .forEach { witness -> assertEquals(witness.targetId, witness.callable.id) }
    assertTrue(context.semanticBinaryOperatorWitnessesAreIncomplete)
  }
}
