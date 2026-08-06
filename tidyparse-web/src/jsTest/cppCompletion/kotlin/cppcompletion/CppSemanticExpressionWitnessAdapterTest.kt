import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticExpressionWitnessAdapterTest {
  @Test
  fun expressionWitnessesSurviveTheEndpointAndWorkerDtoBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
        graph: {nodes: [], isIncomplete: false}, operations: {
          nodes: [], templates: [], conversions: [], callWitnesses: [],
          expressionWitnessesIncomplete: true,
          expressionWitnesses: [{
            syntax: 'dynamicCast', validation: 'semaExpressionBuild', authoritative: true,
            typeOperand: {type: 'Leaf *', canonicalType: 'Leaf *', typeInfo: {
              canonicalId: 'type:leaf-pointer', valueCanonicalId: 'type:leaf-pointer',
              kind: 'pointer', pointeeCanonicalId: 'type:leaf',
              pointeeIsConst: false, pointeeIsVolatile: false,
              isConst: false, isVolatile: false, isDependent: false,
              isInstantiationDependent: false,
              isSourceSpellable: true, isComplete: true}},
            expressionOperand: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'Root *', canonicalType: 'Root *',
              valueCategory: 'lvalue', typeInfo: {canonicalId: 'type:root-pointer',
                valueCanonicalId: 'type:root-pointer', kind: 'pointer',
                pointeeCanonicalId: 'type:root', pointeeIsConst: false,
                pointeeIsVolatile: false, isConst: false, isVolatile: false,
                isDependent: false, isInstantiationDependent: false,
                isSourceSpellable: true, isComplete: true}},
            result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'Leaf *', canonicalType: 'Leaf *',
              valueCategory: 'prvalue', typeInfo: {canonicalId: 'type:leaf-pointer',
                valueCanonicalId: 'type:leaf-pointer', kind: 'pointer',
                pointeeCanonicalId: 'type:leaf', pointeeIsConst: false,
                pointeeIsVolatile: false, isConst: false, isVolatile: false,
                isDependent: false, isInstantiationDependent: false,
                isSourceSpellable: true, isComplete: true}}
          }, {
            syntax: 'typeidType', validation: 'semaExpressionBuild', authoritative: true,
            typeOperand: {type: 'Root', canonicalType: 'Root', typeInfo: {
              canonicalId: 'type:root', valueCanonicalId: 'type:root', kind: 'record',
              isConst: false, isVolatile: false, isDependent: false,
              isInstantiationDependent: false,
              isSourceSpellable: true, isComplete: true}},
            result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'const TypeInfo', canonicalType: 'const TypeInfo',
              valueCategory: 'lvalue', typeInfo: {canonicalId: 'type:const-type-info',
                valueCanonicalId: 'type:type-info', kind: 'record', isConst: true,
                isVolatile: false, isDependent: false, isInstantiationDependent: false,
                isSourceSpellable: true, isComplete: true}}
          }]}
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))

    assertTrue(context.semanticExpressionWitnessesAreIncomplete)
    assertEquals(listOf("dynamicCast", "typeidType"), context.expressionWitnesses.map { it.syntax })
    val cast = context.expressionWitnesses.first()
    assertTrue(cast.authoritative)
    assertEquals("semaExpressionBuild", cast.validation)
    assertEquals("Leaf *", cast.typeOperand?.type)
    assertEquals("type:leaf-pointer", cast.typeOperand?.typeInfo?.valueCanonicalId)
    assertEquals("lvalue", cast.expressionOperand?.valueCategory)
    assertEquals("", cast.expressionOperand?.spelling)
    assertEquals("ordinary", cast.expressionOperand?.objectKind)
    assertEquals("prvalue", cast.result.valueCategory)
    val typeid = context.expressionWitnesses.last()
    assertEquals(null, typeid.expressionOperand)
    assertEquals(true, typeid.result.typeInfo?.isConst)
  }

  @Test
  fun olderWorkerDtosDecodeWithoutExpressionWitnessFields() {
    val context = cppCompletionContextFromDto(js("({identifiers: ['legacy']})"))

    assertEquals(setOf("legacy"), context.identifiers)
    assertTrue(context.expressionWitnesses.isEmpty())
    assertFalse(context.semanticExpressionWitnessesAreIncomplete)
  }

  @Test
  fun expressionSpellingAndObjectKindCorruptionFailClosedAtBothBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val payloads = js(
      """(function() {
        function info(id, isConst) { return {canonicalId: id, valueCanonicalId: id,
          kind: 'record', isConst: !!isConst, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true}; }
        var root = info('type:root', false);
        var typeInfo = info('type:type-info', true);
        function result() { return {kind: 'opaque', spelling: '', objectKind: 'ordinary',
          type: 'const TypeInfo', canonicalType: 'const TypeInfo', valueCategory: 'lvalue',
          typeInfo: typeInfo}; }
        function witness(operand) { return {syntax: 'typeidExpression',
          validation: 'semaExpressionBuild', authoritative: true,
          expressionOperand: operand, result: result()}; }
        var bitField = {kind: 'opaque', spelling: '', objectKind: 'bitField', type: 'Root',
          canonicalType: 'Root', valueCategory: 'lvalue', typeInfo: root};
        var spelledOpaque = {kind: 'opaque', spelling: 'root', objectKind: 'ordinary',
          type: 'Root', canonicalType: 'Root', valueCategory: 'lvalue', typeInfo: root};
        var wrongZero = {kind: 'integerZero', spelling: '7', objectKind: 'ordinary', type: 'Root',
          canonicalType: 'Root', valueCategory: 'prvalue', typeInfo: root};
        var witnesses = [witness(bitField), witness(spelledOpaque), witness(wrongZero)];
        return {
          endpoint: {schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
            graph: {nodes: [], isIncomplete: false}, operations: {nodes: [], templates: [],
              conversions: [], callWitnesses: [], expressionWitnesses: witnesses}},
          worker: {expressionWitnesses: witnesses}
        };
      })()"""
    )

    val endpoint = cppCompletionContextFromDto(
      cppSemanticCompletionContextDto(payloads.endpoint, snapshot)
    )
    val worker = cppCompletionContextFromDto(payloads.worker)
    assertTrue(endpoint.expressionWitnesses.isEmpty())
    assertTrue(worker.expressionWitnesses.isEmpty())
  }

  @Test
  fun malformedPresentProfilesFailClosedAtBothDtoBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val endpoint = js(
      """(function() {
        function typeInfo(id, kind) { return {canonicalId: id, valueCanonicalId: id,
          kind: kind, isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true}; }
        var root = typeInfo('type:root', 'record');
        var result = typeInfo('type:type-info', 'record'); result.isConst = true;
        var brokenPointer = typeInfo('type:leaf-pointer', 'pointer');
        brokenPointer.pointeeCanonicalId = 'type:leaf';
        brokenPointer.pointeeIsConst = false;
        return {schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
          graph: {nodes: [], isIncomplete: false}, operations: {
            nodes: [], templates: [], conversions: [], callWitnesses: [], expressionWitnesses: [{
              syntax: 'typeidType', validation: 'semaExpressionBuild', authoritative: true,
              typeOperand: {type: 'Root', canonicalType: 'Root', typeInfo: root},
              expressionOperand: {kind: 'opaque'},
              result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'const TypeInfo', canonicalType: 'const TypeInfo',
                valueCategory: 'lvalue', typeInfo: result}
            }, {
              syntax: 'dynamicCast', validation: 'semaExpressionBuild', authoritative: true,
              typeOperand: {type: 'Leaf *', canonicalType: 'Leaf *', typeInfo: brokenPointer},
              expressionOperand: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'Root', canonicalType: 'Root',
                valueCategory: 'lvalue', typeInfo: root},
              result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'Root', canonicalType: 'Root',
                valueCategory: 'prvalue', typeInfo: root}
            }, {
              syntax: 'typeidExpression', validation: 'semaExpressionBuild', authoritative: true,
              expressionOperand: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'Root', canonicalType: 'Root',
                typeInfo: root},
              result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'const TypeInfo', canonicalType: 'const TypeInfo',
                valueCategory: 'lvalue', typeInfo: result}
            }]}};
      })()"""
    )
    val endpointContext = cppCompletionContextFromDto(
      cppSemanticCompletionContextDto(endpoint, snapshot)
    )

    val worker = js(
      """(function() {
        var root = {canonicalId: 'type:root', valueCanonicalId: 'type:root', kind: 'record',
          isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true};
        var result = {canonicalId: 'type:type-info', valueCanonicalId: 'type:type-info',
          kind: 'record', isConst: true, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true};
        var brokenPointer = {canonicalId: 'type:leaf-pointer',
          valueCanonicalId: 'type:leaf-pointer', kind: 'pointer',
          pointeeCanonicalId: 'type:leaf', pointeeIsConst: false,
          isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true};
        return {expressionWitnesses: [{syntax: 'typeidType', validation: 'semaExpressionBuild',
          authoritative: true,
          typeOperand: {type: 'Root', canonicalType: 'Root', typeInfo: root},
          expressionOperand: {kind: 'opaque'},
          result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'const TypeInfo', canonicalType: 'const TypeInfo',
            valueCategory: 'lvalue', typeInfo: result}
        }, {syntax: 'dynamicCast', validation: 'semaExpressionBuild', authoritative: true,
          typeOperand: {type: 'Leaf *', canonicalType: 'Leaf *', typeInfo: brokenPointer},
          expressionOperand: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'Root', canonicalType: 'Root',
            valueCategory: 'lvalue', typeInfo: root},
          result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'Root', canonicalType: 'Root',
            valueCategory: 'prvalue', typeInfo: root}
        }, {syntax: 'typeidExpression', validation: 'semaExpressionBuild', authoritative: true,
          expressionOperand: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'Root', canonicalType: 'Root', typeInfo: root},
          result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'const TypeInfo', canonicalType: 'const TypeInfo',
            valueCategory: 'lvalue', typeInfo: result}}]};
      })()"""
    )

    assertTrue(endpointContext.expressionWitnesses.isEmpty())
    assertTrue(cppCompletionContextFromDto(worker).expressionWitnesses.isEmpty())
  }
}
