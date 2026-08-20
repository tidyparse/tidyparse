import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CppSemanticFreeCallWitnessAdapterTest {
  @Test
  fun freeCallTypeAndValueVectorsSurviveBothDtoBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val result = js(
      """({schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
        graph: {nodes: [], isIncomplete: false}, operations: {
          nodes: [], templates: [], conversions: [], expressionWitnesses: [],
          callWitnesses: [{
            name: 'api::transform', syntax: 'freeCall',
            validation: 'recursiveDefinitionInstantiation', authoritative: true,
            targetId: 'template:transform',
            primaryTemplateId: 'template:transform',
            explicitTypeArguments: [{type: 'Target', canonicalType: 'Target', typeInfo: {
              canonicalId: 'type:target', valueCanonicalId: 'type:target', kind: 'record',
              isConst: false, isVolatile: false, isDependent: false,
              isInstantiationDependent: false,
              isSourceSpellable: true, isComplete: true}},
              {type: 'Policy', canonicalType: 'Policy', typeInfo: {
                canonicalId: 'type:policy', valueCanonicalId: 'type:policy', kind: 'record',
                isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false,
                isSourceSpellable: true, isComplete: true}}],
            arguments: [{kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'Source', canonicalType: 'Source',
              valueCategory: 'lvalue', typeInfo: {canonicalId: 'type:source',
                valueCanonicalId: 'type:source', kind: 'record',
                isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false,
                isSourceSpellable: true, isComplete: true}}],
            callable: {id: 'specialization:transform', primaryTemplateId: 'template:transform',
              qualifiedName: 'identity::selected',
              kind: 'Function', provenance: {sema: true, index: false}, isCallable: true,
              isMember: false, returnType: 'Target', canonicalReturnType: 'Target',
              returnTypeInfo: {canonicalId: 'type:target', valueCanonicalId: 'type:target',
                kind: 'record', isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false, isSourceSpellable: true, isComplete: true},
                parameters: []},
            result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'Target', canonicalType: 'Target',
              valueCategory: 'prvalue', typeInfo: {canonicalId: 'type:target',
                valueCanonicalId: 'type:target', kind: 'record',
                isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false,
                isSourceSpellable: true, isComplete: true}}
          }, {
            name: 'api::deduce', syntax: 'freeCall',
            validation: 'recursiveDefinitionInstantiation', authoritative: true,
            targetId: 'template:deduce',
            primaryTemplateId: 'template:deduce',
            arguments: [],
            callable: {id: 'specialization:deduce', primaryTemplateId: 'template:deduce',
              qualifiedName: 'identity::deduced',
              kind: 'Function', provenance: {sema: true, index: false}, isCallable: true,
              isMember: false, returnType: 'Target', canonicalReturnType: 'Target',
              returnTypeInfo: {canonicalId: 'type:target', valueCanonicalId: 'type:target',
                kind: 'record', isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false, isSourceSpellable: true, isComplete: true},
                parameters: []},
            result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
              type: 'Target', canonicalType: 'Target',
              valueCategory: 'prvalue', typeInfo: {canonicalId: 'type:target',
                valueCanonicalId: 'type:target', kind: 'record',
                isConst: false, isVolatile: false, isDependent: false,
                isInstantiationDependent: false,
                isSourceSpellable: true, isComplete: true}}
          }]}
      })"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val explicit = context.callWitnesses.first()
    val deduced = context.callWitnesses.last()

    assertEquals("freeCall", explicit.syntax)
    assertEquals("api::transform", explicit.name)
    assertEquals("template:transform", explicit.targetId)
    assertNull(explicit.receiver)
    assertEquals(listOf("Target", "Policy"), explicit.explicitTypeArguments.map { it.type })
    assertEquals(
      listOf("type:target", "type:policy"),
      explicit.explicitTypeArguments.map { it.typeInfo.valueCanonicalId }
    )
    assertEquals(listOf("Source"), explicit.arguments.map { it.type })
    assertEquals(listOf("ordinary"), explicit.arguments.map { it.objectKind })
    assertEquals("identity::selected", explicit.callable.qualifiedName)
    assertTrue(deduced.explicitTypeArguments.isEmpty(), "a missing legacy field defaults to empty")
  }

  @Test
  fun orderedTaggedArgumentsSurviveEndpointAndWorkerDtoBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val result = js(
      """(function() {
        function info(id, kind) { return {canonicalId: id, valueCanonicalId: id,
          kind: kind, isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true}; }
        var target = info('type:target', 'record');
        var integer = info('type:int', 'builtin');
        var source = info('type:source', 'record');
        return {schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
          graph: {nodes: [], isIncomplete: false}, operations: {nodes: [], templates: [],
            conversions: [], expressionWitnesses: [], callWitnesses: [{
              name: 'api::get', syntax: 'freeCall',
              validation: 'recursiveDefinitionInstantiation', authoritative: true,
              targetId: 'template:get',
              primaryTemplateId: 'template:get',
              explicitTemplateArguments: [
                {kind: 'type', type: {type: 'Target', canonicalType: 'Target', typeInfo: target}},
                {kind: 'exactIntegerLiteral', type: {type: 'int', canonicalType: 'int',
                  typeInfo: integer}, spelling: '1', canonicalValue: '1'}],
              arguments: [{kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'Source', canonicalType: 'Source',
                valueCategory: 'lvalue', typeInfo: source}],
              callable: {id: 'specialization:get', primaryTemplateId: 'template:get',
                qualifiedName: 'identity::get',
                kind: 'Function', provenance: {sema: true, index: false}, isCallable: true,
                isMember: false, returnType: 'Target', canonicalReturnType: 'Target',
                returnTypeInfo: target, parameters: []},
              result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
                type: 'Target', canonicalType: 'Target',
                valueCategory: 'prvalue', typeInfo: target}
            }]}};
      })()"""
    )

    val context = cppCompletionContextFromDto(cppSemanticCompletionContextDto(result, snapshot))
    val witness = context.callWitnesses.single()
    assertTrue(witness.explicitTypeArguments.isEmpty())
    assertEquals(listOf("type", "exactIntegerLiteral"),
      witness.explicitTemplateArguments.map { it.kind })
    assertEquals(listOf("Target", "int"),
      witness.explicitTemplateArguments.map { it.type.type })
    assertEquals("1", witness.explicitTemplateArguments.last().spelling)
    assertEquals("1", witness.explicitTemplateArguments.last().canonicalValue)
  }

  @Test
  fun ordinaryTargetIdentityAndCallExpressionValidationSurviveBothDtoBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val result = js(
      """(function() {
        function info(id, kind) { return {canonicalId: id, valueCanonicalId: id,
          kind: kind, isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true}; }
        var target = info('type:target', 'record');
        var source = info('type:source', 'record');
        function ordinary(name) { return {
          name: 'api::' + name, syntax: 'freeCall', validation: 'semaCallExpression',
          authoritative: true, targetId: 'function:introduce', primaryTemplateId: '',
          explicitTemplateArguments: [],
          arguments: [{kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'Source', canonicalType: 'Source', valueCategory: 'lvalue', typeInfo: source}],
          callable: {name: 'identity::introduce', id: 'function:introduce', primaryTemplateId: '',
            qualifiedName: 'identity::introduce', kind: 'Function',
            provenance: {sema: true, index: false}, isCallable: true, isMember: false,
            returnType: 'Target', canonicalReturnType: 'Target',
            returnTypeInfo: target, parameters: []},
          result: {kind: 'opaque', spelling: '', objectKind: 'ordinary', type: 'Target',
            canonicalType: 'Target', valueCategory: 'prvalue', typeInfo: target}}; }
        function clone(base, name) {
          var value = Object.assign({}, base); value.name = 'api::' + name;
          value.callable = Object.assign({}, base.callable); return value;
        }
        var valid = ordinary('introduce');
        var missingTarget = clone(valid, 'missingTarget'); delete missingTarget.targetId;
        var wrongTarget = clone(valid, 'wrongTarget'); wrongTarget.targetId = 'function:other';
        var witnessPrimary = clone(valid, 'witnessPrimary');
        witnessPrimary.primaryTemplateId = 'template:introduce';
        var selectedPrimary = clone(valid, 'selectedPrimary');
        selectedPrimary.callable.primaryTemplateId = 'template:introduce';
        var malformedSelectedPrimary = clone(valid, 'malformedSelectedPrimary');
        malformedSelectedPrimary.callable.primaryTemplateId = 7;
        var paddedTarget = clone(valid, 'paddedTarget');
        paddedTarget.targetId = ' function:introduce ';
        var explicitOrdinary = clone(valid, 'explicitOrdinary');
        explicitOrdinary.explicitTemplateArguments = [{kind: 'type', type: {
          type: 'Target', canonicalType: 'Target', typeInfo: target}}];
        var shallowTemplate = clone(valid, 'shallowTemplate');
        shallowTemplate.targetId = shallowTemplate.primaryTemplateId = 'template:introduce';
        shallowTemplate.callable.primaryTemplateId = 'template:introduce';
        return {schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
          graph: {nodes: [], isIncomplete: false}, operations: {nodes: [], templates: [],
            conversions: [], expressionWitnesses: [], callWitnesses: [valid, missingTarget,
              wrongTarget, witnessPrimary, selectedPrimary, malformedSelectedPrimary, paddedTarget,
              explicitOrdinary, shallowTemplate]}};
      })()"""
    )

    val endpointContext = cppCompletionContextFromDto(
      cppSemanticCompletionContextDto(result, snapshot)
    )
    val workerDto = js("({})")
    workerDto.callWitnesses = result.operations.callWitnesses
    val workerContext = cppCompletionContextFromDto(workerDto)

    listOf(endpointContext, workerContext).forEach { context ->
      val witness = context.callWitnesses.single()
      assertEquals("api::introduce", witness.name)
      assertEquals("semaCallExpression", witness.validation)
      assertEquals("function:introduce", witness.targetId)
      assertNull(witness.primaryTemplateId)
      assertEquals("function:introduce", witness.callable.id)
      assertTrue(witness.callable.primaryTemplateId.isNullOrBlank())
      assertTrue(witness.explicitTemplateArguments.isEmpty())
    }
  }

  @Test
  fun arrayMetadataIsLosslessAndMalformedShapesFailClosedAtBothDtoBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val result = js(
      """(function() {
        function info(id, kind) { return {canonicalId: id, valueCanonicalId: id,
          kind: kind, isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true}; }
        function arrayInfo(id, incomplete, bound) {
          var value = {canonicalId: id, valueCanonicalId: id, kind: 'array',
            isConst: false, isVolatile: false, elementCanonicalId: 'type:int',
            elementIsConst: false, elementIsVolatile: false,
            isIncompleteArray: incomplete, isDependent: false,
            isInstantiationDependent: false, isSourceSpellable: true,
            isComplete: !incomplete};
          if (bound !== undefined) value.arrayBound = bound;
          return value;
        }
        var target = info('type:target', 'record');
        function witness(name, spelling, typeInfo) { return {
          name: 'api::' + name, syntax: 'freeCall',
          validation: 'recursiveDefinitionInstantiation', authoritative: true,
          targetId: 'template:' + name,
          primaryTemplateId: 'template:' + name,
          explicitTemplateArguments: [{kind: 'type', type: {
            type: spelling, canonicalType: spelling, typeInfo: typeInfo}}],
          arguments: [],
          callable: {name: 'selected_' + name, qualifiedName: 'selected_' + name,
            id: 'specialization:' + name, primaryTemplateId: 'template:' + name,
            kind: 'Function', provenance: {sema: true, index: false},
            isCallable: true, isMember: false, returnType: 'Target',
            canonicalReturnType: 'Target', returnTypeInfo: target, parameters: []},
          result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'Target', canonicalType: 'Target', valueCategory: 'prvalue',
            typeInfo: target}}; }
        var incomplete = witness('incomplete', 'int[]',
          arrayInfo('array:incomplete', true));
        var bounded = witness('bounded', 'int[7]',
          arrayInfo('array:bounded', false, '7'));
        var missingElement = arrayInfo('array:missing-element', true);
        delete missingElement.elementCanonicalId;
        var missing = witness('missing', 'int[]', missingElement);
        var contradictory = witness('contradictory', 'int[]',
          arrayInfo('array:contradictory', true, '7'));
        var missingBound = witness('missingBound', 'int[7]',
          arrayInfo('array:missing-bound', false));
        var zeroBound = witness('zeroBound', 'int[0]',
          arrayInfo('array:zero-bound', false, '0'));
        return {schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
          graph: {nodes: [], isIncomplete: false}, operations: {nodes: [], templates: [],
            conversions: [], expressionWitnesses: [],
            callWitnesses: [incomplete, bounded, missing, contradictory, missingBound,
              zeroBound]}};
      })()"""
    )

    val endpointContext = cppCompletionContextFromDto(
      cppSemanticCompletionContextDto(result, snapshot)
    )
    val workerDto = js("({})")
    workerDto.callWitnesses = result.operations.callWitnesses
    val workerContext = cppCompletionContextFromDto(workerDto)

    listOf(endpointContext, workerContext).forEach { context ->
      assertEquals(listOf("api::incomplete", "api::bounded"),
        context.callWitnesses.map { it.name })
      val incomplete = context.callWitnesses[0]
        .explicitTemplateArguments.single().type.typeInfo
      assertEquals("type:int", incomplete.elementCanonicalId)
      assertEquals(false, incomplete.elementIsConst)
      assertEquals(false, incomplete.elementIsVolatile)
      assertEquals(true, incomplete.isIncompleteArray)
      assertNull(incomplete.arrayBound)
      val bounded = context.callWitnesses[1]
        .explicitTemplateArguments.single().type.typeInfo
      assertEquals(false, bounded.isIncompleteArray)
      assertEquals("7", bounded.arrayBound)
    }
  }

  @Test
  fun malformedArraysAndPresentReceiverFailClosedAtBothDtoBoundaries() {
    val snapshot = assertNotNull(cppEditorStatementSnapshot("  ", 0, 2))
    val endpoint = js(
      """(function() {
        var target = {canonicalId: 'type:target', valueCanonicalId: 'type:target',
          kind: 'record', isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true};
        var integer = {canonicalId: 'type:int', valueCanonicalId: 'type:int',
          kind: 'builtin', isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true};
        var taggedType = {kind: 'type', type: {type: 'Target', canonicalType: 'Target',
          typeInfo: target}};
        function exact(spelling, value) { return {kind: 'exactIntegerLiteral',
          type: {type: 'int', canonicalType: 'int', typeInfo: integer},
          spelling: spelling, canonicalValue: value}; }
        function witness(name) { return {name: name, syntax: 'freeCall',
          validation: 'recursiveDefinitionInstantiation', authoritative: true,
          targetId: 'template:' + name,
          primaryTemplateId: 'template:' + name,
          callable: {id: 'specialization:' + name, primaryTemplateId: 'template:' + name,
            qualifiedName: 'identity::' + name,
            kind: 'Function', provenance: {sema: true, index: false}, isCallable: true,
            isMember: false, returnType: 'Target', canonicalReturnType: 'Target',
            returnTypeInfo: target, parameters: []},
          result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'Target', canonicalType: 'Target',
            valueCategory: 'prvalue', typeInfo: target}}; }
        var badArguments = witness('badArguments'); badArguments.arguments = {};
        var badExplicit = witness('badExplicit'); badExplicit.arguments = [];
        badExplicit.explicitTypeArguments = {};
        var badReceiver = witness('badReceiver'); badReceiver.arguments = [];
        badReceiver.receiver = {kind: 'opaque'};
        var badReturn = witness('badReturn'); badReturn.arguments = [];
        badReturn.callable.returnTypeInfo = {canonicalId: 'type:target',
          valueCanonicalId: 'type:target', kind: 'record', isVolatile: false,
          isDependent: false, isInstantiationDependent: false, isSourceSpellable: true};
        var badTagged = witness('badTagged'); badTagged.arguments = [];
        badTagged.explicitTemplateArguments = {};
        var hybrid = witness('hybrid'); hybrid.arguments = [];
        hybrid.explicitTemplateArguments = [exact('1', '1')];
        hybrid.explicitTypeArguments = [taggedType.type];
        var badUdl = witness('badUdl'); badUdl.arguments = [];
        badUdl.explicitTemplateArguments = [exact('1_km', '1')];
        var badInjection = witness('badInjection'); badInjection.arguments = [];
        badInjection.explicitTemplateArguments = [exact('1, 2', '1')];
        var badCanonical = witness('badCanonical'); badCanonical.arguments = [];
        badCanonical.explicitTemplateArguments = [exact('1', '2')];
        var badKind = witness('badKind'); badKind.arguments = [];
        badKind.explicitTemplateArguments = [{kind: 'value', type: taggedType.type}];
        return {schemaVersion: 2, context: {kind: 'Expression'}, items: [], scopeItems: [],
          graph: {nodes: [], isIncomplete: false}, operations: {nodes: [], templates: [],
            conversions: [], expressionWitnesses: [],
            callWitnesses: [badArguments, badExplicit, badReceiver, badReturn, badTagged,
              hybrid, badUdl, badInjection, badCanonical, badKind]}};
      })()"""
    )
    val endpointContext = cppCompletionContextFromDto(
      cppSemanticCompletionContextDto(endpoint, snapshot)
    )

    val worker = js(
      """(function() {
        var target = {canonicalId: 'type:target', valueCanonicalId: 'type:target',
          kind: 'record', isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true};
        var integer = {canonicalId: 'type:int', valueCanonicalId: 'type:int',
          kind: 'builtin', isConst: false, isVolatile: false, isDependent: false,
          isInstantiationDependent: false, isSourceSpellable: true, isComplete: true};
        var taggedType = {kind: 'type', type: {type: 'Target', canonicalType: 'Target',
          typeInfo: target}};
        function exact(spelling, value) { return {kind: 'exactIntegerLiteral',
          type: {type: 'int', canonicalType: 'int', typeInfo: integer},
          spelling: spelling, canonicalValue: value}; }
        function witness(name) { return {name: name, syntax: 'freeCall',
          validation: 'recursiveDefinitionInstantiation', authoritative: true,
          targetId: 'template:' + name,
          primaryTemplateId: 'template:' + name,
          callable: {name: 'identity::' + name, primaryTemplateId: 'template:' + name,
            kind: 'function', isCallable: true,
            isMember: false, returnType: 'Target', canonicalReturnType: 'Target',
            returnTypeInfo: target},
          result: {kind: 'opaque', spelling: '', objectKind: 'ordinary',
            type: 'Target', canonicalType: 'Target',
            valueCategory: 'prvalue', typeInfo: target}}; }
        var badArguments = witness('badArguments'); badArguments.arguments = 'not-an-array';
        var badExplicit = witness('badExplicit'); badExplicit.arguments = [];
        badExplicit.explicitTypeArguments = {};
        var badReceiver = witness('badReceiver'); badReceiver.arguments = [];
        badReceiver.receiver = {kind: 'opaque'};
        var badReturn = witness('badReturn'); badReturn.arguments = [];
        badReturn.callable.returnTypeInfo = {canonicalId: 'type:target',
          valueCanonicalId: 'type:target', kind: 'record', isVolatile: false,
          isDependent: false, isInstantiationDependent: false, isSourceSpellable: true};
        var badTagged = witness('badTagged'); badTagged.arguments = [];
        badTagged.explicitTemplateArguments = 'not-an-array';
        var hybrid = witness('hybrid'); hybrid.arguments = [];
        hybrid.explicitTemplateArguments = [exact('1', '1')];
        hybrid.explicitTypeArguments = [taggedType.type];
        var badUdl = witness('badUdl'); badUdl.arguments = [];
        badUdl.explicitTemplateArguments = [exact('1_km', '1')];
        var badInjection = witness('badInjection'); badInjection.arguments = [];
        badInjection.explicitTemplateArguments = [exact('1; evil', '1')];
        var badCanonical = witness('badCanonical'); badCanonical.arguments = [];
        badCanonical.explicitTemplateArguments = [exact('1', '2')];
        var badKind = witness('badKind'); badKind.arguments = [];
        badKind.explicitTemplateArguments = [{kind: 'value', type: taggedType.type}];
        return {callWitnesses: [badArguments, badExplicit, badReceiver, badReturn, badTagged,
          hybrid, badUdl, badInjection, badCanonical, badKind]};
      })()"""
    )

    assertTrue(endpointContext.callWitnesses.isEmpty())
    assertTrue(cppCompletionContextFromDto(worker).callWitnesses.isEmpty())
  }
}
