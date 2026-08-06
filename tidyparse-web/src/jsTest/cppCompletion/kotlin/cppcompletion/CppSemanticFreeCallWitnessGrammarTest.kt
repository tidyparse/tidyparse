package cppcompletion

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppSemanticFreeCallWitnessGrammarTest {
  private val sourceA = info("type:SourceA")
  private val sourceB = info("type:SourceB")
  private val extraA = info("type:ExtraA")
  private val extraB = info("type:ExtraB")
  private val targetA = info("type:TargetA")
  private val targetB = info("type:TargetB")
  private val policyA = info("type:PolicyA")
  private val policyB = info("type:PolicyB")
  private val int = info("type:int", kind = "builtin")

  @Test
  fun explicitNondeducibleCallsKeepTypeAndValueVectorsIndivisible() {
    val transformA = callable("selectedA", "TargetA", targetA)
    val transformB = callable("selectedB", "TargetB", targetB)
    val context = context(
      values = listOf(
        value("sourceA", "SourceA", sourceA), value("sourceB", "SourceB", sourceB),
        value("extraA", "ExtraA", extraA), value("extraB", "ExtraB", extraB)
      ),
      witnesses = listOf(
        freeWitness(
          name = "api::transform",
          callable = transformA,
          explicitTypes = listOf(type("TargetA", targetA), type("PolicyA", policyA)),
          arguments = listOf(opaque("SourceA", sourceA), opaque("ExtraA", extraA))
        ),
        freeWitness(
          name = "api::transform",
          callable = transformB,
          explicitTypes = listOf(type("TargetB", targetB), type("PolicyB", policyB)),
          arguments = listOf(opaque("SourceB", sourceB), opaque("ExtraB", extraB))
        )
      )
    )
    val language = language(context)

    assertTrue(language.recognizes("api::transform<TargetA, PolicyA>(sourceA, extraA);"))
    assertTrue(language.recognizes("api::transform<TargetB, PolicyB>(sourceB, extraB);"))
    assertFalse(
      language.recognizes("api::transform<TargetA, PolicyA>(sourceB, extraB);"),
      "an explicit type vector must not combine with another witness's value vector"
    )
    assertFalse(language.recognizes("api::transform<TargetA, PolicyB>(sourceA, extraA);"))
    assertFalse(language.recognizes("api::transform<TargetA, PolicyA>(sourceA, extraB);"))
    assertFalse(
      language.recognizes("api::transform(sourceA, extraA);"),
      "the nondeducible explicit arguments cannot be silently omitted"
    )
    assertFalse(
      language.recognizes("identity::selectedA<TargetA, PolicyA>(sourceA, extraA);"),
      "the selected callable identity is not a source spelling"
    )
  }

  @Test
  fun emptyExplicitTypeVectorLowersAnOrdinaryDeducedFreeCall() {
    val witness = freeWitness(
      name = "api::deduce",
      callable = callable("selectedDeduce", "TargetA", targetA),
      explicitTypes = emptyList(),
      arguments = listOf(opaque("SourceA", sourceA))
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = listOf(witness)
    ))

    assertTrue(language.recognizes("api::deduce(sourceA);"))
    assertFalse(language.recognizes("api::deduce<TargetA>(sourceA);"))
  }

  @Test
  fun exactSourceLvalueTemplateWitnessSupportsPointerQualificationWithoutProducts() {
    val addressable = info("type:Addressable")
    val other = info("type:Other")
    fun pointer(id: String, pointee: CppTypeInfo, pointeeConst: Boolean = false) = CppTypeInfo(
      id = id,
      canonicalId = id,
      valueCanonicalId = id,
      kind = "pointer",
      pointeeCanonicalId = pointee.valueCanonicalId,
      pointeeIsConst = pointeeConst,
      isComplete = true,
      isSourceSpellable = true
    )
    val addressablePointer = pointer("pointer:Addressable", addressable)
    val constAddressablePointer = pointer(
      "pointer:const-Addressable", addressable, pointeeConst = true
    )
    val otherPointer = pointer("pointer:Other", other)
    val selected = callable("selectedAddress", "Addressable *", addressablePointer)
    val exact = freeWitness(
      name = "std::address_like",
      callable = selected,
      explicitTypes = emptyList(),
      arguments = listOf(opaque("Addressable", addressable)),
      result = opaque("Addressable *", addressablePointer, category = "prvalue")
    )
    val mismatchedResult = freeWitness(
      name = "api::mismatched_address",
      callable = selected,
      explicitTypes = emptyList(),
      arguments = listOf(opaque("Addressable", addressable)),
      result = opaque("Other *", otherPointer, category = "prvalue")
    )
    val language = language(context(
      values = listOf(
        value("addressable", "Addressable", addressable),
        value("other", "Other", other)
      ),
      types = listOf(
        typeDeclaration("Addressable", addressable),
        typeDeclaration("Other", other),
        typeDeclaration("Addressable *", addressablePointer),
        typeDeclaration("const Addressable *", constAddressablePointer),
        typeDeclaration("Other *", otherPointer)
      ),
      witnesses = listOf(exact, mismatchedResult)
    ))

    assertTrue(language.recognizes("std::address_like(addressable);"))
    assertTrue(
      language.recognizes(
        "const Addressable * produced = std::address_like(addressable);"
      ),
      "the exact result may undergo only the independently proven pointer qualification"
    )
    assertFalse(
      language.recognizes("std::address_like(other);"),
      "a deduced lvalue witness must not combine with another source value"
    )
    assertFalse(language.recognizes("api::mismatched_address(addressable);"))
    assertFalse(language.recognizes("std::unwitnessed_address(addressable);"))
  }

  @Test
  fun exactOrdinaryFreeFunctionIdentitySupportsDeclarationOnlyCallValidation() {
    val selected = callable("selectedInspect", "TargetA", targetA).copy(
      id = "function:inspect",
      primaryTemplateId = null
    )
    fun ordinary(name: String) = CppCallWitness(
      name = "api::$name",
      syntax = "freeCall",
      validation = "semaCallExpression",
      targetId = "function:inspect",
      arguments = listOf(opaque("SourceA", sourceA)),
      callable = selected,
      result = opaque("TargetA", targetA, category = "prvalue"),
      authoritative = true
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = listOf(ordinary("declared"))
    ))

    assertTrue(language.recognizes("api::declared(sourceA);"))
    assertFalse(language.recognizes("identity::selectedInspect(sourceA);"))
  }

  @Test
  fun ordinaryObjectWitnessesCannotSubstituteBitFieldMemberStates() {
    val state = info("type:State")
    val unsigned = info("type:unsigned", kind = "builtin")
    val long = info("type:long", kind = "builtin")
    fun field(name: String, type: String, typeInfo: CppTypeInfo, bitField: Boolean?) =
      CppReference(
        name = name,
        type = type,
        canonicalType = type,
        kind = "field",
        ownerType = "State",
        canonicalOwnerType = "State",
        typeInfo = typeInfo,
        ownerTypeInfo = state,
        isValue = true,
        isMember = true,
        isBitField = bitField
      )
    fun ordinary(target: String, argument: CppExpressionProfile) = CppCallWitness(
      name = "take",
      syntax = "freeCall",
      validation = "semaCallExpression",
      targetId = target,
      arguments = listOf(argument),
      callable = callable("selectedTake", "TargetA", targetA).copy(
        id = target,
        primaryTemplateId = null
      ),
      result = opaque("TargetA", targetA, category = "prvalue"),
      authoritative = true
    )
    val language = language(context(
      values = listOf(value("state", "State", state)),
      types = listOf(typeDeclaration("State", state)),
      membersByType = listOf(CppTypeMembers("State", listOf(
        field("bits", "unsigned int", unsigned, true),
        field("unproven", "unsigned int", unsigned, null),
        field("value", "long", long, false)
      ))),
      witnesses = listOf(
        ordinary("function:take-bits", opaque("unsigned int", unsigned)),
        ordinary("function:take-value", opaque("long", long))
      )
    ))

    assertFalse(language.recognizes("take(state.bits);"))
    assertFalse(language.recognizes("take(state.unproven);"))
    assertTrue(
      language.recognizes("take(state.value);"),
      "a sibling field proven not to be a bit-field should remain witness-consumable"
    )
  }

  @Test
  fun opaqueArithmeticPrvaluesCannotCollapseToAbstractLiteralZero() {
    val long = info("type:long", kind = "builtin")
    fun ordinary(
      name: String,
      target: String,
      argument: CppExpressionProfile
    ) = CppCallWitness(
      name = name,
      syntax = "freeCall",
      validation = "semaCallExpression",
      targetId = target,
      arguments = listOf(argument),
      callable = callable("selected", "TargetA", targetA).copy(
        id = target,
        primaryTemplateId = null
      ),
      result = opaque("TargetA", targetA, category = "prvalue"),
      authoritative = true
    )
    val opaqueLong = ordinary(
      name = "pick",
      target = "function:pick-long",
      argument = opaque("long", long, category = "prvalue")
    )
    val exactInteger = ordinary(
      name = "exact_pick",
      target = "function:pick-int",
      argument = CppExpressionProfile(
        kind = "integerZero",
        spelling = "0",
        objectKind = "ordinary",
        type = "int",
        canonicalType = "int",
        typeInfo = int,
        valueCategory = "prvalue"
      )
    )
    val language = language(context(
      values = emptyList(),
      witnesses = listOf(opaqueLong, exactInteger)
    ))

    assertFalse(
      language.recognizes("pick(0);"),
      "an opaque long selected against pick(long)/pick(void*) cannot reuse integer zero"
    )
    assertTrue(language.recognizes("exact_pick(0);"))
  }

  @Test
  fun targetIdentitySchemesCannotBeMixedOrDowngraded() {
    val selected = callable("selectedOrdinary", "TargetA", targetA).copy(
      id = "function:ordinary",
      primaryTemplateId = null
    )
    val ordinary = CppCallWitness(
      name = "api::ordinary",
      syntax = "freeCall",
      validation = "semaCallExpression",
      targetId = "function:ordinary",
      arguments = listOf(opaque("SourceA", sourceA)),
      callable = selected,
      result = opaque("TargetA", targetA, category = "prvalue"),
      authoritative = true
    )
    val template = freeWitness(
      name = "api::templated",
      callable = callable("selectedTemplate", "TargetA", targetA),
      explicitTypes = emptyList(),
      arguments = listOf(opaque("SourceA", sourceA))
    )
    val witnesses = listOf(
      ordinary.copy(name = "api::missingTarget", targetId = null),
      ordinary.copy(name = "api::wrongTarget", targetId = "function:other"),
      ordinary.copy(name = "api::missingSelected", callable = selected.copy(id = null)),
      ordinary.copy(name = "api::witnessPrimary", primaryTemplateId = "template:ordinary"),
      ordinary.copy(
        name = "api::selectedPrimary",
        callable = selected.copy(primaryTemplateId = "template:ordinary")
      ),
      ordinary.copy(
        name = "api::ordinaryTemplateId",
        explicitTypeArguments = listOf(type("TargetA", targetA))
      ),
      ordinary.copy(
        name = "ordinaryMember",
        syntax = "memberCall",
        receiver = opaque("SourceA", sourceA)
      ),
      ordinary.copy(name = "api::recursiveOrdinary",
        validation = "recursiveDefinitionInstantiation"),
      template.copy(name = "api::missingTemplateTarget", targetId = null),
      template.copy(name = "api::wrongTemplateTarget", targetId = "template:other"),
      template.copy(name = "api::shallowTemplate", validation = "semaCallExpression")
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = witnesses
    ))

    listOf(
      "missingTarget", "wrongTarget", "missingSelected", "witnessPrimary",
      "selectedPrimary", "ordinaryTemplateId<TargetA>",
      "recursiveOrdinary", "missingTemplateTarget", "wrongTemplateTarget", "shallowTemplate"
    ).forEach { name ->
      assertFalse(language.recognizes("api::$name(sourceA);"), name)
    }
    assertFalse(language.recognizes("sourceA.ordinaryMember(sourceA);"))
  }

  @Test
  fun orderedTypeAndExactIntegerArgumentsRemainCorrelatedWithTheirValueVector() {
    val first = taggedFreeWitness(
      name = "api::get",
      callable = callable("selectedGetA", "TargetA", targetA),
      explicitArguments = listOf(templateType("TargetA", targetA), exactInteger("1")),
      arguments = listOf(opaque("SourceA", sourceA))
    )
    val second = taggedFreeWitness(
      name = "api::get",
      callable = callable("selectedGetB", "TargetB", targetB),
      explicitArguments = listOf(templateType("TargetB", targetB), exactInteger("2")),
      arguments = listOf(opaque("SourceB", sourceB))
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA), value("sourceB", "SourceB", sourceB)),
      witnesses = listOf(first, second)
    ))

    assertTrue(language.recognizes("api::get<TargetA, 1>(sourceA);"))
    assertTrue(language.recognizes("api::get<TargetB, 2>(sourceB);"))
    assertFalse(language.recognizes("api::get<TargetA, 0>(sourceA);"))
    assertFalse(language.recognizes("api::get<TargetA, 2>(sourceA);"))
    assertFalse(language.recognizes("api::get<TargetA, 1>(sourceB);"))
    assertFalse(language.recognizes("api::get<1, TargetA>(sourceA);"))
  }

  @Test
  fun arrayTypeArgumentsRequireIndependentElementIdentityAndExactSemaShape() {
    fun array(
      id: String,
      elementId: String? = int.valueCanonicalId,
      incomplete: Boolean? = true,
      bound: String? = null,
      elementConst: Boolean = false,
      elementVolatile: Boolean = false
    ) = CppTypeInfo(
      id = id,
      canonicalId = id,
      valueCanonicalId = id,
      kind = "array",
      isConst = elementConst,
      isVolatile = elementVolatile,
      elementCanonicalId = elementId,
      elementIsConst = elementConst,
      elementIsVolatile = elementVolatile,
      isIncompleteArray = incomplete,
      arrayBound = bound,
      isComplete = incomplete == false,
      isSourceSpellable = true
    )
    val selected = callable("selectedArray", "TargetA", targetA)
    val argument = listOf(opaque("SourceA", sourceA))
    fun witness(name: String, spelling: String, info: CppTypeInfo) = freeWitness(
      name = "api::$name",
      callable = selected,
      explicitTypes = listOf(type(spelling, info)),
      arguments = argument
    )
    val witnesses = listOf(
      witness("incompleteArray", "int[]", array("array:valid-incomplete")),
      witness(
        "boundedArray", "volatile int[7]",
        array(
          "array:valid-bounded", incomplete = false, bound = "7", elementVolatile = true
        )
      ),
      witness(
        "unknownElement", "Ghost[]",
        array("array:unknown-element", elementId = "type:Ghost")
      ),
      witness("missingElement", "int[]", array("array:missing-element", elementId = null)),
      witness(
        "boundedClaimForIncomplete", "int[]",
        array("array:bounded-claim", incomplete = false, bound = "7")
      ),
      witness(
        "incompleteClaimForBounded", "int[7]",
        array("array:incomplete-claim", incomplete = true)
      ),
      witness(
        "wrongBound", "int[7]",
        array("array:wrong-bound", incomplete = false, bound = "8")
      ),
      witness(
        "zeroBound", "int[0]",
        array("array:zero-bound", incomplete = false, bound = "0")
      ),
      witness(
        "wrongElementCv", "int[]",
        array("array:wrong-cv", elementConst = true)
      )
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = witnesses
    ))

    assertTrue(language.recognizes("api::incompleteArray<int[]>(sourceA);"))
    assertTrue(language.recognizes("api::boundedArray<volatile int[7]>(sourceA);"))
    listOf(
      "unknownElement<Ghost[]>",
      "missingElement<int[]>",
      "boundedClaimForIncomplete<int[]>",
      "incompleteClaimForBounded<int[7]>",
      "wrongBound<int[7]>",
      "zeroBound<int[0]>",
      "wrongElementCv<int[]>"
    ).forEach { call ->
      assertFalse(language.recognizes("api::$call(sourceA);"), call)
    }
  }

  @Test
  fun malformedExactIntegerProfilesAndHybridSchemasFailClosed() {
    val base = taggedFreeWitness(
      name = "api::goodExact",
      callable = callable("selectedExact", "TargetA", targetA),
      explicitArguments = listOf(exactInteger("1")),
      arguments = listOf(opaque("SourceA", sourceA))
    )
    val witnesses = listOf(
      base.copy(
        name = "api::udl",
        explicitTemplateArguments = listOf(exactInteger("1_km", canonicalValue = "1"))
      ),
      base.copy(
        name = "api::injection",
        explicitTemplateArguments = listOf(exactInteger("1, 2", canonicalValue = "1"))
      ),
      base.copy(
        name = "api::multiToken",
        explicitTemplateArguments = listOf(exactInteger("-1", canonicalValue = "1"))
      ),
      base.copy(
        name = "api::wrongValue",
        explicitTemplateArguments = listOf(exactInteger("1", canonicalValue = "2"))
      ),
      base.copy(
        name = "api::nonIntegral",
        explicitTemplateArguments = listOf(exactInteger("1", typeInfo = targetA))
      ),
      base.copy(
        name = "api::hybrid",
        explicitTypeArguments = listOf(type("TargetA", targetA))
      )
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = witnesses
    ))

    listOf("udl<1_km>", "injection<1, 2>", "multiToken<-1>", "wrongValue<1>",
      "nonIntegral<1>", "hybrid<1>").forEach { call ->
      assertFalse(language.recognizes("api::$call(sourceA);"), call)
    }
  }

  @Test
  fun witnessTypeSpellingsCannotAuthenticateTheirOwnIdentityOrEscapeTemplateArguments() {
    val selected = callable("selected", "TargetA", targetA)
    val argument = listOf(opaque("SourceA", sourceA))
    val witnesses = listOf(
      freeWitness(
        "api::comma", selected,
        listOf(type("TargetA, SourceA", targetA)), argument
      ),
      freeWitness(
        "api::unbalanced", selected,
        listOf(type("TargetA<SourceA", targetA)), argument
      ),
      freeWitness(
        "api::expression", selected,
        listOf(type("TargetA()", targetA)), argument
      ),
      freeWitness(
        "api::wrongIdentity", selected,
        listOf(type("TargetB", targetA)), argument
      )
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = witnesses
    ))

    assertFalse(language.recognizes("api::comma<TargetA, SourceA>(sourceA);"))
    assertFalse(language.recognizes("api::unbalanced<TargetA<SourceA>(sourceA);"))
    assertFalse(language.recognizes("api::expression<TargetA()>(sourceA);"))
    assertFalse(language.recognizes("api::wrongIdentity<TargetB>(sourceA);"))
  }

  @Test
  fun independentlyEstablishedPublicAliasAuthenticatesReservedCanonicalMetadata() {
    val canonical = "implementation::__CanonicalTarget"
    val publicAlias = CppReference(
      name = "public_api::AliasA",
      type = canonical,
      canonicalType = canonical,
      kind = "typeAlias",
      isType = true,
      typeInfo = targetA,
      completionVisible = true
    )
    val witness = freeWitness(
      name = "api::alias",
      callable = callable("selectedAlias", "TargetA", targetA),
      explicitTypes = listOf(CppTypeProfile(
        type = "public_api::AliasA",
        canonicalType = canonical,
        typeInfo = targetA
      )),
      arguments = listOf(opaque("SourceA", sourceA))
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      types = listOf(publicAlias),
      witnesses = listOf(witness)
    ))

    assertTrue(language.recognizes("api::alias<public_api::AliasA>(sourceA);"))
    assertFalse(language.recognizes("api::alias<TargetA>(sourceA);"))
  }

  @Test
  fun constReferenceReturnsUseTheCallExpressionKindAndRejectMutableResultProfiles() {
    val callableResult = targetA.copy(
      canonicalId = "reference:const-target-a",
      kind = "lvalueReference",
      isConst = true
    )
    val constValueResult = targetA.copy(isConst = true)
    val selected = callable("selectedConstRef", "const TargetA &", callableResult)
    val valid = freeWitness(
      name = "api::constRef",
      callable = selected,
      explicitTypes = emptyList(),
      arguments = listOf(opaque("SourceA", sourceA)),
      result = opaque("const TargetA", constValueResult, category = "lvalue")
    )
    val mutableResult = valid.copy(
      name = "api::badConstRef",
      result = opaque("TargetA", targetA, category = "lvalue")
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = listOf(valid, mutableResult)
    ))

    assertTrue(language.recognizes("api::constRef(sourceA);"))
    assertFalse(language.recognizes("api::badConstRef(sourceA);"))
  }

  @Test
  fun malformedNamesReceiversResultsAndNonTypeArgumentsAreRejected() {
    val baseCallable = callable("selected", "TargetA", targetA)
    val argument = listOf(opaque("SourceA", sourceA))
    val fakeTemplate = info("template:Container", kind = "template")
    val witnesses = listOf(
      freeWitness("api::__secret", baseCallable, listOf(type("TargetA", targetA)), argument),
      freeWitness("api::bad<TargetA>", baseCallable, emptyList(), argument),
      freeWitness("return", baseCallable, emptyList(), argument),
      freeWitness("api::valueStyle", baseCallable, listOf(type("7", int)), argument),
      freeWitness("api::reservedType", baseCallable, listOf(type("_Secret", targetA)), argument),
      freeWitness(
        "api::templateStyle", baseCallable, listOf(type("Container", fakeTemplate)), argument
      ),
      freeWitness("api::withReceiver", baseCallable, emptyList(), argument)
        .copy(receiver = opaque("SourceA", sourceA)),
      freeWitness(
        "api::badResult", baseCallable, listOf(type("TargetA", targetA)), argument,
        result = opaque("TargetB", targetB, category = "prvalue")
      ),
      freeWitness("api::missingPrimary", baseCallable, emptyList(), argument)
        .copy(primaryTemplateId = null),
      freeWitness("api::mismatchedPrimary", baseCallable, emptyList(), argument)
        .let { witness -> witness.copy(
          callable = witness.callable.copy(primaryTemplateId = "template:someoneElse")
        ) }
    )
    val language = language(context(
      values = listOf(value("sourceA", "SourceA", sourceA)),
      witnesses = witnesses
    ))

    assertFalse(language.recognizes("api::__secret<TargetA>(sourceA);"))
    assertFalse(language.recognizes("api::bad<TargetA>(sourceA);"))
    assertFalse(language.recognizes("return(sourceA);"))
    assertFalse(language.recognizes("api::valueStyle<7>(sourceA);"))
    assertFalse(language.recognizes("api::reservedType<_Secret>(sourceA);"))
    assertFalse(language.recognizes("api::templateStyle<Container>(sourceA);"))
    assertFalse(language.recognizes("api::withReceiver(sourceA);"))
    assertFalse(language.recognizes("api::badResult<TargetA>(sourceA);"))
    assertFalse(language.recognizes("api::missingPrimary(sourceA);"))
    assertFalse(language.recognizes("api::mismatchedPrimary(sourceA);"))
  }

  private fun freeWitness(
    name: String,
    callable: CppReference,
    explicitTypes: List<CppTypeProfile>,
    arguments: List<CppExpressionProfile>,
    result: CppExpressionProfile = opaque(
      callable.returnType.orEmpty(), callable.returnTypeInfo!!, category = "prvalue"
    )
  ) = CppCallWitness(
    name = name,
    syntax = "freeCall",
    validation = "recursiveDefinitionInstantiation",
    targetId = "template:$name",
    primaryTemplateId = "template:$name",
    explicitTypeArguments = explicitTypes,
    arguments = arguments,
    callable = callable.copy(primaryTemplateId = "template:$name"),
    result = result,
    authoritative = true
  )

  private fun taggedFreeWitness(
    name: String,
    callable: CppReference,
    explicitArguments: List<CppTemplateArgumentProfile>,
    arguments: List<CppExpressionProfile>
  ) = freeWitness(name, callable, emptyList(), arguments).copy(
    explicitTemplateArguments = explicitArguments
  )

  private fun templateType(spelling: String, info: CppTypeInfo) =
    CppTemplateArgumentProfile(kind = "type", type = type(spelling, info))

  private fun exactInteger(
    spelling: String,
    canonicalValue: String = spelling,
    typeInfo: CppTypeInfo = int
  ) = CppTemplateArgumentProfile(
    kind = "exactIntegerLiteral",
    type = type(if (typeInfo == int) "int" else "TargetA", typeInfo),
    spelling = spelling,
    canonicalValue = canonicalValue
  )

  private fun callable(name: String, spelling: String, result: CppTypeInfo) = CppReference(
    name = name,
    qualifiedName = "identity::$name",
    kind = "function",
    returnType = spelling,
    canonicalReturnType = spelling,
    returnTypeInfo = result,
    isCallable = true,
    isMember = false
  )

  private fun type(spelling: String, info: CppTypeInfo) = CppTypeProfile(
    type = spelling,
    canonicalType = spelling,
    typeInfo = info
  )

  private fun opaque(
    spelling: String,
    info: CppTypeInfo,
    category: String = "lvalue"
  ) = CppExpressionProfile(
    kind = "opaque",
    spelling = null,
    objectKind = "ordinary",
    type = spelling,
    canonicalType = spelling,
    typeInfo = info,
    valueCategory = category
  )

  private fun value(name: String, spelling: String, info: CppTypeInfo) = CppReference(
    name = name,
    type = spelling,
    canonicalType = spelling,
    kind = "variable",
    typeInfo = info,
    isValue = true
  )

  private fun info(id: String, kind: String = "record") = CppTypeInfo(
    id = id,
    canonicalId = id,
    valueCanonicalId = id,
    kind = kind,
    isComplete = true,
    isSourceSpellable = true
  )

  private fun context(
    values: List<CppReference>,
    types: List<CppReference> = emptyList(),
    membersByType: List<CppTypeMembers> = emptyList(),
    witnesses: List<CppCallWitness>
  ): CppCompletionContext {
    val independentTypes = listOf(
      typeDeclaration("TargetA", targetA), typeDeclaration("TargetB", targetB),
      typeDeclaration("PolicyA", policyA), typeDeclaration("PolicyB", policyB),
      typeDeclaration("int", int)
    ) + types
    val identifiers = buildSet {
      addAll(listOf(
        "api", "transform", "deduce", "TargetA", "TargetB", "PolicyA", "PolicyB",
        "SourceA", "SourceB", "ExtraA", "ExtraB", "Container", "constRef", "badConstRef",
        "mismatchedPrimary"
      ))
      values.mapTo(this, CppReference::name)
      independentTypes.flatMapTo(this) { it.name.split("::") }
      witnesses.flatMapTo(this) { it.name.split("::") }
      membersByType.flatMapTo(this) { group -> group.members.map(CppReference::name) }
    }
    return CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      completionKind = "expression",
      values = values,
      types = independentTypes,
      membersByType = membersByType,
      callWitnesses = witnesses
    )
  }

  private fun typeDeclaration(name: String, info: CppTypeInfo) = CppReference(
    name = name,
    type = name,
    canonicalType = name,
    kind = if (info.kind == "builtin") "builtinType" else "class",
    isType = true,
    typeInfo = info,
    completionVisible = true
  )

  private fun language(context: CppCompletionContext): CppSuffixGrammar =
    CppCompletionGrammar().generate(context, emptyList())

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)
}
