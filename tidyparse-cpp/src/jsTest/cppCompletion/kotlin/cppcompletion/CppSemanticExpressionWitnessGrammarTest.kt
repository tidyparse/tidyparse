package cppcompletion

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppSemanticExpressionWitnessGrammarTest {
  private val root = info("type:Root")
  private val leaf = info("type:Leaf")
  private val other = info("type:Other")
  private val typeInfo = info("type:TypeInfo")
  private val rootPointer = pointer("type:RootPtr", root)
  private val leafPointer = pointer("type:LeafPtr", leaf)
  private val otherPointer = pointer("type:OtherPtr", other)

  @Test
  fun namedCastWitnessesKeepTargetsAndOperandsCorrelated() {
    val language = language(
      values = listOf(
        value("rootPointer", "Root *", rootPointer),
        value("leafPointer", "Leaf *", leafPointer),
        value("root", "Root", root)
      ),
      functions = listOf(function("makeRootPointer", "Root *", rootPointer)),
      witnesses = listOf(
        cast(
          "dynamicCast", type("Leaf *", leafPointer),
          opaque("Root *", rootPointer), opaque("Leaf *", leafPointer, "prvalue")
        ),
        cast(
          "reinterpretCast", type("Other *", otherPointer),
          opaque("Leaf *", leafPointer), opaque("Other *", otherPointer, "prvalue")
        ),
        cast(
          "dynamicCast", type("const Leaf &", leaf.reference(isConst = true)),
          opaque("Root", root), opaque("const Leaf", leaf.copy(isConst = true))
        )
      )
    )

    assertTrue(language.recognizes("dynamic_cast<Leaf *>(rootPointer);"))
    assertTrue(language.recognizes("reinterpret_cast<Other *>(leafPointer);"))
    assertTrue(language.recognizes("dynamic_cast<const Leaf &>(root);"))
    assertFalse(
      language.recognizes("dynamic_cast<Leaf *>(leafPointer);"),
      "a target from one witness must not combine with another operand"
    )
    assertFalse(language.recognizes("reinterpret_cast<Other *>(rootPointer);"))
    assertFalse(
      language.recognizes("dynamic_cast<Leaf *>(makeRootPointer());"),
      "the lvalue operand profile must not widen to a prvalue"
    )
    assertFalse(
      language.recognizes("dynamic_cast<Leaf &>(root);"),
      "the exact cv-qualified reference type-id must be retained"
    )
  }

  @Test
  fun namedCastTargetUsesOnlyIndependentSpellingsOfTheWitnessBaseType() {
    val language = language(
      values = listOf(value("rootPointer", "Root *", rootPointer)),
      witnesses = listOf(
        cast(
          "dynamicCast", type("::Leaf *", leafPointer),
          opaque("Root *", rootPointer), opaque("Leaf *", leafPointer, "prvalue")
        ),
        // The spelling names an independently indexed but different canonical type. Supplying the
        // Leaf pointer identity in the witness must not turn `Other` into an alias for Leaf.
        cast(
          "dynamicCast", type("Other *", leafPointer),
          opaque("Root *", rootPointer), opaque("Leaf *", leafPointer, "prvalue")
        )
      )
    )

    assertTrue(language.recognizes("dynamic_cast<::Leaf *>(rootPointer);"))
    assertTrue(
      language.recognizes("dynamic_cast<Leaf *>(rootPointer);"),
      "an independently indexed spelling of the same base type should retain the exact declarator"
    )
    assertFalse(
      language.recognizes("dynamic_cast<Other *>(rootPointer);"),
      "a same-shaped spelling with a different canonical identity is not an alias"
    )
  }

  @Test
  fun typeidFormsRetainTheirShapeAndConstLvalueResult() {
    val constTypeInfo = typeInfo.copy(isConst = true)
    val language = language(
      values = listOf(value("root", "Root", root)),
      functions = listOf(function("makeRoot", "Root", root)),
      witnesses = listOf(
        expressionWitness(
          syntax = "typeidExpression",
          expressionOperand = opaque("Root", root),
          result = opaque("const TypeInfo", constTypeInfo)
        ),
        expressionWitness(
          syntax = "typeidType",
          typeOperand = type("Root", root),
          result = opaque("const TypeInfo", constTypeInfo)
        )
      )
    )

    assertTrue(language.recognizes("typeid(root);"))
    assertTrue(language.recognizes("typeid(Root);"))
    assertTrue(
      language.recognizes("const_cast<TypeInfo &>(typeid(root));"),
      "the typeid result must enter the exact const-lvalue lattice state"
    )
    assertFalse(language.recognizes("typeid(makeRoot());"), "operand category must remain exact")
  }

  @Test
  fun typeidExpressionDoesNotRequireTheOperandsStaticTypeToBeSourceSpellable() {
    val closureType = info("type:closure", isSourceSpellable = false)
    val constTypeInfo = typeInfo.copy(isConst = true)
    val language = language(
      values = listOf(value("closure", "Closure", closureType)),
      witnesses = listOf(expressionWitness(
        syntax = "typeidExpression",
        expressionOperand = opaque("Closure", closureType),
        result = opaque("const TypeInfo", constTypeInfo)
      ))
    )

    assertTrue(language.recognizes("typeid(closure);"))
    assertFalse(language.recognizes("typeid(Closure);"), "a value witness does not spell a type-id")
  }

  @Test
  fun malformedOrUnprovenExpressionWitnessProfilesPublishNothing() {
    val incomplete = info("type:Incomplete", isComplete = false)
    val hidden = info("type:Hidden", isSourceSpellable = false)
    val exactNonTypeSpecialization = info("type:Wrapper7")
    val constTypeInfo = typeInfo.copy(isConst = true)
    val witnesses = listOf(
      cast(
        "dynamicCast", type("Leaf *", leafPointer),
        opaque("Root *", rootPointer), opaque("Leaf *", leafPointer, "prvalue")
      ).copy(authoritative = false),
      cast(
        "reinterpretCast", type("Other *", otherPointer),
        opaque("Root *", rootPointer), opaque("Other *", otherPointer, "prvalue")
      ).copy(validation = "overloadResolution"),
      expressionWitness(
        syntax = "dynamicCast",
        typeOperand = type("Leaf *", leafPointer),
        result = opaque("Leaf *", leafPointer, "prvalue")
      ),
      cast(
        "dynamicCast", type("Leaf *", leafPointer),
        opaque("Root *", rootPointer), opaque("Other *", otherPointer, "prvalue")
      ),
      expressionWitness(
        syntax = "typeidType", typeOperand = type("Incomplete", incomplete),
        result = opaque("const TypeInfo", constTypeInfo)
      ),
      expressionWitness(
        syntax = "typeidType", typeOperand = type("Hidden", hidden),
        result = opaque("const TypeInfo", constTypeInfo)
      ),
      expressionWitness(
        syntax = "typeidType",
        typeOperand = type("Wrapper<7>", exactNonTypeSpecialization),
        result = opaque("const TypeInfo", constTypeInfo)
      ),
      expressionWitness(
        syntax = "typeidExpression", typeOperand = type("Root", root),
        expressionOperand = opaque("Root", root),
        result = opaque("const TypeInfo", constTypeInfo)
      ),
      expressionWitness(
        syntax = "typeidType", typeOperand = type("Root", root),
        result = opaque("TypeInfo", typeInfo, "prvalue")
      )
    )
    val language = language(
      values = listOf(value("rootPointer", "Root *", rootPointer), value("root", "Root", root)),
      witnesses = witnesses
    )

    assertFalse(language.recognizes("dynamic_cast<Leaf *>(rootPointer);"))
    assertFalse(language.recognizes("reinterpret_cast<Other *>(rootPointer);"))
    assertFalse(language.recognizes("typeid(Incomplete);"))
    assertFalse(language.recognizes("typeid(Hidden);"))
    assertFalse(
      language.recognizes("typeid(Wrapper<7>);"),
      "abstracted literal terminals cannot preserve an exact specialization type-id"
    )
    assertFalse(language.recognizes("typeid(root);"), "typeid(expr) cannot carry a type operand")
    assertFalse(language.recognizes("typeid(Root);"), "typeid must have a const-lvalue result")
  }

  private fun cast(
    syntax: String,
    target: CppTypeProfile,
    operand: CppExpressionProfile,
    result: CppExpressionProfile
  ) = expressionWitness(syntax, target, operand, result)

  private fun expressionWitness(
    syntax: String,
    typeOperand: CppTypeProfile? = null,
    expressionOperand: CppExpressionProfile? = null,
    result: CppExpressionProfile
  ) = CppExpressionWitness(
    syntax = syntax,
    validation = "semaExpressionBuild",
    typeOperand = typeOperand,
    expressionOperand = expressionOperand,
    result = result,
    authoritative = true
  )

  private fun type(spelling: String, info: CppTypeInfo) = CppTypeProfile(
    type = spelling, canonicalType = spelling, typeInfo = info
  )

  private fun opaque(
    spelling: String,
    info: CppTypeInfo,
    category: String = "lvalue"
  ) = CppExpressionProfile(
    kind = "opaque", spelling = null, objectKind = "ordinary",
    type = spelling, canonicalType = spelling,
    typeInfo = info, valueCategory = category
  )

  private fun value(name: String, spelling: String, info: CppTypeInfo) = CppReference(
    name = name, type = spelling, canonicalType = spelling,
    kind = "variable", isValue = true, typeInfo = info
  )

  private fun function(name: String, spelling: String, info: CppTypeInfo) = CppReference(
    name = name, returnType = spelling, canonicalReturnType = spelling,
    kind = "function", isCallable = true, returnTypeInfo = info
  )

  private fun info(
    id: String,
    kind: String = "record",
    isComplete: Boolean = true,
    isSourceSpellable: Boolean = true
  ) = CppTypeInfo(
    id = id, canonicalId = id, valueCanonicalId = id, kind = kind,
    isSourceSpellable = isSourceSpellable, isComplete = isComplete
  )

  private fun pointer(id: String, pointee: CppTypeInfo) = info(id, kind = "pointer").copy(
    pointeeCanonicalId = pointee.valueCanonicalId
  )

  private fun CppTypeInfo.reference(isConst: Boolean = false) = copy(
    id = "ref:$valueCanonicalId:$isConst",
    canonicalId = "ref:$valueCanonicalId:$isConst",
    kind = "lvalueReference",
    isConst = isConst
  )

  private fun language(
    values: List<CppReference> = emptyList(),
    functions: List<CppReference> = emptyList(),
    witnesses: List<CppExpressionWitness>
  ): CppSuffixGrammar {
    val identifiers = buildSet {
      addAll(listOf(
        "Root", "Leaf", "Other", "TypeInfo", "Incomplete", "Hidden", "Wrapper"
      ))
      values.mapTo(this, CppReference::name)
      functions.mapTo(this, CppReference::name)
    }
    return CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = identifiers,
        sourceIdentifiers = identifiers,
        completionKind = "expression",
        values = values,
        functions = functions,
        // Type operands/results must be independently present in the Sema index; a witness does
        // not grant its own display spelling permission to appear in another type-id production.
        types = listOf(
          typeDeclaration("Root", root), typeDeclaration("Leaf", leaf),
          typeDeclaration("Other", other), typeDeclaration("TypeInfo", typeInfo)
        ),
        expressionWitnesses = witnesses
      ),
      emptyList()
    )
  }

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)

  private fun typeDeclaration(name: String, info: CppTypeInfo) = CppReference(
    name = name, type = name, canonicalType = name,
    kind = "class", isType = true, typeInfo = info, completionVisible = true
  )
}
