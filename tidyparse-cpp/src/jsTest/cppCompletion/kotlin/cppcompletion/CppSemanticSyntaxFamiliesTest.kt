package cppcompletion

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppSemanticSyntaxFamiliesTest {
  private val intInfo = info("builtin:int", "builtin")

  @Test
  fun reportedOperatorSignaturesSupportLongTypedChainsWithoutWideningOperands() {
    val emitter = info("record:Emitter", "record")
    val glyph = info("record:Glyph", "record")
    val other = info("record:Other", "record")
    val emitterRef = reference(emitter, "lvalueReference")
    val glyphConstRef = reference(glyph, "lvalueReference", isConst = true)
    val values = listOf(value("out", "Emitter", emitter)) +
      (1..8).map { value("g$it", "Glyph", glyph) } + value("other", "Other", other)
    val insertion = CppReference(
      name = "operator<<", returnType = "Emitter &", kind = "function",
      isCallable = true, returnTypeInfo = emitterRef,
      parameters = listOf(
        CppParameter(type = "Emitter &", typeInfo = emitterRef),
        CppParameter(type = "const Glyph &", typeInfo = glyphConstRef)
      )
    )
    val language = language(
      values = values,
      types = listOf(type("Emitter", emitter), type("Glyph", glyph), type("Other", other)),
      functions = listOf(insertion)
    )

    assertTrue(language.recognizes("out << g1 << g2 << g3 << g4 << g5 << g6 << g7 << g8;"))
    assertFalse(language.recognizes("out << g1 << g2 << other;"))
  }

  @Test
  fun operatorOperandsDoNotEscapeTheirSemanticGroupingThroughCppPrecedence() {
    val emitter = info("record:PrecedenceEmitter", "record")
    val emitterRef = reference(emitter, "lvalueReference")
    val doubleInfo = info("builtin:double", "builtin")
    val boolInfo = info("builtin:bool", "builtin")
    val insertion = CppReference(
      name = "operator<<", returnType = "Emitter &", kind = "function",
      isCallable = true, returnTypeInfo = emitterRef,
      parameters = listOf(
        CppParameter(type = "Emitter &", typeInfo = emitterRef),
        CppParameter(type = "double", typeInfo = doubleInfo)
      )
    )
    val booleanInsertion = insertion.copy(
      parameters = listOf(
        CppParameter(type = "Emitter &", typeInfo = emitterRef),
        CppParameter(type = "bool", typeInfo = boolInfo)
      )
    )
    val language = language(
      values = listOf(
        value("out", "Emitter", emitter),
        value("left", "double", doubleInfo),
        value("right", "double", doubleInfo)
      ),
      types = listOf(
        type("Emitter", emitter), type("double", doubleInfo), type("bool", boolInfo)
      ),
      functions = listOf(insertion, booleanInsertion)
    )

    assertTrue(language.recognizes("out << left << right;"), "left-associative stream chain")
    assertTrue(
      language.recognizes("out << left + right;"),
      "an additive expression binds tightly enough to be the shift operator's right operand"
    )
    assertFalse(
      language.recognizes("out << left != right;"),
      "surface precedence parses this as (out << left) != right, not out << (left != right)"
    )
    assertTrue(
      language.recognizes("out << (left != right);"),
      "the converted boolean operand is valid when its grouping is explicit"
    )
  }

  @Test
  fun weakerSemanticOperatorOnTheLeftRequiresExplicitGrouping() {
    val a = info("record:LeftWeak:A", "record")
    val b = info("record:LeftWeak:B", "record")
    val c = info("record:LeftWeak:C", "record")
    val sum = info("record:LeftWeak:Sum", "record")
    val product = info("record:LeftWeak:Product", "record")
    val language = language(
      values = listOf(value("a", "A", a), value("b", "B", b), value("c", "C", c)),
      types = listOf(
        type("A", a), type("B", b), type("C", c), type("Sum", sum),
        type("Product", product)
      ),
      functions = listOf(
        binaryOperator("+", "A", a, "B", b, "Sum", sum),
        binaryOperator("*", "Sum", sum, "C", c, "Product", product)
      )
    )

    assertTrue(language.recognizes("(a + b) * c;"))
    assertFalse(
      language.recognizes("a + b * c;"),
      "C++ groups the unparenthesized spelling as a + (b * c), whose overloads are absent"
    )
  }

  @Test
  fun strongerSemanticOperatorCanFeedTheLeftOfAWeakerOperator() {
    val a = info("record:StrongLeft:A", "record")
    val b = info("record:StrongLeft:B", "record")
    val c = info("record:StrongLeft:C", "record")
    val product = info("record:StrongLeft:Product", "record")
    val result = info("record:StrongLeft:Result", "record")
    val language = language(
      values = listOf(value("a", "A", a), value("b", "B", b), value("c", "C", c)),
      types = listOf(
        type("A", a), type("B", b), type("C", c), type("Product", product),
        type("Result", result)
      ),
      functions = listOf(
        binaryOperator("*", "A", a, "B", b, "Product", product),
        binaryOperator("+", "Product", product, "C", c, "Result", result)
      )
    )

    assertTrue(language.recognizes("a * b + c;"))
  }

  @Test
  fun rightOperandAtTheSamePrecedenceRequiresParenthesesForLeftAssociativeOperators() {
    val a = info("record:RightAssoc:A", "record")
    val b = info("record:RightAssoc:B", "record")
    val c = info("record:RightAssoc:C", "record")
    val finalInfo = info("record:RightAssoc:Final", "record")
    val language = language(
      values = listOf(value("a", "A", a), value("b", "B", b), value("c", "C", c)),
      types = listOf(type("A", a), type("B", b), type("C", c), type("Final", finalInfo)),
      functions = listOf(
        binaryOperator("+", "A", a, "B", b, "Final", finalInfo),
        binaryOperator("+", "B", b, "C", c, "B", b)
      )
    )

    assertFalse(
      language.recognizes("a + b + c;"),
      "the surface spelling is (a + b) + c and Final + C is not a reported overload"
    )
    assertTrue(language.recognizes("a + (b + c);"))
  }

  @Test
  fun mixedOperatorsAtOnePrecedenceRemainLeftAssociative() {
    val a = info("record:Mixed:A", "record")
    val b = info("record:Mixed:B", "record")
    val c = info("record:Mixed:C", "record")
    val intermediate = info("record:Mixed:Intermediate", "record")
    val finalInfo = info("record:Mixed:Final", "record")
    val language = language(
      values = listOf(value("a", "A", a), value("b", "B", b), value("c", "C", c)),
      types = listOf(
        type("A", a), type("B", b), type("C", c),
        type("Intermediate", intermediate), type("Final", finalInfo)
      ),
      functions = listOf(
        binaryOperator("+", "A", a, "B", b, "Intermediate", intermediate),
        binaryOperator("-", "Intermediate", intermediate, "C", c, "Final", finalInfo)
      )
    )

    assertTrue(language.recognizes("a + b - c;"))
    assertFalse(language.recognizes("a + (b - c);"))
  }

  @Test
  fun operatorReferenceParametersPreserveTheNestedResultCategory() {
    val emitter = info("record:Binding:Emitter", "record")
    val number = info("record:Binding:Number", "record")
    val emitterRef = reference(emitter, "lvalueReference")
    val constNumberRef = reference(number, "lvalueReference", isConst = true)
    val mutableNumberRef = reference(number, "lvalueReference")
    val addition = binaryOperator("+", "Number", number, "Number", number, "Number", number)
    val insertion = CppReference(
      name = "operator<<", returnType = "Emitter &", kind = "function",
      isCallable = true, returnTypeInfo = emitterRef,
      parameters = listOf(
        CppParameter(type = "Emitter &", typeInfo = emitterRef),
        CppParameter(type = "const Number &", typeInfo = constNumberRef)
      )
    )
    val values = listOf(
      value("out", "Emitter", emitter), value("x", "Number", number),
      value("y", "Number", number)
    )
    val types = listOf(type("Emitter", emitter), type("Number", number))

    assertTrue(
      language(values, types, listOf(addition, insertion)).recognizes("out << x + y;"),
      "a const lvalue reference can bind the prvalue result of the nested addition"
    )
    val mutableInsertion = insertion.copy(
      parameters = listOf(
        CppParameter(type = "Emitter &", typeInfo = emitterRef),
        CppParameter(type = "Number &", typeInfo = mutableNumberRef)
      )
    )
    val mutableLanguage = language(values, types, listOf(addition, mutableInsertion))
    assertTrue(mutableLanguage.recognizes("out << x;"), "the mutable lvalue itself can bind")
    assertFalse(
      mutableLanguage.recognizes("out << x + y;"),
      "a mutable lvalue reference cannot bind the prvalue result of the nested addition"
    )
  }

  @Test
  fun memberOperatorReceiversPreserveRefQualifiersAcrossNestedPrecedence() {
    val box = info("record:Receiver:Box", "record")
    val seed = info("record:Receiver:Seed", "record")
    val term = info("record:Receiver:Term", "record")
    val result = info("record:Receiver:Result", "record")
    val boxFactory = binaryOperator("*", "Seed", seed, "Seed", seed, "Box", box)
    val member = CppReference(
      name = "operator+", returnType = "Result", kind = "method", ownerType = "Box",
      canonicalOwnerType = "Box", isCallable = true, isMember = true,
      ownerTypeInfo = box, returnTypeInfo = result, refQualifier = "&",
      parameters = listOf(CppParameter(type = "Term", typeInfo = term))
    )
    val values = listOf(
      value("box", "Box", box), value("s1", "Seed", seed), value("s2", "Seed", seed),
      value("term", "Term", term)
    )
    val types = listOf(
      type("Box", box), type("Seed", seed), type("Term", term), type("Result", result)
    )

    val lvalueQualified = language(values, types, listOf(boxFactory, member))
    assertTrue(lvalueQualified.recognizes("box + term;"))
    assertFalse(lvalueQualified.recognizes("(s1 * s2) + term;"))

    val rvalueQualified = language(
      values, types, listOf(boxFactory, member.copy(refQualifier = "&&"))
    )
    assertFalse(rvalueQualified.recognizes("box + term;"))
    assertTrue(rvalueQualified.recognizes("(s1 * s2) + term;"))
  }

  @Test
  fun constCastsPreserveTheExactSemaPointeeType() {
    val base = info("record:Base", "record")
    val basePointer = pointer("pointer:Base", base)
    val constBasePointer = pointer("pointer:const-Base", base, pointeeConst = true)
    val language = language(
      values = listOf(
        value("base", "Base *", basePointer),
        value("readonly", "const Base *", constBasePointer),
        value("view", "const Base &", base.copy(isConst = true))
      ),
      types = listOf(
        type("Base", base),
        type("Base *", basePointer), type("const Base *", constBasePointer)
      )
    )

    assertTrue(language.recognizes("const_cast<Base *>(readonly);"), "pointer const_cast")
    assertTrue(language.recognizes("const_cast<Base &>(view);"), "reference const_cast")
  }

  @Test
  fun activeVariadicConstructorFactsDriveListInitialization() {
    val bag = info("record:Bag", "record")
    val constructor = CppReference(
      name = "Bag", returnType = "Bag", kind = "constructor", ownerType = "Bag",
      isCallable = true, isMember = true, activeCallable = true,
      ownerTypeInfo = bag, returnTypeInfo = bag,
      parameters = listOf(CppParameter(type = "int", typeInfo = intInfo, isPack = true))
    )
    val context = context(
      types = listOf(type("Bag", bag)),
      functions = listOf(constructor),
      requiredIdentifier = "bag"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes("Bag bag{1, 2, 3, 4};"))
    assertFalse(language.recognizes("Bag bag{\"wrong\"};"))
  }

  private fun language(
    values: List<CppReference> = emptyList(),
    types: List<CppReference> = emptyList(),
    functions: List<CppReference> = emptyList()
  ) = CppCompletionGrammar().generate(
    context(values, types, functions), emptyList()
  )

  private fun context(
    values: List<CppReference> = emptyList(),
    types: List<CppReference> = emptyList(),
    functions: List<CppReference> = emptyList(),
    requiredIdentifier: String? = null
  ): CppCompletionContext {
    val all = values + types + functions
    val identifiers = all.flatMap { it.name.split("::") }
      .filter { it.matches(Regex("[A-Za-z_][A-Za-z0-9_]*")) }.toSet()
    return CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      completionKind = "expression",
      values = values,
      types = types,
      functions = functions,
      binaryOperatorWitnesses = binaryOperatorWitnesses(values, types, functions),
      requiredIdentifier = requiredIdentifier
    )
  }

  /**
   * These are model-level grammar tests, so project each declared test operator into the exact
   * type/category lanes that the native BuildBinOp probe would return. Production code never
   * performs this signature projection.
   */
  private fun binaryOperatorWitnesses(
    values: List<CppReference>,
    types: List<CppReference>,
    functions: List<CppReference>
  ): List<CppBinaryOperatorWitness> {
    val facts = values + types
    fun valueInfo(info: CppTypeInfo?): CppTypeInfo? {
      val id = info?.valueCanonicalId ?: return null
      return facts.asSequence().mapNotNull(CppReference::typeInfo)
        .firstOrNull { candidate ->
          candidate.valueCanonicalId == id &&
            candidate.kind !in setOf("lvalueReference", "rvalueReference")
        }
    }
    fun valueType(spelling: String): String = spelling.trim()
      .removeSuffix("&&").removeSuffix("&").trim()
      .removePrefix("const ").removePrefix("volatile ").trim()
    fun categories(info: CppTypeInfo?): List<String> = when (info?.kind) {
      "lvalueReference" -> if (info.isConst && !info.isVolatile)
        listOf("lvalue", "xvalue", "prvalue") else listOf("lvalue")
      "rvalueReference" -> listOf("xvalue", "prvalue")
      else -> listOf("lvalue", "xvalue", "prvalue")
    }
    fun profile(type: String, info: CppTypeInfo, category: String) = CppExpressionProfile(
      kind = "opaque", spelling = null, objectKind = "ordinary",
      type = type, canonicalType = type, typeInfo = info, valueCategory = category
    )

    return functions.mapIndexedNotNull { index, original ->
      val spelling = original.name.substringAfterLast("::").removePrefix("operator").trim()
        .takeIf { it in setOf(
          "+", "-", "*", "/", "%", "<<", ">>", "<=>", "<", "<=", ">", ">=",
          "==", "!=", "&", "^", "|", "&&", "||"
        ) } ?: return@mapIndexedNotNull null
      val leftParameter = if (original.isMember == true) null
      else original.parameters.getOrNull(0) ?: return@mapIndexedNotNull null
      val rightParameter = original.parameters.getOrNull(if (original.isMember == true) 0 else 1)
        ?: return@mapIndexedNotNull null
      val leftInfo = if (leftParameter == null) valueInfo(original.ownerTypeInfo)
      else valueInfo(leftParameter.typeInfo)
      val rightInfo = valueInfo(rightParameter.typeInfo)
      val resultInfo = valueInfo(original.returnTypeInfo)
      if (leftInfo == null || rightInfo == null || resultInfo == null)
        return@mapIndexedNotNull null
      val leftType = if (leftParameter == null)
        valueType(original.canonicalOwnerType ?: original.ownerType.orEmpty())
      else valueType(leftParameter.canonicalType ?: leftParameter.type)
      val rightType = valueType(rightParameter.canonicalType ?: rightParameter.type)
      val resultType = valueType(original.canonicalReturnType ?: original.returnType.orEmpty())
      val leftCategories = if (leftParameter != null) categories(leftParameter.typeInfo)
      else when (original.refQualifier) {
        "&" -> listOf("lvalue")
        "&&" -> listOf("xvalue", "prvalue")
        else -> listOf("lvalue", "xvalue", "prvalue")
      }
      val resultCategory = when (original.returnTypeInfo?.kind) {
        "lvalueReference" -> "lvalue"
        "rvalueReference" -> "xvalue"
        else -> "prvalue"
      }
      val callable = original.copy(id = "test:operator:$index")
      leftCategories.flatMap { leftCategory ->
        categories(rightParameter.typeInfo).map { rightCategory ->
          CppBinaryOperatorWitness(
            name = "operator$spelling",
            syntax = "binaryOperator",
            operatorSpelling = spelling,
            validation = "semaBinaryOperatorExpression",
            targetId = requireNotNull(callable.id),
            left = profile(leftType, leftInfo, leftCategory),
            right = profile(rightType, rightInfo, rightCategory),
            callable = callable,
            result = profile(resultType, resultInfo, resultCategory),
            authoritative = true
          )
        }
      }
    }.flatten()
  }

  private fun info(id: String, kind: String) = CppTypeInfo(
    id = id, canonicalId = id, valueCanonicalId = id, kind = kind,
    isSourceSpellable = true
  )

  private fun reference(info: CppTypeInfo, kind: String, isConst: Boolean = false) = info.copy(
    id = "$kind:${info.valueCanonicalId}:$isConst",
    canonicalId = "$kind:${info.valueCanonicalId}:$isConst",
    kind = kind,
    isConst = isConst
  )

  private fun pointer(
    id: String,
    pointee: CppTypeInfo,
    pointeeConst: Boolean = false
  ) = CppTypeInfo(
    id = id, canonicalId = id, valueCanonicalId = id, kind = "pointer",
    pointeeCanonicalId = pointee.valueCanonicalId, pointeeIsConst = pointeeConst,
    isSourceSpellable = true
  )

  private fun type(name: String, info: CppTypeInfo) = CppReference(
    name = name, type = name, canonicalType = name, kind = "type",
    isType = true, typeInfo = info
  )

  private fun value(name: String, type: String, info: CppTypeInfo) = CppReference(
    name = name, type = type, canonicalType = type, kind = "variable",
    isValue = true, typeInfo = info
  )

  private fun binaryOperator(
    token: String,
    leftType: String,
    leftInfo: CppTypeInfo,
    rightType: String,
    rightInfo: CppTypeInfo,
    resultType: String,
    resultInfo: CppTypeInfo
  ) = CppReference(
    name = "operator$token", returnType = resultType, canonicalReturnType = resultType,
    kind = "function", isCallable = true, returnTypeInfo = resultInfo,
    parameters = listOf(
      CppParameter(type = leftType, canonicalType = leftType, typeInfo = leftInfo),
      CppParameter(type = rightType, canonicalType = rightType, typeInfo = rightInfo)
    )
  )

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)
}
