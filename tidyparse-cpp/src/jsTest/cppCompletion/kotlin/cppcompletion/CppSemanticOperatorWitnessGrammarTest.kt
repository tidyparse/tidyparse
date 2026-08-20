package cppcompletion

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppSemanticOperatorWitnessGrammarTest {
  private val bool = info("builtin:bool", "builtin")

  @Test
  fun deadOperatorOperandChoicesAreClassifiedAsGeneratedNonterminals() {
    assertTrue(isGeneratedExpressionSymbol("OPERATOR_WITNESS_OPERAND_17"))
  }

  @Test
  fun rewrittenSurfaceRelationDoesNotUseTheSelectedCallablesParameterOrderOrResult() {
    val left = info("record:Left")
    val right = info("record:Right")
    val ordering = info("record:Ordering")
    val selected = callable(
      id = "function:compare",
      name = "compare::operator<=>",
      resultType = "Ordering",
      resultInfo = ordering,
      parameters = listOf(parameter("Right", right), parameter("Left", left))
    )
    val valid = witness(
      spelling = "<",
      validation = "semaDefaultedDefinition",
      targetId = "function:compare",
      callable = selected,
      left = opaque("Left", left),
      right = opaque("Right", right),
      result = opaque("bool", bool, "prvalue")
    )
    val context = context(
      values = listOf(value("left", "Left", left), value("right", "Right", right)),
      types = listOf(
        type("Left", left), type("Right", right), type("Ordering", ordering),
        type("bool", bool)
      ),
      functions = listOf(selected),
      witnesses = listOf(
        valid,
        valid.copy(
          name = "operator>", operatorSpelling = ">", validation = "overloadResolution"
        ),
        valid.copy(
          name = "operator==", operatorSpelling = "==", targetId = "function:other"
        )
      )
    )
    assertTrue(
      valid.copy(validation = "recursiveDefinitionInstantiation")
        .hasWellFormedTargetIdentity(),
      "a class-template member instantiation has an exact function identity but no function primary"
    )
    val language = language(context)

    assertTrue(language.recognizes("left < right;"))
    assertFalse(
      language.recognizes("right < left;"),
      "rewritten operand order must come from the BuildBinOp relation"
    )
    assertFalse(
      language.recognizes("left <=> right;"),
      "the selected operator name must not replace the authenticated surface spelling"
    )
    assertFalse(language.recognizes("left > right;"), "shallow validation is not authority")
    assertFalse(language.recognizes("left == right;"), "mismatched target identity fails closed")
  }

  @Test
  fun exactSelfFeedingWitnessSupportsLongChainsWithoutWideningTheRightOperand() {
    val stream = info("record:Stream")
    val payload = info("record:Payload")
    val other = info("record:Other")
    val streamReference = reference(stream, "lvalueReference")
    val selected = callable(
      id = "function:insert",
      name = "operator<<",
      resultType = "Stream &",
      resultInfo = streamReference,
      parameters = listOf(parameter("Payload", payload), parameter("Stream &", streamReference))
    )
    val insertion = witness(
      spelling = "<<",
      validation = "semaBinaryOperatorExpression",
      targetId = "function:insert",
      callable = selected,
      left = opaque("Stream", stream),
      right = opaque("Payload", payload),
      result = opaque("Stream", streamReference)
    )
    val values = listOf(value("out", "Stream", stream)) +
      (1..8).map { value("p$it", "Payload", payload) } +
      value("other", "Other", other)
    val language = language(context(
      values = values,
      types = listOf(type("Stream", stream), type("Payload", payload), type("Other", other)),
      functions = listOf(selected),
      witnesses = listOf(insertion)
    ))

    assertTrue(language.recognizes("out << p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8;"))
    assertFalse(language.recognizes("out << p1 << p2 << other;"))
  }

  @Test
  fun exactLiteralLeftOperandCannotBeReplacedByThePriorOpaqueResult() {
    val int = info("builtin:int", "builtin")
    val payload = info("record:Payload")
    val selected = callable(
      id = "function:literal-left",
      name = "operator+",
      resultType = "int",
      resultInfo = int,
      parameters = listOf(parameter("int", int), parameter("Payload", payload))
    )
    val exactZero = CppExpressionProfile(
      kind = "integerZero",
      spelling = "0",
      objectKind = "ordinary",
      type = "int",
      canonicalType = "int",
      typeInfo = int,
      valueCategory = "prvalue"
    )
    val language = language(context(
      values = listOf(
        value("first", "Payload", payload), value("second", "Payload", payload)
      ),
      types = listOf(type("int", int), type("Payload", payload)),
      functions = listOf(selected),
      witnesses = listOf(witness(
        spelling = "+",
        validation = "semaBinaryOperatorExpression",
        targetId = "function:literal-left",
        callable = selected,
        left = exactZero,
        right = opaque("Payload", payload),
        result = opaque("int", int, "prvalue")
      ))
    ))

    assertTrue(language.recognizes("0 + first;"))
    assertFalse(
      language.recognizes("0 + first + second;"),
      "the result of a literal-constrained edge is not another exact literal-zero operand"
    )
  }

  @Test
  fun threeWayComparisonRetainsItsDistinctPrecedenceAndSurfaceToken() {
    val item = info("record:Item")
    val ordering = info("record:Ordering")
    val selected = callable(
      id = "function:spaceship",
      name = "operator<=>",
      resultType = "Ordering",
      resultInfo = ordering,
      parameters = listOf(parameter("Item", item), parameter("Item", item))
    )
    val language = language(context(
      values = listOf(value("left", "Item", item), value("right", "Item", item)),
      types = listOf(type("Item", item), type("Ordering", ordering)),
      functions = listOf(selected),
      witnesses = listOf(witness(
        spelling = "<=>",
        validation = "semaDefaultedDefinition",
        targetId = "function:spaceship",
        callable = selected,
        left = opaque("Item", item),
        right = opaque("Item", item),
        result = opaque("Ordering", ordering, "prvalue")
      ))
    ))

    assertEquals(
      listOf("left", "<=", ">", "right", ";"),
      cppLines("left <=> right;").single().tokens.map { it.text }
    )
    assertEquals("<=>", listOf("<=", ">").renderCppTokens())
    assertTrue(language.recognizes("left <=> right;"))
  }

  @Test
  fun weakerExactOperandIsRecoveredOnlyThroughSoundParenthesization() {
    val factor = info("record:Factor")
    val left = info("record:LeftTerm")
    val right = info("record:RightTerm")
    val intermediate = info("record:Intermediate")
    val product = info("record:Product")
    val intermediateReference = reference(intermediate, "lvalueReference")
    val addition = callable(
      id = "function:add",
      name = "operator+",
      resultType = "Intermediate &",
      resultInfo = intermediateReference,
      parameters = listOf(parameter("LeftTerm", left), parameter("RightTerm", right))
    )
    val multiplication = callable(
      id = "function:multiply",
      name = "operator*",
      resultType = "Product",
      resultInfo = product,
      parameters = listOf(parameter("Factor", factor), parameter("Intermediate", intermediate))
    )
    val language = language(context(
      values = listOf(
        value("factor", "Factor", factor), value("left", "LeftTerm", left),
        value("right", "RightTerm", right)
      ),
      types = listOf(
        type("Factor", factor), type("LeftTerm", left), type("RightTerm", right),
        type("Intermediate", intermediate), type("Product", product)
      ),
      functions = listOf(addition, multiplication),
      witnesses = listOf(
        witness(
          spelling = "+",
          validation = "semaBinaryOperatorExpression",
          targetId = "function:add",
          callable = addition,
          left = opaque("LeftTerm", left),
          right = opaque("RightTerm", right),
          result = opaque("Intermediate", intermediate, "lvalue")
        ),
        witness(
          spelling = "*",
          validation = "semaBinaryOperatorExpression",
          targetId = "function:multiply",
          callable = multiplication,
          left = opaque("Factor", factor),
          right = opaque("Intermediate", intermediate, "lvalue"),
          result = opaque("Product", product, "prvalue")
        )
      )
    ))

    assertTrue(language.recognizes("factor * (left + right);"))
    assertFalse(
      language.recognizes("factor * left + right;"),
      "the weaker exact RHS cannot cross the multiplicative precedence boundary unparenthesized"
    )
  }

  @Test
  fun compoundAssignmentWitnessAddsOnlyItsExactStatementRelation() {
    val accumulator = info("record:Accumulator")
    val delta = info("record:Delta")
    val other = info("record:Other")
    val bool = info("builtin:bool", "builtin")
    val accumulatorReference = reference(accumulator, "lvalueReference")
    val selected = callable(
      id = "function:accumulate",
      name = "Accumulator::operator+=",
      resultType = "Accumulator &",
      resultInfo = accumulatorReference,
      parameters = listOf(parameter("Delta", delta))
    )
    val valid = witness(
      spelling = "+=",
      validation = "semaBinaryOperatorExpression",
      targetId = "function:accumulate",
      callable = selected,
      left = opaque("Accumulator", accumulator),
      right = opaque("Delta", delta),
      result = opaque("Accumulator", accumulatorReference)
    )
    val language = language(context(
      values = listOf(
        value("accumulator", "Accumulator", accumulator),
        value("alternate", "Accumulator", accumulator),
        value("delta", "Delta", delta), value("other", "Other", other),
        value("choose", "bool", bool)
      ),
      types = listOf(
        type("Accumulator", accumulator), type("Delta", delta), type("Other", other),
        type("bool", bool)
      ),
      functions = listOf(selected),
      witnesses = listOf(
        valid,
        valid.copy(name = "operator*=", operatorSpelling = "*=")
      )
    ))

    assertTrue(language.recognizes("accumulator += delta;"))
    assertFalse(language.recognizes("other += delta;"))
    assertFalse(language.recognizes("accumulator += other;"))
    assertTrue(language.recognizes("(choose ? accumulator : alternate) += delta;"))
    assertFalse(
      language.recognizes("choose ? accumulator : alternate += delta;"),
      "an unparenthesized conditional would move the witnessed mutation into one branch"
    )
    assertFalse(
      language.recognizes("accumulator *= delta;"),
      "a corrupt surface spelling cannot be paired with a selected operator+="
    )
  }

  private fun witness(
    spelling: String,
    validation: String,
    targetId: String,
    callable: CppReference,
    left: CppExpressionProfile,
    right: CppExpressionProfile,
    result: CppExpressionProfile
  ) = CppBinaryOperatorWitness(
    name = "operator$spelling",
    syntax = "binaryOperator",
    operatorSpelling = spelling,
    validation = validation,
    targetId = targetId,
    left = left,
    right = right,
    callable = callable,
    result = result,
    authoritative = true
  )

  private fun callable(
    id: String,
    name: String,
    resultType: String,
    resultInfo: CppTypeInfo,
    parameters: List<CppParameter>
  ) = CppReference(
    id = id,
    name = name,
    qualifiedName = name,
    kind = "function",
    returnType = resultType,
    canonicalReturnType = resultType,
    parameters = parameters,
    isCallable = true,
    returnTypeInfo = resultInfo
  )

  private fun parameter(type: String, info: CppTypeInfo) = CppParameter(
    type = type, canonicalType = type, typeInfo = info
  )

  private fun opaque(
    type: String,
    info: CppTypeInfo,
    valueCategory: String = "lvalue"
  ) = CppExpressionProfile(
    kind = "opaque",
    spelling = null,
    objectKind = "ordinary",
    type = type,
    canonicalType = type,
    typeInfo = info,
    valueCategory = valueCategory
  )

  private fun context(
    values: List<CppReference>,
    types: List<CppReference>,
    functions: List<CppReference>,
    witnesses: List<CppBinaryOperatorWitness>
  ): CppCompletionContext {
    val identifiers = (values + types + functions).flatMap { reference ->
      Regex("[A-Za-z_][A-Za-z_0-9]*").findAll(reference.name).map { it.value }.toList()
    }.toSet()
    return CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      completionKind = "expression",
      values = values,
      types = types,
      functions = functions,
      binaryOperatorWitnesses = witnesses
    )
  }

  private fun language(context: CppCompletionContext): CppSuffixGrammar =
    CppCompletionGrammar().generate(context, emptyList())

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)

  private fun info(id: String, kind: String = "record") = CppTypeInfo(
    id = id,
    canonicalId = id,
    valueCanonicalId = id,
    kind = kind,
    isSourceSpellable = true
  )

  private fun reference(info: CppTypeInfo, kind: String) = info.copy(
    id = "$kind:${info.valueCanonicalId}",
    canonicalId = "$kind:${info.valueCanonicalId}",
    kind = kind
  )

  private fun type(name: String, info: CppTypeInfo) = CppReference(
    name = name,
    type = name,
    canonicalType = name,
    kind = "type",
    isType = true,
    typeInfo = info
  )

  private fun value(name: String, type: String, info: CppTypeInfo) = CppReference(
    name = name,
    type = type,
    canonicalType = type,
    kind = "variable",
    isValue = true,
    typeInfo = info
  )
}
