package cppcompletion

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppSemanticCallWitnessGrammarTest {
  private val box = info("record:Box")
  private val result = info("record:Result")
  private val tokenA = info("record:Token:A")
  private val tokenB = info("record:Token:B")
  private val text = info("record:Text")
  private val count = info("record:Count")
  private val int = info("builtin:int", "builtin")
  private val double = info("builtin:double", "builtin")
  private val char = info("builtin:char", "builtin")
  private val bool = info("builtin:bool", "builtin")
  private val stringLiteral = info("array:const-char-1", "constantArray")
  private val nullptr = info("builtin:nullptr", "builtin")

  @Test
  fun authoritativeMemberWitnessesPreserveWholeVectorsAndOpaqueTypeIds() {
    val insert = memberCallable("insert", box, "Result &", result.reference("lvalueReference"))
    val context = context(
      values = listOf(
        value("box", "Box", box), value("key", "Token", tokenA),
        value("sameSpellingWrongId", "Token", tokenB), value("text", "Text", text),
        value("count", "Count", count)
      ),
      witnesses = listOf(
        memberWitness(insert, listOf(opaque("Token", tokenA), opaque("Text", text))),
        memberWitness(insert, listOf(opaque("Count", count), literal("booleanTrue", "bool", bool)))
      )
    )
    val language = language(context)

    assertTrue(language.recognizes("box.insert(key, text);"))
    assertTrue(language.recognizes("box.insert(count, true);"))
    assertFalse(language.recognizes("box.insert(key, true);"), "hybrid first vector")
    assertFalse(language.recognizes("box.insert(count, text);"), "hybrid second vector")
    assertFalse(
      language.recognizes("box.insert(sameSpellingWrongId, text);"),
      "display spelling must not replace the exact opaque type ID"
    )
  }

  @Test
  fun receiverAndArgumentProfilesUseExactCvAndValueCategories() {
    val consume = memberCallable("consume", box, "Result &", result.reference("lvalueReference"))
    val context = context(
      values = listOf(
        value("box", "Box", box),
        value("constBox", "Box", box.copy(isConst = true)),
        value("token", "Token", tokenA)
      ),
      functions = listOf(
        function("makeBox", "Box", box),
        function("makeToken", "Token", tokenA),
        function("moveToken", "Token &&", tokenA.reference("rvalueReference"))
      ),
      witnesses = listOf(memberWitness(
        consume,
        listOf(opaque("Token", tokenA, valueCategory = "xvalue"))
      ))
    )
    val language = language(context)

    assertTrue(language.recognizes("box.consume(moveToken());"))
    assertFalse(language.recognizes("box.consume(makeToken());"), "prvalue is not an xvalue")
    assertFalse(language.recognizes("box.consume(token);"), "lvalue is not an xvalue")
    assertFalse(language.recognizes("makeBox().consume(moveToken());"), "receiver must be an lvalue")
    assertFalse(language.recognizes("constBox.consume(moveToken());"), "receiver cv must match")
  }

  @Test
  fun constructionWitnessesRetainSyntaxAndEveryProjectedLiteralKind() {
    val parcel = info("record:Parcel")
    val constructor = constructorCallable("Parcel", parcel)
    val literalVector = listOf(
      literal("integerZero", "int", int),
      literal("floatingZero", "double", double),
      literal("characterZero", "char", char),
      literal("emptyString", "const char[1]", stringLiteral, valueCategory = "lvalue"),
      literal("booleanTrue", "bool", bool),
      literal("nullptr", "std::nullptr_t", nullptr)
    )
    val context = context(
      types = listOf(typeDeclaration("Parcel", parcel)),
      values = listOf(value("text", "Text", text)),
      witnesses = listOf(
        constructionWitness(constructor, parcel, "parenConstruction", literalVector),
        constructionWitness(
          constructor, parcel, "listConstruction", listOf(opaque("Text", text))
        )
      )
    )
    val language = language(context)

    assertTrue(language.recognizes("Parcel(0, 0.0, '\\0', \"\", true, nullptr);"))
    assertTrue(language.recognizes("Parcel{text};"))
    assertFalse(language.recognizes("Parcel{0, 0.0, '\\0', \"\", true, nullptr};"))
    assertFalse(language.recognizes("Parcel(text);"))
    assertFalse(language.recognizes("Parcel(7, 0.0, '\\0', \"\", true, nullptr);"))
    assertFalse(language.recognizes("Parcel(0, 2.5, '\\0', \"\", true, nullptr);"))
    assertFalse(language.recognizes("Parcel(0, 0.0, 'x', \"\", true, nullptr);"))
    assertFalse(language.recognizes("Parcel(0, 0.0, '\\0', \"value\", true, nullptr);"))
    assertFalse(language.recognizes("Parcel(0, 0.0, '\\0', \"\", false, nullptr);"))
  }

  @Test
  fun projectedLiteralKindsRejectContradictoryValueCategories() {
    val parcel = info("record:Parcel")
    val constructor = constructorCallable("Parcel", parcel)
    val context = context(types = listOf(typeDeclaration("Parcel", parcel)), witnesses = listOf(
      constructionWitness(
        constructor, parcel, "parenConstruction",
        listOf(literal("emptyString", "const char[1]", stringLiteral))
      ),
      constructionWitness(
        constructor, parcel, "listConstruction",
        listOf(literal("integerZero", "int", int, valueCategory = "lvalue"))
      )
    ))
    val language = language(context)

    assertFalse(language.recognizes("Parcel(\"value\");"), "a string literal is an lvalue")
    assertFalse(language.recognizes("Parcel{7};"), "an integer literal is a prvalue")
  }

  @Test
  fun opaqueSpellingsAndUnsupportedObjectKindsFailClosed() {
    val consume = memberCallable("consume", box, "Result &", result.reference("lvalueReference"))
    val language = language(context(
      values = listOf(value("box", "Box", box), value("token", "Token", tokenA)),
      witnesses = listOf(
        memberWitness(consume, listOf(opaque("Token", tokenA).copy(spelling = "token"))),
        memberWitness(consume, listOf(opaque("Token", tokenA).copy(objectKind = "bitField")))
      )
    ))

    assertFalse(language.recognizes("box.consume(token);"))
  }

  @Test
  fun nonAuthoritativeOrShallowValidationWitnessesPublishNoCalls() {
    val unchecked = memberCallable("unchecked", box, "Result &", result.reference("lvalueReference"))
    val shallow = memberCallable("shallow", box, "Result &", result.reference("lvalueReference"))
    val arguments = listOf(opaque("Token", tokenA))
    val context = context(
      values = listOf(value("box", "Box", box), value("token", "Token", tokenA)),
      witnesses = listOf(
        memberWitness(unchecked, arguments).copy(authoritative = false),
        memberWitness(shallow, arguments).copy(validation = "overloadResolution")
      )
    )
    val language = language(context)

    assertFalse(language.recognizes("box.unchecked(token);"))
    assertFalse(language.recognizes("box.shallow(token);"))
  }

  private fun memberWitness(
    callable: CppReference,
    arguments: List<CppExpressionProfile>
  ) = CppCallWitness(
    name = callable.name,
    syntax = "memberCall",
    validation = "recursiveDefinitionInstantiation",
    targetId = "template:${callable.name}",
    primaryTemplateId = "template:${callable.name}",
    receiver = opaque("Box", box),
    arguments = arguments,
    callable = callable.copy(primaryTemplateId = "template:${callable.name}"),
    result = opaque("Result", result),
    authoritative = true
  )

  private fun constructionWitness(
    callable: CppReference,
    target: CppTypeInfo,
    syntax: String,
    arguments: List<CppExpressionProfile>
  ) = CppCallWitness(
    name = callable.name,
    syntax = syntax,
    validation = "recursiveDefinitionInstantiation",
    targetId = "template:${callable.name}",
    primaryTemplateId = "template:${callable.name}",
    arguments = arguments,
    callable = callable.copy(primaryTemplateId = "template:${callable.name}"),
    result = opaque(callable.name, target, valueCategory = "prvalue"),
    authoritative = true
  )

  private fun opaque(
    type: String,
    info: CppTypeInfo,
    valueCategory: String = "lvalue"
  ) = CppExpressionProfile(
    kind = "opaque", spelling = null, objectKind = "ordinary",
    type = type, canonicalType = type,
    typeInfo = info, valueCategory = valueCategory
  )

  private fun literal(
    kind: String,
    type: String,
    info: CppTypeInfo,
    valueCategory: String = "prvalue"
  ) = CppExpressionProfile(
    kind = kind,
    spelling = when (kind) {
      "integerZero" -> "0"
      "floatingZero" -> "0.0"
      "characterZero" -> "'\\0'"
      "emptyString" -> "\"\""
      "booleanTrue" -> "true"
      "nullptr" -> "nullptr"
      else -> error("Unknown synthetic literal profile $kind")
    },
    objectKind = "ordinary",
    type = type, canonicalType = type,
    typeInfo = info, valueCategory = valueCategory
  )

  private fun memberCallable(
    name: String,
    owner: CppTypeInfo,
    returnType: String,
    returnInfo: CppTypeInfo
  ) = CppReference(
    name = name, qualifiedName = "Box::$name", kind = "method",
    returnType = returnType, canonicalReturnType = returnType,
    ownerType = "Box", canonicalOwnerType = "Box",
    isCallable = true, isMember = true,
    ownerTypeInfo = owner, returnTypeInfo = returnInfo
  )

  private fun constructorCallable(name: String, owner: CppTypeInfo) = CppReference(
    name = name, qualifiedName = name, kind = "constructor",
    returnType = name, canonicalReturnType = name,
    ownerType = name, canonicalOwnerType = name,
    isCallable = true, isMember = true,
    ownerTypeInfo = owner, returnTypeInfo = owner
  )

  private fun function(
    name: String,
    returnType: String,
    returnInfo: CppTypeInfo
  ) = CppReference(
    name = name, kind = "function", returnType = returnType,
    canonicalReturnType = returnType, isCallable = true,
    returnTypeInfo = returnInfo
  )

  private fun value(name: String, type: String, info: CppTypeInfo) = CppReference(
    name = name, type = type, canonicalType = type,
    kind = "variable", isValue = true, typeInfo = info
  )

  private fun context(
    types: List<CppReference> = emptyList(),
    values: List<CppReference> = emptyList(),
    functions: List<CppReference> = emptyList(),
    witnesses: List<CppCallWitness>
  ): CppCompletionContext {
    val identifiers = (types.map(CppReference::name) + values.map(CppReference::name) +
      functions.map(CppReference::name) +
      witnesses.map(CppCallWitness::name) + listOf("Box", "Result", "Parcel", "Text", "Token"))
      .flatMap { it.split("::") }.toSet()
    return CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      completionKind = "expression",
      types = types,
      values = values,
      functions = functions,
      callWitnesses = witnesses
    )
  }

  private fun info(id: String, kind: String = "record") = CppTypeInfo(
    id = id, canonicalId = id, valueCanonicalId = id,
    kind = kind, isSourceSpellable = true
  )

  private fun CppTypeInfo.reference(kind: String) = copy(
    id = "$kind:$valueCanonicalId", canonicalId = "$kind:$valueCanonicalId", kind = kind
  )

  private fun typeDeclaration(name: String, info: CppTypeInfo) = CppReference(
    name = name, type = name, canonicalType = name,
    kind = "class", isType = true, typeInfo = info, completionVisible = true
  )

  private fun language(context: CppCompletionContext): CppSuffixGrammar =
    CppCompletionGrammar().generate(context, emptyList())

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)
}
