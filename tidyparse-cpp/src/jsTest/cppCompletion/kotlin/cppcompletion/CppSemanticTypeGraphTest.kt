package cppcompletion

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Regression coverage for the structured type identities supplied by tidyparse/semanticCompletion. */
class CppSemanticTypeGraphTest {
  private val intType = typeInfo("builtin:int", "builtin", sourceSpellable = true)
  private val widgetType = typeInfo("record:Widget", "record", sourceSpellable = true)

  @Test
  fun constRvalueReferencesBindWithoutBecomingMutableRvalues() {
    val mutableRvalue = widgetType.reference("rvalueReference")
    val constRvalue = widgetType.reference("rvalueReference", isConst = true)
    val language = language(
      functions = listOf(
        function("makeWidget", "Widget", widgetType),
        function("makeConstWidget", "const Widget &&", constRvalue),
        function("takeWidget", "int", intType, parameter("Widget &&", mutableRvalue)),
        function("takeConstWidget", "int", intType, parameter("const Widget &&", constRvalue))
      )
    )

    assertTrue(language.recognizes("takeWidget(makeWidget());"))
    assertFalse(language.recognizes("takeWidget(makeConstWidget());"))
    assertTrue(language.recognizes("takeConstWidget(makeWidget());"))
    assertTrue(language.recognizes("takeConstWidget(makeConstWidget());"))
  }

  @Test
  fun cvAndRefQualifiedMethodsUseTheExactReceiverCategory() {
    fun value(name: String, isConst: Boolean = false, isVolatile: Boolean = false) = CppReference(
      name, type = "Widget", isValue = true,
      typeInfo = widgetType.copy(isConst = isConst, isVolatile = isVolatile)
    )
    fun method(
      name: String,
      isConst: Boolean = false,
      isVolatile: Boolean = false,
      ref: String
    ) = CppReference(
      name, returnType = "int", kind = "method", ownerType = "Widget",
      isCallable = true, isMember = true, isConstMethod = isConst,
      isVolatileMethod = isVolatile, refQualifier = ref,
      ownerTypeInfo = widgetType, returnTypeInfo = intType
    )
    val members = listOf(
      method("plainL", ref = "&"),
      method("constL", isConst = true, ref = "&"),
      method("volatileL", isVolatile = true, ref = "&"),
      method("cvL", isConst = true, isVolatile = true, ref = "&"),
      method("plainR", ref = "&&"),
      method("constR", isConst = true, ref = "&&")
    )
    val constRvalue = widgetType.reference("rvalueReference", isConst = true)
    val language = language(
      values = listOf(
        value("widget"), value("constWidget", isConst = true),
        value("volatileWidget", isVolatile = true),
        value("cvWidget", isConst = true, isVolatile = true)
      ),
      functions = listOf(
        function("makeWidget", "Widget", widgetType),
        function("makeConstWidget", "const Widget &&", constRvalue)
      ),
      members = members
    )

    assertTrue(language.recognizes("widget.plainL();"))
    assertFalse(language.recognizes("constWidget.plainL();"))
    assertFalse(language.recognizes("volatileWidget.plainL();"))
    assertTrue(language.recognizes("widget.constL();"))
    assertTrue(language.recognizes("constWidget.constL();"))
    assertFalse(language.recognizes("volatileWidget.constL();"))
    assertTrue(language.recognizes("widget.volatileL();"))
    assertTrue(language.recognizes("volatileWidget.volatileL();"))
    assertFalse(language.recognizes("constWidget.volatileL();"))
    assertTrue(language.recognizes("cvWidget.cvL();"))
    assertFalse(language.recognizes("widget.plainR();"))
    assertTrue(language.recognizes("makeWidget().plainR();"))
    assertFalse(language.recognizes("makeConstWidget().plainR();"))
    assertTrue(language.recognizes("makeWidget().constR();"))
    assertTrue(language.recognizes("makeConstWidget().constR();"))
    assertFalse(language.recognizes("makeWidget().plainL();"))
  }

  @Test
  fun topLevelConstPointerAndPointerToConstRemainDifferentFacts() {
    val mutablePointer = pointerInfo("pointer:Widget", pointeeConst = false)
    val constPointer = mutablePointer.copy(isConst = true)
    val pointerToConst = pointerInfo("pointer:const-Widget", pointeeConst = true)
    val mutate = CppReference(
      "mutate", returnType = "int", kind = "method", ownerType = "Widget",
      isCallable = true, isMember = true, isConstMethod = false,
      ownerTypeInfo = widgetType, returnTypeInfo = intType
    )
    val inspect = mutate.copy(name = "inspect", isConstMethod = true)
    val language = language(
      values = listOf(
        CppReference("fixed", type = "Widget *const", isValue = true, typeInfo = constPointer),
        CppReference("cursor", type = "Widget *", isValue = true, typeInfo = mutablePointer),
        CppReference("view", type = "const Widget *", isValue = true, typeInfo = pointerToConst)
      ),
      members = listOf(mutate, inspect)
    )

    assertFalse(language.recognizes("fixed = cursor;"))
    assertTrue(language.recognizes("cursor = fixed;"))
    assertTrue(language.recognizes("fixed->mutate();"))
    assertFalse(language.recognizes("view->mutate();"))
    assertTrue(language.recognizes("view->inspect();"))
  }

  @Test
  fun inheritanceAndObjectPointerConversionsPreserveTheirSemaAndCvBoundaries() {
    val baseType = typeInfo("record:Base", "record", sourceSpellable = true)
    val derivedType = typeInfo("record:Derived", "record", sourceSpellable = true)
    val adaptedType = typeInfo("record:Adapted", "record", sourceSpellable = true)
    val voidType = typeInfo("builtin:void", "builtin", sourceSpellable = true)
    val functionType = typeInfo("function:Callback", "function", sourceSpellable = true)
    fun pointer(
      id: String,
      pointee: CppTypeInfo,
      pointeeConst: Boolean = false
    ) = CppTypeInfo(
      id = id,
      canonicalId = id,
      valueCanonicalId = id,
      kind = "pointer",
      pointeeCanonicalId = pointee.valueCanonicalId,
      pointeeIsConst = pointeeConst,
      isSourceSpellable = true
    )
    val basePointer = pointer("pointer:Base", baseType)
    val constBasePointer = pointer("pointer:const-Base", baseType, pointeeConst = true)
    val derivedPointer = pointer("pointer:Derived", derivedType)
    val constDerivedPointer = pointer("pointer:const-Derived", derivedType, pointeeConst = true)
    val voidPointer = pointer("pointer:void", voidType)
    val constVoidPointer = pointer("pointer:const-void", voidType, pointeeConst = true)
    val functionPointer = pointer("pointer:Callback", functionType)
    fun declaration(name: String, info: CppTypeInfo, kind: String = "class") = CppReference(
      name = name, type = name, canonicalType = name, kind = kind,
      isType = true, typeInfo = info
    )
    val context = context(
      values = listOf(
        CppReference("derived", type = "Derived", isValue = true, typeInfo = derivedType),
        CppReference(
          "readonlyDerivedObject", type = "const Derived", isValue = true,
          typeInfo = derivedType.copy(isConst = true)
        ),
        CppReference("adapted", type = "Adapted", isValue = true, typeInfo = adaptedType),
        CppReference("basePointer", type = "Base *", isValue = true, typeInfo = basePointer),
        CppReference(
          "derivedPointer", type = "Derived *", isValue = true, typeInfo = derivedPointer
        ),
        CppReference(
          "readonlyDerived", type = "const Derived *", isValue = true,
          typeInfo = constDerivedPointer
        ),
        CppReference(
          "callback", type = "Callback *", isValue = true, typeInfo = functionPointer
        )
      ),
      types = listOf(
        declaration("Base", baseType), declaration("Derived", derivedType),
        declaration("Adapted", adaptedType), declaration("void", voidType, kind = "builtin"),
        declaration("Callback", functionType, kind = "typeAlias"),
        declaration("Base *", basePointer, kind = "pointerType"),
        declaration("const Base *", constBasePointer, kind = "pointerType"),
        declaration("Derived *", derivedPointer, kind = "pointerType"),
        declaration("const Derived *", constDerivedPointer, kind = "pointerType"),
        declaration("void *", voidPointer, kind = "pointerType"),
        declaration("const void *", constVoidPointer, kind = "pointerType"),
        declaration("Callback *", functionPointer, kind = "pointerType")
      ),
      extraIdentifiers = setOf("result"),
      conversions = listOf(
        CppConversion("Derived", "Base", kind = "base"),
        CppConversion("Adapted", "Base", kind = "conversion")
      )
    ).copy(requiredIdentifier = "result")
    val language = CppCompletionGrammar().generate(context, emptyList())
    val expressionLanguage = CppCompletionGrammar().generate(
      context.copy(requiredIdentifier = null), emptyList()
    )

    assertTrue(language.recognizes("Base * result = &derived;"), "derived address upcast")
    assertTrue(
      language.recognizes("const Base * result = derivedPointer;"),
      "derived pointer upcast with added const"
    )
    assertFalse(language.recognizes("Derived * result = basePointer;"))
    assertFalse(language.recognizes("Base * result = readonlyDerived;"))
    assertFalse(language.recognizes("Base * result = &readonlyDerivedObject;"))
    assertTrue(language.recognizes("const Base * result = &readonlyDerivedObject;"))
    assertFalse(language.recognizes("Base * result = &adapted;"))

    assertTrue(language.recognizes("void * result = derivedPointer;"), "object pointer to void")
    assertTrue(
      language.recognizes("const void * result = readonlyDerived;"),
      "const object pointer to const void"
    )
    assertFalse(language.recognizes("void * result = readonlyDerived;"))
    assertFalse(language.recognizes("void * result = callback;"))
    assertTrue(
      expressionLanguage.recognizes("static_cast<Base *>(derivedPointer);"),
      "static derived upcast"
    )
    assertFalse(expressionLanguage.recognizes("static_cast<Derived *>(basePointer);"))
    assertTrue(
      expressionLanguage.recognizes("static_cast<const void *>(readonlyDerived);"),
      "static void erasure"
    )
    assertFalse(expressionLanguage.recognizes("static_cast<void *>(readonlyDerived);"))
  }

  @Test
  fun fieldsPreserveBaseCvAndValueCategory() {
    val intLvalue = intType.reference("lvalueReference")
    val intRvalue = intType.reference("rvalueReference")
    val constIntRvalue = intType.reference("rvalueReference", isConst = true)
    fun field(name: String, mutable: Boolean = false) = CppReference(
      name, type = "int", kind = "field", ownerType = "Widget",
      isValue = true, isMember = true, isMutableField = mutable,
      ownerTypeInfo = widgetType, typeInfo = intType
    )
    val constRvalue = widgetType.reference("rvalueReference", isConst = true)
    val language = language(
      values = listOf(CppReference(
        "constWidget", type = "Widget", isValue = true,
        typeInfo = widgetType.copy(isConst = true)
      )),
      functions = listOf(
        function("makeWidget", "Widget", widgetType),
        function("makeConstWidget", "const Widget &&", constRvalue),
        function("takeLvalue", "int", intType, parameter("int &", intLvalue)),
        function("takeRvalue", "int", intType, parameter("int &&", intRvalue)),
        function("takeConstRvalue", "int", intType, parameter("const int &&", constIntRvalue))
      ),
      members = listOf(field("ordinary"), field("scratch", mutable = true))
    )

    assertFalse(language.recognizes("takeLvalue(constWidget.ordinary);"))
    assertTrue(language.recognizes("takeLvalue(constWidget.scratch);"))
    assertFalse(language.recognizes("takeLvalue(makeWidget().ordinary);"))
    assertTrue(language.recognizes("takeRvalue(makeWidget().ordinary);"))
    assertFalse(language.recognizes("takeRvalue(makeConstWidget().ordinary);"))
    assertTrue(language.recognizes("takeConstRvalue(makeConstWidget().ordinary);"))
    assertTrue(language.recognizes("takeRvalue(makeConstWidget().scratch);"))
  }

  @Test
  fun factoredExpressionTiersPreserveDepthCategoryAndCvBoundaries() {
    val lvalue = widgetType.reference("lvalueReference")
    val rvalue = widgetType.reference("rvalueReference")
    val language = language(
      values = listOf(CppReference("widget", type = "Widget", isValue = true, typeInfo = widgetType)),
      functions = listOf(
        function("makeWidget", "Widget", widgetType),
        function("identity", "Widget", widgetType, parameter("Widget", widgetType)),
        function("takeLvalue", "int", intType, parameter("Widget &", lvalue)),
        function("takeRvalue", "int", intType, parameter("Widget &&", rvalue))
      )
    )

    assertTrue(language.recognizes("widget;"))
    assertTrue(language.recognizes("identity(identity(widget));"))
    assertTrue(language.recognizes("takeLvalue(widget);"))
    assertTrue(language.recognizes("takeRvalue(makeWidget());"))
    assertFalse(language.recognizes("takeLvalue(makeWidget());"))
    assertFalse(language.recognizes("takeRvalue(widget);"))

    val nonterminals = language.sourceSyntax.mapTo(linkedSetOf()) { it.first }
    val terminals = language.sourceSyntax.flatMap { it.second }.filterNot { it in nonterminals }
    assertFalse(
      terminals.any {
        it.startsWith("TYPE_") || it.startsWith("BOOLEAN_CONDITION_D") ||
          it.startsWith("REFERENCE_CHOICE_") || it.startsWith("RECEIVER_CHOICE_") ||
          it.startsWith("OBJECT_CHOICE_")
      },
      "Dead generated expression leaves must be pruned as nonterminals, never published as tokens"
    )
  }

  @Test
  fun aliasesWithOneValueCanonicalIdShareOneSemanticNode() {
    val record = typeInfo("record:Canonical", "record", sourceSpellable = true)
    val language = language(
      extraIdentifiers = setOf("Canonical", "Alias"),
      types = listOf(
        CppReference(
          "Canonical", type = "Canonical", kind = "class", isType = true,
          typeInfo = record
        ),
        CppReference(
          "Alias", type = "Alias", canonicalType = "Canonical", kind = "typeAlias",
          isType = true, typeInfo = record.copy(id = "alias:Alias")
        )
      ),
      values = listOf(CppReference("aliased", type = "Alias", isValue = true, typeInfo = record)),
      functions = listOf(
        function("makeAlias", "Alias", record),
        function("consume", "int", intType, parameter("Canonical", record))
      )
    )

    assertTrue(language.recognizes("consume(aliased);"))
    assertTrue(language.recognizes("consume(makeAlias());"))
  }

  @Test
  fun collidingPrintedSpellingsDoNotUnifyDistinctOpaqueIds() {
    val leftType = typeInfo("record:left-token", "record", sourceSpellable = true)
    val rightType = typeInfo("record:right-token", "record", sourceSpellable = true)
    val language = language(
      extraIdentifiers = setOf("Token"),
      types = listOf(
        CppReference("Token", type = "Token", kind = "class", isType = true, typeInfo = leftType),
        CppReference("Token", type = "Token", kind = "class", isType = true, typeInfo = rightType)
      ),
      values = listOf(
        CppReference("left", type = "Token", isValue = true, typeInfo = leftType),
        CppReference("right", type = "Token", isValue = true, typeInfo = rightType)
      ),
      functions = listOf(function(
        "consumeLeft", "int", intType,
        parameter("Token", leftType)
      ))
    )

    assertTrue(language.recognizes("consumeLeft(left);"))
    assertFalse(language.recognizes("consumeLeft(right);"))
  }

  @Test
  fun unspellableClosureIsCallableButNeverEmittedAsTypeSyntax() {
    val closureSpelling = "(lambda at source.cc:8:3)"
    val closureType = typeInfo(
      "record:anonymous-closure", "record", sourceSpellable = false
    )
    val callOperator = CppReference(
      "operator()", returnType = "int", kind = "method", ownerType = closureSpelling,
      isCallable = true, isMember = true, isConstMethod = true,
      ownerTypeInfo = closureType, returnTypeInfo = intType
    )
    val context = context(
      values = listOf(CppReference(
        "closure", type = closureSpelling, isValue = true, typeInfo = closureType
      )),
      members = listOf(callOperator)
    )
    val language = CppCompletionGrammar().generate(context, emptyList())
    val terminals = language.sourceSyntax.flatMapTo(linkedSetOf()) { it.second }

    assertTrue(language.recognizes(cppLines("closure();").single().tokens))
    assertFalse(terminals.any { "lambda" in it || "source.cc" in it })
    assertFalse(language.recognizes(cppLines("$closureSpelling object;").single().tokens))
  }

  @Test
  fun scopedEnumConstantsRequireTheirExactQualifiedSemaSpelling() {
    val permissionType = typeInfo("enum:Permission", "enum", sourceSpellable = true)
    fun enumerator(name: String) = CppReference(
      name = name,
      type = "Permission",
      kind = "enumMember",
      detail = "scoped",
      ownerType = "Permission",
      typeInfo = permissionType,
      ownerTypeInfo = permissionType
    )
    val language = language(
      values = listOf(CppReference(
        "currentPermission", type = "Permission", isValue = true, typeInfo = permissionType
      )),
      types = listOf(CppReference(
        "Permission", type = "Permission", kind = "enum", isType = true,
        typeInfo = permissionType
      )),
      members = listOf(enumerator("Permission::allow"), enumerator("deny"))
    )

    assertTrue(language.recognizes("Permission::allow;"))
    assertFalse(language.recognizes("allow;"))
    assertFalse(language.recognizes("deny;"))
    assertFalse(language.recognizes("currentPermission.allow;"))
    assertFalse(language.recognizes("Permission::allow = Permission::allow;"))
  }

  @Test
  fun builtinCompoundAssignmentsRespectModifiableLvaluesAndOperandCategories() {
    val doubleType = typeInfo("builtin:double", "builtin", sourceSpellable = true)
    val convertedType = typeInfo("record:Converted", "record", sourceSpellable = true)
    val intPointer = CppTypeInfo(
      id = "pointer:int",
      canonicalId = "pointer:int",
      valueCanonicalId = "pointer:int",
      kind = "pointer",
      pointeeCanonicalId = intType.valueCanonicalId,
      isSourceSpellable = true
    )
    val language = language(
      values = listOf(
        CppReference("count", type = "int", isValue = true, typeInfo = intType),
        CppReference(
          "fixed", type = "const int", isValue = true,
          typeInfo = intType.copy(isConst = true)
        ),
        CppReference("ratio", type = "double", isValue = true, typeInfo = doubleType),
        CppReference("converted", type = "Converted", isValue = true, typeInfo = convertedType),
        CppReference("cursor", type = "int *", isValue = true, typeInfo = intPointer)
      ),
      types = listOf(CppReference(
        "Converted", type = "Converted", kind = "class", isType = true,
        typeInfo = convertedType
      )),
      conversions = listOf(CppConversion("Converted", "int"))
    )

    assertTrue(language.recognizes("count += ratio;"))
    assertTrue(language.recognizes("count %= converted;"))
    assertTrue(language.recognizes("count <<= converted;"))
    assertFalse(language.recognizes("ratio %= count;"))
    assertFalse(language.recognizes("fixed += count;"))
    assertTrue(language.recognizes("cursor += count;"))
    assertFalse(language.recognizes("cursor += ratio;"))
    assertFalse(language.recognizes("cursor *= count;"))
  }

  @Test
  fun declarationOnlyCompoundAssignmentsFailClosedWithoutAnAuthoritativeWitness() {
    val accumulatorType = typeInfo("record:Accumulator", "record", sourceSpellable = true)
    val deltaType = typeInfo("record:Delta", "record", sourceSpellable = true)
    val adaptedType = typeInfo("record:Adapted", "record", sourceSpellable = true)
    val plusAssign = CppReference(
      name = "operator+=",
      returnType = "Accumulator &",
      parameters = listOf(parameter("Delta", deltaType)),
      kind = "method",
      ownerType = "Accumulator",
      isCallable = true,
      isMember = true,
      isConstMethod = false,
      refQualifier = "&",
      canonicalOwnerType = "Accumulator",
      ownerTypeInfo = accumulatorType,
      returnTypeInfo = accumulatorType.reference("lvalueReference")
    )
    val language = language(
      values = listOf(
        CppReference(
          "accumulator", type = "Accumulator", isValue = true, typeInfo = accumulatorType
        ),
        CppReference(
          "fixedAccumulator", type = "const Accumulator", isValue = true,
          typeInfo = accumulatorType.copy(isConst = true)
        ),
        CppReference("delta", type = "Delta", isValue = true, typeInfo = deltaType),
        CppReference("adapted", type = "Adapted", isValue = true, typeInfo = adaptedType)
      ),
      types = listOf(
        CppReference(
          "Accumulator", type = "Accumulator", kind = "class", isType = true,
          typeInfo = accumulatorType
        ),
        CppReference("Delta", type = "Delta", kind = "class", isType = true, typeInfo = deltaType),
        CppReference(
          "Adapted", type = "Adapted", kind = "class", isType = true,
          typeInfo = adaptedType
        )
      ),
      members = listOf(plusAssign),
      conversions = listOf(CppConversion("Adapted", "Delta"))
    )

    assertFalse(language.recognizes("accumulator += delta;"))
    assertFalse(language.recognizes("accumulator += adapted;"))
    assertFalse(language.recognizes("fixedAccumulator += delta;"))
    assertFalse(language.recognizes("delta += accumulator;"))
  }

  @Test
  fun structuredDeclarationsRequireAnAffirmativeDefaultConstructibilityFact() {
    val ready = typeInfo("record:Ready", "record", sourceSpellable = true).copy(
      isComplete = true,
      isDefaultConstructible = true
    )
    val disabled = typeInfo("record:Disabled", "record", sourceSpellable = true).copy(
      isComplete = true,
      isDefaultConstructible = false
    )
    val unknown = typeInfo("record:Unknown", "record", sourceSpellable = true).copy(
      isComplete = true
    )
    fun declaration(name: String, info: CppTypeInfo) = CppReference(
      name = name,
      type = name,
      canonicalType = name,
      kind = "class",
      isType = true,
      typeInfo = info,
      completionVisible = true
    )
    val context = context(
      types = listOf(
        declaration("Ready", ready),
        declaration("Disabled", disabled),
        declaration("Unknown", unknown)
      ),
      extraIdentifiers = setOf("fresh")
    ).copy(requiredIdentifier = "fresh")
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes("Ready fresh;"))
    assertTrue(language.recognizes("Ready fresh{};"))
    assertFalse(language.recognizes("Disabled fresh;"))
    assertFalse(language.recognizes("Unknown fresh;"))
  }

  @Test
  fun bareOperationGraphStaticMembersStillRequireAnOwner() {
    fun staticMember(name: String, callable: Boolean) = CppReference(
      name = name,
      type = "int".takeUnless { callable },
      returnType = "int".takeIf { callable },
      kind = if (callable) "method" else "field",
      ownerType = "Widget",
      isValue = !callable,
      isCallable = callable,
      isMember = true,
      isStatic = true,
      ownerTypeInfo = widgetType,
      typeInfo = intType.takeUnless { callable },
      returnTypeInfo = intType.takeIf { callable }
    )
    val language = language(
      values = listOf(CppReference(
        "widget", type = "Widget", isValue = true, typeInfo = widgetType
      )),
      members = listOf(staticMember("limit", false), staticMember("create", true))
    )

    assertFalse(language.recognizes("limit;"))
    assertFalse(language.recognizes("create();"))
    assertTrue(language.recognizes("widget.limit;"))
    assertTrue(language.recognizes("widget.create();"))
  }

  @Test
  fun referenceReturnTypesPreserveTheirRequiredValueCategory() {
    val values = listOf(CppReference(
      "widget", type = "Widget", isValue = true, typeInfo = widgetType
    ))
    val functions = listOf(function("makeWidget", "Widget", widgetType))
    fun returnLanguage(info: CppTypeInfo) = CppCompletionGrammar().generate(
      context(values = values, functions = functions).copy(
        enclosingReturnType = if (info.isConst) "const Widget &" else "Widget &",
        canonicalEnclosingReturnType = if (info.isConst) "const Widget &" else "Widget &",
        enclosingReturnTypeInfo = info
      ),
      emptyList()
    )

    val mutable = returnLanguage(widgetType.reference("lvalueReference"))
    assertTrue(mutable.recognizes("return widget;"))
    assertFalse(mutable.recognizes("return makeWidget();"))

    val readOnly = returnLanguage(widgetType.reference("lvalueReference", isConst = true))
    assertTrue(readOnly.recognizes("return widget;"))
    assertTrue(readOnly.recognizes("return makeWidget();"))
  }

  @Test
  fun structuredReferenceDeclarationsUseDirectBindingRatherThanValueConversions() {
    val base = typeInfo("record:Base", "record", sourceSpellable = true)
    val derived = typeInfo("record:Derived", "record", sourceSpellable = true)
    val adapted = typeInfo("record:Adapted", "record", sourceSpellable = true)
    val double = typeInfo("builtin:double", "builtin", sourceSpellable = true)
    fun value(name: String, type: String, info: CppTypeInfo) = CppReference(
      name, type = type, canonicalType = type, kind = "variable",
      isValue = true, typeInfo = info
    )
    val context = context(
      values = listOf(
        value("base", "Base", base),
        value("constBase", "const Base", base.copy(isConst = true)),
        value("volatileBase", "volatile Base", base.copy(isVolatile = true)),
        value(
          "cvBase", "const volatile Base",
          base.copy(isConst = true, isVolatile = true)
        ),
        value("derived", "Derived", derived),
        value("constDerived", "const Derived", derived.copy(isConst = true)),
        value("adapted", "Adapted", adapted),
        value("integer", "int", intType),
        value("floating", "double", double)
      ),
      functions = listOf(
        function("makeBase", "Base", base),
        function("makeDerived", "Derived", derived)
      ),
      extraIdentifiers = setOf("ref"),
      conversions = listOf(
        CppConversion("Derived", "Base", kind = "base"),
        CppConversion("Adapted", "Base", kind = "conversion")
      )
    ).copy(requiredIdentifier = "ref")
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes("Base & ref = base;"))
    assertTrue(language.recognizes("Base & ref = derived;"), "public direct base binding")
    assertFalse(language.recognizes("Base & ref = constBase;"), "mutable from const")
    assertFalse(language.recognizes("Base & ref = volatileBase;"), "mutable from volatile")
    assertFalse(language.recognizes("Base & ref = makeBase();"), "prvalue temporary")
    assertFalse(language.recognizes("Base & ref = makeDerived();"), "derived prvalue")
    assertFalse(language.recognizes("double & ref = integer;"), "arithmetic temporary")
    assertFalse(language.recognizes("Base & ref = adapted;"), "user-conversion temporary")

    assertTrue(language.recognizes("const Base & ref = base;"))
    assertTrue(language.recognizes("const Base & ref = constBase;"))
    assertTrue(language.recognizes("const Base & ref = constDerived;"))
    assertTrue(language.recognizes("const Base & ref = makeBase();"))
    assertTrue(language.recognizes("const Base & ref = makeDerived();"))
    assertFalse(language.recognizes("const Base & ref = volatileBase;"), "const from volatile")
    assertFalse(language.recognizes("const Base & ref = adapted;"), "const user conversion")
    assertTrue(language.recognizes("const double & ref = integer;"), "builtin temporary")

    assertTrue(language.recognizes("volatile Base & ref = base;"))
    assertTrue(language.recognizes("volatile Base & ref = volatileBase;"))
    assertFalse(language.recognizes("volatile Base & ref = constBase;"), "volatile from const")
    assertTrue(language.recognizes("const volatile Base & ref = cvBase;"))
    assertTrue(language.recognizes("const volatile Base & ref = volatileBase;"))
    assertFalse(
      language.recognizes("const volatile Base & ref = makeBase();"),
      "const volatile from prvalue"
    )
  }

  @Test
  fun structuredPointerReferencesKeepPointeeAndTopLevelCvSeparate() {
    val base = typeInfo("record:PointerBase", "record", sourceSpellable = true)
    val derived = typeInfo("record:PointerDerived", "record", sourceSpellable = true)
    fun pointer(id: String, pointee: CppTypeInfo, pointeeConst: Boolean = false) = CppTypeInfo(
      id = id, canonicalId = id, valueCanonicalId = id, kind = "pointer",
      pointeeCanonicalId = pointee.valueCanonicalId, pointeeIsConst = pointeeConst,
      isSourceSpellable = true
    )
    val basePointer = pointer("pointer:PointerBase", base)
    val constBasePointer = pointer("pointer:const-PointerBase", base, pointeeConst = true)
    val derivedPointer = pointer("pointer:PointerDerived", derived)
    fun value(name: String, type: String, info: CppTypeInfo) = CppReference(
      name, type = type, canonicalType = type, kind = "variable",
      isValue = true, typeInfo = info
    )
    fun type(name: String, info: CppTypeInfo) = CppReference(
      name, type = name, canonicalType = name, kind = "type",
      isType = true, typeInfo = info
    )
    val context = context(
      values = listOf(
        value("basePointer", "PointerBase *", basePointer),
        value("derivedPointer", "PointerDerived *", derivedPointer),
        value("readonlyBasePointer", "const PointerBase *", constBasePointer)
      ),
      types = listOf(type("PointerBase", base), type("PointerDerived", derived)),
      extraIdentifiers = setOf("ref"),
      conversions = listOf(CppConversion("PointerDerived", "PointerBase", kind = "base"))
    ).copy(requiredIdentifier = "ref")
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes("PointerBase * & ref = basePointer;"))
    assertFalse(language.recognizes("PointerBase * & ref = derivedPointer;"))
    assertTrue(language.recognizes("PointerBase * const & ref = derivedPointer;"))
    assertTrue(language.recognizes("const PointerBase * & ref = readonlyBasePointer;"))
    assertFalse(language.recognizes("const PointerBase * & ref = basePointer;"))
    assertTrue(language.recognizes("const PointerBase * const & ref = basePointer;"))
    assertTrue(language.recognizes("const PointerBase * const ref = readonlyBasePointer;"))
    assertFalse(language.recognizes("const const PointerBase * & ref = readonlyBasePointer;"))
    assertFalse(language.recognizes("const const PointerBase * ref = readonlyBasePointer;"))
  }

  @Test
  fun concreteCallAndConstructorReferenceParametersUseTheSameDirectBindingRelation() {
    val base = typeInfo("record:ArgumentBase", "record", sourceSpellable = true)
    val derived = typeInfo("record:ArgumentDerived", "record", sourceSpellable = true)
    val adapted = typeInfo("record:ArgumentAdapted", "record", sourceSpellable = true)
    val holder = typeInfo("record:Holder", "record", sourceSpellable = true)
    val double = typeInfo("builtin:double", "builtin", sourceSpellable = true)
    val baseRef = base.reference("lvalueReference")
    val constBaseRef = base.reference("lvalueReference", isConst = true)
    val doubleRef = double.reference("lvalueReference")
    val constDoubleRef = double.reference("lvalueReference", isConst = true)
    fun value(name: String, type: String, info: CppTypeInfo) = CppReference(
      name, type = type, canonicalType = type, kind = "variable",
      isValue = true, typeInfo = info
    )
    val constructor = CppReference(
      name = "Holder", kind = "constructor", ownerType = "Holder",
      canonicalOwnerType = "Holder", returnType = "Holder", canonicalReturnType = "Holder",
      parameters = listOf(parameter("ArgumentBase &", baseRef)),
      isCallable = true, isMember = true, ownerTypeInfo = holder, returnTypeInfo = holder
    )
    val context = context(
      values = listOf(
        value("base", "ArgumentBase", base),
        value("constBase", "const ArgumentBase", base.copy(isConst = true)),
        value("volatileBase", "volatile ArgumentBase", base.copy(isVolatile = true)),
        value("derived", "ArgumentDerived", derived),
        value("adapted", "ArgumentAdapted", adapted),
        value("integer", "int", intType),
        value("floating", "double", double)
      ),
      functions = listOf(
        function("takeBase", "int", intType, parameter("ArgumentBase &", baseRef)),
        function(
          "inspectBase", "int", intType,
          parameter("const ArgumentBase &", constBaseRef)
        ),
        function("takeDouble", "int", intType, parameter("double &", doubleRef)),
        function(
          "inspectDouble", "int", intType,
          parameter("const double &", constDoubleRef)
        ),
        function("makeBase", "ArgumentBase", base),
        constructor
      ),
      conversions = listOf(
        CppConversion("ArgumentDerived", "ArgumentBase", kind = "base"),
        CppConversion("ArgumentAdapted", "ArgumentBase", kind = "constructor")
      )
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes("takeBase(base);"))
    assertTrue(language.recognizes("takeBase(derived);"))
    assertFalse(language.recognizes("takeBase(constBase);"))
    assertFalse(language.recognizes("takeBase(adapted);"))
    assertFalse(language.recognizes("takeBase(makeBase());"))
    assertTrue(language.recognizes("takeDouble(floating);"))
    assertFalse(language.recognizes("takeDouble(integer);"))

    assertTrue(language.recognizes("inspectBase(constBase);"))
    assertTrue(language.recognizes("inspectBase(makeBase());"))
    assertFalse(language.recognizes("inspectBase(volatileBase);"))
    assertFalse(language.recognizes("inspectBase(adapted);"))
    assertTrue(language.recognizes("inspectDouble(integer);"))

    assertTrue(language.recognizes("Holder{base};"))
    assertTrue(language.recognizes("Holder{derived};"))
    assertFalse(language.recognizes("Holder{adapted};"))
    assertFalse(language.recognizes("Holder{makeBase()};"))
  }

  @Test
  fun repeatedTrailingCvDoesNotChangeReferenceBindingOrNullptrReturns() {
    val pointer = pointerInfo("pointer:cv-normalization-Widget", pointeeConst = false)
    val cvPointer = pointer.copy(isConst = true, isVolatile = true)
    val values = listOf(
      CppReference("widget", type = "Widget", isValue = true, typeInfo = widgetType),
      CppReference("pointer", type = "Widget *", isValue = true, typeInfo = pointer),
      CppReference(
        "cvPointer", type = "Widget * volatile const", isValue = true, typeInfo = cvPointer
      )
    )
    val functions = listOf(function("makePointer", "Widget *", pointer))
    val declarationLanguage = CppCompletionGrammar().generate(
      context(values = values, functions = functions, extraIdentifiers = setOf("ref"))
        .copy(requiredIdentifier = "ref"),
      emptyList()
    )

    assertTrue(declarationLanguage.recognizes("Widget * const volatile & ref = cvPointer;"))
    assertFalse(
      declarationLanguage.recognizes("Widget * volatile const & ref = makePointer();"),
      "a leaked volatile spelling must not enter the const-only temporary branch"
    )

    fun returnLanguage(raw: String, info: CppTypeInfo) = CppCompletionGrammar().generate(
      context(values = values, functions = functions).copy(
        enclosingReturnType = raw,
        canonicalEnclosingReturnType = raw,
        enclosingReturnTypeInfo = info
      ),
      emptyList()
    )
    assertFalse(
      returnLanguage("Widget * &", pointer.reference("lvalueReference"))
        .recognizes("return nullptr;")
    )
    assertTrue(
      returnLanguage("Widget * const &", pointer.reference("lvalueReference", isConst = true))
        .recognizes("return nullptr;")
    )
    assertTrue(
      returnLanguage("Widget * &&", pointer.reference("rvalueReference"))
        .recognizes("return nullptr;")
    )
  }

  @Test
  fun memberReceiversPreserveRoleReferenceCategoryAndTransitivePublicBases() {
    val base = typeInfo("record:ReceiverBase", "record", sourceSpellable = true)
    val middle = typeInfo("record:ReceiverMiddle", "record", sourceSpellable = true)
    val derived = typeInfo("record:ReceiverDerived", "record", sourceSpellable = true)
    val holder = typeInfo("record:ReferenceHolder", "record", sourceSpellable = true)
    val diamondBase = typeInfo("record:DiamondBase", "record", sourceSpellable = true)
    val diamondLeft = typeInfo("record:DiamondLeft", "record", sourceSpellable = true)
    val diamondRight = typeInfo("record:DiamondRight", "record", sourceSpellable = true)
    val diamondLeaf = typeInfo("record:DiamondLeaf", "record", sourceSpellable = true)
    fun pointer(id: String, pointee: CppTypeInfo, pointeeConst: Boolean = false) = CppTypeInfo(
      id = id, canonicalId = id, valueCanonicalId = id, kind = "pointer",
      pointeeCanonicalId = pointee.valueCanonicalId, pointeeIsConst = pointeeConst,
      isSourceSpellable = true
    )
    val constBasePointer = pointer("pointer:const-ReceiverBase", base, pointeeConst = true)
    val derivedPointer = pointer("pointer:ReceiverDerived", derived)
    fun field(
      name: String,
      owner: String,
      ownerInfo: CppTypeInfo,
      info: CppTypeInfo = intType,
      static: Boolean = false
    ) = CppReference(
      name, type = "int", canonicalType = "int", kind = "field", ownerType = owner,
      canonicalOwnerType = owner, isValue = true, isMember = true, isStatic = static,
      typeInfo = info, ownerTypeInfo = ownerInfo
    )
    fun method(
      name: String,
      owner: String = "ReceiverBase",
      ownerInfo: CppTypeInfo = base,
      static: Boolean = false
    ) = CppReference(
      name, returnType = "int", canonicalReturnType = "int", kind = "method",
      ownerType = owner, canonicalOwnerType = owner,
      isCallable = true, isMember = true, isStatic = static,
      ownerTypeInfo = ownerInfo, returnTypeInfo = intType
    )
    val referenceField = field(
      "ref", "ReferenceHolder", holder, intType.reference("lvalueReference")
    )
    // Keep the field before the nonconst method: the old receiver cache reused the field's
    // permissive pointer set for the callable solely because both had U/no-ref-qualifier keys.
    val members = listOf(
      field("value", "ReceiverBase", base), method("mutate"), method("reset", static = true),
      field("counter", "ReceiverBase", base, static = true), referenceField,
      method("ambiguous", "DiamondBase", diamondBase)
    )
    val language = language(
      values = listOf(
        CppReference(
          "baseView", type = "const ReceiverBase *", isValue = true,
          typeInfo = constBasePointer
        ),
        CppReference(
          "constBase", type = "const ReceiverBase", isValue = true,
          typeInfo = base.copy(isConst = true)
        ),
        CppReference("derived", type = "ReceiverDerived", isValue = true, typeInfo = derived),
        CppReference(
          "derivedPointer", type = "ReceiverDerived *", isValue = true,
          typeInfo = derivedPointer
        ),
        CppReference("diamond", type = "DiamondLeaf", isValue = true, typeInfo = diamondLeaf)
      ),
      functions = listOf(
        function("makeHolder", "ReferenceHolder", holder),
        function("takeLvalue", "int", intType, parameter("int &", intType.reference("lvalueReference"))),
        function("takeRvalue", "int", intType, parameter("int &&", intType.reference("rvalueReference")))
      ),
      members = members,
      conversions = listOf(
        CppConversion("ReceiverDerived", "ReceiverMiddle", kind = "base"),
        CppConversion("ReceiverMiddle", "ReceiverBase", kind = "base"),
        CppConversion("DiamondLeaf", "DiamondLeft", kind = "base"),
        CppConversion("DiamondLeaf", "DiamondRight", kind = "base"),
        CppConversion("DiamondLeaf", "DiamondBase", kind = "base"),
        CppConversion("DiamondLeft", "DiamondBase", kind = "base"),
        CppConversion("DiamondRight", "DiamondBase", kind = "base")
      )
    )

    assertTrue(language.recognizes("baseView->value;"))
    assertFalse(language.recognizes("baseView->mutate();"))
    assertTrue(language.recognizes("constBase.reset();"), "static method ignores receiver cv")
    assertTrue(
      language.recognizes("takeLvalue(constBase.counter);"),
      "static field ignores receiver cv"
    )
    assertTrue(language.recognizes("derived.mutate();"))
    assertTrue(language.recognizes("derivedPointer->mutate();"))
    assertFalse(language.recognizes("diamond.ambiguous();"), "two base paths are ambiguous")
    assertTrue(language.recognizes("takeLvalue(makeHolder().ref);"))
    assertFalse(language.recognizes("takeRvalue(makeHolder().ref);"))
  }

  @Test
  fun builtInConvertedTemporariesAndNullptrRespectReferenceKind() {
    val double = typeInfo("builtin:converted-double", "builtin", sourceSpellable = true)
    val pointer = pointerInfo("pointer:argument-Widget", pointeeConst = false)
    val holder = typeInfo("record:PointerHolder", "record", sourceSpellable = true)
    val constDoubleRef = double.reference("lvalueReference", isConst = true)
    val doubleRvalueRef = double.reference("rvalueReference")
    val pointerLvalueRef = pointer.reference("lvalueReference")
    val constPointerRef = pointer.reference("lvalueReference", isConst = true)
    val pointerRvalueRef = pointer.reference("rvalueReference")
    val constructor = CppReference(
      name = "PointerHolder", kind = "constructor", ownerType = "PointerHolder",
      canonicalOwnerType = "PointerHolder", returnType = "PointerHolder",
      canonicalReturnType = "PointerHolder", parameters = listOf(parameter("Widget *", pointer)),
      isCallable = true, isMember = true, ownerTypeInfo = holder, returnTypeInfo = holder
    )
    val values = listOf(
      CppReference(
        "volatileInteger", type = "volatile int", isValue = true,
        typeInfo = intType.copy(isVolatile = true)
      ),
      CppReference("integer", type = "int", isValue = true, typeInfo = intType),
      CppReference("floating", type = "double", isValue = true, typeInfo = double),
      CppReference("widget", type = "Widget", isValue = true, typeInfo = widgetType),
      CppReference("pointer", type = "Widget *", isValue = true, typeInfo = pointer)
    )
    val functions = listOf(
      function("readDouble", "int", intType, parameter("const double &", constDoubleRef)),
      function("moveDouble", "int", intType, parameter("double &&", doubleRvalueRef)),
      function("takePointer", "int", intType, parameter("Widget *", pointer)),
      function("mutatePointer", "int", intType, parameter("Widget * &", pointerLvalueRef)),
      function("inspectPointer", "int", intType, parameter("Widget * const &", constPointerRef)),
      function("movePointer", "int", intType, parameter("Widget * &&", pointerRvalueRef)),
      constructor
    )
    val language = language(values = values, functions = functions)

    assertTrue(language.recognizes("readDouble(volatileInteger);"))
    assertTrue(language.recognizes("moveDouble(integer);"))
    assertFalse(language.recognizes("moveDouble(floating);"), "same-type lvalue is not converted")
    assertTrue(language.recognizes("takePointer(nullptr);"))
    assertFalse(language.recognizes("mutatePointer(nullptr);"))
    assertTrue(language.recognizes("inspectPointer(nullptr);"))
    assertTrue(language.recognizes("movePointer(nullptr);"))
    assertTrue(language.recognizes("PointerHolder{nullptr};"))

    val declarations = CppCompletionGrammar().generate(
      context(values = values, functions = functions, extraIdentifiers = setOf("ref"))
        .copy(requiredIdentifier = "ref"),
      emptyList()
    )
    assertTrue(declarations.recognizes("const double & ref = volatileInteger;"))
    assertTrue(declarations.recognizes("double && ref = volatileInteger;"))
    assertFalse(declarations.recognizes("double && ref = floating;"))
  }

  @Test
  fun exactConditionalStatesBindReferencesAndVoidPointersCannotBeDereferenced() {
    val bool = typeInfo("builtin:conditional-bool", "builtin", sourceSpellable = true)
    val void = typeInfo("builtin:conditional-void", "builtin", sourceSpellable = true)
    val voidPointer = CppTypeInfo(
      id = "pointer:conditional-void", canonicalId = "pointer:conditional-void",
      valueCanonicalId = "pointer:conditional-void", kind = "pointer",
      pointeeCanonicalId = void.valueCanonicalId, isSourceSpellable = true
    )
    val values = listOf(
      CppReference("flag", type = "bool", isValue = true, typeInfo = bool),
      CppReference("first", type = "Widget", isValue = true, typeInfo = widgetType),
      CppReference("second", type = "Widget", isValue = true, typeInfo = widgetType),
      CppReference("raw", type = "void *", isValue = true, typeInfo = voidPointer)
    )
    val functions = listOf(
      function("nothing", "void", void),
      function(
        "takeLvalue", "int", intType,
        parameter("Widget &", widgetType.reference("lvalueReference"))
      ),
      function(
        "takeRvalue", "int", intType,
        parameter("Widget &&", widgetType.reference("rvalueReference"))
      )
    )
    val language = language(values = values, functions = functions)

    assertTrue(language.recognizes("takeLvalue(flag ? first : second);"))
    assertFalse(language.recognizes("takeRvalue(flag ? first : second);"))
    assertTrue(language.recognizes("nothing();"))
    assertFalse(language.recognizes("*raw;"))

    val declarations = CppCompletionGrammar().generate(
      context(values = values, functions = functions, extraIdentifiers = setOf("ref"))
        .copy(requiredIdentifier = "ref"),
      emptyList()
    )
    assertTrue(declarations.recognizes("Widget & ref = flag ? first : second;"))
  }

  private fun language(
    values: List<CppReference> = emptyList(),
    types: List<CppReference> = emptyList(),
    functions: List<CppReference> = emptyList(),
    members: List<CppReference> = emptyList(),
    extraIdentifiers: Set<String> = emptySet(),
    conversions: List<CppConversion> = emptyList()
  ) = CppCompletionGrammar().generate(
    context(values, types, functions, members, extraIdentifiers, conversions), emptyList()
  )

  private fun context(
    values: List<CppReference> = emptyList(),
    types: List<CppReference> = emptyList(),
    functions: List<CppReference> = emptyList(),
    members: List<CppReference> = emptyList(),
    extraIdentifiers: Set<String> = emptySet(),
    conversions: List<CppConversion> = emptyList()
  ): CppCompletionContext {
    val identifiers = buildSet {
      addAll(extraIdentifiers)
      (values + types + functions + members).forEach { reference ->
        reference.name.split("::").filterTo(this) { it.matches(Regex("[A-Za-z_][A-Za-z0-9_]*")) }
      }
    }
    return CppCompletionContext(
      identifiers = identifiers,
      sourceIdentifiers = identifiers,
      completionKind = "expression",
      values = values,
      types = types,
      functions = functions,
      conversions = conversions,
      membersByType = members.groupBy { it.ownerType.orEmpty() }
        .map { (owner, declarations) -> CppTypeMembers(owner, declarations) }
    )
  }

  private fun function(
    name: String,
    returnType: String,
    returnInfo: CppTypeInfo,
    vararg parameters: CppParameter
  ) = CppReference(
    name, returnType = returnType, parameters = parameters.toList(), kind = "function",
    isCallable = true, returnTypeInfo = returnInfo
  )

  private fun parameter(type: String, info: CppTypeInfo) = CppParameter(type = type, typeInfo = info)

  private fun typeInfo(
    canonicalId: String,
    kind: String,
    sourceSpellable: Boolean
  ) = CppTypeInfo(
    id = canonicalId,
    canonicalId = canonicalId,
    valueCanonicalId = canonicalId,
    kind = kind,
    isDependent = false,
    isInstantiationDependent = false,
    isSourceSpellable = sourceSpellable
  )

  private fun CppTypeInfo.reference(
    kind: String,
    isConst: Boolean = false,
    isVolatile: Boolean = false
  ) = copy(
    id = "$kind:${valueCanonicalId}:${if (isConst) "const" else "mutable"}:${if (isVolatile) "volatile" else "plain"}",
    canonicalId = "$kind:${valueCanonicalId}:${if (isConst) "const" else "mutable"}:${if (isVolatile) "volatile" else "plain"}",
    kind = kind,
    isConst = isConst,
    isVolatile = isVolatile
  )

  private fun pointerInfo(canonicalId: String, pointeeConst: Boolean) = CppTypeInfo(
    id = canonicalId,
    canonicalId = canonicalId,
    valueCanonicalId = canonicalId,
    kind = "pointer",
    pointeeCanonicalId = widgetType.valueCanonicalId,
    pointeeIsConst = pointeeConst,
    pointeeIsVolatile = false,
    isDependent = false,
    isInstantiationDependent = false,
    isSourceSpellable = true
  )

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)
}
