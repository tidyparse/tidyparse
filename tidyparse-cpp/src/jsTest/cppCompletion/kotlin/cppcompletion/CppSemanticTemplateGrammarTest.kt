package cppcompletion

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CppSemanticTemplateGrammarTest {
  private val intInfo = typeInfo("builtin:int", "builtin")
  private val visibleInfo = typeInfo("record:Visible", "record")
  private val hiddenInfo = typeInfo("record:Hidden", "record", sourceSpellable = false)
  private val dependentTemplate = CppTypeInfo(
    id = "template:dependent",
    canonicalId = "template:dependent",
    valueCanonicalId = "template:dependent",
    kind = "record",
    isDependent = true,
    isInstantiationDependent = true,
    isSourceSpellable = false
  )

  @Test
  fun dependentClassTemplatesOmitUnconstrainedDefaultedArguments() {
    val language = language(
      template(
        "Box", "work::Box",
        CppParameter(name = "T", type = "type"),
        CppParameter(name = "Policy", type = "type", hasDefault = true)
      )
    )

    assertTrue(language.recognizes("work::Box<Visible>* fresh;"))
    assertTrue(language.recognizes("Box<int>* fresh;"))
    assertFalse(language.recognizes("work::Box<Hidden>* fresh;"))
    assertFalse(language.recognizes("work::Box<Visible, int>* fresh;"))
    assertFalse(language.recognizes("work::Box<Visible, int, double>* fresh;"))
  }

  @Test
  fun factDrivenTemplateIdsFormAliasesAndOrdinaryDeclarations() {
    val template = template(
      "qualified::Template", "qualified::Template",
      CppParameter(name = "First", type = "type"),
      CppParameter(name = "Second", type = "type"),
      CppParameter(name = "Third", type = "type", hasDefault = true)
    )
    val otherInfo = typeInfo("record:Other", "record")
    val types = listOf(type("Other", otherInfo), alias("Alias", "Other", otherInfo))
    fun context(declarator: String) = CppCompletionContext(
      identifiers = setOf("qualified", "Template", "Other", "Alias", declarator),
      sourceIdentifiers = setOf("qualified", "Template", "Other", "Alias", declarator),
      types = listOf(template) + types,
      completionKind = "type",
      requiredIdentifier = declarator
    )
    fun language(declarator: String) =
      CppCompletionGrammar().generate(context(declarator), emptyList())

    val alias = language("Alias")
    val aliasStatement = "using Alias = qualified::Template<int, Other>;"
    assertRecognizedAtEveryBoundary(context("Alias"), aliasStatement)
    assertFalse(alias.recognizes("using Alias = qualified::Template<int>;"))
    assertFalse(alias.recognizes("using Alias = qualified::Template<int, Other, double>;"))
    assertFalse(alias.recognizes("using Alias = qualified::Template<int, Other, double, int>;"))
    assertFalse(alias.recognizes("using Alias = qualified::Template<int, Missing, double>;"))
    assertFalse(alias.recognizes("using Alias = qualified::Unknown<int, Other>;"))

    val declaration = language("value")
    val declarationStatement = "qualified::Template<int, Alias> value;"
    assertRecognizedAtEveryBoundary(context("value"), declarationStatement)
    assertFalse(declaration.recognizes("qualified::Template<int, Alias, double> value;"))
    assertFalse(declaration.recognizes("qualified::Template<int> value;"))
    assertFalse(declaration.recognizes("qualified::Template<int, Alias, double, Other> value;"))
    assertFalse(declaration.recognizes("qualified::Template<int, Undeclared> value;"))
  }

  @Test
  fun typedPacksAreFactoredToTheFiniteHorizonAndNonTypeArgumentsUseTypedLiterals() {
    val language = language(
      template(
        "Bundle", "meta::Bundle",
        CppParameter(name = "T", type = "type"),
        CppParameter(name = "Rest", type = "type", isPack = true)
      ),
      template(
        "Slots", "meta::Slots",
        CppParameter(name = "T", type = "type"),
        CppParameter(name = "N", type = "int", canonicalType = "int", typeInfo = intInfo)
      )
    )

    assertTrue(language.recognizes("meta::Bundle<Visible>* fresh;"))
    assertTrue(language.recognizes("meta::Bundle<Visible, int>* fresh;"))
    assertTrue(language.recognizes("meta::Bundle<Visible, int, double>* fresh;"))
    assertFalse(language.recognizes("meta::Bundle<Visible, Missing>* fresh;"))
    assertTrue(language.recognizes("meta::Slots<Visible, 4>* fresh;"))
    assertFalse(language.recognizes("meta::Slots<Visible, visible>* fresh;"))
  }

  @Test
  fun templateTemplateParametersRequireAConcreteSpecializationFact() {
    val required = language(template(
      "Adaptor", "meta::Adaptor",
      CppParameter(name = "T", type = "type"),
      CppParameter(name = "Container", type = "template")
    ))
    val defaulted = language(template(
      "DefaultAdaptor", "meta::DefaultAdaptor",
      CppParameter(name = "T", type = "type"),
      CppParameter(name = "Container", type = "template", hasDefault = true)
    ))

    assertFalse(required.recognizes("meta::Adaptor<Visible>* fresh;"))
    assertFalse(required.recognizes("meta::Adaptor<Visible, Box>* fresh;"))
    assertTrue(defaulted.recognizes("meta::DefaultAdaptor<Visible>* fresh;"))
    assertFalse(defaulted.recognizes("meta::DefaultAdaptor<Visible, Box>* fresh;"))
  }

  @Test
  fun templateArgumentsKeepQualifiedSemaParameterTypeSpellings() {
    val fromRange = typeInfo("record:std::from_range_t", "record")
    val context = CppCompletionContext(
      identifiers = setOf("work", "Box", "std", "from_range_t", "fresh"),
      sourceIdentifiers = setOf("work", "Box", "std", "from_range_t", "fresh"),
      types = listOf(template("Box", "work::Box", CppParameter(name = "T", type = "type"))),
      functions = listOf(CppReference(
        name = "consume",
        kind = "Function",
        provenance = "sema",
        completionVisible = true,
        isCallable = true,
        parameters = listOf(CppParameter(
          type = "std::from_range_t",
          canonicalType = "std::from_range_t",
          typeInfo = fromRange
        ))
      )),
      completionKind = "type",
      requiredIdentifier = "fresh"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes("work::Box<std::from_range_t>* fresh;"))
    assertFalse(language.recognizes("work::Box<from_range_t>* fresh;"))
  }

  @Test
  fun variableTemplatesAreNotPublishedAsBareExpressions() {
    val language = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = setOf("meta", "enabled_v", "Visible"),
        sourceIdentifiers = setOf("meta", "enabled_v", "Visible"),
        values = listOf(CppReference(
          name = "meta::enabled_v",
          type = "bool",
          canonicalType = "bool",
          kind = "VarTemplate",
          provenance = "sema",
          typeInfo = typeInfo("builtin:bool", "builtin"),
          isValue = true,
          templateParameters = listOf(CppParameter(name = "T", type = "type"))
        ))
      ),
      emptyList()
    )

    assertFalse(language.recognizes("meta::enabled_v;"))
  }

  @Test
  fun declarationOnlyTypesKeepPointerAndTemplateSpellingsWithoutBecomingExpressions() {
    val passive = type("catalog::Passive", typeInfo("record:catalog::Passive", "record"))
    val box = template("catalog::Box", "catalog::Box", CppParameter(name = "T", type = "type"))
    val context = CppCompletionContext(
      identifiers = setOf("catalog", "Passive", "Box", "fresh"),
      sourceIdentifiers = setOf("catalog", "Passive", "Box", "fresh"),
      types = listOf(passive, box),
      completionKind = "type",
      requiredIdentifier = "fresh"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    // Pointer syntax is factored over the Sema-approved pointee spelling; it does not require a
    // synthetic pointer QualType or an expression lattice for Passive.
    assertTrue(language.recognizes("catalog::Passive* fresh;"))
    assertTrue(language.recognizes("catalog::Passive* fresh = nullptr;"))
    assertTrue(language.recognizes("using fresh = catalog::Passive*;"))
    assertTrue(language.recognizes("catalog::Box<catalog::Passive*>* fresh;"))

    // A TypeDecl alone is not evidence for construction or another value expression.
    assertFalse(language.recognizes("catalog::Passive{};"))
    assertFalse(language.recognizes("catalog::Passive fresh;"))
  }

  @Test
  fun compilerBindingProfilesRetainExactFactoredPointerDeclarations() {
    val passive = type("catalog::Passive", typeInfo("record:catalog::Passive", "record"))
    val rejected = type("catalog::Rejected", typeInfo("record:catalog::Rejected", "record"))
    val gate = CppSingletonBindingGate(
      binder = "fresh",
      accepted = setOf(CppBindingProfile("catalog::Passive *")),
      probed = setOf(
        CppBindingProfile("catalog::Passive *"),
        CppBindingProfile("catalog::Rejected *")
      )
    )
    val language = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = setOf("catalog", "Passive", "Rejected", "fresh"),
        sourceIdentifiers = setOf("catalog", "Passive", "Rejected", "fresh"),
        types = listOf(passive, rejected),
        completionKind = "type",
        requiredBinderObligation = CppRequiredBinderObligation(setOf("fresh"), gate)
      ),
      emptyList()
    )

    assertTrue(language.recognizes("catalog::Passive* fresh;"))
    assertFalse(language.recognizes("catalog::Rejected* fresh;"))
  }

  @Test
  fun declarationOnlySemaGraphScalesLinearlyAndKeepsItsTailSpelling() {
    fun prepared(size: Int): PreparedCppCompletionGrammar {
      val names = (0 until size).map { "Type$it" }
      val types = names.map { name ->
        type("catalog::$name", typeInfo("record:catalog::$name", "record"))
      }
      return CppCompletionGrammar().prepare(CppCompletionContext(
        identifiers = (names + "catalog" + "fresh").toSet(),
        sourceIdentifiers = (names + "catalog" + "fresh").toSet(),
        types = types,
        completionKind = "type",
        requiredIdentifier = "fresh"
      ))
    }

    val small = prepared(2_048)
    val large = prepared(4_096)
    assertTrue(large.sourceProductionCount <= small.sourceProductionCount * 2)
    assertTrue(large.sourceProductionCount < 20_000)
    assertTrue(large.recognizes(cppLines("catalog::Type4095* fresh;").single().tokens))
    assertTrue(large.recognizes(cppLines("using fresh = catalog::Type4095;").single().tokens))
    assertFalse(large.recognizes(cppLines("catalog::Type4095{};").single().tokens))
  }

  private fun language(vararg templates: CppReference): CppSuffixGrammar =
    CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = setOf("work", "meta", "Box", "Bundle", "Slots", "Visible", "Hidden", "fresh"),
        sourceIdentifiers = setOf("work", "meta", "Box", "Bundle", "Slots", "Visible", "Hidden", "fresh"),
        types = templates.toList() + listOf(type("Visible", visibleInfo), type("Hidden", hiddenInfo)),
        completionKind = "type",
        requiredIdentifier = "fresh"
      ),
      emptyList()
    )

  private fun template(
    name: String,
    qualifiedName: String,
    vararg parameters: CppParameter
  ) = CppReference(
    name = name,
    kind = "ClassTemplate",
    qualifiedName = qualifiedName,
    provenance = "sema",
    typeInfo = dependentTemplate.copy(
      id = "template:$qualifiedName",
      canonicalId = "template:$qualifiedName",
      valueCanonicalId = "template:$qualifiedName"
    ),
    isType = true,
    templateParameters = parameters.toList(),
    completionVisible = true
  )

  private fun type(name: String, info: CppTypeInfo) = CppReference(
    name = name,
    type = name,
    canonicalType = name,
    kind = "Class",
    provenance = "sema",
    typeInfo = info,
    isType = true,
    completionVisible = true
  )

  private fun alias(name: String, target: String, info: CppTypeInfo) = CppReference(
    name = name,
    type = target,
    canonicalType = target,
    kind = "TypeAlias",
    provenance = "sema",
    typeInfo = info,
    isType = true,
    completionVisible = true
  )

  private fun typeInfo(id: String, kind: String, sourceSpellable: Boolean = true) = CppTypeInfo(
    id = id,
    canonicalId = id,
    valueCanonicalId = id,
    kind = kind,
    isSourceSpellable = sourceSpellable
  )

  private fun CppSuffixGrammar.recognizes(statement: String): Boolean =
    recognizes(cppLines(statement).single().tokens)

  private fun assertRecognizedAtEveryBoundary(
    context: CppCompletionContext,
    statement: String
  ) {
    val line = cppLines(statement).single()
    val prepared = CppCompletionGrammar().prepare(context)
    cppTruncations(line).forEach { truncation ->
      assertTrue(
        prepared.generate(truncation.prefix).recognizes(truncation.suffix),
        "$statement rejected at token ${truncation.prefix.size}"
      )
    }
  }
}
