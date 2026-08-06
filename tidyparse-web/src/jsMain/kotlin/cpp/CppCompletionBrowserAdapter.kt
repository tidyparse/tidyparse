import ai.hypergraph.tidyparse.lexCppTokenSpans
import cppcompletion.CppCompletionContext
import cppcompletion.CppCompletionQuery
import cppcompletion.CppBinaryOperatorWitness
import cppcompletion.CppCallWitness
import cppcompletion.CppBindingProfile
import cppcompletion.CppConversion
import cppcompletion.CppExpressionProfile
import cppcompletion.CppExpressionWitness
import cppcompletion.CppParameter
import cppcompletion.CppReceiver
import cppcompletion.CppReference
import cppcompletion.CppRequiredBinderObligation
import cppcompletion.CppSignature
import cppcompletion.CppSingletonBindingGate
import cppcompletion.CppToken
import cppcompletion.CppTokenKind
import cppcompletion.CppTemplateArgumentProfile
import cppcompletion.CppTypeInfo
import cppcompletion.CppTypeMembers
import cppcompletion.CppTypeProfile
import cppcompletion.CPP_MAX_INTERACTIVE_COMPLETIONS
import cppcompletion.cppLines
import cppcompletion.isWellFormedCppExpressionProfile
import cppcompletion.isWellFormedCppTemplateArgument
import cppcompletion.hasWellFormedTargetIdentity

// Strict statement preparation must not silently treat a truncated latency-tier response as a
// complete Sema index. These remain finite transport guards; the endpoint reports incompleteness
// whenever either guard is reached, and interactive callers may request a smaller provisional tier.
internal const val CPP_SEMANTIC_GRAPH_LIMIT = 4_096
internal const val CPP_SEMANTIC_GRAPH_DEPTH = 2
internal const val CPP_SEMANTIC_OPERATION_LIMIT = 1_024
internal const val CPP_SEMANTIC_OPERATION_DEPTH = 2
internal const val CPP_SEMANTIC_EXPRESSION_WITNESS_LIMIT = 64
internal const val CPP_SEMANTIC_CALL_WITNESS_LIMIT = 256
internal const val CPP_SEMANTIC_CALL_WITNESS_MAX_ARITY = 4

/** Immutable, editor-independent input for one full-statement completion request. */
data class CppEditorStatementSnapshot(
  val line: Int,
  val character: Int,
  val statementStartCharacter: Int,
  val prefixText: String,
  val semanticPrefixText: String,
  val tokens: List<CppToken>,
  val replacementEndCharacter: Int,
  val cacheKey: String,
  val seed: Int
) {
  /** An identifier being prefix-completed, or a lexer token genuinely intersected by the caret. */
  val activeFragment: CppToken?
    get() = tokens.lastOrNull()?.takeIf { token ->
      token.end == prefixText.length && token.isCppCompletionFragment()
    }

  /** Fully committed tokens before [activeFragment]. */
  val stableTokens: List<CppToken>
    get() = if (activeFragment == null) tokens else tokens.dropLast(1)

  /** Exact source preceding [activeFragment], including the original statement indentation. */
  val stablePrefixText: String
    get() = prefixText.substring(0, activeFragment?.start ?: prefixText.length)
}

fun CppEditorStatementSnapshot.completionQuery(
  identifiers: Set<String>,
  limit: Int = CPP_MAX_INTERACTIVE_COMPLETIONS,
  seed: Int = this.seed
) = CppCompletionQuery(stableTokens, stablePrefixText, identifiers, activeFragment, limit, seed)

/**
 * LSP position used to acquire declarations for this statement completion.
 *
 * Immediately after a template opener clang is completing the first template argument, so Sema
 * no longer reports the template declaration needed to construct that argument list. Querying the
 * same unchanged document immediately before a syntactic simple-template-id opener retains that
 * declaration. The editor replacement range continues to use [CppEditorStatementSnapshot.character].
 */
internal fun cppSemanticCompletionCharacter(snapshot: CppEditorStatementSnapshot): Int {
  val tokens = snapshot.stableTokens
  val opener = tokens.lastOrNull()
  val templateName = tokens.getOrNull(tokens.lastIndex - 1)
  return if (
    opener?.text == "<" && opener.completeText == null &&
    opener.end == snapshot.prefixText.length && templateName?.kind == CppTokenKind.IDENTIFIER
  ) snapshot.statementStartCharacter + opener.start
  else snapshot.character
}

/** One clangd completion response together with the cursor that produced it. */
data class CppClangdCompletionGroup(
  val result: dynamic,
  val receiverMember: Boolean = false,
  val receiverOperator: String? = null
)

/** Typed view of the declaration payload kept alive by clang's completion Sema. */
private external interface CppSemanticSymbolDto {
  val kind: String?
  val provenance: dynamic
  val isCallable: Boolean?
  val isValue: Boolean?
  val isMember: Boolean?
  val isType: Boolean?
  val isStatic: Boolean?
  val isConstMethod: Boolean?
  val isVolatileMethod: Boolean?
  val isMutableField: Boolean?
  val isBitField: Boolean?
  val isVariadic: Boolean?
  val isAbstract: Boolean?
  val isEmptyAggregate: Boolean?
  val isDependent: Boolean?
  val isSourceSpellable: Boolean?
  val isExplicit: Boolean?
  val refQualifier: String?
  val type: String?
  val returnType: String?
  val ownerType: String?
  val canonicalType: String?
  val canonicalReturnType: String?
  val canonicalOwnerType: String?
  val typeInfo: dynamic
  val returnTypeInfo: dynamic
  val ownerTypeInfo: dynamic
  val id: String?
  val primaryTemplateId: String?
  val qualifiedName: String?
  val parameters: dynamic
  val templateParameters: dynamic
}

private val CPP_ARRAY_TYPE_INFO_KINDS = setOf(
  "array", "constantArray", "incompleteArray", "variableArray"
)
private val CPP_ARRAY_BOUND = Regex("[1-9][0-9]*")

private fun cppTypeInfo(value: dynamic): CppTypeInfo? {
  if (!cppDefined(value)) return null
  return CppTypeInfo(
    id = value.id as? String,
    canonicalId = value.canonicalId as? String,
    valueCanonicalId = value.valueCanonicalId as? String,
    kind = value.kind as? String,
    isConst = value.isConst as? Boolean ?: false,
    isVolatile = value.isVolatile as? Boolean ?: false,
    pointeeCanonicalId = value.pointeeCanonicalId as? String,
    pointeeIsConst = value.pointeeIsConst as? Boolean ?: false,
    pointeeIsVolatile = value.pointeeIsVolatile as? Boolean ?: false,
    elementCanonicalId = value.elementCanonicalId as? String,
    elementIsConst = value.elementIsConst as? Boolean ?: false,
    elementIsVolatile = value.elementIsVolatile as? Boolean ?: false,
    isIncompleteArray = value.isIncompleteArray as? Boolean,
    arrayBound = value.arrayBound as? String,
    isDependent = value.isDependent as? Boolean ?: false,
    isInstantiationDependent = value.isInstantiationDependent as? Boolean ?: false,
    isSourceSpellable = value.isSourceSpellable as? Boolean,
    isComplete = value.isComplete as? Boolean,
    isDefaultConstructible = value.isDefaultConstructible as? Boolean
  )
}

private fun cppBindingProfileFromDto(value: dynamic): CppBindingProfile? {
  if (!cppDefined(value)) return null
  val type = (value.type as? String)?.takeIf(String::isNotBlank) ?: return null
  val declarationKind = value.declarationKind as? String ?: return null
  if (declarationKind !in setOf("object", "lvalueReference", "rvalueReference")) return null
  return CppBindingProfile(
    type = type,
    canonicalType = value.canonicalType as? String,
    typeInfo = cppTypeInfo(value.typeInfo),
    declarationKind = declarationKind
  )
}

private fun cppRequiredBinderObligationFromDto(value: dynamic): CppRequiredBinderObligation? {
  if (!cppDefined(value)) return null
  val binders = cppStringSet(value.binders)
  if (binders.any { !CPP_IDENTIFIER_REGEX.matches(it) }) return null
  val rawGate = value.singletonGate
  val gate = if (!cppDefined(rawGate)) null else {
    val binder = rawGate.binder as? String ?: return null
    CppSingletonBindingGate(
      binder = binder,
      accepted = cppDynamicList(rawGate.accepted).mapNotNull(::cppBindingProfileFromDto).toSet(),
      probed = cppDynamicList(rawGate.probed).mapNotNull(::cppBindingProfileFromDto).toSet(),
      complete = rawGate.complete as? Boolean ?: false
    )
  }
  return runCatching { CppRequiredBinderObligation(binders, gate) }.getOrNull()
}

private fun CppTypeInfo?.cppSemanticId(): String? =
  this?.valueCanonicalId?.takeIf(String::isNotBlank)
    ?: this?.canonicalId?.takeIf(String::isNotBlank)

/** Exact declaring-owner facts required before a Sema declaration can become a member fact. */
private data class CppSemanticOwner(
  val sourceType: String,
  val canonicalType: String?,
  val typeInfo: CppTypeInfo,
  val canonicalId: String
)

private fun cppSemanticOwner(declaration: CppSemanticSymbolDto): CppSemanticOwner? {
  val sourceType = declaration.ownerType?.takeIf(String::isNotBlank) ?: return null
  val canonicalType = declaration.canonicalOwnerType?.takeIf(String::isNotBlank)
  if (cppDefined(declaration.canonicalOwnerType) && canonicalType == null) return null
  val typeInfo = cppTypeInfo(declaration.ownerTypeInfo) ?: return null
  val canonicalId = typeInfo.cppSemanticId() ?: return null
  if (typeInfo.isDependent || typeInfo.isInstantiationDependent) return null
  return CppSemanticOwner(sourceType, canonicalType, typeInfo, canonicalId)
}

private fun cppHasSemanticOwnerPayload(declaration: CppSemanticSymbolDto): Boolean =
  cppDefined(declaration.ownerType) || cppDefined(declaration.canonicalOwnerType) ||
    cppDefined(declaration.ownerTypeInfo)

/**
 * Witness metadata is a compiler-authoritative relation, so every field that distinguishes two
 * C++ expression states must survive schema-v2 decoding explicitly.  The general semantic graph
 * decoder remains intentionally lenient for older payloads; witnesses fail closed instead.
 */
private fun cppWitnessTypeInfo(value: dynamic): CppTypeInfo? {
  if (!cppDefined(value)) return null
  val canonicalId = (value.canonicalId as? String)?.takeIf(String::isNotBlank) ?: return null
  val valueCanonicalId = (value.valueCanonicalId as? String)?.takeIf(String::isNotBlank)
    ?: return null
  val kind = (value.kind as? String)?.takeIf(String::isNotBlank) ?: return null
  val isConst = value.isConst as? Boolean ?: return null
  val isVolatile = value.isVolatile as? Boolean ?: return null
  val isDependent = value.isDependent as? Boolean ?: return null
  val isInstantiationDependent = value.isInstantiationDependent as? Boolean ?: return null
  val isSourceSpellable = value.isSourceSpellable as? Boolean ?: return null
  val pointeeCanonicalId = if (kind == "pointer" || cppDefined(value.pointeeCanonicalId))
    (value.pointeeCanonicalId as? String)?.takeIf(String::isNotBlank) ?: return null
  else null
  val pointeeIsConst = if (pointeeCanonicalId != null)
    value.pointeeIsConst as? Boolean ?: return null
  else false
  val pointeeIsVolatile = if (pointeeCanonicalId != null)
    value.pointeeIsVolatile as? Boolean ?: return null
  else false
  val hasArrayMetadata = cppDefined(value.elementCanonicalId) ||
    cppDefined(value.elementIsConst) || cppDefined(value.elementIsVolatile) ||
    cppDefined(value.isIncompleteArray) || cppDefined(value.arrayBound)
  val arrayKind = kind in CPP_ARRAY_TYPE_INFO_KINDS
  if (arrayKind != hasArrayMetadata) return null
  val elementCanonicalId = if (arrayKind)
    (value.elementCanonicalId as? String)?.takeIf(String::isNotBlank) ?: return null
  else null
  val elementIsConst = if (arrayKind) value.elementIsConst as? Boolean ?: return null else false
  val elementIsVolatile = if (arrayKind)
    value.elementIsVolatile as? Boolean ?: return null
  else false
  val isIncompleteArray = if (arrayKind)
    value.isIncompleteArray as? Boolean ?: return null
  else null
  val hasArrayBound = cppDefined(value.arrayBound)
  val arrayBound = if (hasArrayBound)
    (value.arrayBound as? String)?.takeIf(CPP_ARRAY_BOUND::matches) ?: return null
  else null
  if (arrayKind && (
      isIncompleteArray == true && hasArrayBound ||
        isIncompleteArray == false && !hasArrayBound ||
        kind == "incompleteArray" && isIncompleteArray != true ||
        kind == "constantArray" && isIncompleteArray != false
      )
  ) return null
  return CppTypeInfo(
    id = value.id as? String,
    canonicalId = canonicalId,
    valueCanonicalId = valueCanonicalId,
    kind = kind,
    isConst = isConst,
    isVolatile = isVolatile,
    pointeeCanonicalId = pointeeCanonicalId,
    pointeeIsConst = pointeeIsConst,
    pointeeIsVolatile = pointeeIsVolatile,
    elementCanonicalId = elementCanonicalId,
    elementIsConst = elementIsConst,
    elementIsVolatile = elementIsVolatile,
    isIncompleteArray = isIncompleteArray,
    arrayBound = arrayBound,
    isDependent = isDependent,
    isInstantiationDependent = isInstantiationDependent,
    isSourceSpellable = isSourceSpellable,
    isComplete = value.isComplete as? Boolean,
    isDefaultConstructible = value.isDefaultConstructible as? Boolean
  )
}

/** Conversion identities are optional for legacy payloads. Once present, however, conversion
 * metadata is compiler-authoritative and must fail closed under exactly the same cv, dependency,
 * kind, and pointer-pointee checks as a correlated expression witness. */
private fun cppConversionTypeInfo(value: dynamic): CppTypeInfo? = cppWitnessTypeInfo(value)

private fun cppExpressionProfile(value: dynamic): CppExpressionProfile? {
  if (!cppDefined(value)) return null
  val kind = value.kind as? String ?: return null
  val spelling = if (cppDefined(value.spelling)) value.spelling as? String ?: return null else null
  val objectKind = value.objectKind as? String ?: return null
  val valueCategory = value.valueCategory as? String
    ?: return null
  if (valueCategory !in setOf("lvalue", "xvalue", "prvalue")) return null
  val profile = CppExpressionProfile(
    kind = kind,
    spelling = spelling,
    objectKind = objectKind,
    type = value.type as? String,
    canonicalType = value.canonicalType as? String,
    typeInfo = cppWitnessTypeInfo(value.typeInfo) ?: return null,
    valueCategory = valueCategory
  )
  return profile.takeIf(CppExpressionProfile::isWellFormedCppExpressionProfile)
}

private fun cppTypeProfile(value: dynamic): CppTypeProfile? {
  if (!cppDefined(value)) return null
  return CppTypeProfile(
    type = value.type as? String ?: return null,
    canonicalType = value.canonicalType as? String,
    typeInfo = cppWitnessTypeInfo(value.typeInfo) ?: return null
  )
}

private fun cppTemplateArgumentProfile(value: dynamic): CppTemplateArgumentProfile? {
  if (!cppDefined(value)) return null
  val profile = CppTemplateArgumentProfile(
    kind = value.kind as? String ?: return null,
    type = cppTypeProfile(value.type) ?: return null,
    spelling = value.spelling as? String,
    canonicalValue = value.canonicalValue as? String
  )
  return profile.takeIf(CppTemplateArgumentProfile::isWellFormedCppTemplateArgument)
}

private fun cppSemanticReferenceKind(
  declaration: CppSemanticSymbolDto,
  fallback: String = "unknown"
): String {
  val kind = declaration.kind?.lowercase().orEmpty()
  return when {
    "constructor" in kind -> "constructor"
    kind == "namespace" || kind == "namespacealias" -> "namespace"
    "enumconstant" in kind -> "enumMember"
    kind == "enum" -> "enum"
    "classtemplatespecialization" in kind -> "classTemplateSpecialization"
    "classtemplate" in kind -> "classTemplate"
    "vartemplatespecialization" in kind -> "varTemplateSpecialization"
    "vartemplate" in kind -> "varTemplate"
    "typealiastemplate" in kind -> "typeAliasTemplate"
    "typealias" in kind || "typedef" in kind -> "typeAlias"
    "templateparm" in kind -> "typeParameter"
    declaration.isMember == true && declaration.isCallable == true -> "method"
    declaration.isCallable == true -> "function"
    declaration.isMember == true -> "field"
    declaration.isType == true -> when {
      "struct" in kind -> "struct"
      else -> "class"
    }
    declaration.isValue == true -> "variable"
    else -> fallback
  }
}

/** Converts one declaration using only exact Sema facts; index-only symbols contribute names. */
private fun cppSemanticReference(
  raw: dynamic,
  name: String,
  fallbackKind: String = "unknown",
  detail: String? = null,
  fallbackType: String? = null,
  fallbackReturnType: String? = null,
  receiverMember: Boolean = false,
  completionVisible: Boolean = false,
  activeCallable: Boolean = false
): CppReference? {
  if (!cppDefined(raw) || name.isBlank()) return null
  val declaration: CppSemanticSymbolDto = raw
  // Clang's declaration name for a variable-template specialization can omit its template
  // arguments. Unlike a class specialization, its QualType is merely the value's type and cannot
  // reconstruct the template-id; publishing the bare primary name would be ill-formed.
  if (declaration.kind?.contains("VarTemplateSpecialization") == true && '<' !in name) return null
  val provenance = declaration.provenance
  val fromSema = provenance?.sema as? Boolean == true
  val fromIndex = provenance?.index as? Boolean == true
  val callable = fromSema && declaration.isCallable == true
  val value = fromSema && declaration.isValue == true
  val member = fromSema && declaration.isMember == true
  // A member is useful only together with its declaration identity and exact canonical owner.
  // Conversely, owner/static payload on a declaration not classified as a member is internally
  // inconsistent and must not be laundered into a free declaration by this adapter.
  val owner = if (member) {
    if (declaration.id.isNullOrBlank()) return null
    cppSemanticOwner(declaration) ?: return null
  } else null
  if (fromSema && !member &&
    (declaration.isStatic == true || cppHasSemanticOwnerPayload(declaration))
  ) return null
  val parameters = if (!fromSema) emptyList() else
    cppDynamicList(declaration.parameters).map { parameter ->
      val hasDefault = parameter?.hasDefault as? Boolean == true
      CppParameter(
        name = parameter?.name as? String ?: "",
        type = parameter?.type as? String ?: parameter?.canonicalType as? String ?: "",
        defaultValue = "".takeIf { hasDefault },
        canonicalType = parameter?.canonicalType as? String,
        typeInfo = cppTypeInfo(parameter?.typeInfo),
        hasDefault = hasDefault,
        isPack = parameter?.isPack as? Boolean == true
      )
    }
  val declaredType = declaration.type
  val declaredReturnType = declaration.returnType
  val semanticProvenance = when {
    fromSema && fromIndex -> "sema+index"
    fromSema -> "sema"
    else -> "index"
  }
  return CppReference(
    name = name,
    type = if (value && !callable) declaredType ?: fallbackType else null,
    returnType = if (callable) declaredReturnType ?: fallbackReturnType else null,
    parameters = parameters,
    kind = cppSemanticReferenceKind(declaration, fallbackKind),
    detail = detail,
    receiverMember = member && receiverMember,
    ownerType = owner?.sourceType,
    source = semanticProvenance,
    abstract = declaration.isAbstract ?: false,
    emptyAggregate = declaration.isEmptyAggregate ?: false,
    id = declaration.id,
    primaryTemplateId = declaration.primaryTemplateId,
    qualifiedName = declaration.qualifiedName,
    provenance = semanticProvenance,
    canonicalType = declaration.canonicalType,
    canonicalReturnType = declaration.canonicalReturnType,
    canonicalOwnerType = owner?.canonicalType,
    typeInfo = cppTypeInfo(declaration.typeInfo),
    returnTypeInfo = cppTypeInfo(declaration.returnTypeInfo),
    ownerTypeInfo = owner?.typeInfo,
    isType = fromSema && declaration.isType == true,
    isValue = value,
    isCallable = callable,
    isMember = member,
    isStatic = fromSema && declaration.isStatic == true,
    isConstMethod = declaration.isConstMethod,
    isVolatileMethod = declaration.isVolatileMethod,
    refQualifier = declaration.refQualifier,
    isMutableField = declaration.isMutableField,
    isBitField = declaration.isBitField,
    isVariadic = declaration.isVariadic == true,
    isExplicit = declaration.isExplicit,
    templateParameters = if (!fromSema) emptyList() else
      cppDynamicList(declaration.templateParameters).map { parameter ->
        val hasDefault = parameter?.hasDefault as? Boolean
        CppParameter(
          name = parameter?.name as? String ?: "",
          type = parameter?.type as? String ?: parameter?.kind as? String ?: "",
          defaultValue = "".takeIf { hasDefault == true },
          canonicalType = parameter?.canonicalType as? String,
          typeInfo = cppTypeInfo(parameter?.typeInfo),
          hasDefault = hasDefault,
          isPack = parameter?.isPack as? Boolean == true
        )
      },
    completionVisible = completionVisible,
    activeCallable = activeCallable
  )
}

private data class CppPhysicalLine(
  val start: Int,
  val contentEnd: Int,
  val text: String
)

private data class CppCommentSpan(val start: Int, val end: Int, val inclusiveEnd: Boolean)

/**
 * Extracts the current physical statement prefix using 0-based LSP coordinates.
 *
 * The hot path lexes only the selected line. A token intersected by the caret retains both its
 * typed fragment and complete lexer spelling, so grammar filtering and whole-statement replacement
 * work for identifiers, literals and maximal-munch operators alike. Prefix length is not capped.
 */
fun cppEditorStatementSnapshot(source: String, line: Int, character: Int): CppEditorStatementSnapshot? {
  if (line < 0 || character < 0) return null
  val physical = cppPhysicalLine(source, line) ?: return null
  if (character > physical.text.length || cppIsPreprocessorLine(source, line)) return null

  val comments = cppCommentSpans(source, physical.start, physical.contentEnd)
  val caretOffset = physical.start + character
  if (comments.any { span ->
      caretOffset >= span.start &&
        (caretOffset < span.end || span.inclusiveEnd && caretOffset == span.end)
    }) return null

  val lineTokens = cppLines(physical.text).single().tokens
  val intersectedToken = lineTokens.singleOrNull { token ->
    token.start < character && character < token.end
  }
  val statementStartCharacter = cppStatementStartCharacter(lineTokens, character)
  val completePrefixTokens = lineTokens.filter { token ->
    token.start >= statementStartCharacter && token.end <= character
  }
  val prefixTokens = completePrefixTokens.map { token ->
    token.copy(
      start = token.start - statementStartCharacter,
      end = token.end - statementStartCharacter
    )
  }.toMutableList()
  if (intersectedToken != null && intersectedToken.start >= statementStartCharacter) {
    val fragmentText = physical.text.substring(intersectedToken.start, character)
    prefixTokens += intersectedToken.copy(
      text = fragmentText,
      start = intersectedToken.start - statementStartCharacter,
      end = character - statementStartCharacter,
      completeText = intersectedToken.text
    )
  }

  val replacementEndCharacter = cppStatementReplacementEndCharacter(
    lineText = physical.text,
    lineStartOffset = physical.start,
    lineTokens = lineTokens,
    comments = comments,
    statementStartCharacter = statementStartCharacter,
    character = character
  )
  val prefixText = physical.text.substring(statementStartCharacter, character)
  val activeFragment = prefixTokens.lastOrNull()?.takeIf { token ->
    token.end == prefixText.length && token.isCppCompletionFragment()
  }
  val semanticPrefixText = when {
    activeFragment?.isCppWordFragment() == true ->
      prefixText.substring(0, activeFragment.start)
    intersectedToken != null ->
      physical.text.substring(statementStartCharacter, intersectedToken.end)
    else -> prefixText
  }
  val identity = "$line:$statementStartCharacter:$character:$replacementEndCharacter:" +
    "${prefixText.hashCode()}:${prefixTokens.hashCode()}"
  return CppEditorStatementSnapshot(
    line = line,
    character = character,
    statementStartCharacter = statementStartCharacter,
    prefixText = prefixText,
    semanticPrefixText = semanticPrefixText,
    tokens = prefixTokens,
    replacementEndCharacter = replacementEndCharacter.coerceAtLeast(character),
    cacheKey = identity,
    seed = identity.hashCode()
  )
}

/** Decodes the declaration-backed completion slice emitted directly by clangd/Sema. */
fun cppSemanticCompletionContextDto(
  result: dynamic,
  snapshot: CppEditorStatementSnapshot
): dynamic {
  val schemaVersion = cppInt(result?.schemaVersion, -1)
  if (!cppDefined(result) || schemaVersion !in 1..2)
    return cppCompletionContextToDto(CppCompletionContext(emptySet()))

  val semanticContext = result.context
  val baseType = semanticContext?.baseType as? String
  val canonicalBaseType = semanticContext?.canonicalBaseType as? String
  val receiverOperator = cppReceiverOperator(snapshot.semanticPrefixText)
  val activeCallee = cppActiveCallee(snapshot.semanticPrefixText)
  val activeReferences = cppDynamicList(result.activeCallables).mapNotNull { raw ->
    val symbol: CppSemanticSymbolDto = raw
    val name = symbol.qualifiedName?.takeIf(String::isNotBlank) ?: return@mapNotNull null
    // The already-typed callee is the exact route Sema resolved. qualifiedName remains declaration
    // identity only; it can name a physical namespace/owner hidden by an alias at this cursor.
    val insertion = activeCallee ?: name
    cppSemanticReference(
      raw = raw,
      name = insertion,
      fallbackReturnType = symbol.returnType,
      receiverMember = receiverOperator != null,
      completionVisible = activeCallee != null,
      activeCallable = true
    )
  }
  val items = cppDynamicList(result.items) + cppDynamicList(result.scopeItems)
  val itemReferences = items.flatMap { item ->
    val itemName = item?.insertText as? String
      ?: ((item?.requiredQualifier as? String).orEmpty() + (item?.name as? String).orEmpty())
    if (itemName.isBlank()) return@flatMap emptyList()
    val itemKind = cppReferenceKind(cppInt(item.kind, -1))
    cppDynamicList(item.symbols).mapNotNull { raw ->
      cppSemanticReference(
        raw = raw,
        name = itemName,
        fallbackKind = itemKind,
        detail = item?.signature as? String,
        fallbackType = item?.returnType as? String,
        fallbackReturnType = item?.returnType as? String,
        receiverMember = receiverOperator != null,
        completionVisible = true
      )
    }
  }.filterNot { cppContainsReservedIdentifier(it.name) }
  val itemDeclarationPaths = itemReferences.mapNotNullTo(hashSetOf()) { reference ->
    reference.id?.let { it to reference.name }
  }
  val operations = result.operations
  val operationNodes = if (schemaVersion == 2) cppDynamicList(operations?.nodes) else emptyList()
  val operationTemplates = if (schemaVersion == 2)
    cppDynamicList(operations?.templates) else emptyList()
  val rawOperationExpressionWitnesses = if (schemaVersion == 2)
    cppDynamicList(operations?.expressionWitnesses) else emptyList()
  val rawOperationWitnesses = if (schemaVersion == 2)
    cppDynamicList(operations?.callWitnesses) else emptyList()
  val rawOperationBinaryOperatorWitnesses = rawOperationWitnesses.filter {
    it?.syntax as? String == "binaryOperator"
  }
  val rawOperationCallWitnesses = rawOperationWitnesses.filterNot {
    it?.syntax as? String == "binaryOperator"
  }
  val operationTypeIds = operationNodes.mapNotNullTo(hashSetOf()) { raw ->
    if (!cppDefined(raw) || raw["role"] as? String != "type") return@mapNotNullTo null
    val declaration: CppSemanticSymbolDto = raw
    val typeInfo = cppTypeInfo(declaration.typeInfo) ?: return@mapNotNullTo null
    if (declaration.provenance?.sema as? Boolean != true || declaration.isType != true ||
      typeInfo.isDependent || typeInfo.isInstantiationDependent ||
      typeInfo.isSourceSpellable != true || declaration.typeInfo?.isComplete as? Boolean == false
    ) return@mapNotNullTo null
    typeInfo.valueCanonicalId ?: typeInfo.canonicalId
  }
  val graph = result.graph
  val graphNodes = if (schemaVersion == 2) cppDynamicList(graph?.nodes) else emptyList()
  val graphReferences = graphNodes.mapNotNull { raw ->
    val insertion = raw?.name as? String ?: return@mapNotNull null
    val declaration: CppSemanticSymbolDto = raw
    val hasExactItemPath = declaration.id?.let { it to insertion in itemDeclarationPaths } == true
    val typeInfo = cppTypeInfo(declaration.typeInfo)
    val semanticTypeId = typeInfo?.valueCanonicalId ?: typeInfo?.canonicalId
    // The operation closure starts from declarations visible at this cursor. Joining its exact
    // canonical type identities back to a qualified graph declaration proves that the spelling is
    // a context-relevant type alias, without admitting every transitive alias in an expanded
    // namespace (for example an unrelated forward-declared library alias).
    val hasOperationTypePath = declaration.isType == true && typeInfo != null &&
      !typeInfo.isDependent && !typeInfo.isInstantiationDependent &&
      typeInfo.isSourceSpellable == true && declaration.typeInfo?.isComplete as? Boolean != false &&
      semanticTypeId != null && semanticTypeId in operationTypeIds
    val primaryTemplate = declaration.kind?.let { kind ->
      "Template" in kind && "Specialization" !in kind
    } == true
    val classTemplate = declaration.kind?.let { kind ->
      "ClassTemplate" in kind && "Specialization" !in kind
    } == true
    val indexedForCompletion = declaration.provenance?.index as? Boolean == true
    // A graph namespace walk proves that a qualified declaration exists, but it does not perform
    // ordinary completion lookup for its path. Unqualified paths need the exact item spelling,
    // and primary templates need item evidence even when graph traversal qualified them: otherwise
    // every constrained/internal template in an expanded namespace becomes an invented product.
    if (cppContainsReservedIdentifier(insertion) ||
      "::" !in insertion && !hasExactItemPath ||
      primaryTemplate && !hasExactItemPath && !(classTemplate && indexedForCompletion)
    ) return@mapNotNull null
    // A ClassTemplateSpecializationDecl's declaration name and qualified name both omit its
    // arguments. Its nondependent, source-spellable QualType supplies the template-id suffix, but
    // not the route: that prefix must remain the exact nested-name-specifier authenticated by the
    // graph lookup (aliases and inline namespaces can make declaration identity spell differently).
    val classSpecialization = declaration.kind?.contains("ClassTemplateSpecialization") == true
    val specializationType = (declaration.type ?: declaration.canonicalType)?.takeIf { type ->
      classSpecialization && '<' in type &&
        declaration.typeInfo?.isDependent as? Boolean != true &&
        declaration.typeInfo?.isInstantiationDependent as? Boolean != true &&
        declaration.typeInfo?.isSourceSpellable as? Boolean == true
    }
    // Function/variable specializations must retain their template-id in the graph name itself.
    // qualifiedName is declaration identity and cannot rescue an endpoint-erased source route.
    val templateIdSuffix = specializationType?.let { source ->
      source.indexOf('<').takeIf { it >= 0 }?.let(source::substring)
    }
    val name = if ('<' in insertion || templateIdSuffix == null) insertion
      else insertion + templateIdSuffix
    if (classSpecialization && '<' !in name) return@mapNotNull null
    if (cppContainsReservedIdentifier(name)) return@mapNotNull null
    cppSemanticReference(
      raw = raw,
      name = name,
      completionVisible = hasExactItemPath || hasOperationTypePath ||
        declaration.isMember == true
    )
  }
  val operationReferences = if (schemaVersion == 2)
    operationNodes.mapNotNull { raw ->
      val name = raw?.name as? String ?: return@mapNotNull null
      val role = if (cppDefined(raw)) raw["role"] as? String else null
      // Roles are a closed wire contract. In particular, a constructor is a
      // construction fact rather than a receiver access, and an enumerator is
      // already emitted with its exact ambient spelling. Template calls are
      // represented only by correlated Sema call witnesses.
      val receiverOperation = when (role) {
        "member" -> true
        "constructor", "type", "enumerator" -> false
        else -> return@mapNotNull null
      }
      val declaration: CppSemanticSymbolDto = raw
      // Operation member names are exact owner-relative declaration names. A static member gains
      // an ambient qualified-id only when the endpoint also supplied an exact, source-spellable
      // owner route. qualifiedName is physical declaration identity and is never route evidence.
      val ambientOwner = cppSemanticOwner(declaration)?.takeIf { owner ->
        receiverOperation && declaration.isMember == true && declaration.isStatic == true &&
          owner.typeInfo.isSourceSpellable == true
      }
      val referenceName = ambientOwner?.let { "${it.sourceType}::$name" } ?: name
      cppSemanticReference(
        raw = raw,
        name = referenceName,
        receiverMember = receiverOperation,
        completionVisible = ambientOwner != null
      )
    }.filterNot { cppContainsReservedIdentifier(it.name) }
  else emptyList()
  val operationTemplateReferences = if (schemaVersion == 2)
    operationTemplates.mapNotNull { schema ->
      if (!cppDefined(schema)) return@mapNotNull null
      val name = schema["name"] as? String ?: return@mapNotNull null
      val pattern = schema["pattern"]
      val role = schema["role"] as? String ?: return@mapNotNull null
      if (role !in setOf("member", "constructor")) return@mapNotNull null
      // A primary function template is an advisory substitution schema, not an overload that Sema
      // proved viable with concrete arguments. Retain its exact owner/parameter/result type closure
      // for later specialization, but neutralize every classifier that could publish a call,
      // member access, value, or constructor production. In particular the kind must not contain
      // `constructor`, because constructor classification intentionally precedes callable flags.
      cppSemanticReference(
        raw = pattern,
        name = name,
        detail = role,
        receiverMember = false
      )?.copy(
        kind = "primaryTemplateAdvisory",
        detail = role,
        receiverMember = false,
        isType = false,
        isValue = false,
        isCallable = false,
        isMember = false,
        isStatic = false,
        completionVisible = false,
        activeCallable = false
      )
    }.filterNot { cppContainsReservedIdentifier(it.name) }
  else emptyList()
  val operationConversions = if (schemaVersion == 2)
    cppDynamicList(operations?.conversions).mapNotNull { edge ->
      if (!cppDefined(edge)) return@mapNotNull null
      val kind = edge["kind"] as? String
      if (kind !in setOf("base", "conversion", "constructor"))
        return@mapNotNull null
      val canonicalFromType = (edge["canonicalFromType"] as? String)
        ?.takeIf(String::isNotBlank) ?: return@mapNotNull null
      val canonicalToType = (edge["canonicalToType"] as? String)
        ?.takeIf(String::isNotBlank) ?: return@mapNotNull null
      val from = (edge["fromType"] as? String)?.takeIf(String::isNotBlank)
        ?: return@mapNotNull null
      val to = (edge["toType"] as? String)?.takeIf(String::isNotBlank)
        ?: return@mapNotNull null
      val fromTypeInfo = cppConversionTypeInfo(edge["fromTypeInfo"])
        ?.takeIf { it.isSourceSpellable == true && it.cppSemanticId() != null }
        ?: return@mapNotNull null
      val toTypeInfo = cppConversionTypeInfo(edge["toTypeInfo"])
        ?.takeIf { it.isSourceSpellable == true && it.cppSemanticId() != null }
        ?: return@mapNotNull null
      CppConversion(
        from = from,
        to = to,
        kind = kind,
        canonicalFromType = canonicalFromType,
        canonicalToType = canonicalToType,
        fromTypeInfo = fromTypeInfo,
        toTypeInfo = toTypeInfo
      )
    }.distinct()
  else emptyList()
  val operationExpressionWitnesses = if (schemaVersion == 2)
    rawOperationExpressionWitnesses.mapNotNull { raw ->
      if (!cppDefined(raw)) return@mapNotNull null
      val typeOperand = cppTypeProfile(raw.typeOperand)
      if (cppDefined(raw.typeOperand) && typeOperand == null) return@mapNotNull null
      val expressionOperand = cppExpressionProfile(raw.expressionOperand)
      if (cppDefined(raw.expressionOperand) && expressionOperand == null)
        return@mapNotNull null
      CppExpressionWitness(
        syntax = raw.syntax as? String ?: return@mapNotNull null,
        validation = raw.validation as? String ?: "",
        typeOperand = typeOperand,
        expressionOperand = expressionOperand,
        result = cppExpressionProfile(raw.result) ?: return@mapNotNull null,
        authoritative = raw.authoritative as? Boolean ?: false
      )
    }
  else emptyList()
  val operationCallWitnesses = if (schemaVersion == 2)
    rawOperationCallWitnesses.mapNotNull { raw ->
      if (!cppDefined(raw)) return@mapNotNull null
      val name = raw.name as? String ?: return@mapNotNull null
      val syntax = raw.syntax as? String ?: return@mapNotNull null
      val callableRaw = raw.callable
      val targetId = cppOptionalCallIdentity(raw.targetId) ?: return@mapNotNull null
      val primaryTemplateId = cppOptionalCallIdentity(raw.primaryTemplateId)
        ?: return@mapNotNull null
      val callableId = cppOptionalCallIdentity(callableRaw?.id) ?: return@mapNotNull null
      val callablePrimaryTemplateId = cppOptionalCallIdentity(callableRaw?.primaryTemplateId)
        ?: return@mapNotNull null
      val callableSymbol: CppSemanticSymbolDto = callableRaw
      val callableName = when (syntax) {
        "memberCall" -> name
        else -> callableSymbol.qualifiedName?.takeIf(String::isNotBlank) ?: name
      }
      val callableReturnInfo = cppWitnessTypeInfo(callableRaw.returnTypeInfo)
        ?: return@mapNotNull null
      val callable = cppSemanticReference(
        raw = callableRaw,
        name = callableName,
        receiverMember = syntax == "memberCall"
      )?.copy(
        id = callableId.value,
        primaryTemplateId = callablePrimaryTemplateId.value,
        returnTypeInfo = callableReturnInfo
      ) ?: return@mapNotNull null
      val resultProfile = cppExpressionProfile(raw.result) ?: return@mapNotNull null
      if (!cppIsArray(raw.arguments)) return@mapNotNull null
      val rawArguments = cppDynamicList(raw.arguments)
      val arguments = rawArguments.mapNotNull(::cppExpressionProfile)
      if (arguments.size != rawArguments.size) return@mapNotNull null
      val hasExplicitTemplateArguments = cppDefined(raw.explicitTemplateArguments)
      if (hasExplicitTemplateArguments && !cppIsArray(raw.explicitTemplateArguments))
        return@mapNotNull null
      val rawExplicitTemplateArguments = cppDynamicList(raw.explicitTemplateArguments)
      val explicitTemplateArguments =
        rawExplicitTemplateArguments.mapNotNull(::cppTemplateArgumentProfile)
      if (explicitTemplateArguments.size != rawExplicitTemplateArguments.size)
        return@mapNotNull null
      if (cppDefined(raw.explicitTypeArguments) && !cppIsArray(raw.explicitTypeArguments))
        return@mapNotNull null
      val rawExplicitTypeArguments = cppDynamicList(raw.explicitTypeArguments)
      val explicitTypeArguments = rawExplicitTypeArguments.mapNotNull(::cppTypeProfile)
      if (explicitTypeArguments.size != rawExplicitTypeArguments.size) return@mapNotNull null
      // Legacy migration is allowed only when the tagged field is absent. Never combine schemas.
      if (hasExplicitTemplateArguments && explicitTypeArguments.isNotEmpty())
        return@mapNotNull null
      val receiverProfile = cppExpressionProfile(raw.receiver)
      if (cppDefined(raw.receiver) && receiverProfile == null) return@mapNotNull null
      if (syntax == "memberCall" && receiverProfile == null) return@mapNotNull null
      CppCallWitness(
        name = name,
        syntax = syntax,
        validation = raw.validation as? String ?: "",
        targetId = targetId.value,
        primaryTemplateId = primaryTemplateId.value,
        explicitTemplateArguments = explicitTemplateArguments,
        explicitTypeArguments = if (hasExplicitTemplateArguments) emptyList()
        else explicitTypeArguments,
        receiver = receiverProfile,
        arguments = arguments,
        callable = callable,
        result = resultProfile,
        authoritative = raw.authoritative as? Boolean ?: false
      ).takeIf(CppCallWitness::hasWellFormedTargetIdentity)
    }
  else emptyList()
  val operationBinaryOperatorWitnesses = if (schemaVersion == 2)
    rawOperationBinaryOperatorWitnesses.mapNotNull { raw ->
      if (!cppDefined(raw) || raw.syntax as? String != "binaryOperator")
        return@mapNotNull null
      if (!cppIsArray(raw.arguments) || cppDynamicList(raw.arguments).size != 1)
        return@mapNotNull null
      if (cppDefined(raw.explicitTemplateArguments) &&
        (!cppIsArray(raw.explicitTemplateArguments) ||
          cppDynamicList(raw.explicitTemplateArguments).isNotEmpty())
      ) return@mapNotNull null
      if (cppDefined(raw.explicitTypeArguments) &&
        (!cppIsArray(raw.explicitTypeArguments) ||
          cppDynamicList(raw.explicitTypeArguments).isNotEmpty())
      ) return@mapNotNull null
      val name = raw.name as? String ?: return@mapNotNull null
      val operatorSpelling = raw.operatorSpelling as? String ?: return@mapNotNull null
      val callableRaw = raw.callable
      val targetId = cppOptionalCallIdentity(raw.targetId) ?: return@mapNotNull null
      val primaryTemplateId = cppOptionalCallIdentity(raw.primaryTemplateId)
        ?: return@mapNotNull null
      val callableId = cppOptionalCallIdentity(callableRaw?.id) ?: return@mapNotNull null
      val callablePrimaryTemplateId = cppOptionalCallIdentity(callableRaw?.primaryTemplateId)
        ?: return@mapNotNull null
      val callableReturnInfo = cppWitnessTypeInfo(callableRaw?.returnTypeInfo)
        ?: return@mapNotNull null
      val callableSymbol: CppSemanticSymbolDto = callableRaw
      val callableName = callableSymbol.qualifiedName?.takeIf(String::isNotBlank) ?: name
      val callable = cppSemanticReference(raw = callableRaw, name = callableName)?.copy(
        id = callableId.value,
        primaryTemplateId = callablePrimaryTemplateId.value,
        returnTypeInfo = callableReturnInfo
      ) ?: return@mapNotNull null
      CppBinaryOperatorWitness(
        name = name,
        syntax = "binaryOperator",
        operatorSpelling = operatorSpelling,
        validation = raw.validation as? String ?: "",
        targetId = targetId.value,
        primaryTemplateId = primaryTemplateId.value,
        left = cppExpressionProfile(raw.receiver) ?: return@mapNotNull null,
        right = cppExpressionProfile(cppDynamicList(raw.arguments).single())
          ?: return@mapNotNull null,
        callable = callable,
        result = cppExpressionProfile(raw.result) ?: return@mapNotNull null,
        authoritative = raw.authoritative as? Boolean ?: false
      ).takeIf(CppBinaryOperatorWitness::hasWellFormedTargetIdentity)
    }
  else emptyList()
  val references = cppDistinctReferences(
    itemReferences + activeReferences + graphReferences + operationReferences +
      operationTemplateReferences
  )
  val values = references.filter { it.isValue == true && it.isCallable != true }
  val types = references.filter { it.isType == true }
  val functions = references.filter { it.isCallable == true }
  val members = references.filter { reference ->
    reference.receiverMember && reference.ownerType != null &&
      reference.ownerTypeInfo.cppSemanticId() != null
  }
  val receiverType = baseType ?: canonicalBaseType
  val receiverTypeInfo = cppTypeInfo(semanticContext?.baseTypeInfo)
  val receiverTypeIds = buildSet {
    val pointeeId = receiverTypeInfo?.pointeeCanonicalId?.takeIf(String::isNotBlank)
    if (receiverOperator == "->" && pointeeId != null) add(pointeeId)
    else receiverTypeInfo.cppSemanticId()?.let { add(it) }
  }
  fun legacyObjectSpelling(type: String?): String? {
    var spelling = type?.trim()?.replace(Regex("\\s+"), " ")?.takeIf(String::isNotBlank)
      ?: return null
    while (true) {
      val unqualified = when {
        spelling.startsWith("const ") -> spelling.removePrefix("const ").trimStart()
        spelling.startsWith("volatile ") -> spelling.removePrefix("volatile ").trimStart()
        spelling.endsWith(" const") -> spelling.removeSuffix(" const").trimEnd()
        spelling.endsWith(" volatile") -> spelling.removeSuffix(" volatile").trimEnd()
        spelling.endsWith("&&") -> spelling.removeSuffix("&&").trimEnd()
        spelling.endsWith("&") -> spelling.removeSuffix("&").trimEnd()
        receiverOperator == "->" && spelling.endsWith("*") ->
          spelling.removeSuffix("*").trimEnd()
        else -> spelling
      }
      if (unqualified == spelling) return spelling
      spelling = unqualified
    }
  }
  val legacyReceiverSpellings = listOf(baseType, canonicalBaseType)
    .mapNotNull(::legacyObjectSpelling).toSet()
  val receiverMembers = members.filter { member ->
    val ownerId = member.ownerTypeInfo.cppSemanticId()
    when {
      ownerId != null && receiverTypeIds.isNotEmpty() -> ownerId in receiverTypeIds
      schemaVersion == 1 -> listOf(member.ownerType, member.canonicalOwnerType)
        .mapNotNull(::legacyObjectSpelling).any { it in legacyReceiverSpellings }
      else -> false
    }
  }
  val receiver = receiverOperator?.let { operator ->
    CppReceiver(
      operator = operator,
      expression = cppReceiverExpression(snapshot.semanticPrefixText, operator).orEmpty(),
      type = receiverType,
      members = receiverMembers
    )
  }
  val preferredTypes = listOfNotNull(
    semanticContext?.preferredType as? String,
    semanticContext?.canonicalPreferredType as? String
  ).toSet()
  fun CppTypeInfo?.concreteSpellings(vararg spellings: String?): Sequence<String> =
    if (this?.isSourceSpellable == true && !isDependent && !isInstantiationDependent)
      spellings.asSequence().filterNotNull()
    else emptySequence()
  fun CppReference.semanticSpellings(): Sequence<String> =
    sequenceOf(name, qualifiedName).filterNotNull() +
      typeInfo.concreteSpellings(type, canonicalType) +
      returnTypeInfo.concreteSpellings(returnType, canonicalReturnType) +
      ownerTypeInfo.concreteSpellings(ownerType, canonicalOwnerType) +
      parameters.asSequence().flatMap { parameter ->
        parameter.typeInfo.concreteSpellings(parameter.type, parameter.canonicalType)
      }
  val declarationItemSpellings = items.asSequence()
    .filter { item -> cppDynamicList(item?.symbols).isNotEmpty() }
    .flatMap { item -> sequenceOf(
      item?.name as? String,
      item?.insertText as? String
    ).filterNotNull() }
  val identifiers = (
    references.asSequence().flatMap { it.semanticSpellings() } + declarationItemSpellings
  )
    .flatMap { CPP_IDENTIFIER_REGEX.findAll(it).map(MatchResult::value) }
    .toCollection(linkedSetOf())
  return cppCompletionContextToDto(CppCompletionContext(
    identifiers = identifiers,
    sourceIdentifiers = references.filter { it.provenance?.contains("sema") == true }
      .flatMapTo(linkedSetOf()) { CPP_IDENTIFIER_REGEX.findAll(it.name).map(MatchResult::value).toList() },
    typeNames = types.mapTo(linkedSetOf()) { it.name },
    values = values,
    types = types,
    functions = functions,
    completions = references,
    expectedTypes = preferredTypes,
    enclosingReturnType = semanticContext?.enclosingReturnType as? String,
    canonicalEnclosingReturnType = semanticContext?.canonicalEnclosingReturnType as? String,
    enclosingReturnTypeInfo = cppTypeInfo(semanticContext?.enclosingReturnTypeInfo),
    enclosingClassType = semanticContext?.enclosingClassType as? String,
    canonicalEnclosingClassType = semanticContext?.canonicalEnclosingClassType as? String,
    enclosingClassTypeInfo = cppTypeInfo(semanticContext?.enclosingClassTypeInfo),
    thisType = semanticContext?.thisType as? String,
    canonicalThisType = semanticContext?.canonicalThisType as? String,
    thisTypeInfo = cppTypeInfo(semanticContext?.thisTypeInfo),
    completionKind = semanticContext?.kind as? String,
    preferredType = semanticContext?.preferredType as? String,
    canonicalPreferredType = semanticContext?.canonicalPreferredType as? String,
    preferredTypeInfo = cppTypeInfo(semanticContext?.preferredTypeInfo),
    baseType = baseType,
    canonicalBaseType = canonicalBaseType,
    baseTypeInfo = cppTypeInfo(semanticContext?.baseTypeInfo),
    queryScopes = cppDynamicList(semanticContext?.queryScopes).mapNotNull { it as? String },
    accessibleScopes = cppDynamicList(semanticContext?.accessibleScopes).mapNotNull { it as? String },
    semanticGraphNodeCount = graphNodes.size,
    semanticGraphIsIncomplete = schemaVersion == 2 && (graph?.isIncomplete as? Boolean == true),
    semanticOperationNodeCount = operationNodes.size,
    semanticOperationTemplateCount = operationTemplates.size,
    semanticOperationsAreIncomplete = schemaVersion == 2 && (
      operations?.isIncomplete as? Boolean == true ||
        operations?.nodesIncomplete as? Boolean == true ||
        operations?.templatesIncomplete as? Boolean == true ||
        operations?.conversionsIncomplete as? Boolean == true
      ),
    semanticExpressionWitnessesAreIncomplete = schemaVersion == 2 &&
      (operations?.expressionWitnessesIncomplete as? Boolean == true),
    semanticCallWitnessesAreIncomplete = schemaVersion == 2 &&
      (operations?.callWitnessesIncomplete as? Boolean == true),
    semanticBinaryOperatorWitnessesAreIncomplete = schemaVersion == 2 &&
      (operations?.binaryOperatorWitnessesIncomplete as? Boolean == true),
    receiver = receiver,
    conversions = operationConversions,
    expressionWitnesses = operationExpressionWitnesses,
    callWitnesses = operationCallWitnesses,
    binaryOperatorWitnesses = operationBinaryOperatorWitnesses,
    membersByType = members.groupBy { it.ownerType!! }.map { (type, owned) ->
      CppTypeMembers(type, owned)
    }
  ))
}

/** Source spelling of the innermost open direct call (`visit(`, `ns::visit(`, ...). */
private fun cppActiveCallee(prefix: String): String? {
  val tokens = cppLines(prefix).single().tokens
  val opens = mutableListOf<Int>()
  tokens.forEachIndexed { index, token -> when (token.text) {
    "(" -> opens += index
    ")" -> if (opens.isNotEmpty()) opens.removeAt(opens.lastIndex)
  } }
  var index = opens.lastOrNull()?.minus(1) ?: return null
  if (tokens.getOrNull(index)?.kind != CppTokenKind.IDENTIFIER) return null
  val end = index
  while (index >= 2 && tokens[index - 1].text == "::" &&
    tokens[index - 2].kind == CppTokenKind.IDENTIFIER) index -= 2
  return tokens.subList(index, end + 1).joinToString("") { it.text }
}

private fun CppTokenKind.isCppLiteral(): Boolean = when (this) {
  CppTokenKind.INTEGER,
  CppTokenKind.FLOATING,
  CppTokenKind.CHARACTER,
  CppTokenKind.STRING,
  CppTokenKind.USER_DEFINED_INTEGER,
  CppTokenKind.USER_DEFINED_FLOATING,
  CppTokenKind.USER_DEFINED_CHARACTER,
  CppTokenKind.USER_DEFINED_STRING -> true
  else -> false
}

private fun CppToken.isCppWordFragment(): Boolean =
  !kind.isCppLiteral() && text.matches(CPP_IDENTIFIER_REGEX)

/** Fully lexed punctuators are committed; only names and true maximal-munch fragments are replaced. */
private fun CppToken.isCppCompletionFragment(): Boolean =
  kind == CppTokenKind.IDENTIFIER || completeText != null

private data class CppStatementScope(
  var startCharacter: Int,
  val roundBaseline: Int,
  val squareBaseline: Int,
  val expressionBraceBaseline: Int
)

private data class CppBraceScope(
  val statementScope: Boolean,
  val advancesParentStatement: Boolean = statementScope
)

/** Finds the lexical statement boundary at the cursor's current compound-statement depth. */
private fun cppStatementStartCharacter(tokens: List<CppToken>, character: Int): Int {
  val scopes = mutableListOf(CppStatementScope(0, 0, 0, 0))
  val braces = mutableListOf<CppBraceScope>()
  var round = 0
  var square = 0
  var expressionBraces = 0
  tokens.forEachIndexed { index, token ->
    if (token.end > character) return@forEachIndexed
    when (token.text) {
      "(" -> round++
      ")" -> round = maxOf(0, round - 1)
      "[" -> square++
      "]" -> square = maxOf(0, square - 1)
      "{" -> {
        val braceScope = cppBraceScope(tokens, index)
        braces += braceScope
        if (braceScope.statementScope) {
          scopes += CppStatementScope(token.end, round, square, expressionBraces)
        } else expressionBraces++
      }
      "}" -> if (braces.isNotEmpty()) {
        val braceScope = braces.removeAt(braces.lastIndex)
        if (braceScope.statementScope) {
          if (scopes.size > 1) scopes.removeAt(scopes.lastIndex)
          if (braceScope.advancesParentStatement) scopes.last().startCharacter = token.end
        } else expressionBraces = maxOf(0, expressionBraces - 1)
      }
      ";" -> scopes.last().let { scope ->
        if (round == scope.roundBaseline && square == scope.squareBaseline &&
          expressionBraces == scope.expressionBraceBaseline
        ) scope.startCharacter = token.end
      }
    }
  }
  return scopes.last().startCharacter.coerceIn(0, character)
}

/** Distinguishes compound/lambda bodies from braced initializers for statement segmentation. */
private fun cppBraceScope(tokens: List<CppToken>, braceIndex: Int): CppBraceScope {
  val previous = tokens.getOrNull(braceIndex - 1)?.text ?: return CppBraceScope(true)
  val boundary = (braceIndex - 1 downTo 0).firstOrNull { index ->
    tokens[index].text in setOf(";", "{", "}")
  } ?: -1
  val clause = tokens.subList(boundary + 1, braceIndex).map(CppToken::text)
  val lambdaBody = previous == "]" ||
    ("[" in clause && "]" in clause && "operator" !in clause)
  return when {
    lambdaBody -> CppBraceScope(statementScope = true, advancesParentStatement = false)
    previous in setOf(")", ";", "{", "}", "else", "do", "try", "catch") ->
      CppBraceScope(true)
    ")" in clause -> CppBraceScope(true)
    clause.any { it in setOf("class", "struct", "union", "namespace", "enum") } ->
      CppBraceScope(true)
    clause.firstOrNull() in setOf("if", "for", "while", "switch") -> CppBraceScope(true)
    else -> CppBraceScope(false)
  }
}

/**
 * Replaces only the selected statement's remaining text. The next statement, a surrounding block
 * close, and trailing comments stay outside Monaco's edit range.
 */
private fun cppStatementReplacementEndCharacter(
  lineText: String,
  lineStartOffset: Int,
  lineTokens: List<CppToken>,
  comments: List<CppCommentSpan>,
  statementStartCharacter: Int,
  character: Int
): Int {
  val trailingComment = comments.asSequence().filter { span ->
    val localStart = span.start - lineStartOffset
    val localEnd = span.end - lineStartOffset
    localStart >= character && lineTokens.none { token -> token.start >= localEnd }
  }.map { span ->
    var start = span.start - lineStartOffset
    while (start > character && lineText[start - 1].isWhitespace()) start--
    start
  }.minOrNull() ?: lineText.length

  var round = 0
  var square = 0
  var brace = 0
  fun consume(token: CppToken) {
    when (token.text) {
      "(" -> round++
      ")" -> round = maxOf(0, round - 1)
      "[" -> square++
      "]" -> square = maxOf(0, square - 1)
      "{" -> brace++
      "}" -> brace = maxOf(0, brace - 1)
    }
  }
  lineTokens.asSequence()
    .filter { it.start >= statementStartCharacter && it.end <= character }
    .forEach(::consume)
  lineTokens.asSequence().filter { it.start >= character }.forEach { token ->
    if (token.text == "}" && brace == 0)
      return minOf(token.start, trailingComment).coerceAtLeast(character)
    if (token.text == ";" && round == 0 && square == 0 && brace == 0)
      return minOf(token.end, trailingComment).coerceAtLeast(character)
    consume(token)
  }
  return trailingComment.coerceIn(character, lineText.length)
}

private fun cppPhysicalLine(source: String, requestedLine: Int): CppPhysicalLine? {
  var line = 0
  var start = 0
  while (true) {
    val newline = source.indexOf('\n', start)
    val rawEnd = if (newline < 0) source.length else newline
    val contentEnd = if (rawEnd > start && source[rawEnd - 1] == '\r') rawEnd - 1 else rawEnd
    if (line == requestedLine)
      return CppPhysicalLine(start, contentEnd, source.substring(start, contentEnd))
    if (newline < 0) return null
    start = newline + 1
    line++
  }
}

/** Includes continuation lines belonging to a backslash-continued preprocessor directive. */
private fun cppIsPreprocessorLine(source: String, requestedLine: Int): Boolean {
  var line = 0
  var start = 0
  var continuing = false
  while (true) {
    val newline = source.indexOf('\n', start)
    val rawEnd = if (newline < 0) source.length else newline
    val contentEnd = if (rawEnd > start && source[rawEnd - 1] == '\r') rawEnd - 1 else rawEnd
    val text = source.substring(start, contentEnd)
    val directive = continuing || text.trimStart().startsWith('#')
    if (line == requestedLine) return directive
    continuing = directive && text.trimEnd().let { trimmed ->
      var slashes = 0
      var index = trimmed.lastIndex
      while (index >= 0 && trimmed[index--] == '\\') slashes++
      slashes % 2 == 1
    }
    if (newline < 0) return false
    start = newline + 1
    line++
  }
}

/** Finds only comment spans intersecting the requested line while respecting quoted literals. */
private fun cppCommentSpans(source: String, lineStart: Int, lineEnd: Int): List<CppCommentSpan> {
  val result = mutableListOf<CppCommentSpan>()
  var index = 0
  while (index < lineEnd) {
    when {
      source.startsWith("//", index) -> {
        val end = source.indexOf('\n', index + 2).let { if (it < 0) source.length else it }
        if (end >= lineStart && index <= lineEnd)
          result += CppCommentSpan(maxOf(index, lineStart), minOf(end, lineEnd), true)
        index = end
      }
      source.startsWith("/*", index) -> {
        val close = source.indexOf("*/", index + 2)
        val end = if (close < 0) source.length else close + 2
        if (end >= lineStart && index <= lineEnd)
          result += CppCommentSpan(
            maxOf(index, lineStart),
            minOf(end, lineEnd),
            inclusiveEnd = close < 0 || end > lineEnd
          )
        index = end
      }
      else -> {
        val raw = cppRawStringAt(source, index)
        if (raw != null) {
          index = raw
        } else if (source[index] == '"' || source[index] == '\'') {
          val quote = source[index++]
          while (index < source.length) {
            when {
              source[index] == '\\' -> index = (index + 2).coerceAtMost(source.length)
              source[index] == quote -> { index++; break }
              else -> index++
            }
          }
        } else index++
      }
    }
  }
  return result
}

/** Returns the end of a C++ raw string beginning at [index], or null when none begins there. */
private fun cppRawStringAt(source: String, index: Int): Int? {
  val prefixes = arrayOf("u8R\"", "uR\"", "UR\"", "LR\"", "R\"")
  val prefix = prefixes.firstOrNull { source.startsWith(it, index) } ?: return null
  val delimiterStart = index + prefix.length
  val open = source.indexOf('(', delimiterStart)
  if (open < 0 || open - delimiterStart > 16) return null
  val delimiter = source.substring(delimiterStart, open)
  if (delimiter.any { it.isWhitespace() || it == '\\' || it == '(' || it == ')' }) return null
  val terminator = ")$delimiter\""
  val close = source.indexOf(terminator, open + 1)
  return if (close < 0) source.length else close + terminator.length
}

/**
 * Builds a structured-clone-safe context DTO from the source and optional clangd facts.
 * Kotlin collections and data classes never cross the worker boundary.
 */
fun cppCompletionContextDto(
  source: String,
  completionGroups: List<CppClangdCompletionGroup> = emptyList(),
  signatures: dynamic = null,
  hover: dynamic = null,
  diagnostics: dynamic = null,
  ast: dynamic = null,
  snapshot: CppEditorStatementSnapshot? = null,
  lexicalFacts: CppLexicalSourceFacts? = null
): dynamic {
  val lexical = lexicalFacts ?: cppLexicalSourceFacts(source)
  val completionReferences = cppDistinctReferences(completionGroups.flatMap { group ->
    cppCompletionItems(group.result).mapNotNull { item ->
      cppReferenceFromCompletion(item, group.receiverMember)
    }
  }).filterNot { reference ->
    // A method is never a free function. clangd may report a partially typed member at an
    // ordinary completion cursor as one collapsed Method item (for example `[4 overloads]` with
    // an ellipsis signature). Without receiver/implicit-this evidence that item is not a usable
    // callable fact; the scoped AST and receiver member tables remain authoritative.
    reference.kind == "method" && !reference.receiverMember
  }
  val astContext = cppContextFromNormalizedAst(ast)
  val baseCompletions = completionReferences.filterNot(CppReference::receiverMember)
  val values = cppDistinctReferences(
    astContext.values + baseCompletions.filter { reference ->
      reference.kind in CPP_VALUE_REFERENCE_KINDS &&
        reference.name.substringAfterLast("::") in lexical.identifiers
    }
  )
  val types = cppDistinctReferences(
    lexical.types + astContext.types + baseCompletions.filter { it.kind in CPP_TYPE_REFERENCE_KINDS }
  )
  val functions = cppDistinctReferences(
    astContext.functions + baseCompletions.filter { it.kind in CPP_FUNCTION_REFERENCE_KINDS }
  )
  val hoverFacts = cppHoverFacts(hover)
  val normalizedSignatures = cppSignatures(signatures, hoverFacts)
  val expectedTypes = normalizedSignatures.mapNotNullTo(linkedSetOf()) { signature ->
    signature.activeParameter?.let(signature.parameters::getOrNull)?.type?.takeIf(String::isNotBlank)
  }
  val memberGroups = completionGroups.filter(CppClangdCompletionGroup::receiverMember)
  val receiverMembers = completionReferences.filter(CppReference::receiverMember)
  val receiverOperator = memberGroups.firstNotNullOfOrNull(CppClangdCompletionGroup::receiverOperator)
    ?: cppReceiverOperator(snapshot?.semanticPrefixText)
  val receiverExpression = cppReceiverExpression(snapshot?.semanticPrefixText, receiverOperator)
  val receiverType = hoverFacts.type
    ?: values.firstOrNull { it.name == receiverExpression }?.type
  val ownedMembers = receiverMembers.map { member ->
    if (member.ownerType != null || receiverType.isNullOrBlank()) member
    else member.copy(ownerType = receiverType)
  }
  val receiver = receiverOperator?.let { operator ->
    CppReceiver(operator, receiverExpression.orEmpty(), receiverType, ownedMembers)
  }
  val memberTables = cppDistinctTypeMembers(
    astContext.membersByType + listOfNotNull(
      receiverType?.takeIf { ownedMembers.isNotEmpty() }?.let { CppTypeMembers(it, ownedMembers) }
    )
  )
  val unresolved = cppUnresolvedIdentifiers(source, diagnostics, snapshot)
  val identifiers = linkedSetOf<String>().apply {
    addAll(lexical.identifiers)
    addAll(lexical.typeNames)
    (completionReferences + values + types + functions).forEach { reference ->
      addAll(CPP_IDENTIFIER_REGEX.findAll(reference.name).map(MatchResult::value))
    }
  }
  return cppCompletionContextToDto(CppCompletionContext(
    identifiers = identifiers,
    sourceIdentifiers = lexical.identifiers,
    headers = lexical.headers,
    typeNames = lexical.typeNames,
    values = values,
    types = types,
    functions = functions,
    completions = completionReferences,
    signatures = normalizedSignatures,
    expectedTypes = expectedTypes,
    receiver = receiver,
    membersByType = memberTables,
    conversions = astContext.conversions,
    unresolvedIdentifiers = unresolved,
    requiredIdentifier = unresolved.firstOrNull(),
    requiredTypes = astContext.requiredTypes,
    probedRequiredTypes = astContext.probedRequiredTypes,
    defaultConstructibleTypes = astContext.defaultConstructibleTypes,
    enclosingReturnType = astContext.enclosingReturnType,
    enclosingClassType = astContext.enclosingClassType,
    thisType = astContext.thisType,
    mutableFields = astContext.mutableFields
  ))
}

/** Rehydrates the worker-local immutable context from a plain structured clone. */
fun cppCompletionContextFromDto(dto: dynamic): CppCompletionContext {
  if (!cppDefined(dto)) return CppCompletionContext(emptySet())
  return CppCompletionContext(
    identifiers = cppStringSet(dto.identifiers),
    sourceIdentifiers = cppStringSet(dto.sourceIdentifiers),
    headers = cppStringSet(dto.headers),
    typeNames = cppStringSet(dto.typeNames),
    values = cppDynamicList(dto.values).mapNotNull(::cppReferenceFromDto),
    types = cppDynamicList(dto.types).mapNotNull(::cppReferenceFromDto),
    functions = cppDynamicList(dto.functions).mapNotNull(::cppReferenceFromDto),
    completions = cppDynamicList(dto.completions).mapNotNull(::cppReferenceFromDto),
    signatures = cppDynamicList(dto.signatures).mapNotNull(::cppSignatureFromDto),
    expectedTypes = cppStringSet(dto.expectedTypes),
    receiver = cppReceiverFromDto(dto.receiver),
    membersByType = cppDynamicList(dto.membersByType).mapNotNull(::cppTypeMembersFromDto),
    conversions = cppDynamicList(dto.conversions).mapNotNull(::cppConversionFromDto),
    expressionWitnesses = cppDynamicList(dto.expressionWitnesses)
      .mapNotNull(::cppExpressionWitnessFromDto),
    callWitnesses = cppDynamicList(dto.callWitnesses).mapNotNull(::cppCallWitnessFromDto),
    binaryOperatorWitnesses = cppDynamicList(dto.binaryOperatorWitnesses)
      .mapNotNull(::cppBinaryOperatorWitnessFromDto),
    unresolvedIdentifiers = cppStringSet(dto.unresolvedIdentifiers),
    requiredBinderObligation = cppRequiredBinderObligationFromDto(dto.requiredBinderObligation),
    requiredIdentifier = dto.requiredIdentifier as? String,
    requiredTypes = cppStringSet(dto.requiredTypes),
    probedRequiredTypes = cppStringSet(dto.probedRequiredTypes),
    defaultConstructibleTypes = cppStringSet(dto.defaultConstructibleTypes),
    enclosingReturnType = dto.enclosingReturnType as? String,
    canonicalEnclosingReturnType = dto.canonicalEnclosingReturnType as? String,
    enclosingReturnTypeInfo = cppTypeInfo(dto.enclosingReturnTypeInfo),
    enclosingClassType = dto.enclosingClassType as? String,
    canonicalEnclosingClassType = dto.canonicalEnclosingClassType as? String,
    enclosingClassTypeInfo = cppTypeInfo(dto.enclosingClassTypeInfo),
    thisType = dto.thisType as? String,
    canonicalThisType = dto.canonicalThisType as? String,
    thisTypeInfo = cppTypeInfo(dto.thisTypeInfo),
    mutableFields = cppStringSet(dto.mutableFields),
    completionKind = dto.completionKind as? String,
    preferredType = dto.preferredType as? String,
    canonicalPreferredType = dto.canonicalPreferredType as? String,
    preferredTypeInfo = cppTypeInfo(dto.preferredTypeInfo),
    baseType = dto.baseType as? String,
    canonicalBaseType = dto.canonicalBaseType as? String,
    baseTypeInfo = cppTypeInfo(dto.baseTypeInfo),
    queryScopes = cppDynamicList(dto.queryScopes).mapNotNull { it as? String },
    accessibleScopes = cppDynamicList(dto.accessibleScopes).mapNotNull { it as? String },
    semanticGraphNodeCount = cppInt(dto.semanticGraphNodeCount),
    semanticGraphIsIncomplete = dto.semanticGraphIsIncomplete as? Boolean ?: false,
    semanticOperationNodeCount = cppInt(dto.semanticOperationNodeCount),
    semanticOperationTemplateCount = cppInt(dto.semanticOperationTemplateCount),
    semanticOperationsAreIncomplete = dto.semanticOperationsAreIncomplete as? Boolean ?: false,
    semanticExpressionWitnessesAreIncomplete =
      dto.semanticExpressionWitnessesAreIncomplete as? Boolean ?: false,
    semanticCallWitnessesAreIncomplete =
      dto.semanticCallWitnessesAreIncomplete as? Boolean ?: false,
    semanticBinaryOperatorWitnessesAreIncomplete =
      dto.semanticBinaryOperatorWitnessesAreIncomplete as? Boolean ?: false
  )
}

data class CppLexicalSourceFacts(
  val identifiers: Set<String>,
  val headers: Set<String>,
  val typeNames: Set<String>,
  val types: List<CppReference>
)

private val CPP_IDENTIFIER_REGEX = Regex("[A-Za-z_][A-Za-z_0-9]*")
private val CPP_BUILTIN_TYPE_WORDS = setOf(
  "void", "bool", "char", "wchar_t", "char8_t", "char16_t", "char32_t", "short",
  "int", "long", "float", "double", "signed", "unsigned", "auto"
)
private val CPP_VALUE_REFERENCE_KINDS = setOf("constant", "enumMember", "field", "property", "value", "variable")
private val CPP_FUNCTION_REFERENCE_KINDS = setOf("constructor", "function", "method", "operator")
private val CPP_TYPE_REFERENCE_KINDS = setOf("class", "concept", "enum", "struct", "typeAlias", "typeParameter")

fun cppLexicalSourceFacts(source: String): CppLexicalSourceFacts {
  val directiveRanges = cppDirectiveRanges(source)
  var directiveIndex = 0
  val tokens = lexCppTokenSpans(source).filterNot { token ->
    while (directiveRanges.getOrNull(directiveIndex)?.second?.let { it <= token.startIndex } == true)
      directiveIndex++
    directiveRanges.getOrNull(directiveIndex)?.let { range ->
      token.startIndex >= range.first && token.startIndex < range.second
    } ?: false
  }
  val identifiers = tokens.filter { it.type == "Identifier" }
    .mapTo(linkedSetOf()) { it.text }
  val typeNames = linkedSetOf<String>()
  val declaredTypes = mutableListOf<CppReference>()
  tokens.forEachIndexed { index, token ->
    val candidate = when (token.text) {
      "class", "struct", "union", "namespace" -> tokens.getOrNull(index + 1)
      "enum" -> tokens.getOrNull(index + 1)?.let { next ->
        if (next.text == "class" || next.text == "struct") tokens.getOrNull(index + 2) else next
      }
      "using" -> tokens.getOrNull(index + 1)?.takeIf { next -> tokens.getOrNull(index + 2)?.text == "=" }
      else -> null
    }
    if (candidate?.type == "Identifier") {
      typeNames += candidate.text
      val kind = when (token.text) {
        "struct" -> "struct"
        "class", "union" -> "class"
        "enum" -> "enum"
        "using" -> "typeAlias"
        else -> null
      }
      if (kind != null && kind != "typeAlias")
        declaredTypes += CppReference(candidate.text, kind = kind, source = "source")
    }
  }
  var declarationStart = 0
  tokens.forEachIndexed { index, token ->
    if (token.text == ";") {
      val declaration = tokens.subList(declarationStart, index)
      if (declaration.firstOrNull()?.text == "typedef") {
        declaration.lastOrNull { it.type == "Identifier" }?.text?.let { name ->
          typeNames += name
          declaredTypes += CppReference(name, kind = "typeAlias", source = "source")
        }
      }
      declarationStart = index + 1
    }
  }
  tokens.filter { it.text in CPP_BUILTIN_TYPE_WORDS }.mapTo(typeNames) { it.text }

  val headers = Regex(
    "^\\s*#\\s*include(?:_next)?\\s*[<\"]([^>\"]+)[>\"]",
    RegexOption.MULTILINE
  )
    .findAll(source).mapTo(linkedSetOf()) { it.groupValues[1] }
  val aliases = Regex("\\busing\\s+([A-Za-z_][A-Za-z_0-9]*)\\s*=\\s*([^;{}\\r\\n]+)\\s*;")
    .findAll(source)
    .mapNotNull { match ->
      val name = match.groupValues[1]
      if (name !in typeNames) null else match.groupValues[2].trim().let { target ->
        CppReference(name, type = target, kind = "typeAlias", detail = target, source = "source")
      }
    }.toList()
  return CppLexicalSourceFacts(
    identifiers,
    headers,
    typeNames,
    cppDistinctReferences(declaredTypes.filterNot { declaration ->
      declaration.kind == "typeAlias" && aliases.any { it.name == declaration.name }
    } + aliases)
  )
}

private fun cppDirectiveRanges(source: String): List<Pair<Int, Int>> {
  val result = mutableListOf<Pair<Int, Int>>()
  var start = 0
  while (start <= source.length) {
    val newline = source.indexOf('\n', start)
    val end = if (newline < 0) source.length else newline
    if (source.substring(start, end).trimStart().startsWith('#')) result += start to end
    if (newline < 0) break
    start = newline + 1
  }
  return result
}

private fun cppCompletionItems(result: dynamic): List<dynamic> = when {
  !cppDefined(result) -> emptyList()
  cppIsArray(result) -> cppDynamicList(result)
  else -> cppDynamicList(result.items)
}

private fun cppReferenceFromCompletion(item: dynamic, receiverMember: Boolean): CppReference? {
  if (!cppDefined(item)) return null
  val label = (if (item.label is String) item.label else item.label?.label) as? String
  val name = cppCompletionSemanticName(item, label) ?: return null
  val detail = (item.detail as? String)?.trim()?.takeIf(String::isNotEmpty)
  val kind = cppReferenceKind(cppInt(item.kind))
  val parameters = if (kind in CPP_FUNCTION_REFERENCE_KINDS) {
    val signatureDetail = item.labelDetails?.detail as? String
    cppParameterClause(signatureDetail ?: label ?: name)
  } else emptyList()
  return CppReference(
    name = name,
    type = detail.takeIf { kind in CPP_VALUE_REFERENCE_KINDS },
    returnType = when {
      kind == "constructor" -> name
      kind in CPP_FUNCTION_REFERENCE_KINDS -> detail
      else -> null
    },
    parameters = parameters,
    kind = kind,
    detail = detail,
    receiverMember = receiverMember,
    source = "completion"
  )
}

private fun cppReferenceKind(kind: Int): String = when (kind) {
  2 -> "method"
  3 -> "function"
  4 -> "constructor"
  5 -> "field"
  6 -> "variable"
  7 -> "class"
  8 -> "typeAlias"
  9 -> "namespace"
  10 -> "property"
  12 -> "value"
  13 -> "enum"
  20 -> "enumMember"
  21 -> "constant"
  22 -> "struct"
  24 -> "operator"
  25 -> "typeParameter"
  else -> "unknown"
}

/** Uses display/filter names as semantic facts and treats edit/snippet text as a last resort. */
private fun cppCompletionSemanticName(item: dynamic, label: String?): String? =
  listOf<dynamic>(label, item.filterText, item.textEdit?.newText, item.insertText)
    .mapNotNull { it as? String }
    .mapNotNull(::cppNormalizeCompletionName)
    .firstOrNull()

private fun cppNormalizeCompletionName(candidate: String): String? {
  if ('\n' in candidate || '\r' in candidate) return null
  val cleaned = cppStripSnippetSyntax(candidate).trim()
  if (cleaned.isEmpty()) return null
  if (cleaned.startsWith("operator")) {
    val operatorName = when {
      cleaned.startsWith("operator()") -> "operator()"
      cleaned.startsWith("operator[]") -> "operator[]"
      else -> cleaned.substringBefore('(').substringBefore(" -> ").trim()
    }
    return operatorName.takeIf(String::isNotEmpty)
  }
  return Regex("^(?:(?:[A-Za-z_][A-Za-z_0-9]*)::)*(?:~?[A-Za-z_][A-Za-z_0-9]*)")
    .find(cleaned)?.value
}

/** Minimal TextMate-snippet decoding; enough to prevent placeholders from becoming C++ names. */
private fun cppStripSnippetSyntax(snippet: String): String = buildString {
  var index = 0
  while (index < snippet.length) {
    when {
      snippet[index] == '\\' && snippet.getOrNull(index + 1) == '$' -> {
        append('$')
        index += 2
      }
      snippet[index] == '$' && snippet.getOrNull(index + 1) == '{' -> {
        val close = snippet.indexOf('}', index + 2)
        if (close < 0) {
          index++
          continue
        }
        val placeholder = snippet.substring(index + 2, close)
        val default = placeholder.substringAfter(':', "")
          .ifEmpty {
            placeholder.substringAfter('|', "").substringBefore(',').substringBefore('|')
          }
        append(default)
        index = close + 1
      }
      snippet[index] == '$' && snippet.getOrNull(index + 1)?.isDigit() == true -> {
        index += 2
        while (snippet.getOrNull(index)?.isDigit() == true) index++
      }
      else -> append(snippet[index++])
    }
  }
}

private data class CppHoverFacts(val type: String? = null, val returnType: String? = null)

private fun cppHoverFacts(hover: dynamic): CppHoverFacts {
  if (!cppDefined(hover)) return CppHoverFacts()
  val contents = hover.contents
  val text = when {
    contents is String -> contents
    cppDefined(contents?.value) -> contents.value as? String ?: ""
    cppIsArray(contents) -> cppDynamicList(contents).joinToString("\n") { item ->
      item as? String ?: item?.value as? String ?: ""
    }
    else -> ""
  }
  val type = Regex("^Type:\\s*(.+)$", RegexOption.MULTILINE)
    .find(text)?.groupValues?.get(1)?.trim()?.trim('`')
  val returnType = Regex("^[→]\\s*(.+)$", RegexOption.MULTILINE)
    .find(text)?.groupValues?.get(1)?.trim()?.trim('`')
  return CppHoverFacts(type, returnType)
}

private fun cppSignatures(result: dynamic, hover: CppHoverFacts): List<CppSignature> {
  if (!cppDefined(result)) return emptyList()
  val raw = cppDynamicList(result.signatures)
  val activeSignature = cppInt(result.activeSignature).coerceIn(0, maxOf(0, raw.lastIndex))
  val activeParameter = cppInt(result.activeParameter)
  return raw.mapIndexedNotNull { index, signature ->
    val label = signature?.label as? String ?: return@mapIndexedNotNull null
    val parameters = cppDynamicList(signature.parameters).mapNotNull { parameter ->
      val parameterLabel = parameter?.label
      val rawLabel = when {
        parameterLabel is String -> parameterLabel
        cppIsArray(parameterLabel) -> {
          val bounds = cppDynamicList(parameterLabel)
          val start = cppInt(bounds.getOrNull(0)).coerceIn(0, label.length)
          val end = cppInt(bounds.getOrNull(1)).coerceIn(start, label.length)
          label.substring(start, end)
        }
        else -> return@mapNotNull null
      }
      cppParameterFromLabel(rawLabel)
    }
    val arrow = label.lastIndexOf("->")
    CppSignature(
      label = label,
      returnType = if (index == activeSignature) hover.returnType
        ?: label.substring((arrow + 2).coerceAtMost(label.length)).trim().takeIf { arrow >= 0 }
      else label.substring((arrow + 2).coerceAtMost(label.length)).trim().takeIf { arrow >= 0 },
      parameters = parameters,
      activeParameter = if (parameters.isEmpty()) null else activeParameter.coerceIn(0, parameters.lastIndex)
    )
  }
}

private fun cppParameterClause(text: String): List<CppParameter> {
  val open = text.indexOf('(')
  if (open < 0) return emptyList()
  var depth = 0
  for (index in open until text.length) {
    if (text[index] == '(') depth++
    if (text[index] == ')' && --depth == 0)
      return cppSplitTopLevel(text.substring(open + 1, index)).map(::cppParameterFromLabel)
  }
  return emptyList()
}

private fun cppSplitTopLevel(text: String, separator: Char = ','): List<String> {
  val result = mutableListOf<String>()
  var start = 0
  var angle = 0
  var round = 0
  var square = 0
  var brace = 0
  text.forEachIndexed { index, character ->
    when (character) {
      '<' -> angle++
      '>' -> angle = maxOf(0, angle - 1)
      '(' -> round++
      ')' -> round = maxOf(0, round - 1)
      '[' -> square++
      ']' -> square = maxOf(0, square - 1)
      '{' -> brace++
      '}' -> brace = maxOf(0, brace - 1)
      separator -> if (angle + round + square + brace == 0) {
        result += text.substring(start, index).trim()
        start = index + 1
      }
    }
  }
  result += text.substring(start).trim()
  return result.filter { it.isNotEmpty() && it != "void" }
}

private fun cppParameterFromLabel(label: String): CppParameter {
  val raw = label.trim()
  val equalParts = cppSplitTopLevel(raw, '=')
  val declaration = equalParts.firstOrNull() ?: raw
  val nameMatch = Regex("(?:^|[\\s*&])([A-Za-z_][A-Za-z_0-9]*)\\s*(\\[[^\\]]*\\])?\\s*$")
    .find(declaration)
  val name = nameMatch?.groupValues?.get(1).orEmpty()
  val nameOffset = nameMatch?.let { match -> match.range.first + match.value.lastIndexOf(name) } ?: -1
  val prefix = if (nameOffset < 0) "" else declaration.substring(0, nameOffset).trim()
  val hasSeparateName = name.isNotEmpty() && prefix.isNotEmpty()
  val arraySuffix = nameMatch?.groupValues?.getOrNull(2).orEmpty()
  return CppParameter(
    label = raw,
    name = name.takeIf { hasSeparateName }.orEmpty(),
    type = (if (hasSeparateName) prefix + arraySuffix else declaration).trim(),
    defaultValue = raw.substringAfter('=', "").trim().takeIf { '=' in raw }
  )
}

private fun cppReceiverOperator(prefix: String?): String? = prefix?.trimEnd()?.let {
  when {
    it.endsWith("->") -> "->"
    it.endsWith("::") -> "::"
    it.endsWith('.') -> "."
    else -> null
  }
}

private fun cppReceiverExpression(prefix: String?, operator: String?): String? {
  if (prefix == null || operator == null) return null
  val before = prefix.trimEnd().removeSuffix(operator).trimEnd()
  return Regex("(?:[A-Za-z_][A-Za-z_0-9]*|\\([^()]*\\)|\\[[^\\]]*\\])$")
    .find(before)?.value
}

private fun cppUnresolvedIdentifiers(
  source: String,
  diagnostics: dynamic,
  snapshot: CppEditorStatementSnapshot?
): Set<String> {
  val raw = if (cppIsArray(diagnostics)) cppDynamicList(diagnostics)
  else if (cppDefined(diagnostics)) cppDynamicList(diagnostics.diagnostics) else emptyList()
  return raw.asSequence().filter { diagnostic ->
    val code = diagnostic?.code?.toString().orEmpty()
    val message = diagnostic?.message as? String ?: ""
    (code == "undeclared_var_use" || code == "undeclared_var_use_suggest" ||
      Regex("(?:use of )?undeclared identifier", RegexOption.IGNORE_CASE).containsMatchIn(message)) &&
      (snapshot == null || cppDiagnosticBelongsToStatement(diagnostic, snapshot))
  }.mapNotNull { diagnostic ->
    cppTextAtRange(source, diagnostic?.range) ?: run {
      val message = diagnostic?.message as? String ?: return@mapNotNull null
      Regex("undeclared identifier\\s+['‘]([A-Za-z_][A-Za-z_0-9]*)['’]", RegexOption.IGNORE_CASE)
        .find(message)?.groupValues?.get(1)
    }
  }.filter(CPP_IDENTIFIER_REGEX::matches).toCollection(linkedSetOf())
}

private fun cppDiagnosticBelongsToStatement(diagnostic: dynamic, snapshot: CppEditorStatementSnapshot): Boolean {
  val start = diagnostic?.range?.start
  if (!cppDefined(start) || cppInt(start.line, -1) != snapshot.line) return false
  val character = cppInt(start.character, -1)
  return character in snapshot.statementStartCharacter..snapshot.replacementEndCharacter
}

private fun cppTextAtRange(source: String, range: dynamic): String? {
  if (!cppDefined(range?.start) || !cppDefined(range?.end)) return null
  val startLine = cppInt(range.start.line, -1)
  val endLine = cppInt(range.end.line, -1)
  if (startLine < 0 || startLine != endLine) return null
  val physical = cppPhysicalLine(source, startLine) ?: return null
  val start = cppInt(range.start.character, -1)
  val end = cppInt(range.end.character, -1)
  if (start < 0 || end < start || end > physical.text.length) return null
  return physical.text.substring(start, end).trim()
}

private fun cppDistinctReferences(references: List<CppReference>): List<CppReference> {
  val unique = linkedMapOf<String, CppReference>()
  references.forEach { reference ->
    val ownerId = reference.ownerTypeInfo.cppSemanticId()
    val key = listOf(reference.id, reference.name, reference.kind,
      reference.canonicalType ?: reference.type,
      reference.canonicalReturnType ?: reference.returnType,
      reference.parameters.joinToString(",") { it.canonicalType ?: it.type },
      reference.receiverMember, reference.isMember, reference.isStatic,
      ownerId, reference.ownerType, reference.canonicalOwnerType)
      .joinToString("\u0000")
    val previous = unique[key]
    if (previous == null) {
      unique[key] = reference
    } else {
      val preferred = when {
        previous.provenance == "index" && reference.provenance != "index" -> reference
        reference.activeCallable && !previous.activeCallable -> reference
        else -> previous
      }
      // Route and active-overload evidence can arrive through different slices for the same exact
      // declaration. Deduplication must union those proofs instead of allowing input order to erase
      // one of them.
      unique[key] = preferred.copy(
        completionVisible = previous.completionVisible || reference.completionVisible,
        activeCallable = previous.activeCallable || reference.activeCallable
      )
    }
  }
  return unique.values.toList()
}

/** Implementation-reserved identifiers are not user-facing source spellings. */
private fun cppContainsReservedIdentifier(spelling: String): Boolean =
  Regex("[A-Za-z_][A-Za-z_0-9]*").findAll(spelling).any { match ->
    val identifier = match.value
    "__" in identifier || identifier.length > 1 && identifier[0] == '_' && identifier[1].isUpperCase()
  }

private fun cppDistinctTypeMembers(tables: List<CppTypeMembers>): List<CppTypeMembers> =
  tables.groupBy(CppTypeMembers::type).map { (type, grouped) ->
    CppTypeMembers(type, cppDistinctReferences(grouped.flatMap(CppTypeMembers::members)))
  }

private const val CPP_AST_NODE_LIMIT = 12_000
private const val CPP_AST_DEPTH_LIMIT = 96
private const val CPP_AST_PARAMETER_NODE_LIMIT = 256
internal const val CPP_AST_CONTEXT_REQUEST_FIELD = "__tidyparseCppCompletionAstContext"
internal const val CPP_NORMALIZED_AST_CONTEXT_FIELD = "__tidyparseCppCompletionNormalizedAst"

private val CPP_AST_SCOPE_KINDS = setOf(
  "Compound", "CompoundStmt", "CXXConstructor", "CXXForRange", "CXXMethod",
  "For", "Function", "If", "Lambda", "Switch", "While"
)

private val CPP_AST_METHOD_KINDS = setOf(
  "CXXConstructor", "CXXConversion", "CXXDestructor", "CXXMethod"
)

/**
 * Reduces clangd's raw `textDocument/ast` tree to the small, structured-clone-safe semantic DTO
 * consumed by [cppCompletionContextDto]. Only declarations visible at the requested cursor and
 * declaration metadata needed by the statement grammar are retained.
 *
 * clangd ASTs can be much larger than the edited file because recovery nodes are verbose. The
 * traversal therefore has explicit node/depth ceilings and never copies expression subtrees into
 * the worker request. An incomplete traversal is still conservative: lexical and completion facts
 * remain available to the caller.
 */
fun cppClangdAstContextDto(
  rawAst: dynamic,
  source: String,
  cursorLine: Int,
  cursorCharacter: Int
): dynamic {
  val empty = CppCompletionContext(emptySet())
  if (!cppDefined(rawAst) || cursorLine < 0 || cursorCharacter < 0)
    return cppCompletionContextToDto(empty)
  return try {
    val root = if (cppDefined(rawAst.ast)) rawAst.ast else rawAst
    cppCompletionContextToDto(
      CppClangdAstNormalizer(source, cursorLine, cursorCharacter).normalize(root)
    )
  } catch (_: Throwable) {
    // AST is an optional enrichment; malformed recovery data must not suppress lexical fallback.
    cppCompletionContextToDto(empty)
  }
}

private data class CppNormalizedAstRecord(val name: String, val bases: List<String>, val members: List<CppReference>)

private class CppClangdAstNormalizer(source: String, private val cursorLine: Int, private val cursorCharacter: Int) {
  private val sourceLines = source.replace("\r\n", "\n").replace('\r', '\n').split('\n')
  private val values = mutableListOf<CppReference>()
  private val functions = mutableListOf<CppReference>()
  private val types = mutableListOf<CppReference>()
  private val records = mutableListOf<CppNormalizedAstRecord>()
  private val conversions = mutableListOf<CppConversion>()
  private val defaultConstructibleTypes = linkedSetOf<String>()
  private val mutableFields = linkedSetOf<String>()
  private val ancestors = mutableListOf<dynamic>()
  private var visitedNodes = 0
  private var enclosingReturnType: String? = null
  private var enclosingClassType: String? = null
  private var thisType: String? = null

  fun normalize(root: dynamic): CppCompletionContext {
    visit(root, 0)
    val recordsByName = linkedMapOf<String, CppNormalizedAstRecord>()
    records.forEach { recordsByName[it.name] = it }

    fun inheritedMembers(type: String, seen: MutableSet<String>): List<CppReference> {
      if (!seen.add(type)) return emptyList()
      val record = recordsByName[type] ?: return emptyList()
      return record.members + record.bases.flatMap { base ->
        inheritedMembers(base, seen).filterNot { it.kind == "constructor" }
      }
    }

    val membersByType = recordsByName.values.map { record ->
      CppTypeMembers(record.name, cppDistinctReferences(inheritedMembers(record.name, linkedSetOf())))
    }
    val normalizedValues = cppDistinctReferences(values)
    val normalizedFunctions = cppDistinctReferences(functions)
    val normalizedTypes = cppDistinctReferences(types)
    val identifiers = linkedSetOf<String>()
    (normalizedValues + normalizedFunctions + normalizedTypes + membersByType.flatMap { it.members })
      .forEach { reference ->
        CPP_IDENTIFIER_REGEX.findAll(reference.name).mapTo(identifiers, MatchResult::value)
        reference.parameters.mapNotNullTo(identifiers) { it.name.takeIf(String::isNotBlank) }
      }
    return CppCompletionContext(
      identifiers = identifiers,
      values = normalizedValues,
      types = normalizedTypes,
      functions = normalizedFunctions,
      membersByType = membersByType,
      conversions = conversions.distinct(),
      defaultConstructibleTypes = defaultConstructibleTypes,
      enclosingReturnType = enclosingReturnType,
      enclosingClassType = enclosingClassType,
      thisType = thisType,
      mutableFields = mutableFields
    )
  }

  private fun visit(node: dynamic, depth: Int) {
    if (!cppDefined(node) || jsTypeOf(node) != "object" || depth > CPP_AST_DEPTH_LIMIT ||
      visitedNodes >= CPP_AST_NODE_LIMIT) return
    visitedNodes++

    val kind = node.kind as? String
    val role = node.role as? String
    val name = (node.detail as? String)?.trim()?.takeIf(String::isNotEmpty)
    val declaredBeforeCursor = cppAstPositionAtOrBefore(node.range?.start)

    if (kind == "Enum" && role == "declaration" && name != null && declaredBeforeCursor) {
      types += CppReference(name, type = name, kind = "enum", detail = name, source = "ast")
    }
    if (kind == "EnumConstant" && role == "declaration" && name != null && declaredBeforeCursor) {
      val owner = nearestAncestor { ancestor ->
        ancestor.kind as? String == "Enum" && ancestor.role as? String == "declaration" &&
          (ancestor.detail as? String).isNullOrBlank().not()
      }
      val ownerName = owner?.detail as? String
      if (ownerName != null) values += CppReference(
        name = name,
        type = ownerName,
        kind = "enumMember",
        detail = if (Regex("\\b(?:class|struct|scoped)\\b")
            .containsMatchIn(owner.arcana as? String ?: "")) "scoped" else "unscoped",
        ownerType = ownerName,
        source = "ast"
      )
    }

    if (role == "declaration" && cppAstRangeContainsCursor(node.range)) {
      val enclosingRecord = nearestAncestor { ancestor ->
        ancestor.kind as? String == "CXXRecord" && ancestor.role as? String == "declaration" &&
          (ancestor.detail as? String).isNullOrBlank().not()
      }
      if (kind in CPP_AST_METHOD_KINDS && cppDefined(enclosingRecord)) {
        val recordName = enclosingRecord.detail as String
        enclosingClassType = recordName
        val signature = cppQuotedAstType(node.arcana as? String).orEmpty()
        val methodConst = Regex("\\)\\s+const(?:\\s|$)").containsMatchIn(signature)
        thisType = "${if (methodConst) "const " else ""}$recordName *"
        cppDynamicList(enclosingRecord.children).filter { child ->
          child.kind as? String == "Field" && child.detail is String &&
            Regex("\\bmutable\\b").containsMatchIn(child.arcana as? String ?: "")
        }.mapNotNullTo(mutableFields) { it.detail as? String }
      }
      enclosingReturnType = when (kind) {
        "CXXConstructor", "CXXDestructor" -> "void"
        "Function", "CXXMethod", "CXXConversion" -> cppAstReturnType(node)
        else -> enclosingReturnType
      }
    }

    if ((kind == "Var" || kind == "ParmVar") && role == "declaration" &&
      name != null && declaredBeforeCursor && cppAstSpelledBeforeCursor(node, name)) {
      val scope = nearestAncestor { it.kind as? String in CPP_AST_SCOPE_KINDS }
      if (!cppDefined(scope) || cppAstRangeContainsCursor(scope.range)) {
        val type = cppQuotedAstType(node.arcana as? String).orEmpty()
        values += CppReference(name, type = type, kind = "variable", detail = type, source = "ast")
      }
    }

    if (kind == "Function" && role == "declaration" && name != null &&
      declaredBeforeCursor && nearestAncestor { it.kind as? String == "CXXRecord" } == null) {
      cppAstCallable(node, "function", name)?.let(functions::add)
    }

    if (kind == "CXXRecord" && role == "declaration" && name != null && declaredBeforeCursor &&
      (Regex("\\bdefinition\\b", RegexOption.IGNORE_CASE)
        .containsMatchIn(node.arcana as? String ?: "") ||
        "DefinitionData" in (node.arcana as? String ?: ""))) {
      collectRecord(node, name)
    }

    ancestors.add(node)
    cppDynamicList(node.children).forEach { child -> visit(child, depth + 1) }
    ancestors.removeAt(ancestors.lastIndex)
  }

  private fun collectRecord(record: dynamic, name: String) {
    val arcana = record.arcana as? String ?: ""
    val kind = if (Regex("\\bstruct\\s+").containsMatchIn(arcana)) "struct" else "class"
    val abstract = Regex("\\babstract\\b").containsMatchIn(arcana)
    val children = cppDynamicList(record.children)
    val bases = children.asSequence()
      .filter { it.role as? String == "base" && it.kind as? String == "public" }
      .mapNotNull(::cppFirstAstTypeName)
      .distinct()
      .toList()
    // clang does not materialize an implicit CXXConstructor node until it is used. Preserve the
    // stronger DefinitionData fact for method-only visitors such as `Describe{}`. The
    // `empty aggregate` marker by itself is insufficient: clang also uses it for records with a deleted
    // destructor and for empty derived aggregates whose base cannot be default-constructed.
    val implicitEmptyConstruction = !abstract && bases.isEmpty() &&
      "empty aggregate" in arcana &&
      Regex("\\bDefaultConstructor\\b[^\\r\\n]*\\bneeds_implicit\\b").containsMatchIn(arcana) &&
      Regex("\\bDestructor\\b[^\\r\\n]*\\bneeds_implicit\\b").containsMatchIn(arcana) &&
      children.none { child ->
        child.role as? String == "base" || child.kind as? String in setOf(
          "Field", "CXXConstructor", "ConstructorUsingShadow", "CXXDestructor"
        )
      }
    if (implicitEmptyConstruction) defaultConstructibleTypes += name
    val members = mutableListOf<CppReference>()
    var access = if (kind == "struct") "public" else "private"
    children.forEach { child ->
      val childKind = child.kind as? String
      if (childKind == "AccessSpec") {
        access = Regex("\\b(public|private|protected)\\s*$")
          .find(child.arcana as? String ?: "")?.groupValues?.get(1) ?: access
        return@forEach
      }
      if (access != "public") return@forEach
      val childName = (child.detail as? String)?.trim()?.takeIf(String::isNotEmpty)
      val childArcana = child.arcana as? String ?: ""
      val implicit = Regex("(?:^|\\s)implicit(?:\\s|$)").containsMatchIn(childArcana)
      when {
        childKind == "Field" && childName != null -> {
          val type = cppQuotedAstType(childArcana).orEmpty()
          members += CppReference(
            name = childName,
            type = type,
            kind = "field",
            detail = type,
            receiverMember = true,
            ownerType = name,
            source = "ast"
          )
        }
        childKind == "CXXMethod" && !implicit -> {
          val callableKind = if (childName?.startsWith("operator") == true) "operator" else "method"
          cppAstCallable(child, callableKind, childName)?.let { callable ->
            members += callable.copy(receiverMember = true, ownerType = name)
          }
        }
        childKind == "CXXConstructor" && !abstract -> {
          val deleted = Regex("\\b(?:deleted|delete)\\b|default_delete")
            .containsMatchIn(childArcana)
          val inherited = Regex("(?:^|\\s)implicit\\s+used(?:\\s|$)")
            .containsMatchIn(childArcana)
          val callable = cppAstCallable(child, "constructor", name)
          val implicitDefault = implicit && callable?.parameters?.isEmpty() == true
          if (!deleted && (!implicit || inherited || implicitDefault) && callable != null) {
            functions += callable.copy(ownerType = name)
            if (callable.parameters.isEmpty()) defaultConstructibleTypes += name
          }
        }
        childKind == "ConstructorUsingShadow" && !abstract -> {
          val signature = Regex("'([^'\\r\\n]+)'").findAll(childArcana)
            .map { it.groupValues[1] }
            .lastOrNull { '(' in it }
          val parameters = signature?.let(::cppParameterClause).orEmpty()
          val inheritedCopy = parameters.size == 1 && bases.any { base ->
            parameters.single().type
              .replace(Regex("\\b(?:const|volatile)\\b"), "")
              .replace(Regex("(?:&&|&)\\s*$"), "")
              .trim() == base
          }
          if (signature != null && !inheritedCopy) functions += CppReference(
            name = name,
            returnType = name,
            parameters = parameters,
            kind = "constructor",
            detail = signature,
            ownerType = name,
            source = "ast"
          )
        }
      }
    }
    types += CppReference(name, type = name, kind = kind, detail = name, source = "ast", abstract = abstract)
    records += CppNormalizedAstRecord(name, bases, cppDistinctReferences(members))
    bases.forEach { base -> conversions += CppConversion(name, base, kind = "base") }
  }

  private fun cppAstCallable(
    node: dynamic,
    kind: String,
    fallbackName: String?
  ): CppReference? {
    val name = (node.detail as? String)?.trim()?.takeIf(String::isNotEmpty) ?: fallbackName ?: return null
    val signature = cppQuotedAstType(node.arcana as? String).orEmpty()
    val parameters = cppAstParameterNodes(node).map(::cppAstParameter)
      .ifEmpty { cppParameterClause(signature) }
    return CppReference(
      name = name,
      returnType = if (kind == "constructor") fallbackName else cppAstReturnType(node),
      parameters = parameters,
      kind = kind,
      detail = signature,
      source = "ast"
    )
  }

  private fun cppAstParameter(node: dynamic): CppParameter {
    val type = cppQuotedAstType(node.arcana as? String).orEmpty()
    val name = node.detail as? String ?: ""
    val label = cppAstRangeText(node.range) ?: type
    val parsed = cppParameterFromLabel(label)
    return parsed.copy(
      label = label,
      name = name.ifBlank { parsed.name },
      type = type.ifBlank { parsed.type }
    )
  }

  private fun cppAstParameterNodes(root: dynamic): List<dynamic> {
    val result = mutableListOf<dynamic>()
    var visited = 0
    fun collect(node: dynamic, depth: Int) {
      if (!cppDefined(node) || depth > 32 || visited++ >= CPP_AST_PARAMETER_NODE_LIMIT) return
      if (node.kind as? String == "ParmVar" && node.role as? String == "declaration") {
        result.add(node)
        return
      }
      if (node !== root && node.role as? String in setOf("statement", "expression")) return
      cppDynamicList(node.children).forEach { collect(it, depth + 1) }
    }
    collect(root, 0)
    return result
  }

  private fun cppFirstAstTypeName(root: dynamic): String? {
    var visited = 0
    fun find(node: dynamic, depth: Int): String? {
      if (!cppDefined(node) || depth > 24 || visited++ >= CPP_AST_PARAMETER_NODE_LIMIT) return null
      if (node.role as? String == "type")
        (node.detail as? String)?.takeIf(String::isNotBlank)?.let { return it }
      cppDynamicList(node.children).forEach { child -> find(child, depth + 1)?.let { return it } }
      return null
    }
    return find(root, 0)
  }

  private fun cppAstReturnType(node: dynamic): String? {
    val signature = cppQuotedAstType(node.arcana as? String) ?: return null
    val open = signature.indexOf('(')
    return signature.substring(0, if (open < 0) signature.length else open).trim()
      .takeIf(String::isNotEmpty)
  }

  private fun cppQuotedAstType(arcana: String?): String? {
    arcana ?: return null
    Regex("(?:^|\\s)'([^'\\r\\n]+)'\\s*:\\s*'([^'\\r\\n]+)'")
      .find(arcana)?.groupValues?.get(2)?.let { return it }
    return Regex("(?:^|\\s)'([^'\\r\\n]+)'").find(arcana)?.groupValues?.get(1)
  }

  private fun cppAstRangeText(range: dynamic): String? {
    if (!cppDefined(range?.start) || !cppDefined(range?.end)) return null
    val startLine = cppInt(range.start.line, -1)
    val endLine = cppInt(range.end.line, -1)
    if (startLine < 0 || startLine != endLine) return null
    val line = sourceLines.getOrNull(startLine) ?: return null
    val start = cppInt(range.start.character, -1)
    val end = cppInt(range.end.character, -1)
    if (start < 0 || end < start || end > line.length) return null
    return line.substring(start, end).trim().takeIf(String::isNotEmpty)
  }

  private fun cppAstSpelledBeforeCursor(node: dynamic, name: String): Boolean {
    if (cppInt(node.range?.start?.line, -1) != cursorLine) return true
    val line = sourceLines.getOrNull(cursorLine).orEmpty()
    val prefix = line.substring(0, cursorCharacter.coerceIn(0, line.length))
    return Regex("(?:^|\\W)${Regex.escape(name)}(?:$|\\W)").containsMatchIn(prefix)
  }

  private fun cppAstPositionAtOrBefore(position: dynamic): Boolean {
    if (!cppDefined(position)) return false
    val line = cppInt(position.line, -1)
    val character = cppInt(position.character, -1)
    return line >= 0 && character >= 0 &&
      (line < cursorLine || line == cursorLine && character <= cursorCharacter)
  }

  private fun cppAstCursorAtOrBefore(position: dynamic): Boolean {
    if (!cppDefined(position)) return false
    val line = cppInt(position.line, -1)
    val character = cppInt(position.character, -1)
    return line > cursorLine || line == cursorLine && character >= cursorCharacter
  }

  private fun cppAstRangeContainsCursor(range: dynamic): Boolean =
    cppDefined(range) && cppAstPositionAtOrBefore(range.start) && cppAstCursorAtOrBefore(range.end)

  private inline fun nearestAncestor(predicate: (dynamic) -> Boolean): dynamic {
    for (index in ancestors.lastIndex downTo 0)
      if (predicate(ancestors[index])) return ancestors[index]
    return null
  }
}

private fun cppContextFromNormalizedAst(ast: dynamic): CppCompletionContext {
  if (!cppDefined(ast)) return CppCompletionContext(emptySet())
  return CppCompletionContext(
    identifiers = cppStringSet(ast.identifiers),
    values = cppDynamicList(ast.values).mapNotNull(::cppReferenceFromDto),
    types = cppDynamicList(ast.types).mapNotNull(::cppReferenceFromDto),
    functions = cppDynamicList(ast.functions).mapNotNull(::cppReferenceFromDto),
    membersByType = cppDynamicList(ast.membersByType).mapNotNull(::cppTypeMembersFromDto),
    conversions = cppDynamicList(ast.conversions).mapNotNull(::cppConversionFromDto),
    requiredTypes = cppStringSet(ast.requiredTypes),
    probedRequiredTypes = cppStringSet(ast.probedRequiredTypes),
    defaultConstructibleTypes = cppStringSet(ast.defaultConstructibleTypes),
    enclosingReturnType = ast.enclosingReturnType as? String,
    enclosingClassType = ast.enclosingClassType as? String,
    thisType = ast.thisType as? String,
    mutableFields = cppStringSet(ast.mutableFields)
  )
}

private fun cppCompletionContextToDto(context: CppCompletionContext): dynamic {
  val dto = js("({})")
  dto.identifiers = context.identifiers.sorted().toTypedArray()
  dto.sourceIdentifiers = context.sourceIdentifiers.sorted().toTypedArray()
  dto.headers = context.headers.sorted().toTypedArray()
  dto.typeNames = context.typeNames.sorted().toTypedArray()
  dto.values = context.values.map(::cppReferenceToDto).toTypedArray()
  dto.types = context.types.map(::cppReferenceToDto).toTypedArray()
  dto.functions = context.functions.map(::cppReferenceToDto).toTypedArray()
  dto.completions = context.completions.map(::cppReferenceToDto).toTypedArray()
  dto.signatures = context.signatures.map(::cppSignatureToDto).toTypedArray()
  dto.expectedTypes = context.expectedTypes.sorted().toTypedArray()
  dto.receiver = context.receiver?.let(::cppReceiverToDto)
  dto.membersByType = context.membersByType.map(::cppTypeMembersToDto).toTypedArray()
  dto.conversions = context.conversions.map(::cppConversionToDto).toTypedArray()
  dto.expressionWitnesses = context.expressionWitnesses
    .map(::cppExpressionWitnessToDto).toTypedArray()
  dto.callWitnesses = context.callWitnesses.map(::cppCallWitnessToDto).toTypedArray()
  dto.binaryOperatorWitnesses = context.binaryOperatorWitnesses
    .map(::cppBinaryOperatorWitnessToDto).toTypedArray()
  dto.unresolvedIdentifiers = context.unresolvedIdentifiers.sorted().toTypedArray()
  dto.requiredBinderObligation = context.requiredBinderObligation
    ?.let(::cppRequiredBinderObligationToDto)
  dto.requiredIdentifier = context.requiredIdentifier
  dto.requiredTypes = context.requiredTypes.sorted().toTypedArray()
  dto.probedRequiredTypes = context.probedRequiredTypes.sorted().toTypedArray()
  dto.defaultConstructibleTypes = context.defaultConstructibleTypes.sorted().toTypedArray()
  dto.enclosingReturnType = context.enclosingReturnType
  dto.canonicalEnclosingReturnType = context.canonicalEnclosingReturnType
  dto.enclosingReturnTypeInfo = context.enclosingReturnTypeInfo?.let(::cppTypeInfoToDto)
  dto.enclosingClassType = context.enclosingClassType
  dto.canonicalEnclosingClassType = context.canonicalEnclosingClassType
  dto.enclosingClassTypeInfo = context.enclosingClassTypeInfo?.let(::cppTypeInfoToDto)
  dto.thisType = context.thisType
  dto.canonicalThisType = context.canonicalThisType
  dto.thisTypeInfo = context.thisTypeInfo?.let(::cppTypeInfoToDto)
  dto.mutableFields = context.mutableFields.sorted().toTypedArray()
  dto.completionKind = context.completionKind
  dto.preferredType = context.preferredType
  dto.canonicalPreferredType = context.canonicalPreferredType
  dto.preferredTypeInfo = context.preferredTypeInfo?.let(::cppTypeInfoToDto)
  dto.baseType = context.baseType
  dto.canonicalBaseType = context.canonicalBaseType
  dto.baseTypeInfo = context.baseTypeInfo?.let(::cppTypeInfoToDto)
  dto.queryScopes = context.queryScopes.toTypedArray()
  dto.accessibleScopes = context.accessibleScopes.toTypedArray()
  dto.semanticGraphNodeCount = context.semanticGraphNodeCount
  dto.semanticGraphIsIncomplete = context.semanticGraphIsIncomplete
  dto.semanticOperationNodeCount = context.semanticOperationNodeCount
  dto.semanticOperationTemplateCount = context.semanticOperationTemplateCount
  dto.semanticOperationsAreIncomplete = context.semanticOperationsAreIncomplete
  dto.semanticExpressionWitnessesAreIncomplete =
    context.semanticExpressionWitnessesAreIncomplete
  dto.semanticCallWitnessesAreIncomplete = context.semanticCallWitnessesAreIncomplete
  dto.semanticBinaryOperatorWitnessesAreIncomplete =
    context.semanticBinaryOperatorWitnessesAreIncomplete
  return dto
}

private fun cppParameterToDto(parameter: CppParameter): dynamic {
  val dto = js("({})")
  dto.label = parameter.label
  dto.name = parameter.name
  dto.type = parameter.type
  dto.defaultValue = parameter.defaultValue
  dto.canonicalType = parameter.canonicalType
  dto.typeInfo = parameter.typeInfo?.let(::cppTypeInfoToDto)
  dto.hasDefault = parameter.hasDefault
  dto.isPack = parameter.isPack
  return dto
}

private fun cppReferenceToDto(reference: CppReference): dynamic {
  val dto = js("({})")
  dto.name = reference.name
  dto.type = reference.type
  dto.returnType = reference.returnType
  dto.parameters = reference.parameters.map(::cppParameterToDto).toTypedArray()
  dto.kind = reference.kind
  dto.detail = reference.detail
  dto.receiverMember = reference.receiverMember
  dto.ownerType = reference.ownerType
  dto.source = reference.source
  dto.abstract = reference.abstract
  dto.emptyAggregate = reference.emptyAggregate
  dto.id = reference.id
  dto.primaryTemplateId = reference.primaryTemplateId
  dto.qualifiedName = reference.qualifiedName
  dto.provenance = reference.provenance
  dto.canonicalType = reference.canonicalType
  dto.canonicalReturnType = reference.canonicalReturnType
  dto.canonicalOwnerType = reference.canonicalOwnerType
  dto.typeInfo = reference.typeInfo?.let(::cppTypeInfoToDto)
  dto.returnTypeInfo = reference.returnTypeInfo?.let(::cppTypeInfoToDto)
  dto.ownerTypeInfo = reference.ownerTypeInfo?.let(::cppTypeInfoToDto)
  dto.isType = reference.isType
  dto.isValue = reference.isValue
  dto.isCallable = reference.isCallable
  dto.isMember = reference.isMember
  dto.isStatic = reference.isStatic
  dto.isConstMethod = reference.isConstMethod
  dto.isVolatileMethod = reference.isVolatileMethod
  dto.refQualifier = reference.refQualifier
  dto.isMutableField = reference.isMutableField
  dto.isBitField = reference.isBitField
  dto.isVariadic = reference.isVariadic
  dto.isExplicit = reference.isExplicit
  dto.templateParameters = reference.templateParameters.map(::cppParameterToDto).toTypedArray()
  dto.completionVisible = reference.completionVisible
  dto.activeCallable = reference.activeCallable
  return dto
}

private fun cppTypeInfoToDto(type: CppTypeInfo): dynamic {
  val dto = js("({})")
  dto.id = type.id
  dto.canonicalId = type.canonicalId
  dto.valueCanonicalId = type.valueCanonicalId
  dto.kind = type.kind
  dto.isConst = type.isConst
  dto.isVolatile = type.isVolatile
  dto.pointeeCanonicalId = type.pointeeCanonicalId
  dto.pointeeIsConst = type.pointeeIsConst
  dto.pointeeIsVolatile = type.pointeeIsVolatile
  if (type.kind in CPP_ARRAY_TYPE_INFO_KINDS || type.elementCanonicalId != null ||
    type.isIncompleteArray != null || type.arrayBound != null
  ) {
    dto.elementCanonicalId = type.elementCanonicalId
    dto.elementIsConst = type.elementIsConst
    dto.elementIsVolatile = type.elementIsVolatile
    dto.isIncompleteArray = type.isIncompleteArray
    dto.arrayBound = type.arrayBound
  }
  dto.isDependent = type.isDependent
  dto.isInstantiationDependent = type.isInstantiationDependent
  dto.isSourceSpellable = type.isSourceSpellable
  dto.isComplete = type.isComplete
  dto.isDefaultConstructible = type.isDefaultConstructible
  return dto
}

private fun cppBindingProfileToDto(profile: CppBindingProfile): dynamic {
  val dto = js("({})")
  dto.type = profile.type
  dto.canonicalType = profile.canonicalType
  dto.typeInfo = profile.typeInfo?.let(::cppTypeInfoToDto)
  dto.declarationKind = profile.declarationKind
  return dto
}

private fun cppRequiredBinderObligationToDto(
  obligation: CppRequiredBinderObligation
): dynamic {
  val dto = js("({})")
  dto.binders = obligation.binders.sorted().toTypedArray()
  dto.singletonGate = obligation.singletonGate?.let { gate ->
    val item = js("({})")
    item.binder = gate.binder
    item.accepted = gate.accepted.map(::cppBindingProfileToDto).toTypedArray()
    item.probed = gate.probed.map(::cppBindingProfileToDto).toTypedArray()
    item.complete = gate.complete
    item
  }
  return dto
}

private fun cppSignatureToDto(signature: CppSignature): dynamic {
  val dto = js("({})")
  dto.label = signature.label
  dto.returnType = signature.returnType
  dto.parameters = signature.parameters.map(::cppParameterToDto).toTypedArray()
  dto.activeParameter = signature.activeParameter
  return dto
}

private fun cppReceiverToDto(receiver: CppReceiver): dynamic {
  val dto = js("({})")
  dto.operator = receiver.operator
  dto.expression = receiver.expression
  dto.type = receiver.type
  dto.members = receiver.members.map(::cppReferenceToDto).toTypedArray()
  return dto
}

private fun cppTypeMembersToDto(table: CppTypeMembers): dynamic {
  val dto = js("({})")
  dto.type = table.type
  dto.members = table.members.map(::cppReferenceToDto).toTypedArray()
  return dto
}

private fun cppConversionToDto(conversion: CppConversion): dynamic {
  val dto = js("({})")
  dto.from = conversion.from
  dto.to = conversion.to
  dto.kind = conversion.kind
  dto.canonicalFromType = conversion.canonicalFromType
  dto.canonicalToType = conversion.canonicalToType
  dto.fromTypeInfo = conversion.fromTypeInfo?.let(::cppTypeInfoToDto)
  dto.toTypeInfo = conversion.toTypeInfo?.let(::cppTypeInfoToDto)
  return dto
}

private fun cppExpressionProfileToDto(profile: CppExpressionProfile): dynamic {
  val dto = js("({})")
  dto.kind = profile.kind
  dto.spelling = profile.spelling
  dto.objectKind = profile.objectKind
  dto.type = profile.type
  dto.canonicalType = profile.canonicalType
  dto.typeInfo = profile.typeInfo?.let(::cppTypeInfoToDto)
  dto.valueCategory = profile.valueCategory
  return dto
}

private fun cppTypeProfileToDto(profile: CppTypeProfile): dynamic {
  val dto = js("({})")
  dto.type = profile.type
  dto.canonicalType = profile.canonicalType
  dto.typeInfo = cppTypeInfoToDto(profile.typeInfo)
  return dto
}

private fun cppTemplateArgumentProfileToDto(profile: CppTemplateArgumentProfile): dynamic {
  val dto = js("({})")
  dto.kind = profile.kind
  dto.type = cppTypeProfileToDto(profile.type)
  dto.spelling = profile.spelling
  dto.canonicalValue = profile.canonicalValue
  return dto
}

private fun cppExpressionWitnessToDto(witness: CppExpressionWitness): dynamic {
  val dto = js("({})")
  dto.syntax = witness.syntax
  dto.validation = witness.validation
  dto.typeOperand = witness.typeOperand?.let(::cppTypeProfileToDto)
  dto.expressionOperand = witness.expressionOperand?.let(::cppExpressionProfileToDto)
  dto.result = cppExpressionProfileToDto(witness.result)
  dto.authoritative = witness.authoritative
  return dto
}

private fun cppCallWitnessToDto(witness: CppCallWitness): dynamic {
  val dto = js("({})")
  dto.name = witness.name
  dto.syntax = witness.syntax
  dto.validation = witness.validation
  dto.targetId = witness.targetId
  dto.primaryTemplateId = witness.primaryTemplateId
  when {
    witness.explicitTemplateArguments.isNotEmpty() -> {
      dto.explicitTemplateArguments = witness.explicitTemplateArguments
        .map(::cppTemplateArgumentProfileToDto).toTypedArray()
      // Preserve an invalid in-memory hybrid so the receiving boundary rejects it.
      if (witness.explicitTypeArguments.isNotEmpty())
        dto.explicitTypeArguments = witness.explicitTypeArguments
          .map(::cppTypeProfileToDto).toTypedArray()
    }
    witness.explicitTypeArguments.isNotEmpty() ->
      dto.explicitTypeArguments = witness.explicitTypeArguments
        .map(::cppTypeProfileToDto).toTypedArray()
    else -> dto.explicitTemplateArguments = emptyArray<dynamic>()
  }
  dto.receiver = witness.receiver?.let(::cppExpressionProfileToDto)
  dto.arguments = witness.arguments.map(::cppExpressionProfileToDto).toTypedArray()
  dto.callable = cppReferenceToDto(witness.callable)
  dto.result = cppExpressionProfileToDto(witness.result)
  dto.authoritative = witness.authoritative
  return dto
}

private fun cppBinaryOperatorWitnessToDto(witness: CppBinaryOperatorWitness): dynamic {
  val dto = js("({})")
  dto.name = witness.name
  dto.syntax = witness.syntax
  dto.operatorSpelling = witness.operatorSpelling
  dto.validation = witness.validation
  dto.targetId = witness.targetId
  dto.primaryTemplateId = witness.primaryTemplateId
  dto.left = cppExpressionProfileToDto(witness.left)
  dto.right = cppExpressionProfileToDto(witness.right)
  dto.callable = cppReferenceToDto(witness.callable)
  dto.result = cppExpressionProfileToDto(witness.result)
  dto.authoritative = witness.authoritative
  return dto
}

private fun cppParameterFromDto(value: dynamic): CppParameter? {
  if (!cppDefined(value)) return null
  val type = value.type as? String ?: ""
  return CppParameter(
    label = value.label as? String ?: type,
    name = value.name as? String ?: "",
    type = type,
    defaultValue = value.defaultValue as? String,
    canonicalType = value.canonicalType as? String,
    typeInfo = cppTypeInfo(value.typeInfo),
    hasDefault = value.hasDefault as? Boolean,
    isPack = value.isPack as? Boolean ?: false
  )
}

private fun cppReferenceFromDto(value: dynamic): CppReference? =
  if (!cppDefined(value)) null
  else CppReference(
    name = value.name as? String ?: return null,
    type = value.type as? String,
    returnType = value.returnType as? String,
    parameters = cppDynamicList(value.parameters).mapNotNull(::cppParameterFromDto),
    kind = value.kind as? String ?: "value",
    detail = value.detail as? String,
    receiverMember = value.receiverMember as? Boolean ?: false,
    ownerType = value.ownerType as? String,
    source = value.source as? String,
    abstract = value.abstract as? Boolean ?: false,
    emptyAggregate = value.emptyAggregate as? Boolean ?: false,
    id = value.id as? String,
    primaryTemplateId = value.primaryTemplateId as? String,
    qualifiedName = value.qualifiedName as? String,
    provenance = value.provenance as? String,
    canonicalType = value.canonicalType as? String,
    canonicalReturnType = value.canonicalReturnType as? String,
    canonicalOwnerType = value.canonicalOwnerType as? String,
    typeInfo = cppTypeInfo(value.typeInfo),
    returnTypeInfo = cppTypeInfo(value.returnTypeInfo),
    ownerTypeInfo = cppTypeInfo(value.ownerTypeInfo),
    isType = value.isType as? Boolean,
    isValue = value.isValue as? Boolean,
    isCallable = value.isCallable as? Boolean,
    isMember = value.isMember as? Boolean,
    isStatic = value.isStatic as? Boolean,
    isConstMethod = value.isConstMethod as? Boolean,
    isVolatileMethod = value.isVolatileMethod as? Boolean,
    refQualifier = value.refQualifier as? String,
    isMutableField = value.isMutableField as? Boolean,
    isBitField = value.isBitField as? Boolean,
    isVariadic = value.isVariadic as? Boolean ?: false,
    isExplicit = value.isExplicit as? Boolean,
    templateParameters = cppDynamicList(value.templateParameters).mapNotNull(::cppParameterFromDto),
    completionVisible = value.completionVisible as? Boolean ?: false,
    activeCallable = value.activeCallable as? Boolean ?: false
  )

private fun cppSignatureFromDto(value: dynamic): CppSignature? =
  if (!cppDefined(value)) null
  else CppSignature(
    label = value.label as? String ?: return null,
    returnType = value.returnType as? String,
    parameters = cppDynamicList(value.parameters).mapNotNull(::cppParameterFromDto),
    activeParameter = (value.activeParameter as? Number)?.toInt()
  )

private fun cppReceiverFromDto(value: dynamic): CppReceiver? =
  if (!cppDefined(value)) null
  else CppReceiver(
    operator = value.operator as? String ?: return null,
    expression = value.expression as? String ?: "",
    type = value.type as? String,
    members = cppDynamicList(value.members).mapNotNull(::cppReferenceFromDto)
  )

private fun cppTypeMembersFromDto(value: dynamic): CppTypeMembers? =
  if (!cppDefined(value)) null
  else CppTypeMembers(
    type = value.type as? String ?: return null,
    members = cppDynamicList(value.members).mapNotNull(::cppReferenceFromDto)
  )

private fun cppConversionFromDto(value: dynamic): CppConversion? {
  if (!cppDefined(value)) return null
  val canonicalFromType = (value.canonicalFromType as? String)?.takeIf(String::isNotBlank)
  val canonicalToType = (value.canonicalToType as? String)?.takeIf(String::isNotBlank)
  if (cppDefined(value.canonicalFromType) && canonicalFromType == null ||
    cppDefined(value.canonicalToType) && canonicalToType == null
  ) return null
  val fromTypeInfo = if (cppDefined(value.fromTypeInfo))
    cppConversionTypeInfo(value.fromTypeInfo) ?: return null
  else null
  val toTypeInfo = if (cppDefined(value.toTypeInfo))
    cppConversionTypeInfo(value.toTypeInfo) ?: return null
  else null
  return CppConversion(
    from = (value.from as? String)?.takeIf(String::isNotBlank) ?: return null,
    to = (value.to as? String)?.takeIf(String::isNotBlank) ?: return null,
    kind = value.kind as? String,
    canonicalFromType = canonicalFromType,
    canonicalToType = canonicalToType,
    fromTypeInfo = fromTypeInfo,
    toTypeInfo = toTypeInfo
  )
}

private fun cppExpressionProfileFromDto(value: dynamic): CppExpressionProfile? =
  if (!cppDefined(value)) null
  else {
    val spelling = if (cppDefined(value.spelling)) value.spelling as? String ?: return null else null
    val objectKind = value.objectKind as? String ?: return null
    val valueCategory = value.valueCategory as? String ?: return null
    if (valueCategory !in setOf("lvalue", "xvalue", "prvalue")) return null
    CppExpressionProfile(
      kind = value.kind as? String ?: return null,
      spelling = spelling,
      objectKind = objectKind,
      type = value.type as? String,
      canonicalType = value.canonicalType as? String,
      typeInfo = cppWitnessTypeInfo(value.typeInfo) ?: return null,
      valueCategory = valueCategory
    ).takeIf(CppExpressionProfile::isWellFormedCppExpressionProfile)
  }

private fun cppTypeProfileFromDto(value: dynamic): CppTypeProfile? =
  if (!cppDefined(value)) null
  else CppTypeProfile(
    type = value.type as? String ?: return null,
    canonicalType = value.canonicalType as? String,
    typeInfo = cppWitnessTypeInfo(value.typeInfo) ?: return null
  )

private fun cppTemplateArgumentProfileFromDto(value: dynamic): CppTemplateArgumentProfile? {
  if (!cppDefined(value)) return null
  val profile = CppTemplateArgumentProfile(
    kind = value.kind as? String ?: return null,
    type = cppTypeProfileFromDto(value.type) ?: return null,
    spelling = value.spelling as? String,
    canonicalValue = value.canonicalValue as? String
  )
  return profile.takeIf(CppTemplateArgumentProfile::isWellFormedCppTemplateArgument)
}

private fun cppExpressionWitnessFromDto(value: dynamic): CppExpressionWitness? {
  if (!cppDefined(value)) return null
  val typeOperand = cppTypeProfileFromDto(value.typeOperand)
  if (cppDefined(value.typeOperand) && typeOperand == null) return null
  val expressionOperand = cppExpressionProfileFromDto(value.expressionOperand)
  if (cppDefined(value.expressionOperand) && expressionOperand == null) return null
  return CppExpressionWitness(
    syntax = value.syntax as? String ?: return null,
    validation = value.validation as? String ?: "",
    typeOperand = typeOperand,
    expressionOperand = expressionOperand,
    result = cppExpressionProfileFromDto(value.result) ?: return null,
    authoritative = value.authoritative as? Boolean ?: false
  )
}

private fun cppCallWitnessFromDto(value: dynamic): CppCallWitness? {
  if (!cppDefined(value)) return null
  if (!cppIsArray(value.arguments)) return null
  val rawArguments = cppDynamicList(value.arguments)
  val arguments = rawArguments.mapNotNull(::cppExpressionProfileFromDto)
  if (arguments.size != rawArguments.size) return null
  val hasExplicitTemplateArguments = cppDefined(value.explicitTemplateArguments)
  if (hasExplicitTemplateArguments && !cppIsArray(value.explicitTemplateArguments)) return null
  val rawExplicitTemplateArguments = cppDynamicList(value.explicitTemplateArguments)
  val explicitTemplateArguments =
    rawExplicitTemplateArguments.mapNotNull(::cppTemplateArgumentProfileFromDto)
  if (explicitTemplateArguments.size != rawExplicitTemplateArguments.size) return null
  if (cppDefined(value.explicitTypeArguments) && !cppIsArray(value.explicitTypeArguments))
    return null
  val rawExplicitTypeArguments = cppDynamicList(value.explicitTypeArguments)
  val explicitTypeArguments = rawExplicitTypeArguments.mapNotNull(::cppTypeProfileFromDto)
  if (explicitTypeArguments.size != rawExplicitTypeArguments.size) return null
  if (hasExplicitTemplateArguments && explicitTypeArguments.isNotEmpty()) return null
  val syntax = value.syntax as? String ?: return null
  val receiver = cppExpressionProfileFromDto(value.receiver)
  if (cppDefined(value.receiver) && receiver == null) return null
  if (syntax == "memberCall" && receiver == null) return null
  val callableValue = value.callable
  val targetId = cppOptionalCallIdentity(value.targetId) ?: return null
  val primaryTemplateId = cppOptionalCallIdentity(value.primaryTemplateId) ?: return null
  val callableId = cppOptionalCallIdentity(callableValue?.id) ?: return null
  val callablePrimaryTemplateId = cppOptionalCallIdentity(callableValue?.primaryTemplateId)
    ?: return null
  val callableReturnInfo = cppWitnessTypeInfo(callableValue?.returnTypeInfo) ?: return null
  val callable = cppReferenceFromDto(callableValue)?.copy(
    id = callableId.value,
    primaryTemplateId = callablePrimaryTemplateId.value,
    returnTypeInfo = callableReturnInfo
  ) ?: return null
  return CppCallWitness(
    name = value.name as? String ?: return null,
    syntax = syntax,
    validation = value.validation as? String ?: "",
    targetId = targetId.value,
    primaryTemplateId = primaryTemplateId.value,
    explicitTemplateArguments = explicitTemplateArguments,
    explicitTypeArguments = if (hasExplicitTemplateArguments) emptyList()
    else explicitTypeArguments,
    receiver = receiver,
    arguments = arguments,
    callable = callable,
    result = cppExpressionProfileFromDto(value.result) ?: return null,
    authoritative = value.authoritative as? Boolean ?: false
  ).takeIf(CppCallWitness::hasWellFormedTargetIdentity)
}

private fun cppBinaryOperatorWitnessFromDto(value: dynamic): CppBinaryOperatorWitness? {
  if (!cppDefined(value)) return null
  val callableValue = value.callable
  val targetId = cppOptionalCallIdentity(value.targetId) ?: return null
  val primaryTemplateId = cppOptionalCallIdentity(value.primaryTemplateId) ?: return null
  val callableId = cppOptionalCallIdentity(callableValue?.id) ?: return null
  val callablePrimaryTemplateId = cppOptionalCallIdentity(callableValue?.primaryTemplateId)
    ?: return null
  val callableReturnInfo = cppWitnessTypeInfo(callableValue?.returnTypeInfo) ?: return null
  val callable = cppReferenceFromDto(callableValue)?.copy(
    id = callableId.value,
    primaryTemplateId = callablePrimaryTemplateId.value,
    returnTypeInfo = callableReturnInfo
  ) ?: return null
  return CppBinaryOperatorWitness(
    name = value.name as? String ?: return null,
    syntax = value.syntax as? String ?: return null,
    operatorSpelling = value.operatorSpelling as? String ?: return null,
    validation = value.validation as? String ?: "",
    targetId = targetId.value,
    primaryTemplateId = primaryTemplateId.value,
    left = cppExpressionProfileFromDto(value.left) ?: return null,
    right = cppExpressionProfileFromDto(value.right) ?: return null,
    callable = callable,
    result = cppExpressionProfileFromDto(value.result) ?: return null,
    authoritative = value.authoritative as? Boolean ?: false
  ).takeIf(CppBinaryOperatorWitness::hasWellFormedTargetIdentity)
}

private data class CppOptionalCallIdentity(val value: String?)

/** Empty is the producer's legacy spelling for an absent optional ID; malformed IDs fail closed. */
private fun cppOptionalCallIdentity(value: dynamic): CppOptionalCallIdentity? {
  if (!cppDefined(value) || value == null) return CppOptionalCallIdentity(null)
  val identity = value as? String ?: return null
  if (identity.isEmpty()) return CppOptionalCallIdentity(null)
  return CppOptionalCallIdentity(identity.takeIf { it.isNotBlank() && it == it.trim() }
    ?: return null)
}

private fun cppDefined(value: dynamic): Boolean = value != null && jsTypeOf(value) != "undefined"

private fun cppIsArray(value: dynamic): Boolean = cppDefined(value) && js("Array.isArray(value)") as Boolean

private fun cppDynamicList(value: dynamic): List<dynamic> =
  if (!cppIsArray(value)) emptyList() else (0 until cppInt(value.length)).map { value[it] }

private fun cppStringSet(value: dynamic): Set<String> =
  cppDynamicList(value).mapNotNullTo(linkedSetOf()) { it as? String }

private fun cppInt(value: dynamic, fallback: Int = 0): Int = (value as? Number)?.toInt() ?: fallback
