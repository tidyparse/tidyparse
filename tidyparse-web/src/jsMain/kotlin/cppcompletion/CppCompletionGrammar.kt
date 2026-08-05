package cppcompletion

import ai.hypergraph.kaliningraph.parsing.BoundedAcyclicCFG
import ai.hypergraph.kaliningraph.parsing.BoundedCountWorkspace
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.PTree
import ai.hypergraph.kaliningraph.parsing.PreindexedAcyclicCFG
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import ai.hypergraph.kaliningraph.parsing.freeze
import ai.hypergraph.tidyparse.lexCppTokenSpans
import com.ionspin.kotlin.bignum.integer.BigInteger
import kotlin.time.TimeSource

const val CPP_CFG_BUDGET_MILLIS = 500L
// The longest committed statement is the 48-token associative report. Keeping the horizon equal
// to that locked corpus maximum avoids length-64 count vectors for every much shorter cursor.
const val CPP_MAX_STATEMENT_TOKENS = 48
const val CPP_SUFFIX_HORIZON = CPP_MAX_STATEMENT_TOKENS
const val CPP_FRESH = "@fresh"
internal const val CPP_SYNTAX_IDENTIFIER = "@identifier"
const val CPP_BIND_PREFIX = "@bind:"
private const val CPP_SEMANTIC_DEPTH = 6

internal const val CPP_INTEGER = "@integer"
private const val CPP_FLOATING = "@floating"
private const val CPP_CHARACTER = "@character"
private const val CPP_STRING = "@string"
private const val CPP_BOOLEAN = "@boolean"
private const val CPP_NULLPTR = "@nullptr"
private const val SOURCE_EPSILON_RULE = 0
private const val SOURCE_TERMINAL_RULE = 1
private const val SOURCE_UNIT_RULE = 2
private const val SOURCE_BINARY_RULE = 3

/**
 * Read-only CFG view with the one cache primitive Galoisenne's extension properties require.
 * `CFG.freeze()` also computes all display statistics eagerly; cursor generation only needs a
 * stable hash, while `stats()` can populate its ordinary lazy caches later when the report renders.
 */
/**
 * Set-semantic view over immutable production groups whose LHS namespaces are disjoint. The
 * conditioner has already established a dense child-before-parent index, so Galoisenne can use it
 * directly rather than grouping and validating the same thousands of productions at each cursor.
 */
private class IndexedChunkedCppCfg(
  private val chunks: List<Collection<Pair<String, List<String>>>>,
  override val size: Int,
  override val acyclicCountingOrder: List<String>,
  override val acyclicNonterminalIndex: Map<String, Int>,
  override val acyclicStructuralStats: String,
  private val productionLookup: (String) -> List<Pair<String, List<String>>>
) : AbstractSet<Pair<String, List<String>>>(), PreindexedAcyclicCFG {
  // Most residuals never serve as keys in Galoisenne's general CFG property caches. Compute the
  // ordinary Set hash only on demand, while retaining exact Set equality for those rare callers.
  private val cachedHash by lazy {
    chunks.fold(0) { hash, group -> group.fold(hash) { sum, production ->
      sum + production.hashCode()
    } }
  }

  override fun productionsFor(nonterminal: String): List<Pair<String, List<String>>> =
    if (nonterminal in acyclicNonterminalIndex) productionLookup(nonterminal) else emptyList()

  override fun contains(element: Pair<String, List<String>>): Boolean =
    chunks.any { element in it }

  override fun iterator(): Iterator<Pair<String, List<String>>> = object :
    Iterator<Pair<String, List<String>>> {
    private val groups = chunks.iterator()
    private var rules: Iterator<Pair<String, List<String>>> = emptyList<Pair<String, List<String>>>().iterator()

    override fun hasNext(): Boolean {
      while (!rules.hasNext() && groups.hasNext()) rules = groups.next().iterator()
      return rules.hasNext()
    }

    override fun next(): Pair<String, List<String>> {
      if (!hasNext()) throw NoSuchElementException()
      return rules.next()
    }
  }

  override fun hashCode(): Int = cachedHash
  override fun equals(other: Any?): Boolean =
    other === this || other is Set<*> && size == other.size && all { it in other }
}

enum class CppTokenKind {
  IDENTIFIER,
  INTEGER,
  FLOATING,
  CHARACTER,
  STRING,
  USER_DEFINED_INTEGER,
  USER_DEFINED_FLOATING,
  USER_DEFINED_CHARACTER,
  USER_DEFINED_STRING,
  BOOLEAN,
  NULLPTR,
  OTHER
}

data class CppToken(
  val text: String,
  val start: Int,
  val end: Int,
  val kind: CppTokenKind,
  val completeText: String? = null
)

data class CppLine(
  val number: Int,
  val start: Int,
  val contentEnd: Int,
  val end: Int,
  val text: String,
  val tokens: List<CppToken>
)

data class CppTruncation(
  val line: CppLine,
  val prefix: List<CppToken>,
  val suffix: List<CppToken>,
  val prefixText: String
)

/** Suffix token positions that must be alpha-renamed together for guarded wildcard recall. */
data class CppFreshMatch(val groups: List<List<Int>>)

internal data class CppConditioningMetrics(
  val derivativeMillis: Long = 0,
  val reachableMillis: Long = 0,
  val boundedMillis: Long = 0
)

data class CppCompletionContext(
  val identifiers: Set<String>,
  val sourceIdentifiers: Set<String> = emptySet(),
  val headers: Set<String> = emptySet(),
  val typeNames: Set<String> = emptySet(),
  val values: List<CppReference> = emptyList(),
  val types: List<CppReference> = emptyList(),
  val functions: List<CppReference> = emptyList(),
  val completions: List<CppReference> = emptyList(),
  val signatures: List<CppSignature> = emptyList(),
  val expectedTypes: Set<String> = emptySet(),
  val receiver: CppReceiver? = null,
  val membersByType: List<CppTypeMembers> = emptyList(),
  val conversions: List<CppConversion> = emptyList(),
  val unresolvedIdentifiers: Set<String> = emptySet(),
  val requiredIdentifier: String? = null,
  val requiredTypes: Set<String> = emptySet(),
  /** Types actually tried by the compiler oracle; absence from this set is not a rejection. */
  val probedRequiredTypes: Set<String> = emptySet(),
  /** User types whose ordinary `T name;` form clang++ proved valid at this damaged line. */
  val defaultConstructibleTypes: Set<String> = emptySet(),
  val enclosingReturnType: String? = null,
  val canonicalEnclosingReturnType: String? = null,
  val enclosingReturnTypeInfo: CppTypeInfo? = null,
  val enclosingClassType: String? = null,
  val canonicalEnclosingClassType: String? = null,
  val enclosingClassTypeInfo: CppTypeInfo? = null,
  val thisType: String? = null,
  val canonicalThisType: String? = null,
  val thisTypeInfo: CppTypeInfo? = null,
  val mutableFields: Set<String> = emptySet(),
  /** Structured cursor state reported directly by clang/Sema. */
  val completionKind: String? = null,
  val preferredType: String? = null,
  val canonicalPreferredType: String? = null,
  val preferredTypeInfo: CppTypeInfo? = null,
  val baseType: String? = null,
  val canonicalBaseType: String? = null,
  val baseTypeInfo: CppTypeInfo? = null,
  val queryScopes: List<String> = emptyList(),
  val accessibleScopes: List<String> = emptyList()
)

/** Opaque type identity and safety facts captured while clang's Sema AST is alive. */
data class CppTypeInfo(
  val id: String? = null,
  val canonicalId: String? = null,
  val valueCanonicalId: String? = null,
  val kind: String? = null,
  val isConst: Boolean = false,
  val isVolatile: Boolean = false,
  val pointeeCanonicalId: String? = null,
  val pointeeIsConst: Boolean = false,
  val pointeeIsVolatile: Boolean = false,
  val isDependent: Boolean = false,
  val isInstantiationDependent: Boolean = false,
  val isSourceSpellable: Boolean? = null
)

/** Compact, transport-friendly clang semantic facts used to specialize one cursor CFG. */
data class CppParameter(
  val label: String = "",
  val name: String = "",
  val type: String = "",
  val defaultValue: String? = null,
  val canonicalType: String? = null,
  val typeInfo: CppTypeInfo? = null,
  val hasDefault: Boolean? = null,
  val isPack: Boolean = false
)

data class CppReference(
  val name: String,
  val type: String? = null,
  val returnType: String? = null,
  val parameters: List<CppParameter> = emptyList(),
  val kind: String = "value",
  val detail: String? = null,
  val receiverMember: Boolean = false,
  val ownerType: String? = null,
  val source: String? = null,
  /** Clang's AST marks records containing an unimplemented pure virtual member explicitly. */
  val abstract: Boolean = false,
  val id: String? = null,
  val qualifiedName: String? = null,
  val provenance: String? = source,
  val canonicalType: String? = null,
  val canonicalReturnType: String? = null,
  val canonicalOwnerType: String? = null,
  val typeInfo: CppTypeInfo? = null,
  val returnTypeInfo: CppTypeInfo? = null,
  val ownerTypeInfo: CppTypeInfo? = null,
  val isType: Boolean? = null,
  val isValue: Boolean? = null,
  val isCallable: Boolean? = null,
  val isMember: Boolean? = null,
  val isStatic: Boolean? = null,
  val isConstMethod: Boolean? = null,
  val isVolatileMethod: Boolean? = null,
  val refQualifier: String? = null,
  val isMutableField: Boolean? = null,
  val isVariadic: Boolean = false,
  val templateParameters: List<CppParameter> = emptyList()
)

data class CppSignature(
  val label: String,
  val returnType: String? = null,
  val parameters: List<CppParameter> = emptyList(),
  val activeParameter: Int? = null
)

data class CppReceiver(
  val operator: String,
  val expression: String,
  val type: String? = null,
  val members: List<CppReference> = emptyList()
)

data class CppTypeMembers(val type: String, val members: List<CppReference>)
data class CppConversion(val from: String, val to: String)

/**
 * A prefix-conditioned, finite completion language. The syntax is a native Galoisenne CFG and
 * [forest] is its shared finite parse forest; [bounded] supplies uniform derivation sampling over
 * every nonempty suffix within the configured token horizon.
 */
class CppSuffixGrammar internal constructor(
  internal val bounded: BoundedAcyclicCFG,
  val rawPrefix: List<CppToken>,
  val projectedPrefix: List<String>,
  val templateTokens: Int,
  /** The cursor-independent grammar, retained for syntax-fragment regression checks. */
  val sourceSyntax: CFG = bounded.grammar,
  internal val conditioningMetrics: CppConditioningMetrics = CppConditioningMetrics(),
  internal val projectionMode: CppProjectionMode = CppProjectionMode.SEMANTIC,
  internal val identifierInventory: Set<String> = emptySet(),
  private val syntaxFallbackFactory: ((CppToken?) -> CppSuffixGrammar?)? = null,
  private val recognizesCompleteSyntax: Boolean = false
) {
  private val syntaxFallback: CppSuffixGrammar? by lazy { syntaxFallbackFactory?.invoke(null) }
  val syntax: CFG get() = bounded.grammar
  val forest: PTree? get() = bounded.forest
  val isEmpty: Boolean get() = bounded.isEmpty && syntaxFallback?.isEmpty != false
  internal val derivationCount: BigInteger get() = bounded.derivationCount

  private fun accepts(candidate: List<String>): Boolean = bounded.recognizes(candidate)

  fun recognizes(rawSuffix: List<CppToken>): Boolean {
    val full = projectCppCompletionTokens(rawPrefix + rawSuffix, projectionMode)
    val candidate = full.drop(projectedPrefix.size)
    if (candidate.size <= templateTokens && accepts(candidate)) return true
    if (recognizesCompleteSyntax && rawSuffix.asSequence()
        .filter { it.kind == CppTokenKind.IDENTIFIER }
        .all { it.text in identifierInventory } &&
      cppSingleStatementSyntaxRecognizes(rawPrefix + rawSuffix)
    ) return true
    // A semantic residual retains semantic membership semantics for compiler-backed precision
    // checks. Its syntax floor is a sampling fallback, not a widening of this predicate.
    return false
  }

  /** Returns alpha-renaming alignments that are admitted and must be compiler-guarded. */
  fun freshMatches(rawSuffix: List<CppToken>): List<CppFreshMatch> {
    val matches = mutableListOf<CppFreshMatch>()
    rawSuffix.indices.forEach { suffixIndex ->
      val suffix = rawSuffix.mapIndexed { index, token ->
        if (index == suffixIndex) token.copy(text = CPP_FRESH, kind = CppTokenKind.OTHER) else token
      }
      val full = projectCppCompletionTokens(rawPrefix + suffix, projectionMode)
      val projected = full.drop(projectedPrefix.size)
      if (projected.size <= templateTokens && accepts(projected)) {
        matches += CppFreshMatch(listOf(listOf(suffixIndex)))
      }
    }

    val nonterminals = syntax.mapTo(linkedSetOf()) { it.first }
    val binders = syntax.flatMap { it.second }
      .filter { it !in nonterminals && it.startsWith(CPP_BIND_PREFIX) }
      .distinct()
    if (binders.isEmpty()) return matches
    val identifiers = rawSuffix.withIndex()
      .filter { it.value.kind == CppTokenKind.IDENTIFIER }
      .groupBy({ it.value.text }, { it.index })
    fun assign(
      selectedBinders: List<String>,
      slot: Int,
      remaining: List<String>,
      chosen: MutableMap<String, String>
    ) {
      if (slot == selectedBinders.size) {
        val suffix = rawSuffix.map { token ->
          val binder = chosen[token.text]
          if (binder == null) token else token.copy(text = binder, kind = CppTokenKind.OTHER)
        }
        val projected = projectCppCompletionTokens(rawPrefix + suffix, projectionMode)
          .drop(projectedPrefix.size)
        if (projected.size <= templateTokens && accepts(projected)) {
          matches += CppFreshMatch(chosen.keys.mapNotNull(identifiers::get))
        }
        return
      }
      remaining.forEach { identifier ->
        chosen[identifier] = selectedBinders[slot]
        assign(selectedBinders, slot + 1, remaining - identifier, chosen)
        chosen.remove(identifier)
      }
    }
    binders.groupBy { binder ->
      val suffix = binder.removePrefix(CPP_BIND_PREFIX)
      if (suffix.substringAfterLast(':').all(Char::isDigit) && ':' in suffix)
        binder.substringBeforeLast(':')
      else if (suffix.all(Char::isDigit)) CPP_BIND_PREFIX
      else binder
    }.values.forEach { group ->
      if (identifiers.size >= group.size)
        assign(group, 0, identifiers.keys.toList(), linkedMapOf())
    }
    return matches.distinct()
  }

  internal fun completionFallback(tokenPrefix: CppToken? = null): CppSuffixGrammar? =
    if (tokenPrefix == null) syntaxFallback else syntaxFallbackFactory?.invoke(tokenPrefix)

  internal fun withSyntaxFallback(identifiers: Set<String>): CppSuffixGrammar = CppSuffixGrammar(
    bounded = bounded,
    rawPrefix = rawPrefix,
    projectedPrefix = projectedPrefix,
    templateTokens = templateTokens,
    sourceSyntax = sourceSyntax,
    conditioningMetrics = conditioningMetrics,
    projectionMode = projectionMode,
    identifierInventory = identifiers,
    syntaxFallbackFactory = { cppSingleStatementSyntaxCompletion(rawPrefix, it, identifiers) }
  )
}

internal enum class CppProjectionMode { SEMANTIC, SYNTAX }

/** Constructs one finite, cursor-specialized statement grammar from clang's scoped facts. */
class CppCompletionGrammar {
  fun prepare(context: CppCompletionContext): PreparedCppCompletionGrammar =
    PreparedCppCompletionGrammar(
      SemanticCppGrammar(context, emptyList()).build(),
      context.syntaxIdentifierInventory()
    )

  fun prepare(context: CppCompletionContext, prefix: List<CppToken>): PreparedCppCompletionGrammar =
    PreparedCppCompletionGrammar(
      SemanticCppGrammar(context, prefix).build(),
      context.syntaxIdentifierInventory() + prefix.sourceIdentifierInventory()
    )

  fun generate(context: CppCompletionContext, prefix: List<CppToken>): CppSuffixGrammar = when {
    prefix.endsCompleteStatement() -> completedStatementGrammar(prefix)
    else -> prepare(context, prefix).generate(prefix)
  }
}

/** Reuses one line's scoped semantic grammar while deriving an exact residual at every cursor. */
class PreparedCppCompletionGrammar internal constructor(
  private val sourceSyntax: CFG,
  private val identifierInventory: Set<String> = emptySet()
) {
  private val conditioner = FiniteCppConditioner(sourceSyntax)

  /** Exact prepared-language membership without materializing a residual CFG or CYK index. */
  fun recognizes(statement: List<CppToken>): Boolean =
    conditioner.recognizesExactly(projectCppTokens(statement))

  fun generate(prefix: List<CppToken>): CppSuffixGrammar {
    if (prefix.endsCompleteStatement()) return completedStatementGrammar(prefix, sourceSyntax)
    val projectedPrefix = projectCppTokens(prefix)
    if (projectedPrefix.size > CPP_MAX_STATEMENT_TOKENS)
      return emptyCppSuffixGrammar(prefix, sourceSyntax).withSyntaxFallback(identifierInventory)
    val suffixTokens = CPP_SUFFIX_HORIZON.coerceAtMost(
      (CPP_MAX_STATEMENT_TOKENS - projectedPrefix.size).coerceAtLeast(0)
    )
    val bounded = conditioner.condition(projectedPrefix, suffixTokens)
    return CppSuffixGrammar(
      bounded = bounded,
      rawPrefix = prefix,
      projectedPrefix = projectedPrefix,
      templateTokens = suffixTokens,
      sourceSyntax = sourceSyntax,
      conditioningMetrics = conditioner.lastMetrics,
      identifierInventory = identifierInventory
    ).withSyntaxFallback(identifierInventory)
  }
}

private fun CppCompletionContext.syntaxIdentifierInventory(): Set<String> =
  identifiers.filterTo(linkedSetOf(), String::isCppIdentifierName)

private fun List<CppToken>.sourceIdentifierInventory(): Set<String> = asSequence()
  .filter { it.kind == CppTokenKind.IDENTIFIER && it.text.isCppIdentifierName() }
  .mapTo(linkedSetOf(), CppToken::text)

private fun emptyCppSuffixGrammar(prefix: List<CppToken>, sourceSyntax: CFG): CppSuffixGrammar =
  CppSuffixGrammar(
    bounded = emptySet<Pair<String, List<String>>>().boundedAcyclic(0),
    rawPrefix = prefix,
    projectedPrefix = projectCppTokens(prefix),
    templateTokens = 0,
    sourceSyntax = sourceSyntax
  )

private fun completedStatementGrammar(prefix: List<CppToken>, sourceSyntax: CFG? = null): CppSuffixGrammar {
  val epsilon = setOf("START" to emptyList<String>()).freeze()
  return CppSuffixGrammar(
    bounded = epsilon.boundedAcyclic(0),
    rawPrefix = prefix,
    projectedPrefix = projectCppTokens(prefix),
    templateTokens = 0,
    sourceSyntax = sourceSyntax ?: epsilon
  )
}

/**
 * Builds an acyclic, depth-indexed expression grammar from clang's cursor-local facts. Unlike a
 * broad syntactic grammar, every emitted identifier has a scoped declaration and every emitted
 * call has arguments assignable to its parameter types. Recursive-looking fluent chains are
 * unrolled to a fixed depth, so the generated language (and its Galoisenne parse forest) is finite.
 */
private class SemanticCppGrammar(
  private val context: CppCompletionContext,
  private val prefix: List<CppToken>
) {
  private val productions = linkedSetOf<Pair<String, List<String>>>()
  private val values = linkedSetOf<CppReference>()
  private val functions = linkedSetOf<CppReference>()
  private val members = linkedSetOf<CppReference>()
  private val constructors = linkedSetOf<CppReference>()
  private val rawTypes = linkedSetOf<String>()
  private val spellings = mutableMapOf<String, LinkedHashSet<String>>()
  private val normalizedTypes = mutableMapOf<String, String?>()
  private val typeKeysByCanonicalId = mutableMapOf<String, String>()
  private val canonicalIdByTypeKey = mutableMapOf<String, String>()
  private val typeAliases = mutableMapOf<String, String>()
  private val pointerTypes = mutableMapOf<PointerShape, String>()
  private val pointerShapes = mutableMapOf<String, PointerShape>()
  private val pointerInfos = mutableMapOf<String, CppTypeInfo>()
  private val sourceSpellableTypes = linkedSetOf<String>()
  private val tokenizedNames = mutableMapOf<String, List<String>>()
  private val typeSpellingSymbols = mutableMapOf<String, String>()
  private val compatibleTypes = mutableMapOf<String, List<String>>()
  private val argumentChoices = mutableMapOf<List<CppParameter>, List<List<String>>>()
  private val receiverChoices = mutableMapOf<Pair<String, String>, List<Pair<String, String>>>()
  private val qualifiedChoices = mutableMapOf<String, String>()
  private lateinit var typeSymbols: Map<String, String>

  private data class PointerShape(
    val pointee: String,
    val isConst: Boolean = false,
    val isVolatile: Boolean = false
  )

  private data class Cv(val isConst: Boolean = false, val isVolatile: Boolean = false) {
    val code: String get() = when {
      isConst && isVolatile -> "CV"
      isConst -> "C"
      isVolatile -> "V"
      else -> "U"
    }
  }

  private enum class ValueCategory { LVALUE, RVALUE }

  private val cvVariants = listOf(Cv(), Cv(isConst = true), Cv(isVolatile = true), Cv(true, true))

  /** Schema-v1 Sema facts are exact; presentation strings are legacy input only. */
  private val structuredTypes = context.completionKind != null || sequenceOf(
    context.preferredTypeInfo,
    context.baseTypeInfo,
    context.enclosingReturnTypeInfo,
    context.enclosingClassTypeInfo,
    context.thisTypeInfo
  ).any { it != null } ||
    (context.values + context.types + context.functions + context.completions +
      context.membersByType.flatMap(CppTypeMembers::members))
      .any { it.typeInfo != null || it.returnTypeInfo != null || it.ownerTypeInfo != null }

  private val aliases = linkedMapOf<String, String>()
  private val abstractTypes = linkedSetOf<String>()
  private val enumTypes = linkedSetOf<String>()
  private val defaultConstructibleTypes by lazy {
    if (structuredTypes) emptySet()
    else context.defaultConstructibleTypes.mapNotNullTo(linkedSetOf(), ::canonicalType)
  }
  private val explicitConversions by lazy {
    buildSet {
      context.conversions.forEach { conversion ->
        val from = canonicalType(conversion.from)
        val to = canonicalType(conversion.to)
        if (from != null && to != null) add(from to to)
      }
      if (!structuredTypes)
        aliases.forEach { (alias, target) -> add(alias to target); add(target to alias) }
    }
  }

  fun build(): CFG {
    collectFacts()
    resolvePointerShapes()
    listOf("bool", "char", "int", "double", "const char *").forEach(::recordType)
    // A pointer type is language syntax over an accessible pointee type, not a new declaration.
    // It guarantees a sound declaration completion even when Sema reports no constructor facts.
    rawTypes.toList().filterNot { it.typeShape() == "void" || isPointer(it) }
      .forEach(::recordPointerType)
    typeSymbols = rawTypes.sorted().mapIndexed { index, type -> type to "TYPE_$index" }.toMap()

    production("START", "SEMANTIC_STATEMENT")
    addAtoms()
    addBooleanCondition(0)
    for (depth in 1..CPP_SEMANTIC_DEPTH) {
      inheritExpressions(depth)
      addLanguageExpressions(depth)
      addCalls(depth)
      addMemberAccesses(depth)
      addOperators(depth)
      addBooleanCondition(depth)
    }
    addStatements()
    return finiteAcyclicCnf(productions)
  }

  /** The endpoint is the authority: this layer classifies facts, but never manufactures a name. */
  private fun collectFacts() {
    fun add(reference: CppReference, owner: String? = reference.ownerType) {
      val fact = if (owner == reference.ownerType) reference else reference.copy(ownerType = owner)
      if (fact.denotesConstructor()) {
        constructors += fact
      } else {
        if (fact.denotesMember()) members += fact
        if (fact.denotesCallable() && (!fact.denotesMember() || fact.isStaticFact())) functions += fact
        val implicitOwnerId = context.thisTypeInfo?.pointeeCanonicalId
        val ownerId = fact.ownerTypeInfo.semanticId()
        val implicitOwner = canonicalType(context.thisType)?.rawPointee()
          ?.removePrefix("const ")?.removePrefix("volatile ")
        val ownerType = canonicalType(fact.canonicalOwnerType ?: fact.ownerType)
        val isImplicitMember = if (implicitOwnerId != null && ownerId != null)
          implicitOwnerId == ownerId else ownerType == implicitOwner
        if (fact.denotesValue() && !fact.denotesCallable() &&
          (!fact.denotesMember() || fact.isStaticFact() || isImplicitMember)) values += fact
      }
      recordReferenceTypes(fact)
    }

    context.values.forEach(::add)
    context.functions.forEach(::add)
    context.types.forEach { reference ->
      recordTypeReference(reference)
      if (reference.denotesConstructor() && isConcrete(reference.typeInfo)) constructors += reference
    }
    context.completions.forEach(::add)
    context.membersByType.forEach { group ->
      if (!structuredTypes) recordType(group.type)
      group.members.forEach { add(it, it.ownerType ?: group.type) }
    }
    context.receiver?.let { receiver ->
      if (!structuredTypes) recordType(receiver.type)
      receiver.members.forEach { add(it, it.ownerType ?: receiver.type) }
    }

    if (!structuredTypes) {
      context.typeNames.forEach(::recordType)
      context.expectedTypes.forEach(::recordType)
      context.requiredTypes.forEach(::recordType)
      context.conversions.forEach { recordType(it.from); recordType(it.to) }
    }
    recordSemanticType(context.preferredType, context.canonicalPreferredType, context.preferredTypeInfo)
    recordSemanticType(context.baseType, context.canonicalBaseType, context.baseTypeInfo)
    recordSemanticType(
      context.enclosingReturnType,
      context.canonicalEnclosingReturnType,
      context.enclosingReturnTypeInfo
    )
    recordSemanticType(
      context.enclosingClassType,
      context.canonicalEnclosingClassType,
      context.enclosingClassTypeInfo
    )
    recordSemanticType(context.thisType, context.canonicalThisType, context.thisTypeInfo)

    // Taking the address of a declaration is a language operation. Its spelling contains no new
    // identifier, and recording it before indexing types keeps `&value` fully generic.
    values.forEach { value ->
      val type = canonicalType(value.semanticType(), value.typeInfo) ?: return@forEach
      if (type.typeShape() != "void") recordPointerType(type)
    }
    deduplicate(values)
    deduplicate(functions)
    deduplicate(members)
    deduplicate(constructors)
  }

  private fun recordTypeReference(reference: CppReference) {
    if (!isConcrete(reference.typeInfo)) return
    val canonical = reference.canonicalType ?: reference.type ?: reference.name
    recordType(reference.name, canonical, reference.typeInfo)
    recordType(reference.type, canonical, reference.typeInfo)
    if (reference.abstract) canonicalType(canonical, reference.typeInfo)?.let(abstractTypes::add)
    if (reference.kind.contains("enum", ignoreCase = true))
      canonicalType(canonical, reference.typeInfo)?.let(enumTypes::add)
    if (reference.kind.contains("alias", ignoreCase = true)) {
      val alias = canonicalType(reference.name, reference.typeInfo)
      val target = canonicalType(
        reference.canonicalType ?: reference.type ?: reference.detail,
        reference.typeInfo
      )
      if (alias != null && target != null && alias != target) {
        aliases[alias] = target
      }
    }
    recordReferenceTypes(reference)
  }

  private fun recordReferenceTypes(reference: CppReference) {
    if (!reference.denotesCallable())
      recordSemanticType(reference.type, reference.canonicalType, reference.typeInfo)
    recordSemanticType(reference.returnType, reference.canonicalReturnType, reference.returnTypeInfo)
    recordSemanticType(reference.ownerType, reference.canonicalOwnerType, reference.ownerTypeInfo)
    reference.parameters.forEach { recordSemanticType(it.type, it.canonicalType, it.typeInfo) }
  }

  private fun recordSemanticType(display: String?, canonical: String?, info: CppTypeInfo?) {
    if (!structuredTypes || info != null) recordType(display, canonical, info)
  }

  private fun isConcrete(info: CppTypeInfo?): Boolean =
    info.isConcrete() && (!structuredTypes || info != null)

  private fun recordType(display: String?, canonical: String? = null, info: CppTypeInfo? = null) {
    if (info?.isDependent == true || info?.isInstantiationDependent == true) return
    val normalized = cppType(canonical ?: display) ?: return
    val canonicalId = info.semanticId()
    if (info != null && canonicalId == null) return
    val type = canonicalId?.let { id -> typeKeysByCanonicalId.getOrPut(id) {
      val key = if (normalized !in canonicalIdByTypeKey) normalized else "$normalized\u0000$id"
      canonicalIdByTypeKey[key] = id
      key
    } } ?: normalized
    if (normalized !in typeAliases) typeAliases[normalized] = type
    cppType(display)?.let { if (it !in typeAliases) typeAliases[it] = type }
    rawTypes += type
    if (info?.kind == "pointer") {
      pointerInfos[type] = info
    }
    if (info?.isSourceSpellable != false) cppType(display)?.let { spelling ->
      spellings.getOrPut(type, ::linkedSetOf) += spelling
      sourceSpellableTypes += type
    }
  }

  private fun resolvePointerShapes() = pointerInfos.forEach { (type, info) ->
    info.pointeeCanonicalId?.let(typeKeysByCanonicalId::get)?.let { pointee ->
      val shape = PointerShape(pointee, info.pointeeIsConst, info.pointeeIsVolatile)
      pointerTypes[shape] = type
      pointerShapes[type] = shape
    }
  }

  private fun recordPointerType(pointee: String) {
    val shape = PointerShape(pointee)
    val pointer = pointerTypes.getOrPut(shape) {
      "${pointee.typeShape()} *\u0000ptr:${canonicalIdByTypeKey[pointee] ?: pointee}"
    }
    pointerShapes[pointer] = shape
    rawTypes += pointer
    if (pointee in sourceSpellableTypes) {
      spellings.getOrPut(pointer, ::linkedSetOf) += spellings.getValue(pointee).map { "$it *" }
      sourceSpellableTypes += pointer
    }
  }

  private fun deduplicate(references: MutableSet<CppReference>) {
    val unique = linkedMapOf<String, CppReference>()
    references.forEach { reference ->
      val key = buildString {
        append(reference.id ?: reference.semanticName()); append('|')
        append(reference.ownerTypeInfo.semanticId() ?: canonicalType(reference.ownerType)); append('|')
        append(reference.returnTypeInfo.semanticId() ?: reference.typeInfo.semanticId()
          ?: canonicalType(reference.semanticReturnType() ?: reference.semanticType())); append('|')
        reference.parameters.forEach {
          append(it.typeInfo.semanticId() ?: canonicalType(it.semanticType())); append(';')
        }
      }
      val previous = unique[key]
      if (previous == null || previous.provenance == "index" && reference.provenance == "sema")
        unique[key] = reference
    }
    references.clear()
    references += unique.values
  }

  private fun addAtoms() {
    values.forEach { reference ->
      if (!isConcrete(reference.typeInfo)) return@forEach
      val type = canonicalType(reference.semanticType(), reference.typeInfo)
        ?.takeIf { it in typeSymbols } ?: return@forEach
      val name = reference.semanticName().cachedNameTokens()
      if (name.isEmpty()) return@forEach
      val cv = reference.typeInfo?.let { Cv(it.isConst, it.isVolatile) }
        ?: Cv(isConst = !reference.isMutableValueInContext())
      exactPostfixExpression(type, 0, ValueCategory.LVALUE, cv, name)
      val pointer = pointerTypes[PointerShape(type)]?.takeIf { it in typeSymbols }
      if (pointer != null && type.typeShape() != "void") {
        movableStableExpression(pointer, 0, listOf("&") + name)
        movablePostfixExpression(pointer, 0, listOf("(", "&") + name + ")")
      }
    }

    functions.filter { it.isStaticFact() && !it.denotesCallable() }.forEach { reference ->
      val type = canonicalType(reference.semanticType(), reference.typeInfo)
        ?.takeIf { it in typeSymbols } ?: return@forEach
      postfixExpression(type, 0, reference.semanticName().cachedNameTokens())
    }
    canonicalType(context.thisType, context.thisTypeInfo)?.takeIf { it in typeSymbols }?.let { type ->
      movablePostfixExpression(type, 0, listOf("this"))
    }
    typeSymbols.keys.forEach { type ->
      when {
        type.typeShape() == "bool" -> movablePostfixExpression(type, 0, listOf(CPP_BOOLEAN))
        type.typeShape() == "char" -> movablePostfixExpression(type, 0, listOf(CPP_CHARACTER))
        type.isIntegralCppType() -> movablePostfixExpression(type, 0, listOf(CPP_INTEGER))
        type.isFloatingCppType() -> {
          movablePostfixExpression(type, 0, listOf(CPP_FLOATING))
          movablePostfixExpression(type, 0, listOf(CPP_INTEGER))
        }
        type.typeShape() == "const char *" -> movablePostfixExpression(type, 0, listOf(CPP_STRING))
      }
    }
  }

  private fun inheritExpressions(depth: Int) = typeSymbols.keys.forEach { type ->
    production(expression(type, depth), expression(type, depth - 1))
    production(postfix(type, depth), postfix(type, depth - 1))
    production(stable(type, depth), stable(type, depth - 1))
    production(glvalue(type, depth), glvalue(type, depth - 1))
    production(rvalue(type, depth), rvalue(type, depth - 1))
    production(lvalue(type, depth), lvalue(type, depth - 1))
    production(movable(type, depth), movable(type, depth - 1))
    production(mutablePostfix(type, depth), mutablePostfix(type, depth - 1))
    cvVariants.forEach { cv -> ValueCategory.entries.forEach { category ->
      production(qualified(type, depth, category, cv), qualified(type, depth - 1, category, cv))
      production(
        qualifiedPostfix(type, depth, category, cv),
        qualifiedPostfix(type, depth - 1, category, cv)
      )
    } }
  }

  private fun addLanguageExpressions(depth: Int) {
    val previous = depth - 1
    typeSymbols.keys.forEach { type ->
      postfixExpression(type, depth, listOf("(", expression(type, previous), ")"))
      cvVariants.forEach { cv -> ValueCategory.entries.forEach { category ->
        exactPostfixExpression(
          type, depth, category, cv,
          listOf("(", qualified(type, previous, category, cv), ")")
        )
      } }
      if (type.isNumericCppType()) {
        val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: type
        listOf("+", "-").forEach { operator ->
          movableStableExpression(result, depth, listOf(operator, stable(type, previous)))
        }
      }
      if (type.isIntegralCppType()) {
        val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: type
        movableStableExpression(result, depth, listOf("~", stable(type, previous)))
      }
      if (isPointer(type)) {
        val pointer = pointerShapes[type]
        val rawPointee = type.rawPointee() ?: return@forEach
        val pointee = pointer?.pointee
          ?: canonicalType(rawPointee)?.takeIf { it in typeSymbols }
          ?: return@forEach
        val rhs = listOf("*", stable(type, previous))
        val cv = pointer?.let { Cv(it.isConst, it.isVolatile) }
          ?: Cv(
            isConst = rawPointee.startsWith("const "),
            isVolatile = rawPointee.startsWith("volatile ")
          )
        exactStableExpression(pointee, depth, ValueCategory.LVALUE, cv, rhs)
      }
    }

    val numeric = typeSymbols.keys.filter { it.isArithmeticCppType() || it in enumTypes }
    numeric.filter { it in sourceSpellableTypes }.forEach { target -> numeric.forEach { source ->
      movablePostfixExpression(
        target, depth,
        listOf("static_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
      )
    } }
  }

  private fun addCalls(depth: Int) {
    functions.filter { !it.denotesMember() || it.isStaticFact() }.forEach { callable ->
      if (!isConcrete(callable.returnTypeInfo)) return@forEach
      val result = canonicalType(callable.semanticReturnType(), callable.returnTypeInfo)
        ?.takeIf { it in typeSymbols }
        ?: return@forEach
      val name = callable.semanticName().cachedNameTokens()
      if (name.isEmpty() || callable.operatorToken() != null) return@forEach
      addCallProductions(
        result, depth, name, callable.parameters,
        callable.semanticReturnType(), callable.returnTypeInfo
      )
    }

    constructors.groupBy {
      if (!isConcrete(it.ownerTypeInfo ?: it.returnTypeInfo)) null
      else canonicalType(
        it.canonicalOwnerType ?: it.ownerType ?: it.semanticReturnType() ?: it.name,
        it.ownerTypeInfo ?: it.returnTypeInfo
      )
    }
      .forEach { (type, overloads) ->
        if (type == null || type !in typeSymbols || type in abstractTypes ||
          type !in sourceSpellableTypes) return@forEach
        overloads.forEach { constructor ->
          argumentTypeChoices(constructor.parameters).forEach { actuals ->
            val arguments = commaSeparatedExpressions(actuals, constructor.parameters, depth - 1)
            movablePostfixExpression(
              type, depth, listOf(typeSpelling(type), "{") + arguments + "}"
            )
            movablePostfixExpression(
              type, depth, listOf(typeSpelling(type), "(") + arguments + ")"
            )
          }
        }
      }
  }

  private fun addMemberAccesses(depth: Int) {
    members.forEach { member ->
      if (!isConcrete(member.ownerTypeInfo) ||
        !isConcrete(member.returnTypeInfo) && member.denotesCallable() ||
        !isConcrete(member.typeInfo) && !member.denotesCallable()) return@forEach
      val owner = canonicalType(
        member.canonicalOwnerType ?: member.ownerType ?: context.canonicalBaseType ?: context.baseType,
        member.ownerTypeInfo ?: context.baseTypeInfo
      )
        ?: return@forEach
      val result = canonicalType(
        member.semanticReturnType() ?: member.semanticType(),
        if (member.denotesCallable()) member.returnTypeInfo else member.typeInfo
      )
        ?.takeIf { it in typeSymbols } ?: return@forEach
      val memberName = member.semanticName().substringAfterLast("::")
      val receivers = receiversFor(owner, member)

      fun receiverHead(receiverType: String, connector: String): List<String> {
        val symbol = if (connector != ".") postfix(receiverType, depth - 1)
        else memberReceiver(receiverType, depth - 1, member)
        return listOf(symbol, connector)
      }

      if (memberName == "operator()" && member.denotesCallable()) {
        receivers.forEach { (receiverType, connector) ->
          val receiver = receiverHead(receiverType, connector).first()
          val callableHead = if (connector == ".") listOf(receiver)
          else listOf("(", "*", receiver, ")")
          addCallProductions(
            result, depth, callableHead,
            member.parameters,
            member.semanticReturnType(), member.returnTypeInfo
          )
        }
        return@forEach
      }
      if (memberName == "operator[]" && member.parameters.size == 1) {
        val parameter = member.parameters.single()
        val actuals = assignableTypes(
          canonicalType(parameter.semanticType(), parameter.typeInfo) ?: return@forEach
        )
        receivers.forEach { (receiverType, connector) -> actuals.forEach { actual ->
          val receiver = receiverHead(receiverType, connector).first()
          val base = if (connector == ".") listOf(receiver) else listOf("(", "*", receiver, ")")
          val rhs = base + listOf(
            "[",
            argumentExpression(actual, parameter, depth - 1), "]"
          )
          emitPostfixResult(
            result, depth, rhs, member.semanticReturnType(), member.returnTypeInfo
          )
        } }
        return@forEach
      }
      if (!IDENTIFIER_REGEX.matches(memberName)) return@forEach

      val thisPointee = context.thisTypeInfo?.pointeeCanonicalId
        ?.let(typeKeysByCanonicalId::get)
        ?: canonicalType(context.thisType, context.thisTypeInfo)?.rawPointee()
      if (thisPointee == owner && member.refQualifier != "&&" &&
        (!member.denotesCallable() || member.acceptsCv(Cv(
          context.thisTypeInfo?.pointeeIsConst == true,
          context.thisTypeInfo?.pointeeIsVolatile == true
        )))) {
        val head = memberName.cachedNameTokens()
        if (member.denotesCallable()) addCallProductions(
          result, depth, head, member.parameters,
          member.semanticReturnType(), member.returnTypeInfo
        ) else emitField(
          result, depth, head, ValueCategory.LVALUE,
          Cv(
            context.thisTypeInfo?.pointeeIsConst == true,
            context.thisTypeInfo?.pointeeIsVolatile == true
          ), member
        )
      }
      receivers.forEach { (receiverType, connector) ->
        val name = memberName.cachedNameTokens()
        if (member.denotesCallable()) {
          addCallProductions(
            result, depth, receiverHead(receiverType, connector) + name, member.parameters,
            member.semanticReturnType(), member.returnTypeInfo
          )
        } else if (connector == "->") {
          val baseCv = pointerShapes[receiverType]?.let { Cv(it.isConst, it.isVolatile) } ?: Cv()
          emitField(
            result, depth,
            listOf(postfix(receiverType, depth - 1), connector) + name,
            ValueCategory.LVALUE, baseCv, member
          )
        } else {
          ValueCategory.entries.forEach { category -> cvVariants.forEach { baseCv ->
            emitField(
              result, depth,
              listOf(qualifiedPostfix(receiverType, depth - 1, category, baseCv), connector) + name,
              category, baseCv, member
            )
          } }
        }
      }
    }
  }

  private fun addCallProductions(
    result: String,
    depth: Int,
    head: List<String>,
    parameters: List<CppParameter>,
    rawReturnType: String?,
    returnTypeInfo: CppTypeInfo? = null
  ) {
    argumentTypeChoices(parameters).forEach { actuals ->
      val rhs = head + "(" + commaSeparatedExpressions(actuals, parameters, depth - 1) + ")"
      emitPostfixResult(result, depth, rhs, rawReturnType, returnTypeInfo)
    }
  }

  private fun emitPostfixResult(
    result: String,
    depth: Int,
    rhs: List<String>,
    rawType: String?,
    info: CppTypeInfo?
  ) = when {
    info?.kind == "lvalueReference" -> exactPostfixExpression(
      result, depth, ValueCategory.LVALUE, Cv(info.isConst, info.isVolatile), rhs
    )
    info?.kind == "rvalueReference" || info != null -> exactPostfixExpression(
      result, depth, ValueCategory.RVALUE, Cv(info.isConst, info.isVolatile), rhs
    )
    rawType.isConstRvalueReferenceType() -> rvaluePostfixExpression(result, depth, rhs)
    rawType.isConstLvalueReferenceType() -> glvaluePostfixExpression(result, depth, rhs)
    rawType.isLvalueReferenceType() -> lvalueExpression(result, depth, rhs)
    else -> movablePostfixExpression(result, depth, rhs)
  }

  private fun emitField(
    result: String,
    depth: Int,
    rhs: List<String>,
    category: ValueCategory,
    baseCv: Cv,
    field: CppReference
  ) {
    val declared = field.typeInfo?.let { Cv(it.isConst, it.isVolatile) } ?: Cv()
    val cv = Cv(
      declared.isConst || baseCv.isConst && field.isMutableField != true,
      declared.isVolatile || baseCv.isVolatile
    )
    exactPostfixExpression(result, depth, category, cv, rhs)
  }

  private fun argumentTypeChoices(parameters: List<CppParameter>): List<List<String>> =
    argumentChoices.getOrPut(parameters) {
      // clang must expand dependent packs before they can be represented as concrete CFG edges.
      if (parameters.any {
          it.isPack || !isConcrete(it.typeInfo) ||
            canonicalType(it.semanticType(), it.typeInfo) !in typeSymbols
        })
        return@getOrPut emptyList()
      val required = parameters.indexOfFirst(CppParameter::isOptional)
        .let { if (it < 0) parameters.size else it }
      (required..parameters.size).flatMap { arity ->
        parameters.take(arity).fold(listOf(emptyList())) { choices, parameter ->
          val expected = canonicalType(parameter.semanticType(), parameter.typeInfo)!!
          choices.flatMap { chosen -> assignableTypes(expected).map { chosen + it } }
        }
      }
    }

  private fun commaSeparatedExpressions(
    types: List<String>,
    parameters: List<CppParameter>,
    depth: Int
  ): List<String> = buildList {
    types.forEachIndexed { index, type ->
      if (index > 0) add(",")
      add(argumentExpression(type, parameters[index], depth))
    }
  }

  private fun argumentExpression(actual: String, parameter: CppParameter, depth: Int): String {
    val expected = parameter.type.ifBlank { parameter.semanticType() }
    parameter.typeInfo?.let { info -> when (info.kind) {
      "lvalueReference" -> return qualifiedReferenceExpression(
        actual, depth, ValueCategory.LVALUE, Cv(info.isConst, info.isVolatile),
        includeRvalues = info.isConst && !info.isVolatile
      )
      "rvalueReference" -> return qualifiedReferenceExpression(
        actual, depth, ValueCategory.RVALUE, Cv(info.isConst, info.isVolatile)
      )
    } }
    return when {
      expected.isLvalueReferenceType() && !expected.isConstLvalueReferenceType() -> lvalue(actual, depth)
      expected.isConstRvalueReferenceType() -> rvalue(actual, depth)
      expected.trim().endsWith("&&") -> movable(actual, depth)
      else -> expression(actual, depth)
    }
  }

  private fun qualifiedReferenceExpression(
    type: String,
    depth: Int,
    category: ValueCategory,
    target: Cv,
    includeRvalues: Boolean = false
  ): String {
    val key = "$type:$depth:${category.name}:${target.code}:$includeRvalues"
    return qualifiedChoices.getOrPut(key) {
      val symbol = "REFERENCE_CHOICE_${qualifiedChoices.size}"
      val categories = if (includeRvalues) ValueCategory.entries else listOf(category)
      categories.forEach { actualCategory -> cvVariants.filter { actual ->
        (!actual.isConst || target.isConst) && (!actual.isVolatile || target.isVolatile)
      }.forEach { cv -> production(symbol, qualified(type, depth, actualCategory, cv)) } }
      symbol
    }
  }

  private fun addOperators(depth: Int) {
    val previous = depth - 1
    typeSymbols.keys.filter(String::isNumericCppType).forEach { type ->
      val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: type
      val arithmetic = if (type.isIntegralCppType()) listOf("+", "-", "*", "/", "%")
      else listOf("+", "-", "*", "/")
      arithmetic.forEach { operator ->
        movableStableExpression(result, depth, listOf(stable(type, previous), operator, stable(type, previous)))
      }
      if (type.isIntegralCppType()) listOf("&", "|", "^", "<", ">>").forEach { operator ->
        val tokens = if (operator == "<") listOf("<", "<") else listOf(operator)
        movableStableExpression(
          result, depth, listOf(stable(type, previous)) + tokens + stable(type, previous)
        )
      }
      booleanType()?.let { boolean -> listOf("==", "!=", "<", "<=", ">", ">=").forEach { operator ->
        movableStableExpression(boolean, depth, listOf(stable(type, previous), operator, stable(type, previous)))
      }
      }
    }

    (functions + members).forEach { callable ->
      val operator = callable.operatorToken() ?: return@forEach
      val result = canonicalType(callable.semanticReturnType(), callable.returnTypeInfo)
        ?.takeIf { it in typeSymbols }
        ?: return@forEach
      val operands = if (callable.denotesMember()) {
        val owner = canonicalType(
          callable.canonicalOwnerType ?: callable.ownerType,
          callable.ownerTypeInfo
        ) ?: return@forEach
        if (callable.parameters.size != 1) return@forEach
        listOf(owner, canonicalType(
          callable.parameters.single().semanticType(),
          callable.parameters.single().typeInfo
        ) ?: return@forEach)
      } else {
        if (callable.parameters.size != 2) return@forEach
        callable.parameters.map {
          canonicalType(it.semanticType(), it.typeInfo) ?: return@forEach
        }
      }
      if (operands.any { it !in typeSymbols }) return@forEach
      val operatorTokens = if (operator == "<<") listOf("<", "<") else listOf(operator)
      fun parameterOperand(type: String, parameter: CppParameter): String {
        val spelling = parameter.type.ifBlank { parameter.semanticType() }
        parameter.typeInfo?.let { info -> when (info.kind) {
          "lvalueReference" -> return qualifiedReferenceExpression(
            type, previous, ValueCategory.LVALUE, Cv(info.isConst, info.isVolatile),
            includeRvalues = info.isConst && !info.isVolatile
          )
          "rvalueReference" -> return qualifiedReferenceExpression(
            type, previous, ValueCategory.RVALUE, Cv(info.isConst, info.isVolatile)
          )
        } }
        return when {
          spelling.isLvalueReferenceType() && !spelling.isConstLvalueReferenceType() -> lvalue(type, previous)
          spelling.isConstRvalueReferenceType() -> rvalue(type, previous)
          spelling.trim().endsWith("&&") -> movable(type, previous)
          else -> stable(type, previous)
        }
      }
      val left = if (callable.denotesMember()) {
        memberObject(operands[0], previous, callable)
      } else parameterOperand(operands[0], callable.parameters[0])
      val rightParameter = callable.parameters[if (callable.denotesMember()) 0 else 1]
      val rhs = listOf(left) + operatorTokens + parameterOperand(operands[1], rightParameter)
      when {
        callable.returnTypeInfo.isLvalueReference() -> exactStableExpression(
          result, depth, ValueCategory.LVALUE,
          Cv(callable.returnTypeInfo?.isConst == true, callable.returnTypeInfo?.isVolatile == true), rhs
        )
        callable.returnTypeInfo?.kind == "rvalueReference" -> exactStableExpression(
          result, depth, ValueCategory.RVALUE,
          Cv(callable.returnTypeInfo.isConst, callable.returnTypeInfo.isVolatile), rhs
        )
        callable.returnTypeInfo != null -> exactStableExpression(
          result, depth, ValueCategory.RVALUE,
          Cv(callable.returnTypeInfo.isConst, callable.returnTypeInfo.isVolatile), rhs
        )
        callable.semanticReturnType().isConstRvalueReferenceType() ->
          stableRvalueExpression(result, depth, rhs)
        callable.semanticReturnType().isConstLvalueReferenceType() ->
          stableGlvalueExpression(result, depth, rhs)
        callable.semanticReturnType().isLvalueReferenceType() -> stableLvalueExpression(result, depth, rhs)
        else -> movableStableExpression(result, depth, rhs)
      }
    }

    booleanType()?.let { boolean ->
      listOf("&&", "||").forEach { operator ->
        movableStableExpression(boolean, depth, listOf(condition(previous), operator, condition(previous)))
      }
      movableStableExpression(boolean, depth, listOf("!", condition(previous)))
      typeSymbols.keys.filter(::isPointer).forEach { pointer ->
        listOf("==", "!=").forEach { operator ->
          movableStableExpression(
            boolean, depth, listOf(stable(pointer, previous), operator, CPP_NULLPTR)
          )
        }
      }
    }

    if (booleanType() != null) typeSymbols.keys.filterNot { it.typeShape() == "void" }.forEach { type ->
      expression(
        type, depth,
        listOf(condition(previous), "?", expression(type, previous), ":", expression(type, previous))
      )
    }
  }

  private fun addBooleanCondition(depth: Int) {
    if (booleanType() == null) return
    typeSymbols.keys.filter { it.isArithmeticCppType() || isPointer(it) }
      .forEach { type -> production(condition(depth), stable(type, depth)) }
  }

  private fun addStatements() {
    val requiredName = requiredDeclarator()
    val names = requiredName?.let { listOf(encodeIdentifier(it)) } ?: buildList {
      add(CPP_FRESH)
      prefix.filter { it.kind == CppTokenKind.IDENTIFIER && it.text !in CPP_KEYWORDS }
        .mapTo(this) { encodeIdentifier(it.text) }
    }.distinct()
    addDeclarations(names)
    if (requiredName != null) {
      production("SEMANTIC_STATEMENT", "SIMPLE_STATEMENT")
      return
    }

    typeSymbols.keys.forEach { type -> production("SIMPLE_STATEMENT", expression(type, CPP_SEMANTIC_DEPTH), ";") }
    addAssignments()
    addReturns()
    production("SEMANTIC_STATEMENT", "SIMPLE_STATEMENT")
    if (booleanType() != null)
      production("SEMANTIC_STATEMENT", "if", "(", condition(CPP_SEMANTIC_DEPTH), ")", "SIMPLE_STATEMENT")
  }

  private fun requiredDeclarator(): String? {
    val diagnostic = sequenceOf(context.requiredIdentifier)
      .plus(context.unresolvedIdentifiers.asSequence())
      .filterNotNull().firstOrNull(IDENTIFIER_REGEX::matches)
    if (prefix.isEmpty()) return diagnostic
    val projected = projectCppTokens(prefix)
    val typePrefix = sourceSpellableTypes.asSequence()
      .filterNot { it.typeShape() == "void" }
      .flatMap { spellings[it].orEmpty().asSequence() }
      .flatMap { it.typeSpellingVariants().asSequence() }
      .toList()
      .sortedByDescending(List<String>::size)
      .firstOrNull { candidate ->
        val common = minOf(candidate.size, projected.size)
        candidate.take(common) == projected.take(common)
      } ?: return null
    return projected.getOrNull(typePrefix.size)?.removePrefix("@id:") ?: diagnostic
  }

  private fun addDeclarations(names: List<String>) {
    val depth = CPP_SEMANTIC_DEPTH
    val constructorsByType = constructors.groupBy {
      canonicalType(
        it.canonicalOwnerType ?: it.ownerType ?: it.semanticReturnType() ?: it.name,
        it.ownerTypeInfo ?: it.returnTypeInfo
      )
    }
    names.forEach { name ->
      typeSymbols.keys.filterNot {
        it.typeShape() == "void" || it in abstractTypes || it !in sourceSpellableTypes
      }.forEach { type ->
        val spelling = listOf(typeSpelling(type))
        val constructors = constructorsByType[type].orEmpty()
        if (type in defaultConstructibleTypes || type.isLanguageDefaultConstructible() ||
          constructors.any { constructor -> constructor.parameters.all(CppParameter::isOptional) }) {
          production("SIMPLE_STATEMENT", spelling + name + ";")
          production("SIMPLE_STATEMENT", spelling + name + listOf("{", "}", ";"))
        }
        assignableTypes(type).forEach { actual ->
          production("SIMPLE_STATEMENT", spelling + name + listOf("=", expression(actual, depth), ";"))
          production(
            "SIMPLE_STATEMENT",
            listOf("const") + spelling + name + listOf("=", expression(actual, depth), ";")
          )
          if (!isPointer(type)) {
            production("SIMPLE_STATEMENT", spelling + listOf("&", name, "=", lvalue(actual, depth), ";"))
            production(
              "SIMPLE_STATEMENT",
              listOf("const") + spelling + listOf("&", name, "=", expression(actual, depth), ";")
            )
          }
        }
        if (isPointer(type)) {
          production("SIMPLE_STATEMENT", spelling + name + listOf("=", CPP_NULLPTR, ";"))
          production("SIMPLE_STATEMENT", spelling + name + listOf("{", CPP_NULLPTR, "}", ";"))
        }
        constructors.forEach { constructor -> argumentTypeChoices(constructor.parameters).forEach { actuals ->
          val arguments = commaSeparatedExpressions(actuals, constructor.parameters, depth)
          production("SIMPLE_STATEMENT", spelling + name + listOf("{") + arguments + listOf("}", ";"))
          production("SIMPLE_STATEMENT", spelling + name + listOf("(") + arguments + listOf(")", ";"))
        } }
      }

      typeSymbols.keys.filterNot { it.typeShape() == "void" || it in abstractTypes }.forEach { actual ->
        production("SIMPLE_STATEMENT", listOf("auto", name, "=", expression(actual, depth), ";"))
      }
      // A type alias introduces a fresh binder and is valid for every clang-spelled type.
      sourceSpellableTypes.forEach { type ->
        production("SIMPLE_STATEMENT", listOf("using", name, "=", typeSpelling(type), ";"))
      }
    }
  }

  private fun addAssignments() {
    val depth = CPP_SEMANTIC_DEPTH
    typeSymbols.keys.forEach { target ->
      assignableTypes(target).forEach { actual ->
        production("SIMPLE_STATEMENT", lvalue(target, depth), "=", expression(actual, depth), ";")
      }
      if (isPointer(target))
        production("SIMPLE_STATEMENT", lvalue(target, depth), "=", CPP_NULLPTR, ";")
      if (target.isArithmeticCppType()) typeSymbols.keys.filter(String::isArithmeticCppType)
        .forEach { actual -> listOf("+=", "-=", "*=", "/=").forEach { operator ->
          production("SIMPLE_STATEMENT", lvalue(target, depth), operator, expression(actual, depth), ";")
        } }
    }
  }

  private fun addReturns() {
    val raw = context.enclosingReturnType ?: return
    val expected = canonicalType(raw, context.enclosingReturnTypeInfo) ?: return
    if (expected.typeShape() == "void") {
      production("SIMPLE_STATEMENT", "return", ";")
      return
    }
    if (expected !in typeSymbols) return
    assignableTypes(expected).forEach { actual ->
      production("SIMPLE_STATEMENT", "return", expression(actual, CPP_SEMANTIC_DEPTH), ";")
    }
    if (isPointer(expected)) production("SIMPLE_STATEMENT", "return", CPP_NULLPTR, ";")
  }

  private fun receiversFor(owner: String, member: CppReference): List<Pair<String, String>> =
    receiverChoices.getOrPut(
      owner to "${member.methodCv().code}:${member.refQualifier.orEmpty()}"
    ) {
      buildList {
        typeSymbols.keys.forEach { candidate ->
          val pointer = pointerShapes[candidate]
          val pointee = pointer?.pointee ?: candidate.rawPointee()?.let(::canonicalType)
          when {
            candidate == owner -> add(candidate to ".")
            member.refQualifier != "&&" && pointee == owner &&
              (!member.denotesCallable() || pointer == null ||
                member.acceptsCv(Cv(pointer.isConst, pointer.isVolatile))) ->
              add(candidate to "->")
          }
        }
      }.distinct()
    }

  private fun memberReceiver(type: String, depth: Int, member: CppReference): String {
    val target = member.methodCv()
    val key = "receiver:$type:$depth:${target.code}:${member.refQualifier.orEmpty()}"
    return qualifiedChoices.getOrPut(key) {
      val symbol = "RECEIVER_CHOICE_${qualifiedChoices.size}"
      val categories = when (member.refQualifier) {
        "&" -> listOf(ValueCategory.LVALUE)
        "&&" -> listOf(ValueCategory.RVALUE)
        else -> ValueCategory.entries
      }
      categories.forEach { category -> cvVariants.filter { member.acceptsCv(it) }.forEach { cv ->
        production(symbol, qualifiedPostfix(type, depth, category, cv))
      } }
      symbol
    }
  }

  private fun memberObject(type: String, depth: Int, member: CppReference): String {
    val target = member.methodCv()
    val key = "object:$type:$depth:${target.code}:${member.refQualifier.orEmpty()}"
    return qualifiedChoices.getOrPut(key) {
      val symbol = "OBJECT_CHOICE_${qualifiedChoices.size}"
      val categories = when (member.refQualifier) {
        "&" -> listOf(ValueCategory.LVALUE)
        "&&" -> listOf(ValueCategory.RVALUE)
        else -> ValueCategory.entries
      }
      categories.forEach { category -> cvVariants.filter { member.acceptsCv(it) }.forEach { cv ->
        production(symbol, qualified(type, depth, category, cv))
      } }
      symbol
    }
  }

  private fun CppReference.methodCv(): Cv = Cv(isConstMember(), isVolatileMember())

  private fun CppReference.acceptsCv(actual: Cv): Boolean = methodCv().let { target ->
    (!actual.isConst || target.isConst) && (!actual.isVolatile || target.isVolatile)
  }

  private fun assignableTypes(expected: String): List<String> = compatibleTypes.getOrPut(expected) {
    typeSymbols.keys.filter { actual -> isAssignable(actual, expected) }
  }

  private fun isAssignable(actual: String, expected: String): Boolean {
    if (actual == expected || actual to expected in explicitConversions) return true
    if (actual.isArithmeticCppType() && expected.isArithmeticCppType()) return true
    if (expected.typeShape() == "bool" && isPointer(actual)) return true
    val from = pointerShapes[actual]
    val to = pointerShapes[expected]
    if (from != null && to != null) return from.pointee == to.pointee &&
      (!from.isConst || to.isConst) && (!from.isVolatile || to.isVolatile)
    return !structuredTypes && isAssignable(actual.typeShape(), expected.typeShape(), explicitConversions)
  }

  private fun CppReference.isMutableValueInContext(): Boolean {
    if (!isMutableLvalue()) return false
    if (!denotesMember() || isStaticFact() || isMutableField == true) return true
    val owner = canonicalType(canonicalOwnerType ?: ownerType, ownerTypeInfo) ?: return true
    context.thisTypeInfo?.pointeeCanonicalId?.let { pointeeId ->
      val sameOwner = pointeeId == ownerTypeInfo.semanticId()
      return !sameOwner || context.thisTypeInfo.pointeeIsConst != true
    }
    val implicitObject = canonicalType(context.thisType)?.rawPointee() ?: return true
    return implicitObject.removePrefix("const ").removePrefix("volatile ") != owner ||
      !implicitObject.startsWith("const ")
  }

  private fun typeSpelling(type: String): String = typeSpellingSymbols.getOrPut(type) {
    val symbol = "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_SPELLING"
    val candidates = spellings[type].orEmpty()
    check(candidates.isNotEmpty()) { "C++ type $type has no Sema-approved source spelling" }
    candidates.flatMap(String::typeSpellingVariants).distinct().forEach { production(symbol, it) }
    symbol
  }

  private fun canonicalType(raw: String?): String? {
    if (raw == null) return null
    val normalized = normalizedTypes.getOrPut(raw) { cppType(raw) } ?: return null
    return typeAliases[normalized] ?: normalized
  }

  private fun canonicalType(raw: String?, info: CppTypeInfo?): String? =
    if (info == null) canonicalType(raw)
    else if (!info.isConcrete()) null
    else info.semanticId()?.let(typeKeysByCanonicalId::get)

  private fun String.typeShape(): String = substringBefore('\u0000')

  private fun booleanType(): String? = typeSymbols.keys.firstOrNull { it.typeShape() == "bool" }
  private fun isPointer(type: String): Boolean = type in pointerShapes || type.isRawPointer()

  private fun String.cachedNameTokens(): List<String> =
    tokenizedNames.getOrPut(this) { cppNameTokens() }

  private fun expression(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_D$depth"
  private fun postfix(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_POSTFIX_D$depth"
  private fun stable(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_STABLE_D$depth"
  private fun glvalue(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_GLVALUE_D$depth"
  private fun rvalue(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_RVALUE_D$depth"
  private fun lvalue(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_LVALUE_D$depth"
  private fun movable(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_MOVABLE_D$depth"
  private fun mutablePostfix(type: String, depth: Int): String =
    "${typeSymbols.getValue(type)}_MUTABLE_POSTFIX_D$depth"
  private fun qualified(type: String, depth: Int, category: ValueCategory, cv: Cv): String =
    "${typeSymbols.getValue(type)}_${category.name}_${cv.code}_D$depth"
  private fun qualifiedPostfix(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv
  ): String = "${typeSymbols.getValue(type)}_POSTFIX_${category.name}_${cv.code}_D$depth"
  private fun condition(depth: Int): String = "BOOLEAN_CONDITION_D$depth"

  private fun expression(type: String, depth: Int, rhs: List<String>) = production(expression(type, depth), rhs)
  private fun postfixExpression(type: String, depth: Int, rhs: List<String>) {
    expression(type, depth, rhs); production(postfix(type, depth), rhs); production(stable(type, depth), rhs)
  }
  private fun stableExpression(type: String, depth: Int, rhs: List<String>) {
    expression(type, depth, rhs); production(stable(type, depth), rhs)
  }
  private fun exactPostfixExpression(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv,
    rhs: List<String>
  ) {
    postfixExpression(type, depth, rhs)
    production(qualified(type, depth, category, cv), rhs)
    production(qualifiedPostfix(type, depth, category, cv), rhs)
    if (category == ValueCategory.LVALUE) {
      production(glvalue(type, depth), rhs)
      if (!cv.isConst) production(lvalue(type, depth), rhs)
      if (!cv.isConst && !cv.isVolatile) production(mutablePostfix(type, depth), rhs)
    } else {
      production(rvalue(type, depth), rhs)
      if (!cv.isConst && !cv.isVolatile) {
        production(movable(type, depth), rhs)
        production(mutablePostfix(type, depth), rhs)
      }
    }
  }
  private fun exactStableExpression(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv,
    rhs: List<String>
  ) {
    stableExpression(type, depth, rhs)
    production(qualified(type, depth, category, cv), rhs)
    if (category == ValueCategory.LVALUE) {
      production(glvalue(type, depth), rhs)
      if (!cv.isConst) production(lvalue(type, depth), rhs)
    } else {
      production(rvalue(type, depth), rhs)
      if (!cv.isConst && !cv.isVolatile) production(movable(type, depth), rhs)
    }
  }
  private fun glvaluePostfixExpression(
    type: String,
    depth: Int,
    rhs: List<String>,
    cv: Cv = Cv(isConst = true)
  ) {
    exactPostfixExpression(type, depth, ValueCategory.LVALUE, cv, rhs)
  }
  private fun stableGlvalueExpression(
    type: String,
    depth: Int,
    rhs: List<String>,
    cv: Cv = Cv(isConst = true)
  ) {
    exactStableExpression(type, depth, ValueCategory.LVALUE, cv, rhs)
  }
  private fun movablePostfixExpression(type: String, depth: Int, rhs: List<String>) {
    exactPostfixExpression(type, depth, ValueCategory.RVALUE, Cv(), rhs)
  }
  private fun rvaluePostfixExpression(
    type: String,
    depth: Int,
    rhs: List<String>,
    cv: Cv = Cv(isConst = true)
  ) {
    exactPostfixExpression(type, depth, ValueCategory.RVALUE, cv, rhs)
  }
  private fun movableStableExpression(type: String, depth: Int, rhs: List<String>) {
    exactStableExpression(type, depth, ValueCategory.RVALUE, Cv(), rhs)
  }
  private fun stableRvalueExpression(
    type: String,
    depth: Int,
    rhs: List<String>,
    cv: Cv = Cv(isConst = true)
  ) {
    exactStableExpression(type, depth, ValueCategory.RVALUE, cv, rhs)
  }
  private fun lvalueExpression(
    type: String,
    depth: Int,
    rhs: List<String>,
    isVolatile: Boolean = false
  ) {
    exactPostfixExpression(type, depth, ValueCategory.LVALUE, Cv(isVolatile = isVolatile), rhs)
  }
  private fun stableLvalueExpression(
    type: String,
    depth: Int,
    rhs: List<String>,
    isVolatile: Boolean = false
  ) {
    exactStableExpression(type, depth, ValueCategory.LVALUE, Cv(isVolatile = isVolatile), rhs)
  }

  private fun production(lhs: String, vararg rhs: String) = production(lhs, rhs.toList())
  private fun production(lhs: String, rhs: List<String>) {
    if (rhs.isNotEmpty()) productions += lhs to rhs
  }
}

/** Incremental exact left quotients for one prepared statement grammar. */
private class FiniteCppConditioner(private val source: CFG) {
  private data class OrderedGrammar(
    val syntax: CFG,
    val countingOrder: List<String>
  )

  private data class Derivative(
    val symbol: String,
    var nullable: Boolean = false,
    var nonempty: Boolean = false
  )

  private data class IndexedSourceRule(
    val kind: Int,
    val left: Int = -1,
    val right: Int = -1,
    val terminal: String = ""
  )

  private val sourceNonterminals = source.mapTo(linkedSetOf()) { it.first }
  private val sourceRules = source.groupBy { it.first }
  private val sourceSymbols = sourceNonterminals.sorted()
  private val sourceIndex = sourceSymbols.withIndex()
    .associate { (index, symbol) -> symbol to index }
  private val indexedSourceRules = Array(sourceSymbols.size) { index ->
    sourceRules[sourceSymbols[index]].orEmpty().map { (_, rhs) -> when {
      rhs.isEmpty() -> IndexedSourceRule(SOURCE_EPSILON_RULE)
      rhs.size == 1 && rhs[0] !in sourceNonterminals ->
        IndexedSourceRule(SOURCE_TERMINAL_RULE, terminal = rhs[0])
      rhs.size == 1 -> IndexedSourceRule(
        SOURCE_UNIT_RULE,
        left = sourceIndex.getValue(rhs[0])
      )
      else -> IndexedSourceRule(
        SOURCE_BINARY_RULE,
        left = sourceIndex.getValue(rhs[0]),
        right = sourceIndex.getValue(rhs[1])
      )
    } }
  }
  private val renamedSource = sourceIndex.mapValues { (_, index) -> "PREPARED_SOURCE_N_$index" }
  private val renamedSourceByIndex = sourceSymbols.map { renamedSource.getValue(it) }
  private val renamedSourceRules: Map<String, List<Pair<String, List<String>>>> = source
    .map { (lhs, rhs) ->
      renamedSource.getValue(lhs) to rhs.map { renamedSource[it] ?: it }
    }
    .groupBy { it.first }
  private val renamedSourceNonterminals = renamedSource.values.toSet()
  private val renamedSourceChildren = renamedSourceRules.mapValues { (_, productions) ->
    productions.flatMapTo(linkedSetOf()) { (_, rhs) ->
      rhs.filter { it in renamedSourceNonterminals }
    }.toList()
  }
  private val renamedSourceCountingOrder = sourceChildBeforeParentOrder()
    .map(renamedSource::getValue)
  private val spanBase = CPP_MAX_STATEMENT_TOKENS + 1
  private val exactMemo = mutableMapOf<Int, Boolean>()
  private val derivativeMemo = mutableMapOf<Int, Derivative>()
  private val derivativeRules = mutableMapOf<String, MutableSet<Pair<String, List<String>>>>()
  /** Completed derivative groups are snapshotted once and reused by every later cursor. */
  private val derivativeRuleLists = mutableMapOf<String, List<Pair<String, List<String>>>>()
  private val derivativeCountingOrder = mutableListOf<String>()
  private val countWorkspace = BoundedCountWorkspace()
  private var cachedPrefix = emptyList<String>()
  private var activePrefix = emptyList<String>()
  var lastMetrics: CppConditioningMetrics = CppConditioningMetrics()
    private set

  /** Computes the immutable source grammar's stable order once for every cursor residual. */
  private fun sourceChildBeforeParentOrder(): List<String> {
    val visiting = mutableSetOf<String>()
    val visited = mutableSetOf<String>()
    val order = mutableListOf<String>()
    fun visit(nonterminal: String) {
      if (nonterminal in visited) return
      check(visiting.add(nonterminal)) { "Prepared C++ source grammar contains a cycle at $nonterminal" }
      sourceRules[nonterminal].orEmpty().forEach { (_, rhs) ->
        rhs.filter { it in sourceNonterminals }.forEach(::visit)
      }
      visiting.remove(nonterminal)
      visited += nonterminal
      order += nonterminal
    }
    sourceNonterminals.sorted().forEach(::visit)
    return order
  }

  private fun spanKey(nonterminal: Int, start: Int, end: Int): Int =
    (nonterminal * spanBase + start) * spanBase + end

  private fun reset(prefix: List<String>) {
    exactMemo.clear()
    derivativeMemo.clear()
    derivativeRules.clear()
    derivativeRuleLists.clear()
    derivativeCountingOrder.clear()
    countWorkspace.clear()
    cachedPrefix = emptyList()
    activePrefix = prefix
  }

  private fun generatesExactly(nonterminal: Int, start: Int, end: Int): Boolean =
    generatesExactly(nonterminal, start, end, activePrefix, exactMemo)

  private fun generatesExactly(
    nonterminal: Int,
    start: Int,
    end: Int,
    tokens: List<String>,
    memo: MutableMap<Int, Boolean>
  ): Boolean {
    if (start >= end) return false
    val key = spanKey(nonterminal, start, end)
    memo[key]?.let { return it }
    var matches = false
    indexedSourceRules[nonterminal].forEach { rule ->
      matches = matches || when (rule.kind) {
        SOURCE_TERMINAL_RULE ->
          end == start + 1 && rule.terminal == tokens[start]
        SOURCE_UNIT_RULE -> generatesExactly(rule.left, start, end, tokens, memo)
        SOURCE_BINARY_RULE -> (start + 1 until end).any { split ->
          generatesExactly(rule.left, start, split, tokens, memo) &&
            generatesExactly(rule.right, split, end, tokens, memo)
        }
        else -> false
      }
    }
    memo[key] = matches
    return matches
  }

  /** Uses a local span chart so membership does not disturb the incremental quotient workspace. */
  fun recognizesExactly(tokens: List<String>): Boolean {
    if (tokens.isEmpty() || tokens.size > CPP_MAX_STATEMENT_TOKENS || source.isEmpty()) return false
    return generatesExactly(sourceIndex.getValue("START"), 0, tokens.size, tokens, mutableMapOf())
  }

  private fun derivative(nonterminal: Int, start: Int, end: Int): Derivative {
    require(start < end)
    val key = spanKey(nonterminal, start, end)
    derivativeMemo[key]?.let { return it }
    val result = Derivative("PREPARED_D_${nonterminal}_${start}_$end")
    derivativeMemo[key] = result
    val productions = derivativeRules.getOrPut(result.symbol) { linkedSetOf() }
    fun add(vararg rhs: String) {
      productions += result.symbol to rhs.toList()
      result.nonempty = true
    }
    indexedSourceRules[nonterminal].forEach { rule -> when (rule.kind) {
      SOURCE_TERMINAL_RULE ->
        if (end == start + 1 && rule.terminal == activePrefix[start]) result.nullable = true
      SOURCE_UNIT_RULE -> {
        val child = derivative(rule.left, start, end)
        if (child.nonempty) add(child.symbol)
        result.nullable = result.nullable || child.nullable
      }
      SOURCE_BINARY_RULE -> {
        val left = derivative(rule.left, start, end)
        if (left.nonempty) add(left.symbol, renamedSourceByIndex[rule.right])
        if (left.nullable) add(renamedSourceByIndex[rule.right])
        for (split in start + 1 until end) if (generatesExactly(rule.left, start, split)) {
          val right = derivative(rule.right, split, end)
          if (right.nonempty) add(right.symbol)
          result.nullable = result.nullable || right.nullable
        }
      }
    } }
    // All derivative children have completed recursively, so append this node in postorder.
    derivativeRuleLists[result.symbol] = productions.toList()
    derivativeCountingOrder += result.symbol
    return result
  }

  private fun reachableGrammar(rootRules: Collection<Pair<String, List<String>>>): OrderedGrammar {
    val chunks = mutableListOf<Collection<Pair<String, List<String>>>>(rootRules)
    var productionCount = rootRules.size
    val queue = rootRules.flatMapTo(mutableListOf()) { (_, rhs) -> rhs }
    val visited = linkedSetOf<String>()
    val nonterminals = rootRules.mapTo(linkedSetOf()) { it.first }
    var next = 0
    while (next < queue.size) {
      val symbol = queue[next++]
      if (!visited.add(symbol)) continue
      val productions = when {
        symbol in renamedSourceNonterminals -> renamedSourceRules[symbol].orEmpty()
        else -> derivativeRuleLists[symbol].orEmpty()
      }
      if (productions.isEmpty()) continue
      nonterminals += symbol
      chunks += productions
      productionCount += productions.size
      if (symbol in renamedSourceNonterminals) {
        queue += renamedSourceChildren[symbol].orEmpty()
      } else {
        productions.forEach { (_, rhs) -> queue += rhs }
      }
    }
    // Source groups are immutable, and completed derivative groups are never mutated after they
    // are published. Their LHS namespaces are disjoint and each group is already duplicate-free,
    // so this chunked view retains exact Set semantics without hashing/copying every production.
    val order = buildList(nonterminals.size) {
      renamedSourceCountingOrder.filterTo(this) { it in nonterminals }
      derivativeCountingOrder.filterTo(this) { it in nonterminals }
      rootRules.mapTo(linkedSetOf()) { it.first }
        .filterTo(this) { it in nonterminals }
    }
    check(order.size == nonterminals.size) {
      "Prepared C++ residual order omitted ${nonterminals - order.toSet()}"
    }
    val nonterminalIndex = buildMap(order.size) {
      order.forEachIndexed { index, symbol -> put(symbol, index) }
    }
    val terminals = linkedSetOf<String>()
    var nonterminalProductions = 0
    chunks.forEach { group ->
      group.forEach { (_, rhs) ->
        if (rhs.size == 1 && rhs[0] !in nonterminals) terminals += rhs[0]
        else nonterminalProductions++
      }
    }
    val rootLists = rootRules.groupBy { it.first }
    val syntax = IndexedChunkedCppCfg(
      chunks = chunks.toList(),
      size = productionCount,
      acyclicCountingOrder = order,
      acyclicNonterminalIndex = nonterminalIndex,
      acyclicStructuralStats =
        "CFG(|Σ|=${terminals.size}, |V|=${nonterminals.size}, |P|=$nonterminalProductions)",
      productionLookup = { symbol ->
        rootLists[symbol]
          ?: renamedSourceRules[symbol]
          ?: derivativeRuleLists[symbol]
          ?: emptyList()
      }
    )
    return OrderedGrammar(syntax, order)
  }

  fun condition(prefix: List<String>, maxSuffixTokens: Int): BoundedAcyclicCFG {
    if (maxSuffixTokens < 0 || source.isEmpty()) {
      lastMetrics = CppConditioningMetrics()
      return emptySet<Pair<String, List<String>>>()
        .boundedAcyclic(maxSuffixTokens.coerceAtLeast(0))
    }
    val derivativeClock = TimeSource.Monotonic.markNow()
    require(prefix.size <= CPP_MAX_STATEMENT_TOKENS) {
      "C++ statement prefix exceeds the $CPP_MAX_STATEMENT_TOKENS-token finite horizon"
    }
    val extendsCached = prefix.size >= cachedPrefix.size &&
      prefix.subList(0, cachedPrefix.size) == cachedPrefix
    if (!extendsCached) reset(prefix) else activePrefix = prefix
    cachedPrefix = prefix.toList()

    val rootSymbol = "PREPARED_ROOT_${prefix.size}"
    val rootRules = linkedSetOf<Pair<String, List<String>>>()
    if (prefix.isEmpty()) {
      rootRules += rootSymbol to listOf(renamedSource.getValue("START"))
    } else {
      // For A -> B C, w^-1(BC) contains (w^-1 B)C and, for every non-empty
      // split w=uv generated by B, v^-1 C. Nullable derivatives are spliced into parents.
      val residual = derivative(sourceIndex.getValue("START"), 0, prefix.size)
      if (!residual.nonempty && !residual.nullable) {
        lastMetrics = CppConditioningMetrics(
          derivativeMillis = derivativeClock.elapsedNow().inWholeMilliseconds
        )
        return emptySet<Pair<String, List<String>>>().boundedAcyclic(maxSuffixTokens)
      }
      if (residual.nonempty) rootRules += rootSymbol to listOf(residual.symbol)
      if (residual.nullable) rootRules += rootSymbol to emptyList()
    }
    val derivativeMillis = derivativeClock.elapsedNow().inWholeMilliseconds
    val reachableClock = TimeSource.Monotonic.markNow()
    val residual = reachableGrammar(rootRules)
    val reachableMillis = reachableClock.elapsedNow().inWholeMilliseconds
    val boundedClock = TimeSource.Monotonic.markNow()
    val bounded = residual.syntax.boundedAcyclic(
      maxLength = maxSuffixTokens,
      startSymbol = rootSymbol,
      workspace = countWorkspace,
      countingOrder = residual.countingOrder
    )
    lastMetrics = CppConditioningMetrics(
      derivativeMillis = derivativeMillis,
      reachableMillis = reachableMillis,
      boundedMillis = boundedClock.elapsedNow().inWholeMilliseconds
    )
    return bounded
  }
}

/** Terminal-lifts and binarizes an already acyclic grammar while preserving compact unit edges. */
internal fun finiteAcyclicCnf(input: Collection<Pair<String, List<String>>>): CFG {
  val grammar = pruneSemanticGrammar(input)
  if (grammar.isEmpty()) return emptySet()
  val nonterminals = grammar.mapTo(linkedSetOf()) { it.first }
  // [input] is already set-semantic and filtering preserves uniqueness. The final CNF set still
  // provides the ordinary defensive deduplication, so hashing every intermediate rule again is
  // redundant on this hot path.
  val lifted = ArrayList<Pair<String, List<String>>>(grammar.size)
  val terminalSymbols = linkedMapOf<String, String>()
  grammar.forEach { (lhs, rhs) ->
    if (rhs.size == 1) lifted += lhs to rhs
    else lifted += lhs to rhs.map { symbol ->
      if (symbol in nonterminals) symbol
      else terminalSymbols.getOrPut(symbol) { "FINITE_TERMINAL_${terminalSymbols.size}" }
    }
  }
  terminalSymbols.forEach { (terminal, symbol) -> lifted += symbol to listOf(terminal) }

  val cnf = linkedSetOf<Pair<String, List<String>>>()
  val suffixSymbols = linkedMapOf<List<String>, String>()
  var suffixCount = 0
  fun suffixSymbol(rhs: List<String>): String {
    suffixSymbols[rhs]?.let { return it }
    val symbol = "FINITE_SUFFIX_${suffixCount++}"
    suffixSymbols[rhs] = symbol
    val tail = if (rhs.size == 2) rhs else listOf(rhs.first(), suffixSymbol(rhs.drop(1)))
    cnf += symbol to tail
    return symbol
  }
  lifted.forEach { (lhs, rhs) ->
    cnf += if (rhs.size <= 2) lhs to rhs
    else lhs to listOf(rhs.first(), suffixSymbol(rhs.drop(1)))
  }
  // [grammar] is already productive and START-reachable. Lifting and suffix binarization introduce
  // only referenced terminal/tail nodes whose children are productive, so a second whole-grammar
  // fixed point is redundant (and was the dominant cold cost for the largest cursor context).
  return cnf.freeze()
}

private fun pruneSemanticGrammar(
  grammar: Collection<Pair<String, List<String>>>
): List<Pair<String, List<String>>> {
  if (grammar.isEmpty()) return emptyList()
  val nonterminals = grammar.mapTo(linkedSetOf()) { it.first }
  // A depth/type symbol can be referenced before it receives any productive atom or call rule.
  // Treat it as a dead nonterminal, not as a literal terminal such as `TYPE_7_D0`. Cache the exact
  // classifier by spelling: large semantic grammars repeat each depth symbol thousands of times.
  val generatedSymbol = mutableMapOf<String, Boolean>()
  grammar.forEach { (_, rhs) -> rhs.forEach { symbol ->
    if (symbol !in nonterminals && generatedSymbol.getOrPut(symbol) {
        GENERATED_EXPRESSION_SYMBOL.matches(symbol)
      }) nonterminals += symbol
  } }
  val productions = if (grammar is List) grammar else grammar.toList()
  val generating = linkedSetOf<String>()
  val remainingChildren = IntArray(productions.size)
  val waiting = mutableMapOf<String, MutableList<Int>>()
  val ready = ArrayList<Int>(productions.size)
  productions.forEachIndexed { index, (_, rhs) ->
    rhs.forEach { child -> if (child in nonterminals) {
      remainingChildren[index]++
      waiting.getOrPut(child) { mutableListOf() } += index
    } }
    if (remainingChildren[index] == 0) ready += index
  }
  var readyIndex = 0
  while (readyIndex < ready.size) {
    val (lhs) = productions[ready[readyIndex++]]
    if (!generating.add(lhs)) continue
    waiting[lhs].orEmpty().forEach { parent ->
      remainingChildren[parent]--
      if (remainingChildren[parent] == 0) ready += parent
    }
  }
  if ("START" !in generating) return emptyList()

  val productive = ArrayList<Pair<String, List<String>>>(productions.size)
  productions.filterTo(productive) { (lhs, rhs) ->
    lhs in generating && rhs.all { it !in nonterminals || it in generating }
  }
  val byLhs = mutableMapOf<String, MutableList<Pair<String, List<String>>>>()
  productive.forEach { production ->
    byLhs.getOrPut(production.first) { mutableListOf() } += production
  }
  val reachable = linkedSetOf("START")
  val queue = ArrayList<String>().apply { add("START") }
  var next = 0
  while (next < queue.size) {
    byLhs[queue[next++]].orEmpty().forEach { (_, rhs) ->
      rhs.forEach { symbol -> if (symbol in nonterminals && reachable.add(symbol)) queue += symbol }
    }
  }
  return productive.filterTo(ArrayList(productive.size)) { it.first in reachable }
}

private val GENERATED_EXPRESSION_SYMBOL = Regex(
  "(?:SEMANTIC_STATEMENT|SIMPLE_STATEMENT|BOOLEAN_CONDITION_D[0-9]+|" +
    "TYPE_[0-9]+_(?:(?:POSTFIX|STABLE|GLVALUE|RVALUE|LVALUE|MOVABLE|MUTABLE_POSTFIX)_)?D[0-9]+|" +
    "TYPE_[0-9]+_(?:POSTFIX_)?(?:LVALUE|RVALUE)_(?:U|C|V|CV)_D[0-9]+)"
)


private fun CppParameter.semanticType(): String = canonicalType?.takeIf(String::isNotBlank) ?: type
private fun CppParameter.isOptional(): Boolean = hasDefault ?: (defaultValue != null)
private fun CppTypeInfo?.semanticId(): String? =
  this?.valueCanonicalId?.takeIf(String::isNotBlank)
    ?: this?.canonicalId?.takeIf(String::isNotBlank)

private fun CppTypeInfo?.isConcrete(): Boolean = this == null ||
  (!isDependent && !isInstantiationDependent && semanticId() != null)

private fun CppTypeInfo?.isLvalueReference(): Boolean = this?.kind == "lvalueReference"

/** [name] is clangd's context-correct insertion spelling; qualifiedName is identity only. */
private fun CppReference.semanticName(): String = name

private fun CppReference.denotesConstructor(): Boolean =
  kind.contains("constructor", ignoreCase = true)

private fun CppReference.denotesMember(): Boolean = isMember ?: (
  receiverMember || canonicalOwnerType != null || ownerType != null ||
    kind.contains("method", ignoreCase = true) || kind.contains("field", ignoreCase = true)
  )

private fun CppReference.denotesCallable(): Boolean = isCallable ?: (
  parameters.isNotEmpty() || denotesConstructor() ||
    kind.contains("function", ignoreCase = true) || kind.contains("method", ignoreCase = true) ||
    kind.contains("operator", ignoreCase = true)
  )

private fun CppReference.denotesType(): Boolean = isType ?: kind.lowercase().let {
  "type" in it || "class" in it || "struct" in it || "enum" in it || "alias" in it
}

private fun CppReference.denotesValue(): Boolean = isValue ?: (
  !denotesType() && !denotesCallable() && !type.isNullOrBlank()
  )

private fun CppReference.isStaticFact(): Boolean = isStatic ?: run {
  Regex("(?:^|\\s)static(?:\\s|$)").containsMatchIn(detail.orEmpty())
}

private fun CppReference.semanticType(): String? =
  canonicalType?.takeIf(String::isNotBlank) ?: type?.takeIf(String::isNotBlank)

private fun CppReference.semanticReturnType(): String? =
  canonicalReturnType?.takeIf(String::isNotBlank)
    ?: returnType?.takeIf(String::isNotBlank)
    ?: detail?.substringBefore('(')?.trim()?.takeIf { denotesCallable() && it.isNotBlank() }
    ?: type?.substringBefore('(')?.trim()?.takeIf { denotesCallable() && it.isNotBlank() }

private fun CppReference.isConstMember(): Boolean =
  isConstMethod ?: Regex("\\)\\s*const(?:\\s|$)").containsMatchIn(detail.orEmpty())

private fun CppReference.isVolatileMember(): Boolean =
  isVolatileMethod ?: Regex("\\)\\s*(?:const\\s+)?volatile(?:\\s|$)")
    .containsMatchIn(detail.orEmpty())

private fun CppReference.requiresMutableReceiver(): Boolean =
  denotesCallable() && !isStaticFact() && !isConstMember() && !isVolatileMember()

private fun CppReference.isMutableLvalue(): Boolean {
  if (!denotesValue()) return false
  if (isMutableField == true) return true
  typeInfo?.let { return !it.isConst }
  val spelling = type.orEmpty().trim()
  if (spelling.isEmpty()) return false
  if (spelling.startsWith("const ") && '*' !in spelling) return false
  return !Regex("\\bconst\\s*(?:&&|&)\\s*$").containsMatchIn(spelling)
}

/** Operators are syntax; their operand and result types still come exclusively from Sema. */
private fun CppReference.operatorToken(): String? {
  val spelling = semanticName().substringAfterLast("::").removePrefix("operator").trim()
  return spelling.takeIf {
    it in setOf("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "==", "!=", "<", "<=", ">", ">=")
  }
}

private fun String?.isLvalueReferenceType(): Boolean =
  this?.trim()?.let { it.endsWith('&') && !it.endsWith("&&") } == true

private fun String?.isConstLvalueReferenceType(): Boolean {
  val spelling = this?.trim() ?: return false
  return spelling.isLvalueReferenceType() &&
    Regex("(?:^|\\s)const(?:\\s|$)").containsMatchIn(spelling)
}

private fun String?.isConstRvalueReferenceType(): Boolean {
  val spelling = this?.trim() ?: return false
  return spelling.endsWith("&&") &&
    Regex("(?:^|\\s)const(?:\\s|$)").containsMatchIn(spelling)
}

/** Normalizes clang type spelling without attaching semantics to any library identifier. */
private fun cppType(raw: String?): String? {
  if (raw.isNullOrBlank()) return null
  var type = raw.trim()
    .removePrefix("class ").removePrefix("struct ").removePrefix("enum ")
    .substringBefore(" noexcept").substringBefore(" __attribute__")
    .removeSuffix("&&").removeSuffix("&").trim()
  type = type.replace(Regex("\\s+(?:const|volatile)\\s*$"), "").trim()
  if ('*' !in type) type = type.replace(Regex("^(?:const|volatile)\\s+"), "").trim()
  else type = type.replace(Regex("^(.+?)\\s+const\\s*\\*")) { "const ${it.groupValues[1]} *" }
    .replace(Regex("^(.+?)\\s+volatile\\s*\\*")) { "volatile ${it.groupValues[1]} *" }
  return type.replace(Regex("\\s*::\\s*"), "::")
    .replace(Regex("\\s*<\\s*"), "<")
    .replace(Regex("\\s*>\\s*"), ">")
    .replace(Regex("\\s*,\\s*"), ",")
    .replace(Regex("\\s*\\*\\s*"), " *")
    .replace(Regex("\\s+"), " ").trim().removePrefix("::")
    .takeIf(String::isNotBlank)
}

private fun String.cppNameTokens(): List<String> = lexCppLine(this).map { token ->
  if (token.kind == CppTokenKind.IDENTIFIER) encodeIdentifier(token.text) else token.text
}

private fun String.typeShape(): String = substringBefore('\u0000')
private fun String.isRawPointer(): Boolean = typeShape().endsWith(" *")
private fun String.rawPointee(): String? =
  if (isRawPointer()) typeShape().removeSuffix(" *").trim() else null

private fun String.typeSpellingVariants(): List<List<String>> = buildList {
  add(cppNameTokens())
  when (this@typeSpellingVariants) {
    "signed int" -> add(listOf("signed"))
    "unsigned int" -> add(listOf("unsigned"))
    "short int", "signed short int" -> add(listOf("short"))
    "unsigned short int" -> add(listOf("unsigned", "short"))
    "long int", "signed long int" -> add(listOf("long"))
    "unsigned long int" -> add(listOf("unsigned", "long"))
    "long long int", "signed long long int" -> add(listOf("long", "long"))
    "unsigned long long int" -> add(listOf("unsigned", "long", "long"))
  }
}.distinct()

private fun String.isIntegralCppType(): Boolean = typeShape() in setOf(
  "char", "signed char", "unsigned char", "short", "short int", "signed short int",
  "unsigned short", "unsigned short int", "int", "signed", "signed int", "unsigned",
  "unsigned int", "long", "long int", "signed long int", "unsigned long", "unsigned long int",
  "long long", "long long int", "signed long long int", "unsigned long long",
  "unsigned long long int", "wchar_t", "char8_t", "char16_t", "char32_t"
)

private fun String.isFloatingCppType(): Boolean = typeShape() in setOf("float", "double", "long double")
private fun String.isNumericCppType(): Boolean = isIntegralCppType() || isFloatingCppType()
private fun String.isArithmeticCppType(): Boolean = typeShape() == "bool" || isNumericCppType()
private fun String.promotedArithmeticType(): String = when {
  typeShape() in setOf(
    "char", "signed char", "unsigned char", "short", "short int", "signed short int",
    "unsigned short", "unsigned short int", "wchar_t", "char8_t", "char16_t"
  ) -> "int"
  else -> typeShape()
}

private fun String.isLanguageDefaultConstructible(): Boolean =
  isArithmeticCppType() || isRawPointer()

private fun isAssignable(
  from: String,
  to: String,
  explicit: Set<Pair<String, String>>
): Boolean {
  if (from == to || from to to in explicit) return true
  if (to == "bool" && (from.isArithmeticCppType() || from.isRawPointer())) return true
  if (from.isArithmeticCppType() && to.isArithmeticCppType()) return true
  val fromPointee = from.rawPointee() ?: return false
  val toPointee = to.rawPointee() ?: return false
  val fromConst = fromPointee.startsWith("const ")
  val toConst = toPointee.startsWith("const ")
  if (fromConst && !toConst) return false
  val rawFrom = fromPointee.removePrefix("const ").removePrefix("volatile ")
  val rawTo = toPointee.removePrefix("const ").removePrefix("volatile ")
  return rawFrom == rawTo || rawFrom to rawTo in explicit
}

fun encodeIdentifier(identifier: String): String = "@id:$identifier"

private val CPP_ALTERNATIVE_OPERATOR_CANONICAL = mapOf(
  "and" to "&&",
  "or" to "||",
  "not" to "!",
  "bitand" to "&",
  "bitor" to "|",
  "xor" to "^",
  "compl" to "~",
  "and_eq" to "&=",
  "or_eq" to "|=",
  "xor_eq" to "^=",
  "not_eq" to "!="
)

private val CPP_DIGRAPH_CANONICAL = mapOf(
  "<:" to "[",
  ":>" to "]",
  "<%" to "{",
  "%>" to "}"
)

private val CPP_COMPLETION_SOURCE_ALIASES =
  (CPP_ALTERNATIVE_OPERATOR_CANONICAL.entries + CPP_DIGRAPH_CANONICAL.entries)
    .groupBy({ it.value }, { it.key })

internal fun projectCppCompletionTokens(
  tokens: List<CppToken>,
  mode: CppProjectionMode
): List<String> = when (mode) {
  CppProjectionMode.SEMANTIC -> projectCppTokens(tokens)
  // The pinned lexer exposes every `>>` as two Greater tokens. Preserve that lossless stream for
  // the generated parser: name-based template-depth guesses reject valid casts/operator calls.
  CppProjectionMode.SYNTAX -> projectCppTokens(tokens, preserveAdjacentGreater = true).map { terminal ->
    if (terminal.startsWith("@id:")) CPP_SYNTAX_IDENTIFIER else terminal
  }
}

fun projectCppTokens(tokens: List<CppToken>): List<String> =
  projectCppTokens(tokens, preserveAdjacentGreater = false)

private fun projectCppTokens(
  tokens: List<CppToken>,
  preserveAdjacentGreater: Boolean
): List<String> = buildList {
  var templateDepth = 0
  var index = 0
  while (index < tokens.size) {
    val token = tokens[index]
    val next = tokens.getOrNull(index + 1)
    val literalTerminal = CPP_LITERAL_TERMINAL[token.kind]
    val adjacentPair = next?.takeIf { token.end == it.start }?.let { token.text + it.text }
    val digraph = CPP_DIGRAPH_CANONICAL[adjacentPair]
    when {
      digraph != null -> {
        add(digraph)
        index++
      }
      token.text in CPP_DIGRAPH_CANONICAL -> add(CPP_DIGRAPH_CANONICAL.getValue(token.text))
      token.text in CPP_ALTERNATIVE_OPERATOR_CANONICAL ->
        add(CPP_ALTERNATIVE_OPERATOR_CANONICAL.getValue(token.text))
      token.text == ">" && next?.text == ">" && token.end == next.start -> {
        if (preserveAdjacentGreater || templateDepth >= 2) {
          add(">")
          add(">")
          if (!preserveAdjacentGreater) templateDepth -= 2
        } else add(">>")
        index++
      }
      token.text == CPP_FRESH -> add(CPP_FRESH)
      token.kind == CppTokenKind.IDENTIFIER -> add(encodeIdentifier(token.text))
      literalTerminal != null -> add(literalTerminal)
      token.kind == CppTokenKind.BOOLEAN -> add(CPP_BOOLEAN)
      token.kind == CppTokenKind.NULLPTR -> add(CPP_NULLPTR)
      token.text == "<" -> {
        add("<")
        if (!preserveAdjacentGreater && looksLikeTemplateOpen(tokens, index)) templateDepth++
      }
      token.text == ">>" && (preserveAdjacentGreater || templateDepth >= 2) -> {
        add(">")
        add(">")
        if (!preserveAdjacentGreater) templateDepth -= 2
      }
      token.text == ">" -> {
        add(">")
        if (templateDepth > 0) templateDepth--
      }
      else -> add(token.text)
    }
    index++
  }
}

fun materializeCppTerminal(terminal: String, fresh: () -> String): String = when {
  terminal.startsWith("@id:") -> terminal.removePrefix("@id:")
  terminal == CPP_FRESH || terminal.startsWith(CPP_BIND_PREFIX) -> fresh()
  terminal == CPP_SYNTAX_IDENTIFIER ->
    error("C++ syntax Identifier reached sampling without a clang/Sema spelling")
  terminal == CPP_INTEGER -> "0"
  terminal == CPP_FLOATING -> "0.0"
  terminal == CPP_CHARACTER -> "'\\0'"
  terminal == CPP_STRING -> "\"\""
  terminal == CPP_BOOLEAN -> "true"
  terminal == CPP_NULLPTR -> "nullptr"
  else -> terminal
}

private val CPP_LITERAL_TERMINAL = mapOf(
  CppTokenKind.INTEGER to CPP_INTEGER,
  CppTokenKind.FLOATING to CPP_FLOATING,
  CppTokenKind.CHARACTER to CPP_CHARACTER,
  CppTokenKind.STRING to CPP_STRING,
  CppTokenKind.USER_DEFINED_INTEGER to CPP_INTEGER,
  CppTokenKind.USER_DEFINED_FLOATING to CPP_FLOATING,
  CppTokenKind.USER_DEFINED_CHARACTER to CPP_CHARACTER,
  CppTokenKind.USER_DEFINED_STRING to CPP_STRING
)

/** Source spellings represented by one grammar terminal at a partial-token cursor. */
internal fun cppCompletionTerminalSpellings(
  terminal: String,
  prefix: CppToken
): List<String> = buildList {
  when {
    terminal.startsWith("@id:") -> add(terminal.removePrefix("@id:"))
    terminal == CPP_FRESH || terminal.startsWith(CPP_BIND_PREFIX) ->
      if (prefix.kind == CppTokenKind.IDENTIFIER) {
        add(prefix.text)
        prefix.completeText?.let(::add)
      }
    CPP_LITERAL_TERMINAL[prefix.kind] == terminal -> add(prefix.completeText ?: prefix.text)
    terminal == CPP_BOOLEAN -> {
      add("true")
      add("false")
    }
    terminal == CPP_NULLPTR -> {
      add("nullptr")
    }
    !terminal.startsWith('@') -> {
      add(terminal)
      addAll(CPP_COMPLETION_SOURCE_ALIASES[terminal].orEmpty())
    }
  }
}.distinct()

/** The shortest concrete spelling of [terminal] that preserves the characters already typed. */
internal fun cppCompletionTerminalSpelling(terminal: String, prefix: CppToken): String? =
  cppCompletionTerminalSpellings(terminal, prefix)
    .filter { it.startsWith(prefix.text) }
    .minWithOrNull(compareBy(String::length).thenBy { it })

fun cppLines(source: String): List<CppLine> = buildList {
  var start = 0
  var number = 0
  while (start <= source.length) {
    val newline = source.indexOf('\n', start)
    val rawEnd = if (newline < 0) source.length else newline
    val contentEnd = if (rawEnd > start && source[rawEnd - 1] == '\r') rawEnd - 1 else rawEnd
    val end = if (newline < 0) rawEnd else newline + 1
    val text = source.substring(start, contentEnd)
    add(CppLine(number++, start, contentEnd, end, text, lexCppLine(text)))
    if (newline < 0) break
    start = newline + 1
  }
}

fun cppSemicolonLines(source: String): List<CppLine> = cppLines(source).filter { line ->
  line.text.trimEnd().endsWith(';') && line.tokens.lastOrNull()?.text == ";"
}

/** Every semicolon-ended line lexically enclosed by a function body, without ranking/filtering. */
fun cppStatementLines(source: String): List<CppLine> {
  val result = mutableListOf<CppLine>()
  val functionScopes = mutableListOf<Boolean>()
  // A function declarator and its opening brace need not share a physical line. Retain the tokens
  // since the previous declaration delimiter so Allman-style definitions are classified exactly
  // like `int f() {`. The selected benchmark unit remains a complete physical line ending in `;`.
  val sinceDelimiter = mutableListOf<String>()
  cppLines(source).forEach { line ->
    if (functionScopes.lastOrNull() == true &&
      line.text.trimEnd().endsWith(';') && line.tokens.lastOrNull()?.text == ";"
    ) result += line
    line.tokens.forEach { token -> when (token.text) {
      "{" -> {
        val alreadyInFunction = functionScopes.lastOrNull() == true
        val opensRecordOrNamespace = sinceDelimiter.any { it in setOf("class", "struct", "union", "namespace") }
        val opensFunction = !alreadyInFunction && !opensRecordOrNamespace && ")" in sinceDelimiter
        functionScopes += alreadyInFunction || opensFunction
        sinceDelimiter.clear()
      }
      "}" -> {
        if (functionScopes.isNotEmpty()) functionScopes.removeAt(functionScopes.lastIndex)
        sinceDelimiter.clear()
      }
      ";" -> sinceDelimiter.clear()
      else -> sinceDelimiter += token.text
    } }
  }
  return result
}

fun cppTruncations(line: CppLine): List<CppTruncation> {
  val indentation = line.text.takeWhile { it == ' ' || it == '\t' }
  return (0..line.tokens.size).map { count ->
    val cursor = if (count == 0) indentation.length else line.tokens[count - 1].end
    CppTruncation(
      line = line,
      prefix = line.tokens.take(count),
      suffix = line.tokens.drop(count),
      prefixText = line.text.substring(0, cursor)
    )
  }
}

/** A semicolon closes the selected statement only when it is not inside a lambda/call/group. */
private fun List<CppToken>.endsCompleteStatement(): Boolean {
  if (lastOrNull()?.text != ";") return false
  var round = 0
  var square = 0
  var brace = 0
  dropLast(1).forEach { token -> when (token.text) {
    "(" -> round++
    ")" -> round--
    "[" -> square++
    "]" -> square--
    "{" -> brace++
    "}" -> brace--
  } }
  return round == 0 && square == 0 && brace == 0
}

fun replaceCppLine(source: String, line: CppLine, replacement: String): String {
  require('\n' !in replacement && '\r' !in replacement)
  return source.replaceRange(line.start, line.contentEnd, replacement)
}

fun truncateCppSource(source: String, truncation: CppTruncation): String =
  replaceCppLine(source, truncation.line, truncation.prefixText)

private fun looksLikeTemplateOpen(tokens: List<CppToken>, index: Int): Boolean {
  val previous = tokens.getOrNull(index - 1) ?: return false
  val next = tokens.getOrNull(index + 1)
  if (next?.text == "<" && tokens[index].end == next.start) return false
  return previous.kind == CppTokenKind.IDENTIFIER || previous.text in setOf(">", ">>", "::")
}

private fun lexCppLine(line: String): List<CppToken> = buildList {
  val lexicalTokens = lexCppTokenSpans(line)
  var index = 0
  while (index < lexicalTokens.size) {
    val lexical = lexicalTokens[index]
    val next = lexicalTokens.getOrNull(index + 1)
    val adjacentPair = next?.takeIf { lexical.endIndexExclusive == it.startIndex }
      ?.let { lexical.text + it.text }
    if (adjacentPair != null && adjacentPair in CPP_DIGRAPH_CANONICAL) {
      val digraphEnd = checkNotNull(next)
      add(CppToken(
        text = adjacentPair,
        start = lexical.startIndex,
        end = digraphEnd.endIndexExclusive,
        kind = CppTokenKind.OTHER
      ))
      index += 2
      continue
    }
    add(CppToken(
      text = lexical.text,
      start = lexical.startIndex,
      end = lexical.endIndexExclusive,
      kind = cppTokenKind(lexical.type, lexical.text)
    ))
    index++
  }
}

private fun cppTokenKind(symbolicType: String, text: String): CppTokenKind =
  when (symbolicType) {
    "Identifier" -> if (text in CPP_KEYWORDS) CppTokenKind.OTHER else CppTokenKind.IDENTIFIER
    "IntegerLiteral" -> CppTokenKind.INTEGER
    "FloatingLiteral" -> CppTokenKind.FLOATING
    "CharacterLiteral" -> CppTokenKind.CHARACTER
    "StringLiteral" -> CppTokenKind.STRING
    "BooleanLiteral" -> CppTokenKind.BOOLEAN
    "PointerLiteral" -> CppTokenKind.NULLPTR
    "UserDefinedIntegerLiteral" -> CppTokenKind.USER_DEFINED_INTEGER
    "UserDefinedFloatingLiteral" -> CppTokenKind.USER_DEFINED_FLOATING
    "UserDefinedCharacterLiteral" -> CppTokenKind.USER_DEFINED_CHARACTER
    "UserDefinedStringLiteral" -> CppTokenKind.USER_DEFINED_STRING
    "UserDefinedLiteral" -> userDefinedCppTokenKind(text)
    else -> CppTokenKind.OTHER
  }

private fun userDefinedCppTokenKind(text: String): CppTokenKind = when {
  '"' in text -> CppTokenKind.USER_DEFINED_STRING
  '\'' in text -> CppTokenKind.USER_DEFINED_CHARACTER
  '.' in text || (if (text.startsWith("0x", ignoreCase = true))
    Regex("[pP][+-]?[0-9]") else Regex("[eE][+-]?[0-9]"))
    .containsMatchIn(text) -> CppTokenKind.USER_DEFINED_FLOATING
  else -> CppTokenKind.USER_DEFINED_INTEGER
}

private val IDENTIFIER_REGEX = Regex("[A-Za-z_][A-Za-z_0-9]*")

internal fun String.isCppIdentifierName(): Boolean =
  this !in CPP_KEYWORDS && IDENTIFIER_REGEX.matches(this)

private val CPP_KEYWORDS = setOf(
  "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool",
  "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class",
  "compl", "concept", "const", "consteval", "constexpr", "constinit", "const_cast",
  "continue", "co_await", "co_return", "co_yield", "decltype", "default", "delete", "do",
  "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false",
  "final", "float", "for", "friend", "goto", "if", "import", "inline", "int", "long",
  "module", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
  "operator", "or", "or_eq", "override", "private", "protected", "public", "register",
  "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
  "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local",
  "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
  "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
)
