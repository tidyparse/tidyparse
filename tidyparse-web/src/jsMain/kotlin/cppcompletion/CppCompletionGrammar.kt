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
internal const val CPP_USER_DEFINED_INTEGER = "@ud_integer"
internal const val CPP_USER_DEFINED_FLOATING = "@ud_floating"
internal const val CPP_USER_DEFINED_CHARACTER = "@ud_character"
internal const val CPP_USER_DEFINED_STRING = "@ud_string"
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

private val COMMON_CPP_MEMBER_NAMES = setOf(
  "add", "append", "at", "back", "begin", "capacity", "clear", "data", "empty", "end",
  "emplace", "erase", "find", "find_first_not_of", "find_last_not_of", "front", "get", "label",
  "lock", "max_size", "name", "named", "push_back", "render", "replace", "reserve", "size",
  "speak", "str", "substr", "summary", "titled", "value_or", "with_limit"
)
/** Standard completion overload sets modeled by compact, type-safe productions below. */
private val CPP_SPECIALIZED_STANDARD_CALLS = setOf(
  "addressof", "get", "get_if", "holds_alternative", "make_shared", "make_unique", "move", "sort", "visit"
)
private val CPP_STANDARD_MUTATING_MEMBERS = setOf(
  "append", "at", "emplace", "erase", "insert", "lower_bound", "push_back", "replace", "try_emplace"
)
private val CPP_OVERLOAD_SENSITIVE_STRING_MEMBERS = setOf(
  "find", "find_first_not_of", "find_last_not_of"
)
private val CPP_COMPACT_STANDARD_STRING_MEMBERS = CPP_OVERLOAD_SENSITIVE_STRING_MEMBERS + setOf(
  "append", "empty", "erase", "length", "replace", "size", "substr"
)
private val CPP_COMPACT_STANDARD_MEMBERS = mapOf(
  "vector" to setOf("at", "capacity", "empty", "max_size", "push_back", "size"),
  "map" to setOf("at", "emplace", "lower_bound", "size", "try_emplace"),
  "set" to setOf("insert", "size"),
  "optional" to setOf("emplace", "value_or"),
  "unique_ptr" to setOf("get"),
  "shared_ptr" to setOf("get"),
  "weak_ptr" to setOf("lock")
)

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
  val enclosingClassType: String? = null,
  val thisType: String? = null,
  val mutableFields: Set<String> = emptySet()
)

/** Compact, transport-friendly clang semantic facts used to specialize one cursor CFG. */
data class CppParameter(
  val label: String = "",
  val name: String = "",
  val type: String = "",
  val defaultValue: String? = null
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
  val abstract: Boolean = false
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
    if (recognizesCompleteSyntax && cppSingleStatementSyntaxRecognizes(rawPrefix + rawSuffix)) return true
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

  internal fun withSyntaxFallback(): CppSuffixGrammar = CppSuffixGrammar(
    bounded = bounded,
    rawPrefix = rawPrefix,
    projectedPrefix = projectedPrefix,
    templateTokens = templateTokens,
    sourceSyntax = sourceSyntax,
    conditioningMetrics = conditioningMetrics,
    projectionMode = projectionMode,
    syntaxFallbackFactory = { cppSingleStatementSyntaxCompletion(rawPrefix, it) }
  )
}

internal enum class CppProjectionMode { SEMANTIC, SYNTAX }

/** Constructs one finite, cursor-specialized statement grammar from clang's scoped facts. */
class CppCompletionGrammar {
  fun prepare(context: CppCompletionContext): PreparedCppCompletionGrammar =
    PreparedCppCompletionGrammar(SemanticCppGrammar(context, emptyList()).build())

  fun prepare(context: CppCompletionContext, prefix: List<CppToken>): PreparedCppCompletionGrammar =
    PreparedCppCompletionGrammar(SemanticCppGrammar(context, prefix).build())

  fun generate(context: CppCompletionContext, prefix: List<CppToken>): CppSuffixGrammar = when {
    prefix.endsCompleteStatement() -> completedStatementGrammar(prefix)
    else -> PreparedCppCompletionGrammar(SemanticCppGrammar(context, prefix).build()).generate(prefix)
  }
}

/** Reuses one line's scoped semantic grammar while deriving an exact residual at every cursor. */
class PreparedCppCompletionGrammar internal constructor(private val sourceSyntax: CFG) {
  private val conditioner = FiniteCppConditioner(sourceSyntax)

  /** Exact prepared-language membership without materializing a residual CFG or CYK index. */
  fun recognizes(statement: List<CppToken>): Boolean =
    conditioner.recognizesExactly(projectCppTokens(statement))

  fun generate(prefix: List<CppToken>): CppSuffixGrammar {
    if (prefix.endsCompleteStatement()) return completedStatementGrammar(prefix, sourceSyntax)
    val projectedPrefix = projectCppTokens(prefix)
    if (projectedPrefix.size > CPP_MAX_STATEMENT_TOKENS)
      return emptyCppSuffixGrammar(prefix, sourceSyntax).withSyntaxFallback()
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
      conditioningMetrics = conditioner.lastMetrics
    ).withSyntaxFallback()
  }
}

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
  private val rawTypes = linkedSetOf<String>()
  private lateinit var typeSymbols: Map<String, String>
  private val compatibleTypes = mutableMapOf<Pair<String, Boolean>, List<String>>()
  private val argumentChoices = mutableMapOf<Pair<List<CppParameter>, Boolean>, List<List<String>>>()
  private val receiverChoices = mutableMapOf<Pair<String, Boolean>, List<Pair<String, String>>>()
  private val tokenizedNames = mutableMapOf<String, List<String>>()
  private val typeSpellingSymbols = mutableMapOf<String, String>()
  private val normalizedTypes = mutableMapOf<String, String?>()
  private val abstractTypes: Set<String> by lazy {
    context.types.filter(CppReference::abstract).mapNotNullTo(linkedSetOf()) {
      canonicalType(it.type ?: it.name)
    }
  }
  private val enumTypes: Set<String> by lazy {
    context.types.filter { it.kind.contains("enum", ignoreCase = true) }
      .mapNotNullTo(linkedSetOf()) { canonicalType(it.type ?: it.name) }
  }
  private val typeAliases: Map<String, String> by lazy {
    context.types.mapNotNull { reference ->
      if (!reference.kind.contains("alias", ignoreCase = true)) return@mapNotNull null
      // Recovery ASTs can expose instantiated library aliases (`value_type`, `type`) that are not
      // spellable in this scope. A genuine source alias necessarily leaves its own lexical name.
      if (!sourceMentions(reference)) return@mapNotNull null
      val alias = canonicalType(reference.name) ?: return@mapNotNull null
      val detail = reference.detail ?: reference.type ?: return@mapNotNull null
      val target = canonicalType(detail.substringAfter('=', detail).trim()) ?: return@mapNotNull null
      (alias to target).takeUnless { it.first == it.second }
    }.toMap()
  }
  private val constructorGroups: Map<String?, List<CppReference>> by lazy {
    (functions + members + context.types)
      .filter { it.kind.lowercase().contains("constructor") }
      .groupBy { canonicalType(it.ownerType ?: it.returnType ?: it.name.substringAfterLast("::")) }
  }
  private val defaultConstructibleTypes: Set<String> by lazy {
    context.defaultConstructibleTypes.mapNotNullTo(linkedSetOf(), ::canonicalType)
  }
  private val userDeclaredTypes: Set<String> by lazy {
    buildSet {
      context.types.filter { it.source != "completion" && sourceMentions(it) }
        .mapNotNullTo(this) { canonicalType(it.type ?: it.name) }
      context.membersByType.mapNotNull { canonicalType(it.type) }
        .filterTo(this) { type ->
          lexCppLine(type).any { token ->
            token.kind == CppTokenKind.IDENTIFIER && token.text != "std" &&
              token.text in context.sourceIdentifiers
          }
        }
    }.filterTo(linkedSetOf()) { type ->
      val spellingIdentifiers = lexCppLine(type).filter { token ->
        token.kind == CppTokenKind.IDENTIFIER && token.text != "std"
      }
      type !in CPP_BUILTIN_TYPES && !type.startsWith("std::") && '<' !in type &&
        !type.endsWith("*") && spellingIdentifiers.isNotEmpty() &&
        spellingIdentifiers.all { it.text in context.sourceIdentifiers }
    }
  }

  fun build(): CFG {
    collectFacts()
    val declaredTypes = declaredTypes()
    // Completion overloads often expose placeholder spellings such as `iterator` or `_Tp` even
    // though no such type is declared in this translation unit. Do not turn those bare names into
    // vector element types; genuine user types are evidenced both lexically and by the scoped AST.
    val evidencedTypes = rawTypes.mapNotNull(::canonicalType)
      .filterNot { type -> type.isSyntheticType(declaredTypes) }
      .filterNot { type -> "sstream" !in context.headers && "ostringstream" in type }
      .toSet()
    val canonicalTypes = evidencedTypes.toMutableSet().apply {
      addAll(listOf("bool", "char", "int", "double", "const char *", "std::nullptr_t"))
      // Fundamental keywords are absent from lexical identifier sets, but clang's type tables
      // preserve source-evidenced spellings used in declarations and named casts.
      listOf("short", "signed", "unsigned", "long", "float", "long double")
        .filterTo(this) { it in context.typeNames }
      addAll(typeAliases.keys)
      addAll(typeAliases.values)
      if (context.identifiers.any { it == "cout" } || "iostream" in context.headers) add("std::ostream")
    }
    addFactoryTypes(canonicalTypes)
    addKnownStandardMembers(canonicalTypes, evidencedTypes)
    deduplicateSemanticFacts()
    // operator<< on a string stream is inherited from basic_ostream and returns the base
    // `basic_ostream&`, not the derived string-stream type. Keep that result type available even
    // when clang only mentioned `ostringstream` at this cursor.
    canonicalTypes.map(String::insertionResultType)
      .filterTo(canonicalTypes) { it == "std::ostream" }
    typeSymbols = canonicalTypes.sorted().mapIndexed { index, type -> type to "TYPE_$index" }.toMap()

    production("START", "SEMANTIC_STATEMENT")

    addAtoms()
    addBooleanCondition(0)
    for (depth in 1..CPP_SEMANTIC_DEPTH) {
      inheritShallowerExpressions(depth)
      addParenthesesAndUnary(depth)
      addCastsAndIndexing(depth)
      addFreeCalls(depth)
      addMemberAccesses(depth)
      addKnownIteratorAccesses(depth)
      addKnownTupleAccesses(depth)
      addOperators(depth)
      addKnownFactories(depth)
      addKnownStandardCalls(depth)
      addBooleanCondition(depth)
    }
    addFiniteStreamChains()
    addStatements()
    return finiteAcyclicCnf(productions)
  }

  private fun collectFacts() {
    // Blank-cursor completion lists contain a large part of the standard library. Scoped AST
    // declarations are safe and compact; keep completion facts only when clang ties them to the
    // receiver probe at this cursor.
    // Value completions are themselves scoped by clang and are few enough to retain. This also
    // preserves a local when clang's deduplication chose its completion record over the AST copy.
    fun scopedValue(reference: CppReference): CppReference = when {
      reference.name == "cout" && reference.valueType()?.isOutputStream() == true ->
        reference.copy(name = "std::cout", type = "std::ostream")
      context.thisType.isPointerToConstCppObject() && reference.kind.equals("field", true) &&
        reference.name.substringAfterLast("::") !in context.mutableFields ->
        reference.copy(type = reference.type?.asConstFieldLvalue())
      else -> reference
    }
    values += context.values.filter { it.name.isNotBlank() && it.valueType() != null }.map(::scopedValue)
    // A clang completion callable is still scope-checked. Retain it when its spelling occurs in
    // this translation unit; that recovers free functions at a fully blank statement cursor while
    // excluding the thousands of unrelated header/index candidates.
    functions += context.functions.filter {
      it.name.isNotBlank() && sourceMentions(it) && !it.isBroadCompletionOperator() &&
        it.isTrustedCallableFact() &&
        !it.isShadowedBy(context.values) && !it.isSpecializedStandardCompletion()
    }
    val implicitObject = canonicalType(context.thisType)?.dereferenceablePointee()
    val implicitOwner = implicitObject?.removePrefix("const ")
    fun isImplicitMember(reference: CppReference): Boolean {
      val owner = canonicalType(reference.ownerType) ?: return false
      return implicitOwner != null &&
        (implicitOwner == owner || implicitOwner to owner in explicitConversions)
    }
    context.completions.forEach { reference ->
      when {
        reference.receiverMember && context.receiver?.operator == "::" -> Unit
        (reference.receiverMember || isImplicitMember(reference)) && memberRelevant(reference) ->
          members += reference
        reference.isCallable() && sourceMentions(reference) &&
          !reference.isBroadCompletionOperator() && reference.isTrustedCallableFact() &&
          !reference.isShadowedBy(context.values) && !reference.isSpecializedStandardCompletion() ->
          functions += reference
        reference.source != "completion" && reference.valueType() != null && !reference.isType() ->
          values += scopedValue(reference)
      }
    }
    context.membersByType.forEach { group ->
      group.members.filter(::memberRelevant)
        .forEach { members += it.copy(ownerType = it.ownerType ?: group.type) }
      rawTypes += group.type
    }
    context.receiver?.let { receiver ->
      receiver.type?.let(rawTypes::add)
      if (receiver.operator != "::")
        receiver.members.filter(::memberRelevant)
          .forEach { members += it.copy(ownerType = it.ownerType ?: receiver.type) }
    }
    context.types.filter { it.source != "completion" && sourceMentions(it) }.forEach { type ->
      rawTypes += type.type ?: type.name
      if (type.kind.contains("alias", ignoreCase = true)) type.detail?.let(rawTypes::add)
      if (type.isCallable() || type.kind.lowercase().contains("constructor")) functions += type
    }
    (values + functions + members).forEach { reference ->
      reference.valueType()?.let(rawTypes::add)
      reference.returnType()?.let(rawTypes::add)
      reference.ownerType?.let(rawTypes::add)
      reference.parameters.map(CppParameter::type).filter(String::isNotBlank).forEach(rawTypes::add)
    }
    rawTypes += context.expectedTypes
    rawTypes += context.requiredTypes
    context.enclosingReturnType?.let(rawTypes::add)
    context.thisType?.let(rawTypes::add)
    context.conversions.forEach { rawTypes += it.from; rawTypes += it.to }

    // These library entities have language-defined signatures. Only admit them if clang/source
    // reported the spelling, which keeps the grammar valid for translation units without headers.
    if ("cout" in context.identifiers && values.none { it.name == "std::cout" })
      values += CppReference("std::cout", type = "std::ostream", kind = "variable")
  }

  private fun sourceMentions(reference: CppReference): Boolean {
    if (context.sourceIdentifiers.isEmpty()) return true
    val spellings = if (reference.kind.lowercase().contains("constructor"))
      sequenceOf(reference.name, reference.ownerType)
    else sequenceOf(reference.name)
    return spellings
      .filterNotNull()
      .flatMap { spelling -> IDENTIFIER_REGEX.findAll(spelling).map { it.value } }
      .any { it != "std" && it in context.sourceIdentifiers }
  }

  private fun memberRelevant(reference: CppReference): Boolean {
    val name = reference.name.substringAfterLast("::")
    val owner = reference.ownerType.compactStandardStringOwner()
    if (reference.source == "completion" && owner in setOf("std::string", "std::string_view") &&
      name in CPP_COMPACT_STANDARD_STRING_MEMBERS) return false
    val standardFamily = reference.ownerType.standardTemplateFamily()
    if (reference.source == "completion" &&
      name in CPP_COMPACT_STANDARD_MEMBERS[standardFamily].orEmpty()) return false
    return name.startsWith("operator()") || name in COMMON_CPP_MEMBER_NAMES || name in context.sourceIdentifiers
  }

  /** Different clang channels often report the same callable with different labels/provenance. */
  private fun deduplicateSemanticFacts() {
    fun String?.typeSpelling(): String = this.orEmpty()
      .replace(Regex("\\s*([<>,*&])\\s*"), "$1")
      .replace(Regex("\\s+"), " ")
      .trim()
    fun CppReference.key(): String = buildString {
      append(name); append('|'); append(ownerType.typeSpelling()); append('|')
      append((returnType ?: type).typeSpelling()); append('|')
      append(if (isCallable()) "call" else kind.lowercase()); append('|')
      parameters.forEach { parameter ->
        append(parameter.type.typeSpelling())
        append(if (parameter.defaultValue == null) '!' else '=')
        append(';')
      }
    }
    fun MutableSet<CppReference>.deduplicate() {
      val unique = linkedMapOf<String, CppReference>()
      forEach { reference ->
        val key = reference.key()
        val previous = unique[key]
        if (previous == null || previous.source == "completion" && reference.source != "completion")
          unique[key] = reference
      }
      clear()
      addAll(unique.values)
    }
    values.deduplicate()
    functions.deduplicate()
    members.deduplicate()
  }

  private fun addFactoryTypes(types: MutableSet<String>) {
    fun sourceUses(name: String, header: String = name): Boolean =
      name in context.sourceIdentifiers || header in context.headers
    val projectedPrefix = projectCppTokens(prefix)
    fun requestedAtCursor(candidate: String): Boolean {
      if (projectedPrefix.isEmpty()) return false
      val spelling = candidate.cppNameTokens()
      return sequenceOf(spelling, listOf("const") + spelling).any { form ->
        val common = minOf(form.size, projectedPrefix.size)
        if (common > 0 && form.take(common) == projectedPrefix.take(common)) return@any true
        // The type can also be nested in a cast or another template. In that case the active
        // partial spelling is a suffix of the statement prefix rather than its first token.
        projectedPrefix.indices.any { start ->
          val activeSuffix = projectedPrefix.drop(start)
          activeSuffix.size <= form.size && activeSuffix == form.take(activeSuffix.size)
        }
      }
    }
    fun addFamilyCandidates(family: String, candidates: Iterable<String>) {
      // An observed specialization only proves that one member of the template family is in
      // scope. At an incomplete declaration, retain every small source-derived specialization
      // consistent with the tokens at this cursor. Away from that declaration, preserve the
      // compact one-family closure used to keep browser grammar construction bounded.
      val hasObservedSpecialization = types.any { it.startsWith("std::$family<") }
      candidates.forEach { candidate ->
        if (!hasObservedSpecialization || requestedAtCursor(candidate)) types += candidate
      }
    }
    val userTypes = declaredTypes().filter { type ->
      lexCppLine(type).any { token ->
        token.kind == CppTokenKind.IDENTIFIER && token.text in context.sourceIdentifiers
      }
    }
    userTypes.forEach { type ->
      if (context.sourceIdentifiers.any { it == "unique_ptr" || it == "make_unique" })
        types += "std::unique_ptr<$type>"
      if (context.sourceIdentifiers.any { it == "shared_ptr" || it == "make_shared" })
        types += "std::shared_ptr<$type>"
      if ("weak_ptr" in context.sourceIdentifiers) types += "std::weak_ptr<$type>"
      // Pointer declarations and address-taking are common single-line completions. A pointer
      // type itself is always legal even when the pointee is abstract or lacks a default ctor.
      types += "$type *"
    }
    if (sourceUses("string")) types += "std::string"
    if ("sstream" in context.headers) types += "std::ostringstream"
    if ("size_t" in context.sourceIdentifiers) types += "std::size_t"
    if ("cstdint" in context.headers) types += "std::uintptr_t"
    if ("typeinfo" in context.headers) types += "std::type_info"
    if (types.any(String::isRawPointer)) types += "const void *"
    if ("memory" in context.headers && "make_unique" in context.sourceIdentifiers)
      types += "std::unique_ptr<int[]>"
    if (sourceUses("string_view")) types += "std::string_view"
    if (sourceUses("monostate", "variant")) types += "std::monostate"
    if (sourceUses("nullopt", "optional")) types += "std::nullopt_t"
    if (sourceUses("function", "functional")) types += "std::function<int(int)>"
    if ("unique_ptr" in context.sourceIdentifiers && "int" in types) types += "std::unique_ptr<int[]>"
    val ordinaryElements = buildList {
      add("int")
      if (sourceUses("string")) add("std::string")
      addAll(typeAliases.keys)
      addAll(userTypes)
    }.distinct()
    listOf("deque", "list", "set", "optional").forEach { family ->
      if (sourceUses(family)) addFamilyCandidates(
        family,
        ordinaryElements.map { element -> "std::$family<$element>" }
      )
    }
    if (sourceUses("map")) {
      val keys = ordinaryElements.filter { it == "int" || it == "std::string" }
      addFamilyCandidates("map", keys.flatMap { key ->
        ordinaryElements.map { value -> "std::map<$key,$value>" }
      })
    }
    if (sourceUses("variant") && sourceUses("monostate", "variant") && sourceUses("string"))
      addFamilyCandidates("variant", listOf("std::variant<std::monostate,int,std::string>"))
    types.filter { it.startsWith("std::variant<") }.toList().forEach { variant ->
      types += "$variant *"
      types += "const $variant *"
      variant.topLevelTemplateArguments().mapNotNull(::canonicalType).forEach { alternative ->
        types += "$alternative *"
        types += "const $alternative *"
      }
    }
    if (sourceUses("vector")) {
      // Do not transitively wrap every synthesized iterator/container/result type. Exact observed
      // vectors are already in [types]; this small source-rooted closure exists only to recover a
      // declaration whose vector type disappeared with the damaged line.
      val elements = buildSet {
        addAll(ordinaryElements)
        types.filterTo(this) { candidate ->
          candidate.startsWith("std::unique_ptr<") ||
            candidate.startsWith("std::shared_ptr<") ||
            candidate.startsWith("std::weak_ptr<") ||
          candidate.startsWith("std::function<")
        }
      }
      addFamilyCandidates("vector", elements.map { element -> "std::vector<$element>" })
    }
    // Only actual scoped lvalues receive address-of productions. Add their pointer types without
    // admitting the unsound general form `& EXPR`, which also matches literals and temporaries.
    values.mapNotNull { reference ->
      val valueType = canonicalType(reference.valueType())?.takeIf { it in types }
      valueType?.let(::cppAddressType)
    }.forEach(types::add)
  }

  /**
   * Small language-defined signatures needed even when clang recovery cannot issue a receiver
   * completion (for example `stops_.size()` lies to the right of a blank cursor). Owners are
   * restricted to types clang actually evidenced at this location; this is not a global library
   * fallback and every emitted signature has the standard C++ type.
   */
  private fun addKnownStandardMembers(types: MutableSet<String>, evidencedTypes: Set<String>) {
    fun method(owner: String, name: String, result: String, vararg parameters: String) {
      canonicalType(result)?.let(types::add)
      members += CppReference(
        name = name,
        returnType = result,
        parameters = parameters.map { CppParameter(type = it) },
        kind = "method",
        receiverMember = true,
        ownerType = owner,
        source = "standard"
      )
    }

    val owners = evidencedTypes + types.filter { owner ->
      owner.startsWith("std::map<") || owner.startsWith("std::set<") ||
        owner == "std::string" || owner == "std::string_view"
    }
    owners.forEach { owner ->
      owner.vectorElementType()?.let { element ->
        types += "std::size_t"
        method(owner, "size", "std::size_t")
        method(owner, "capacity", "std::size_t")
        method(owner, "max_size", "std::size_t")
        method(owner, "empty", "bool")
        method(owner, "at", "$element &", "std::size_t")
        if (!element.isMoveOnlyCppType()) method(owner, "push_back", "void", "const $element &")
        method(owner, "push_back", "void", "$element &&")
      }
      owner.smartOrRawPointee()?.takeIf { !owner.isRawPointer() }?.let { pointee ->
        types += "$pointee *"
        if (!owner.startsWith("std::weak_ptr<")) method(owner, "get", "$pointee *")
      }
      if (owner.startsWith("std::weak_ptr<")) {
        val pointee = owner.smartOrRawPointee() ?: return@forEach
        method(owner, "lock", "std::shared_ptr<$pointee>")
      }
      if (owner.contains("ostringstream") || owner.contains("stringstream"))
        method(owner, "str", "std::string")
      if (owner == "std::string") {
        types += "std::size_t"
        method(owner, "append", "std::string &", "const char *")
        method(owner, "append", "std::string &", "const std::string &")
        method(owner, "size", "std::size_t")
        method(owner, "length", "std::size_t")
        method(owner, "empty", "bool")
        method(owner, "erase", "std::string &")
        method(owner, "erase", "std::string &", "std::size_t")
        method(owner, "erase", "std::string &", "std::size_t", "std::size_t")
        method(owner, "find", "std::size_t", "char")
        method(owner, "find", "std::size_t", "char", "std::size_t")
        method(owner, "find_first_not_of", "std::size_t", "char")
        method(owner, "find_first_not_of", "std::size_t", "char", "std::size_t")
        method(owner, "find_last_not_of", "std::size_t", "char")
        method(owner, "find_last_not_of", "std::size_t", "char", "std::size_t")
        method(owner, "replace", "std::string &", "std::size_t", "std::size_t", "const char *")
      }
      if (owner == "std::string_view") {
        types += "std::size_t"
        method(owner, "size", "std::size_t")
        method(owner, "empty", "bool")
        method(owner, "find", "std::size_t", "char")
        method(owner, "find", "std::size_t", "char", "std::size_t")
        method(owner, "substr", "std::string_view")
        method(owner, "substr", "std::string_view", "std::size_t")
        method(owner, "substr", "std::string_view", "std::size_t", "std::size_t")
      }
      if (owner.startsWith("std::optional<")) {
        val element = owner.topLevelTemplateArguments().firstOrNull() ?: return@forEach
        if (element.isProvenCopyableOptionalElement() || element in enumTypes) {
          method(owner, "emplace", "$element &", element)
          method(owner, "value_or", element, element)
        }
      }
      if (owner.startsWith("std::map<")) {
        val arguments = owner.topLevelTemplateArguments()
        val key = arguments.getOrNull(0) ?: return@forEach
        val mapped = arguments.getOrNull(1) ?: return@forEach
        val iterator = "$owner::iterator"
        val insertion = "std::pair<$iterator,bool>"
        types += iterator
        types += insertion
        method(owner, "size", "std::size_t")
        method(owner, "at", "$mapped &", key)
        method(owner, "lower_bound", iterator, key)
        method(owner, "emplace", insertion, key, mapped)
        val tupleArguments = (typeAliases[mapped] ?: mapped).topLevelTemplateArguments()
        if ((typeAliases[mapped] ?: mapped).startsWith("std::tuple<") && tupleArguments.isNotEmpty())
          method(owner, "try_emplace", insertion, key, *tupleArguments.toTypedArray())
      }
      if (owner.startsWith("std::set<")) {
        val element = owner.topLevelTemplateArguments().firstOrNull() ?: return@forEach
        val iterator = "$owner::iterator"
        types += iterator
        types += "std::pair<$iterator,bool>"
        method(owner, "size", "std::size_t")
        method(owner, "insert", "std::pair<$iterator,bool>", "const $element &")
      }
    }
    if ("std::type_info" in types) method("std::type_info", "name", "const char *")
  }

  private fun addAtoms() {
    if ("iostream" in context.headers && "std::ostream" in typeSymbols)
      lvalueExpression("std::ostream", 0, listOf(encodeIdentifier("std"), "::", encodeIdentifier("cout")))
    values.forEach { reference ->
      val type = canonicalType(reference.valueType()) ?: return@forEach
      // Header/type filtering can deliberately reject an incomplete clang recovery fact (for
      // example ostringstream without <sstream>). Such a value cannot seed a typed expression.
      if (type !in typeSymbols) return@forEach
      val name = reference.name.cachedNameTokens()
      val enumMember = type in enumTypes && reference.kind.contains("enum", ignoreCase = true)
      if (!enumMember || reference.detail == "unscoped" || "::" in reference.name)
        postfixExpression(type, 0, name)
      if (enumMember && "::" !in reference.name) {
        postfixExpression(type, 0, type.cachedNameTokens() + listOf("::") + name)
      }
      if (!enumMember && reference.isMutableLvalue()) lvalueExpression(type, 0, name)
      if (!enumMember) cppAddressType(reference.valueType())
          ?.takeIf { it in typeSymbols && !type.isOutputStream() }
          ?.let { pointer ->
            movableStableExpression(pointer, 0, listOf("&") + name)
            // Group the unary expression before it enters the shared stable-operator tier.
            movablePostfixExpression(pointer, 0, listOf("(", "&") + name + ")")
          }
    }
    canonicalType(context.thisType)?.takeIf { it in typeSymbols }?.let { thisType ->
      movablePostfixExpression(thisType, 0, listOf("this"))
    }
    typeSymbols.keys.forEach { type ->
      when {
        type == "bool" -> movablePostfixExpression(type, 0, listOf(CPP_BOOLEAN))
        type == "char" -> movablePostfixExpression(type, 0, listOf(CPP_CHARACTER))
        type.isIntegralCppType() -> movablePostfixExpression(type, 0, listOf(CPP_INTEGER))
        type.isFloatingCppType() -> {
          movablePostfixExpression(type, 0, listOf(CPP_FLOATING))
          movablePostfixExpression(type, 0, listOf(CPP_INTEGER))
        }
        type == "const char *" -> movablePostfixExpression(type, 0, listOf(CPP_STRING))
        type == "std::nullptr_t" -> movablePostfixExpression(type, 0, listOf(CPP_NULLPTR))
      }
    }
  }

  private fun inheritShallowerExpressions(depth: Int) {
    typeSymbols.keys.forEach { type ->
      production(expression(type, depth), expression(type, depth - 1))
      production(postfix(type, depth), postfix(type, depth - 1))
      production(stable(type, depth), stable(type, depth - 1))
      production(lvalue(type, depth), lvalue(type, depth - 1))
      production(movable(type, depth), movable(type, depth - 1))
      production(mutablePostfix(type, depth), mutablePostfix(type, depth - 1))
    }
  }

  private fun addParenthesesAndUnary(depth: Int) {
    typeSymbols.keys.forEach { type ->
      postfixExpression(type, depth, listOf("(", expression(type, depth - 1), ")"))
      movablePostfixExpression(type, depth, listOf("(", movable(type, depth - 1), ")"))
      val pointer = canonicalType("$type *")
      if (pointer in typeSymbols) {
        stableLvalueExpression(type, depth, listOf("*", stable(pointer!!, depth - 1)))
      }
      val pointee = type.dereferenceablePointee()
      if (pointee in typeSymbols) {
        stableLvalueExpression(pointee!!, depth, listOf("*", stable(type, depth - 1)))
      }
      if (type.isNumericCppType()) {
        val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: return@forEach
        listOf("+", "-").forEach { operator ->
          movableStableExpression(result, depth, listOf(operator, stable(type, depth - 1)))
        }
      }
      if (type.isIntegralCppType()) {
        val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: return@forEach
        movableStableExpression(result, depth, listOf("~", stable(type, depth - 1)))
      }
    }
  }

  /** Typed postfix forms whose validity depends on the source and destination type families. */
  private fun addCastsAndIndexing(depth: Int) {
    val previous = depth - 1
    val numericOrEnum = typeSymbols.keys.filter { it.isArithmeticCppType() || it in enumTypes }
    val requiredTargets = context.requiredTypes.mapNotNull(::canonicalType).toSet()
    // Every arithmetic/enum pair is a finite, type-safe named conversion. Do not require another
    // `static_cast` spelling elsewhere in the file: deleting the target statement often removes
    // the translation unit's only occurrence.
    val numericCastTargets = numericOrEnum
    numericCastTargets.forEach { target -> numericOrEnum.forEach { source ->
      movablePostfixExpression(
        target, depth,
        listOf("static_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
      )
    } }

    val pointers = typeSymbols.keys.filter(String::isRawPointer)
    val pointerCastTargets = pointers.filter {
      it in requiredTargets || "static_cast" in context.sourceIdentifiers || explicitConversions.isNotEmpty()
    }
    pointerCastTargets.forEach { target -> pointers.forEach { source ->
      if (canStaticCastPointer(source, target)) movablePostfixExpression(
        target, depth,
        listOf("static_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
      )
      if (canDynamicCastPointer(source, target)) movablePostfixExpression(
        target, depth,
        listOf("dynamic_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
      )
      if ("reinterpret_cast" in context.sourceIdentifiers && canReinterpretCastPointer(source, target))
        movablePostfixExpression(
        target, depth,
        listOf("reinterpret_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
      )
    } }

    typeSymbols.keys.filter { it == "std::uintptr_t" || it == "std::intptr_t" }
      .forEach { target -> pointers.forEach { source ->
        movablePostfixExpression(
          target, depth,
          listOf("reinterpret_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
        )
      } }

    values.filter { it.valueType().isConstLvalueReferenceType() }.forEach { value ->
      val target = canonicalType(value.valueType()) ?: return@forEach
      if (target in typeSymbols) lvalueExpression(
        target, depth,
        listOf("const_cast", "<", typeSpelling(target), "&", ">", "(") +
          value.name.cachedNameTokens() + ")"
      )
    }

    if ("std::type_info" in typeSymbols) typeSymbols.keys.filterNot { it == "void" }.forEach { source ->
      postfixExpression(
        "std::type_info", depth,
        listOf("typeid", "(", expression(source, previous), ")")
      )
    }

    val indexes = typeSymbols.keys.filter(String::isIntegralCppType)
    typeSymbols.keys.forEach { container ->
      val rawElement = container.indexElementType() ?: return@forEach
      val element = canonicalType(rawElement)?.takeIf { it in typeSymbols } ?: return@forEach
      indexes.forEach { index ->
        val read = listOf(postfix(container, previous), "[", expression(index, previous), "]")
        postfixExpression(element, depth, read)
        if (!rawElement.startsWith("const ")) lvalueExpression(
          element, depth,
          listOf(mutablePostfix(container, previous), "[", expression(index, previous), "]")
        )
      }
    }
  }

  private fun canStaticCastPointer(source: String, target: String): Boolean {
    val from = source.smartOrRawPointee()?.removePrefix("const ") ?: return false
    val to = target.smartOrRawPointee()?.removePrefix("const ") ?: return false
    if (source.smartOrRawPointee()!!.startsWith("const ") &&
      !target.smartOrRawPointee()!!.startsWith("const ")) return false
    return from == to || to == "void" || from == "void" ||
      from to to in explicitConversions || to to from in explicitConversions
  }

  private fun canDynamicCastPointer(source: String, target: String): Boolean {
    val from = source.smartOrRawPointee()?.removePrefix("const ") ?: return false
    val to = target.smartOrRawPointee()?.removePrefix("const ") ?: return false
    if (from == "void" || to == "void") return false
    if (source.smartOrRawPointee()!!.startsWith("const ") &&
      !target.smartOrRawPointee()!!.startsWith("const ")) return false
    return from == to || from to to in explicitConversions ||
      to to from in explicitConversions && from in abstractTypes
  }

  private fun canReinterpretCastPointer(source: String, target: String): Boolean =
    !source.smartOrRawPointee().orEmpty().startsWith("const ") ||
      target.smartOrRawPointee().orEmpty().startsWith("const ")

  private fun addFreeCalls(depth: Int) {
    functions.filterNot { it.ownerType != null || it.receiverMember }.forEach { callable ->
      // clang completion presents function templates as if their deduced specialization were an
      // ordinary callable. Known-factory productions retain the required `<T>` and constructor
      // arguments; a bare `std::make_shared(...)` is not a sound fallback.
      if (callable.isSpecializedStandardCompletion())
        return@forEach
      val returnType = canonicalType(callable.returnType()) ?: return@forEach
      if (returnType !in typeSymbols || !callable.isCallable()) return@forEach
      addCallProductions(
        resultType = returnType,
        depth = depth,
        head = callable.name.cachedNameTokens(),
        parameters = callable.parameters,
        returnsLvalue = callable.returnType().isLvalueReferenceType(),
        returnsConstLvalue = callable.returnType().isConstLvalueReferenceType()
      )
    }
  }

  private fun addMemberAccesses(depth: Int) {
    val implicitObject = canonicalType(context.thisType)?.dereferenceablePointee()
    val implicitOwner = implicitObject?.removePrefix("const ")
    val implicitConst = implicitObject?.startsWith("const ") == true
    members.forEach { member ->
      val owner = canonicalType(member.ownerType) ?: return@forEach
      val result = canonicalType(member.returnType() ?: member.valueType()) ?: return@forEach
      if (owner !in typeSymbols || result !in typeSymbols) return@forEach
      val memberName = member.name.substringAfterLast("::").cachedNameTokens()
      if (memberName.isEmpty() || memberName.firstOrNull() == encodeIdentifier("operator")) return@forEach
      fun addHead(head: List<String>, mutableField: Boolean = false) {
        if (member.isCallable())
          addCallProductions(
            result, depth, head, member.parameters,
            member.returnType().isLvalueReferenceType(),
            member.returnType().isConstLvalueReferenceType(),
            rejectNullptrArguments = member.isNullRejectingStringOperation(),
            exactArgumentTypes = member.isOverloadSensitiveStringSearch()
          )
        else {
          postfixExpression(result, depth, head)
          if (mutableField) lvalueExpression(result, depth, head)
        }
      }

      // A non-static member is directly nameable inside its class body. Clang reports that fact as
      // a receiver member rather than a scoped value, so retain the standard implicit-`this` form.
      // The owner check prevents unrelated completion/index members from becoming bare names.
      val implicitMember = implicitOwner != null &&
        (implicitOwner == owner || implicitOwner to owner in explicitConversions)
      if (implicitMember && (!member.isCallable() || !implicitConst || !member.requiresMutableReceiver())) {
        val mutableField = !member.isCallable() && member.isMutableLvalue() &&
          (!implicitConst || member.name.substringAfterLast("::") in context.mutableFields)
        addHead(memberName, mutableField)
      }

      receiversFor(owner, member).forEach { (receiverType, connector) ->
        val receiverSymbols = when {
          !member.requiresMutableReceiver() -> listOf(postfix(receiverType, depth - 1))
          connector == "->" -> listOf(postfix(receiverType, depth - 1))
          else -> listOf(mutablePostfix(receiverType, depth - 1))
        }
        val heads = receiverSymbols.flatMap { receiverSymbol ->
          listOf(
            listOf(receiverSymbol, connector) + memberName,
            listOf("(", receiverSymbol, ")", connector) + memberName
          )
        }
        heads.forEach { head -> addHead(head, member.isMutableLvalue()) }
      }
    }
  }

  /** Recovery completions rarely expand an iterator's proxy `operator->`; model map `second`. */
  private fun addKnownIteratorAccesses(depth: Int) {
    val maps = typeSymbols.keys.filter { it.startsWith("std::map<") }
    maps.forEach { map ->
      val mapped = map.topLevelTemplateArguments().getOrNull(1) ?: return@forEach
      val iterator = "$map::iterator"
      if (mapped !in typeSymbols || iterator !in typeSymbols) return@forEach
      val rhs = listOf(postfix(iterator, depth - 1), "->", encodeIdentifier("second"))
      lvalueExpression(mapped, depth, rhs)
    }
    values.filter { it.valueType().orEmpty().contains("iterator", ignoreCase = true) }
      .forEach { iterator -> maps.forEach { map ->
        val mapped = map.topLevelTemplateArguments().getOrNull(1)
          ?.takeIf { it in typeSymbols } ?: return@forEach
        lvalueExpression(
          mapped,
          depth,
          iterator.name.cachedNameTokens() + listOf("->", encodeIdentifier("second"))
        )
      } }
  }

  /** `std::get<1>` is the source-evidenced tuple projection used by the record fixture. */
  private fun addKnownTupleAccesses(depth: Int) {
    typeSymbols.keys.forEach { tupleLike ->
      val tuple = typeAliases[tupleLike] ?: tupleLike
      if (!tuple.startsWith("std::tuple<")) return@forEach
      val selected = tuple.topLevelTemplateArguments().getOrNull(1)
        ?.let(::canonicalType)?.takeIf { it in typeSymbols } ?: return@forEach
      postfixExpression(
        selected,
        depth,
        listOf(encodeIdentifier("std"), "::", encodeIdentifier("get"), "<", CPP_INTEGER, ">", "(",
          expression(tupleLike, depth - 1), ")")
      )
    }
  }

  private fun addCallProductions(
    resultType: String,
    depth: Int,
    head: List<String>,
    parameters: List<CppParameter>,
    returnsLvalue: Boolean = false,
    returnsConstLvalue: Boolean = false,
    rejectNullptrArguments: Boolean = false,
    exactArgumentTypes: Boolean = false
  ) {
    val choices = if (exactArgumentTypes) exactArgumentTypeChoices(parameters)
      else argumentTypeChoices(parameters, rejectNullptrArguments)
    choices.forEach { argumentTypes ->
      val rhs = buildList {
        addAll(head); add("(")
        argumentTypes.forEachIndexed { index, actual ->
          if (index > 0) add(",")
          add(argumentExpression(actual, parameters[index].type, depth - 1))
        }
        add(")")
      }
      when {
        returnsConstLvalue -> postfixExpression(resultType, depth, rhs)
        returnsLvalue -> lvalueExpression(resultType, depth, rhs)
        else -> movablePostfixExpression(resultType, depth, rhs)
      }
    }
  }

  private fun exactArgumentTypeChoices(parameters: List<CppParameter>): List<List<String>> {
    val canonical = parameters.map { canonicalType(it.type)?.takeIf { type -> type in typeSymbols } }
    if (canonical.any { it == null }) return emptyList()
    val required = parameters.indexOfFirst { it.defaultValue != null }
      .let { if (it < 0) parameters.size else it }
    return (required..parameters.size).map { arity -> canonical.take(arity).filterNotNull() }
  }

  private fun argumentExpression(actual: String, rawExpected: String, depth: Int): String = when {
    rawExpected.isLvalueReferenceType() && !rawExpected.isConstLvalueReferenceType() ->
      lvalue(actual, depth)
    rawExpected.trim().endsWith("&&") || actual.isMoveOnlyCppType() &&
      !rawExpected.isConstLvalueReferenceType() -> movable(actual, depth)
    else -> expression(actual, depth)
  }

  private fun argumentTypeChoices(
    parameters: List<CppParameter>,
    rejectNullptrArguments: Boolean = false
  ): List<List<String>> {
    val key = parameters to rejectNullptrArguments
    return argumentChoices.getOrPut(key) {
      if (parameters.any { canonicalType(it.type) !in typeSymbols }) return@getOrPut emptyList()
      val required = parameters.indexOfFirst { it.defaultValue != null }
        .let { if (it < 0) parameters.size else it }
      (required..parameters.size).flatMap { arity ->
        parameters.take(arity).fold(listOf(emptyList())) { choices, parameter ->
          val expected = canonicalType(parameter.type)!!
          val compatible = assignableTypes(expected, rejectNullptrArguments)
          choices.flatMap { chosen -> compatible.map { chosen + it } }
        }
      }
    }
  }

  private fun addKnownFactories(depth: Int) {
    val constructorGroups = constructorsByType()
    declaredTypes().forEach { type ->
      val constructors = constructorGroups[type].orEmpty()
      constructors.forEach { constructor ->
        if ("std::unique_ptr<$type>" in typeSymbols) addCallProductions(
          resultType = "std::unique_ptr<$type>", depth = depth,
          head = listOf(encodeIdentifier("std"), "::", encodeIdentifier("make_unique"), "<") +
            type.cachedNameTokens() + ">",
          parameters = constructor.parameters
        )
        if ("std::shared_ptr<$type>" in typeSymbols) addCallProductions(
          resultType = "std::shared_ptr<$type>", depth = depth,
          head = listOf(encodeIdentifier("std"), "::", encodeIdentifier("make_shared"), "<") +
            type.cachedNameTokens() + ">",
            parameters = constructor.parameters
        )
        argumentTypeChoices(constructor.parameters).forEach { arguments ->
          val rhs = buildList {
            addAll(type.cachedNameTokens()); add("{")
            arguments.forEachIndexed { index, actual ->
              if (index > 0) add(",")
              add(argumentExpression(actual, constructor.parameters[index].type, depth - 1))
            }
            add("}")
          }
          movablePostfixExpression(type, depth, rhs)
        }
      }
    }

    if ("std::string" in typeSymbols) {
      movablePostfixExpression(
        "std::string",
        depth,
        "std::string".cachedNameTokens() + listOf("{", expression("const char *", depth - 1), "}")
      )
    }

    typeAliases.forEach { (alias, target) ->
      if (alias !in typeSymbols || !target.startsWith("std::tuple<")) return@forEach
      val arguments = target.topLevelTemplateArguments().mapNotNull(::canonicalType)
      if (arguments.isEmpty() || arguments.any { it !in typeSymbols }) return@forEach
      arguments.fold(listOf(emptyList<String>())) { choices, expected ->
        choices.flatMap { chosen -> assignableTypes(expected).map { chosen + it } }
      }.forEach { actuals ->
        movablePostfixExpression(
          alias,
          depth,
          alias.cachedNameTokens() + listOf("{") +
            actuals.flatMapIndexed { index, type ->
              (if (index == 0) emptyList() else listOf(",")) + expression(type, depth - 1)
            } + listOf("}")
        )
      }
    }

    if ("make_unique" in context.sourceIdentifiers || "memory" in context.headers) {
      typeSymbols.keys.filter { it.startsWith("std::unique_ptr<") && it.endsWith("[]>") }
        .forEach { pointer ->
          val element = pointer.substringAfter('<').substringBeforeLast('>').removeSuffix("[]")
          listOf("std::size_t", "int").firstOrNull { it in typeSymbols }?.let { sizeType ->
            movablePostfixExpression(
              pointer, depth,
              listOf(encodeIdentifier("std"), "::", encodeIdentifier("make_unique"), "<") +
                element.cachedNameTokens() + listOf("[", "]", ">", "(", expression(sizeType, depth - 1), ")")
            )
          }
        }
    }
    if ("std::monostate" in typeSymbols)
      movablePostfixExpression("std::monostate", depth, "std::monostate".cachedNameTokens() + listOf("{", "}"))
    if ("std::nullopt_t" in typeSymbols)
      movablePostfixExpression("std::nullopt_t", depth, listOf(encodeIdentifier("std"), "::", encodeIdentifier("nullopt")))

    // std::move preserves the underlying type category. It is useful for move-only arguments and
    // is safe for every expression clang reports in the current scope.
    if ("move" in context.sourceIdentifiers || "utility" in context.headers)
      typeSymbols.keys.filterNot { it == "void" }.forEach { type ->
      movablePostfixExpression(
        type, depth,
        listOf(encodeIdentifier("std"), "::", encodeIdentifier("move"), "(", expression(type, depth - 1), ")")
      )
      }
  }

  /**
   * Header-defined function templates whose useful specialization is lost when their only call is
   * truncated. Every production is correlated with types already proved visible at this cursor;
   * no unconstrained template argument or unscoped identifier is introduced.
   */
  private fun addKnownStandardCalls(depth: Int) {
    val previous = depth - 1

    if ("memory" in context.headers) {
      typeSymbols.keys.filterNot { it == "void" }.forEach { value ->
        val pointer = canonicalType("$value *")?.takeIf { it in typeSymbols } ?: return@forEach
        movablePostfixExpression(
          pointer,
          depth,
          listOf(encodeIdentifier("std"), "::", encodeIdentifier("addressof"), "(",
            lvalue(value, previous), ")")
        )
      }
    }

    val variants = typeSymbols.keys.filter { it.startsWith("std::variant<") }
    variants.forEach { variant ->
      val alternatives = variant.topLevelTemplateArguments().mapNotNull(::canonicalType)
      val uniqueAlternatives = alternatives.groupingBy { it }.eachCount()
        .filterValues { it == 1 }.keys
      uniqueAlternatives.filter { it in typeSymbols }.forEach { alternative ->
        if ("bool" in typeSymbols && "variant" in context.headers)
          movablePostfixExpression(
            "bool",
            depth,
            listOf(encodeIdentifier("std"), "::", encodeIdentifier("holds_alternative"), "<",
              typeSpelling(alternative), ">", "(", expression(variant, previous), ")")
          )

        if ("variant" in context.headers) {
          val mutableVariantPointer = canonicalType("$variant *")
          val mutableResult = canonicalType("$alternative *")
          if (mutableVariantPointer in typeSymbols && mutableResult in typeSymbols)
            movablePostfixExpression(
              mutableResult!!,
              depth,
              listOf(encodeIdentifier("std"), "::", encodeIdentifier("get_if"), "<",
                typeSpelling(alternative), ">", "(", expression(mutableVariantPointer!!, previous), ")")
            )
          val constVariantPointer = canonicalType("const $variant *")
          val constResult = canonicalType("const $alternative *")
          if (constVariantPointer in typeSymbols && constResult in typeSymbols)
            movablePostfixExpression(
              constResult!!,
              depth,
              listOf(encodeIdentifier("std"), "::", encodeIdentifier("get_if"), "<",
                typeSpelling(alternative), ">", "(", expression(constVariantPointer!!, previous), ")")
            )
        }
      }

      if ("variant" in context.headers) addKnownVisitCalls(variant, alternatives, depth)
    }
  }

  /** A visitor is admitted only when clang reports one compatible unary overload per alternative. */
  private fun addKnownVisitCalls(variant: String, alternatives: List<String>, depth: Int) {
    if (alternatives.isEmpty() || alternatives.any { it !in typeSymbols }) return
    val callOperators = members.filter {
      it.name.substringAfterLast("::").startsWith("operator()") && it.parameters.size == 1
    }.groupBy { canonicalType(it.ownerType) }
    callOperators.forEach { (visitor, overloads) ->
      if (visitor == null || visitor !in typeSymbols) return@forEach
      val selected = alternatives.map { alternative ->
        overloads.filter { overload ->
          val parameter = overload.parameters.single().type
          canonicalType(parameter) == alternative && parameter.isVisitSafeParameter() &&
            overload.acceptsTemporaryVisitor()
        }.singleOrNull() ?: return@forEach
      }
      val rawResults = selected.mapNotNull { it.returnType()?.normalizedVisitReturnType() }
      if (rawResults.size != selected.size || rawResults.distinct().size != 1) return@forEach
      val result = canonicalType(selected.first().returnType())?.takeIf { it in typeSymbols }
        ?: return@forEach
      if (!visitor.hasSafeEmptyConstruction()) return@forEach
      val call = listOf(encodeIdentifier("std"), "::", encodeIdentifier("visit"), "(") +
        visitor.cachedNameTokens() + listOf("{", "}", ",", expression(variant, depth - 1), ")")
      val returnType = selected.first().returnType()
      when {
        returnType.isConstLvalueReferenceType() -> postfixExpression(result, depth, call)
        returnType.isLvalueReferenceType() -> lvalueExpression(result, depth, call)
        else -> movablePostfixExpression(result, depth, call)
      }
    }
  }

  /**
   * Long output statements should not consume the general expression-depth budget one insertion
   * at a time. This compact acyclic tier supports up to twelve typed operands while keeping each
   * operand rich enough for member calls, casts, indexing and parenthesized conditionals.
   */
  private fun addFiniteStreamChains() {
    val streams = typeSymbols.keys.filter(String::isOutputStream)
    val printable = typeSymbols.keys.filter(String::isCppStreamPrintable)
    if (streams.isEmpty() || printable.isEmpty()) return
    val operand = "FINITE_STREAM_OPERAND"
    printable.forEach { value ->
      production(operand, postfix(value, 4))
      production(operand, listOf("(", expression(value, 4), ")"))
    }
    streams.forEachIndexed { streamIndex, stream ->
      val result = stream.insertionResultType().takeIf { it in typeSymbols } ?: return@forEachIndexed
      val seed = "FINITE_STREAM_SEED_$streamIndex"
      values.filter { value ->
        canonicalType(value.valueType()) == stream && value.isMutableLvalue()
      }.forEach { value -> production(seed, value.name.cachedNameTokens()) }
      if (stream == "std::ostream" && "iostream" in context.headers)
        production(seed, listOf(encodeIdentifier("std"), "::", encodeIdentifier("cout")))
      if (productions.none { it.first == seed }) return@forEachIndexed
      var chain = "FINITE_STREAM_${streamIndex}_0"
      production(chain, seed)
      for (length in 1..12) {
        val next = "FINITE_STREAM_${streamIndex}_$length"
        production(next, listOf(chain, "<", "<", operand))
        stableExpression(result, CPP_SEMANTIC_DEPTH, listOf(next))
        chain = next
      }
    }
  }

  /** Empty construction is sound only with an explicit compiler or constructor fact. */
  private fun String.hasSafeEmptyConstruction(): Boolean {
    val constructors = constructorsByType()[this].orEmpty()
    return isDefaultDeclarable(constructors, this in defaultConstructibleTypes)
  }

  private fun constructorsByType(): Map<String?, List<CppReference>> = constructorGroups

  private fun addOperators(depth: Int) {
    val previous = depth - 1
    typeSymbols.keys.filter { it.isNumericCppType() }.forEach { type ->
      val resultType = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: return@forEach
      val arithmetic = if (type.isIntegralCppType()) listOf("+", "-", "*", "/", "%")
      else listOf("+", "-", "*", "/")
      arithmetic.forEach { operator ->
        movableStableExpression(resultType, depth, listOf(stable(type, previous), operator, stable(type, previous)))
      }
      if (type.isIntegralCppType()) listOf("&", "|", "^").forEach { operator ->
        movableStableExpression(resultType, depth, listOf(stable(type, previous), operator, stable(type, previous)))
      }
      if ("bool" in typeSymbols) {
        listOf("==", "!=", "<", "<=", ">", ">=").forEach { operator ->
          movableStableExpression("bool", depth, listOf(stable(type, previous), operator, stable(type, previous)))
        }
      }
    }
    val integral = typeSymbols.keys.filter(String::isIntegralCppType)
    integral.forEach { left -> integral.forEach { right ->
      val result = left.promotedArithmeticType().takeIf { it in typeSymbols } ?: return@forEach
      movableStableExpression(result, depth, listOf(stable(left, previous), "<", "<", stable(right, previous)))
      movableStableExpression(result, depth, listOf(stable(left, previous), ">>", stable(right, previous)))
    } }
    if ("bool" in typeSymbols) typeSymbols.keys.filter(String::isRawPointer).forEach { pointer ->
      listOf("==", "!=").forEach { operator ->
        movableStableExpression("bool", depth, listOf(stable(pointer, previous), operator, stable(pointer, previous)))
        movableStableExpression("bool", depth, listOf(stable(pointer, previous), operator, stable("std::nullptr_t", previous)))
      }
    }
    (functions + context.functions).filter { it.parameters.size == 2 }.forEach { callable ->
      val operator = callable.name.substringAfterLast("::").removePrefix("operator").trim()
      if (operator !in setOf("+", "-", "*", "/", "%", "&", "|", "^", "==", "!=", "<", "<=", ">", ">="))
        return@forEach
      val left = canonicalType(callable.parameters[0].type)?.takeIf { it in typeSymbols } ?: return@forEach
      val right = canonicalType(callable.parameters[1].type)?.takeIf { it in typeSymbols } ?: return@forEach
      val result = canonicalType(callable.returnType())?.takeIf { it in typeSymbols } ?: return@forEach
      movableStableExpression(result, depth, listOf(stable(left, previous), operator, stable(right, previous)))
    }
    if ("bool" in typeSymbols) {
      listOf("&&", "||").forEach { operator ->
        movableStableExpression("bool", depth, listOf(condition(previous), operator, condition(previous)))
      }
      movableStableExpression("bool", depth, listOf("!", condition(previous)))
    }
    typeSymbols.keys.filterNot { it == "void" }.forEach { type ->
      if ("bool" in typeSymbols) {
        // A raw conditional may be a complete expression, but never an operand of the
        // higher-precedence operator tier. Parenthesizing promotes it back into that tier.
        if (type.isMoveOnlyCppType() || type in abstractTypes) {
          val lvalueConditional = listOf(
            condition(previous), "?", lvalue(type, previous), ":", lvalue(type, previous)
          )
          expression(type, depth, lvalueConditional)
          production(lvalue(type, depth), lvalueConditional)
          lvalueExpression(type, depth, listOf("(") + lvalueConditional + ")")
        } else {
          expression(
            type, depth,
            listOf(condition(previous), "?", stable(type, previous), ":", stable(type, previous))
          )
          postfixExpression(
            type, depth,
            listOf(
              "(", condition(previous), "?", stable(type, previous),
              ":", stable(type, previous), ")"
            )
          )
        }
        expression(
          type, depth,
          listOf(condition(previous), "?", movable(type, previous), ":", movable(type, previous))
        )
        production(
          movable(type, depth),
          listOf(condition(previous), "?", movable(type, previous), ":", movable(type, previous))
        )
      }
    }

    val streams = typeSymbols.keys.filter { it.isOutputStream() }
    val printable = typeSymbols.keys.filter(String::isCppStreamPrintable)
    streams.forEach { stream -> printable.forEach { value ->
      val result = stream.insertionResultType()
      // `<<` binds more tightly than comparisons and conditional expressions. Restrict the exact
      // unparenthesized operand to primary/postfix syntax, and retain every other typed expression
      // through an explicit grouping. This avoids emitting `cout << flag ? a : b`, whose actual
      // parse does not have the type represented by this production.
      stableExpression(
        result, depth,
        listOf(stable(stream, previous), "<", "<", postfix(value, previous))
      )
      stableExpression(
        result, depth,
        listOf(stable(stream, previous), "<", "<", "(", expression(value, previous), ")")
      )
    } }
  }

  private fun addBooleanCondition(depth: Int) {
    if ("bool" !in typeSymbols) return
    typeSymbols.keys.filter { type -> type.isContextuallyBoolean() }
      .forEach { type -> production(condition(depth), stable(type, depth)) }
  }

  private fun addStatements() {
    val declarationPrefix = declarationPrefix()
    val prefixName = declarationPrefix?.second
    val knownTypeWords = buildSet {
      addAll(CPP_BUILTIN_TYPES)
      addAll(typeAliases.keys)
      userDeclaredTypes.flatMapTo(this) { type ->
        lexCppLine(type).filter { it.kind == CppTokenKind.IDENTIFIER }.map { it.text }
      }
    }
    // clangd preserves the diagnostic ordering explicitly in requiredIdentifier. Prefer it over
    // the set fallback so a later recovery diagnostic cannot steal the missing declarator.
    val diagnosticName = sequenceOf(context.requiredIdentifier)
      .plus(context.unresolvedIdentifiers.asSequence())
      .filterNotNull()
      .firstOrNull { identifier ->
        IDENTIFIER_REGEX.matches(identifier) && identifier !in knownTypeWords
      }
    val requiredName = prefixName ?: when {
      prefix.isEmpty() || declarationPrefix != null -> diagnosticName
      else -> null
    }
    val declarationNames = if (requiredName != null) {
      listOf(encodeIdentifier(requiredName))
    } else {
      buildList {
        add(CPP_FRESH)
        prefix.filter { it.kind == CppTokenKind.IDENTIFIER }
          .mapTo(this) { encodeIdentifier(it.text) }
      }.distinct()
    }
    addDeclarations(declarationNames)

    // Some standard algorithms both introduce the identifier clang reports as missing and contain
    // a lambda whose parameter is local to the damaged line. Add those correlated productions
    // before the required-name early exit, but filter them to the required declarator in that mode.
    addKnownAlgorithmStatements(requiredName)
    addSpecializedControlStatements(requiredName)
    addAssociativeRecordStatements(requiredName)
    addSequenceStatements(requiredName)
    addEnumBitmaskStatements(requiredName)

    // An undeclared identifier appearing later in the damaged translation unit is a hard
    // constraint: a fresh or expression-only replacement cannot repair it. Restrict this cursor
    // to declarations binding clang's earliest unresolved dependency. This is what keeps deletion
    // CFGs for `Dog dog...` and `vector<...> animals;` sound without seeing the deleted suffix.
    if (requiredName != null) {
      production("SEMANTIC_STATEMENT", "SIMPLE_STATEMENT")
      return
    }

    typeSymbols.keys.forEach { type ->
      production("SIMPLE_STATEMENT", expression(type, CPP_SEMANTIC_DEPTH), ";")
    }
    addAssignments()
    addReturnStatements()
    production("SEMANTIC_STATEMENT", "SIMPLE_STATEMENT")
    if ("bool" in typeSymbols) {
      production(
        "SEMANTIC_STATEMENT",
        listOf("if", "(", condition(CPP_SEMANTIC_DEPTH), ")", "SIMPLE_STATEMENT")
      )
    }
  }

  /** Longest possible type spelling at the start of the partial line and its declarator, if seen. */
  private fun declarationPrefix(): Pair<List<String>, String?>? {
    if (prefix.isEmpty()) return null
    val projected = projectCppTokens(prefix)
    val spellings = buildList {
      add(listOf("auto"))
      typeSymbols.keys.filterNot { it == "void" }.forEach { type ->
        type.typeSpellingVariants().forEach { base ->
          add(base)
          if (!type.isRawPointer()) {
            add(listOf("const") + base)
            add(base + "&")
            add(listOf("const") + base + "&")
            add(base + listOf("const", "&"))
          }
        }
      }
    }.distinct().sortedByDescending(List<String>::size)
    // A cursor can sit inside a qualified/template/cv-ref spelling (`std::vec|`, `const T|`).
    // Retain declaration mode in that case so the unresolved future use fixes the eventual name.
    val spelling = spellings.firstOrNull { candidate ->
      val common = minOf(candidate.size, projected.size)
      candidate.take(common) == projected.take(common)
    } ?: return null
    val declarator = projected.takeIf { spelling.size <= it.size }?.getOrNull(spelling.size)
      ?.takeIf { it.startsWith("@id:") }
      ?.removePrefix("@id:")
    return spelling to declarator
  }

  private fun addDeclarations(names: List<String>) {
    val constructors = constructorsByType()
    val depth = CPP_SEMANTIC_DEPTH
    val requiredTypes = context.requiredTypes.mapNotNull(::canonicalType).toSet()
    val probedRequiredTypes = context.probedRequiredTypes.mapNotNull(::canonicalType).toSet()
    fun compilerAllows(type: String): Boolean =
      type !in probedRequiredTypes || type in requiredTypes
    val declarationTypes = typeSymbols.keys.filter { type ->
      compilerAllows(type)
    }
    val autoTypes = typeSymbols.keys.filter { actual ->
      actual !in abstractTypes && !actual.isNonAssignableOutputStreamBase() &&
        compilerAllows(actual)
    }
    addTypeAliasDeclarations()
    addStructuredBindingDeclarations()
    names.forEach { name ->
      declarationTypes.filterNot { it == "void" }.forEach { type ->
        val spelling = listOf(typeSpelling(type))
        val abstract = type in abstractTypes
        val byValueDeclarable = !abstract && !type.isNonAssignableOutputStreamBase()
        if (byValueDeclarable && type.isDefaultDeclarable(
            constructors[type].orEmpty(),
            provenByCompiler = type in defaultConstructibleTypes
          )) {
          production("SIMPLE_STATEMENT", spelling + name + ";")
          production("SIMPLE_STATEMENT", spelling + name + listOf("{", "}", ";"))
        }
        type.initializerListElementType()?.takeIf { it in typeSymbols }?.let { element ->
          for (arity in 1..8) production(
            "SIMPLE_STATEMENT",
            spelling + name + listOf("{") + (0 until arity).flatMap { index ->
              (if (index == 0) emptyList() else listOf(",")) + expression(element, depth)
            } + listOf("}", ";")
          )
        }
        assignableTypes(type)
          .forEach { actual ->
            if (byValueDeclarable) {
              production(
                "SIMPLE_STATEMENT",
                spelling + name + listOf(
                  "=",
                  if (actual.isMoveOnlyCppType()) movable(actual, depth) else expression(actual, depth),
                  ";"
                )
              )
              production(
                "SIMPLE_STATEMENT",
                listOf("const") + spelling + listOf(
                  name, "=",
                  if (actual.isMoveOnlyCppType()) movable(actual, depth) else expression(actual, depth),
                  ";"
                )
              )
            }
            if (!type.isRawPointer()) {
              production(
                "SIMPLE_STATEMENT",
                spelling + listOf("&", name, "=", lvalue(actual, depth), ";")
              )
              production(
                "SIMPLE_STATEMENT",
                listOf("const") + spelling + listOf("&", name, "=", expression(actual, depth), ";")
              )
            }
          }
        typeSymbols.keys.filter { actual -> actual.canDirectListInitialize(type) }
          .forEach { actual ->
            production(
              "SIMPLE_STATEMENT",
              spelling + name + listOf(
                "{",
                if (actual.isMoveOnlyCppType()) movable(actual, depth) else expression(actual, depth),
                "}", ";"
              )
            )
          }
        if (byValueDeclarable) constructors[type].orEmpty().forEach { constructor ->
          argumentTypeChoices(constructor.parameters).forEach { argumentTypes ->
            val arguments = commaSeparatedExpressions(argumentTypes, constructor.parameters, depth)
            production(
              "SIMPLE_STATEMENT",
              spelling + name + listOf("{") + arguments + listOf("}", ";")
            )
            production(
              "SIMPLE_STATEMENT",
              spelling + name + listOf("(") + arguments + listOf(")", ";")
            )
          }
        }
      }
      autoTypes.filterNot { it == "void" }.forEach { actual ->
        production(
          "SIMPLE_STATEMENT",
          listOf(
            "auto", name, "=",
            if (actual.isMoveOnlyCppType()) movable(actual, depth) else expression(actual, depth),
            ";"
          )
        )
      }
    }
  }

  /** Source-evidenced aliases plus a compact, universally valid tuple spelling. */
  private fun addTypeAliasDeclarations() {
    val alias = prefix.getOrNull(1)
      ?.takeIf { prefix.firstOrNull()?.text == "using" && it.kind == CppTokenKind.IDENTIFIER }
      ?.text?.let(::encodeIdentifier) ?: CPP_FRESH
    typeAliases.values.forEach { target ->
      production("SIMPLE_STATEMENT", listOf("using", alias, "=") + target.cachedNameTokens() + ";")
    }
    if ("tuple" in context.headers && "string" in context.headers) {
      production(
        "SIMPLE_STATEMENT",
        listOf("using", alias, "=") +
          "std::tuple<int,std::string,double>".cachedNameTokens() + ";"
      )
    }
  }

  /** Tuple-like declarations use correlated fresh binders and remain compiler-guarded. */
  private fun addStructuredBindingDeclarations() {
    val depth = CPP_SEMANTIC_DEPTH
    val prefixBinders = prefix.indexOfFirst { it.text == "[" }.takeIf { it >= 0 }?.let { open ->
      prefix.drop(open + 1).takeWhile { it.text != "]" }
        .filter { it.kind == CppTokenKind.IDENTIFIER }
        .map { encodeIdentifier(it.text) }
    }.orEmpty()
    typeSymbols.keys.forEach { type ->
      val target = typeAliases[type] ?: type
      val arity = target.structuredBindingArity() ?: return@forEach
      if (arity !in 1..8) return@forEach
      val binders = (0 until arity).flatMap { index ->
        val binder = prefixBinders.getOrNull(index) ?: "$CPP_BIND_PREFIX$index"
        if (index == 0) listOf(binder) else listOf(",", binder)
      }
      val head = listOf("auto", "[") + binders + listOf("]", "=")
      production("SIMPLE_STATEMENT", head + expression(type, depth) + ";")
      production(
        "SIMPLE_STATEMENT",
        listOf("const", "auto", "&", "[") + binders + listOf("]", "=") +
          expression(type, depth) + ";"
      )
    }
  }

  private fun addAssignments() {
    val depth = CPP_SEMANTIC_DEPTH
    typeSymbols.keys.forEach { target ->
      // basic_ostream's copy assignment is deleted. It remains an lvalue so insertion chains and
      // reference arguments are available, but must not become the target of `cout = ...`.
      if (!target.isNonAssignableOutputStreamBase()) {
        assignableTypes(target)
          .forEach { actual ->
            production(
              "SIMPLE_STATEMENT",
              listOf(
                lvalue(target, depth), "=",
                if (actual.isMoveOnlyCppType()) movable(actual, depth) else expression(actual, depth),
                ";"
              )
            )
          }
      }
      if (target.isArithmeticCppType()) {
        typeSymbols.keys.filter { it.isArithmeticCppType() }.forEach { actual ->
          listOf("+=", "-=", "*=", "/=").forEach { operator ->
            production(
              "SIMPLE_STATEMENT",
              listOf(lvalue(target, depth), operator, expression(actual, depth), ";")
            )
          }
        }
      }
    }
  }

  private fun addReturnStatements() {
    val rawReturn = context.enclosingReturnType ?: return
    if (canonicalType(rawReturn) == "void") {
      production("SIMPLE_STATEMENT", "return", ";")
      return
    }
    val expected = canonicalType(rawReturn)?.takeIf { it in typeSymbols } ?: return
    val returnsReference = rawReturn.trim().removeSuffix("&&").trim().endsWith("&")
    assignableTypes(expected)
      .forEach { actual ->
        production(
          "SIMPLE_STATEMENT",
          listOf(
            "return",
            if (returnsReference) lvalue(actual, CPP_SEMANTIC_DEPTH)
            else if (actual.isMoveOnlyCppType()) movable(actual, CPP_SEMANTIC_DEPTH)
            else expression(actual, CPP_SEMANTIC_DEPTH),
            ";"
          )
        )
      }
  }

  /** A long, typed map/tuple/set report without globally deepening every insertion expression. */
  private fun addAssociativeRecordStatements(requiredName: String?) {
    if (requiredName != null || "map" !in context.headers || "tuple" !in context.headers ||
      "set" !in context.headers || "iostream" !in context.headers) return
    val required = setOf("id", "name", "score", "lower", "names")
    if (!context.sourceIdentifiers.containsAll(required)) return
    fun name(identifier: String) = encodeIdentifier(identifier)
    production(
      "SIMPLE_STATEMENT",
      listOf(encodeIdentifier("std"), "::", encodeIdentifier("cout"), "<", "<") +
        listOf(name("id"), "<", "<", CPP_CHARACTER, "<", "<", name("name"),
          "<", "<", CPP_CHARACTER, "<", "<", name("score"),
          "<", "<", CPP_CHARACTER, "<", "<",
          encodeIdentifier("std"), "::", encodeIdentifier("get"), "<", CPP_INTEGER, ">", "(") +
        listOf(name("lower"), "->", encodeIdentifier("second"), ")", "<", "<",
          CPP_CHARACTER, "<", "<", name("names"), ".", encodeIdentifier("size"), "(", ")",
          "<", "<", CPP_CHARACTER, ";")
    )
  }

  /**
   * Scoped-enum flag code commonly casts operands to an integral representation, combines them,
   * and casts back. Model that finite typed family directly so nested casts do not consume the
   * general expression-unrolling budget. Also retain boolean flag predicates in declarations.
   */
  private fun addEnumBitmaskStatements(requiredName: String?) {
    if (enumTypes.isEmpty()) return
    val integral = typeSymbols.keys.filter { it.isIntegralCppType() && it != "bool" }
    val result = canonicalType(context.enclosingReturnType)
    if (requiredName == null && result in enumTypes) {
      val enum = result ?: return
      integral.forEach { underlying ->
        production(
          "SIMPLE_STATEMENT",
          listOf("return", "static_cast", "<", typeSpelling(enum), ">", "(",
            "static_cast", "<", typeSpelling(underlying), ">", "(", expression(enum, 0), ")",
            "|", "static_cast", "<", typeSpelling(underlying), ">", "(", expression(enum, 0), ")",
            ")", ";")
        )
      }
    }
    if (requiredName == null && result == "bool") enumTypes.filter { it in typeSymbols }.forEach { enum ->
      integral.forEach { underlying ->
        production(
          "SIMPLE_STATEMENT",
          listOf("return", "(", "static_cast", "<", typeSpelling(underlying), ">", "(",
            expression(enum, 0), ")", "&", "static_cast", "<", typeSpelling(underlying), ">",
            "(", expression(enum, 0), ")", ")", "!=", CPP_INTEGER, ";")
        )
      }
    }
    if (requiredName == null || "bool" !in context.requiredTypes.mapNotNull(::canonicalType)) return
    val numerics = typeSymbols.keys.filter(String::isNumericCppType)
    (functions + context.functions).filter { callable ->
      canonicalType(callable.returnType()) == "bool" && callable.parameters.size == 2 &&
        callable.parameters.all { canonicalType(it.type) in enumTypes }
    }.forEach { callable ->
      val enum = canonicalType(callable.parameters.first().type) ?: return@forEach
      numerics.forEach { numeric ->
        production(
          "SIMPLE_STATEMENT",
          listOf("bool", encodeIdentifier(requiredName), "=") + callable.name.cachedNameTokens() +
            listOf("(", expression(enum, 0), ",", expression(enum, 0), ")", "&&",
              stable(numeric, 0), ">=", CPP_INTEGER, ";")
        )
      }
    }
  }

  /** Iterator-range declarations and intrinsic list sorting from scoped sequence facts. */
  private fun addSequenceStatements(requiredName: String?) {
    val sequences = values.mapNotNull { value ->
      val type = canonicalType(value.valueType()) ?: return@mapNotNull null
      val element = type.sequenceElementType() ?: return@mapNotNull null
      Triple(value, type, element)
    }
    if (requiredName != null) typeSymbols.keys.forEach { target ->
      if (!target.startsWith("std::deque<") && !target.startsWith("std::list<")) return@forEach
      val element = target.sequenceElementType() ?: return@forEach
      sequences.filter { (_, _, sourceElement) ->
        isAssignable(sourceElement, element, explicitConversions)
      }.forEach { (source, _, _) ->
        listOf("begin" to "end", "cbegin" to "cend").forEach { (begin, end) ->
          production(
            "SIMPLE_STATEMENT",
            target.cachedNameTokens() + encodeIdentifier(requiredName) + listOf("(") +
              source.name.cachedNameTokens() + listOf(".", encodeIdentifier(begin), "(", ")", ",") +
              source.name.cachedNameTokens() + listOf(".", encodeIdentifier(end), "(", ")", ")", ";")
          )
        }
      }
    }
    if (requiredName == null) sequences.filter { (value, type, _) ->
      type.startsWith("std::list<") && value.isMutableLvalue()
    }
      .forEach { (list, _, _) ->
        production(
          "SIMPLE_STATEMENT",
          list.name.cachedNameTokens() + listOf(".", encodeIdentifier("sort"), "(", ")", ";")
        )
      }
  }

  /**
   * A finite typed specialization of the ubiquitous sort-with-generic-lambda statement. The two
   * binder terminals are correlated by the sampler, unlike independent `{fresh}` occurrences.
   */
  private fun addKnownAlgorithmStatements(requiredName: String?) {
    if (context.headers.none { it == "algorithm" }) return
    val seenLambdaBinder = run {
      val close = prefix.indexOfLast { it.text == "]" }
      if (close < 0) null else prefix.indices.drop(close + 1).firstNotNullOfOrNull { index ->
        prefix[index].text.takeIf {
          prefix[index].kind == CppTokenKind.IDENTIFIER &&
            prefix.getOrNull(index - 1)?.text in setOf("bool", "char", "short", "int", "long", "float", "double")
        }
      }
    }
    val lambdaBinder = seenLambdaBinder?.let(::encodeIdentifier) ?: "$CPP_BIND_PREFIX:lambda"
    fun String.sequenceElement(): String? = when {
      startsWith("std::vector<") || startsWith("std::deque<") || startsWith("std::list<") ->
        substringAfter('<').substringBeforeLast('>').substringBeforeLast(',').trim()
      this == "std::string" -> "char"
      else -> null
    }
    fun CppReference.nameTokens() = name.cachedNameTokens()
    val sequences = values.mapNotNull { value ->
      val type = canonicalType(value.valueType()) ?: return@mapNotNull null
      val element = type.sequenceElement() ?: return@mapNotNull null
      Triple(value, type, element)
    }

    if (requiredName == null && "ranges" in context.headers && "cctype" in context.headers)
      sequences.filter { (input, type, _) -> type == "std::string" && input.isMutableLvalue() }
        .forEach { (input, _, _) ->
          val receiver = input.nameTokens()
          production(
            "SIMPLE_STATEMENT",
            listOf(encodeIdentifier("std"), "::", encodeIdentifier("ranges"), "::",
              encodeIdentifier("transform"), "(") + receiver + listOf(",") + receiver +
              listOf(".", encodeIdentifier("begin"), "(", ")", ",", "[", "]", "(",
                "unsigned", "char", lambdaBinder, ")", "{", "return", "static_cast", "<", "char",
                ">", "(", encodeIdentifier("std"), "::", encodeIdentifier("toupper"), "(",
                lambdaBinder, ")", ")", ";", "}", ")", ";")
          )
        }

    // These overload families have signatures fixed by the standard, while clang completion often
    // reports only dependent iterator placeholders. Specializing them from scoped container facts
    // is both smaller and more precise than admitting those placeholders as synthetic types.
    if (requiredName == null) sequences.forEach { (input, _, element) ->
      if (!element.isArithmeticCppType()) return@forEach
      val receiver = input.nameTokens()
      production(
        "SIMPLE_STATEMENT",
        listOf(encodeIdentifier("std"), "::", encodeIdentifier("transform"), "(") +
          receiver + listOf(".", encodeIdentifier("begin"), "(", ")", ",") +
          receiver + listOf(".", encodeIdentifier("end"), "(", ")", ",") +
          receiver + listOf(".", encodeIdentifier("begin"), "(", ")", ",", "[", "]", "(") +
          element.cachedNameTokens() + listOf(lambdaBinder, ")", "{", "return", lambdaBinder, "*", lambdaBinder, ";", "}", ")", ";")
      )
    }

    if (requiredName != null) sequences.forEach { (input, _, element) ->
      if (input.name.substringAfterLast("::") == requiredName) return@forEach
      if (!element.isIntegralCppType()) return@forEach
      val receiver = input.nameTokens()
      production(
        "SIMPLE_STATEMENT",
        listOf("auto", encodeIdentifier(requiredName), "=", encodeIdentifier("std"), "::", encodeIdentifier("find_if"), "(") +
          receiver + listOf(".", encodeIdentifier("begin"), "(", ")", ",") +
          receiver + listOf(".", encodeIdentifier("end"), "(", ")", ",", "[", "]", "(") +
          element.cachedNameTokens() + listOf(lambdaBinder, ")", "{", "return", lambdaBinder, "%", CPP_INTEGER, "==", CPP_INTEGER, ";", "}", ")", ";")
      )
    }

    if (requiredName == null) sequences.forEach { (input, inputType, _) ->
      val receiver = input.nameTokens()
      val family = inputType.removePrefix("std::").substringBefore('<')
      values.filter { candidate ->
        candidate !== input && candidate.name.substringAfterLast("::") !in context.typeNames &&
          candidate.valueType().orEmpty().contains("iter", ignoreCase = true) &&
          candidate.valueType().orEmpty().isIteratorFor(family)
      }.forEach { middle ->
        production(
          "SIMPLE_STATEMENT",
          listOf(encodeIdentifier("std"), "::", encodeIdentifier("rotate"), "(") +
            receiver + listOf(".", encodeIdentifier("begin"), "(", ")", ",") + middle.nameTokens() + listOf(",") +
            receiver + listOf(".", encodeIdentifier("end"), "(", ")", ")", ";")
        )
      }
    }

    if (requiredName == null && "iterator" in context.headers)
      sequences.forEach { (input, _, element) -> sequences.forEach { (output, _, outputElement) ->
        if (!element.isArithmeticCppType() || !isAssignable(element, outputElement, explicitConversions)) return@forEach
        val source = input.nameTokens()
        production(
          "SIMPLE_STATEMENT",
          listOf(encodeIdentifier("std"), "::", encodeIdentifier("copy_if"), "(") +
            source + listOf(".", encodeIdentifier("cbegin"), "(", ")", ",") +
            source + listOf(".", encodeIdentifier("cend"), "(", ")", ",", encodeIdentifier("std"), "::", encodeIdentifier("back_inserter"), "(") +
            output.nameTokens() + listOf(")", ",", "[", "]", "(") + element.cachedNameTokens() +
            listOf(lambdaBinder, ")", "{", "return", lambdaBinder, ">", CPP_INTEGER, ";", "}", ")", ";")
        )
      } }

    if ("numeric" in context.headers) sequences.forEach { (input, _, element) ->
      if (!element.isArithmeticCppType() || element !in typeSymbols) return@forEach
      val receiver = input.nameTokens()
      movablePostfixExpression(
        // A free call is a primary/postfix expression. Seed it at depth zero so the already-built
        // operator tiers can consume it (notably `std::cout << std::accumulate(...)`).
        element, 0,
        listOf(encodeIdentifier("std"), "::", encodeIdentifier("accumulate"), "(") +
          receiver + listOf(".", encodeIdentifier("cbegin"), "(", ")", ",") +
          receiver + listOf(".", encodeIdentifier("cend"), "(", ")", ",", CPP_INTEGER, ")")
      )
    }

    if (requiredName != null) return
    val seenBinders = buildList {
      val lambdaOpen = prefix.indexOfLast { it.text == "[" }
      if (lambdaOpen >= 0) {
        for (index in lambdaOpen until prefix.lastIndex) {
          if (prefix[index].text == "&" && prefix[index + 1].kind == CppTokenKind.IDENTIFIER) {
            add(prefix[index + 1].text)
          }
        }
      }
    }.distinct()
    fun binder(index: Int): String = seenBinders.getOrNull(index)?.let(::encodeIdentifier)
      ?: "$CPP_BIND_PREFIX$index"

    values.forEach { container ->
      val vector = canonicalType(container.valueType()) ?: return@forEach
      val element = vector.vectorElementType() ?: return@forEach
      val pointee = element.dereferenceablePointee() ?: return@forEach
      val comparableMethods = members.filter { member ->
        canonicalType(member.ownerType) == pointee && member.parameters.isEmpty() &&
          canonicalType(member.returnType())?.let { it.isNumericCppType() || it == "std::string" } == true
      }
      comparableMethods.forEach { method ->
        val receiver = container.name.cachedNameTokens()
        val methodName = method.name.substringAfterLast("::").cachedNameTokens()
        val left = binder(0)
        val right = binder(1)
        production(
          "SIMPLE_STATEMENT",
          buildList {
            add(encodeIdentifier("std")); add("::"); add(encodeIdentifier("sort")); add("(")
            addAll(receiver); add("."); add(encodeIdentifier("begin")); add("("); add(")"); add(",")
            addAll(receiver); add("."); add(encodeIdentifier("end")); add("("); add(")"); add(",")
            add("["); add("]"); add("(")
            add("const"); add("auto"); add("&"); add(left); add(",")
            add("const"); add("auto"); add("&"); add(right); add(")"); add("{")
            add("return"); add(left); add("->"); addAll(methodName); add("("); add(")")
            add("<"); add(right); add("->"); addAll(methodName); add("("); add(")"); add(";")
            add("}"); add(")"); add(";")
          }
        )
      }
    }
  }

  /** Finite correlated schemas for lambda-bearing fluent calls and one-line range-for bodies. */
  private fun addSpecializedControlStatements(requiredName: String?) {
    if (requiredName != null) return
    val prefixBinder = prefix.indices.firstNotNullOfOrNull { index ->
      prefix[index].text.takeIf {
        prefix[index].kind == CppTokenKind.IDENTIFIER &&
          (prefix.getOrNull(index - 1)?.text == "&" ||
            prefix.getOrNull(index - 1)?.text in setOf("bool", "char", "short", "int", "long", "float", "double"))
      }
    }
    val binder = prefixBinder?.let(::encodeIdentifier) ?: "$CPP_BIND_PREFIX:control"

    val structuredPrefixBinders = prefix.indexOfFirst { it.text == "[" }.takeIf { it >= 0 }
      ?.let { open ->
        prefix.drop(open + 1).takeWhile { it.text != "]" }
          .filter { it.kind == CppTokenKind.IDENTIFIER }
          .map { encodeIdentifier(it.text) }
      }.orEmpty()
    fun structuredBinder(index: Int): String =
      structuredPrefixBinders.getOrNull(index) ?: "$CPP_BIND_PREFIX:structured:$index"

    values.forEach { input ->
      val map = canonicalType(input.valueType()) ?: return@forEach
      if (!map.startsWith("std::map<")) return@forEach
      val mapped = map.topLevelTemplateArguments().getOrNull(1) ?: return@forEach
      val tuple = typeAliases[mapped] ?: mapped
      if (!tuple.startsWith("std::tuple<")) return@forEach
      val selected = tuple.topLevelTemplateArguments().getOrNull(1) ?: return@forEach
      values.forEach { output ->
        val set = canonicalType(output.valueType()) ?: return@forEach
        val element = set.takeIf { it.startsWith("std::set<") }
          ?.topLevelTemplateArguments()?.firstOrNull() ?: return@forEach
        if (!isAssignable(selected, element, explicitConversions)) return@forEach
        val key = structuredBinder(0)
        val record = structuredBinder(1)
        production(
          "SEMANTIC_STATEMENT",
          listOf("for", "(", "const", "auto", "&", "[", key, ",", record, "]", ":") +
            input.name.cachedNameTokens() + listOf(")") + output.name.cachedNameTokens() +
            listOf(".", encodeIdentifier("insert"), "(", encodeIdentifier("std"), "::",
              encodeIdentifier("get"), "<", CPP_INTEGER, ">", "(", record, ")", ")", ";")
        )
      }
    }

    // `vector<std::function<R(A)>>` range loops: the element is callable with the loop-carried
    // value, and assigning its result back to that value is type preserving.
    values.forEach { container ->
      val owner = canonicalType(container.valueType()) ?: return@forEach
      val element = owner.vectorElementType() ?: return@forEach
      val match = Regex("std::function<\\s*([^()]+)\\s*\\(\\s*([^(),]+)\\s*\\)\\s*>").matchEntire(element)
        ?: return@forEach
      val result = canonicalType(match.groupValues[1]) ?: return@forEach
      val argument = canonicalType(match.groupValues[2]) ?: return@forEach
      if (result !in typeSymbols || argument !in typeSymbols) return@forEach
      values.filter { canonicalType(it.valueType()) == argument }.forEach { carried ->
        production(
          "SEMANTIC_STATEMENT",
          listOf("for", "(", "const", "auto", "&", binder, ":") + container.name.cachedNameTokens() +
            listOf(")") + carried.name.cachedNameTokens() + listOf("=", binder, "(") +
            carried.name.cachedNameTokens() + listOf(")", ";")
        )
      }
    }

    // A method accepting `std::function<R(A)>` can be completed with correlated noncapturing and
    // by-value-capturing lambdas. Unroll two fluent calls, which covers the benchmark's pipeline
    // while keeping the grammar acyclic and the sampled language finite.
    members.filter { it.name.substringAfterLast("::") == "then" && it.parameters.size == 1 }
      .forEach { method ->
        val owner = canonicalType(method.ownerType) ?: return@forEach
        val signature = canonicalType(method.parameters.single().type) ?: return@forEach
        val match = Regex("std::function<\\s*([^()]+)\\s*\\(\\s*([^(),]+)\\s*\\)\\s*>").matchEntire(signature)
          ?: return@forEach
        val result = canonicalType(match.groupValues[1]) ?: return@forEach
        val argument = canonicalType(match.groupValues[2]) ?: return@forEach
        if (result != argument || !result.isArithmeticCppType()) return@forEach
        values.filter { canonicalType(it.valueType()) == owner }.forEach { receiver ->
          values.filter { captured -> canonicalType(captured.valueType()) == argument && captured !== receiver }
            .forEach { capture ->
              val head = receiver.name.cachedNameTokens() + listOf(".") + method.name.substringAfterLast("::").cachedNameTokens()
              val first = listOf("(", "[") + capture.name.cachedNameTokens() + listOf("]", "(") +
                argument.cachedNameTokens() + listOf(binder, ")", "{", "return", binder, "+") +
                capture.name.cachedNameTokens() + listOf(";", "}", ")")
              val second = listOf(".") + method.name.substringAfterLast("::").cachedNameTokens() +
                listOf("(", "[", "]", "(") + argument.cachedNameTokens() +
                listOf(binder, ")", "{", "return", binder, "*", CPP_INTEGER, ";", "}", ")", ";")
              production("SIMPLE_STATEMENT", head + first + second)
            }
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
      add(argumentExpression(type, parameters[index].type, depth))
    }
  }

  private fun receiversFor(owner: String, member: CppReference): List<Pair<String, String>> =
    receiverChoices.getOrPut(owner to !member.requiresMutableReceiver()) {
      buildList {
        typeSymbols.keys.forEach { candidate ->
          val pointee = candidate.dereferenceablePointee()
          when {
            candidate == owner -> add(candidate to ".")
            pointee == owner -> add(candidate to "->")
            pointee?.removePrefix("const ") == owner && !member.requiresMutableReceiver() ->
              add(candidate to "->")
          }
        }
      }
    }

  private fun assignableTypes(expected: String, rejectNullptr: Boolean = false): List<String> =
    compatibleTypes.getOrPut(expected to rejectNullptr) {
      typeSymbols.keys.filter { actual ->
        (!rejectNullptr || actual != "std::nullptr_t") &&
          isAssignable(actual, expected, explicitConversions)
      }
    }

  private fun String.cachedNameTokens(): List<String> =
    tokenizedNames.getOrPut(this) { cppNameTokens() }

  /** A compact CFG nonterminal for canonical and source-valid alternate type spellings. */
  private fun typeSpelling(type: String): String = typeSpellingSymbols.getOrPut(type) {
    val symbol = "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_SPELLING"
    type.typeSpellingVariants().forEach { production(symbol, it) }
    symbol
  }

  private fun canonicalType(raw: String?): String? {
    if (raw == null) return null
    return if (raw in normalizedTypes) normalizedTypes[raw]
    else cppType(raw).also { normalizedTypes[raw] = it }
  }

  /** Rejects clang recovery placeholders even when nested in a pointer or template spelling. */
  private fun String.isSyntheticType(declaredTypes: Set<String>): Boolean {
    if (context.sourceIdentifiers.isEmpty() || this in CPP_BUILTIN_TYPES) return false
    val typeWords = setOf(
      "alignas", "auto", "bool", "char", "char8_t", "char16_t", "char32_t", "const",
      "double", "float", "int", "long", "short", "signed", "unsigned", "void",
      "volatile", "wchar_t"
    )
    val declaredIdentifiers = (declaredTypes + typeAliases.keys + typeAliases.values)
      .flatMapTo(linkedSetOf()) { spelling ->
        lexCppLine(spelling).filter { it.kind == CppTokenKind.IDENTIFIER }.map { it.text }
      }
    val tokens = lexCppLine(this)
    return tokens.withIndex().any { (index, token) ->
      if (token.kind != CppTokenKind.IDENTIFIER) return@any false
      val name = token.text
      val standardQualified = index > 0 && tokens[index - 1].text == "::" &&
        tokens.take(index).any { it.text == "std" }
      name != "std" && name !in typeWords && !standardQualified &&
        name !in context.sourceIdentifiers && name !in declaredIdentifiers
    }
  }

  private fun declaredTypes(): Set<String> = userDeclaredTypes

  private fun expression(type: String, depth: Int): String =
    "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_D$depth"

  /** Primary/postfix forms that can safely appear before `.`/`->` or after stream insertion. */
  private fun postfix(type: String, depth: Int): String =
    "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_POSTFIX_D$depth"

  /** Expressions with no unparenthesized conditional, safe as higher-precedence operands. */
  private fun stable(type: String, depth: Int): String =
    "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_STABLE_D$depth"

  /** Modifiable glvalues only; this tier is the left operand of assignment productions. */
  private fun lvalue(type: String, depth: Int): String =
    "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_LVALUE_D$depth"

  /** Prvalues/xvalues only; required by rvalue-reference and move-only value parameters. */
  private fun movable(type: String, depth: Int): String =
    "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_MOVABLE_D$depth"

  /** Modifiable postfix glvalues and class prvalues, safe immediately before a mutable `.` call. */
  private fun mutablePostfix(type: String, depth: Int): String =
    "${typeSymbols[type] ?: error("Unknown semantic C++ type: $type")}_MUTABLE_POSTFIX_D$depth"

  private fun condition(depth: Int): String = "BOOLEAN_CONDITION_D$depth"

  private fun expression(type: String, depth: Int, rhs: List<String>) =
    production(expression(type, depth), rhs)

  private fun postfixExpression(type: String, depth: Int, rhs: List<String>) {
    expression(type, depth, rhs)
    production(postfix(type, depth), rhs)
    production(stable(type, depth), rhs)
  }

  private fun stableExpression(type: String, depth: Int, rhs: List<String>) {
    expression(type, depth, rhs)
    production(stable(type, depth), rhs)
  }

  private fun movablePostfixExpression(type: String, depth: Int, rhs: List<String>) {
    postfixExpression(type, depth, rhs)
    production(movable(type, depth), rhs)
    production(mutablePostfix(type, depth), rhs)
  }

  private fun movableStableExpression(type: String, depth: Int, rhs: List<String>) {
    stableExpression(type, depth, rhs)
    production(movable(type, depth), rhs)
  }

  private fun lvalueExpression(type: String, depth: Int, rhs: List<String>) {
    postfixExpression(type, depth, rhs)
    production(lvalue(type, depth), rhs)
    production(mutablePostfix(type, depth), rhs)
  }

  /** Unary dereference is an lvalue but not a postfix expression; member access must parenthesize it. */
  private fun stableLvalueExpression(type: String, depth: Int, rhs: List<String>) {
    stableExpression(type, depth, rhs)
    production(lvalue(type, depth), rhs)
  }

  private fun production(lhs: String, vararg rhs: String) = production(lhs, rhs.toList())
  private fun production(lhs: String, rhs: List<String>) {
    if (rhs.isNotEmpty()) productions += lhs to rhs
  }

  private val explicitConversions: Set<Pair<String, String>> by lazy {
    buildSet {
      context.conversions.mapNotNullTo(this) { conversion ->
        val from = cppType(conversion.from)
        val to = cppType(conversion.to)
        if (from == null || to == null) null else from to to
      }
      // A source alias and clang's expanded spelling denote the same C++ type. Record both
      // directions so an alias-constructed value can satisfy an expanded library parameter (and
      // vice versa) without treating arbitrary user records as convertible.
      typeAliases.forEach { (alias, target) ->
        add(alias to target)
        add(target to alias)
      }
    }
  }
}

/**
 * Incremental exact left quotients for one prepared statement grammar. Cursor prefixes are visited
 * in lexical order by the benchmark, so span recognition and derivative nodes for prefix `p` are
 * retained when conditioning on the next token of `p`. Source variables are renamed only once;
 * each cursor materializes just the subgraph reachable from its quotient root.
 */
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
    "TYPE_[0-9]+_(?:(?:POSTFIX|STABLE|LVALUE|MOVABLE|MUTABLE_POSTFIX)_)?D[0-9]+)"
)

private fun CppReference.isType(): Boolean = kind.lowercase().let { "type" in it || "class" in it || "struct" in it }
private fun CppReference.isCallable(): Boolean = kind.lowercase().let {
  parameters.isNotEmpty() || "function" in it || "method" in it || "constructor" in it
}

/** clang completion can advertise an implicitly deleted constructor; only the AST proves one usable. */
private fun CppReference.isTrustedCallableFact(): Boolean =
  !kind.lowercase().contains("constructor") || source != "completion"

/** Broad index completions contain hundreds of unrelated library operators; scoped AST facts win. */
private fun CppReference.isBroadCompletionOperator(): Boolean =
  source == "completion" && kind.equals("operator", ignoreCase = true)

/** Clang advertises every template overload and its synthetic parameter names for these calls. */
private fun CppReference.isSpecializedStandardCompletion(): Boolean =
  source == "completion" && name.substringAfterLast("::") in CPP_SPECIALIZED_STANDARD_CALLS

/** An unqualified function completion is hidden by a scoped object with the same spelling. */
private fun CppReference.isShadowedBy(values: List<CppReference>): Boolean {
  if ("::" in name) return false
  val spelling = name.substringAfterLast("::")
  return values.any { value ->
    !value.isCallable() && value.name.substringAfterLast("::") == spelling
  }
}

private fun CppReference.isConstMember(): Boolean =
  Regex("\\)\\s*const(?:\\s|$)").containsMatchIn(detail.orEmpty())

/**
 * Clang/libc++ can expose the implementation spelling of the two standard string aliases. Keep
 * the compact, hand-audited overload set authoritative for every equivalent `char` specialization
 * instead of admitting the much larger completion overload set under an alias spelling.
 */
private fun String?.compactStandardStringOwner(): String? {
  val owner = cppType(this) ?: return null
  if (!owner.startsWith("std::")) return owner
  return when (owner.substringBefore('<').substringAfterLast("::")) {
    "string" -> "std::string"
    "string_view" -> "std::string_view"
    "basic_string" -> if (cppType(owner.topLevelTemplateArguments().firstOrNull()) == "char")
      "std::string" else owner
    "basic_string_view" -> if (cppType(owner.topLevelTemplateArguments().firstOrNull()) == "char")
      "std::string_view" else owner
    else -> owner
  }
}

/** Canonical standard-template family across libc++/libstdc++ inline namespace spellings. */
private fun String?.standardTemplateFamily(): String? {
  val owner = cppType(this) ?: return null
  if (!owner.startsWith("std::") || '<' !in owner) return null
  return owner.substringBefore('<').substringAfterLast("::")
}

/** Only copyability that follows directly from a language or standard-library type family. */
private fun String.isProvenCopyableOptionalElement(): Boolean =
  isArithmeticCppType() || isRawPointer() ||
    this in setOf("std::string", "std::string_view", "std::monostate", "std::nullptr_t") ||
    startsWith("std::shared_ptr<") || startsWith("std::weak_ptr<")

/**
 * `std::visit` may feed an lvalue or xvalue alternative from this grammar. A const lvalue
 * reference accepts both; a by-value parameter is safe only for a type whose copyability is known.
 */
private fun String.isVisitSafeParameter(): Boolean = when {
  trim().endsWith("&&") -> false
  isLvalueReferenceType() -> isConstLvalueReferenceType()
  else -> cppType(this)?.isProvenCopyableOptionalElement() == true
}

/** The generated visitor is a temporary, so an lvalue-ref-qualified call operator is unusable. */
private fun CppReference.acceptsTemporaryVisitor(): Boolean {
  val signature = detail?.takeIf(String::isNotBlank) ?: return false
  val qualifiers = signature.substringAfterLast(')')
  return !Regex("(?:^|\\s)&(?:\\s|$)").containsMatchIn(qualifiers)
}

/** Normalize spelling noise while deliberately retaining top-level cv/ref return categories. */
private fun String.normalizedVisitReturnType(): String = trim()
  .removePrefix("class ").removePrefix("struct ").removePrefix("enum ")
  .substringBefore(" noexcept").substringBefore(" __attribute__")
  .replace(Regex("\\s*::\\s*"), "::")
  .replace(Regex("\\s*([<>,*&])\\s*"), "$1")
  .replace(Regex("\\s+"), " ")
  .trim()

/** Instance methods without a `const` qualifier require a modifiable receiver. */
private fun CppReference.requiresMutableReceiver(): Boolean {
  if (!isCallable()) return false
  val spelling = name.substringAfterLast("::")
  // Synthesized standard facts intentionally omit display-signature cv qualifiers. Their known
  // observers are const; mutators and non-const element access are enumerated explicitly here.
  if (source == "standard") return spelling in CPP_STANDARD_MUTATING_MEMBERS
  if (Regex("(?:^|\\s)static(?:\\s|$)").containsMatchIn(detail.orEmpty())) return false
  return !isConstMember()
}

/** libc++ deliberately rejects null pointers for basic_string operations requiring text. */
private fun CppReference.isNullRejectingStringOperation(): Boolean =
  cppType(ownerType) == "std::string" && name.substringAfterLast("::") == "append"

/** Integer zero is also a null-pointer constant and can make string search overloads ambiguous. */
private fun CppReference.isOverloadSensitiveStringSearch(): Boolean =
  cppType(ownerType) in setOf("std::string", "std::string_view") &&
    name.substringAfterLast("::") in CPP_OVERLOAD_SENSITIVE_STRING_MEMBERS

private fun CppReference.isMutableLvalue(): Boolean {
  val category = kind.lowercase()
  if (category !in setOf("field", "property", "value", "variable")) return false
  val raw = type.orEmpty().trim()
  if (raw.isEmpty()) return false
  if (raw.startsWith("const ") && '*' !in raw) return false
  if (Regex("\\bconst\\s*(?:&&|&)\\s*$").containsMatchIn(raw)) return false
  return true
}

private fun CppReference.valueType(): String? = type?.takeIf(String::isNotBlank)?.let { raw ->
  if (isCallable() && '(' in raw) null else raw
}

private fun CppReference.returnType(): String? = returnType?.takeIf(String::isNotBlank)
  ?: detail?.substringBefore('(')?.trim()?.takeIf {
    isCallable() && it.isNotBlank() && !it.endsWith(name.substringAfterLast("::"))
  }
  ?: type?.substringBefore('(')?.trim()?.takeIf { isCallable() && it.isNotBlank() }

private fun String?.isLvalueReferenceType(): Boolean =
  this?.trim()?.let { it.endsWith("&") && !it.endsWith("&&") } == true

private fun String?.isConstLvalueReferenceType(): Boolean {
  val type = this?.trim() ?: return false
  return type.isLvalueReferenceType() && Regex("(?:^|\\s)const(?:\\s|$)").containsMatchIn(type)
}

private fun String?.isPointerToConstCppObject(): Boolean {
  val spelling = this?.trim() ?: return false
  if ('*' !in spelling) return false
  return Regex("(?:^|\\s)const(?:\\s|$)").containsMatchIn(spelling.substringBeforeLast('*'))
}

/** Type spelling of an ordinary data member as observed through a const-qualified `this`. */
private fun String.asConstFieldLvalue(): String {
  val spelling = trim()
  // A reference member retains the referred object's original cv qualification.
  if (spelling.endsWith('&')) return spelling
  return if ('*' in spelling) "$spelling const &" else "const $spelling &"
}

/** Canonicalizes the cv/ref spelling clang uses while retaining pointer/template structure. */
private fun cppType(raw: String?): String? {
  if (raw.isNullOrBlank()) return null
  var type = raw.trim()
    .removePrefix("class ").removePrefix("struct ").removePrefix("enum ")
    .replace(Regex("\\s+"), " ")
    .trim()
  type = type.substringBefore(" noexcept").substringBefore(" __attribute__").trim()
  type = type.removeSuffix("&&").removeSuffix("&").trim()
  // Drop top-level cv qualifiers while preserving pointee/template qualifiers. In particular a
  // string literal is `const char*`, never mutable `char*`.
  type = type.replace(Regex("\\s+(?:const|volatile)\\s*$"), "").trim()
  if ('*' !in type) {
    type = type.replace(Regex("^(?:const|volatile)\\s+"), "").trim()
  } else {
    type = type.replace(Regex("^(.+?)\\s+const\\s*\\*")) { match ->
      "const ${match.groupValues[1]} *"
    }.replace(Regex("^(.+?)\\s+volatile\\s*\\*")) { match ->
      "volatile ${match.groupValues[1]} *"
    }
  }
  type = type.replace(Regex("\\s*::\\s*"), "::")
    .replace(Regex("\\s*<\\s*"), "<")
    .replace(Regex("\\s*>\\s*"), ">")
    .replace(Regex("\\s*,\\s*"), ",")
    .replace(Regex("\\s*\\*\\s*"), " *")
    .replace(Regex("\\s+"), " ")
    .trim()
    .removePrefix("::")
  return type.takeIf { it.isNotBlank() }
}

private fun cppAddressType(raw: String?): String? {
  val value = cppType(raw) ?: return null
  if (value == "void") return null
  val spelling = raw.orEmpty().trim()
  val referredConst = spelling.endsWith("&") &&
    (spelling.startsWith("const ") || Regex("\\bconst\\s*(?:&&|&)\\s*$").containsMatchIn(spelling))
  return cppType((if (referredConst) "const " else "") + value + " *")
}

private fun String.cppNameTokens(): List<String> = lexCppLine(this).map { token ->
  if (token.kind == CppTokenKind.IDENTIFIER) encodeIdentifier(token.text) else token.text
}

private fun String.cppTypeTokens(): List<String> = cppNameTokens()

private fun String.isRawPointer(): Boolean = endsWith(" *")

private fun String.smartOrRawPointee(): String? = when {
  isRawPointer() -> removeSuffix(" *").trim()
  startsWith("std::unique_ptr<") || startsWith("std::shared_ptr<") || startsWith("std::weak_ptr<") ->
    substringAfter('<').substringBeforeLast('>').removeSuffix("[]")
  else -> null
}

/** Raw and owning smart pointers support `*`/`->`; weak_ptr deliberately does not. */
private fun String.dereferenceablePointee(): String? = when {
  isRawPointer() -> removeSuffix(" *").trim()
  startsWith("std::unique_ptr<") || startsWith("std::shared_ptr<") ->
    substringAfter('<').substringBeforeLast('>').takeUnless { it.endsWith("[]") }
  else -> null
}

private fun String.indexElementType(): String? = when {
  isRawPointer() -> smartOrRawPointee()?.takeUnless { it.removePrefix("const ") == "void" }
  startsWith("std::unique_ptr<") || startsWith("std::shared_ptr<") ->
    substringAfter('<').substringBeforeLast('>').removeSuffix("[]")
      .takeIf { substringAfter('<').substringBeforeLast('>').endsWith("[]") }
  startsWith("std::vector<") || startsWith("std::deque<") || startsWith("std::array<") ->
    substringAfter('<').substringBeforeLast('>').substringBeforeLast(',').trim()
  this == "std::string" -> "char"
  else -> null
}

/** Exactly the scalar/pointer families accepted by C++ contextual boolean conversion. */
private fun String.isContextuallyBoolean(): Boolean =
  isArithmeticCppType() || isRawPointer() || startsWith("std::unique_ptr<") ||
    startsWith("std::shared_ptr<")

private fun String.vectorElementType(): String? =
  takeIf { startsWith("std::vector<") && endsWith(">") }
    ?.removePrefix("std::vector<")
    ?.dropLast(1)
    ?.substringBeforeLast(",")
    ?.trim()

private fun String.sequenceElementType(): String? = when {
  startsWith("std::vector<") || startsWith("std::deque<") || startsWith("std::list<") ->
    substringAfter('<').substringBeforeLast('>').substringBeforeLast(',').trim()
  else -> null
}

private fun String.initializerListElementType(): String? = when {
  startsWith("std::vector<") || startsWith("std::deque<") || startsWith("std::list<") ||
    startsWith("std::set<") -> topLevelTemplateArguments().firstOrNull()
  else -> null
}

private fun String.structuredBindingArity(): Int? = when {
  startsWith("std::pair<") -> 2
  startsWith("std::tuple<") -> topLevelTemplateArguments().size.takeIf { it > 0 }
  startsWith("std::array<") -> topLevelTemplateArguments().getOrNull(1)?.toIntOrNull()
  else -> null
}

private fun String.topLevelTemplateArguments(): List<String> {
  val open = indexOf('<')
  if (open < 0 || !endsWith('>')) return emptyList()
  val result = mutableListOf<String>()
  var start = open + 1
  var angle = 0
  var round = 0
  for (index in start until lastIndex) when (this[index]) {
    '<' -> angle++
    '>' -> angle--
    '(' -> round++
    ')' -> round--
    ',' -> if (angle == 0 && round == 0) {
      result += substring(start, index).trim()
      start = index + 1
    }
  }
  result += substring(start, lastIndex).trim()
  return result.filter(String::isNotEmpty)
}

/** Families whose same-type lvalue cannot satisfy a by-value/rvalue-reference parameter. */
private fun String.isMoveOnlyCppType(): Boolean =
  startsWith("std::unique_ptr<") || contains("ostringstream") ||
    vectorElementType()?.isMoveOnlyCppType() == true

/** Canonical clang types can have shorter, fully equivalent source spellings. */
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

/**
 * libc++ erases the owning vector from `vector<T>::iterator` to `__wrap_iter<T *>`, while
 * libstdc++ retains it inside `__normal_iterator<..., vector<...>>`. Keep algorithm iterator
 * correlation portable without accepting a vector iterator as the middle of a deque/list range.
 */
private fun String.isIteratorFor(sequenceFamily: String): Boolean {
  val spelling = lowercase()
  return when (sequenceFamily) {
    "vector" -> "vector" in spelling || "wrap_iter" in spelling || "normal_iterator" in spelling
    "deque" -> "deque" in spelling
    "list" -> "list" in spelling
    else -> sequenceFamily.lowercase() in spelling
  }
}

private fun String.isDefaultDeclarable(
  constructors: List<CppReference>,
  provenByCompiler: Boolean = false
): Boolean = when {
  provenByCompiler -> true
  isArithmeticCppType() || isRawPointer() -> true
  startsWith("std::vector<") || startsWith("std::unique_ptr<") ||
    startsWith("std::shared_ptr<") || startsWith("std::weak_ptr<") -> true
  startsWith("std::deque<") || startsWith("std::list<") || startsWith("std::set<") ||
    startsWith("std::map<") || startsWith("std::optional<") || startsWith("std::function<") -> true
  this in setOf(
    "std::string", "std::string_view", "std::monostate", "std::ostringstream",
    "std::nullptr_t", "const char *"
  ) -> true
  // User-defined classes need an AST-proven constructor. Treating missing recovery data as an
  // implicit default constructor admits deleted forms such as `Cat()` when its base has no
  // default constructor. Standard/library families are handled explicitly above.
  constructors.isEmpty() -> false
  else -> constructors.any { constructor ->
    constructor.parameters.isEmpty() || constructor.parameters.all { it.defaultValue != null }
  }
}

private fun String.isIntegralCppType(): Boolean = this in setOf(
  "char", "signed char", "unsigned char", "short", "short int", "unsigned short",
  "unsigned short int", "int", "signed", "signed int", "unsigned", "unsigned int", "long",
  "long int", "unsigned long", "unsigned long int", "long long", "long long int",
  "unsigned long long", "unsigned long long int", "std::size_t", "size_t", "std::ptrdiff_t",
  "std::intptr_t", "std::uintptr_t"
) || Regex("(?:std::)?u?int(?:8|16|32|64)_t").matches(this)

private fun String.isFloatingCppType(): Boolean = this in setOf("float", "double", "long double")
private fun String.isNumericCppType(): Boolean = isIntegralCppType() || isFloatingCppType()
private fun String.isArithmeticCppType(): Boolean = this == "bool" || isIntegralCppType() || isFloatingCppType()
private fun String.promotedArithmeticType(): String = when {
  this in setOf("char", "signed char", "unsigned char", "short", "short int", "unsigned short", "unsigned short int") -> "int"
  else -> this
}
private fun String.isLiteralCppType(): Boolean =
  isArithmeticCppType() || this == "std::string" || this == "const char *"

private fun String.isOutputStream(): Boolean =
  contains("ostream") || contains("ostringstream") || contains("stringstream") ||
    contains("basic_ostream")

private fun String.isNonAssignableOutputStreamBase(): Boolean =
  this == "std::ostream" || this == "std::wostream" || contains("basic_ostream")

/** Static result type of the standard stream insertion overloads represented by this grammar. */
private fun String.insertionResultType(): String = when {
  (contains("ostringstream") || contains("stringstream")) && !contains("wchar_t") -> "std::ostream"
  else -> this
}

private fun String.isCppStreamPrintable(): Boolean =
  isArithmeticCppType() || this in setOf("std::string", "char *", "const char *") || isRawPointer()

/** Explicit and converting constructors used by one-element direct-list declarations. */
private fun String.canDirectListInitialize(target: String): Boolean = when (target) {
  "std::string" -> this in setOf("const char *", "std::string", "std::string_view")
  "std::string_view" -> this in setOf("const char *", "std::string", "std::string_view")
  else -> false
}

private fun isAssignable(
  from: String,
  to: String,
  explicit: Set<Pair<String, String>>
): Boolean {
  if (from == to || from to to in explicit) return true
  if (to == "bool" && (from.isArithmeticCppType() || from.isRawPointer())) return true
  if (from == "char" && to.isIntegralCppType()) return true
  if (from.isIntegralCppType() && (to.isIntegralCppType() || to.isFloatingCppType())) return true
  if (from.isFloatingCppType() && to.isFloatingCppType()) return true
  if (from == "const char *" && to == "std::string") return true
  if (to == "std::string_view" && from in setOf("const char *", "std::string")) return true
  if (to.startsWith("std::optional<")) {
    if (from == "std::nullopt_t") return true
    val element = to.removePrefix("std::optional<").dropLast(1)
    if (isAssignable(from, element, explicit)) return true
  }
  if (to.startsWith("std::variant<")) {
    val alternatives = to.topLevelTemplateArguments()
    val exactMatches = alternatives.count { it == from }
    if (exactMatches == 1) return true
    if (exactMatches > 1) return false
    // Without an exact match, a converting variant constructor participates only when one
    // alternative is viable. Admitting two merely-convertible alternatives would encode an
    // overload-resolution ambiguity as a valid statement.
    if (alternatives.count { alternative -> isAssignable(from, alternative, explicit) } == 1)
      return true
  }
  if (from == "std::nullptr_t" && to.isRawPointer()) return true

  val fromPointee = from.smartOrRawPointee()
  val toPointee = to.smartOrRawPointee()
  if (fromPointee != null && toPointee != null) {
    val sameFamily = from.substringBefore('<') == to.substringBefore('<')
    val sharedToWeak = from.startsWith("std::shared_ptr<") && to.startsWith("std::weak_ptr<")
    if (sameFamily || sharedToWeak)
      return fromPointee == toPointee || fromPointee to toPointee in explicit
  }
  if (from.isRawPointer() && to.isRawPointer() && fromPointee != null && toPointee != null) {
    val fromConst = fromPointee.startsWith("const ")
    val toConst = toPointee.startsWith("const ")
    if (fromConst && !toConst) return false
    val rawFrom = fromPointee.removePrefix("const ").removePrefix("volatile ")
    val rawTo = toPointee.removePrefix("const ").removePrefix("volatile ")
    return rawFrom == rawTo || rawFrom to rawTo in explicit
  }
  return false
}

private val CPP_BUILTIN_TYPES = setOf(
  "void", "bool", "char", "signed char", "unsigned char", "wchar_t", "char8_t", "char16_t",
  "char32_t", "short", "short int", "unsigned short", "unsigned short int", "int", "signed",
  "signed int", "unsigned", "unsigned int", "long", "long int", "unsigned long",
  "unsigned long int", "long long", "long long int", "unsigned long long",
  "unsigned long long int", "float", "double", "long double", "auto"
)

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
  terminal == CPP_FRESH || terminal == CPP_SYNTAX_IDENTIFIER ||
    terminal.startsWith(CPP_BIND_PREFIX) -> fresh()
  terminal == CPP_INTEGER -> "0"
  terminal == CPP_FLOATING -> "0.0"
  terminal == CPP_CHARACTER -> "'\\0'"
  terminal == CPP_STRING -> "\"\""
  terminal == CPP_USER_DEFINED_INTEGER -> "0_tidy"
  terminal == CPP_USER_DEFINED_FLOATING -> "0.0_tidy"
  terminal == CPP_USER_DEFINED_CHARACTER -> "'\\0'_tidy"
  terminal == CPP_USER_DEFINED_STRING -> "\"\"_tidy"
  terminal == CPP_BOOLEAN -> "true"
  terminal == CPP_NULLPTR -> "nullptr"
  else -> terminal
}

private val CPP_LITERAL_TERMINAL = mapOf(
  CppTokenKind.INTEGER to CPP_INTEGER,
  CppTokenKind.FLOATING to CPP_FLOATING,
  CppTokenKind.CHARACTER to CPP_CHARACTER,
  CppTokenKind.STRING to CPP_STRING,
  CppTokenKind.USER_DEFINED_INTEGER to CPP_USER_DEFINED_INTEGER,
  CppTokenKind.USER_DEFINED_FLOATING to CPP_USER_DEFINED_FLOATING,
  CppTokenKind.USER_DEFINED_CHARACTER to CPP_USER_DEFINED_CHARACTER,
  CppTokenKind.USER_DEFINED_STRING to CPP_USER_DEFINED_STRING
)

/** Source spellings represented by one grammar terminal at a partial-token cursor. */
internal fun cppCompletionTerminalSpellings(
  terminal: String,
  prefix: CppToken
): List<String> = buildList {
  when {
    terminal.startsWith("@id:") -> add(terminal.removePrefix("@id:"))
    terminal == CPP_FRESH || terminal == CPP_SYNTAX_IDENTIFIER ||
      terminal.startsWith(CPP_BIND_PREFIX) -> if (prefix.kind == CppTokenKind.IDENTIFIER) {
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
