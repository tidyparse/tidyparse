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
private const val CPP_EXACT_LITERAL_PREFIX = "@exact-literal:"
private const val CPP_OBSERVED_LITERAL_PREFIX = "@observed-literal:"
private const val SOURCE_EPSILON_RULE = 0
private const val SOURCE_TERMINAL_RULE = 1
private const val SOURCE_UNIT_RULE = 2
private const val SOURCE_BINARY_RULE = 3

private data class CppLiteralClass(
  val tag: String,
  val tokenKind: CppTokenKind,
  val abstractTerminal: String
)

private val CPP_EXACT_LITERAL_CLASSES = listOf(
  CppLiteralClass("integer", CppTokenKind.INTEGER, CPP_INTEGER),
  CppLiteralClass("floating", CppTokenKind.FLOATING, CPP_FLOATING),
  CppLiteralClass("character", CppTokenKind.CHARACTER, CPP_CHARACTER),
  CppLiteralClass("string", CppTokenKind.STRING, CPP_STRING),
  CppLiteralClass("boolean", CppTokenKind.BOOLEAN, CPP_BOOLEAN)
)
private val CPP_EXACT_LITERAL_CLASS_BY_KIND =
  CPP_EXACT_LITERAL_CLASSES.associateBy(CppLiteralClass::tokenKind)
private val CPP_EXACT_LITERAL_CLASS_BY_TAG =
  CPP_EXACT_LITERAL_CLASSES.associateBy(CppLiteralClass::tag)

private data class CppExactLiteral(val literalClass: CppLiteralClass, val spelling: String)

/** One compiler-authoritative standard literal token; public projection remains category-only. */
internal fun cppExactLiteralTerminal(kind: CppTokenKind, spelling: String): String {
  val literalClass = requireNotNull(CPP_EXACT_LITERAL_CLASS_BY_KIND[kind]) {
    "$kind has no exact standard-literal terminal"
  }
  return "$CPP_EXACT_LITERAL_PREFIX${literalClass.tag}:$spelling"
}

internal fun cppExactIntegerTerminal(spelling: String): String =
  cppExactLiteralTerminal(CppTokenKind.INTEGER, spelling)

private fun String.literalMarker(prefix: String): CppExactLiteral? {
  if (!startsWith(prefix)) return null
  val payload = removePrefix(prefix)
  val separator = payload.indexOf(':')
  if (separator <= 0) return null
  val literalClass = CPP_EXACT_LITERAL_CLASS_BY_TAG[payload.substring(0, separator)] ?: return null
  return CppExactLiteral(literalClass, payload.substring(separator + 1))
}

private fun String.exactLiteral(): CppExactLiteral? = literalMarker(CPP_EXACT_LITERAL_PREFIX)
private fun String.observedLiteral(): CppExactLiteral? = literalMarker(CPP_OBSERVED_LITERAL_PREFIX)

private fun cppObservedLiteralTerminal(kind: CppTokenKind, spelling: String): String {
  val literalClass = requireNotNull(CPP_EXACT_LITERAL_CLASS_BY_KIND[kind])
  return "$CPP_OBSERVED_LITERAL_PREFIX${literalClass.tag}:$spelling"
}

/** The sole terminal relation used by exact membership and incremental derivatives. */
private fun cppTerminalMatches(expected: String, observed: String): Boolean {
  if (expected == observed) return true
  val actual = observed.observedLiteral() ?: return false
  val exact = expected.exactLiteral()
  return expected == actual.literalClass.abstractTerminal ||
    exact?.literalClass == actual.literalClass && exact.spelling == actual.spelling
}

/** Metadata carried by semantic/prepared DAGs so ordinary grammars retain exact-equality paths. */
private interface CppExactLiteralGrammar {
  val hasExactLiteralTerminals: Boolean
}

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
  override val hasExactLiteralTerminals: Boolean,
  private val productionLookup: (String) -> List<Pair<String, List<String>>>
) : AbstractSet<Pair<String, List<String>>>(), PreindexedAcyclicCFG, CppExactLiteralGrammar {
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

/** One exact Sema-approved type-name spelling usable at the start of a declaration. */
internal data class CppDeclaratorTypePrefix(
  val tokens: List<String>,
  val requiresTemplateArguments: Boolean = false
)

private val CPP_DECLARATOR_LEADING_CV = setOf("const", "volatile")
private val CPP_DECLARATOR_OPERATORS = setOf("*", "&", "&&")

/**
 * Finds a declared identifier in an already-typed statement prefix. A match must start with the
 * longest available Sema spelling, so expression operators later in a statement are never treated
 * as declarators merely because they precede an identifier.
 */
internal fun cppDeclaratorPrefixBinder(
  prefix: List<CppToken>,
  typePrefixes: Collection<CppDeclaratorTypePrefix>
): String? {
  val projected = projectCppTokens(prefix)
  if (projected.firstOrNull() == "using") {
    return projected.getOrNull(1)?.decodeCppIdentifier()
  }

  val starts = buildList {
    add(0)
    var start = 0
    while (projected.getOrNull(start) in CPP_DECLARATOR_LEADING_CV) {
      start++
      add(start)
    }
  }
  data class Match(val binder: String, val typeEnd: Int, val typeLength: Int)

  fun templateEnd(open: Int): Int? {
    if (projected.getOrNull(open) != "<") return null
    var depth = 0
    for (index in open until projected.size) {
      when (projected[index]) {
        "<" -> depth++
        ">" -> if (--depth == 0) return index + 1
      }
    }
    return null
  }

  fun matchesAt(tokens: List<String>, start: Int): Boolean =
    start + tokens.size <= projected.size && tokens.indices.all { index ->
      tokens[index] == projected[start + index]
    }

  return starts.asSequence().flatMap { start ->
    typePrefixes.asSequence().mapNotNull { type ->
      if (type.tokens.isEmpty() || !matchesAt(type.tokens, start))
        return@mapNotNull null
      var end = start + type.tokens.size
      if (type.requiresTemplateArguments) end = templateEnd(end) ?: return@mapNotNull null

      var sawDeclaratorOperator = false
      while (true) {
        when (projected.getOrNull(end)) {
          in CPP_DECLARATOR_OPERATORS -> {
            sawDeclaratorOperator = true
            end++
          }
          in CPP_DECLARATOR_LEADING_CV -> {
            if (!sawDeclaratorOperator) break
            end++
          }
          else -> break
        }
      }
      val binder = projected.getOrNull(end)?.decodeCppIdentifier() ?: return@mapNotNull null
      Match(binder, end, type.tokens.size)
    }
  }.maxWithOrNull(compareBy<Match>({ it.typeEnd }, { it.typeLength }))?.binder
}

/** Lightweight statement-level inventory for binder detection before a prefix-specific CFG build. */
internal fun cppDeclaratorTypePrefixes(context: CppCompletionContext): List<CppDeclaratorTypePrefix> {
  val prefixes = linkedMapOf<Pair<List<String>, Boolean>, CppDeclaratorTypePrefix>()
  fun add(spelling: String?, requiresTemplateArguments: Boolean = false) {
    val source = spelling?.trim()?.takeIf(String::isNotEmpty) ?: return
    if (source.containsReservedCppIdentifier()) return
    val tokens = source.cppNameTokens()
    if (tokens.isEmpty()) return
    val prefix = CppDeclaratorTypePrefix(tokens, requiresTemplateArguments)
    prefixes[tokens to requiresTemplateArguments] = prefix
  }

  listOf(
    "auto", "void", "bool", "char", "signed char", "unsigned char", "short", "short int",
    "signed short", "signed short int", "unsigned short", "unsigned short int", "int", "signed",
    "signed int", "unsigned", "unsigned int", "long", "long int", "signed long",
    "signed long int", "unsigned long", "unsigned long int", "long long", "long long int",
    "signed long long", "signed long long int", "unsigned long long", "unsigned long long int",
    "float", "double", "long double", "wchar_t", "char8_t", "char16_t", "char32_t"
  ).forEach(::add)
  (context.types.asSequence() + context.completions.asSequence())
    .filter(CppReference::denotesType)
    .mapNotNull { reference ->
      val primaryTemplate = reference.kind.contains("classTemplate", ignoreCase = true) &&
        !reference.kind.contains("specialization", ignoreCase = true)
      // A primary template's dependent QualType is intentionally not source-spellable, while its
      // declaration name is the source spelling consumed by the balanced `<...>` parser.
      if (!primaryTemplate && reference.typeInfo?.isSourceSpellable == false) null
      else reference.name to primaryTemplate
    }.distinct().forEach { (name, primaryTemplate) -> add(name, primaryTemplate) }
  return prefixes.values.toList()
}

private fun String.decodeCppIdentifier(): String? = removePrefix("@id:")
  .takeIf { it != this && IDENTIFIER_REGEX.matches(it) }

internal data class CppConditioningMetrics(
  val derivativeMillis: Long = 0,
  val reachableMillis: Long = 0,
  val boundedMillis: Long = 0
)

/** One exact declaration effect considered by the compiler's downstream-binder probe. */
data class CppBindingProfile(
  val type: String,
  val canonicalType: String? = null,
  val typeInfo: CppTypeInfo? = null,
  val declarationKind: String = "object"
)

/** Positive and negative evidence belongs to one binder; it must never be reused for another. */
data class CppSingletonBindingGate(
  val binder: String,
  val accepted: Set<CppBindingProfile> = emptySet(),
  val probed: Set<CppBindingProfile> = emptySet(),
  val complete: Boolean = false
)

/**
 * Necessary value binders obtained by compiling the whole TU with this statement deleted.
 * A present empty set means the compiler proved there is no downstream obligation; null on the
 * context means the probe was unavailable or inconclusive. Multiple binders stay one correlated
 * obligation and are never flattened into independent declaration alternatives.
 */
data class CppRequiredBinderObligation(
  val binders: Set<String>,
  val singletonGate: CppSingletonBindingGate? = null
) {
  init {
    require(binders.all(IDENTIFIER_REGEX::matches))
    require(singletonGate == null || binders == setOf(singletonGate.binder))
  }
}

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
  /** Whole correlated language expressions constructed successfully by clang/Sema. */
  val expressionWitnesses: List<CppExpressionWitness> = emptyList(),
  /** Whole correlated argument vectors recursively validated by clang/Sema. */
  val callWitnesses: List<CppCallWitness> = emptyList(),
  /** Whole source-order binary expressions accepted by BuildBinOp and definition validation. */
  val binaryOperatorWitnesses: List<CppBinaryOperatorWitness> = emptyList(),
  val unresolvedIdentifiers: Set<String> = emptySet(),
  val requiredBinderObligation: CppRequiredBinderObligation? = null,
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
  val accessibleScopes: List<String> = emptyList(),
  val semanticGraphNodeCount: Int = 0,
  val semanticGraphIsIncomplete: Boolean = false,
  val semanticOperationNodeCount: Int = 0,
  val semanticOperationTemplateCount: Int = 0,
  val semanticOperationsAreIncomplete: Boolean = false,
  val semanticExpressionWitnessesAreIncomplete: Boolean = false,
  val semanticCallWitnessesAreIncomplete: Boolean = false,
  val semanticBinaryOperatorWitnessesAreIncomplete: Boolean = false
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
  /** Canonical identity and immediate cv-shape of an array's element type. */
  val elementCanonicalId: String? = null,
  val elementIsConst: Boolean = false,
  val elementIsVolatile: Boolean = false,
  /** True for `T[]`, false for `T[N]`; null means no authoritative array shape. */
  val isIncompleteArray: Boolean? = null,
  /** Exact decimal outer bound emitted by Sema for a bounded array. */
  val arrayBound: String? = null,
  val isDependent: Boolean = false,
  val isInstantiationDependent: Boolean = false,
  val isSourceSpellable: Boolean? = null,
  /** Whether the type's definition is complete at the semantic query point. */
  val isComplete: Boolean? = null,
  /** Exact Sema result for an ordinary empty initialization, when the endpoint queried it. */
  val isDefaultConstructible: Boolean? = null
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
  /** Clang reports a complete, nonabstract aggregate with no bases or fields. */
  val emptyAggregate: Boolean = false,
  val id: String? = null,
  /** Opaque identity of the primary template that selected this specialization, when applicable. */
  val primaryTemplateId: String? = null,
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
  /** Exact FieldDecl object-kind fact; null means this source did not prove ordinary vs bit-field. */
  val isBitField: Boolean? = null,
  val isVariadic: Boolean = false,
  /** Exact constructor explicitness; null means the endpoint did not report it. */
  val isExplicit: Boolean? = null,
  val templateParameters: List<CppParameter> = emptyList(),
  /** The declaration has a cursor-relevant source path: an exact completion/scope item or a
   * qualified graph type joined by canonical identity to the exact operation closure. */
  val completionVisible: Boolean = false,
  /** This declaration is a viable overload of the call open at the completion cursor. */
  val activeCallable: Boolean = false
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
data class CppConversion(
  val from: String,
  val to: String,
  /** Exact Sema edge category (`base`, `conversion`, or `constructor`), when available. */
  val kind: String? = null,
  /** Canonical endpoint spellings retained for diagnostics and legacy identity fallback. */
  val canonicalFromType: String? = null,
  val canonicalToType: String? = null,
  /** Opaque Sema identities for the two conversion endpoints, when the endpoint supplied them. */
  val fromTypeInfo: CppTypeInfo? = null,
  val toTypeInfo: CppTypeInfo? = null
)

/** Exact projected expression shape used as one position in a compiler-validated call witness. */
data class CppExpressionProfile(
  val kind: String,
  /** Exact source token for synthetic literal profiles; opaque expressions never carry text. */
  val spelling: String? = null,
  /** Clang's ExprObjectKind. Only ordinary objects currently have an exact grammar route. */
  val objectKind: String = "ordinary",
  val type: String? = null,
  val canonicalType: String? = null,
  val typeInfo: CppTypeInfo? = null,
  val valueCategory: String = "prvalue"
)

private data class CppSyntheticLiteralProfile(
  val spelling: String,
  val tokenKind: CppTokenKind,
  val category: String
)

private val CPP_SYNTHETIC_LITERAL_PROFILES = mapOf(
  "integerZero" to CppSyntheticLiteralProfile("0", CppTokenKind.INTEGER, "prvalue"),
  "floatingZero" to CppSyntheticLiteralProfile("0.0", CppTokenKind.FLOATING, "prvalue"),
  "characterZero" to CppSyntheticLiteralProfile("'\\0'", CppTokenKind.CHARACTER, "prvalue"),
  "emptyString" to CppSyntheticLiteralProfile("\"\"", CppTokenKind.STRING, "lvalue"),
  "booleanTrue" to CppSyntheticLiteralProfile("true", CppTokenKind.BOOLEAN, "prvalue"),
  "nullptr" to CppSyntheticLiteralProfile("nullptr", CppTokenKind.NULLPTR, "prvalue")
)

/** Strict schema validation shared by the live endpoint and worker-transport DTO boundaries. */
internal fun CppExpressionProfile.isWellFormedCppExpressionProfile(): Boolean {
  if (objectKind != "ordinary" || typeInfo == null ||
    valueCategory !in setOf("lvalue", "xvalue", "prvalue")
  ) return false
  if (kind == "opaque") return spelling.isNullOrEmpty()
  val literal = CPP_SYNTHETIC_LITERAL_PROFILES[kind] ?: return false
  val source = spelling ?: return false
  if (source != literal.spelling || valueCategory != literal.category) return false
  val token = lexCppLine(source).singleOrNull() ?: return false
  return token.start == 0 && token.end == source.length && token.text == source &&
    token.kind == literal.tokenKind
}

/** Exact source type-id and opaque semantic identity used by an expression witness. */
data class CppTypeProfile(
  val type: String,
  val canonicalType: String? = null,
  val typeInfo: CppTypeInfo
)

/**
 * One ordered explicit function-template argument selected by Sema. [type] is the exact type
 * argument for `type`, and the compiler-reported literal type for `exactIntegerLiteral`.
 */
data class CppTemplateArgumentProfile(
  val kind: String,
  val type: CppTypeProfile,
  val spelling: String? = null,
  val canonicalValue: String? = null
)

/**
 * One indivisible language-expression relation accepted by Sema. The optional operands are
 * interpreted only by [syntax]; a client must never combine either operand across witnesses.
 */
data class CppExpressionWitness(
  val syntax: String,
  val validation: String,
  val typeOperand: CppTypeProfile? = null,
  val expressionOperand: CppExpressionProfile? = null,
  val result: CppExpressionProfile,
  val authoritative: Boolean = false
)

/**
 * A single, inseparable call relation accepted by overload resolution. Function-template
 * witnesses additionally prove deduction, constraints, and recursive definition instantiation;
 * ordinary non-template calls may be authoritative after the call expression selects their exact
 * canonical declaration. Argument positions from distinct witnesses are deliberately never merged.
 */
data class CppCallWitness(
  val name: String,
  val syntax: String,
  val validation: String,
  /** Canonical identity the producer intended the source route to select. */
  val targetId: String? = null,
  val primaryTemplateId: String? = null,
  /** Ordered tagged schema. New producers should use this field exclusively. */
  val explicitTemplateArguments: List<CppTemplateArgumentProfile> = emptyList(),
  /** Legacy type-only schema, accepted only when [explicitTemplateArguments] is absent/empty. */
  val explicitTypeArguments: List<CppTypeProfile> = emptyList(),
  val receiver: CppExpressionProfile? = null,
  val arguments: List<CppExpressionProfile> = emptyList(),
  val callable: CppReference,
  val result: CppExpressionProfile,
  val authoritative: Boolean = false
)

/**
 * One indivisible source-order binary expression accepted by Sema. [left] and [right] are the
 * actual operands passed to BuildBinOp, which is essential for C++20 rewritten comparisons: the
 * selected [callable]'s parameter order and even its operator name may differ from the surface
 * expression. No operand position may therefore be reconstructed from [callable].
 */
data class CppBinaryOperatorWitness(
  val name: String,
  val syntax: String,
  val operatorSpelling: String,
  val validation: String,
  val targetId: String? = null,
  val primaryTemplateId: String? = null,
  val left: CppExpressionProfile,
  val right: CppExpressionProfile,
  val callable: CppReference,
  val result: CppExpressionProfile,
  val authoritative: Boolean = false
)

private val CPP_BINARY_OPERATOR_VALIDATIONS = setOf(
  "semaBinaryOperatorExpression",
  "recursiveDefinitionInstantiation",
  "semaDefaultedDefinition"
)
private val CPP_INFIX_BINARY_OPERATOR_SPELLINGS = setOf(
  "+", "-", "*", "/", "%", "<<", ">>", "<=>", "<", "<=", ">", ">=",
  "==", "!=", "&", "^", "|", "&&", "||"
)
private val CPP_COMPOUND_ASSIGNMENT_OPERATOR_SPELLINGS = setOf(
  "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "^=", "|="
)
private val CPP_BINARY_OPERATOR_SPELLINGS =
  CPP_INFIX_BINARY_OPERATOR_SPELLINGS + CPP_COMPOUND_ASSIGNMENT_OPERATOR_SPELLINGS

/**
 * Checks the producer/selection identity without interpreting the selected declaration's
 * parameters. A specialization is identified by its primary template; an ordinary selected
 * function is identified by its canonical FunctionDecl.
 */
internal fun CppBinaryOperatorWitness.hasWellFormedTargetIdentity(): Boolean {
  if (syntax != "binaryOperator" || name != "operator$operatorSpelling" ||
    operatorSpelling !in CPP_BINARY_OPERATOR_SPELLINGS ||
    validation !in CPP_BINARY_OPERATOR_VALIDATIONS ||
    !selectedOperatorCanImplementSurfaceOperator(
      operatorSpelling, callable.operatorToken()
    )
  ) return false
  val target = targetId.strictCppCallIdentity() ?: return false
  val expectedPrimary = primaryTemplateId.strictCppCallIdentity()
  val selected = callable.id.strictCppCallIdentity() ?: return false
  val selectedPrimary = callable.primaryTemplateId.strictCppCallIdentity()
  return if (expectedPrimary != null || selectedPrimary != null) {
    expectedPrimary != null && selectedPrimary != null &&
      target == expectedPrimary && target == selectedPrimary &&
      validation == "recursiveDefinitionInstantiation"
  } else {
    // A non-function-template operator in a class-template instantiation has an exact selected
    // FunctionDecl identity and an instantiation pattern, but no function primary-template ID.
    // Native recursion validates that selected body before publishing this third label.
    target == selected && validation in CPP_BINARY_OPERATOR_VALIDATIONS
  }
}

internal enum class CppCallTargetKind { FUNCTION_TEMPLATE, ORDINARY_FREE_FUNCTION }

private fun String?.strictCppCallIdentity(): String? =
  this?.takeIf { it.isNotBlank() && it == it.trim() }

/**
 * Validates the redundant producer/selection identity relation before any witness can publish a
 * production. Template targets are identified by their primary; an ordinary target is identified
 * by the selected canonical FunctionDecl itself. Mixing either identity scheme fails closed.
 */
internal fun CppCallWitness.validatedTargetKind(): CppCallTargetKind? {
  val target = targetId.strictCppCallIdentity() ?: return null
  val expectedPrimary = primaryTemplateId.strictCppCallIdentity()
  val selectedPrimary = callable.primaryTemplateId.strictCppCallIdentity()
  return if (expectedPrimary != null || selectedPrimary != null) {
    CppCallTargetKind.FUNCTION_TEMPLATE.takeIf {
      expectedPrimary != null && selectedPrimary != null &&
        target == expectedPrimary && target == selectedPrimary &&
        validation == "recursiveDefinitionInstantiation"
    }
  } else {
    val selected = callable.id.strictCppCallIdentity()
    CppCallTargetKind.ORDINARY_FREE_FUNCTION.takeIf {
      syntax == "freeCall" && selected != null && target == selected &&
        explicitTemplateArguments.isEmpty() && explicitTypeArguments.isEmpty() &&
        validation == "semaCallExpression"
    }
  }
}

internal fun CppCallWitness.hasWellFormedTargetIdentity(): Boolean =
  validatedTargetKind() != null

private val CPP_INTEGER_LITERAL_PROFILE = Regex(
  "(?:(0[xX][0-9A-Fa-f](?:'?[0-9A-Fa-f])*)|" +
    "(0[bB][01](?:'?[01])*)|([0-9](?:'?[0-9])*))" +
    "(?:[uU](?:(?:ll|LL)|[lLzZ])?|(?:(?:ll|LL)|[lLzZ])[uU]?)?"
)

/** Recomputes the mathematical value of one complete, suffix-bearing C++ integer token. */
private fun canonicalCppIntegerLiteralValue(spelling: String): String? {
  if (spelling.isBlank() || spelling != spelling.trim()) return null
  val token = lexCppLine(spelling).singleOrNull()
    ?.takeIf { it.start == 0 && it.end == spelling.length && it.text == spelling }
    ?.takeIf { it.kind == CppTokenKind.INTEGER } ?: return null
  val match = CPP_INTEGER_LITERAL_PROFILE.matchEntire(token.text) ?: return null
  val (digits, radix) = when {
    match.groupValues[1].isNotEmpty() -> match.groupValues[1].drop(2) to 16
    match.groupValues[2].isNotEmpty() -> match.groupValues[2].drop(2) to 2
    else -> match.groupValues[3].let { literal ->
      if (literal.length > 1 && literal.startsWith('0')) literal.drop(1) to 8
      else literal to 10
    }
  }
  val normalizedDigits = digits.replace("'", "").ifEmpty { "0" }
  return runCatching { BigInteger.parseString(normalizedDigits, radix).toString() }.getOrNull()
}

/** Schema-level validation shared by both JS DTO boundaries and semantic lowering. */
internal fun CppTemplateArgumentProfile.isWellFormedCppTemplateArgument(): Boolean {
  return when (kind) {
    "type" -> spelling == null && canonicalValue == null
    "exactIntegerLiteral" -> {
      val literal = spelling ?: return false
      val value = canonicalValue ?: return false
      val info = type.typeInfo
      val semanticType = cppType(type.canonicalType ?: type.type)
      literal.isNotEmpty() && value.matches(Regex("0|[1-9][0-9]*")) &&
        canonicalCppIntegerLiteralValue(literal) == value &&
        info.kind == "builtin" && info.isSourceSpellable == true &&
        !info.isDependent && !info.isInstantiationDependent && info.semanticId() != null &&
        semanticType?.isIntegralOrBooleanCppType() == true
    }
    else -> false
  }
}

/** Migrates a legacy type-only vector without allowing the two schemas to form a hybrid call. */
private fun CppCallWitness.orderedExplicitTemplateArguments(): List<CppTemplateArgumentProfile>? {
  if (explicitTemplateArguments.isNotEmpty() && explicitTypeArguments.isNotEmpty()) return null
  return if (explicitTemplateArguments.isNotEmpty()) explicitTemplateArguments
  else explicitTypeArguments.map { CppTemplateArgumentProfile(kind = "type", type = it) }
}

private fun CFG.containsExactLiteralTerminal(): Boolean =
  (this as? CppExactLiteralGrammar)?.hasExactLiteralTerminals
    ?: any { (_, rhs) -> rhs.any { it.exactLiteral() != null } }

/** Matcher-aware finite membership used only by grammars containing exact literal terminals. */
private class ExactCppTerminalRecognizer(
  grammar: CFG,
  private val startSymbol: String
) {
  private data class Span(val symbol: String, val start: Int, val end: Int)

  private val indexed = grammar as? PreindexedAcyclicCFG
  private val grouped = if (indexed == null) grammar.groupBy { it.first } else emptyMap()
  private val nonterminals = indexed?.acyclicNonterminalIndex?.keys ?: grouped.keys
  private val nullable = mutableMapOf<String, Boolean>()
  private val nullableVisiting = mutableSetOf<String>()

  private fun rules(symbol: String): List<Pair<String, List<String>>> =
    indexed?.productionsFor(symbol) ?: grouped[symbol].orEmpty()

  private fun isNullable(symbol: String): Boolean = nullable.getOrPut(symbol) {
    check(nullableVisiting.add(symbol)) { "Exact C++ residual contains a cycle at $symbol" }
    val result = rules(symbol).any { (_, rhs) ->
      rhs.isEmpty() || rhs.all { it in nonterminals && isNullable(it) }
    }
    nullableVisiting.remove(symbol)
    result
  }

  fun recognizes(tokens: List<String>): Boolean {
    val memo = mutableMapOf<Span, Boolean>()
    fun generates(symbol: String, start: Int, end: Int): Boolean {
      val span = Span(symbol, start, end)
      memo[span]?.let { return it }
      val result = if (start == end) isNullable(symbol) else rules(symbol).any { (_, rhs) ->
        when (rhs.size) {
          0 -> false
          1 -> if (rhs[0] in nonterminals) generates(rhs[0], start, end)
          else end == start + 1 && cppTerminalMatches(rhs[0], tokens[start])
          2 -> (start..end).any { split ->
            fun elementGenerates(element: String, from: Int, to: Int): Boolean =
              if (element in nonterminals) generates(element, from, to)
              else to == from + 1 && cppTerminalMatches(element, tokens[from])
            elementGenerates(rhs[0], start, split) &&
              elementGenerates(rhs[1], split, end)
          }
          else -> error("Exact C++ residual is not in binary normal form: $symbol -> $rhs")
        }
      }
      memo[span] = result
      return result
    }
    return startSymbol in nonterminals && generates(startSymbol, 0, tokens.size)
  }
}

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
  private val usesExactLiteralTerminals: Boolean = false,
  private val recognizesCompleteSyntax: Boolean = false
) {
  val syntax: CFG get() = bounded.grammar
  val forest: PTree? get() = bounded.forest
  val isEmpty: Boolean get() = bounded.isEmpty
  internal val derivationCount: BigInteger get() = bounded.derivationCount

  private val exactTerminalRecognizer by lazy {
    ExactCppTerminalRecognizer(syntax, bounded.startSymbol)
  }

  private fun accepts(candidate: List<String>): Boolean =
    if (usesExactLiteralTerminals) exactTerminalRecognizer.recognizes(candidate)
    else bounded.recognizes(candidate)

  fun recognizes(rawSuffix: List<CppToken>): Boolean {
    val full = projectCppCompletionTokens(
      rawPrefix + rawSuffix, projectionMode, usesExactLiteralTerminals
    )
    val candidate = full.drop(projectedPrefix.size)
    if (candidate.size <= templateTokens && accepts(candidate)) return true
    // Explicit syntax-oracle residuals also recognize witnesses longer than their shortest forest.
    // Semantic/editor residuals leave this disabled and never widen membership with untyped names.
    return recognizesCompleteSyntax && rawSuffix.asSequence()
      .filter { it.kind == CppTokenKind.IDENTIFIER }
      .all { it.text in identifierInventory } &&
      cppSingleStatementSyntaxRecognizes(rawPrefix + rawSuffix)
  }

  /** Returns alpha-renaming alignments that are admitted and must be compiler-guarded. */
  fun freshMatches(rawSuffix: List<CppToken>): List<CppFreshMatch> {
    val matches = mutableListOf<CppFreshMatch>()
    rawSuffix.indices.forEach { suffixIndex ->
      val suffix = rawSuffix.mapIndexed { index, token ->
        if (index == suffixIndex) token.copy(text = CPP_FRESH, kind = CppTokenKind.OTHER) else token
      }
      val full = projectCppCompletionTokens(
        rawPrefix + suffix, projectionMode, usesExactLiteralTerminals
      )
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
        val projected = projectCppCompletionTokens(
          rawPrefix + suffix, projectionMode, usesExactLiteralTerminals
        )
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
    else -> prepare(context, prefix).generate(prefix)
  }
}

/** Reuses one line's scoped semantic grammar while deriving an exact residual at every cursor. */
class PreparedCppCompletionGrammar internal constructor(
  private val sourceSyntax: CFG
) {
  internal val sourceProductionCount: Int get() = sourceSyntax.size
  private val usesExactLiteralTerminals = sourceSyntax.containsExactLiteralTerminal()
  private val conditioner by lazy { FiniteCppConditioner(sourceSyntax) }

  /** Exact prepared-language membership without materializing a residual CFG or CYK index. */
  fun recognizes(statement: List<CppToken>): Boolean =
    conditioner.recognizesExactly(projectCppPreparedTokens(statement, usesExactLiteralTerminals))

  fun generate(prefix: List<CppToken>): CppSuffixGrammar {
    if (prefix.endsCompleteStatement()) return completedStatementGrammar(
      prefix, sourceSyntax, usesExactLiteralTerminals
    )
    val projectedPrefix = projectCppPreparedTokens(prefix, usesExactLiteralTerminals)
    if (projectedPrefix.size > CPP_MAX_STATEMENT_TOKENS)
      return emptyCppSuffixGrammar(prefix, sourceSyntax, usesExactLiteralTerminals)
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
      usesExactLiteralTerminals = usesExactLiteralTerminals
    )
  }
}

private fun emptyCppSuffixGrammar(
  prefix: List<CppToken>,
  sourceSyntax: CFG,
  usesExactLiteralTerminals: Boolean = sourceSyntax.containsExactLiteralTerminal()
): CppSuffixGrammar =
  CppSuffixGrammar(
    bounded = emptySet<Pair<String, List<String>>>().boundedAcyclic(0),
    rawPrefix = prefix,
    projectedPrefix = projectCppPreparedTokens(prefix, usesExactLiteralTerminals),
    templateTokens = 0,
    sourceSyntax = sourceSyntax,
    usesExactLiteralTerminals = usesExactLiteralTerminals
  )

private fun completedStatementGrammar(
  prefix: List<CppToken>,
  sourceSyntax: CFG? = null,
  usesExactLiteralTerminals: Boolean = sourceSyntax?.containsExactLiteralTerminal() == true
): CppSuffixGrammar {
  val epsilon = setOf("START" to emptyList<String>()).freeze()
  return CppSuffixGrammar(
    bounded = epsilon.boundedAcyclic(0),
    rawPrefix = prefix,
    projectedPrefix = projectCppPreparedTokens(prefix, usesExactLiteralTerminals),
    templateTokens = 0,
    sourceSyntax = sourceSyntax ?: epsilon,
    usesExactLiteralTerminals = usesExactLiteralTerminals
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
  // Most semantic rules are structurally unique. Deferring set normalization until the productive
  // CNF is materialized avoids hashing hundreds of thousands of dead lattice edges in Kotlin/JS.
  private val productions = ArrayList<Pair<String, List<String>>>()
  private val values = linkedSetOf<CppReference>()
  private val functions = linkedSetOf<CppReference>()
  private val members = linkedSetOf<CppReference>()
  private val constructors = linkedSetOf<CppReference>()
  private val typeTemplates = linkedSetOf<CppReference>()
  private val rawTypes = linkedSetOf<String>()
  /** Types with an exact expression source/sink; declaration-only TypeDecls stay spelling-only. */
  private val expressionTypes = linkedSetOf<String>()
  private val spellings = mutableMapOf<String, LinkedHashSet<String>>()
  private val normalizedTypes = mutableMapOf<String, String?>()
  private val typeKeysByCanonicalId = mutableMapOf<String, String>()
  private val canonicalIdByTypeKey = mutableMapOf<String, String>()
  private val typeAliases = mutableMapOf<String, String>()
  /** A source spelling is stronger evidence than a normalization alias: only the former may be
   * emitted inside a witness-carried type-id. These maps are frozen before witness profiles are
   * visited, so a profile cannot authenticate its own spelling by registering its opaque ID. */
  private val sourceSpellableTypeAliases = mutableMapOf<String, String>()
  private var preWitnessTypeKeysByCanonicalId: Map<String, String> = emptyMap()
  private var preWitnessTypeAliases: Map<String, String> = emptyMap()
  private var preWitnessSourceSpellableTypeAliases: Map<String, String> = emptyMap()
  private val pointerTypes = mutableMapOf<PointerShape, String>()
  private val pointerShapes = mutableMapOf<String, PointerShape>()
  private val pointerInfos = mutableMapOf<String, CppTypeInfo>()
  /** Pointees proven to be C++ object types by structured Sema metadata. */
  private val objectTypes = linkedSetOf<String>()
  private val declarablePointerPointees = linkedSetOf<String>()
  private val exactTypeOperandSpellingSymbols = mutableMapOf<List<List<String>>, String>()
  private val requiredBindingAllowances = mutableMapOf<Triple<String, String, Cv>, Boolean>()
  private val indexedBindingProfiles by lazy(::indexBindingProfiles)
  private val sourceSpellableTypes = linkedSetOf<String>()
  private val tokenizedNames = mutableMapOf<String, List<String>>()
  private val nameSpellingSymbols = mutableMapOf<String, String>()
  private val typeSpellingSymbols = mutableMapOf<String, String>()
  private val typeSpellingChoiceSymbols = mutableMapOf<List<String>, String>()
  private val pointerSpellingChoiceSymbols = mutableMapOf<List<String>, String>()
  private val compatibleTypes = mutableMapOf<String, List<String>>()
  private val directReferenceTypes = mutableMapOf<String, List<String>>()
  private val builtinTemporaryConversionTypes = mutableMapOf<String, List<String>>()
  /**
   * Generic lvalue/xvalue states do not yet distinguish ordinary objects from bit-fields. If any
   * member of a canonical type is definitely a bit-field, or its declaration source could not
   * prove otherwise, that state cannot satisfy an `OK_Ordinary` compiler witness. Prvalues remain
   * usable because lvalue-to-rvalue conversion produces an ordinary value.
   */
  private val witnessObjectKindTaintedTypes by lazy {
    members.asSequence()
      .filter { member ->
        member.denotesValue() && !member.denotesCallable() && member.isBitField != false
      }
      .mapNotNull { member -> canonicalType(member.semanticType(), member.typeInfo) }
      .toSet()
  }
  /** A template requires recursive definition validation. An exact ordinary free function uses
   * call-expression validation because its body cannot affect overload viability. A witness remains
   * one list element throughout lowering so no positional union can manufacture hybrid calls. */
  private val authoritativeCallWitnesses = context.callWitnesses.filter { witness ->
    witness.authoritative && witness.hasWellFormedTargetIdentity() &&
      witness.syntax in setOf("memberCall", "parenConstruction", "listConstruction", "freeCall")
  }
  private val authoritativeExpressionWitnesses = context.expressionWitnesses.filter { witness ->
    witness.authoritative && witness.validation == "semaExpressionBuild" &&
      witness.syntax in setOf(
        "dynamicCast", "reinterpretCast", "typeidExpression", "typeidType"
      )
  }
  private val authoritativeBinaryOperatorWitnesses = context.binaryOperatorWitnesses.filter {
    witness -> witness.authoritative && witness.hasWellFormedTargetIdentity()
  }
  private lateinit var typeOrder: Map<String, Int>
  private lateinit var arithmeticTypes: List<String>
  private lateinit var conversionSourcesByTarget: Map<String, List<String>>
  private lateinit var baseSourcesByTarget: Map<String, List<String>>
  private lateinit var pointerExpressionTypes: List<String>
  private lateinit var pointerExpressionsByPointee: Map<String, List<String>>
  private val receiverChoices = mutableMapOf<Pair<String, String>, List<Pair<String, String>>>()
  private val qualifiedChoices = mutableMapOf<String, String>()
  private val precedenceChoices = mutableMapOf<String, String>()
  private lateinit var typeSymbols: Map<String, String>
  private lateinit var freeCallWork: List<FreeCallWork>
  private lateinit var constructorsBySemanticType: Map<String?, List<CppReference>>

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

  private data class BindingProfileKey(
    val type: String,
    val declarationKind: String,
    val cv: Cv
  )

  private data class IndexedBindingProfiles(
    val accepted: Set<BindingProfileKey>,
    val probed: Set<BindingProfileKey>,
    val acceptedSyntheticPointers: Set<String>,
    val probedSyntheticPointers: Set<String>,
    val hasAccepted: Boolean,
    val complete: Boolean
  )

  /** Clang's exact ExprValueKind. Keep prvalues and xvalues separate for call witnesses even
   * though ordinary C++ rvalue-reference binding deliberately accepts both categories. */
  private enum class ValueCategory {
    LVALUE,
    XVALUE,
    PRVALUE;

    val isRvalue: Boolean get() = this != LVALUE

    companion object {
      val rvalues = listOf(XVALUE, PRVALUE)
    }
  }
  /** Outermost C++ expression precedence, ordered from strongest to weakest binding. */
  private enum class ExpressionPrecedence {
    POSTFIX,
    UNARY,
    POINTER_TO_MEMBER,
    MULTIPLICATIVE,
    ADDITIVE,
    SHIFT,
    THREE_WAY,
    RELATIONAL,
    EQUALITY,
    BIT_AND,
    BIT_XOR,
    BIT_OR,
    LOGICAL_AND,
    LOGICAL_OR,
    CONDITIONAL;

    fun tighter(): ExpressionPrecedence? = entries.getOrNull(ordinal - 1)
    fun admits(native: ExpressionPrecedence): Boolean = native.ordinal <= ordinal
  }

  private enum class OperatorAssociativity { LEFT, RIGHT, NONE }

  private val nonPostfixPrecedences = ExpressionPrecedence.entries.drop(1)
  private val admittedPrecedences = ExpressionPrecedence.entries.associateWith { limit ->
    ExpressionPrecedence.entries.filter(limit::admits)
  }

  private data class OperatorSyntax(
    val spelling: String,
    val tokens: List<String>,
    val precedence: ExpressionPrecedence,
    val associativity: OperatorAssociativity = OperatorAssociativity.LEFT
  ) {
    val leftLimit: ExpressionPrecedence
      get() = if (associativity == OperatorAssociativity.LEFT) precedence
      else requireNotNull(precedence.tighter())
    val rightLimit: ExpressionPrecedence
      get() = if (associativity == OperatorAssociativity.RIGHT) precedence
      else requireNotNull(precedence.tighter())
  }

  private fun binaryOperatorSyntax(spelling: String): OperatorSyntax? {
    val precedence = when (spelling) {
      "*", "/", "%" -> ExpressionPrecedence.MULTIPLICATIVE
      "+", "-" -> ExpressionPrecedence.ADDITIVE
      "<<", ">>" -> ExpressionPrecedence.SHIFT
      "<=>" -> ExpressionPrecedence.THREE_WAY
      "<", "<=", ">", ">=" -> ExpressionPrecedence.RELATIONAL
      "==", "!=" -> ExpressionPrecedence.EQUALITY
      "&" -> ExpressionPrecedence.BIT_AND
      "^" -> ExpressionPrecedence.BIT_XOR
      "|" -> ExpressionPrecedence.BIT_OR
      "&&" -> ExpressionPrecedence.LOGICAL_AND
      "||" -> ExpressionPrecedence.LOGICAL_OR
      else -> return null
    }
    val tokens = when (spelling) {
      "<<" -> listOf("<", "<")
      ">>" -> listOf(">", ">")
      // The pinned CPP14 lexer tokenizes the C++20 overlay spelling as LessEqual, Greater.
      "<=>" -> listOf("<=", ">")
      else -> listOf(spelling)
    }
    return OperatorSyntax(spelling, tokens, precedence)
  }
  private enum class ActiveArgumentMode { ALL, AGGREGATE, VALUE }

  private lateinit var productiveExpressionOnlyStates: BooleanArray
  private lateinit var productivePostfixOnlyStates: BooleanArray
  private lateinit var productiveExpressionStates: BooleanArray
  private lateinit var productivePostfixTypeStates: BooleanArray
  private lateinit var productiveStableTypeStates: BooleanArray
  private lateinit var exactPostfixTypeStates: BooleanArray
  private lateinit var exactStableTypeStates: BooleanArray
  private lateinit var productiveNativeTypeMasks: IntArray
  private lateinit var productivePostfixStates: BooleanArray
  private lateinit var productiveStableStates: BooleanArray
  private lateinit var productiveNativePrecedenceMasks: IntArray
  private lateinit var linkedNativePrecedenceMasks: IntArray

  private fun genericState(type: String, depth: Int): Int =
    typeOrder.getValue(type) * (CPP_SEMANTIC_DEPTH + 1) + depth

  private fun qualifiedState(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv
  ): Int = (genericState(type, depth) * ValueCategory.entries.size + category.ordinal) * 4 +
    (if (cv.isConst) 1 else 0) + (if (cv.isVolatile) 2 else 0)

  private fun markProductiveNativePrecedence(
    state: Int,
    precedence: ExpressionPrecedence
  ) {
    productiveNativeTypeMasks[state] =
      productiveNativeTypeMasks[state] or (1 shl precedence.ordinal)
  }

  private fun hasExpression(type: String, depth: Int): Boolean =
    productiveExpressionStates[genericState(type, depth)]

  private fun hasStableExpression(type: String, depth: Int): Boolean =
    productiveStableTypeStates[genericState(type, depth)]

  private fun hasPostfixExpression(type: String, depth: Int): Boolean =
    productivePostfixTypeStates[genericState(type, depth)]

  private fun hasPrecedenceExpression(
    type: String,
    depth: Int,
    limit: ExpressionPrecedence
  ): Boolean {
    val mask = productiveNativeTypeMasks[genericState(type, depth)]
    if (mask == 0) return false
    return admittedPrecedences.getValue(limit).any { precedence ->
      mask and (1 shl precedence.ordinal) != 0
    }
  }

  private fun hasQualifiedStableExpression(
    type: String,
    depth: Int,
    categories: List<ValueCategory>,
    target: Cv
  ): Boolean = categories.any { category -> cvVariants.any { cv ->
    (!cv.isConst || target.isConst) && (!cv.isVolatile || target.isVolatile) &&
      productiveStableStates[qualifiedState(type, depth, category, cv)]
  } }

  private fun hasLvalueExpression(type: String, depth: Int): Boolean =
    cvVariants.any { cv -> !cv.isConst &&
      productiveStableStates[qualifiedState(type, depth, ValueCategory.LVALUE, cv)]
    }

  private fun hasRvalueExpression(type: String, depth: Int): Boolean =
    ValueCategory.rvalues.any { category -> cvVariants.any { cv ->
      productiveStableStates[qualifiedState(type, depth, category, cv)]
    } }

  private fun hasMovableExpression(type: String, depth: Int): Boolean =
    ValueCategory.rvalues.any { category ->
      productiveStableStates[qualifiedState(type, depth, category, Cv())]
    }

  private fun markProductiveQualifiedPrecedence(
    state: Int,
    precedence: ExpressionPrecedence
  ) {
    productiveNativePrecedenceMasks[state] =
      productiveNativePrecedenceMasks[state] or (1 shl precedence.ordinal)
  }

  private fun hasProductiveQualifiedPrecedence(
    state: Int,
    precedence: ExpressionPrecedence
  ): Boolean = productiveNativePrecedenceMasks[state] and (1 shl precedence.ordinal) != 0

  private data class ExpressionState(val type: String, val category: ValueCategory, val cv: Cv)

  private data class OperatorEdge(
    val witness: CppBinaryOperatorWitness,
    val left: ExpressionState,
    val right: ExpressionState,
    val result: ExpressionState,
    val syntax: OperatorSyntax
  )

  private val semanticOperatorEdgeCache by lazy(::semanticOperatorEdges)

  private data class ArgumentShape(
    val type: String,
    val binding: String,
    val isConst: Boolean,
    val isVolatile: Boolean,
    val optional: Boolean,
    val pack: Boolean
  )

  private data class ArgumentSource(
    val type: String,
    /** The initializer first undergoes a built-in conversion to the parameter's value type. */
    val convertedTemporary: Boolean = false
  )

  private data class FreeCallWork(
    val callable: CppReference,
    val sourceName: String,
    val concreteResult: String?
  )

  private val argumentLists = mutableMapOf<Pair<List<ArgumentShape>, Int>, List<List<String>>>()
  private val argumentSymbols = mutableMapOf<Pair<ArgumentShape, Int>, String>()
  private val argumentAlternatives =
    mutableMapOf<Pair<ArgumentShape, Int>, List<List<String>>>()

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
      .any { it.typeInfo != null || it.returnTypeInfo != null || it.ownerTypeInfo != null } ||
    context.conversions.any { it.fromTypeInfo != null || it.toTypeInfo != null }

  private val aliases = linkedMapOf<String, String>()
  private val abstractTypes = linkedSetOf<String>()
  private val emptyAggregateTypes = linkedSetOf<String>()
  private val enumTypes = linkedSetOf<String>()
  private val semaDefaultConstructibleTypes = linkedSetOf<String>()
  private val activeArgumentSymbols =
    mutableMapOf<Triple<CppParameter, Int, ActiveArgumentMode>, String>()
  private val activePackSymbols = mutableMapOf<Triple<CppParameter, Int, Int>, String>()
  private val templateArgumentSymbols = mutableMapOf<String, String>()
  private val templateArgumentLists = mutableMapOf<List<CppParameter>, List<List<String>>>()
  private val templateArgumentTypes = linkedSetOf<String>()
  private val templatePackArgumentTypes = linkedSetOf<String>()
  private val contextualActiveResults by lazy(::contextualActiveResultTypes)
  private val defaultConstructibleTypes by lazy {
    buildSet {
      addAll(semaDefaultConstructibleTypes)
      if (!structuredTypes)
        context.defaultConstructibleTypes.mapNotNullTo(this, ::canonicalType)
    }
  }
  private val explicitConversions by lazy {
    buildSet {
      context.conversions.forEach { conversion ->
        val from = canonicalType(conversion.semanticFromType(), conversion.fromTypeInfo)
        val to = canonicalType(conversion.semanticToType(), conversion.toTypeInfo)
        if (from != null && to != null) add(from to to)
      }
      if (!structuredTypes)
        aliases.forEach { (alias, target) -> add(alias to target); add(target to alias) }
    }
  }
  private val directBaseConversions by lazy {
    context.conversions.asSequence().filter { it.kind == "base" }.mapNotNull { conversion ->
      val from = canonicalType(conversion.semanticFromType(), conversion.fromTypeInfo)
      val to = canonicalType(conversion.semanticToType(), conversion.toTypeInfo)
      if (from != null && to != null) from to to else null
    }.toSet()
  }
  /** Conservative transitive closure of compiler-reported public base edges. C++ reference
   * binding, implicit-object conversion, and pointer upcasts cross a public base chain only when
   * the target base subobject is unambiguous. Without virtual-base identity, one graph path proves
   * that relation; two paths are conservatively rejected. A direct base specifier participates in
   * the same count because a second indirect path still makes the conversion ambiguous. */
  private val baseConversions by lazy {
    val outgoing = directBaseConversions.groupBy({ it.first }, { it.second })
    buildSet {
      outgoing.keys.forEach { source ->
        val pathCounts = linkedMapOf<String, Int>()
        fun visit(current: String, path: Set<String>) {
          outgoing[current].orEmpty().forEach { target ->
            if (target in path) return@forEach
            val previous = pathCounts[target] ?: 0
            if (previous >= 2) return@forEach
            pathCounts[target] = previous + 1
            // Expand once per distinct incoming path, capped at two. The second expansion marks
            // every descendant reached through both base subobjects as ambiguous as well.
            visit(target, path + target)
          }
        }
        visit(source, setOf(source))
        pathCounts.filterValues { it == 1 }.keys.forEach { target -> add(source to target) }
      }
    }
  }

  fun build(): CFG {
    collectFacts()
    // `void` has no ordinary value expression, but it is the pointee of a language-provided
    // object-pointer conversion. Record its spelling before resolving exact pointer metadata.
    listOf("void", "bool", "char", "int", "double", "const char *").forEach(::recordType)
    resolvePointerShapes()
    // A public base edge is also an exact language proof for raw pointers to that base. Materialize
    // its four pointee-cv targets even when clang did not separately publish a pointer QualType at
    // this damaged cursor (for example while completing `Base *p = &derived`).
    baseConversions.mapTo(linkedSetOf()) { it.second }
      .filter { it in sourceSpellableTypes }
      .forEach { pointee -> cvVariants.forEach { cv -> recordPointerType(pointee, cv) } }
    // Pointer declarators are grammar syntax over a Sema-spelled pointee. Keep them factored below
    // instead of manufacturing a second semantic type/lattice node for every accessible TypeDecl.
    rawTypes.filterTo(declarablePointerPointees) {
      it.typeShape() != "void" && !isPointer(it) && it in sourceSpellableTypes
    }
    activateExpressionTypes()
    typeSymbols = rawTypes.sorted().mapIndexed { index, type -> type to "TYPE_$index" }.toMap()
    freeCallWork = functions.asSequence()
      .filter { !it.denotesMember() || it.isStaticFact() }
      .mapNotNull { callable ->
        val sourceName = callable.semanticName()
        if (sourceName.cachedNameTokens().isEmpty() || callable.operatorToken() != null)
          return@mapNotNull null
        val concreteResult = canonicalType(
          callable.semanticReturnType(), callable.returnTypeInfo
        )?.takeIf { isConcrete(callable.returnTypeInfo) && it in typeSymbols }
        if (concreteResult == null && !callable.activeCallable) return@mapNotNull null
        FreeCallWork(callable, sourceName, concreteResult)
      }.toList()
    constructorsBySemanticType = constructors.groupBy {
      if (!isConcrete(it.ownerTypeInfo ?: it.returnTypeInfo)) null
      else canonicalType(
        it.canonicalOwnerType ?: it.ownerType ?: it.semanticReturnType() ?: it.name,
        it.ownerTypeInfo ?: it.returnTypeInfo
      )
    }
    // A dependent active template has no concrete return QualType until its arguments are
    // supplied.  When addCalls must use the enclosing declaration/return context for that result,
    // activate the same exact type before building the depth-indexed expression tiers so the call
    // can flow into its declaration initializer. Adding it after [typeSymbols] is intentional:
    // contextualActiveResultTypes resolves only independently recorded, source-spellable type
    // identities and never creates a new type.
    val needsContextualActiveResult = functions.any { callable ->
      callable.activeCallable &&
        canonicalType(callable.semanticReturnType(), callable.returnTypeInfo)
          ?.takeIf { isConcrete(callable.returnTypeInfo) && it in typeSymbols } == null
    }
    if (needsContextualActiveResult) expressionTypes += contextualActiveResults
    typeOrder = typeSymbols.keys.withIndex().associate { (index, type) -> type to index }
    val genericStateCount = typeSymbols.size * (CPP_SEMANTIC_DEPTH + 1)
    productiveExpressionOnlyStates = BooleanArray(genericStateCount)
    productivePostfixOnlyStates = BooleanArray(genericStateCount)
    productiveExpressionStates = BooleanArray(genericStateCount)
    productivePostfixTypeStates = BooleanArray(genericStateCount)
    productiveStableTypeStates = BooleanArray(genericStateCount)
    exactPostfixTypeStates = BooleanArray(genericStateCount)
    exactStableTypeStates = BooleanArray(genericStateCount)
    productiveNativeTypeMasks = IntArray(genericStateCount)
    val qualifiedStateCount = genericStateCount * ValueCategory.entries.size * 4
    productivePostfixStates = BooleanArray(qualifiedStateCount)
    productiveStableStates = BooleanArray(qualifiedStateCount)
    productiveNativePrecedenceMasks = IntArray(qualifiedStateCount)
    linkedNativePrecedenceMasks = IntArray(qualifiedStateCount)
    arithmeticTypes = expressionTypes.filter(String::isArithmeticCppType)
    conversionSourcesByTarget = explicitConversions.asSequence()
      .filter { (source, target) -> source in expressionTypes && target in typeSymbols }
      .groupBy({ it.second }, { it.first })
    baseSourcesByTarget = baseConversions.groupBy({ it.second }, { it.first })
    pointerExpressionTypes = expressionTypes.filter(::isPointer)
    pointerExpressionsByPointee = expressionTypes.mapNotNull { type ->
      pointerShapes[type]?.let { it.pointee to type }
    }.groupBy({ it.first }, { it.second })

    production("START", "SEMANTIC_STATEMENT")
    addAtoms()
    addBooleanCondition(0)
    addExpressionTierLinks(0)
    for (depth in 1..CPP_SEMANTIC_DEPTH) {
      inheritExpressions(depth)
      addLanguageExpressions(depth)
      addExpressionWitnesses(depth)
      addCalls(depth)
      addMemberAccesses(depth)
      addCallWitnesses(depth)
      addOperators(depth)
      addBooleanCondition(depth)
      addExpressionTierLinks(depth)
    }
    addStatements()
    // Delay set normalization until dead rules have been pruned and long productions have been
    // structurally interned. Hashing the full typed lattice is substantially more expensive than
    // hashing its much smaller productive CNF, while both paths retain identical set semantics.
    return finiteAcyclicCnf(productions)
  }

  /** The endpoint is the authority: this layer classifies facts, but never manufactures a name. */
  private fun collectFacts() {
    // Seed opaque conversion identities before broader callable signatures. A reference parameter
    // may carry the same valueCanonicalId through cv/ref erasure; the conversion's own canonical
    // endpoint spelling is the stable value-type shape that must own that semantic key.
    context.conversions.forEach { conversion ->
      recordSemanticType(
        conversion.from,
        conversion.canonicalFromType,
        conversion.fromTypeInfo
      )
      recordSemanticType(
        conversion.to,
        conversion.canonicalToType,
        conversion.toTypeInfo
      )
    }

    fun add(reference: CppReference, owner: String? = reference.ownerType) {
      val fact = if (owner == reference.ownerType) reference else reference.copy(ownerType = owner)
      if (fact.isClassTemplateDeclaration()) typeTemplates += fact
      if (fact.denotesConstructor()) {
        constructors += fact
      } else {
        val enumConstant = fact.denotesEnumConstant()
        // An enum's DeclContext is an ownership fact, not an object-member access path. Scoped
        // enumerators are ambient expressions only when Sema supplied their qualified-id; a bare
        // owned enumerator is not made visible merely because the graph retained its enum owner.
        if (fact.denotesMember() && !enumConstant) members += fact
        val implicitOwnerId = context.thisTypeInfo?.pointeeCanonicalId
        val ownerId = fact.ownerTypeInfo.semanticId()
        val implicitOwner = canonicalType(context.thisType)?.rawPointee()
          ?.removePrefix("const ")?.removePrefix("volatile ")
        val ownerType = canonicalType(fact.canonicalOwnerType ?: fact.ownerType)
        val isImplicitMember = if (implicitOwnerId != null && ownerId != null)
          implicitOwnerId == ownerId
        else ownerType != null && implicitOwner != null && ownerType == implicitOwner
        val explicitlyQualified = "::" in fact.semanticName()
        val hasExactAmbientRoute = explicitlyQualified || isImplicitMember || fact.completionVisible
        // Operation-graph member names are relative to their owner. Static members still need an
        // object/qualified-id outside their class; treating a bare `npos` or factory as a global
        // declaration manufactures invalid expressions. Exact qualified completion spellings and
        // members of the current class remain valid direct-call facts.
        if (fact.denotesCallable() && fact.hasDeducibleTemplateArguments() &&
          (!fact.denotesMember() || fact.isStaticFact() && hasExactAmbientRoute)
        ) functions += fact
        if (fact.denotesValue() && !fact.denotesCallable()) {
          val scopedEnumerator = fact.detail?.trim()?.equals("scoped", ignoreCase = true) == true
          val unscopedEnumerator =
            fact.detail?.trim()?.equals("unscoped", ignoreCase = true) == true
          val ambientEnumerator = enumConstant && (
            explicitlyQualified || !scopedEnumerator &&
              (!fact.denotesMember() || unscopedEnumerator)
          )
          if (ambientEnumerator || !enumConstant &&
            (!fact.denotesMember() || isImplicitMember || fact.isStaticFact() && hasExactAmbientRoute)
          ) values += fact
        }
      }
      recordReferenceTypes(fact)
    }

    context.values.forEach(::add)
    context.functions.forEach(::add)
    context.types.forEach { reference ->
      if (reference.isClassTemplateDeclaration()) typeTemplates += reference
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

    // Everything above is independent cursor/Sema evidence. Seal it before touching a witness:
    // an explicit argument such as `Real, Injected` must not become valid merely because its own
    // profile was the first fact carrying the supplied opaque canonical ID.
    preWitnessTypeKeysByCanonicalId = typeKeysByCanonicalId.toMap()
    preWitnessTypeAliases = typeAliases.toMap()
    preWitnessSourceSpellableTypeAliases = sourceSpellableTypeAliases.toMap()

    // Witness callables are selected specializations, not ordinary overload facts: publishing one
    // through [functions]/[members]/[constructors] would immediately re-form a parameter product.
    // Record only their exact type closure; addCallWitnesses lowers the inseparable vector later.
    authoritativeCallWitnesses.forEach { witness ->
      recordWitnessReferenceTypes(witness.callable)
      witness.orderedExplicitTemplateArguments()?.forEach { argument ->
        val profile = argument.type
        recordWitnessSemanticType(profile.type, profile.canonicalType, profile.typeInfo)
      }
      sequenceOf(witness.receiver, witness.result).filterNotNull()
        .plus(witness.arguments.asSequence()).forEach { profile ->
          recordWitnessSemanticType(profile.type, profile.canonicalType, profile.typeInfo)
        }
    }

    // A binary witness's callable is retained for exact selected-target identity only. Its
    // parameters never become operand grammar facts: rewritten comparisons can reverse operands
    // or select operator<=>/operator== for a different source spelling.
    authoritativeBinaryOperatorWitnesses.forEach { witness ->
      recordWitnessReferenceTypes(witness.callable)
      sequenceOf(witness.left, witness.right, witness.result).forEach { profile ->
        recordWitnessSemanticType(profile.type, profile.canonicalType, profile.typeInfo)
      }
    }

    // Expression witnesses are whole Sema relations, not general cast/conversion facts. Register
    // their exact type closure before pointer resolution and [typeSymbols] indexing.
    authoritativeExpressionWitnesses.forEach { witness ->
      witness.typeOperand?.let { profile ->
        recordWitnessSemanticType(profile.type, profile.canonicalType, profile.typeInfo)
      }
      sequenceOf(witness.expressionOperand, witness.result).filterNotNull().forEach { profile ->
        recordWitnessSemanticType(profile.type, profile.canonicalType, profile.typeInfo)
      }
    }

    // Taking the address of a declaration is a language operation. Its spelling contains no new
    // identifier, and recording it before indexing types keeps `&value` fully generic.
    values.forEach { value ->
      val type = canonicalType(value.semanticType(), value.typeInfo) ?: return@forEach
      val cv = value.objectCv()
      // Spelling a pointer to a cv-qualified pointer requires declarator-aware inner cv placement;
      // omit that address form until such a structured pointer node is supplied directly by Sema.
      if (type.typeShape() != "void" && (!isPointer(type) || cv == Cv()))
        recordPointerType(type, cv)
    }
    deduplicate(values)
    deduplicate(functions)
    deduplicate(members)
    deduplicate(constructors)
  }

  /**
   * A TypeDecl contributes a spelling, but not an expression. Activate the expensive typed
   * expression lattice only when an exact Sema fact can consume or produce that type. Literal
   * types and cv-related pointer cast targets are language-provided expression sources.
   */
  private fun activateExpressionTypes() {
    fun add(raw: String?, info: CppTypeInfo? = null) {
      canonicalType(raw, info)?.takeIf { it in rawTypes }?.let(expressionTypes::add)
    }
    fun add(reference: CppReference) {
      if (reference.denotesCallable()) {
        add(reference.semanticReturnType(), reference.returnTypeInfo)
        reference.parameters.forEach { add(it.semanticType(), it.typeInfo) }
      } else if (reference.denotesValue()) {
        add(reference.semanticType(), reference.typeInfo)
      }
      if (reference.denotesMember() || reference.denotesConstructor())
        add(reference.canonicalOwnerType ?: reference.ownerType, reference.ownerTypeInfo)
    }

    values.forEach(::add)
    functions.forEach(::add)
    members.forEach(::add)
    constructors.forEach(::add)
    authoritativeCallWitnesses.forEach { witness ->
      sequenceOf(witness.receiver, witness.result).filterNotNull()
        .plus(witness.arguments.asSequence()).forEach { profile ->
          add(profile.canonicalType ?: profile.type, profile.typeInfo)
        }
    }
    authoritativeBinaryOperatorWitnesses.forEach { witness ->
      sequenceOf(witness.left, witness.right, witness.result).forEach { profile ->
        add(profile.canonicalType ?: profile.type, profile.typeInfo)
      }
    }
    authoritativeExpressionWitnesses.forEach { witness ->
      sequenceOf(witness.expressionOperand, witness.result).filterNotNull().forEach { profile ->
        add(profile.canonicalType ?: profile.type, profile.typeInfo)
      }
    }
    context.conversions.forEach { conversion ->
      add(conversion.semanticFromType(), conversion.fromTypeInfo)
      add(conversion.semanticToType(), conversion.toTypeInfo)
    }
    add(context.preferredType, context.preferredTypeInfo)
    add(context.baseType, context.baseTypeInfo)
    add(context.enclosingReturnType, context.enclosingReturnTypeInfo)
    add(context.enclosingClassType, context.enclosingClassTypeInfo)
    add(context.thisType, context.thisTypeInfo)
    expressionTypes += emptyAggregateTypes
    rawTypes.filterTo(expressionTypes) { type ->
      type.isArithmeticCppType() || type.typeShape() == "char" ||
        type.typeShape() == "const char *"
    }
    // `&value` is an exact language operation even when clang did not spell the pointer type.
    values.forEach { value ->
      canonicalType(value.semanticType(), value.typeInfo)?.let { type ->
        val cv = value.objectCv()
        if (!isPointer(type) || cv == Cv())
          pointerTypes[PointerShape(type, cv.isConst, cv.isVolatile)]?.let(expressionTypes::add)
      }
    }

    // A productive pointer expression makes every exact cv-only const_cast target productive.
    val activePointees = expressionTypes.mapNotNullTo(linkedSetOf()) { pointerShapes[it]?.pointee }
    pointerShapes.forEach { (type, shape) ->
      if (shape.pointee in activePointees) expressionTypes += type
    }

    // Standard pointer conversions also create exact expression result types. Close only over
    // targets whose structured pointer shapes and Sema provenance prove a legal conversion.
    var added: Boolean
    do {
      val activePointers = expressionTypes.mapNotNull { pointerShapes[it] }
      added = false
      pointerShapes.forEach { (type, target) ->
        if (type !in expressionTypes && activePointers.any { source ->
            isImplicitPointerConversion(source, target)
          }) added = expressionTypes.add(type) || added
      }
    } while (added)
  }

  private fun recordTypeReference(reference: CppReference) {
    recordReferenceTypes(reference)
    // A primary template is not itself a type (`using X = vector;` is ill-formed). Its parameter
    // categories are handled separately by addTypeTemplateDeclarations; concrete specialization
    // nodes arrive from Sema without a primary-template parameter list and remain ordinary types.
    if (!isConcrete(reference.typeInfo)) return
    val canonical = reference.canonicalType ?: reference.type ?: reference.name
    // `name` is clangd's context-correct insertion spelling. A declaration's qualified identity
    // is not interchangeable with it: local using-declarations deliberately insert their short
    // spelling, while names that require a qualifier arrive here with that qualifier included.
    recordType(reference.name, canonical, reference.typeInfo)
    val type = canonicalType(canonical, reference.typeInfo)
    if (reference.abstract) type?.let(abstractTypes::add)
    if (reference.emptyAggregate)
      type?.let(emptyAggregateTypes::add)
    if (reference.kind.contains("enum", ignoreCase = true))
      type?.let(enumTypes::add)
    // A graph walk proves that a type exists and is source-spellable, but does not prove it is a
    // suitable substitution in every primary template. Keep the argument domain to exact
    // completion paths and category-safe scalar/enum/empty-aggregate facts. Concrete pointer
    // types are admitted separately below from their exact Sema spelling.
    if (!reference.abstract && (reference.completionVisible ||
        reference.typeInfo?.kind in setOf("builtin", "enum") || reference.emptyAggregate))
      type?.let(templateArgumentTypes::add)
    // A type alias denotes a complete substitution chosen by its declaration rather than a raw
    // graph-only record name. It is safe to retain as an element candidate for a type-parameter
    // pack, whose arity has no positional policy semantics; fixed policy slots stay conservative.
    if (!reference.abstract && reference.kind.contains("alias", ignoreCase = true))
      type?.let(templatePackArgumentTypes::add)
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
  }

  private fun recordReferenceTypes(reference: CppReference) {
    // Type declarations are recorded by recordTypeReference from their completion insertion text,
    // not their possibly unqualified QualType display text.
    if (!reference.denotesCallable() && !reference.denotesType())
      recordSemanticType(reference.type, reference.canonicalType, reference.typeInfo)
    recordSemanticType(reference.returnType, reference.canonicalReturnType, reference.returnTypeInfo)
    recordSemanticType(reference.ownerType, reference.canonicalOwnerType, reference.ownerTypeInfo)
    reference.parameters.forEach { parameter ->
      recordSemanticType(parameter.type, parameter.canonicalType, parameter.typeInfo)
      // An exact completion item carries Sema's context-correct parameter spelling. Preserve that
      // spelling as a type-id candidate; graph-only callable signatures remain excluded so a
      // transitive header walk cannot flood every primary-template argument position.
      if (reference.completionVisible && isConcrete(parameter.typeInfo) &&
        parameter.typeInfo?.isSourceSpellable != false
      ) canonicalType(parameter.semanticType(), parameter.typeInfo)
        ?.let(templateArgumentTypes::add)
    }
    reference.templateParameters.forEach {
      recordSemanticType(it.type, it.canonicalType, it.typeInfo)
    }
  }

  private fun recordSemanticType(display: String?, canonical: String?, info: CppTypeInfo?) {
    if (!structuredTypes || info != null) recordType(display, canonical, info)
  }

  private fun recordWitnessSemanticType(display: String?, canonical: String?, info: CppTypeInfo?) {
    if (!structuredTypes || info != null)
      recordType(display, canonical, info, sourceSpellingEvidence = false)
  }

  private fun recordWitnessReferenceTypes(reference: CppReference) {
    recordWitnessSemanticType(reference.type, reference.canonicalType, reference.typeInfo)
    recordWitnessSemanticType(
      reference.returnType, reference.canonicalReturnType, reference.returnTypeInfo
    )
    recordWitnessSemanticType(reference.ownerType, reference.canonicalOwnerType, reference.ownerTypeInfo)
    reference.parameters.forEach { parameter ->
      recordWitnessSemanticType(parameter.type, parameter.canonicalType, parameter.typeInfo)
    }
  }

  private fun isConcrete(info: CppTypeInfo?): Boolean =
    info.isConcrete() && (!structuredTypes || info != null)

  private fun recordType(
    display: String?,
    canonical: String? = null,
    info: CppTypeInfo? = null,
    sourceSpellingEvidence: Boolean = true
  ) {
    if (info?.isDependent == true || info?.isInstantiationDependent == true) return
    val normalized = cppType(canonical ?: display) ?: return
    val canonicalId = info.semanticId()
    if (info != null && canonicalId == null) return
    val type = canonicalId?.let { id -> typeKeysByCanonicalId.getOrPut(id) {
      val key = if (normalized !in canonicalIdByTypeKey) normalized else "$normalized\u0000$id"
      canonicalIdByTypeKey[key] = id
      key
    } } ?: normalized
    if (info?.isDefaultConstructible == true && info.isComplete != false)
      semaDefaultConstructibleTypes += type
    if (normalized !in typeAliases) typeAliases[normalized] = type
    cppType(display)?.let { if (it !in typeAliases) typeAliases[it] = type }
    rawTypes += type
    if (info?.kind in setOf("builtin", "record", "enum", "pointer", "array"))
      objectTypes += type
    if (info?.kind == "pointer") {
      pointerInfos[type] = info
    }
    if (sourceSpellingEvidence && info?.isSourceSpellable != false &&
      !display.orEmpty().containsReservedCppIdentifier()
    )
      cppType(display)?.let { spelling ->
        // Keep the same first-writer identity policy as [typeAliases]. A later contradictory fact
        // cannot turn an already-known source spelling into evidence for another opaque type.
        // In schema-v1/structured mode, only Sema's affirmative bit may authenticate text that a
        // later witness will emit. Null remains a legacy spelling signal solely for unstructured
        // contexts; it is never promoted into the sealed witness trust map.
        if (typeAliases[spelling] == type) {
          if ((!structuredTypes || info?.isSourceSpellable == true) &&
            spelling !in sourceSpellableTypeAliases
          ) sourceSpellableTypeAliases[spelling] = type
          spellings.getOrPut(type, ::linkedSetOf) += spelling
          sourceSpellableTypes += type
        }
      }
  }

  private fun resolvePointerShapes() = pointerInfos.forEach { (type, info) ->
    val pointee = info.pointeeCanonicalId?.let(typeKeysByCanonicalId::get)
      // `void` has no declaration node to carry its opaque ID. The exact pointer spelling and
      // pointer metadata nevertheless prove this language builtin as the pointee.
      ?: type.rawPointee()?.let(::cppType)?.takeIf { it == "void" }?.let(::canonicalType)
    pointee?.let { resolved ->
      val shape = PointerShape(resolved, info.pointeeIsConst, info.pointeeIsVolatile)
      pointerTypes[shape] = type
      pointerShapes[type] = shape
    }
  }

  private fun recordPointerType(pointee: String, cv: Cv = Cv()) {
    val shape = PointerShape(pointee, cv.isConst, cv.isVolatile)
    val pointer = pointerTypes.getOrPut(shape) {
      val qualifiers = buildList {
        if (cv.isConst) add("const")
        if (cv.isVolatile) add("volatile")
      }.joinToString(" ").let { if (it.isEmpty()) "" else "$it " }
      "$qualifiers${pointee.typeShape()} *\u0000ptr:${canonicalIdByTypeKey[pointee] ?: pointee}:${cv.code}"
    }
    pointerShapes[pointer] = shape
    objectTypes += pointer
    rawTypes += pointer
    if (pointee in sourceSpellableTypes) {
      val qualifiers = buildList {
        if (cv.isConst) add("const")
        if (cv.isVolatile) add("volatile")
      }.joinToString(" ").let { if (it.isEmpty()) "" else "$it " }
      spellings.getOrPut(pointer, ::linkedSetOf) +=
        spellings.getValue(pointee).map { "$qualifiers$it *" }
      sourceSpellableTypes += pointer
    }
  }

  private fun deduplicate(references: MutableSet<CppReference>) {
    val unique = linkedMapOf<String, CppReference>()
    references.forEach { reference ->
      val key = buildString {
        // Distinct declarations that induce the same typed production are one CFG fact. Decl IDs
        // and index provenance do not change the accepted token language.
        append(reference.semanticName()); append('|')
        append(reference.denotesCallable()); append('|')
        append(reference.denotesMember()); append('|')
        append(reference.isStaticFact()); append('|')
        append(reference.isConstMember()); append('|')
        append(reference.isVolatileMember()); append('|')
        append(reference.refQualifier); append('|')
        append(reference.isMutableField); append('|')
        append(reference.isBitField); append('|')
        append(reference.activeCallable); append('|')
        append(reference.templateParameters.isNotEmpty()); append('|')
        append(reference.ownerTypeInfo.semanticId() ?: canonicalType(reference.ownerType)); append('|')
        append(reference.returnTypeInfo.semanticId() ?: reference.typeInfo.semanticId()
          ?: canonicalType(reference.semanticReturnType() ?: reference.semanticType())); append('|')
        append(reference.returnTypeInfo?.kind ?: reference.typeInfo?.kind); append('|')
        append(reference.returnTypeInfo?.isConst ?: reference.typeInfo?.isConst); append('|')
        append(reference.returnTypeInfo?.isVolatile ?: reference.typeInfo?.isVolatile); append('|')
        reference.parameters.forEach {
          append(it.typeInfo.semanticId() ?: canonicalType(it.semanticType())); append(':')
          append(it.typeInfo?.kind); append(':')
          append(it.typeInfo?.isConst); append(':')
          append(it.typeInfo?.isVolatile); append(':')
          append(it.isOptional()); append(':')
          append(it.isPack); append(';')
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
      // A variable template is not an expression until template arguments are supplied. Sema's
      // parameter list does not encode enough constraints to invent a sound specialization.
      if (reference.templateParameters.isNotEmpty() ||
        reference.kind.contains("varTemplate", ignoreCase = true) &&
        !reference.kind.contains("specialization", ignoreCase = true)
      ) return@forEach
      if (!isConcrete(reference.typeInfo)) return@forEach
      val type = canonicalType(reference.semanticType(), reference.typeInfo)
        ?.takeIf { it in typeSymbols } ?: return@forEach
      val name = reference.semanticName().grammarNameTokens()
      if (name.isEmpty()) return@forEach
      val cv = reference.objectCv()
      if (reference.denotesEnumConstant()) {
        // Enumerators are prvalues, including when an AST compatibility fact described them with
        // an ownerType. Publishing them as lvalues would incorrectly permit assignment to them.
        exactPostfixExpression(type, 0, ValueCategory.PRVALUE, Cv(), name)
      } else {
        exactPostfixExpression(type, 0, ValueCategory.LVALUE, cv, name)
      }
      val pointer = if (!isPointer(type) || cv == Cv())
        pointerTypes[PointerShape(type, cv.isConst, cv.isVolatile)]?.takeIf { it in typeSymbols }
      else null
      if (!reference.denotesEnumConstant() && pointer != null && type.typeShape() != "void") {
        movableStableExpression(
          pointer, 0, ExpressionPrecedence.UNARY, listOf("&") + name
        )
        movablePostfixExpression(pointer, 0, listOf("(", "&") + name + ")")
      }
    }
    functions.filter { it.isStaticFact() && !it.denotesCallable() }.forEach { reference ->
      val type = canonicalType(reference.semanticType(), reference.typeInfo)
        ?.takeIf { it in typeSymbols } ?: return@forEach
      postfixExpression(type, 0, reference.semanticName().grammarNameTokens())
    }
    canonicalType(context.thisType, context.thisTypeInfo)?.takeIf { it in typeSymbols }?.let { type ->
      movablePostfixExpression(type, 0, listOf("this"))
    }
    emptyAggregateTypes.filter { it in sourceSpellableTypes }.forEach { type ->
      movablePostfixExpression(type, 0, listOf(typeSpelling(type), "{", "}"))
    }
    expressionTypes.forEach { type ->
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

  private fun inheritExpressions(depth: Int) = expressionTypes.forEach { type ->
    val previousGeneric = genericState(type, depth - 1)
    val currentGeneric = genericState(type, depth)
    if (productiveExpressionOnlyStates[previousGeneric]) {
      production(expressionOnly(type, depth), expressionOnly(type, depth - 1))
      productiveExpressionOnlyStates[currentGeneric] = true
    }
    if (productivePostfixOnlyStates[previousGeneric]) {
      production(postfixOnly(type, depth), postfixOnly(type, depth - 1))
      productivePostfixOnlyStates[currentGeneric] = true
      markProductiveNativePrecedence(currentGeneric, ExpressionPrecedence.POSTFIX)
    }
    val previousNativeMask = productiveNativeTypeMasks[previousGeneric]
    nonPostfixPrecedences.forEach { precedence ->
      if (previousNativeMask and (1 shl precedence.ordinal) != 0) {
        production(
          nativePrecedence(type, depth, precedence),
          nativePrecedence(type, depth - 1, precedence)
        )
        markProductiveNativePrecedence(currentGeneric, precedence)
      }
    }
    cvVariants.forEach { cv -> ValueCategory.entries.forEach { category ->
      val previous = qualifiedState(type, depth - 1, category, cv)
      val current = qualifiedState(type, depth, category, cv)
      if (productivePostfixStates[previous]) {
        production(
          qualifiedPostfix(type, depth, category, cv),
          qualifiedPostfix(type, depth - 1, category, cv)
        )
        productivePostfixStates[current] = true
        exactPostfixTypeStates[currentGeneric] = true
        exactStableTypeStates[currentGeneric] = true
        markProductiveQualifiedPrecedence(current, ExpressionPrecedence.POSTFIX)
        markProductiveNativePrecedence(currentGeneric, ExpressionPrecedence.POSTFIX)
      }
      val previousPrecedenceMask = productiveNativePrecedenceMasks[previous]
      nonPostfixPrecedences.forEach { precedence ->
          if (previousPrecedenceMask and (1 shl precedence.ordinal) != 0) {
            production(
              qualifiedNativePrecedence(type, depth, category, cv, precedence),
              qualifiedNativePrecedence(type, depth - 1, category, cv, precedence)
            )
            markProductiveQualifiedPrecedence(current, precedence)
          }
        }
      if (productiveStableStates[previous]) {
        production(
          qualifiedStable(type, depth, category, cv),
          qualifiedStable(type, depth - 1, category, cv)
        )
        productiveStableStates[current] = true
        exactStableTypeStates[currentGeneric] = true
      }
    } }
  }

  /** Publish only exact states that have an atom, inherited witness, or composite production. */
  private fun addExpressionTierLinks(depth: Int) = expressionTypes.forEach { type ->
    val generic = genericState(type, depth)
    val hasExpressionOnly = productiveExpressionOnlyStates[generic]
    val hasPostfixOnly = productivePostfixOnlyStates[generic]
    val hasStable = hasPostfixOnly || exactStableTypeStates[generic]
    val hasPostfix = hasPostfixOnly || exactPostfixTypeStates[generic]
    if (hasExpressionOnly)
      production(expression(type, depth), expressionOnly(type, depth))
    if (hasStable)
      production(expression(type, depth), stable(type, depth))
    // Generic postfix-only forms carry no recoverable category/cv; exact postfix states flow
    // through their corresponding stable state below.
    if (hasPostfixOnly) {
      production(stable(type, depth), postfixOnly(type, depth))
      production(postfix(type, depth), postfixOnly(type, depth))
    }
    cvVariants.forEach { cv -> ValueCategory.entries.forEach { category ->
      val state = qualifiedState(type, depth, category, cv)
      if (!productiveStableStates[state]) return@forEach
      val postfixLeaf = qualifiedPostfix(type, depth, category, cv)
      val stableLeaf = qualifiedStable(type, depth, category, cv)
      // A stable expression includes every postfix expression with the same exact category/cv.
      // Reusing that state as the general qualified expression removes an equivalent union node
      // for every type/depth/state without changing the represented expression language.
      if (productivePostfixStates[state]) {
        production(stableLeaf, postfixLeaf)
        production(postfix(type, depth), postfixLeaf)
      }
      production(stable(type, depth), stableLeaf)
      when (category) {
        ValueCategory.LVALUE -> {
          production(glvalue(type, depth), stableLeaf)
          if (!cv.isConst) production(lvalue(type, depth), stableLeaf)
          if (!cv.isConst && !cv.isVolatile && productivePostfixStates[state])
            production(mutablePostfix(type, depth), postfixLeaf)
        }
        ValueCategory.XVALUE, ValueCategory.PRVALUE -> {
          production(rvalue(type, depth), stableLeaf)
          if (!cv.isConst && !cv.isVolatile) {
            production(movable(type, depth), stableLeaf)
            if (productivePostfixStates[state])
              production(mutablePostfix(type, depth), postfixLeaf)
          }
        }
      }
    } }
    if (hasExpressionOnly || hasStable) productiveExpressionStates[generic] = true
    if (hasStable) productiveStableTypeStates[generic] = true
    if (hasPostfix) productivePostfixTypeStates[generic] = true
  }

  private fun addLanguageExpressions(depth: Int) {
    val previous = depth - 1
    expressionTypes.forEach { type ->
      if (hasExpression(type, previous))
        postfixExpression(type, depth, listOf("(", expression(type, previous), ")"))
      cvVariants.forEach { cv -> ValueCategory.entries.forEach { category ->
        if (!productiveStableStates[qualifiedState(type, previous, category, cv)])
          return@forEach
        exactPostfixExpression(
          type, depth, category, cv,
          listOf("(", qualified(type, previous, category, cv), ")")
        )
      } }
      if (type.isNumericCppType() &&
        hasPrecedenceExpression(type, previous, ExpressionPrecedence.UNARY)) {
        val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: type
        listOf("+", "-").forEach { operator ->
          movableStableExpression(
            result, depth, ExpressionPrecedence.UNARY,
            listOf(operator, precedenceExpression(type, previous, ExpressionPrecedence.UNARY))
          )
        }
      }
      if (type.isIntegralCppType() &&
        hasPrecedenceExpression(type, previous, ExpressionPrecedence.UNARY)) {
        val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: type
        movableStableExpression(
          result, depth, ExpressionPrecedence.UNARY,
          listOf("~", precedenceExpression(type, previous, ExpressionPrecedence.UNARY))
        )
      }
      if (isPointer(type) &&
        hasPrecedenceExpression(type, previous, ExpressionPrecedence.UNARY)) {
        val pointer = pointerShapes[type]
        val rawPointee = type.rawPointee() ?: return@forEach
        val pointee = pointer?.pointee
          ?: canonicalType(rawPointee)?.takeIf { it in typeSymbols }
          ?: return@forEach
        // Unary `*` requires a pointer to object or function; `void` is neither.
        if (pointee.typeShape() == "void") return@forEach
        val rhs = listOf(
          "*", precedenceExpression(type, previous, ExpressionPrecedence.UNARY)
        )
        val cv = pointer?.let { Cv(it.isConst, it.isVolatile) }
          ?: Cv(
            isConst = rawPointee.startsWith("const "),
            isVolatile = rawPointee.startsWith("volatile ")
          )
        exactStableExpression(
          pointee, depth, ValueCategory.LVALUE, cv, ExpressionPrecedence.UNARY, rhs
        )
      }
    }

    val numeric = expressionTypes.filter { it.isArithmeticCppType() || it in enumTypes }
    numeric.filter { it in sourceSpellableTypes }.forEach { target -> numeric.forEach { source ->
      if (!hasExpression(source, previous)) return@forEach
      movablePostfixExpression(
        target, depth,
        listOf("static_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
      )
    } }
    addPointerStaticCasts(depth)
    addConstCasts(depth)
  }

  /**
   * A pointer standard conversion is valid for a Sema-proven direct public base edge, or from a
   * pointer to a proven object type to (cv) `void`. Both forms may add, but never remove, pointee
   * qualification. User-defined object conversions deliberately do not lift to pointer types.
   */
  private fun addPointerStaticCasts(depth: Int) {
    val previous = depth - 1
    val pointers = expressionTypes.mapNotNull { type ->
      pointerShapes[type]?.let { shape -> type to shape }
    }
    pointers.filter { (target, _) -> target in sourceSpellableTypes }.forEach { (target, to) ->
      pointers.forEach { (source, from) ->
        if (source != target && hasExpression(source, previous) &&
          isImplicitPointerConversion(from, to))
          movablePostfixExpression(
            target, depth,
            listOf("static_cast", "<", typeSpelling(target), ">", "(",
              expression(source, previous), ")")
          )
      }
    }
  }

  /** `const_cast` changes only cv qualification of an exact Sema object type. */
  private fun addConstCasts(depth: Int) {
    val previous = depth - 1
    val pointers = expressionTypes.filter(::isPointer)
    pointers.forEach { source -> pointers.forEach { target ->
      val from = pointerShapes[source] ?: return@forEach
      val to = pointerShapes[target] ?: return@forEach
      if (target !in sourceSpellableTypes) return@forEach
      if (from.pointee == to.pointee &&
        (from.isConst != to.isConst || from.isVolatile != to.isVolatile) &&
        hasExpression(source, previous)) {
        movablePostfixExpression(
          target, depth,
          listOf("const_cast", "<", typeSpelling(target), ">", "(", expression(source, previous), ")")
        )
      }
    } }

    sourceSpellableTypes.filter { it.typeShape() != "void" && !isPointer(it) }.forEach { type ->
      if (productiveStableStates[qualifiedState(
          type, previous, ValueCategory.LVALUE, Cv(isConst = true)
        )]) {
        exactPostfixExpression(
          type, depth, ValueCategory.LVALUE, Cv(),
          listOf("const_cast", "<", typeSpelling(type), "&", ">", "(",
            qualified(type, previous, ValueCategory.LVALUE, Cv(isConst = true)), ")")
        )
      }
      if (productiveStableStates[qualifiedState(
          type, previous, ValueCategory.LVALUE, Cv()
        )]) {
        exactPostfixExpression(
          type, depth, ValueCategory.LVALUE, Cv(isConst = true),
          listOf("const_cast", "<", "const", typeSpelling(type), "&", ">", "(",
            qualified(type, previous, ValueCategory.LVALUE, Cv()), ")")
        )
      }
    }

  }

  /**
   * Lowers a successful Sema expression build as one correlated RHS. A target, operand, or result
   * from another witness is never placed behind a shared choice symbol, so the CFG cannot invent a
   * cast pair that the compiler did not build.
   */
  private fun addExpressionWitnesses(depth: Int) {
    authoritativeExpressionWitnesses.forEach { witness ->
      val result = exactExpressionWitnessProfileType(witness.result) ?: return@forEach
      if (witness.result.kind != "opaque") return@forEach
      val resultCategory = exactProfileCategory(witness.result) ?: return@forEach
      val resultInfo = witness.result.typeInfo ?: return@forEach

      val rhs = when (witness.syntax) {
        "dynamicCast", "reinterpretCast" -> {
          val target = witness.typeOperand ?: return@forEach
          val operand = witness.expressionOperand ?: return@forEach
          val (targetType, targetSpelling) = exactTypeOperand(target) ?: return@forEach
          if (!expressionWitnessResultMatchesTarget(
              target, targetType, witness.result, result, resultCategory
            ) || exactExpressionWitnessProfileType(operand) == null
          ) return@forEach
          val operandExpression = exactProfileExpression(operand, depth - 1) ?: return@forEach
          val operator = if (witness.syntax == "dynamicCast")
            "dynamic_cast" else "reinterpret_cast"
          listOf(operator, "<") + targetSpelling + ">" + "(" + operandExpression + ")"
        }
        "typeidExpression" -> {
          if (witness.typeOperand != null || !isExactTypeidResult(
              witness.result, resultCategory
            )) return@forEach
          val operand = witness.expressionOperand ?: return@forEach
          if (exactExpressionWitnessProfileType(operand) == null) return@forEach
          val operandExpression = exactProfileExpression(operand, depth - 1) ?: return@forEach
          listOf("typeid", "(", operandExpression, ")")
        }
        "typeidType" -> {
          if (witness.expressionOperand != null || !isExactTypeidResult(
              witness.result, resultCategory
            )) return@forEach
          val target = witness.typeOperand ?: return@forEach
          val (_, targetSpelling) = exactTypeOperand(target) ?: return@forEach
          listOf("typeid", "(") + targetSpelling + ")"
        }
        else -> return@forEach
      }
      exactPostfixExpression(
        result, depth, resultCategory, Cv(resultInfo.isConst, resultInfo.isVolatile), rhs
      )
    }
  }

  /** A witness value needs complete opaque provenance, but its static type need not be spellable. */
  private fun exactExpressionWitnessProfileType(profile: CppExpressionProfile): String? {
    val info = profile.typeInfo?.takeIf(CppTypeInfo::isUsableExpressionWitnessValue) ?: return null
    return exactProfileType(profile)?.takeIf { it in expressionTypes && info.semanticId() != null }
  }

  /**
   * The endpoint fixes the exact canonical type and declarator shape. Its display text need not be
   * the only spelling accepted at the cursor: an independently indexed alias for the same base
   * type may replace only the parsed base-name slice. The surrounding cv/ref/pointer/array
   * declarator remains witness-exact, and the witness is never allowed to authenticate its own
   * alternate alias.
   */
  private fun exactTypeOperand(profile: CppTypeProfile): Pair<String, List<String>>? {
    val (type, exactTerminals) =
      exactWitnessTypeSpelling(profile, requireCompleteObject = true) ?: return null
    val parsed = CppExactTypeIdParser(profile.type).parse() ?: return null
    val spellings = linkedSetOf(exactTerminals)

    val baseSpelling = parsed.baseSourceSpelling
    val baseType = cppType(baseSpelling)?.let(preWitnessSourceSpellableTypeAliases::get)
    if (baseType != null) {
      this.spellings[baseType].orEmpty().forEach { candidate ->
        val candidateType = CppExactTypeIdParser(candidate).parse() ?: return@forEach
        // A replacement is a base name, not another abstract declarator or a cv-qualified type.
        if (candidateType.kind != CppExactTypeIdKind.VALUE || candidateType.isConst ||
          candidateType.isVolatile
        ) return@forEach
        candidate.typeSpellingVariants().forEach { replacement ->
          spellings += parsed.terminals.take(parsed.baseStartTerminal) + replacement +
            parsed.terminals.drop(parsed.baseEndTerminal)
        }
      }
    }

    if (spellings.size == 1) return type to exactTerminals
    val alternatives = spellings.toList()
    val symbol = exactTypeOperandSpellingSymbols.getOrPut(alternatives) {
      "EXACT_TYPE_OPERAND_SPELLING_${exactTypeOperandSpellingSymbols.size}".also { choice ->
        alternatives.forEach { production(choice, it) }
      }
    }
    return type to listOf(symbol)
  }

  /** Function-template explicit arguments are types, but may legitimately be incomplete. */
  private fun exactExplicitTypeArgument(profile: CppTypeProfile): List<String>? {
    if (profile.typeInfo.kind !in setOf(
        "builtin", "pointer", "record", "enum", "array", "function", "other",
        "lvalueReference", "rvalueReference"
      )) return null
    return exactWitnessTypeSpelling(profile, requireCompleteObject = false)?.second
  }

  /** Preserves the selected specialization's heterogeneous explicit-argument order exactly. */
  private fun exactExplicitTemplateArgument(
    profile: CppTemplateArgumentProfile
  ): List<String>? {
    if (!profile.isWellFormedCppTemplateArgument()) return null
    return when (profile.kind) {
      "type" -> exactExplicitTypeArgument(profile.type)
      "exactIntegerLiteral" -> {
        val info = profile.type.typeInfo
        canonicalType(profile.type.canonicalType ?: profile.type.type, info)
          ?.takeIf { it in typeSymbols && it.isIntegralOrBooleanCppType() } ?: return null
        listOf(cppExactIntegerTerminal(requireNotNull(profile.spelling)))
      }
      else -> null
    }
  }

  private fun exactWitnessTypeSpelling(
    profile: CppTypeProfile,
    requireCompleteObject: Boolean
  ): Pair<String, List<String>>? {
    val info = profile.typeInfo
    if (!info.isConcrete() || info.isSourceSpellable != true ||
      requireCompleteObject && !info.isUsableExpressionWitnessTypeId()
    ) return null
    val semanticId = info.semanticId() ?: return null
    val spelling = profile.type.trim().takeIf(String::isNotEmpty) ?: return null
    val normalizedSpelling = cppType(spelling) ?: return null
    val parsed = CppExactTypeIdParser(spelling).parse() ?: return null
    val expectedKind = when (info.kind) {
      "lvalueReference" -> CppExactTypeIdKind.LVALUE_REFERENCE
      "rvalueReference" -> CppExactTypeIdKind.RVALUE_REFERENCE
      "pointer" -> CppExactTypeIdKind.POINTER
      "array", "constantArray", "incompleteArray", "variableArray" -> CppExactTypeIdKind.ARRAY
      "builtin", "record", "enum" -> CppExactTypeIdKind.VALUE
      else -> return null
    }
    if (parsed.kind != expectedKind || parsed.isConst != info.isConst ||
      parsed.isVolatile != info.isVolatile
    ) return null

    val canonicalSpelling = (profile.canonicalType ?: profile.type).trim()
    val normalizedCanonical = cppType(canonicalSpelling) ?: return null
    // Canonical metadata may legitimately use an implementation namespace; it is checked but
    // never emitted. Reserved-name rejection therefore applies only to the public display text.
    val parsedCanonical = CppExactTypeIdParser(
      canonicalSpelling, allowReservedIdentifiers = true
    ).parse() ?: return null
    if (parsedCanonical.kind != expectedKind || parsedCanonical.isConst != info.isConst ||
      parsedCanonical.isVolatile != info.isVolatile
    ) return null

    // Resolve both strings through evidence sealed before any witness profile was registered. The
    // display spelling must itself have been independently marked source-spellable; a canonical
    // normalization alias alone is not permission to emit that text.
    val independentlyKnownType = preWitnessTypeKeysByCanonicalId[semanticId]
    val directProof = expectedKind != CppExactTypeIdKind.ARRAY && independentlyKnownType != null &&
      preWitnessTypeAliases[normalizedCanonical] == independentlyKnownType &&
      preWitnessSourceSpellableTypeAliases[normalizedSpelling] == independentlyKnownType

    // A simple pointer spelling may be derived without an independently indexed pointer node:
    // structured pointer metadata binds it to an independently source-spellable pointee identity.
    // This retains `T *`/cv-pointer witnesses while a claimed opaque pointer ID alone proves
    // nothing. Array derivation below follows the same sealed-identity rule with its own shape.
    val derivedPointerProof = expectedKind == CppExactTypeIdKind.POINTER &&
      info.pointeeCanonicalId?.let(preWitnessTypeKeysByCanonicalId::get)?.let { pointee ->
        fun provesPointee(normalized: String, aliases: Map<String, String>): Boolean =
          normalized.rawPointee()?.let(::cppType)?.let(aliases::get) == pointee
        parsed.baseIsConst == info.pointeeIsConst &&
          parsed.baseIsVolatile == info.pointeeIsVolatile &&
          parsedCanonical.baseIsConst == info.pointeeIsConst &&
          parsedCanonical.baseIsVolatile == info.pointeeIsVolatile &&
          provesPointee(normalizedSpelling, preWitnessSourceSpellableTypeAliases) &&
          provesPointee(normalizedCanonical, preWitnessTypeAliases)
      } == true

    // An array witness never authenticates its own opaque array identity. Its immediate element
    // must already have an independently sealed identity and spelling, while Sema's array metadata
    // must agree exactly with both the public and canonical declarator shapes.
    val derivedArrayProof = expectedKind == CppExactTypeIdKind.ARRAY &&
      info.elementCanonicalId?.let(preWitnessTypeKeysByCanonicalId::get)?.let { element ->
        info.isIncompleteArray?.let { incomplete ->
          val bound = info.arrayBound
          val metadataShapeIsExact = if (incomplete) bound == null
          else bound?.matches(Regex("[1-9][0-9]*")) == true
          fun provesArray(
            exact: CppExactTypeId,
            aliases: Map<String, String>
          ): Boolean = exact.arrayElementType?.let(aliases::get) == element &&
            exact.isConst == info.elementIsConst &&
            exact.isVolatile == info.elementIsVolatile &&
            exact.isIncompleteArray == incomplete &&
            (if (incomplete) exact.arrayBound == null
            else exact.arrayBound?.let(::canonicalCppIntegerLiteralValue) == bound)
          metadataShapeIsExact && provesArray(parsed, preWitnessSourceSpellableTypeAliases) &&
            provesArray(parsedCanonical, preWitnessTypeAliases)
        }
      } == true

    if (!directProof && !derivedPointerProof && !derivedArrayProof) return null
    val type = (independentlyKnownType.takeIf { directProof } ?:
      canonicalType(profile.canonicalType ?: profile.type, info))
      ?.takeIf {
        it in typeSymbols && (it in sourceSpellableTypes || derivedPointerProof || derivedArrayProof)
      }
      ?: return null
    return type to parsed.terminals
  }

  /** A named cast's value type/category is fixed by its exact target type-id. */
  private fun expressionWitnessResultMatchesTarget(
    target: CppTypeProfile,
    targetType: String,
    result: CppExpressionProfile,
    resultType: String,
    resultCategory: ValueCategory
  ): Boolean {
    if (targetType != resultType) return false
    val expectedCategory = when (target.typeInfo.kind) {
      "lvalueReference" -> ValueCategory.LVALUE
      "rvalueReference" -> ValueCategory.XVALUE
      else -> ValueCategory.PRVALUE
    }
    if (resultCategory != expectedCategory) return false
    val resultInfo = result.typeInfo ?: return false
    return target.typeInfo.kind !in setOf("lvalueReference", "rvalueReference") ||
      target.typeInfo.isConst == resultInfo.isConst &&
      target.typeInfo.isVolatile == resultInfo.isVolatile
  }

  /** Both standard typeid forms produce a nonvolatile const lvalue. */
  private fun isExactTypeidResult(
    result: CppExpressionProfile,
    category: ValueCategory
  ): Boolean = result.typeInfo?.let { info ->
    category == ValueCategory.LVALUE && info.isConst && !info.isVolatile
  } == true

  private fun addCalls(depth: Int) {
    freeCallWork.forEach { work ->
      val callable = work.callable
      val name = work.sourceName.grammarNameTokens()
      if (work.concreteResult != null) {
        addCallProductions(
          work.concreteResult, depth, name, callable.parameters,
          callable.semanticReturnType(), callable.returnTypeInfo, callable.activeCallable
        )
      } else if (callable.activeCallable) {
        contextualActiveResults.forEach { result ->
          addCallProductions(result, depth, name, callable.parameters, null, null, active = true)
        }
      }
    }

    constructorsBySemanticType.forEach { (type, overloads) ->
        if (type == null || type !in typeSymbols || type in abstractTypes ||
          type !in sourceSpellableTypes) return@forEach
        overloads.forEach { constructor ->
          val argumentLists = if (constructor.activeCallable)
            activeArgumentLists(constructor.parameters, depth - 1)
          else factoredArgumentLists(constructor.parameters, depth - 1)
          argumentLists.forEach { arguments ->
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

  /**
   * Lowers compiler-validated calls as a finite relation of whole argument vectors.
   * Each RHS is assembled from exactly one [CppCallWitness]; there is intentionally no symbol
   * whose alternatives are the union of one argument position across several witnesses.
   */
  private fun addCallWitnesses(depth: Int) {
    authoritativeCallWitnesses.forEach { witness ->
      val targetKind = witness.validatedTargetKind() ?: return@forEach
      val explicitTemplateArguments = witness.orderedExplicitTemplateArguments()
        ?: return@forEach
      val result = exactProfileType(witness.result) ?: return@forEach
      if (witness.result.kind != "opaque" ||
        !witnessResultMatchesCallable(witness, result)) return@forEach
      val resultCategory = exactProfileCategory(witness.result) ?: return@forEach
      val resultCv = witness.result.typeInfo?.let { Cv(it.isConst, it.isVolatile) }
        ?: return@forEach
      val arguments = exactWitnessArguments(witness.arguments, depth - 1) ?: return@forEach

      val rhs = when (witness.syntax) {
        "memberCall" -> {
          if (targetKind != CppCallTargetKind.FUNCTION_TEMPLATE) return@forEach
          if (explicitTemplateArguments.isNotEmpty()) return@forEach
          val receiver = witness.receiver ?: return@forEach
          val receiverType = exactProfileType(receiver) ?: return@forEach
          val owner = canonicalType(
            witness.callable.canonicalOwnerType ?: witness.callable.ownerType,
            witness.callable.ownerTypeInfo
          ) ?: return@forEach
          if (receiverType != owner) return@forEach
          val receiverExpression = exactProfileExpression(
            receiver, depth - 1, requirePostfix = true
          ) ?: return@forEach
          val name = witness.name.grammarNameTokens()
          if (name.isEmpty()) return@forEach
          listOf(receiverExpression, ".") + name + "(" + arguments + ")"
        }
        "parenConstruction", "listConstruction" -> {
          if (targetKind != CppCallTargetKind.FUNCTION_TEMPLATE) return@forEach
          if (witness.receiver != null || explicitTemplateArguments.isNotEmpty())
            return@forEach
          val owner = canonicalType(
            witness.callable.canonicalOwnerType ?: witness.callable.ownerType,
            witness.callable.ownerTypeInfo
          ) ?: return@forEach
          if (owner != result || owner !in sourceSpellableTypes || owner in abstractTypes)
            return@forEach
          val open = if (witness.syntax == "listConstruction") "{" else "("
          val close = if (witness.syntax == "listConstruction") "}" else ")"
          listOf(typeSpelling(owner), open) + arguments + close
        }
        "freeCall" -> {
          if (witness.receiver != null ||
            !witness.callable.denotesCallable() || witness.callable.denotesMember() ||
            witness.callable.denotesConstructor()
          ) return@forEach
          val name = exactFreeCallName(witness.name) ?: return@forEach
          val explicitArguments = mutableListOf<String>()
          explicitTemplateArguments.forEachIndexed { index, profile ->
            val spelling = exactExplicitTemplateArgument(profile) ?: return@forEach
            if (index > 0) explicitArguments += ","
            explicitArguments += spelling
          }
          val templateId = if (explicitArguments.isEmpty()) name
          else name + "<" + explicitArguments + ">"
          templateId + "(" + arguments + ")"
        }
        else -> return@forEach
      }
      exactPostfixExpression(result, depth, resultCategory, resultCv, rhs)
    }
  }

  /** The source call path is carried only by the witness, never reconstructed from identity. */
  private fun exactFreeCallName(raw: String): List<String>? {
    if (raw.isBlank() || raw != raw.trim() || raw.containsReservedCppIdentifier() ||
      !raw.isCppQualifiedName()
    ) return null
    if (raw.removePrefix("::").split("::").any { it in CPP_KEYWORDS }) return null
    return raw.grammarNameTokens().takeIf(List<String>::isNotEmpty)
  }

  /** A selected specialization's result identity must agree with the probed call expression. */
  private fun witnessResultMatchesCallable(witness: CppCallWitness, result: String): Boolean {
    if (witness.result.kind != "opaque") return false
    val callableInfo = witness.callable.returnTypeInfo ?: return false
    val resultInfo = witness.result.typeInfo ?: return false
    if (callableInfo.semanticId() == null || callableInfo.semanticId() != resultInfo.semanticId() ||
      callableInfo.kind !in setOf("lvalueReference", "rvalueReference") &&
        callableInfo.kind != resultInfo.kind ||
      callableInfo.isConst != resultInfo.isConst ||
      callableInfo.isVolatile != resultInfo.isVolatile ||
      canonicalType(witness.callable.semanticReturnType(), callableInfo) != result
    )
      return false
    val category = exactProfileCategory(witness.result) ?: return false
    val callableCategory = when (callableInfo.kind) {
      "lvalueReference" -> ValueCategory.LVALUE
      "rvalueReference" -> ValueCategory.XVALUE
      else -> ValueCategory.PRVALUE
    }
    return category == callableCategory
  }

  /** Maps a profile through its opaque Sema value identity, never through display spelling alone. */
  private fun exactProfileType(profile: CppExpressionProfile): String? {
    if (!profile.isWellFormedCppExpressionProfile()) return null
    val info = profile.typeInfo?.takeIf(CppTypeInfo::isConcrete) ?: return null
    return canonicalType(profile.canonicalType ?: profile.type, info)
      ?.takeIf { it in typeSymbols }
  }

  private fun exactProfileCategory(profile: CppExpressionProfile): ValueCategory? =
    when (profile.valueCategory) {
      "lvalue" -> ValueCategory.LVALUE
      "xvalue" -> ValueCategory.XVALUE
      "prvalue" -> ValueCategory.PRVALUE
      else -> null
    }

  /** Synthetic probe literals retain their exact compiler spelling; opaque profiles select a
   * typed, cv-qualified expression with the exact Clang value category. */
  private fun exactProfileExpression(
    profile: CppExpressionProfile,
    depth: Int,
    requirePostfix: Boolean = false
  ): String? {
    // Even literal profiles must carry a resolvable semantic ID: the terminal alone is not enough
    // to prove that this is the type/category vector clang instantiated.
    val type = exactProfileType(profile) ?: return null
    val category = exactProfileCategory(profile) ?: return null
    if (profile.kind != "opaque") {
      if (requirePostfix) return null
      val literal = CPP_SYNTHETIC_LITERAL_PROFILES[profile.kind] ?: return null
      return if (literal.tokenKind == CppTokenKind.NULLPTR) CPP_NULLPTR
      else cppExactLiteralTerminal(literal.tokenKind, literal.spelling)
    }
    val stringLiteralPointer = pointerShapes[type]?.let { shape ->
      shape.pointee.typeShape() == "char" && shape.isConst
    } == true
    // Generic prvalue states deliberately abstract literal tokens. For these types that erases
    // exact static type (suffixes/promotions) and special conversion facts such as integer zero's
    // null-pointer conversion or string-literal array decay. Only tagged synthetic literals may
    // authenticate such a witness until a literal-free state is split out.
    if (category == ValueCategory.PRVALUE &&
      (type.isArithmeticCppType() || stringLiteralPointer)
    ) return null
    if (category != ValueCategory.PRVALUE && type in witnessObjectKindTaintedTypes)
      return null
    val cv = profile.typeInfo?.let { Cv(it.isConst, it.isVolatile) } ?: return null
    val state = qualifiedState(type, depth, category, cv)
    return if (requirePostfix) {
      if (productivePostfixStates[state]) qualifiedPostfix(type, depth, category, cv) else null
    } else {
      if (productiveStableStates[state]) qualified(type, depth, category, cv) else null
    }
  }

  private fun exactWitnessArguments(
    profiles: List<CppExpressionProfile>,
    depth: Int
  ): List<String>? {
    val arguments = mutableListOf<String>()
    profiles.forEachIndexed { index, profile ->
      val expression = exactProfileExpression(profile, depth) ?: return null
      if (index > 0) arguments += ","
      arguments += expression
    }
    return arguments
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
      val receivers = receiversFor(owner, member).filter { (receiverType, _) ->
        hasPostfixExpression(receiverType, depth - 1)
      }

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
        val arguments = concreteArgumentAlternatives(parameter, depth - 1)
        receivers.forEach { (receiverType, connector) -> arguments.forEach { argument ->
          val receiver = receiverHead(receiverType, connector).first()
          val base = if (connector == ".") listOf(receiver) else listOf("(", "*", receiver, ")")
          val rhs = base + "[" + argument + "]"
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
      val implicitOwnerMatches = thisPointee == owner ||
        thisPointee != null && thisPointee to owner in baseConversions
      if (implicitOwnerMatches &&
        (member.isStaticFact() || member.refQualifier != "&&") &&
        (!member.denotesCallable() || member.isStaticFact() || member.acceptsCv(Cv(
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
            if (!productivePostfixStates[qualifiedState(
                receiverType, depth - 1, category, baseCv
              )]) return@forEach
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
    returnTypeInfo: CppTypeInfo? = null,
    active: Boolean = false
  ) {
    if (active) {
      activeArgumentLists(parameters, depth - 1).forEach { arguments ->
        emitPostfixResult(result, depth, head + "(" + arguments + ")", rawReturnType, returnTypeInfo)
      }
      return
    }
    factoredArgumentLists(parameters, depth - 1).forEach { arguments ->
      val rhs = head + "(" + arguments + ")"
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
    info?.kind == "rvalueReference" -> exactPostfixExpression(
      result, depth, ValueCategory.XVALUE, Cv(info.isConst, info.isVolatile), rhs
    )
    info != null -> exactPostfixExpression(
      result, depth, ValueCategory.PRVALUE, Cv(info.isConst, info.isVolatile), rhs
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
    val referenceMember = field.typeInfo?.kind in setOf("lvalueReference", "rvalueReference")
    // Static members and reference members denote an object independently of the receiver's cv.
    // A named reference data member is always an lvalue, even through an rvalue owner.
    val receiverIndependent = field.isStaticFact() || referenceMember
    val cv = if (receiverIndependent) declared else Cv(
      declared.isConst || baseCv.isConst && field.isMutableField != true,
      declared.isVolatile || baseCv.isVolatile
    )
    val resultCategory = when {
      receiverIndependent -> ValueCategory.LVALUE
      category == ValueCategory.LVALUE -> ValueCategory.LVALUE
      else -> ValueCategory.XVALUE
    }
    exactPostfixExpression(result, depth, resultCategory, cv, rhs)
  }

  /** Exact positional argument products represented as a shared comma-chain CFG. */
  private fun factoredArgumentLists(
    parameters: List<CppParameter>,
    depth: Int
  ): List<List<String>> {
    // clang must expand dependent packs before they can be represented as concrete CFG edges.
    val shapes = parameters.map { parameter ->
      if (parameter.isPack || !isConcrete(parameter.typeInfo)) return emptyList()
      val type = canonicalType(parameter.semanticType(), parameter.typeInfo)
        ?.takeIf { it in typeSymbols } ?: return emptyList()
      val spelling = parameter.type.ifBlank { parameter.semanticType() }
      val binding = parameter.typeInfo?.kind ?: when {
        spelling.isConstLvalueReferenceType() -> "constLvalueReference"
        spelling.isLvalueReferenceType() -> "lvalueReference"
        spelling.isConstRvalueReferenceType() -> "constRvalueReference"
        spelling.trim().endsWith("&&") -> "rvalueReference"
        else -> "value"
      }
      ArgumentShape(
        type = type,
        binding = binding,
        isConst = parameter.typeInfo?.isConst == true,
        isVolatile = parameter.typeInfo?.isVolatile == true,
        optional = parameter.isOptional(),
        pack = parameter.isPack
      )
    }
    return argumentLists.getOrPut(shapes to depth) {
    val required = shapes.indexOfFirst(ArgumentShape::optional)
      .let { if (it < 0) shapes.size else it }
    // Determine the usable arity before publishing any argument symbols. When a later required
    // parameter has no exact expression source, the call is impossible; emitting all preceding
    // position alternatives only creates a large unreachable subgrammar for the final pruner.
    val concreteAlternatives = parameters.mapIndexed { index, parameter ->
      val key = shapes[index].copy(optional = false) to depth
      argumentAlternatives.getOrPut(key) {
        concreteArgumentAlternatives(parameter, depth)
      }
    }
    val maximum = concreteAlternatives.indexOfFirst { it.isEmpty() }
      .let { if (it < 0) concreteAlternatives.size else it }
    if (required > maximum) return@getOrPut emptyList()
    val ordinal = argumentLists.size
    val positions = shapes.take(maximum).mapIndexed { index, shape ->
      val bindingShape = shape.copy(optional = false)
      argumentSymbols.getOrPut(bindingShape to depth) {
        "ARGUMENT_${argumentSymbols.size}_D$depth".also { symbol ->
          concreteAlternatives[index].forEach { production(symbol, it) }
        }
      }
    }
    buildList {
      if (required == 0) add(emptyList())
      if (maximum == 0) return@buildList
      val alternatives = "ARGUMENT_LIST_${ordinal}_D$depth"
      var chain = "${alternatives}_1"
      production(chain, positions[0])
      if (required <= 1) production(alternatives, chain)
      for (arity in 2..maximum) {
        val next = "${alternatives}_$arity"
        production(next, chain, ",", positions[arity - 1])
        chain = next
        if (arity >= required) production(alternatives, chain)
      }
      add(listOf(alternatives))
    }
  }
  }

  /**
   * Active template candidates are the one place where clang intentionally leaves parameter
   * types dependent. Keep the ordinary concrete path exact, and use only Sema-ranked value and
   * empty-aggregate atoms for dependent slots. A shared symbol avoids materializing their
   * Cartesian product; the pack chain stays finite at the statement grammar's existing horizon.
   */
  private fun activeArgumentLists(parameters: List<CppParameter>, depth: Int): List<List<String>> {
    val pack = parameters.indexOfFirst(CppParameter::isPack)
    if (pack >= 0 && (pack != parameters.lastIndex || parameters.count(CppParameter::isPack) != 1))
      return emptyList()
    val fixed = if (pack < 0) parameters else parameters.take(pack)
    val fixedMode = if (pack >= 0 && emptyAggregateTypes.isNotEmpty())
      ActiveArgumentMode.AGGREGATE else ActiveArgumentMode.ALL
    val symbols = fixed.map { activeArgumentSymbol(it, depth, fixedMode) ?: return emptyList() }
    val required = fixed.indexOfFirst(CppParameter::isOptional)
      .let { if (it < 0) fixed.size else it }
    val choices = (required..fixed.size).map { joinArguments(symbols.take(it)) }.toMutableList()
    if (pack >= 0 && symbols.size == fixed.size) {
      val parameter = parameters[pack]
      val maximum = CPP_MAX_STATEMENT_TOKENS / 2
      activePackSymbol(parameter, depth, maximum)?.let { tail ->
        choices += joinArguments(symbols) + if (symbols.isEmpty()) listOf(tail) else listOf(",", tail)
      }
    }
    return choices.distinct()
  }

  private fun activeArgumentSymbol(
    parameter: CppParameter,
    depth: Int,
    mode: ActiveArgumentMode = ActiveArgumentMode.ALL
  ): String? {
    val key = Triple(parameter, depth, mode)
    activeArgumentSymbols[key]?.let { return it }
    val expected = canonicalType(parameter.semanticType(), parameter.typeInfo)
    val concrete = isConcrete(parameter.typeInfo) && expected in typeSymbols
    val alternatives = if (concrete)
      concreteArgumentAlternatives(parameter, depth)
    else {
      val rankedValues = values.asSequence().map { it.semanticName().cachedNameTokens() }
        .filter(List<String>::isNotEmpty).distinct()
        .take((CPP_MAX_INTERACTIVE_COMPLETIONS - 1).coerceAtLeast(1)).toList()
      val aggregates = emptyAggregateTypes.asSequence().filter { it in sourceSpellableTypes }
        .map { listOf(typeSpelling(it), "{", "}") }.take(1).toList()
      when (mode) {
        ActiveArgumentMode.AGGREGATE -> aggregates
        ActiveArgumentMode.VALUE -> rankedValues
        ActiveArgumentMode.ALL -> aggregates + rankedValues
      }
    }
    if (alternatives.isEmpty()) return null
    val symbol = "ACTIVE_ARGUMENT_${activeArgumentSymbols.size}_D$depth"
    activeArgumentSymbols[key] = symbol
    alternatives.forEach { production(symbol, it) }
    return symbol
  }

  private fun activePackSymbol(parameter: CppParameter, depth: Int, maximum: Int): String? {
    if (maximum < 1) return null
    val key = Triple(parameter, depth, maximum)
    activePackSymbols[key]?.let { return it }
    val argument = activeArgumentSymbol(parameter, depth, ActiveArgumentMode.VALUE) ?: return null
    val choice = "ACTIVE_PACK_${activePackSymbols.size}_D$depth"
    var previous = "${choice}_1"
    production(previous, argument)
    production(choice, previous)
    for (size in 2..maximum) {
      val current = "${choice}_$size"
      production(current, argument, ",", previous)
      production(choice, current)
      previous = current
    }
    activePackSymbols[key] = choice
    return choice
  }

  private fun joinArguments(arguments: List<String>): List<String> = buildList {
    arguments.forEachIndexed { index, argument ->
      if (index > 0) add(",")
      add(argument)
    }
  }

  /** Result types already required by the enclosing statement; no template return is guessed. */
  private fun contextualActiveResultTypes(): Set<String> = buildSet {
    val projected = projectCppTokens(prefix)
    if (projected.firstOrNull() == "return") canonicalType(
      context.enclosingReturnType,
      context.enclosingReturnTypeInfo
    )?.takeIf { it in typeSymbols }?.let(::add)

    val specifiers = setOf("const", "volatile", "static", "constexpr")
    val start = projected.indexOfFirst { it !in specifiers }.coerceAtLeast(0)
    sourceSpellableTypes.forEach { type ->
      spellings[type].orEmpty().flatMap(String::typeSpellingVariants).forEach { spelling ->
        if (projected.drop(start).take(spelling.size) != spelling) return@forEach
        val tail = projected.drop(start + spelling.size)
        val declarator = tail.indexOfFirst { it.startsWith("@id:") }
        if (declarator >= 0 && tail.take(declarator).all { it == "*" || it == "&" } &&
          tail.getOrNull(declarator + 1) in setOf("=", "{", "(")) add(type)
      }
    }
    val equals = projected.indexOfLast { it == "=" }
    if (equals >= 0) {
      val lhs = projected.take(equals)
      values.forEach { value ->
        val name = value.semanticName().cachedNameTokens()
        if (name.isNotEmpty() && lhs.takeLast(name.size) == name)
          canonicalType(value.semanticType(), value.typeInfo)
            ?.takeIf { it in typeSymbols }?.let(::add)
      }
    }
  }

  private fun argumentExpression(actual: String, parameter: CppParameter, depth: Int): String {
    val expected = parameter.type.ifBlank { parameter.semanticType() }
    if (parameter.typeInfo?.let { it.kind == "rvalueReference" &&
        (it.isDependent || it.isInstantiationDependent) } == true)
      return expression(actual, depth)
    parameter.typeInfo?.let { info -> when (info.kind) {
      "lvalueReference" -> return qualifiedReferenceExpression(
        actual, depth, ValueCategory.LVALUE, Cv(info.isConst, info.isVolatile),
        includeRvalues = info.isConst && !info.isVolatile
      )
      "rvalueReference" -> return qualifiedReferenceExpression(
        actual, depth, ValueCategory.rvalues, Cv(info.isConst, info.isVolatile)
      )
    } }
    return when {
      expected.isLvalueReferenceType() && !expected.isConstLvalueReferenceType() -> lvalue(actual, depth)
      expected.isConstRvalueReferenceType() -> rvalue(actual, depth)
      expected.trim().endsWith("&&") -> movable(actual, depth)
      else -> expression(actual, depth)
    }
  }

  private fun argumentExpression(
    actual: ArgumentSource,
    parameter: CppParameter,
    depth: Int
  ): String = if (actual.convertedTemporary) expression(actual.type, depth)
  else argumentExpression(actual.type, parameter, depth)

  private fun qualifiedReferenceExpression(
    type: String,
    depth: Int,
    category: ValueCategory,
    target: Cv,
    includeRvalues: Boolean = false
  ): String = qualifiedReferenceExpression(
    type, depth,
    if (includeRvalues) ValueCategory.entries else listOf(category),
    target
  )

  private fun qualifiedReferenceExpression(
    type: String,
    depth: Int,
    categories: List<ValueCategory>,
    target: Cv
  ): String {
    val key = "$type:$depth:${categories.joinToString { it.name }}:${target.code}"
    return qualifiedChoices.getOrPut(key) {
      val symbol = "REFERENCE_CHOICE_${qualifiedChoices.size}"
      categories.forEach { actualCategory -> cvVariants.forEach { cv ->
        if ((cv.isConst && !target.isConst) || (cv.isVolatile && !target.isVolatile))
          return@forEach
        if (productiveStableStates[qualifiedState(type, depth, actualCategory, cv)])
          production(symbol, qualified(type, depth, actualCategory, cv))
      } }
      symbol
    }
  }

  /** Sparse cumulative view over expressions whose outer operator binds at least this tightly. */
  private fun precedenceExpression(
    type: String,
    depth: Int,
    limit: ExpressionPrecedence
  ): String {
    if (limit == ExpressionPrecedence.POSTFIX) return postfix(type, depth)
    val key = "precedence:$type:$depth:${limit.name}"
    return precedenceChoices.getOrPut(key) {
      val symbol = "PRECEDENCE_CHOICE_${precedenceChoices.size}"
      val productiveMask = productiveNativeTypeMasks[genericState(type, depth)]
      admittedPrecedences.getValue(limit).forEach { precedence ->
        if (productiveMask and (1 shl precedence.ordinal) != 0)
          production(symbol, nativePrecedence(type, depth, precedence))
      }
      symbol
    }
  }

  private fun qualifiedPrecedenceExpression(
    type: String,
    depth: Int,
    category: ValueCategory,
    target: Cv,
    limit: ExpressionPrecedence,
    includeRvalues: Boolean = false
  ): String = qualifiedPrecedenceExpression(
    type, depth,
    if (includeRvalues) ValueCategory.entries else listOf(category),
    target, limit
  )

  private fun qualifiedPrecedenceExpression(
    type: String,
    depth: Int,
    categories: List<ValueCategory>,
    target: Cv,
    limit: ExpressionPrecedence
  ): String {
    val key = "qualified-precedence:$type:$depth:${limit.name}:" +
      "${categories.joinToString { it.name }}:${target.code}"
    return precedenceChoices.getOrPut(key) {
      val symbol = "QUALIFIED_PRECEDENCE_CHOICE_${precedenceChoices.size}"
      admittedPrecedences.getValue(limit).forEach { precedence ->
        categories.forEach { actualCategory -> cvVariants.forEach { cv ->
          if ((cv.isConst && !target.isConst) || (cv.isVolatile && !target.isVolatile))
            return@forEach
          val state = qualifiedState(type, depth, actualCategory, cv)
          if (hasProductiveQualifiedPrecedence(state, precedence))
            production(
              symbol,
              qualifiedNativePrecedence(type, depth, actualCategory, cv, precedence)
            )
        } }
      }
      symbol
    }
  }

  private fun exactOperatorProfileState(profile: CppExpressionProfile): ExpressionState? {
    val type = exactProfileType(profile) ?: return null
    val category = exactProfileCategory(profile) ?: return null
    val info = profile.typeInfo ?: return null
    return ExpressionState(type, category, Cv(info.isConst, info.isVolatile))
  }

  /** Exact profile/category/cv operand with the surface operator's precedence boundary. */
  private fun exactOperatorOperand(
    profile: CppExpressionProfile,
    depth: Int,
    limit: ExpressionPrecedence
  ): String? {
    val expression = exactProfileExpression(profile, depth) ?: return null
    if (profile.kind != "opaque") return expression
    val state = exactOperatorProfileState(profile) ?: return null
    val key = "operator-witness:${state.type}:$depth:${state.category.name}:" +
      "${state.cv.code}:${limit.name}"
    return precedenceChoices.getOrPut(key) {
      val symbol = "OPERATOR_WITNESS_OPERAND_${precedenceChoices.size}"
      val qualified = qualifiedState(state.type, depth, state.category, state.cv)
      var hasDirectPrecedence = false
      admittedPrecedences.getValue(limit).forEach { precedence ->
        if (hasProductiveQualifiedPrecedence(qualified, precedence)) {
          hasDirectPrecedence = true
          production(
            symbol,
            qualifiedNativePrecedence(
              state.type, depth, state.category, state.cv, precedence
            )
          )
        }
      }
      // Parentheses turn any exact stable expression into a primary expression without changing
      // its type, category, cv, or ordinary-object status. Use this only when the unparenthesized
      // state binds too weakly for the surface operator position.
      if (!hasDirectPrecedence)
        production(
          symbol,
          "(", qualifiedStable(state.type, depth, state.category, state.cv), ")"
        )
      symbol
    }
  }

  private fun semanticOperatorEdges(): List<OperatorEdge> =
    authoritativeBinaryOperatorWitnesses.mapNotNull { witness ->
      if (!witness.callable.denotesCallable() || witness.callable.denotesConstructor() ||
        witness.callable.operatorToken() == null || witness.result.kind != "opaque"
      ) return@mapNotNull null
      val syntax = binaryOperatorSyntax(witness.operatorSpelling) ?: return@mapNotNull null
      val left = exactOperatorProfileState(witness.left) ?: return@mapNotNull null
      val right = exactOperatorProfileState(witness.right) ?: return@mapNotNull null
      val result = exactOperatorProfileState(witness.result) ?: return@mapNotNull null
      OperatorEdge(witness, left, right, result, syntax)
    }.distinctBy { edge ->
      listOf(
        edge.witness.targetId.orEmpty(), edge.witness.operatorSpelling,
        edge.witness.left.kind, edge.witness.left.spelling.orEmpty(),
        edge.witness.left.objectKind, edge.witness.left.typeInfo.semanticId().orEmpty(),
        edge.left.type, edge.left.category.name, edge.left.cv.code,
        edge.witness.right.kind, edge.witness.right.spelling.orEmpty(),
        edge.witness.right.objectKind, edge.witness.right.typeInfo.semanticId().orEmpty(),
        edge.right.type, edge.right.category.name, edge.right.cv.code,
        edge.witness.result.kind, edge.witness.result.spelling.orEmpty(),
        edge.witness.result.objectKind, edge.witness.result.typeInfo.semanticId().orEmpty(),
        edge.result.type, edge.result.category.name, edge.result.cv.code
      )
    }

  /** A depth-indexed typed automaton preserves long left-associative overload chains compactly. */
  private fun addFactoredOperatorChains() {
    val depth = CPP_SEMANTIC_DEPTH
    // Beyond the ordinary expression-depth budget, retain only stable self-feeding signatures.
    // This covers streams, flags and value builders without multiplying every visible overload.
    semanticOperatorEdgeCache.filter { edge ->
      edge.syntax.associativity == OperatorAssociativity.LEFT &&
        edge.witness.left.kind == "opaque" && edge.left == edge.result
    }.groupBy { edge -> edge.syntax.precedence to edge.syntax.associativity }
      .values.forEachIndexed { group, edges ->
      val states = edges.map(OperatorEdge::result).distinct()
      val stateIds = states.withIndex().associate { (index, state) -> state to index }
      fun symbol(state: ExpressionState, operations: Int) =
        "OPERATOR_CHAIN_${group}_${stateIds.getValue(state)}_$operations"
      val produced = linkedSetOf<ExpressionState>()
      edges.forEach { edge ->
        val left = exactOperatorOperand(
          edge.witness.left, depth, edge.syntax.leftLimit
        ) ?: return@forEach
        val right = exactOperatorOperand(
          edge.witness.right, depth, edge.syntax.rightLimit
        ) ?: return@forEach
        production(
          symbol(edge.result, 1),
          listOf(left) + edge.syntax.tokens + right
        )
        produced += edge.result
      }
      var previousStates: Set<ExpressionState> = produced
      val maximumOperationTokens = edges.maxOf { it.syntax.tokens.size + 1 }
      val maximumOperations = (CPP_MAX_STATEMENT_TOKENS - 2) / maximumOperationTokens
      for (operations in 2..maximumOperations) {
        val nextStates = linkedSetOf<ExpressionState>()
        edges.forEach edgeLoop@ { edge ->
          val right = exactOperatorOperand(
            edge.witness.right, depth, edge.syntax.rightLimit
          ) ?: return@edgeLoop
          previousStates.filter { state -> state == edge.left }.forEach { previous ->
            production(
              symbol(edge.result, operations),
              listOf(symbol(previous, operations - 1)) + edge.syntax.tokens + right
            )
            nextStates += edge.result
          }
        }
        if (nextStates.isEmpty()) break
        nextStates.forEach { state ->
          production("SIMPLE_STATEMENT", symbol(state, operations), ";")
        }
        previousStates = nextStates
      }
    }
  }

  private fun addOperators(depth: Int) {
    val previous = depth - 1

    fun addBuiltinBinary(type: String, result: String, operator: String) {
      val syntax = requireNotNull(binaryOperatorSyntax(operator))
      if (!hasPrecedenceExpression(type, previous, syntax.leftLimit) ||
        !hasPrecedenceExpression(type, previous, syntax.rightLimit)) return
      movableStableExpression(
        result, depth, syntax.precedence,
        listOf(precedenceExpression(type, previous, syntax.leftLimit)) + syntax.tokens +
          precedenceExpression(type, previous, syntax.rightLimit)
      )
    }

    expressionTypes.filter(String::isNumericCppType).forEach { type ->
      val result = type.promotedArithmeticType().takeIf { it in typeSymbols } ?: type
      val arithmetic = if (type.isIntegralCppType()) listOf("+", "-", "*", "/", "%")
      else listOf("+", "-", "*", "/")
      arithmetic.forEach { operator -> addBuiltinBinary(type, result, operator) }
      if (type.isIntegralCppType())
        listOf("&", "|", "^", "<<", ">>").forEach { operator ->
          addBuiltinBinary(type, result, operator)
        }
      booleanType()?.let { boolean ->
        listOf("==", "!=", "<", "<=", ">", ">=").forEach { operator ->
          addBuiltinBinary(type, boolean, operator)
        }
      }
    }

    semanticOperatorEdgeCache.forEach { edge ->
      val syntax = edge.syntax
      val left = exactOperatorOperand(
        edge.witness.left, previous, syntax.leftLimit
      ) ?: return@forEach
      val right = exactOperatorOperand(
        edge.witness.right, previous, syntax.rightLimit
      ) ?: return@forEach
      val rhs = listOf(left) + syntax.tokens + right
      exactStableExpression(
        edge.result.type, depth, edge.result.category, edge.result.cv, syntax.precedence, rhs
      )
    }

    booleanType()?.let { boolean ->
      listOf("&&", "||").forEach { operator ->
        val syntax = requireNotNull(binaryOperatorSyntax(operator))
        movableStableExpression(
          boolean, depth, syntax.precedence,
          listOf(precedenceCondition(previous, syntax.leftLimit)) + syntax.tokens +
            precedenceCondition(previous, syntax.rightLimit)
        )
      }
      movableStableExpression(
        boolean, depth, ExpressionPrecedence.UNARY,
        listOf("!", precedenceCondition(previous, ExpressionPrecedence.UNARY))
      )
      expressionTypes.filter(::isPointer).forEach { pointer ->
        listOf("==", "!=").forEach { operator ->
          val syntax = requireNotNull(binaryOperatorSyntax(operator))
          if (!hasPrecedenceExpression(pointer, previous, syntax.leftLimit)) return@forEach
          movableStableExpression(
            boolean, depth, syntax.precedence,
            listOf(precedenceExpression(pointer, previous, syntax.leftLimit)) +
              syntax.tokens + CPP_NULLPTR
          )
        }
      }
    }

    if (booleanType() != null) expressionTypes.filterNot { it.typeShape() == "void" }.forEach { type ->
      // If both operands have the same type, value category, and cv, the conditional preserves
      // that exact state. Publish it so reference binding/member receivers can consume the result.
      cvVariants.forEach { cv -> ValueCategory.entries.forEach { category ->
        val state = qualifiedState(type, previous, category, cv)
        if (productiveStableStates[state])
          exactStableExpression(
            type, depth, category, cv,
            ExpressionPrecedence.CONDITIONAL,
            listOf(
              precedenceCondition(previous, ExpressionPrecedence.LOGICAL_OR),
              "?", qualified(type, previous, category, cv),
              ":", qualified(type, previous, category, cv)
            )
          )
      } }
      // Retain mixed-category/common-type conditionals in value-only contexts. Without selected
      // conversion facts their result category cannot be recovered soundly.
      if (hasExpression(type, previous))
        expression(
          type, depth, ExpressionPrecedence.CONDITIONAL,
          listOf(
            precedenceCondition(previous, ExpressionPrecedence.LOGICAL_OR),
            "?", expression(type, previous), ":", expression(type, previous)
          )
        )
    }
  }

  private fun precedenceCondition(depth: Int, limit: ExpressionPrecedence): String {
    val key = "condition:$depth:${limit.name}"
    return precedenceChoices.getOrPut(key) {
      val symbol = "BOOLEAN_PRECEDENCE_CHOICE_${precedenceChoices.size}"
      expressionTypes.filter { it.isArithmeticCppType() || isPointer(it) }
        .filter { type -> hasPrecedenceExpression(type, depth, limit) }
        .forEach { type -> production(symbol, precedenceExpression(type, depth, limit)) }
      symbol
    }
  }

  private fun addBooleanCondition(depth: Int) {
    if (booleanType() == null) return
    expressionTypes.filter { it.isArithmeticCppType() || isPointer(it) }
      .filter { type -> hasStableExpression(type, depth) }
      .forEach { type -> production(condition(depth), stable(type, depth)) }
  }

  private fun addStatements() {
    val constraint = requiredDeclaratorConstraint()
    if (constraint.impossible) return
    val requiredName = constraint.name
    val names = requiredName?.let { listOf(encodeIdentifier(it)) } ?: buildList {
      add(CPP_FRESH)
      prefix.filter { it.kind == CppTokenKind.IDENTIFIER && it.text !in CPP_KEYWORDS }
        .mapTo(this) { encodeIdentifier(it.text) }
    }.distinct()
    addDeclarations(names, allowTypeAliases = !constraint.requiresValueBinder)
    addTypeTemplateDeclarations(names, allowTypeAliases = !constraint.requiresValueBinder)
    addUntypedActiveCallStatements(names)
    if (requiredName != null) {
      production("SEMANTIC_STATEMENT", "SIMPLE_STATEMENT")
      return
    }

    expressionTypes.filter { type -> hasExpression(type, CPP_SEMANTIC_DEPTH) }.forEach { type ->
      production("SIMPLE_STATEMENT", expression(type, CPP_SEMANTIC_DEPTH), ";")
    }
    addFactoredOperatorChains()
    addAssignments()
    addReturns()
    production("SEMANTIC_STATEMENT", "SIMPLE_STATEMENT")
    if (booleanType() != null)
      production("SEMANTIC_STATEMENT", "if", "(", condition(CPP_SEMANTIC_DEPTH), ")", "SIMPLE_STATEMENT")
  }

  /** Template-id syntax comes only from an accessible Sema template declaration and its exact
   * parameter categories. It intentionally remains a syntactic type: no specialization,
   * constructor, conversion, or member fact is inferred from an instantiation. */
  private fun addTypeTemplateDeclarations(
    declarators: List<String>,
    allowTypeAliases: Boolean = true
  ) {
    // A primary template-id has no selected specialization identity. Once the downstream compiler
    // found concrete positive binding profiles, this syntactic family cannot claim to match one.
    if (context.requiredBinderObligation?.singletonGate?.accepted?.isNotEmpty() == true) return
    typeTemplates.groupBy { it.id ?: it.qualifiedName ?: it.name }.values
      .forEachIndexed { index, declarations ->
        val parameters = declarations.first().templateParameters
        val arguments = factoredTemplateArgumentLists(parameters)
        if (arguments.isEmpty()) return@forEachIndexed
        val spellings = declarations.flatMap { listOfNotNull(it.name, it.qualifiedName) }
          .map(String::trim).filter(String::isCppQualifiedName)
          .map(String::cppNameTokens).distinct()
        if (spellings.isEmpty()) return@forEachIndexed
        val name = "TYPE_TEMPLATE_${index}_NAME"
        spellings.forEach { production(name, it) }
        val templateId = "TYPE_TEMPLATE_${index}_ID"
        arguments.forEach { list ->
          production(templateId, listOf(name, "<") + list + ">")
        }
        declarators.forEach { declarator ->
          production("SIMPLE_STATEMENT", templateId, declarator, ";")
          production("SIMPLE_STATEMENT", templateId, "*", declarator, ";")
          if (allowTypeAliases)
            production("SIMPLE_STATEMENT", "using", declarator, "=", templateId, ";")
        }
      }
  }

  /** Exact required arity represented by one positional comma chain, never a product.
   *
   * A default proves that the omitted argument is valid; it does not prove that an arbitrary
   * source-spellable type is a valid replacement for that default (policy parameters commonly
   * have semantic requirements not represented by their `typename` category). */
  private fun factoredTemplateArgumentLists(
    parameters: List<CppParameter>
  ): List<List<String>> = templateArgumentLists.getOrPut(parameters) {
    val pack = parameters.indexOfFirst(CppParameter::isPack)
    // A primary class-template pack is trailing. Its Sema-reported category can be unrolled only
    // to this grammar's finite token horizon; unsupported template-template packs stay empty.
    if (pack >= 0 && (pack != parameters.lastIndex || parameters.count(CppParameter::isPack) != 1))
      return@getOrPut emptyList()
    val fixed = if (pack < 0) parameters else parameters.take(pack)
    val required = fixed.indexOfFirst(CppParameter::isOptional)
      .let { if (it < 0) fixed.size else it }
    if (fixed.drop(required).any { !it.isOptional() }) return@getOrPut emptyList()

    val positions = fixed.map { templateArgument(it) }
    val available = positions.indexOfFirst { it == null }
      .let { if (it < 0) positions.size else it }
    if (required > available) return@getOrPut emptyList()
    buildList {
      if (required == 0) add(emptyList())
      val ordinal = templateArgumentLists.size
      val alternatives = "TYPE_TEMPLATE_ARGUMENT_LIST_$ordinal"
      var chain: String? = null
      for (arity in 1..required) {
        val next = "${alternatives}_$arity"
        val argument = requireNotNull(positions[arity - 1])
        if (chain == null) production(next, argument) else production(next, chain, ",", argument)
        chain = next
        if (arity == required) production(alternatives, chain)
      }
      val packArgument = parameters.getOrNull(pack)?.let { templateArgument(it, pack = true) }
        ?.takeIf { required == fixed.size && available == fixed.size }
      val maximumArguments = CPP_MAX_STATEMENT_TOKENS / 2
      if (packArgument != null) {
        repeat((maximumArguments - fixed.size).coerceAtLeast(0)) {
          val arity = fixed.size + it + 1
          val next = "${alternatives}_$arity"
          val previous = chain
          if (previous == null) production(next, packArgument) else
            production(next, previous, ",", packArgument)
          chain = next
          production(alternatives, next)
        }
      }
      if (chain != null) add(listOf(alternatives))
    }
  }

  private fun templateArgument(parameter: CppParameter, pack: Boolean = false): String? {
    val role = parameter.type.ifBlank { parameter.label }.trim().lowercase()
    if (role in setOf("type", "typename", "class")) {
      val key = if (pack) "type:pack" else "type"
      templateArgumentSymbols[key]?.let { return it }
      val candidates = sourceSpellableTypes.filter { type ->
        type.typeShape() != "void" && type !in abstractTypes && (type.isArithmeticCppType() ||
          type in templateArgumentTypes || isPointer(type) ||
          pack && type in templatePackArgumentTypes)
      }
      val syntheticPointers = declarablePointerPointees.filter { pointee ->
        pointee in candidates && pointerTypes[PointerShape(pointee)] !in typeSymbols
      }
      if (candidates.isEmpty() && syntheticPointers.isEmpty()) return null
      return (if (pack) "TYPE_TEMPLATE_PACK_TYPE_ARGUMENT" else "TYPE_TEMPLATE_TYPE_ARGUMENT")
        .also { symbol ->
        templateArgumentSymbols[key] = symbol
        typeSpellingChoice(candidates)?.let { production(symbol, it) }
        pointerTypeSpellingChoice(syntheticPointers)?.let { production(symbol, it) }
      }
    }
    // The endpoint does not yet describe the signature required by a template-template parameter.
    if (role == "template") return null
    val info = parameter.typeInfo?.takeIf(CppTypeInfo::isConcrete) ?: return null
    val type = canonicalType(parameter.semanticType(), info)?.takeIf { it in typeSymbols } ?: return null
    val terminal = when {
      type.typeShape() == "bool" -> CPP_BOOLEAN
      type.isIntegralCppType() -> CPP_INTEGER
      type.isFloatingCppType() -> CPP_FLOATING
      else -> return null
    }
    val key = "value:${info.semanticId()}:$terminal"
    return templateArgumentSymbols[key] ?: "TYPE_TEMPLATE_VALUE_ARGUMENT_${templateArgumentSymbols.size}"
      .also { symbol ->
        templateArgumentSymbols[key] = symbol
        production(symbol, terminal)
      }
  }

  private fun addUntypedActiveCallStatements(names: List<String>) {
    if (contextualActiveResults.isNotEmpty()) return
    if (context.requiredBinderObligation?.singletonGate?.accepted?.isNotEmpty() == true) return
    functions.filter { callable ->
      callable.activeCallable && canonicalType(callable.semanticReturnType(), callable.returnTypeInfo)
        ?.takeIf { isConcrete(callable.returnTypeInfo) && it in typeSymbols } == null
    }.forEach { callable ->
      val head = callable.semanticName().grammarNameTokens()
      if (head.isEmpty() || callable.operatorToken() != null) return@forEach
      activeArgumentLists(callable.parameters, CPP_SEMANTIC_DEPTH - 1).forEach { arguments ->
        val call = head + "(" + arguments + ")"
        production("SIMPLE_STATEMENT", call + ";")
        names.forEach { name ->
          production("SIMPLE_STATEMENT", listOf("auto", name, "=") + call + ";")
        }
      }
    }
  }

  private data class RequiredDeclaratorConstraint(
    val name: String? = null,
    val requiresValueBinder: Boolean = false,
    val impossible: Boolean = false
  )

  private fun requiredDeclaratorConstraint(): RequiredDeclaratorConstraint {
    val diagnostic = sequenceOf(context.requiredIdentifier)
      .plus(context.unresolvedIdentifiers.asSequence())
      .filterNotNull().firstOrNull(IDENTIFIER_REGEX::matches)
    val prefixBinder = prefixDeclarator()
    context.requiredBinderObligation?.let { obligation ->
      if (obligation.binders.size > 1)
        return RequiredDeclaratorConstraint(requiresValueBinder = true, impossible = true)
      val required = obligation.binders.singleOrNull()
      if (required != null) {
        if (prefixBinder != null && prefixBinder != required)
          return RequiredDeclaratorConstraint(requiresValueBinder = true, impossible = true)
        return RequiredDeclaratorConstraint(required, requiresValueBinder = true)
      }
      return RequiredDeclaratorConstraint(prefixBinder)
    }
    return RequiredDeclaratorConstraint(prefixBinder ?: diagnostic)
  }

  private fun prefixDeclarator(): String? {
    if (prefix.isEmpty()) return null
    val ordinaryPrefixes = sourceSpellableTypes.asSequence()
      .filterNot { it.typeShape() == "void" }
      .flatMap { spellings[it].orEmpty().asSequence() }
      .flatMap { it.typeSpellingVariants().asSequence() }
      .map(::CppDeclaratorTypePrefix)
    val templatePrefixes = typeTemplates.asSequence().flatMap { declaration ->
      sequenceOf(declaration.name, declaration.qualifiedName)
        .filterNotNull().map(String::trim).filter(String::isCppQualifiedName)
        .map { CppDeclaratorTypePrefix(it.cppNameTokens(), requiresTemplateArguments = true) }
    }
    val languagePrefixes = sequenceOf(CppDeclaratorTypePrefix(listOf("auto")))
    return cppDeclaratorPrefixBinder(
      prefix,
      (languagePrefixes + ordinaryPrefixes + templatePrefixes).toList()
    )
  }

  /**
   * Applies only evidence compiled for this exact singleton binder and declaration kind. A failed
   * object probe cannot reject an lvalue-reference form (or vice versa), and an unprobed profile
   * remains available unless the oracle explicitly marked its universe complete.
   */
  private fun requiredBindingProfileAllows(
    type: String,
    declarationKind: String,
    targetCv: Cv = Cv()
  ): Boolean = requiredBindingAllowances.getOrPut(Triple(type, declarationKind, targetCv)) {
    val profiles = indexedBindingProfiles ?: return true
    val key = BindingProfileKey(type, declarationKind, targetCv)
    if (key in profiles.accepted) return@getOrPut true
    // Once the bounded oracle found at least one whole-TU-valid binding profile, emit only that
    // compiler-authenticated positive family. The candidate universe may be incomplete, so a gate
    // with no positive result retains unprobed profiles as a recall fallback; it must not let
    // unknown SDK aliases compete with known-valid profiles after a positive route exists.
    if (profiles.hasAccepted) return@getOrPut false
    if (key in profiles.probed) return@getOrPut false
    !profiles.complete
  }

  /** Exact profile gate for the factored `T *` declarator which has no semantic pointer node. */
  private fun requiredSyntheticPointerProfileAllows(pointee: String): Boolean {
    val profiles = indexedBindingProfiles ?: return true
    if (pointee in profiles.acceptedSyntheticPointers) return true
    if (profiles.hasAccepted) return false
    if (pointee in profiles.probedSyntheticPointers) return false
    return !profiles.complete
  }

  /** Parse the compiler's bounded declaration profiles once. The former per-type scan reparsed
   * every profile for every cv/ref candidate, making statement construction quadratic in the
   * Sema inventory even though membership is the only query made by the grammar builder. */
  private fun indexBindingProfiles(): IndexedBindingProfiles? {
    val gate = context.requiredBinderObligation?.singletonGate ?: return null
    val typesByTerminals = mutableMapOf<List<String>, MutableSet<String>>()
    sourceSpellableTypes.forEach { type ->
      spellings[type].orEmpty().flatMap(String::typeSpellingVariants).forEach { terminals ->
        typesByTerminals.getOrPut(terminals, ::linkedSetOf) += type
      }
    }

    data class ProfileIndex(
      val declarations: MutableSet<BindingProfileKey> = linkedSetOf(),
      val syntheticPointers: MutableSet<String> = linkedSetOf()
    )

    fun index(profiles: Set<CppBindingProfile>): ProfileIndex = ProfileIndex().also { result ->
      profiles.forEach { profile ->
        val declarationKind = profile.declarationKind
        var raw = profile.canonicalType ?: profile.type
        raw = when (declarationKind) {
          "lvalueReference" -> raw.replace(Regex("(?<!&)\\s*&\\s*$"), "")
          "rvalueReference" -> raw.replace(Regex("\\s*&&\\s*$"), "")
          else -> raw
        }.trim()
        val parsed = CppExactTypeIdParser(raw).parse() ?: return@forEach
        val matchingTypes = linkedSetOf<String>()
        canonicalType(raw)?.let(matchingTypes::add)
        typesByTerminals[parsed.terminals]?.let(matchingTypes::addAll)
        val cv = Cv(parsed.isConst, parsed.isVolatile)
        matchingTypes.forEach { type ->
          result.declarations += BindingProfileKey(type, declarationKind, cv)
        }
        if (declarationKind == "object" && parsed.kind == CppExactTypeIdKind.POINTER &&
          !parsed.isConst && !parsed.isVolatile &&
          !parsed.baseIsConst && !parsed.baseIsVolatile
        ) canonicalType(parsed.baseSourceSpelling)?.let(result.syntheticPointers::add)
      }
    }

    val accepted = index(gate.accepted)
    val probed = index(gate.probed)
    return IndexedBindingProfiles(
      accepted = accepted.declarations,
      probed = probed.declarations,
      acceptedSyntheticPointers = accepted.syntheticPointers,
      probedSyntheticPointers = probed.syntheticPointers,
      hasAccepted = gate.accepted.isNotEmpty(),
      complete = gate.complete
    )
  }

  private fun addDeclarations(names: List<String>, allowTypeAliases: Boolean = true) {
    val depth = CPP_SEMANTIC_DEPTH
    val syntheticPointers = declarablePointerPointees.filter { pointee ->
      pointerTypes[PointerShape(pointee)] !in typeSymbols &&
        requiredSyntheticPointerProfileAllows(pointee)
    }
    val syntheticPointerSet = syntheticPointers.toHashSet()
    names.forEach { name ->
      typeSymbols.keys.forEach typeLoop@ { type ->
        if (type.typeShape() == "void" || type !in sourceSpellableTypes) return@typeLoop
        val concreteObject = type !in abstractTypes
        val objectAllowed = requiredBindingProfileAllows(type, "object")
        val constObjectAllowed = requiredBindingProfileAllows(
          type, "object", Cv(isConst = true)
        )
        // A positive downstream gate is a closed compiler-authenticated family. Avoid constructing
        // initializer/reference lattices for a source type that cannot spell any member of it; the
        // final reachability pass would discard every one of those rules anyway.
        if (context.requiredBinderObligation?.singletonGate?.accepted?.isNotEmpty() == true) {
          val referenceAllowed = if (structuredTypes) cvVariants.any { cv ->
            requiredBindingProfileAllows(type, "lvalueReference", cv) ||
              requiredBindingProfileAllows(type, "rvalueReference", cv)
          } else requiredBindingProfileAllows(type, "lvalueReference") ||
            requiredBindingProfileAllows(type, "lvalueReference", Cv(isConst = true))
          if (!objectAllowed && !constObjectAllowed && !referenceAllowed) return@typeLoop
        }
        val constructors = constructorsBySemanticType[type].orEmpty()
        val hasDefaultDeclaration = concreteObject && objectAllowed && (
          type in defaultConstructibleTypes || type.isLanguageDefaultConstructible() ||
            constructors.any { it.parameters.all(CppParameter::isOptional) }
        )
        // A declaration-only TypeDecl has no type-specific value language. Its spelling remains
        // available through the factored alias/pointer choices below, without allocating empty
        // initializer and reference states for every declaration in the Sema index.
        if (!hasDefaultDeclaration && type !in expressionTypes &&
          type !in conversionSourcesByTarget && type !in baseSourcesByTarget &&
          !(isPointer(type) && objectAllowed) &&
          !(concreteObject && objectAllowed && constructors.isNotEmpty())
        ) return@typeLoop
        val spelling = listOf(typeSpelling(type))
        if (hasDefaultDeclaration) {
          production("SIMPLE_STATEMENT", spelling + name + ";")
          production("SIMPLE_STATEMENT", spelling + name + listOf("{", "}", ";"))
        }
        assignableTypes(type).filter { actual -> hasExpression(actual, depth) }.forEach { actual ->
          if (concreteObject && objectAllowed) {
            production(
              "SIMPLE_STATEMENT", spelling + name + listOf("=", expression(actual, depth), ";")
            )
            val constSpelling = if (isPointer(type)) spelling + "const"
            else listOf("const") + spelling
            if (constObjectAllowed)
              production(
                "SIMPLE_STATEMENT",
                constSpelling + name + listOf("=", expression(actual, depth), ";")
              )
          }
          // Presentation-only contexts predate exact type/cv metadata. Preserve their conservative
          // legacy reference language; schema-v2 contexts use direct binding states below.
          if (!structuredTypes && !isPointer(type)) {
            if (requiredBindingProfileAllows(type, "lvalueReference"))
              production(
                "SIMPLE_STATEMENT",
                spelling + listOf("&", name, "=", lvalue(actual, depth), ";")
              )
            if (requiredBindingProfileAllows(
                type, "lvalueReference", Cv(isConst = true)
              )) production(
                "SIMPLE_STATEMENT",
                listOf("const") + spelling + listOf("&", name, "=", expression(actual, depth), ";")
              )
          }
        }
        if (structuredTypes)
          addStructuredReferenceDeclarations(type, spelling, name, depth)
        if (isPointer(type) && objectAllowed) {
          production("SIMPLE_STATEMENT", spelling + name + listOf("=", CPP_NULLPTR, ";"))
          production("SIMPLE_STATEMENT", spelling + name + listOf("{", CPP_NULLPTR, "}", ";"))
        }
        constructors.takeIf { concreteObject && objectAllowed }.orEmpty().forEach { constructor ->
          val argumentLists = if (constructor.activeCallable)
            activeArgumentLists(constructor.parameters, depth)
          else factoredArgumentLists(constructor.parameters, depth)
          argumentLists.forEach { arguments ->
            production("SIMPLE_STATEMENT", spelling + name + listOf("{") + arguments + listOf("}", ";"))
            production("SIMPLE_STATEMENT", spelling + name + listOf("(") + arguments + listOf(")", ";"))
          }
        }
      }

      pointerTypeSpellingChoice(syntheticPointers)?.let { pointer ->
        production("SIMPLE_STATEMENT", pointer, name, ";")
        production("SIMPLE_STATEMENT", pointer, name, "{", "}", ";")
        production("SIMPLE_STATEMENT", pointer, name, "=", CPP_NULLPTR, ";")
        production("SIMPLE_STATEMENT", pointer, name, "{", CPP_NULLPTR, "}", ";")
        if (allowTypeAliases)
          production("SIMPLE_STATEMENT", "using", name, "=", pointer, ";")
      }

      expressionTypes.filterNot { it.typeShape() == "void" || it in abstractTypes }
        .filter { actual -> hasExpression(actual, depth) }.forEach { actual ->
        if (requiredBindingProfileAllows(actual, "object"))
          production("SIMPLE_STATEMENT", listOf("auto", name, "=", expression(actual, depth), ";"))
      }
      // A type alias introduces a fresh binder and is valid for every clang-spelled type.
      if (allowTypeAliases) {
        val sharedPointees = typeSpellingChoice(syntheticPointers)
        val remaining = sourceSpellableTypes.filterNot(syntheticPointerSet::contains)
        val choices = listOfNotNull(sharedPointees, typeSpellingChoice(remaining))
        val type = when (choices.size) {
          0 -> null
          1 -> choices.single()
          else -> "TYPE_ALIAS_SPELLING_CHOICE".also { symbol ->
            choices.forEach { production(symbol, it) }
          }
        }
        if (type != null)
          production("SIMPLE_STATEMENT", listOf("using", name, "=", type, ";"))
      }
    }
  }

  /**
   * Reference initialization is direct binding, not value assignment. A type-level arithmetic or
   * user-conversion edge can create a temporary, but that temporary can never initialize mutable
   * `T&`; the current conversion transport also lacks the selected conversion's cv/ref category.
   * Keep structured declarations to exact reference-compatible object identities and direct public
   * base edges reported by Sema. Pointer references are admitted only for the identical structured
   * pointer node—pointer conversions likewise create temporary pointer values.
   */
  private fun addStructuredReferenceDeclarations(
    target: String,
    spelling: List<String>,
    name: String,
    depth: Int
  ) {
    val directSources = directReferenceSources(target)
    if (directSources.isEmpty()) return

    fun declarator(targetCv: Cv, reference: String): List<String> {
      val qualifiers = buildList {
        if (targetCv.isConst) add("const")
        if (targetCv.isVolatile) add("volatile")
      }
      // For a raw pointer spelling, leading `const` qualifies the pointee. Postfix qualifiers are
      // the exact spelling of a cv-qualified pointer object (`T * const &`).
      return if (isPointer(target)) spelling + qualifiers + listOf(reference, name, "=")
      else qualifiers + spelling + listOf(reference, name, "=")
    }

    listOf(Cv(), Cv(isConst = true), Cv(isVolatile = true), Cv(true, true))
      .forEach { targetCv ->
        if (!requiredBindingProfileAllows(
            target, "lvalueReference", targetCv
          )) return@forEach
        // Only a non-volatile const lvalue reference binds a temporary. Mutable, volatile, and
        // const-volatile lvalue references require an exact lvalue expression.
        val categories = if (targetCv.isConst && !targetCv.isVolatile)
          ValueCategory.entries else listOf(ValueCategory.LVALUE)
        directSources.forEach { actual ->
          if (!hasQualifiedStableExpression(actual, depth, categories, targetCv)) return@forEach
          val initializer = qualifiedReferenceExpression(actual, depth, categories, targetCv)
          production(
            "SIMPLE_STATEMENT", declarator(targetCv, "&") + initializer + ";"
          )
        }
        if (targetCv.isConst && !targetCv.isVolatile)
          builtinTemporaryConversionSources(target).forEach { actual ->
            if (!hasExpression(actual, depth)) return@forEach
            production(
              "SIMPLE_STATEMENT", declarator(targetCv, "&") + expression(actual, depth) + ";"
            )
          }
      }

    // An rvalue reference directly binds only xvalues/prvalues. Safe built-in conversions first
    // create a target-typed prvalue, so their source expression may have any category/top-level cv.
    cvVariants.forEach { targetCv ->
      if (!requiredBindingProfileAllows(target, "rvalueReference", targetCv)) return@forEach
      directSources.forEach { actual ->
        if (!hasQualifiedStableExpression(
            actual, depth, ValueCategory.rvalues, targetCv
          )) return@forEach
        production(
          "SIMPLE_STATEMENT",
          declarator(targetCv, "&&") +
            qualifiedReferenceExpression(actual, depth, ValueCategory.rvalues, targetCv) + ";"
        )
      }
      builtinTemporaryConversionSources(target).forEach { actual ->
        if (!hasExpression(actual, depth)) return@forEach
        production(
          "SIMPLE_STATEMENT", declarator(targetCv, "&&") + expression(actual, depth) + ";"
        )
      }
    }
  }

  private fun addAssignments() {
    val depth = CPP_SEMANTIC_DEPTH
    expressionTypes.filter { target -> hasLvalueExpression(target, depth) }.forEach { target ->
      assignableTypes(target).filter { actual -> hasExpression(actual, depth) }.forEach { actual ->
        production("SIMPLE_STATEMENT", lvalue(target, depth), "=", expression(actual, depth), ";")
      }
      if (isPointer(target))
        production("SIMPLE_STATEMENT", lvalue(target, depth), "=", CPP_NULLPTR, ";")
    }

    fun compoundOperands(target: (String) -> Boolean): List<String> = buildSet {
      expressionTypes.filterTo(this, target)
      explicitConversions.forEach { (source, converted) ->
        if (source in expressionTypes && target(converted)) add(source)
      }
    }.sortedBy(typeOrder::getValue)

    val arithmeticOperands = compoundOperands(String::isArithmeticCppType)
      .filter { actual -> hasExpression(actual, depth) }
    val integralOperands = compoundOperands(String::isIntegralOrBooleanCppType)
      .filter { actual -> hasExpression(actual, depth) }
    expressionTypes.filter { target -> hasLvalueExpression(target, depth) }.forEach { target ->
      if (target.isArithmeticCppType()) {
        listOf("+=", "-=", "*=", "/=").forEach { operator ->
          arithmeticOperands.forEach { actual ->
            production("SIMPLE_STATEMENT", lvalue(target, depth), operator,
              expression(actual, depth), ";")
          }
        }
      }
      if (target.isIntegralOrBooleanCppType()) {
        listOf("%=", "<<=", ">>=", "&=", "^=", "|=").forEach { operator ->
          integralOperands.forEach { actual ->
            production("SIMPLE_STATEMENT", lvalue(target, depth), operator,
              expression(actual, depth), ";")
          }
        }
      }
      if (isPointer(target)) listOf("+=", "-=").forEach { operator ->
        integralOperands.forEach { actual ->
          production("SIMPLE_STATEMENT", lvalue(target, depth), operator,
            expression(actual, depth), ";")
        }
      }
    }

    // A compound-assignment witness authenticates one whole source-order mutation relation through
    // BuildBinOp. Keep it statement-only: assignment has weaker, right-associative precedence than
    // the expression automaton above, and publishing it there would permit unwitnessed chains. The
    // result profile is still schema-checked, but its type need not be source-spellable because a
    // discarded expression statement never names that type.
    authoritativeBinaryOperatorWitnesses.forEach { witness ->
      if (witness.operatorSpelling !in CPP_COMPOUND_ASSIGNMENT_OPERATOR_SPELLINGS ||
        !witness.callable.denotesCallable() || witness.callable.denotesConstructor() ||
        witness.result.kind != "opaque" ||
        !witness.result.isWellFormedCppExpressionProfile()
      ) return@forEach
      val left = exactOperatorOperand(
        witness.left, depth, ExpressionPrecedence.LOGICAL_OR
      ) ?: return@forEach
      val right = exactOperatorOperand(
        witness.right, depth, ExpressionPrecedence.CONDITIONAL
      ) ?: return@forEach
      production(
        "SIMPLE_STATEMENT", left, witness.operatorSpelling, right, ";"
      )
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
    val returnParameter = CppParameter(
      type = raw,
      canonicalType = context.canonicalEnclosingReturnType,
      typeInfo = context.enclosingReturnTypeInfo
    )
    compatibleArgumentTypes(returnParameter)
      .filter { actual -> hasArgumentExpression(actual, returnParameter, CPP_SEMANTIC_DEPTH) }
      .forEach { actual ->
      production(
        "SIMPLE_STATEMENT", "return",
        argumentExpression(actual, returnParameter, CPP_SEMANTIC_DEPTH), ";"
      )
    }
    if (isPointer(expected) && returnParameter.acceptsConvertedTemporary())
      production("SIMPLE_STATEMENT", "return", CPP_NULLPTR, ";")
  }

  private fun receiversFor(owner: String, member: CppReference): List<Pair<String, String>> =
    receiverChoices.getOrPut(
      owner to buildString {
        append(if (member.denotesCallable()) "callable" else "field")
        append(':'); append(member.isStaticFact())
        append(':'); append(member.methodCv().code)
        append(':'); append(member.refQualifier.orEmpty())
      }
    ) {
      buildList {
        expressionTypes.forEach { candidate ->
          val pointer = pointerShapes[candidate]
          val pointee = pointer?.pointee ?: candidate.rawPointee()?.let(::canonicalType)
          val objectMatches = candidate == owner ||
            !isPointer(candidate) && candidate to owner in baseConversions
          val pointeeMatches = pointee == owner ||
            pointee != null && pointee to owner in baseConversions
          when {
            objectMatches -> add(candidate to ".")
            (member.isStaticFact() || member.refQualifier != "&&") && pointeeMatches &&
              (!member.denotesCallable() || member.isStaticFact() || pointer == null ||
                member.acceptsCv(Cv(pointer.isConst, pointer.isVolatile))) ->
              add(candidate to "->")
          }
        }
      }.distinct()
    }

  private fun memberReceiver(type: String, depth: Int, member: CppReference): String {
    if (member.isStaticFact()) return postfix(type, depth)
    val target = member.methodCv()
    val key = "receiver:$type:$depth:${target.code}:${member.refQualifier.orEmpty()}"
    return qualifiedChoices.getOrPut(key) {
      val symbol = "RECEIVER_CHOICE_${qualifiedChoices.size}"
      val categories = when (member.refQualifier) {
        "&" -> listOf(ValueCategory.LVALUE)
        "&&" -> ValueCategory.rvalues
        else -> ValueCategory.entries
      }
      categories.forEach { category -> cvVariants.forEach { cv ->
        if (!member.acceptsCv(cv)) return@forEach
        if (productivePostfixStates[qualifiedState(type, depth, category, cv)])
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
        "&&" -> ValueCategory.rvalues
        else -> ValueCategory.entries
      }
      categories.forEach { category -> cvVariants.forEach { cv ->
        if (!member.acceptsCv(cv)) return@forEach
        if (productiveStableStates[qualifiedState(type, depth, category, cv)])
          production(symbol, qualified(type, depth, category, cv))
      } }
      symbol
    }
  }

  private fun CppReference.methodCv(): Cv = Cv(isConstMember(), isVolatileMember())

  private fun CppReference.acceptsCv(actual: Cv): Boolean = methodCv().let { target ->
    (!actual.isConst || target.isConst) && (!actual.isVolatile || target.isVolatile)
  }

  /** Exact structured source types that can bind directly to a reference target. */
  private fun directReferenceSources(target: String): List<String> =
    directReferenceTypes.getOrPut(target) {
      buildSet {
        if (target in expressionTypes) add(target)
        if (!isPointer(target)) baseSourcesByTarget[target].orEmpty().filterTo(this) { actual ->
          actual in expressionTypes && !isPointer(actual)
        }
      }.sortedBy(typeOrder::getValue)
    }

  /** Language-defined conversions whose result temporary does not depend on an untransported
   * constructor/conversion-function ref qualifier. */
  private fun builtinTemporaryConversionSources(target: String): List<String> =
    builtinTemporaryConversionTypes.getOrPut(target) {
      buildSet {
        if (target.isArithmeticCppType())
          addAll(arithmeticTypes)
        if (target.typeShape() == "bool")
          addAll(pointerExpressionTypes)
        pointerShapes[target]?.let { to ->
          val candidatePointees = linkedSetOf(to.pointee)
          candidatePointees += baseSourcesByTarget[to.pointee].orEmpty()
          if (to.pointee.typeShape() == "void")
            pointerExpressionsByPointee.keys.filterTo(candidatePointees) { pointee ->
              pointee in objectTypes && pointee.typeShape() != "void"
            }
          candidatePointees.forEach { pointee ->
            pointerExpressionsByPointee[pointee].orEmpty().filterTo(this) { actual ->
              pointerShapes[actual]?.let { from -> isImplicitPointerConversion(from, to) } == true
            }
          }
        }
      }.asSequence().filter { it != target }.sortedBy(typeOrder::getValue).toList()
    }

  /**
   * Value conversion and reference binding have different source-type relations. In particular,
   * a non-const or volatile lvalue-reference parameter cannot bind the temporary produced by an
   * arithmetic, pointer, or user-defined conversion. The endpoint's direct public base edge is
   * sufficient for an object lvalue binding; pointer references require exact pointer identity.
   */
  private fun compatibleArgumentTypes(parameter: CppParameter): List<ArgumentSource> {
    val expected = canonicalType(parameter.semanticType(), parameter.typeInfo)
      ?.takeIf { it in typeSymbols } ?: return emptyList()
    val info = parameter.typeInfo
    if (!structuredTypes) return assignableTypes(expected).map(::ArgumentSource)
    fun direct() = directReferenceSources(expected).map(::ArgumentSource)
    fun converted() = builtinTemporaryConversionSources(expected)
      .map { ArgumentSource(it, convertedTemporary = true) }
    return when (info?.kind) {
      "lvalueReference" -> if (info.isConst && !info.isVolatile)
        (direct() + converted()).distinct()
      else direct()
      "rvalueReference" -> (direct() + converted()).distinct()
      else -> assignableTypes(expected).map(::ArgumentSource)
    }
  }

  private fun concreteArgumentAlternatives(
    parameter: CppParameter,
    depth: Int
  ): List<List<String>> = buildList {
    compatibleArgumentTypes(parameter).forEach { source ->
      if (hasArgumentExpression(source, parameter, depth))
        add(listOf(argumentExpression(source, parameter, depth)))
    }
    val expected = canonicalType(parameter.semanticType(), parameter.typeInfo)
    if (expected != null && isPointer(expected) && parameter.acceptsConvertedTemporary())
      add(listOf(CPP_NULLPTR))
  }.distinct()

  private fun hasArgumentExpression(
    source: ArgumentSource,
    parameter: CppParameter,
    depth: Int
  ): Boolean {
    if (source.convertedTemporary) return hasExpression(source.type, depth)
    val info = parameter.typeInfo
    if (info?.kind == "rvalueReference" && (info.isDependent || info.isInstantiationDependent))
      return hasExpression(source.type, depth)
    info?.let { exact -> when (exact.kind) {
      "lvalueReference" -> return hasQualifiedStableExpression(
        source.type, depth,
        if (exact.isConst && !exact.isVolatile) ValueCategory.entries
        else listOf(ValueCategory.LVALUE),
        Cv(exact.isConst, exact.isVolatile)
      )
      "rvalueReference" -> return hasQualifiedStableExpression(
        source.type, depth, ValueCategory.rvalues, Cv(exact.isConst, exact.isVolatile)
      )
    } }
    val expected = parameter.type.ifBlank { parameter.semanticType() }
    return when {
      expected.isLvalueReferenceType() && !expected.isConstLvalueReferenceType() ->
        hasLvalueExpression(source.type, depth)
      expected.isConstRvalueReferenceType() -> hasRvalueExpression(source.type, depth)
      expected.trim().endsWith("&&") -> hasMovableExpression(source.type, depth)
      else -> hasExpression(source.type, depth)
    }
  }

  /** Whether a conversion result prvalue may bind to this parameter/reference result. */
  private fun CppParameter.acceptsConvertedTemporary(): Boolean {
    typeInfo?.let { info -> return when (info.kind) {
      "lvalueReference" -> info.isConst && !info.isVolatile
      "rvalueReference" -> true
      else -> true
    } }
    val spelling = type.ifBlank { semanticType() }.trim()
    if (spelling.endsWith("&&")) return true
    if (!spelling.isLvalueReferenceType()) return true
    val referred = spelling.removeSuffix("&").trim()
    val topLevelQualifiers = if ('*' in referred)
      referred.substringAfterLast('*').trim().split(Regex("\\s+")).filter(String::isNotEmpty)
    else referred.split(Regex("\\s+")).filter(String::isNotEmpty)
    return "const" in topLevelQualifiers && "volatile" !in topLevelQualifiers
  }

  private fun assignableTypes(expected: String): List<String> = compatibleTypes.getOrPut(expected) {
    if (!structuredTypes) return@getOrPut expressionTypes.filter { actual ->
      isAssignable(actual, expected)
    }
    val candidates = linkedSetOf(expected)
    candidates += conversionSourcesByTarget[expected].orEmpty()
    if (expected.isArithmeticCppType()) candidates += arithmeticTypes
    if (expected.typeShape() == "bool") candidates += expressionTypes.filter(::isPointer)
    pointerShapes[expected]?.let { target ->
      expressionTypes.filterTo(candidates) { actual ->
        pointerShapes[actual]?.let { source ->
          isImplicitPointerConversion(source, target)
        } == true
      }
    }
    candidates.asSequence().filter { it in expressionTypes }
      .sortedBy(typeOrder::getValue).toList()
  }

  private fun isAssignable(actual: String, expected: String): Boolean {
    if (actual == expected || actual to expected in explicitConversions) return true
    if (actual.isArithmeticCppType() && expected.isArithmeticCppType()) return true
    if (expected.typeShape() == "bool" && isPointer(actual)) return true
    val from = pointerShapes[actual]
    val to = pointerShapes[expected]
    if (from != null && to != null) return isImplicitPointerConversion(from, to)
    return !structuredTypes && isAssignable(actual.typeShape(), expected.typeShape(), explicitConversions)
  }

  private fun isImplicitPointerConversion(from: PointerShape, to: PointerShape): Boolean {
    if (from.isConst && !to.isConst || from.isVolatile && !to.isVolatile) return false
    if (from.pointee == to.pointee) return true
    if (from.pointee to to.pointee in baseConversions) return true
    return to.pointee.typeShape() == "void" && from.pointee in objectTypes &&
      from.pointee.typeShape() != "void"
  }

  private fun CppReference.objectCv(): Cv = typeInfo?.let {
    Cv(it.isConst, it.isVolatile)
  } ?: Cv(
    isConst = !isMutableValueInContext(),
    isVolatile = Regex("(?:^|\\s)volatile(?:\\s|$)").containsMatchIn(semanticType().orEmpty())
  )

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

  /** A syntax-only choice needs no per-type semantic node. This keeps a large declaration index
   * linear: each Sema spelling is emitted once and every declaration form reuses the same choice. */
  private fun typeSpellingChoice(types: Collection<String>): String? {
    val key = types.asSequence().filter { it in typeSymbols && it in sourceSpellableTypes }
      .distinct().sortedBy(typeOrder::getValue).toList()
    if (key.isEmpty()) return null
    return typeSpellingChoiceSymbols.getOrPut(key) {
      "TYPE_SPELLING_CHOICE_${typeSpellingChoiceSymbols.size}".also { symbol ->
        val direct = linkedSetOf<List<String>>()
        key.forEach { type ->
          typeSpellingSymbols[type]?.let { production(symbol, it); return@forEach }
          spellings[type].orEmpty().flatMapTo(direct, String::typeSpellingVariants)
        }
        direct.forEach { production(symbol, it) }
      }
    }
  }

  /** Exact spelling language formerly represented by one synthetic `T *` semantic type. */
  private fun pointerTypeSpellingChoice(pointees: Collection<String>): String? {
    val key = pointees.asSequence().filter { pointee ->
      pointee in typeSymbols && pointee in sourceSpellableTypes
    }.distinct().sortedBy(typeOrder::getValue).toList()
    if (key.isEmpty()) return null
    return pointerSpellingChoiceSymbols.getOrPut(key) {
      val pointee = requireNotNull(typeSpellingChoice(key))
      "POINTER_SPELLING_CHOICE_${pointerSpellingChoiceSymbols.size}".also { symbol ->
        production(symbol, pointee, "*")
      }
    }
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

  private fun booleanType(): String? = expressionTypes.firstOrNull { it.typeShape() == "bool" }
  private fun isPointer(type: String): Boolean = type in pointerShapes || type.isRawPointer()

  private fun String.cachedNameTokens(): List<String> =
    tokenizedNames.getOrPut(this) { cppNameTokens() }

  /** Share an exact multi-token insertion spelling across every depth and overload production. */
  private fun String.grammarNameTokens(): List<String> {
    val tokens = cachedNameTokens()
    if (tokens.size < 2) return tokens
    val symbol = nameSpellingSymbols.getOrPut(this) {
      "SEMANTIC_NAME_${nameSpellingSymbols.size}".also { production(it, tokens) }
    }
    return listOf(symbol)
  }

  private fun expression(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_D$depth"
  private fun expressionOnly(type: String, depth: Int): String =
    "${typeSymbols.getValue(type)}_EXPRESSION_ONLY_D$depth"
  private fun postfix(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_POSTFIX_D$depth"
  private fun postfixOnly(type: String, depth: Int): String =
    "${typeSymbols.getValue(type)}_POSTFIX_ONLY_D$depth"
  private fun stable(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_STABLE_D$depth"
  private fun glvalue(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_GLVALUE_D$depth"
  private fun rvalue(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_RVALUE_D$depth"
  private fun lvalue(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_LVALUE_D$depth"
  private fun movable(type: String, depth: Int): String = "${typeSymbols.getValue(type)}_MOVABLE_D$depth"
  private fun mutablePostfix(type: String, depth: Int): String =
    "${typeSymbols.getValue(type)}_MUTABLE_POSTFIX_D$depth"
  private fun qualified(type: String, depth: Int, category: ValueCategory, cv: Cv): String =
    qualifiedStable(type, depth, category, cv)
  private fun qualifiedPostfix(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv
  ): String = "${typeSymbols.getValue(type)}_POSTFIX_${category.name}_${cv.code}_D$depth"
  private fun qualifiedStable(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv
  ): String = "${typeSymbols.getValue(type)}_STABLE_${category.name}_${cv.code}_D$depth"
  private fun nativePrecedence(
    type: String,
    depth: Int,
    precedence: ExpressionPrecedence
  ): String = if (precedence == ExpressionPrecedence.POSTFIX) postfix(type, depth)
  else "${typeSymbols.getValue(type)}_NATIVE_${precedence.name}_D$depth"
  private fun qualifiedNativePrecedence(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv,
    precedence: ExpressionPrecedence
  ): String = if (precedence == ExpressionPrecedence.POSTFIX)
    qualifiedPostfix(type, depth, category, cv)
  else "${typeSymbols.getValue(type)}_NATIVE_${precedence.name}_${category.name}_${cv.code}_D$depth"
  private fun condition(depth: Int): String = "BOOLEAN_CONDITION_D$depth"

  private fun expression(
    type: String,
    depth: Int,
    precedence: ExpressionPrecedence,
    rhs: List<String>
  ) {
    val state = genericState(type, depth)
    productiveExpressionOnlyStates[state] = true
    markProductiveNativePrecedence(state, precedence)
    production(expressionOnly(type, depth), rhs)
    production(nativePrecedence(type, depth, precedence), rhs)
  }
  private fun postfixExpression(type: String, depth: Int, rhs: List<String>) {
    val state = genericState(type, depth)
    productivePostfixOnlyStates[state] = true
    markProductiveNativePrecedence(state, ExpressionPrecedence.POSTFIX)
    production(postfixOnly(type, depth), rhs)
  }
  private fun exactPostfixExpression(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv,
    rhs: List<String>
  ) {
    val state = qualifiedState(type, depth, category, cv)
    val generic = genericState(type, depth)
    productivePostfixStates[state] = true
    productiveStableStates[state] = true
    exactPostfixTypeStates[generic] = true
    exactStableTypeStates[generic] = true
    markProductiveQualifiedPrecedence(state, ExpressionPrecedence.POSTFIX)
    markProductiveNativePrecedence(generic, ExpressionPrecedence.POSTFIX)
    production(qualifiedPostfix(type, depth, category, cv), rhs)
  }
  private fun exactStableExpression(
    type: String,
    depth: Int,
    category: ValueCategory,
    cv: Cv,
    precedence: ExpressionPrecedence,
    rhs: List<String>
  ) {
    val state = qualifiedState(type, depth, category, cv)
    val generic = genericState(type, depth)
    productiveStableStates[state] = true
    exactStableTypeStates[generic] = true
    markProductiveQualifiedPrecedence(state, precedence)
    markProductiveNativePrecedence(generic, precedence)
    val native = qualifiedNativePrecedence(type, depth, category, cv, precedence)
    production(native, rhs)
    val precedenceBit = 1 shl precedence.ordinal
    if (linkedNativePrecedenceMasks[state] and precedenceBit == 0) {
      linkedNativePrecedenceMasks[state] = linkedNativePrecedenceMasks[state] or precedenceBit
      production(qualifiedStable(type, depth, category, cv), native)
      production(nativePrecedence(type, depth, precedence), native)
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
    precedence: ExpressionPrecedence,
    rhs: List<String>,
    cv: Cv = Cv(isConst = true)
  ) {
    exactStableExpression(type, depth, ValueCategory.LVALUE, cv, precedence, rhs)
  }
  private fun movablePostfixExpression(type: String, depth: Int, rhs: List<String>) {
    exactPostfixExpression(type, depth, ValueCategory.PRVALUE, Cv(), rhs)
  }
  private fun rvaluePostfixExpression(
    type: String,
    depth: Int,
    rhs: List<String>,
    cv: Cv = Cv(isConst = true)
  ) {
    exactPostfixExpression(type, depth, ValueCategory.XVALUE, cv, rhs)
  }
  private fun movableStableExpression(
    type: String,
    depth: Int,
    precedence: ExpressionPrecedence,
    rhs: List<String>
  ) {
    exactStableExpression(type, depth, ValueCategory.PRVALUE, Cv(), precedence, rhs)
  }
  private fun stableRvalueExpression(
    type: String,
    depth: Int,
    precedence: ExpressionPrecedence,
    rhs: List<String>,
    cv: Cv = Cv(isConst = true)
  ) {
    exactStableExpression(type, depth, ValueCategory.XVALUE, cv, precedence, rhs)
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
    precedence: ExpressionPrecedence,
    rhs: List<String>,
    isVolatile: Boolean = false
  ) {
    exactStableExpression(
      type, depth, ValueCategory.LVALUE, Cv(isVolatile = isVolatile), precedence, rhs
    )
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
    var nonempty: Boolean = false,
    var countingOrdinal: Int = -1
  ) {
    val hasLanguage: Boolean get() = nullable || nonempty
  }

  private data class IndexedSourceRule(
    val kind: Int,
    val left: Int = -1,
    val right: Int = -1,
    val terminal: String = ""
  )

  private val preindexedSource = source as? PreindexedAcyclicCFG
  private val fallbackSourceRules = if (preindexedSource == null) source.groupBy { it.first }
    else emptyMap()
  private val sourceSymbols = preindexedSource?.acyclicCountingOrder
    ?: fallbackSourceRules.keys.sorted()
  private val sourceIndex = preindexedSource?.acyclicNonterminalIndex
    ?: sourceSymbols.withIndex().associate { (index, symbol) -> symbol to index }
  private val sourceNonterminals = sourceIndex.keys

  private fun sourceProductions(nonterminal: String): List<Pair<String, List<String>>> =
    preindexedSource?.productionsFor(nonterminal) ?: fallbackSourceRules[nonterminal].orEmpty()

  private val indexedSourceRules = Array(sourceSymbols.size) { index ->
    sourceProductions(sourceSymbols[index]).map { (_, rhs) -> when {
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
  private val sourceChildIndices = Array(sourceSymbols.size) { index ->
    indexedSourceRules[index].flatMapTo(linkedSetOf()) { rule -> when (rule.kind) {
      SOURCE_UNIT_RULE -> listOf(rule.left)
      SOURCE_BINARY_RULE -> listOf(rule.left, rule.right)
      else -> emptyList()
    } }.toIntArray()
  }
  private val sourceTerminals = Array(sourceSymbols.size) { index ->
    indexedSourceRules[index].mapNotNullTo(linkedSetOf()) { rule ->
      rule.terminal.takeIf { rule.kind == SOURCE_TERMINAL_RULE }
    }.toList()
  }
  private val sourceNonterminalProductionCounts = IntArray(sourceSymbols.size) { index ->
    indexedSourceRules[index].count { it.kind != SOURCE_TERMINAL_RULE }
  }
  private val sourceCountingOrder = preindexedSource?.acyclicCountingOrder
    ?: sourceChildBeforeParentOrder()
  private val sourceCountingIndices = sourceCountingOrder.map { sourceIndex.getValue(it) }.toIntArray()
  private val sourceNullable = BooleanArray(sourceSymbols.size)
  private val sourceNonempty = BooleanArray(sourceSymbols.size)
  private val spanBase = CPP_MAX_STATEMENT_TOKENS + 1
  /** One-token derivative memo. It is cleared only when another token is consumed. */
  private val derivativeMemo = mutableMapOf<String, Derivative>()
  /** Completed derivative groups are snapshotted once and reused by every later cursor. */
  private val derivativeRuleLists = mutableMapOf<String, List<Pair<String, List<String>>>>()
  /** Unique nonterminal children of an immutable derivative group, computed once at publication. */
  private val derivativeChildren = mutableMapOf<String, List<String>>()
  private val derivativeStates = mutableMapOf<String, Derivative>()
  private val derivativeCountingOrder = mutableListOf<String>()
  private val sourceReachabilityEpochs = IntArray(sourceSymbols.size)
  private var derivativeReachabilityEpochs = IntArray(0)
  private var reachabilityEpoch = 0
  /** Nullable-only projections preserve the multiplicity of empty left-hand derivations. */
  private val epsilonProjectionMemo = mutableMapOf<String, String>()
  private val countWorkspace = BoundedCountWorkspace()
  private var cachedPrefix = emptyList<String>()
  private var cachedResidual: Derivative? = null
  private var derivativeGeneration = 0
  var lastMetrics: CppConditioningMetrics = CppConditioningMetrics()
    private set

  init {
    // The source grammar is acyclic and [sourceCountingOrder] is child-before-parent. Retain both
    // halves of the language state: nullable-only children must not be mistaken for productive
    // nonempty suffixes when an incremental derivative is assembled.
    sourceCountingOrder.forEach { symbol ->
      val index = sourceIndex.getValue(symbol)
      indexedSourceRules[index].forEach { rule -> when (rule.kind) {
        SOURCE_EPSILON_RULE -> sourceNullable[index] = true
        SOURCE_TERMINAL_RULE -> sourceNonempty[index] = true
        SOURCE_UNIT_RULE -> {
          sourceNullable[index] = sourceNullable[index] || sourceNullable[rule.left]
          sourceNonempty[index] = sourceNonempty[index] || sourceNonempty[rule.left]
        }
        SOURCE_BINARY_RULE -> {
          val rightAny = sourceNullable[rule.right] || sourceNonempty[rule.right]
          sourceNullable[index] = sourceNullable[index] ||
            sourceNullable[rule.left] && sourceNullable[rule.right]
          sourceNonempty[index] = sourceNonempty[index] ||
            sourceNonempty[rule.left] && rightAny ||
            sourceNullable[rule.left] && sourceNonempty[rule.right]
        }
      } }
    }
  }

  /** Computes the immutable source grammar's stable order once for every cursor residual. */
  private fun sourceChildBeforeParentOrder(): List<String> {
    val visiting = mutableSetOf<String>()
    val visited = mutableSetOf<String>()
    val order = mutableListOf<String>()
    fun visit(nonterminal: String) {
      if (nonterminal in visited) return
      check(visiting.add(nonterminal)) { "Prepared C++ source grammar contains a cycle at $nonterminal" }
      sourceProductions(nonterminal).forEach { (_, rhs) ->
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

  private fun reset() {
    derivativeMemo.clear()
    derivativeRuleLists.clear()
    derivativeChildren.clear()
    derivativeStates.clear()
    derivativeCountingOrder.clear()
    derivativeReachabilityEpochs = IntArray(0)
    epsilonProjectionMemo.clear()
    countWorkspace.clear()
    cachedPrefix = emptyList()
    cachedResidual = null
    derivativeGeneration = 0
  }

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
          end == start + 1 && cppTerminalMatches(rule.terminal, tokens[start])
        SOURCE_UNIT_RULE -> generatesExactly(rule.left, start, end, tokens, memo)
        SOURCE_BINARY_RULE ->
          sourceNullable[rule.left] &&
            generatesExactly(rule.right, start, end, tokens, memo) ||
            sourceNullable[rule.right] &&
            generatesExactly(rule.left, start, end, tokens, memo) ||
            (start + 1 until end).any { split ->
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
    if (tokens.size > CPP_MAX_STATEMENT_TOKENS || source.isEmpty()) return false
    if (tokens.isEmpty()) return sourceNullable[sourceIndex.getValue("START")]
    return generatesExactly(sourceIndex.getValue("START"), 0, tokens.size, tokens, mutableMapOf())
  }

  private fun symbolNullable(symbol: String): Boolean =
    sourceIndex[symbol]?.let { sourceNullable[it] }
      ?: derivativeStates[symbol]?.nullable ?: false

  private fun symbolNonempty(symbol: String): Boolean =
    sourceIndex[symbol]?.let { sourceNonempty[it] }
      ?: derivativeStates[symbol]?.nonempty ?: false

  private fun symbolHasLanguage(symbol: String): Boolean =
    symbolNullable(symbol) || symbolNonempty(symbol)

  private fun publishDerivativeGroup(
    state: Derivative,
    productions: Collection<Pair<String, List<String>>>
  ) {
    val rules = productions.toList()
    derivativeRuleLists[state.symbol] = rules
    derivativeChildren[state.symbol] = rules.flatMapTo(linkedSetOf()) { it.second }.toList()
    state.countingOrdinal = derivativeCountingOrder.size
    derivativeStates[state.symbol] = state
    derivativeCountingOrder += state.symbol
  }

  /**
   * Retains exactly the epsilon derivations of [symbol]. A Boolean nullable splice is sufficient
   * for language recognition, but it collapses parses: in D_a(L R), every empty derivation of L
   * must remain paired with D_a(R). The projection is itself an acyclic postordered DAG.
   */
  private fun epsilonProjection(symbol: String): String {
    check(symbolNullable(symbol)) { "Cannot project non-nullable symbol $symbol onto epsilon" }
    if (!symbolNonempty(symbol)) return symbol
    epsilonProjectionMemo[symbol]?.let { return it }

    val result = "PREPARED_E_${epsilonProjectionMemo.size}"
    epsilonProjectionMemo[symbol] = result
    val productions = linkedSetOf<Pair<String, List<String>>>()
    fun addEpsilon() {
      productions += result to emptyList()
    }
    fun addUnit(child: String) {
      if (symbolNullable(child)) productions += result to listOf(epsilonProjection(child))
    }
    fun addBinary(left: String, right: String) {
      if (symbolNullable(left) && symbolNullable(right))
        productions += result to listOf(epsilonProjection(left), epsilonProjection(right))
    }

    val sourceOrdinal = sourceIndex[symbol]
    if (sourceOrdinal != null) {
      indexedSourceRules[sourceOrdinal].forEach { rule -> when (rule.kind) {
        SOURCE_EPSILON_RULE -> addEpsilon()
        SOURCE_UNIT_RULE -> addUnit(sourceSymbols[rule.left])
        SOURCE_BINARY_RULE -> addBinary(sourceSymbols[rule.left], sourceSymbols[rule.right])
      } }
    } else {
      derivativeRuleLists[symbol].orEmpty().forEach { (_, rhs) -> when (rhs.size) {
        0 -> addEpsilon()
        1 -> if (rhs[0] in sourceNonterminals || rhs[0] in derivativeStates) addUnit(rhs[0])
        2 -> addBinary(rhs[0], rhs[1])
        else -> error("Incremental C++ epsilon projection requires binary normal form: $symbol -> $rhs")
      } }
    }
    check(productions.isNotEmpty()) { "Nullable symbol $symbol has no epsilon derivation" }
    publishDerivativeGroup(Derivative(result, nullable = true), productions)
    return result
  }

  /** Exact derivative by one newly committed terminal, over the preceding residual DAG. */
  private fun derivative(symbol: String, terminal: String): Derivative {
    derivativeMemo[symbol]?.let { return it }
    val result = Derivative("PREPARED_D_${derivativeGeneration}_${derivativeMemo.size}")
    derivativeMemo[symbol] = result
    val productions = linkedSetOf<Pair<String, List<String>>>()

    fun addEpsilon() {
      productions += result.symbol to emptyList()
      result.nullable = true
    }
    fun addUnit(child: String) {
      if (!symbolHasLanguage(child)) return
      productions += result.symbol to listOf(child)
      result.nullable = result.nullable || symbolNullable(child)
      result.nonempty = result.nonempty || symbolNonempty(child)
    }
    fun addBinary(left: String, right: String) {
      val leftNullable = symbolNullable(left)
      val leftNonempty = symbolNonempty(left)
      val rightNullable = symbolNullable(right)
      val rightNonempty = symbolNonempty(right)
      val leftAny = leftNullable || leftNonempty
      val rightAny = rightNullable || rightNonempty
      if (!leftAny || !rightAny) return
      productions += result.symbol to listOf(left, right)
      result.nullable = result.nullable || leftNullable && rightNullable
      result.nonempty = result.nonempty ||
        leftNonempty && rightAny || leftNullable && rightNonempty
    }
    fun differentiateUnit(child: String) {
      addUnit(derivative(child, terminal).symbol)
    }
    fun differentiateBinary(left: String, right: String) {
      addBinary(derivative(left, terminal).symbol, right)
      if (symbolNullable(left))
        addBinary(epsilonProjection(left), derivative(right, terminal).symbol)
    }

    val sourceOrdinal = sourceIndex[symbol]
    if (sourceOrdinal != null) {
      indexedSourceRules[sourceOrdinal].forEach { rule -> when (rule.kind) {
        SOURCE_TERMINAL_RULE -> if (cppTerminalMatches(rule.terminal, terminal)) addEpsilon()
        SOURCE_UNIT_RULE -> differentiateUnit(sourceSymbols[rule.left])
        SOURCE_BINARY_RULE -> differentiateBinary(
          sourceSymbols[rule.left], sourceSymbols[rule.right]
        )
      } }
    } else {
      derivativeRuleLists[symbol].orEmpty().forEach { (_, rhs) -> when (rhs.size) {
        0 -> Unit
        1 -> if (rhs[0] in sourceNonterminals || rhs[0] in derivativeStates)
          differentiateUnit(rhs[0]) else if (cppTerminalMatches(rhs[0], terminal)) addEpsilon()
        2 -> differentiateBinary(rhs[0], rhs[1])
        else -> error("Incremental C++ derivative requires binary normal form: $symbol -> $rhs")
      } }
    }
    // All derivative children have completed recursively, so append this node in postorder.
    publishDerivativeGroup(result, productions)
    return result
  }

  private fun reachableGrammar(rootRules: Collection<Pair<String, List<String>>>): OrderedGrammar {
    val chunks = mutableListOf<Collection<Pair<String, List<String>>>>(rootRules)
    var productionCount = rootRules.size
    var nonterminalProductions = rootRules.size
    val terminals = linkedSetOf<String>()
    val queue = rootRules.flatMapTo(ArrayList()) { (_, rhs) -> rhs }
    val rootSymbols = rootRules.mapTo(linkedSetOf()) { it.first }

    // Epoch marks avoid allocating and hashing a large String set at every cursor. Source and
    // derivative nodes already have dense stable ordinals in child-before-parent order.
    if (reachabilityEpoch == Int.MAX_VALUE) {
      sourceReachabilityEpochs.fill(0)
      derivativeReachabilityEpochs.fill(0)
      reachabilityEpoch = 1
    } else reachabilityEpoch++
    val epoch = reachabilityEpoch
    if (derivativeReachabilityEpochs.size < derivativeCountingOrder.size)
      derivativeReachabilityEpochs = derivativeReachabilityEpochs.copyOf(derivativeCountingOrder.size)

    var reachableSourceCount = 0
    var reachableDerivativeCount = 0
    var next = 0
    while (next < queue.size) {
      val symbol = queue[next++]
      val sourceOrdinal = sourceIndex[symbol]
      if (sourceOrdinal != null) {
        if (sourceReachabilityEpochs[sourceOrdinal] == epoch) continue
        sourceReachabilityEpochs[sourceOrdinal] = epoch
        reachableSourceCount++
        val productions = sourceProductions(symbol)
        chunks += productions
        productionCount += productions.size
        nonterminalProductions += sourceNonterminalProductionCounts[sourceOrdinal]
        terminals += sourceTerminals[sourceOrdinal]
        sourceChildIndices[sourceOrdinal].forEach { child -> queue += sourceSymbols[child] }
        continue
      }

      val state = derivativeStates[symbol] ?: continue
      val ordinal = state.countingOrdinal
      check(ordinal >= 0) { "Unpublished prepared C++ derivative $symbol" }
      if (derivativeReachabilityEpochs[ordinal] == epoch) continue
      derivativeReachabilityEpochs[ordinal] = epoch
      reachableDerivativeCount++
      val productions = derivativeRuleLists[symbol].orEmpty()
      chunks += productions
      productionCount += productions.size
      nonterminalProductions += productions.size
      queue += derivativeChildren[symbol].orEmpty()
    }

    // Source groups are immutable, and completed derivative groups are never mutated after they
    // are published. Their LHS namespaces are disjoint and each group is already duplicate-free,
    // so this chunked view retains exact Set semantics without hashing/copying every production.
    val nonterminalCount = reachableSourceCount + reachableDerivativeCount + rootSymbols.size
    val order = buildList(nonterminalCount) {
      sourceCountingIndices.forEach { ordinal ->
        if (sourceReachabilityEpochs[ordinal] == epoch) add(sourceSymbols[ordinal])
      }
      derivativeCountingOrder.forEachIndexed { ordinal, symbol ->
        if (derivativeReachabilityEpochs[ordinal] == epoch) add(symbol)
      }
      addAll(rootSymbols)
    }
    check(order.size == nonterminalCount) {
      "Prepared C++ residual order has ${order.size} symbols, expected $nonterminalCount"
    }
    val nonterminalIndex = buildMap(order.size) {
      order.forEachIndexed { index, symbol -> put(symbol, index) }
    }
    val rootLists = rootRules.groupBy { it.first }
    val syntax = IndexedChunkedCppCfg(
      chunks = chunks.toList(),
      size = productionCount,
      acyclicCountingOrder = order,
      acyclicNonterminalIndex = nonterminalIndex,
      acyclicStructuralStats =
        "CFG(|Σ|=${terminals.size}, |V|=$nonterminalCount, |P|=$nonterminalProductions)",
      hasExactLiteralTerminals = terminals.any { it.exactLiteral() != null },
      productionLookup = { symbol ->
        rootLists[symbol]
          ?: sourceProductions(symbol).takeIf(List<*>::isNotEmpty)
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
    if (!extendsCached) reset()
    for (index in cachedPrefix.size until prefix.size) {
      derivativeMemo.clear()
      derivativeGeneration++
      val input = cachedResidual?.symbol ?: "START"
      cachedResidual = derivative(input, prefix[index])
    }
    cachedPrefix = prefix.toList()

    val rootSymbol = "PREPARED_ROOT_${prefix.size}"
    val rootRules = linkedSetOf<Pair<String, List<String>>>()
    if (prefix.isEmpty()) {
      rootRules += rootSymbol to listOf("START")
    } else {
      // Each cursor extends the previous one by a terminal. The cached residual already denotes
      // w^-1(G), so differentiating that DAG by only the new terminal yields (wa)^-1(G) without
      // revisiting every split of w. Nullable alternatives are explicit productions in the DAG.
      val residual = requireNotNull(cachedResidual)
      if (!residual.hasLanguage) {
        lastMetrics = CppConditioningMetrics(
          derivativeMillis = derivativeClock.elapsedNow().inWholeMilliseconds
        )
        return emptySet<Pair<String, List<String>>>().boundedAcyclic(maxSuffixTokens)
      }
      rootRules += rootSymbol to listOf(residual.symbol)
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
  // Productive pruning also deduplicates each source LHS. Lifting therefore stays injective while
  // avoiding a second hash pass over the substantially larger binary grammar.
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
  // With an unused FINITE_* namespace the transformation is injective: terminal symbols are
  // one-to-one, structural suffix nodes are interned, and distinct source rules retain either a
  // distinct LHS, head, or tail.
  val generatedNamespaceIsFresh = grammar.none { (lhs) ->
    lhs.startsWith("FINITE_TERMINAL_") || lhs.startsWith("FINITE_SUFFIX_")
  }
  val injectiveTransformation = generatedNamespaceIsFresh
  val cnf = ArrayList<Pair<String, List<String>>>(lifted.size)
  data class SuffixKey(val head: String, val terminalTail: String? = null, val child: Int = -1)
  val suffixNodes = linkedMapOf<SuffixKey, Int>()
  val suffixKeys = mutableListOf<SuffixKey>()
  val suffixSymbols = mutableListOf<String?>()
  var suffixCount = 0
  fun suffixNode(rhs: List<String>, start: Int): Int {
    var child = -1
    for (index in rhs.lastIndex - 1 downTo start) {
      val key = if (child < 0) SuffixKey(rhs[index], rhs.last())
      else SuffixKey(rhs[index], child = child)
      child = suffixNodes.getOrPut(key) {
        suffixKeys += key
        suffixSymbols += null
        suffixKeys.lastIndex
      }
    }
    return child
  }
  fun suffixSymbol(node: Int): String {
    suffixSymbols[node]?.let { return it }
    val symbol = "FINITE_SUFFIX_${suffixCount++}"
    suffixSymbols[node] = symbol
    val key = suffixKeys[node]
    val tail = key.terminalTail ?: suffixSymbol(key.child)
    cnf += symbol to listOf(key.head, tail)
    return symbol
  }
  lifted.forEach { (lhs, rhs) ->
    cnf += if (rhs.size <= 2) lhs to rhs
    else lhs to listOf(rhs.first(), suffixSymbol(suffixNode(rhs, 1)))
  }
  // [grammar] is already productive and START-reachable. Lifting and suffix binarization introduce
  // only referenced terminal/tail nodes whose children are productive, so a second whole-grammar
  // fixed point is redundant (and was the dominant cold cost for the largest cursor context).
  return indexedAcyclicGrammar(cnf, deduplicate = !injectiveTransformation)
}

/** Publishes the completed source DAG together with the indexes every quotient will reuse. */
private fun indexedAcyclicGrammar(
  grammar: Collection<Pair<String, List<String>>>,
  deduplicate: Boolean = false
): CFG {
  if (grammar.isEmpty()) return emptySet()
  val byLhs = linkedMapOf<String, MutableList<Pair<String, List<String>>>>()
  if (deduplicate) {
    // Grouping is required by the prepared conditioner anyway. Deduplicate RHSs inside their LHS
    // bucket so the fallback path never allocates and hashes a second global Set<Pair<...>>.
    val rhsByLhs = linkedMapOf<String, LinkedHashSet<List<String>>>()
    grammar.forEach { (lhs, rhs) -> rhsByLhs.getOrPut(lhs, ::linkedSetOf) += rhs }
    rhsByLhs.forEach { (lhs, alternatives) ->
      byLhs[lhs] = alternatives.mapTo(ArrayList(alternatives.size)) { lhs to it }
    }
  } else grammar.forEach { production ->
    byLhs.getOrPut(production.first) { mutableListOf() } += production
  }
  val productionCount = byLhs.values.sumOf { it.size }
  val symbols = byLhs.keys.toList()
  val sourceOrdinal = HashMap<String, Int>(symbols.size)
  symbols.forEachIndexed { ordinal, symbol -> sourceOrdinal[symbol] = ordinal }

  // CNF gives every rule at most two children. Dense child ordinals and byte visitation state
  // avoid rebuilding several large String sets merely to publish the already-established DAG.
  val children = Array(symbols.size) { IntArray(0) }
  val seenChildAtParent = IntArray(symbols.size) { -1 }
  val terminals = linkedSetOf<String>()
  var nonterminalProductions = 0
  symbols.forEachIndexed { parent, symbol ->
    val childOrdinals = ArrayList<Int>()
    byLhs.getValue(symbol).forEach { (_, rhs) ->
      if (rhs.size == 1 && sourceOrdinal[rhs[0]] == null) terminals += rhs[0]
      else nonterminalProductions++
      rhs.forEach { child -> sourceOrdinal[child]?.let { ordinal ->
        if (seenChildAtParent[ordinal] != parent) {
          seenChildAtParent[ordinal] = parent
          childOrdinals += ordinal
        }
      } }
    }
    children[parent] = childOrdinals.toIntArray()
  }

  val visitState = ByteArray(symbols.size)
  val orderedOrdinals = ArrayList<Int>(symbols.size)
  fun visit(nonterminal: Int) {
    when (visitState[nonterminal].toInt()) {
      2 -> return
      1 -> error("Semantic C++ grammar contains a cycle at ${symbols[nonterminal]}")
    }
    visitState[nonterminal] = 1
    children[nonterminal].forEach(::visit)
    visitState[nonterminal] = 2
    orderedOrdinals += nonterminal
  }
  symbols.indices.forEach(::visit)
  val order = orderedOrdinals.mapTo(ArrayList(symbols.size)) { symbols[it] }
  val index = HashMap<String, Int>(order.size)
  order.forEachIndexed { ordinal, symbol -> index[symbol] = ordinal }
  return IndexedChunkedCppCfg(
    chunks = byLhs.values.toList(),
    size = productionCount,
    acyclicCountingOrder = order,
    acyclicNonterminalIndex = index,
    acyclicStructuralStats =
      "CFG(|Σ|=${terminals.size}, |V|=${symbols.size}, |P|=$nonterminalProductions)",
    hasExactLiteralTerminals = terminals.any { it.exactLiteral() != null },
    productionLookup = { symbol -> byLhs[symbol].orEmpty() }
  )
}

private fun pruneSemanticGrammar(
  grammar: Collection<Pair<String, List<String>>>
): List<Pair<String, List<String>>> {
  if (grammar.isEmpty()) return emptyList()
  val productions = if (grammar is List) grammar else grammar.toList()
  val byLhs = linkedMapOf<String, MutableList<Int>>()
  val nonterminals = linkedSetOf<String>()
  productions.forEachIndexed { index, (lhs) ->
    nonterminals += lhs
    byLhs.getOrPut(lhs) { mutableListOf() } += index
  }
  // A depth/type symbol can be referenced before it receives any productive atom or call rule.
  // Treat it as a dead nonterminal, not as a literal terminal such as `TYPE_7_D0`. Cache the exact
  // classifier by spelling: large semantic grammars repeat each depth symbol thousands of times.
  val generatedSymbol = mutableMapOf<String, Boolean>()
  productions.forEach { (_, rhs) -> rhs.forEach { symbol ->
    if (symbol !in nonterminals && generatedSymbol.getOrPut(symbol) {
        isGeneratedExpressionSymbol(symbol)
      }) nonterminals += symbol
  } }

  val symbols = nonterminals.toList()
  val ordinalBySymbol = HashMap<String, Int>(symbols.size)
  symbols.forEachIndexed { ordinal, symbol -> ordinalBySymbol[symbol] = ordinal }
  val rulesBySymbol = Array(symbols.size) { ordinal ->
    byLhs[symbols[ordinal]].orEmpty().toIntArray()
  }

  // Store every nonterminal dependency once in a compact CSR table. Productivity, reachability,
  // and final filtering then use integer arrays instead of repeating String hashing over each RHS.
  val childOffsets = IntArray(productions.size + 1)
  productions.forEachIndexed { production, (_, rhs) ->
    var childCount = 0
    rhs.forEach { if (ordinalBySymbol[it] != null) childCount++ }
    childOffsets[production + 1] = childOffsets[production] + childCount
  }
  val childOrdinals = IntArray(childOffsets.last())
  productions.forEachIndexed { production, (_, rhs) ->
    var next = childOffsets[production]
    rhs.forEach { child -> ordinalBySymbol[child]?.let { childOrdinals[next++] = it } }
  }

  val generating = BooleanArray(symbols.size)
  val generationState = ByteArray(symbols.size)
  val productiveRule = ByteArray(productions.size)
  fun generates(ordinal: Int): Boolean {
    when (generationState[ordinal].toInt()) {
      2 -> return generating[ordinal]
      1 -> error("Semantic C++ grammar contains a cycle at ${symbols[ordinal]}")
    }
    generationState[ordinal] = 1
    var result = false
    val rules = rulesBySymbol[ordinal]
    for (rule in rules) {
      var productive = true
      for (edge in childOffsets[rule] until childOffsets[rule + 1]) {
        if (!generates(childOrdinals[edge])) {
          productive = false
          break
        }
      }
      productiveRule[rule] = if (productive) 2 else 1
      if (productive) {
        result = true
        break
      }
    }
    generating[ordinal] = result
    generationState[ordinal] = 2
    return result
  }
  fun ruleGenerates(rule: Int): Boolean {
    when (productiveRule[rule].toInt()) {
      2 -> return true
      1 -> return false
    }
    for (edge in childOffsets[rule] until childOffsets[rule + 1]) {
      if (!generates(childOrdinals[edge])) {
        productiveRule[rule] = 1
        return false
      }
    }
    productiveRule[rule] = 2
    return true
  }

  val start = ordinalBySymbol["START"] ?: return emptyList()
  if (!generates(start)) return emptyList()

  val reachable = BooleanArray(symbols.size)
  reachable[start] = true
  val queue = ArrayList<Int>().apply { add(start) }
  var next = 0
  while (next < queue.size) {
    rulesBySymbol[queue[next++]].forEach { rule ->
      if (ruleGenerates(rule)) {
        for (edge in childOffsets[rule] until childOffsets[rule + 1]) {
          val child = childOrdinals[edge]
          if (!reachable[child]) {
            reachable[child] = true
            queue += child
          }
        }
      }
    }
  }
  val result = ArrayList<Pair<String, List<String>>>(productions.size)
  val emittedByLhs = mutableMapOf<String, MutableSet<List<String>>>()
  productions.forEachIndexed { rule, production ->
    if (reachable[ordinalBySymbol.getValue(production.first)] && ruleGenerates(rule) &&
      emittedByLhs.getOrPut(production.first, ::linkedSetOf).add(production.second)
    ) result += production
  }
  return result
}

/** Internal expression symbols are never raw C++ terminals: identifiers are encoded as `@id:`. */
internal fun isGeneratedExpressionSymbol(symbol: String): Boolean =
  symbol == "SEMANTIC_STATEMENT" || symbol == "SIMPLE_STATEMENT" ||
    symbol.startsWith("TYPE_") || symbol.startsWith("BOOLEAN_CONDITION_D") ||
    symbol.startsWith("REFERENCE_CHOICE_") || symbol.startsWith("RECEIVER_CHOICE_") ||
    symbol.startsWith("OBJECT_CHOICE_") || symbol.startsWith("PRECEDENCE_CHOICE_") ||
    symbol.startsWith("QUALIFIED_PRECEDENCE_CHOICE_") ||
    symbol.startsWith("OPERATOR_RECEIVER_CHOICE_") ||
    symbol.startsWith("OPERATOR_WITNESS_OPERAND_") ||
    symbol.startsWith("BOOLEAN_PRECEDENCE_CHOICE_")


private fun CppParameter.semanticType(): String = canonicalType?.takeIf(String::isNotBlank) ?: type
private fun CppConversion.semanticFromType(): String =
  canonicalFromType?.takeIf(String::isNotBlank) ?: from
private fun CppConversion.semanticToType(): String =
  canonicalToType?.takeIf(String::isNotBlank) ?: to
private fun CppParameter.isOptional(): Boolean = hasDefault ?: (defaultValue != null)
private fun CppTypeInfo?.semanticId(): String? =
  this?.valueCanonicalId?.takeIf(String::isNotBlank)
    ?: this?.canonicalId?.takeIf(String::isNotBlank)

private fun CppTypeInfo?.isConcrete(): Boolean = this == null ||
  (!isDependent && !isInstantiationDependent && semanticId() != null)

/** An opaque value need not have a source-spellable static type; its exact typed expression does. */
private fun CppTypeInfo.isUsableExpressionWitnessValue(): Boolean =
  !isDependent && !isInstantiationDependent && semanticId() != null &&
    (isComplete != false || kind in setOf("builtin", "function"))

/** `void` and function types are not object-complete; an explicit incomplete object is rejected. */
private fun CppTypeInfo.isUsableExpressionWitnessTypeId(): Boolean =
  isSourceSpellable == true && isUsableExpressionWitnessValue()

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

/** A primary function template without a deducible occurrence needs an explicit template-id. */
private fun CppReference.hasDeducibleTemplateArguments(): Boolean =
  templateParameters.isEmpty() || activeCallable || templateParameters.all { templateParameter ->
    val name = templateParameter.name.takeIf(IDENTIFIER_REGEX::matches) ?: return@all false
    val occurrence = Regex("(?:^|[^A-Za-z_0-9])${Regex.escape(name)}(?:$|[^A-Za-z_0-9])")
    parameters.any { parameter ->
      occurrence.containsMatchIn(parameter.type) ||
        occurrence.containsMatchIn(parameter.canonicalType.orEmpty())
    }
  }

private fun CppReference.denotesType(): Boolean = isType ?: (!denotesEnumConstant() &&
  kind.lowercase().let {
    "type" in it || "class" in it || "struct" in it || "enum" in it || "alias" in it
  })

private fun CppReference.denotesEnumConstant(): Boolean = kind.lowercase().let {
  "enumconstant" in it || "enummember" in it
}

private fun CppReference.isClassTemplateDeclaration(): Boolean =
  denotesType() && templateParameters.isNotEmpty() &&
    kind.contains("classTemplate", ignoreCase = true) &&
    !kind.contains("specialization", ignoreCase = true)

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
  return spelling.takeIf { it in CPP_BINARY_OPERATOR_SPELLINGS }
}

/** The only surface/selected mismatches admitted by C++20 rewritten candidates. */
private fun selectedOperatorCanImplementSurfaceOperator(
  surface: String,
  selected: String?
): Boolean = selected == surface || when (surface) {
  "<", "<=", ">", ">=" -> selected == "<=>"
  "!=" -> selected == "=="
  else -> false
}

private fun String?.isLvalueReferenceType(): Boolean =
  this?.trim()?.let { it.endsWith('&') && !it.endsWith("&&") } == true

private fun String?.isRvalueReferenceType(): Boolean = this?.trim()?.endsWith("&&") == true

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
    // Clang may print nullability as a postfix type qualifier even though it is not part of
    // source-level C++ type identity. Keeping it would split otherwise identical pointer nodes
    // (for example a string literal's `const char *` and a conversion edge's `_Nonnull` source).
    .replace(Regex("\\s+_(?:Nonnull|Nullable|Null_unspecified)\\b"), "")
    .removeSuffix("&&").removeSuffix("&").trim()
  type = type.replace(Regex("(?:\\s+(?:const|volatile))+\\s*$"), "").trim()
  if ('*' !in type)
    type = type.replace(Regex("^(?:(?:const|volatile)\\s+)+"), "").trim()
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

private enum class CppExactTypeIdKind {
  VALUE,
  POINTER,
  LVALUE_REFERENCE,
  RVALUE_REFERENCE,
  ARRAY
}

private data class CppExactTypeId(
  val terminals: List<String>,
  val kind: CppExactTypeIdKind,
  val isConst: Boolean,
  val isVolatile: Boolean,
  val baseIsConst: Boolean,
  val baseIsVolatile: Boolean,
  /** Normalized immediate element type after removing exactly the outer array suffix. */
  val arrayElementType: String? = null,
  /** Source spelling of the outer bound; null denotes an incomplete outer array. */
  val arrayBound: String? = null,
  val isIncompleteArray: Boolean? = null,
  /** Terminal range occupied by the base name, including an optional elaborated/typename prefix. */
  val baseStartTerminal: Int,
  val baseEndTerminal: Int,
  val baseSourceSpelling: String
)

private val CPP_TYPE_ID_CV = setOf("const", "volatile")
private val CPP_TYPE_ID_BUILTIN_WORDS = setOf(
  "bool", "char", "char8_t", "char16_t", "char32_t", "double", "float", "int",
  "long", "short", "signed", "unsigned", "void", "wchar_t"
)
private val CPP_TYPE_ID_ELABORATED = setOf("class", "struct", "enum", "union")

/**
 * Parses one deliberately conservative, expression-free C++ type-id. This is not a replacement
 * for Clang's parser: opaque identity still comes from Sema. Its job is to make an already-known
 * compiler spelling safe to splice between `<...>` or `( ... )` delimiters. In particular, it
 * rejects comments/gaps, top-level commas, calls, casts, operators, and unbalanced delimiters.
 */
private class CppExactTypeIdParser(
  private val spelling: String,
  private val allowReservedIdentifiers: Boolean = false
) {
  private val tokens = lexCppLine(spelling)

  fun parse(): CppExactTypeId? {
    if (spelling.isBlank() || spelling != spelling.trim() || '\n' in spelling || '\r' in spelling ||
      !allowReservedIdentifiers && spelling.containsReservedCppIdentifier() ||
      tokens.isEmpty() || !hasOnlyWhitespaceGaps()
    ) return null
    val parsed = parseTypeId(0, tokens.size) ?: return null
    return CppExactTypeId(
      terminals = tokens.map { token -> when (token.kind) {
        CppTokenKind.IDENTIFIER -> encodeIdentifier(token.text)
        CppTokenKind.INTEGER -> cppExactIntegerTerminal(token.text)
        else -> token.text
      } },
      kind = parsed.kind,
      isConst = parsed.isConst,
      isVolatile = parsed.isVolatile,
      baseIsConst = parsed.baseIsConst,
      baseIsVolatile = parsed.baseIsVolatile,
      arrayElementType = parsed.arrayElementType,
      arrayBound = parsed.arrayBound,
      isIncompleteArray = parsed.isIncompleteArray,
      baseStartTerminal = parsed.baseStartTerminal,
      baseEndTerminal = parsed.baseEndTerminal,
      baseSourceSpelling = spelling.substring(
        tokens[parsed.baseStartTerminal].start,
        tokens[parsed.baseEndTerminal - 1].end
      )
    )
  }

  private data class Shape(
    val kind: CppExactTypeIdKind,
    val isConst: Boolean,
    val isVolatile: Boolean,
    val baseIsConst: Boolean = isConst,
    val baseIsVolatile: Boolean = isVolatile,
    val arrayElementType: String? = null,
    val arrayBound: String? = null,
    val isIncompleteArray: Boolean? = null,
    val baseStartTerminal: Int = -1,
    val baseEndTerminal: Int = -1
  )

  private data class CvWords(val next: Int, val isConst: Boolean, val isVolatile: Boolean)
  private data class ArraySuffixes(
    val bounds: List<String?>,
    val firstCloseExclusive: Int
  )

  private fun hasOnlyWhitespaceGaps(): Boolean {
    var cursor = 0
    tokens.forEach { token ->
      if (token.start < cursor || spelling.substring(cursor, token.start).any { !it.isWhitespace() })
        return false
      cursor = token.end
    }
    return spelling.substring(cursor).all(Char::isWhitespace)
  }

  private fun cvWords(start: Int, end: Int): CvWords {
    var index = start
    var isConst = false
    var isVolatile = false
    while (index < end && tokens[index].text in CPP_TYPE_ID_CV) {
      if (tokens[index].text == "const") isConst = true else isVolatile = true
      index++
    }
    return CvWords(index, isConst, isVolatile)
  }

  private fun parseTypeId(start: Int, end: Int): Shape? {
    if (start >= end) return null
    val leadingCv = cvWords(start, end)
    var index = leadingCv.next
    val baseStart = index
    if (tokens.getOrNull(index)?.text == "typename") index++
    if (tokens.getOrNull(index)?.text in CPP_TYPE_ID_ELABORATED) index++
    if (index >= end) return null

    index = if (tokens[index].text in CPP_TYPE_ID_BUILTIN_WORDS) {
      val builtinStart = index
      while (index < end && tokens[index].text in CPP_TYPE_ID_BUILTIN_WORDS) index++
      val builtin = (builtinStart until index).joinToString(" ") { tokens[it].text }
      if (builtin != "void" && !builtin.isArithmeticCppType()) return null
      index
    } else parseQualifiedTypeName(index, end) ?: return null

    val trailingCv = cvWords(index, end)
    val baseEnd = index
    index = trailingCv.next
    val baseConst = leadingCv.isConst || trailingCv.isConst
    val baseVolatile = leadingCv.isVolatile || trailingCv.isVolatile
    if (index == end)
      return Shape(
        CppExactTypeIdKind.VALUE, baseConst, baseVolatile,
        baseStartTerminal = baseStart, baseEndTerminal = baseEnd
      )

    return parseAbstractDeclarator(
      index, end, baseConst, baseVolatile, baseStart, baseEnd
    )
  }

  private fun parseQualifiedTypeName(start: Int, end: Int): Int? {
    var index = start
    if (tokens[index].text == "::") index++
    var needSegment = true
    while (index < end) {
      if (!needSegment) break
      if (tokens[index].text == "template") index++
      if (index >= end || tokens[index].kind != CppTokenKind.IDENTIFIER) return null
      index++
      if (index < end && tokens[index].text == "<")
        index = parseTemplateArguments(index, end) ?: return null
      if (index < end && tokens[index].text == "::") {
        index++
        needSegment = true
      } else needSegment = false
    }
    return index.takeUnless { needSegment }
  }

  private fun parseTemplateArguments(open: Int, end: Int): Int? {
    var angleDepth = 1
    var squareDepth = 0
    var roundDepth = 0
    var argumentStart = open + 1
    var index = argumentStart
    while (index < end) {
      when (tokens[index].text) {
        "<" -> angleDepth++
        ">" -> {
          angleDepth--
          if (angleDepth == 0) {
            if (!parseTemplateArgument(argumentStart, index)) return null
            return index + 1
          }
          if (angleDepth < 0) return null
        }
        "[" -> squareDepth++
        "]" -> if (--squareDepth < 0) return null
        "(" -> roundDepth++
        ")" -> if (--roundDepth < 0) return null
        "," -> if (angleDepth == 1 && squareDepth == 0 && roundDepth == 0) {
          if (!parseTemplateArgument(argumentStart, index)) return null
          argumentStart = index + 1
        }
      }
      index++
    }
    return null
  }

  private fun parseTemplateArgument(start: Int, end: Int): Boolean =
    start < end && (end == start + 1 && tokens[start].kind == CppTokenKind.INTEGER ||
      parseTypeId(start, end) != null)

  private fun parseAbstractDeclarator(
    start: Int,
    end: Int,
    baseConst: Boolean,
    baseVolatile: Boolean,
    baseStartTerminal: Int,
    baseEndTerminal: Int
  ): Shape? {
    var index = start
    if (tokens[index].text == "(") {
      val close = (index + 1 until end).firstOrNull { tokens[it].text == ")" } ?: return null
      val (rawOperator, operatorEnd) = parsePointerOperators(index + 1, close) ?: return null
      if (operatorEnd != close) return null
      val operator = rawOperator ?: return null
      index = close + 1
      if (index == end || parseArraySuffixes(index, end) == null) return null
      return (if (operator.kind in setOf(
          CppExactTypeIdKind.LVALUE_REFERENCE, CppExactTypeIdKind.RVALUE_REFERENCE
        )) operator.copy(isConst = baseConst, isVolatile = baseVolatile)
      else operator).copy(baseIsConst = baseConst, baseIsVolatile = baseVolatile)
        .copy(
          baseStartTerminal = baseStartTerminal,
          baseEndTerminal = baseEndTerminal
        )
    }

    val operator = parsePointerOperators(index, end, allowEmpty = true) ?: return null
    index = operator.second
    val arrays = if (index < end) parseArraySuffixes(index, end) ?: return null else null
    if (arrays != null) {
      val elementOperator = operator.first
      if (elementOperator?.kind in setOf(
          CppExactTypeIdKind.LVALUE_REFERENCE, CppExactTypeIdKind.RVALUE_REFERENCE
        )) return null
      val elementPrefix = spelling.substring(0, tokens[index].start).trimEnd()
      val elementSuffix = spelling.substring(tokens[arrays.firstCloseExclusive - 1].end)
      val elementType = cppType(elementPrefix + elementSuffix) ?: return null
      return Shape(
        kind = CppExactTypeIdKind.ARRAY,
        isConst = elementOperator?.isConst ?: baseConst,
        isVolatile = elementOperator?.isVolatile ?: baseVolatile,
        baseIsConst = baseConst,
        baseIsVolatile = baseVolatile,
        arrayElementType = elementType,
        arrayBound = arrays.bounds.first(),
        isIncompleteArray = arrays.bounds.first() == null,
        baseStartTerminal = baseStartTerminal,
        baseEndTerminal = baseEndTerminal
      )
    }
    return operator.first?.let { shape ->
      if (shape.kind in setOf(
          CppExactTypeIdKind.LVALUE_REFERENCE, CppExactTypeIdKind.RVALUE_REFERENCE
        )) shape.copy(isConst = baseConst, isVolatile = baseVolatile)
      else shape
    }?.copy(
      baseIsConst = baseConst,
      baseIsVolatile = baseVolatile,
      baseStartTerminal = baseStartTerminal,
      baseEndTerminal = baseEndTerminal
    ) ?: return null
  }

  private fun parsePointerOperators(
    start: Int,
    end: Int,
    allowEmpty: Boolean = false
  ): Pair<Shape?, Int>? {
    var index = start
    var shape: Shape? = null
    while (index < end && tokens[index].text == "*") {
      index++
      val cv = cvWords(index, end)
      index = cv.next
      shape = Shape(CppExactTypeIdKind.POINTER, cv.isConst, cv.isVolatile)
    }
    if (index < end && tokens[index].text in setOf("&", "&&")) {
      if (shape != null) return null
      val kind = if (tokens[index].text == "&") CppExactTypeIdKind.LVALUE_REFERENCE
      else CppExactTypeIdKind.RVALUE_REFERENCE
      shape = Shape(kind, false, false)
      index++
    }
    if (shape == null && !allowEmpty) return null
    return shape to index
  }

  private fun parseArraySuffixes(start: Int, end: Int): ArraySuffixes? {
    var index = start
    val bounds = mutableListOf<String?>()
    var firstCloseExclusive = -1
    while (index < end) {
      if (tokens[index].text != "[") return null
      index++
      val bound = tokens.getOrNull(index)?.takeIf { it.kind == CppTokenKind.INTEGER }?.text
      if (bound != null) index++
      // Only an outermost array may omit its bound in a valid multidimensional type-id.
      if (bounds.isNotEmpty() && bound == null) return null
      if (index >= end || tokens[index].text != "]") return null
      index++
      bounds += bound
      if (firstCloseExclusive < 0) firstCloseExclusive = index
    }
    return bounds.takeIf(List<*>::isNotEmpty)?.let {
      ArraySuffixes(bounds, firstCloseExclusive)
    }
  }
}

private fun String.cppNameTokens(): List<String> = projectCppTokens(lexCppLine(this))

private fun String.containsReservedCppIdentifier(): Boolean =
  IDENTIFIER_REGEX.findAll(this).any { match ->
    val identifier = match.value
    "__" in identifier || identifier.length > 1 && identifier[0] == '_' && identifier[1].isUpperCase()
  }

private fun String.isCppQualifiedName(): Boolean = removePrefix("::").split("::")
  .let { it.isNotEmpty() && it.all(IDENTIFIER_REGEX::matches) }

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

private fun String.isIntegralOrBooleanCppType(): Boolean =
  typeShape() == "bool" || isIntegralCppType()

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
  mode: CppProjectionMode,
  observeExactLiterals: Boolean = false
): List<String> = when (mode) {
  CppProjectionMode.SEMANTIC -> projectCppTokens(
    tokens, preserveAdjacentGreater = true, observeExactLiterals = observeExactLiterals
  )
  // The pinned lexer exposes every `>>` as two Greater tokens. Preserve that lossless stream for
  // the generated parser: name-based template-depth guesses reject valid casts/operator calls.
  CppProjectionMode.SYNTAX -> projectCppTokens(
    tokens, preserveAdjacentGreater = true, observeExactLiterals = observeExactLiterals
  ).map { terminal ->
    if (terminal.startsWith("@id:")) CPP_SYNTAX_IDENTIFIER else terminal
  }
}

private fun projectCppPreparedTokens(
  tokens: List<CppToken>,
  observeExactLiterals: Boolean
): List<String> = projectCppTokens(
  tokens, preserveAdjacentGreater = true, observeExactLiterals = observeExactLiterals
)

fun projectCppTokens(tokens: List<CppToken>): List<String> =
  // Projection must commute with prefix concatenation for exact left quotients. Keeping the
  // lexer's two Greater tokens lossless avoids changing an already-conditioned prefix when its
  // next suffix token is another `>`; grammar producers spell right shift with the same pair.
  projectCppTokens(tokens, preserveAdjacentGreater = true, observeExactLiterals = false)

private fun projectCppTokens(
  tokens: List<CppToken>,
  preserveAdjacentGreater: Boolean,
  observeExactLiterals: Boolean = false
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
      observeExactLiterals && token.kind in CPP_EXACT_LITERAL_CLASS_BY_KIND ->
        add(cppObservedLiteralTerminal(token.kind, token.text))
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
  terminal.exactLiteral() != null -> requireNotNull(terminal.exactLiteral()).spelling
  terminal.observedLiteral() != null ->
    error("Observed C++ literal terminal reached sampling: $terminal")
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
    terminal.exactLiteral() != null -> add(requireNotNull(terminal.exactLiteral()).spelling)
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
