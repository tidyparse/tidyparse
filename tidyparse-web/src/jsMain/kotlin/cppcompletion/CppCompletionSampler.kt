package cppcompletion

import ai.hypergraph.kaliningraph.parsing.BoundedLengthSample
import ai.hypergraph.kaliningraph.parsing.BoundedLengthSampleBatch
import com.ionspin.kotlin.bignum.integer.BigInteger
import kotlin.random.Random

data class CppCompletionSample(
  val tokens: List<String>,
  val freshNames: Set<String>,
  val length: Int = tokens.size
)

/** Maximum number of grammar completions published for one explicit editor request. */
const val CPP_MAX_INTERACTIVE_COMPLETIONS = 10

/**
 * Raw grammar yields may differ only by internal binder labels. Inspect a small multiple of the
 * display cap so alpha-equivalent derivations cannot crowd out source-distinct completions.
 */
private const val CPP_INTERACTIVE_DISCOVERY_MULTIPLIER = 4

/** A shortest suffix together with the text an editor should insert at the cursor. */
data class CppShortestCompletion(
  val tokens: List<String>,
  val insertionText: String,
  val freshNames: Set<String>,
  val length: Int = tokens.size
)

/** Materializes one cached, shortest-first Galoisenne batch and exposes its counted scope. */
class CppCompletionSampler(
  private val language: CppSuffixGrammar,
  identifiersInFile: Set<String>,
  private val random: Random = Random.Default
) {
  private val fresh = FreshCppNames(identifiersInFile, random)
  private var preparedLimit: Int? = null
  private var preparedBatch: BoundedLengthSampleBatch? = null
  private var materializedSamples: List<CppCompletionSample>? = null

  val inspectedDerivationCount: BigInteger
    get() = requireNotNull(preparedBatch) {
      "Prepare samples before reading their inspected derivation count"
    }
      .inspectedDerivationCount

  val inspectedLengths: IntRange
    get() = requireNotNull(preparedBatch) { "Prepare samples before reading their count range" }
      .inspectedLengths

  val coversFullBound: Boolean
    get() = requireNotNull(preparedBatch) { "Prepare samples before reading their count scope" }
      .coversFullBound

  fun prepare(count: Int = 100) {
    require(count >= 0)
    preparedLimit?.let { previous ->
      require(previous == count) {
        "A C++ completion sampler materializes one batch ($previous requested, then $count)"
      }
      return
    }
    val batch = language.bounded.shortestSampleBatch(
      random = random,
      sampleLimit = count,
      samplesPerLength = 10
    )
    val samples = materialize(batch.samples)
    preparedBatch = batch
    materializedSamples = samples
    preparedLimit = count
  }

  /** Returns source-distinct yields in increasing exact terminal length. */
  fun shortestDistinct(count: Int): List<CppCompletionSample> {
    require(count >= 0)
    if (count == 0) return emptyList()
    var batch = language.bounded.shortestDistinctSampleBatch(
      random = random,
      sampleLimit = count
    )
    fun structurallyDistinct() = batch.samples.asSequence()
      .filter { it.terminals.isNotEmpty() }
      .distinctBy { it.terminals.alphaNormalizedCppTerminals() }
      .take(count)
      .toList()
    var structurallyDistinct = structurallyDistinct()
    // Only widen discovery when the raw cap was actually saturated by alpha-equivalent forms.
    if (structurallyDistinct.size < count && batch.samples.size == count) {
      batch = language.bounded.shortestDistinctSampleBatch(
        random = random,
        sampleLimit = count * CPP_INTERACTIVE_DISCOVERY_MULTIPLIER
      )
      structurallyDistinct = structurallyDistinct()
    }
    return materialize(structurallyDistinct)
  }

  private fun materialize(
    sampledBatch: List<BoundedLengthSample>
  ): List<CppCompletionSample> = sampledBatch.map { sampled ->
    val emittedFresh = linkedSetOf<String>()
    val binders = mutableMapOf<String, String>()
    val projected = language.projectedPrefix + sampled.terminals
    val suffix = sampled.terminals.mapIndexed { suffixIndex, terminal ->
      val absoluteIndex = language.projectedPrefix.size + suffixIndex
      if (
        terminal == CPP_INTEGER && projected.getOrNull(absoluteIndex - 1) == "<" &&
        projected.getOrNull(absoluteIndex - 2) == "@id:get"
      ) "1" else materializeCppTerminal(terminal) {
        if (terminal.startsWith(CPP_BIND_PREFIX)) {
          binders.getOrPut(terminal) { fresh.next().also(emittedFresh::add) }
        } else {
          fresh.next().also(emittedFresh::add)
        }
      }
    }
    CppCompletionSample(suffix, emittedFresh, sampled.length)
  }

  fun sample(count: Int = 100): List<CppCompletionSample> {
    prepare(count)
    return requireNotNull(materializedSamples)
  }
}

/**
 * Draws source-distinct suffixes in increasing exact terminal length.
 *
 * Ambiguous derivations and alpha-renamed fresh binders have set semantics, so neither can consume
 * the display cap. If the globally shortest slice contains fewer than [limit] unique source forms,
 * subsequent exact lengths are inspected until the cap or the finite suffix horizon is reached.
 * An injected [random] keeps ordering and generated identifier spellings stable for a document
 * revision and cursor location.
 */
fun CppSuffixGrammar.shortestCompletions(
  prefixText: String,
  identifiersInFile: Set<String>,
  limit: Int = CPP_MAX_INTERACTIVE_COMPLETIONS,
  random: Random = Random.Default
): List<CppShortestCompletion> {
  require(limit >= 0) { "Interactive C++ completion limit must be nonnegative" }
  val sampleLimit = limit.coerceAtMost(CPP_MAX_INTERACTIVE_COMPLETIONS)
  if (sampleLimit == 0) return emptyList()

  val primary = shortestCompletionsFromThisGrammar(
    prefixText = prefixText,
    identifiersInFile = identifiersInFile,
    limit = sampleLimit,
    random = random
  )
  if (primary.size >= sampleLimit) return primary
  val fallback = completionFallback() ?: return primary
  val syntactic = fallback.shortestCompletionsFromThisGrammar(
    prefixText = prefixText,
    identifiersInFile = identifiersInFile,
    limit = sampleLimit,
    random = random
  )
  return (primary + syntactic).withIndex()
    .distinctBy { it.value.tokens }
    .sortedWith(compareBy<IndexedValue<CppShortestCompletion>> { it.value.length }.thenBy { it.index })
    .take(sampleLimit)
    .map(IndexedValue<CppShortestCompletion>::value)
}

private fun CppSuffixGrammar.shortestCompletionsFromThisGrammar(
  prefixText: String,
  identifiersInFile: Set<String>,
  limit: Int,
  random: Random
): List<CppShortestCompletion> {

  // Do not consult CppSuffixGrammar.isEmpty here: BoundedAcyclicCFG's semantic emptiness check
  // forces exact count vectors through the complete suffix horizon. The distinct sampler handles
  // a structurally empty grammar itself and retains the minimum-row fast path whenever that row
  // already supplies the requested result count.
  val samples = CppCompletionSampler(this, identifiersInFile, random)
    .shortestDistinct(limit)
  check(samples.zipWithNext().all { (left, right) -> left.length <= right.length }) {
    "Interactive C++ completions are not ordered by exact terminal length"
  }
  return samples.asSequence()
    .map { sample ->
      CppShortestCompletion(
        tokens = sample.tokens,
        insertionText = renderCppCompletionSuffix(prefixText, sample.tokens),
        freshNames = sample.freshNames,
        length = sample.length
      )
    }
    .toList()
}

/** Canonicalizes grammar-only fresh names while preserving their equality pattern. */
private fun List<String>.alphaNormalizedCppTerminals(): List<String> {
  val binders = linkedMapOf<String, String>()
  var nextAlpha = 0
  return map { terminal ->
    when {
      terminal == CPP_FRESH || terminal == CPP_SYNTAX_IDENTIFIER -> "@alpha:${nextAlpha++}"
      terminal.startsWith(CPP_BIND_PREFIX) -> binders.getOrPut(terminal) {
        "@alpha:${nextAlpha++}"
      }
      else -> terminal
    }
  }
}

/**
 * Renders materialized grammar terminals as valid C++ source.
 *
 * grammars-v4's C++ lexer exposes shifts as adjacent angle-bracket terminals. Separating every
 * terminal with whitespace would turn them into the ill-formed `< <` or `> >`; joining `> >` is
 * also valid for nested template closers in every language mode supported by this editor.
 */
fun List<String>.renderCppTokens(): String = buildString {
  var index = 0
  while (index < size) {
    val terminal = this@renderCppTokens[index]
    val next = this@renderCppTokens.getOrNull(index + 1)
    if ((terminal == "<" && next == "<") || (terminal == ">" && next == ">")) {
      append(terminal)
      append(next)
      index += 2
    } else {
      append(this@renderCppTokens[index++])
    }
    if (index < size) append(' ')
  }
}

/** Returns suffix-only insertion text while preserving a split shift across the cursor. */
fun renderCppCompletionSuffix(prefixText: String, suffixTokens: List<String>): String {
  if (suffixTokens.isEmpty()) return ""
  val trimmedPrefix = prefixText.trimEnd()
  val joinsSplitOperator =
    trimmedPrefix.endsWith('<') && suffixTokens.first() == "<" ||
      trimmedPrefix.endsWith('>') && suffixTokens.first() == ">"
  val joinsQualifiedOrPostfix = listOf("::", "->", ".", "(", "[", "<")
    .any(trimmedPrefix::endsWith)
  val alreadySeparated = prefixText.lastOrNull()?.isWhitespace() == true
  val separator = if (
    prefixText.isNotBlank() && !alreadySeparated && !joinsSplitOperator && !joinsQualifiedOrPostfix
  ) " " else ""
  return separator + suffixTokens.renderCppTokens()
}

private val CPP_DISPLAY_BINARY_OPERATORS = setOf(
  "=", "+", "-", "/", "%", "==", "!=", "<=", ">=", "<=>", "&&", "||", "^", "|",
  "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "?", ":"
)
private val CPP_DISPLAY_MEMBER_OPERATORS = setOf("::", ".", "->", ".*", "->*")
private val CPP_DISPLAY_CONTROL_KEYWORDS = setOf("if", "for", "while", "switch", "catch")
private val CPP_DISPLAY_CLOSING_PUNCTUATION = setOf(")", "]", "}", ",", ";")
private val CPP_DISPLAY_OPENING_PUNCTUATION = setOf("(", "[", "{")
private val CPP_DISPLAY_INCREMENT_OPERATORS = setOf("++", "--")
private val CPP_DISPLAY_PREFIX_OPERATORS = setOf("++", "--", "!", "~")
private val CPP_DISPLAY_DECLARATOR_OPERATORS = setOf("*", "&")

/**
 * Formats a completed statement for a compact suggestion label.
 *
 * This renderer is intentionally display-only. [renderCppCompletionSuffix] remains the canonical
 * insertion renderer: its conservative spaces prevent independently generated grammar terminals
 * from accidentally merging into a different C++ token. Labels operate on the already atomic
 * terminal sequence, so punctuation can be rendered in ordinary C++ style without that risk.
 */
fun formatCppCompletionLabel(
  prefixTokens: List<String>,
  suffixTokens: List<String>
): String {
  val terminals = buildList {
    val input = prefixTokens + suffixTokens
    var index = 0
    while (index < input.size) {
      when {
        input[index] == "<" && input.getOrNull(index + 1) == "<" -> {
          add("<<")
          index += 2
        }
        else -> add(input[index++])
      }
    }
  }
  return buildString {
    terminals.forEachIndexed { index, terminal ->
      val previous = terminals.getOrNull(index - 1)
      if (previous != null && cppCompletionLabelNeedsSpace(previous, terminal)) append(' ')
      append(terminal)
    }
  }
}

private fun cppCompletionLabelNeedsSpace(previous: String, terminal: String): Boolean = when {
  terminal in CPP_DISPLAY_CLOSING_PUNCTUATION -> false
  previous in CPP_DISPLAY_OPENING_PUNCTUATION -> false
  terminal in CPP_DISPLAY_MEMBER_OPERATORS || previous in CPP_DISPLAY_MEMBER_OPERATORS -> false
  terminal == "(" -> previous in CPP_DISPLAY_CONTROL_KEYWORDS
  terminal == "[" || terminal == "{" -> false
  terminal == "<" || terminal == ">" || previous == "<" -> false
  terminal in CPP_DISPLAY_INCREMENT_OPERATORS || previous in CPP_DISPLAY_PREFIX_OPERATORS -> false
  terminal in CPP_DISPLAY_DECLARATOR_OPERATORS -> false
  terminal in CPP_DISPLAY_BINARY_OPERATORS || previous in CPP_DISPLAY_BINARY_OPERATORS -> true
  else -> true
}

class FreshCppNames(
  identifiersInFile: Set<String>,
  private val random: Random = Random.Default
) {
  private val unavailable = identifiersInFile.toMutableSet()

  fun next(): String {
    while (true) {
      val candidate = buildString {
        append("freshId_")
        repeat(12) { append(('a'.code + random.nextInt(26)).toChar()) }
      }
      if (unavailable.add(candidate)) return candidate
    }
  }
}
