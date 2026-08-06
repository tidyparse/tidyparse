package cppcompletion

import ai.hypergraph.kaliningraph.parsing.BoundedAcyclicCFG
import ai.hypergraph.kaliningraph.parsing.BoundedLengthSample
import ai.hypergraph.kaliningraph.parsing.BoundedLengthSampleBatch
import ai.hypergraph.kaliningraph.parsing.PreindexedAcyclicCFG
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
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
  private val random: Random = Random.Default,
  private val tokenPrefix: CppToken? = null
) {
  // Syntax forests constrain the lexical category; this final intersection selects an exact
  // Sema-backed identifier spelling (or keyword/literal spelling) inside that category.
  private val bounded = tokenPrefix?.let(language.bounded::startingWith) ?: language.bounded
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
    val batch = bounded.shortestSampleBatch(
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
    var batch = bounded.shortestDistinctSampleBatch(
      random = random,
      sampleLimit = count
    )
    fun structurallyDistinct(limit: Int) = batch.samples.asSequence()
      .filter { it.terminals.isNotEmpty() }
      .distinctBy { sample -> sample.terminals.cppCompletionShape(tokenPrefix) }
      .take(limit)
      .toList()
    var distinct = structurallyDistinct(count)
    // A small bounded widening prevents aliases or alpha-equivalent derivations from consuming
    // slots intended for distinct source completions.
    if (distinct.size < count && batch.samples.size == count) {
      batch = bounded.shortestDistinctSampleBatch(
        random = random,
        sampleLimit = count * CPP_INTERACTIVE_DISCOVERY_MULTIPLIER
      )
      distinct = structurallyDistinct(count * CPP_INTERACTIVE_DISCOVERY_MULTIPLIER)
    }
    return materialize(distinct).distinctBy(CppCompletionSample::tokens).take(count)
  }

  private fun materialize(
    sampledBatch: List<BoundedLengthSample>
  ): List<CppCompletionSample> = sampledBatch.map { sampled ->
    val emittedFresh = linkedSetOf<String>()
    val binders = mutableMapOf<String, String>()
    val suffix = sampled.terminals.mapIndexed { suffixIndex, terminal ->
      val prefixSpelling = tokenPrefix?.takeIf { suffixIndex == 0 }
        ?.let { cppCompletionTerminalSpelling(terminal, it) }
      if (prefixSpelling != null) {
        if (terminal.startsWith(CPP_BIND_PREFIX)) binders[terminal] = prefixSpelling
        prefixSpelling
      } else materializeCppTerminal(terminal) {
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
  random: Random = Random.Default,
  tokenPrefix: CppToken? = null
): List<CppShortestCompletion> {
  require(limit >= 0) { "Interactive C++ completion limit must be nonnegative" }
  val sampleLimit = limit.coerceAtMost(CPP_MAX_INTERACTIVE_COMPLETIONS)
  if (sampleLimit == 0) return emptyList()

  val primary = shortestCompletionsFromThisGrammar(
    prefixText = prefixText,
    identifiersInFile = identifiersInFile,
    tokenPrefix = tokenPrefix,
    limit = sampleLimit,
    random = random
  )
  // The generated statement syntax is a totality/recognition floor, not semantic evidence.
  // Publishing its untyped derivations can reinterpret committed tokens or place a value where
  // C++ requires a type. Only the Sema-specialized grammar is safe to expose in the editor.
  return primary
}

private fun CppSuffixGrammar.shortestCompletionsFromThisGrammar(
  prefixText: String,
  identifiersInFile: Set<String>,
  tokenPrefix: CppToken?,
  limit: Int,
  random: Random
): List<CppShortestCompletion> {

  // Do not consult CppSuffixGrammar.isEmpty here: BoundedAcyclicCFG's semantic emptiness check
  // forces exact count vectors through the complete suffix horizon. The distinct sampler handles
  // a structurally empty grammar itself and retains the minimum-row fast path whenever that row
  // already supplies the requested result count.
  val samples = CppCompletionSampler(this, identifiersInFile, random, tokenPrefix)
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
    .distinctBy(CppShortestCompletion::insertionText)
    .toList()
}

/** Exact intersection with the regular language whose first terminal extends [prefix]. */
private fun BoundedAcyclicCFG.startingWith(prefix: CppToken): BoundedAcyclicCFG {
  val indexed = grammar as? PreindexedAcyclicCFG
  val grouped = indexed?.let { null } ?: grammar.groupBy { it.first }
  val nonterminals = indexed?.acyclicNonterminalIndex?.keys ?: grouped!!.keys
  fun rules(symbol: String) = indexed?.productionsFor(symbol) ?: grouped!![symbol].orEmpty()
  if (startSymbol !in nonterminals) return this
  val sourceOrder = indexed?.acyclicCountingOrder

  fun fixedPoint(accepts: (List<String>, Set<String>) -> Boolean): Set<String> {
    val result = linkedSetOf<String>()
    if (sourceOrder != null) sourceOrder.forEach { symbol ->
      if (rules(symbol).any { accepts(it.second, result) }) result += symbol
    } else {
      var changed: Boolean
      do {
        val before = result.size
        nonterminals.forEach { symbol ->
          if (symbol !in result && rules(symbol).any { accepts(it.second, result) }) result += symbol
        }
        changed = result.size != before
      } while (changed)
    }
    return result
  }

  val nullable = fixedPoint { rhs, known -> rhs.all(known::contains) }
  val matches = mutableMapOf<String, Boolean>()
  val productive = fixedPoint { rhs, known -> when (rhs.size) {
    0 -> false
    1 -> if (rhs[0] in nonterminals) rhs[0] in known
      else matches.getOrPut(rhs[0]) { cppCompletionTerminalSpelling(rhs[0], prefix) != null }
    else -> rhs[0] in known || rhs[0] in nullable && rhs[1] in known
  } }
  if (startSymbol !in productive)
    return emptySet<Pair<String, List<String>>>().boundedAcyclic(maxLength)

  val matchedNames = mutableMapOf<String, String>()
  fun matched(symbol: String) = matchedNames.getOrPut(symbol) { "$symbol\u0000CPP_FIRST" }
  val matchedQueue = mutableListOf(startSymbol)
  val matchedSeen = linkedSetOf(startSymbol)
  val originalQueue = mutableListOf<String>()
  val originalSeen = linkedSetOf<String>()
  val constrained = linkedSetOf<Pair<String, List<String>>>()
  fun includeMatched(symbol: String) {
    if (symbol in productive && matchedSeen.add(symbol)) matchedQueue += symbol
  }
  fun includeOriginal(symbol: String) {
    if (originalSeen.add(symbol)) originalQueue += symbol
  }

  var next = 0
  while (next < matchedQueue.size) {
    val symbol = matchedQueue[next++]
    rules(symbol).forEach { (_, rhs) -> when (rhs.size) {
      1 -> if (rhs[0] in nonterminals) {
        if (rhs[0] in productive) {
          constrained += matched(symbol) to listOf(matched(rhs[0]))
          includeMatched(rhs[0])
        }
      } else if (matches.getOrPut(rhs[0]) {
          cppCompletionTerminalSpelling(rhs[0], prefix) != null
        }) constrained += matched(symbol) to rhs
      2 -> {
        if (rhs[0] in productive) {
          constrained += matched(symbol) to listOf(matched(rhs[0]), rhs[1])
          includeMatched(rhs[0]); includeOriginal(rhs[1])
        }
        if (rhs[0] in nullable && rhs[1] in productive) {
          constrained += matched(symbol) to listOf(matched(rhs[1]))
          includeMatched(rhs[1])
        }
      }
    } }
  }
  next = 0
  while (next < originalQueue.size) {
    val symbol = originalQueue[next++]
    rules(symbol).forEach { production ->
      constrained += production
      production.second.filter(nonterminals::contains).forEach(::includeOriginal)
    }
  }
  val countingOrder = sourceOrder?.let { order ->
    order.filter(originalSeen::contains) + order.filter(matchedSeen::contains).map(::matched)
  }
  return constrained.boundedAcyclic(
    maxLength = maxLength,
    startSymbol = matched(startSymbol),
    countingOrder = countingOrder
  )
}

/** Canonicalizes grammar-only fresh names while preserving their equality pattern. */
private fun List<String>.alphaNormalizedCppTerminals(): List<String> {
  val binders = linkedMapOf<String, String>()
  var nextAlpha = 0
  return map { terminal ->
    when {
      terminal == CPP_FRESH -> "@alpha:${nextAlpha++}"
      terminal.startsWith(CPP_BIND_PREFIX) -> binders.getOrPut(terminal) {
        "@alpha:${nextAlpha++}"
      }
      else -> terminal
    }
  }
}

private fun List<String>.cppCompletionShape(prefix: CppToken?): List<String> =
  alphaNormalizedCppTerminals().toMutableList().also { shape ->
    prefix?.let { cppCompletionTerminalSpelling(first(), it) }?.let { shape[0] = it }
  }

private const val CPP_LEXICAL_SEPARATOR_CACHE_SIZE = 512

/**
 * Whether two generated terminals need whitespace to retain their intended lexical identities.
 *
 * This is a source codec, not a formatter: it compares the syntax projection with and without a
 * separator and only retains whitespace when concatenation would create a different C++ token.
 * Split angle terminals are the grammar's lossless spelling for shifts and nested template closes.
 */
private val cppLexicalSeparatorCache = linkedMapOf<String, Boolean>()

private fun needsCppLexicalSeparator(left: String, right: String): Boolean {
  if (left.isEmpty() || right.isEmpty()) return false
  val key = "$left\u0000$right"
  cppLexicalSeparatorCache[key]?.let { cached ->
    cppLexicalSeparatorCache.remove(key)
    cppLexicalSeparatorCache[key] = cached
    return cached
  }
  val needsSeparator = when {
    // The syntax grammar deliberately represents both shifts and adjacent template closes this way.
    left == "<" && right == "<" || left == ">" && right == ">" -> false
    // An unterminated block comment is recovered as punctuation by the lexer, so guard it directly.
    left == "/" && (right == "/" || right == "*") -> true
    // Three separately generated dots must not silently become one ellipsis preprocessing token.
    left.endsWith('.') && right.startsWith('.') -> true
    else -> {
      val separated = cppLines("$left $right").single().tokens
      val adjacent = cppLines(left + right).single().tokens
      projectCppCompletionTokens(separated, CppProjectionMode.SYNTAX) !=
        projectCppCompletionTokens(adjacent, CppProjectionMode.SYNTAX)
    }
  }
  cppLexicalSeparatorCache[key] = needsSeparator
  while (cppLexicalSeparatorCache.size > CPP_LEXICAL_SEPARATOR_CACHE_SIZE)
    cppLexicalSeparatorCache.remove(cppLexicalSeparatorCache.keys.first())
  return needsSeparator
}

/** Serializes grammar terminals with only the whitespace required to preserve their token stream. */
fun List<String>.renderCppTokens(): String = buildString {
  this@renderCppTokens.forEachIndexed { index, terminal ->
    val previous = this@renderCppTokens.getOrNull(index - 1)
    if (previous != null && needsCppLexicalSeparator(previous, terminal)) append(' ')
    append(terminal)
  }
}

/**
 * Serializes a generated terminal suffix without making style decisions.
 *
 * clang-format owns the final spelling. This function only prevents the first generated terminal
 * from merging with the last token already present before the caret.
 */
fun renderCppCompletionSuffix(prefixText: String, suffixTokens: List<String>): String {
  if (suffixTokens.isEmpty()) return ""
  val alreadySeparated = prefixText.lastOrNull()?.isWhitespace() == true
  val lastPrefixToken = if (alreadySeparated) null
  else cppLines(prefixText).single().tokens.lastOrNull()
    ?.takeIf { token -> token.end == prefixText.length }
    ?.text
  val separator = if (
    lastPrefixToken != null && needsCppLexicalSeparator(lastPrefixToken, suffixTokens.first())
  ) " " else ""
  return separator + suffixTokens.renderCppTokens()
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
