package cppcompletion

import kotlin.random.Random
import kotlin.time.TimeSource

/** [prefixText]/[prefix] stop before the optional caret-token [tokenPrefix]. */
data class CppCompletionQuery(
  val prefix: List<CppToken>,
  val prefixText: String,
  val identifiersInFile: Set<String>,
  val tokenPrefix: CppToken? = null,
  val limit: Int = CPP_MAX_INTERACTIVE_COMPLETIONS,
  val seed: Int = 0
) {
  init {
    require('\n' !in prefixText && '\r' !in prefixText) {
      "A C++ completion query must contain one physical statement prefix"
    }
    require(tokenPrefix?.text?.isNotEmpty() != false) { "A C++ token prefix cannot be empty" }
    require(limit in 1..CPP_MAX_INTERACTIVE_COMPLETIONS) {
      "Interactive C++ completion limit must be in 1..$CPP_MAX_INTERACTIVE_COMPLETIONS"
    }
  }
}

/** Formatter-ready completion generated without any JavaScript transport types. */
data class CppEditorCompletion(
  val candidateText: String,
  val tokenLength: Int,
  val tokens: List<String>,
  val freshNames: Set<String>
)

/** One deterministic completion result plus phase timings used by the worker diagnostics. */
data class CppCompletionExecution(
  val suggestions: List<CppEditorCompletion>,
  val minimumTokenLength: Int?,
  val generationMillis: Int,
  val samplingMillis: Int
)

/**
 * Runs the exact production quotient, shortest-distinct sampling and lexical serialization.
 *
 * Browser transport and prepared-grammar caching deliberately stay outside this function. Tests
 * can therefore exercise the same completion behavior as the worker without constructing dynamic
 * DTOs, while the worker can continue reusing its small LRU of prepared grammars.
 */
fun PreparedCppCompletionGrammar.completeCppStatement(
  query: CppCompletionQuery
): CppCompletionExecution {
  val generationClock = TimeSource.Monotonic.markNow()
  val suffixGrammar = generate(query.prefix)
  val generationMillis = generationClock.elapsedNow().inWholeMilliseconds.toInt()

  val samplingClock = TimeSource.Monotonic.markNow()
  val identifiers = buildSet {
    addAll(query.identifiersInFile)
    query.prefix.mapTo(this) { it.text }
    query.tokenPrefix?.let { add(it.text) }
  }
  val samples = suffixGrammar.shortestCompletions(
    prefixText = query.prefixText,
    identifiersInFile = identifiers,
    tokenPrefix = query.tokenPrefix,
    limit = query.limit,
    random = Random(query.seed)
  )
  val suggestions = samples.map { sample ->
    CppEditorCompletion(
      candidateText = query.prefixText + sample.insertionText,
      tokenLength = sample.length,
      tokens = sample.tokens,
      freshNames = sample.freshNames
    )
  }
  return CppCompletionExecution(
    suggestions = suggestions,
    minimumTokenLength = samples.firstOrNull()?.length,
    generationMillis = generationMillis,
    samplingMillis = samplingClock.elapsedNow().inWholeMilliseconds.toInt()
  )
}

/** Convenience entry point for uncached callers such as browser-parity tests. */
fun CppCompletionGrammar.completeCppStatement(
  context: CppCompletionContext,
  query: CppCompletionQuery
): CppCompletionExecution = prepare(context, query.prefix).completeCppStatement(query)
