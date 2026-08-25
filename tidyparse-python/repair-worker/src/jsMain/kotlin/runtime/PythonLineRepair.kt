import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.contains
import ai.hypergraph.kaliningraph.parsing.language
import ai.hypergraph.kaliningraph.parsing.levenshteinAlign
import ai.hypergraph.kaliningraph.repair.LED_BUFFER
import ai.hypergraph.kaliningraph.repair.TIMEOUT_MS
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.tidyparse.PyCodeSnippet
import ai.hypergraph.tidyparse.sampleGREUntilTimeout
import ai.hypergraph.tidyparse.wgpu.*

private const val DEFAULT_LED_BUFFER = 2
private const val CPU_TIMEOUT_MS = 1_000

val pythonRepairGrammar: CFG by lazy { pythonStatementCNFAllProds }

data class PythonLineRepairResult(
  val repairMode: Boolean,
  val repairs: List<String>
)

suspend fun repairPythonLine(line: String, maxResults: Int? = null): PythonLineRepairResult {
  require('\n' !in line && '\r' !in line) { "Syntax repair accepts exactly one physical line" }
  val currentLine = line.trim()
  if (currentLine.isBlank() || currentLine.startsWith('#')) {
    return PythonLineRepairResult(repairMode = false, repairs = emptyList())
  }

  val snippet = PyCodeSnippet(currentLine)
  val tokens = snippet.lexedTokens()
    .tokenizeByWhitespace()
    .map { if (it == "|") "OR" else it }

  // Manual completion is allowed on every line; keep the expensive repair path line-local and
  // return immediately when the shared Python statement grammar already accepts the input.
  if (tokens in pythonRepairGrammar.language) {
    return PythonLineRepairResult(repairMode = false, repairs = emptyList())
  }

  val candidates = if (gpuAvailable) {
    repairCode(
      cfg = pythonRepairGrammar,
      code = tokens,
      ledBuffer = DEFAULT_LED_BUFFER,
      rerankerQuery = neuralRerankerQuery(tokens),
      reranker = RepairReranker::rerankOrOriginal
    ).pythonRepairTokens()
  } else {
    LED_BUFFER = DEFAULT_LED_BUFFER
    TIMEOUT_MS = CPU_TIMEOUT_MS
    sampleGREUntilTimeout(tokens, pythonRepairGrammar)
      .map(String::toPythonRepairTokens)
      .distinct()
  }

  val restitched = candidates
    .map { repairTokens ->
      snippet.restitch(
        levenshteinAlign(tokens.dropLast(1), repairTokens.tokenizeByWhitespace())
      )
    }
    .distinct()
  val repairs = if (maxResults == null) restitched.toList() else restitched.take(maxResults).toList()
  return PythonLineRepairResult(repairMode = true, repairs = repairs)
}

private fun IntersectionResults.pythonRepairTokens(): Sequence<String> =
  mapTerminals(String::toPythonTerminal)
    .asSequence()
    .map(String::withoutFinalNewline)
    .distinct()

private fun String.toPythonRepairTokens(): String =
  withoutFinalNewline()
    .tokenizeByWhitespace()
    .joinToString(" ", transform = String::toPythonTerminal)

private fun String.withoutFinalNewline(): String =
  if (this == "NEWLINE") "" else removeSuffix(" NEWLINE")

private fun String.toPythonTerminal(): String = when (this) {
  "OR" -> "|"
  "not_in" -> "not in"
  "is_not" -> "is not"
  else -> this
}
