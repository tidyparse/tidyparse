private data class RawAdmissibleRepair(
  val body: String,
  val line: String
)

/**
 * Filters ranked whole-line repairs without coupling the policy to Monaco or Wasm wrappers.
 *
 * All distinct raw insertions are classified before formatting begins. Every semantic survivor is
 * then formatted and the exact displayed insertion is classified again when Ruff changes it.
 */
internal fun semanticallyAdmissibleRepairs(
  candidates: List<String>,
  originalLine: String,
  completionLimit: Int,
  isCurrent: () -> Boolean,
  sourceWithLine: (String) -> String,
  isSemanticallyAdmissible: (String) -> Boolean,
  formatCandidate: (String) -> String
): List<String> {
  require(completionLimit >= 0) { "completionLimit must not be negative" }

  val rawSurvivors = ArrayList<RawAdmissibleRepair>()
  val seenRawSources = linkedSetOf<String>()

  for (candidate in candidates) {
    if (!isCurrent()) return emptyList()
    val rawCandidate = candidate.trim()
    if (
      rawCandidate.isBlank() ||
      '\n' in rawCandidate ||
      '\r' in rawCandidate
    ) continue

    val rawLine = preserveRepairIndentation(originalLine, rawCandidate)
    if (rawLine == originalLine) continue
    val rawSource = sourceWithLine(rawLine)
    if (!seenRawSources.add(rawSource)) continue
    if (isSemanticallyAdmissible(rawSource)) {
      rawSurvivors += RawAdmissibleRepair(rawCandidate, rawLine)
    }
  }

  if (!isCurrent()) return emptyList()

  val accepted = ArrayList<String>(rawSurvivors.size)
  val seenRendered = linkedSetOf<String>()
  for (survivor in rawSurvivors) {
    if (!isCurrent()) return emptyList()
    val formattedCandidate = formatCandidate(survivor.body)
    if (!isCurrent()) return emptyList()

    val formattedLine = preserveRepairIndentation(originalLine, formattedCandidate)
    val renderedLine = if (
      formattedCandidate.isBlank() ||
      '\n' in formattedCandidate ||
      '\r' in formattedCandidate ||
      formattedLine == originalLine ||
      formattedLine == survivor.line
    ) {
      survivor.line
    } else {
      val formattedSource = sourceWithLine(formattedLine)
      if (isSemanticallyAdmissible(formattedSource)) formattedLine else survivor.line
    }

    if (!isCurrent()) return emptyList()
    if (seenRendered.add(renderedLine)) accepted += renderedLine
  }

  return if (isCurrent()) accepted.take(completionLimit) else emptyList()
}

internal fun preserveRepairIndentation(originalLine: String, candidate: String): String =
  originalLine.takeWhile { it.isWhitespace() } + candidate.trimStart()