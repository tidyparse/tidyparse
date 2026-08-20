/** clang-format configuration shared by the browser worker and completion scratch documents. */
internal val CPP_CLANG_FORMAT_CONFIGURATION = """
  Language: Cpp
  BasedOnStyle: LLVM
  Standard: Latest
  ColumnLimit: 0
  IndentWidth: 4
  ContinuationIndentWidth: 4
  UseTab: Never
  TabWidth: 4
  PointerAlignment: Left
  ReferenceAlignment: Pointer
  DerivePointerAlignment: false
  AccessModifierOffset: -4
  EmptyLineBeforeAccessModifier: Never
  AllowShortBlocksOnASingleLine: Always
  AllowShortFunctionsOnASingleLine: All
  AllowShortIfStatementsOnASingleLine: AllIfsAndElse
  AllowShortLoopsOnASingleLine: true
  AllowShortCaseLabelsOnASingleLine: true
  AllowShortLambdasOnASingleLine: All
""".trimIndent() + "\n"

internal data class CppCompletionFormatBatch(
  val source: String,
  val leadingWhitespace: List<String>,
  val beginMarkers: List<String>,
  val endMarkers: List<String>
)

internal data class CppFormattedCompletion(
  val replacementText: String,
  val displayText: String
)

internal data class CppFormatPosition(val line: Int, val character: Int)

internal data class CppFormatTextEdit(
  val start: CppFormatPosition,
  val end: CppFormatPosition,
  val newText: String
)

/**
 * Places every candidate in its own function so one clangd formatting request can format the
 * complete popup. Markers make extraction independent of clang-format's chosen whitespace.
 */
internal fun cppCompletionFormatBatch(
  candidateTexts: List<String>,
  nonce: Int
): CppCompletionFormatBatch {
  require(candidateTexts.isNotEmpty()) { "A clang-format batch must contain a completion" }
  require(candidateTexts.all { '\n' !in it && '\r' !in it }) {
    "C++ statement completion candidates must occupy one physical line before formatting"
  }
  val leadingWhitespace = candidateTexts.map { text -> text.takeWhile { it == ' ' || it == '\t' } }
  val beginMarkers = candidateTexts.indices.map { index ->
    "// __tidyparse_completion_${nonce}_begin_$index"
  }
  val endMarkers = candidateTexts.indices.map { index ->
    "// __tidyparse_completion_${nonce}_end_$index"
  }
  val source = buildString {
    candidateTexts.forEachIndexed { index, candidate ->
      append("void __tidyparse_completion_")
      append(nonce)
      append('_')
      append(index)
      appendLine("() {")
      append("    ")
      appendLine(beginMarkers[index])
      append("    ")
      appendLine(candidate.drop(leadingWhitespace[index].length))
      append("    ")
      appendLine(endMarkers[index])
      appendLine("}")
    }
  }
  return CppCompletionFormatBatch(source, leadingWhitespace, beginMarkers, endMarkers)
}

/** Extracts clang-format's source spelling while restoring the statement's editor indentation. */
internal fun extractCppFormattedCompletions(
  batch: CppCompletionFormatBatch,
  formattedSource: String
): List<CppFormattedCompletion>? {
  val lines = formattedSource.lines()
  return batch.beginMarkers.indices.map { index ->
    val begin = lines.indexOfFirst { it.trim() == batch.beginMarkers[index] }
    val end = lines.indexOfFirst { it.trim() == batch.endMarkers[index] }
    if (begin < 0 || end <= begin + 1) return null
    val body = lines.subList(begin + 1, end)
      .dropWhile(String::isBlank)
      .dropLastWhile(String::isBlank)
    if (body.isEmpty()) return null
    val commonIndent = body.asSequence().filter(String::isNotBlank)
      .map { line -> line.indexOfFirst { !it.isWhitespace() }.coerceAtLeast(0) }
      .minOrNull() ?: 0
    val dedented = body.map { line ->
      if (line.isBlank()) "" else line.drop(commonIndent).trimEnd()
    }
    val placement = batch.leadingWhitespace[index]
    val replacement = dedented.joinToString("\n") { line -> placement + line }
    val display = dedented.asSequence().map(String::trim).filter(String::isNotEmpty)
      .joinToString(" ")
    if (display.isEmpty()) return null
    CppFormattedCompletion(replacement, display)
  }
}

/** Applies clangd's non-overlapping UTF-16 LSP edits to the scratch source. */
internal fun applyCppFormatTextEdits(
  source: String,
  edits: List<CppFormatTextEdit>
): String? {
  if (edits.isEmpty()) return source
  val lineStarts = buildList {
    add(0)
    source.forEachIndexed { index, character -> if (character == '\n') add(index + 1) }
  }
  fun offset(position: CppFormatPosition): Int? {
    if (position.line !in lineStarts.indices || position.character < 0) return null
    val start = lineStarts[position.line]
    val next = lineStarts.getOrNull(position.line + 1) ?: source.length
    var contentEnd = if (next > start && source[next - 1] == '\n') next - 1 else next
    if (contentEnd > start && source[contentEnd - 1] == '\r') contentEnd--
    return (start + position.character).takeIf { it <= contentEnd }
  }
  data class OffsetEdit(val start: Int, val end: Int, val newText: String)
  val resolved = edits.map { edit ->
    val start = offset(edit.start) ?: return null
    val end = offset(edit.end) ?: return null
    if (end < start) return null
    OffsetEdit(start, end, edit.newText)
  }.sortedWith(compareBy<OffsetEdit> { it.start }.thenBy { it.end })
  resolved.zipWithNext().forEach { (left, right) ->
    if (right.start < left.end || right.start == left.start && right.end == left.end) return null
  }
  return resolved.asReversed().fold(source) { text, edit ->
    text.replaceRange(edit.start, edit.end, edit.newText)
  }
}
