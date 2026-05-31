package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.image.escapeHTML
import com.strumenta.antlrkotlin.parsers.generated.Python3Lexer
import org.antlr.v4.kotlinruntime.CharStreams
import org.antlr.v4.kotlinruntime.Token

data class PyCodeSnippet(val rawCode: String) {
  // Keep all tokens, including whitespace/comments. The hidden channel is included by default.
  val tokens: List<Token> =
    Python3Lexer(CharStreams.fromString(rawCode)).allTokens
      .filter { it.text?.isNotBlank() == true }
      .takeWhile { it.type != Token.EOF }
      .toList()

  /**
   * Returns just the ANTLR "names" of the lexed tokens, e.g. "NAME", "FOR", "IN", etc.
   * (matching the Python3Lexer vocabulary).
   */
  fun lexedTokens(): String = tokens.joinToString(" ") { Python3Lexer.VOCABULARY.getDisplayName(it.type) }
      .replace("'", "").replace("NEWLINE", "") + " NEWLINE"

  private enum class Paint { NONE, GREEN, ORANGE, GRAY }

  /**
   * Paints a Levenshtein-aligned patch onto the original code, highlighting:
   *  - inserted tokens in green,
   *  - deleted tokens as a gray 'blank' of the same length,
   *  - substituted tokens in orange,
   *  - identical tokens as plain text.
   *
   * The patch is given as a list of (oldTokenType?, newTokenType?) pairs.
   */
  fun paintDiff(levAlignedPatch: List<Pair<String?, String?>>, format: (String) -> String): String =
    buildTaggedString(levAlignedPatch).let { tagged ->
      renderTaggedString(tagged, format(tagged.joinToString(" ") { it.second }))
    }

  /**
   * Paints a Levenshtein-aligned patch onto the original code, highlighting:
   *  - inserted tokens in green,
   *  - deleted tokens as a gray 'blank' of the same length,
   *  - substituted tokens in orange,
   *  - identical tokens as plain text.
   *
   * The patch is given as a list of (oldTokenType?, newTokenType?) pairs.
   */
  suspend fun paintDiffAsync(
    levAlignedPatch: List<Pair<String?, String?>>,
    format: suspend (String) -> String
  ): String =
    buildTaggedString(levAlignedPatch).let { tagged ->
      renderTaggedString(tagged, format(tagged.joinToString(" ") { it.second }))
    }

  private fun buildTaggedString(levAlignedPatch: List<Pair<String?, String?>>): List<Pair<Paint, String>> {
    val taggedStr = mutableListOf<Pair<Paint, String>>()
    var indexInOriginal = 0

    for ((oldToken, newToken) in levAlignedPatch) {
      when {
        oldToken == null && newToken != null -> taggedStr.add(Paint.GREEN to newToken).also { indexInOriginal-- }
        oldToken != null && newToken == null -> taggedStr.add(Paint.GRAY to "")
        oldToken != null && newToken != null && oldToken != newToken -> taggedStr.add(Paint.ORANGE to newToken)
        else -> taggedStr.add(Paint.NONE to tokens[indexInOriginal].text!!)
      }
      indexInOriginal++
    }

    return taggedStr
  }

  private fun renderTaggedString(taggedStr: List<Pair<Paint, String>>, formattedCode: String): String {
    // This removes newlines, since the input and output are assumed to be a single line.
    val formattedString =
      formattedCode.replace(Regex("\\s+"), " ")
        .trim()

    // The formatter is expected to only adjust whitespaces between valid tokens. If it removes
    // non-whitespace tokens, fall back to emitting the aligned token with a trailing space.
    val sb = StringBuilder()
    var i = 0
    var ti = 0
    while (i < formattedString.length) {
      if (!formattedString[i].isWhitespace()) {
        while (ti < taggedStr.size && taggedStr[ti].first == Paint.GRAY) sb.append(paint(taggedStr[ti++]))
        if (ti >= taggedStr.size) break
        val ts = taggedStr[ti]
        if (ts.second.startsWith(formattedString[i])) {
          sb.append(paint(ts))
          i += ts.second.length
        } else sb.append(paint(ts.first to ts.second + " "))
        ti++
      } else sb.append(formattedString[i++])
    }
    while (ti < taggedStr.size) sb.append(paint(taggedStr[ti++]))

    return sb.toString()
  }

  private fun paint(ts: Pair<Paint, String>): String = when (ts.first) {
    Paint.GREEN -> """<span style="color: green">${ts.second.escapeHTML()}</span>"""
    Paint.GRAY -> """<span class="spacer"></span>"""
    Paint.ORANGE -> """<span style="color: orange">${ts.second.escapeHTML()}</span>"""
    Paint.NONE -> ts.second.escapeHTML()
  }
}
