import ai.hypergraph.kaliningraph.image.escapeHTML
import com.strumenta.antlrkotlin.parsers.generated.Python3Lexer
import org.antlr.v4.kotlinruntime.*

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
  fun lexedTokens(): String =
    tokens.joinToString(" ") { Python3Lexer.VOCABULARY.getDisplayName(it.type) }
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
  fun paintDiff(levAlignedPatch: List<Pair<String?, String?>>, format: (String) -> String): String {
// log("TOKENS: ${tokens.map { Python3Lexer.VOCABULARY.getDisplayName(it.type) }}")

    val taggedStr = mutableListOf<Pair<Paint, String>>()
    var indexInOriginal = 0

    for ((oldToken, newToken) in levAlignedPatch) {
      when {
        // (1) Insertions (oldToken == null)
        oldToken == null && newToken != null -> taggedStr.add(Paint.GREEN to newToken).also { indexInOriginal-- }
        // (2) Deletions (newToken == null)
        oldToken != null && newToken == null -> taggedStr.add(Paint.GRAY to "")
        // (2) Substitutions (oldToken != null && newToken != null && oldToken != newToken)
        oldToken != null && newToken != null && oldToken != newToken -> taggedStr.add(Paint.ORANGE to newToken)
        // (5) Match (oldToken == newToken)
        else -> taggedStr.add(Paint.NONE to tokens[indexInOriginal].text!!)
      }
      indexInOriginal++
    }

    val formattedString = format(taggedStr.joinToString(" ") { it.second }).replace("\n", " ").trim()

    val sb = StringBuilder(); var i = 0; var ti = 0
    while (i < formattedString.length)
      if (!formattedString[i].isWhitespace()) {
        while (ti < taggedStr.size && taggedStr[ti].first == Paint.GRAY) sb.append(paint(taggedStr[ti++]))
        if (ti >= taggedStr.size) break
        val ts = taggedStr[ti]
        sb.append(paint(ts))
        i += ts.second.length
        ti++
      } else sb.append(formattedString[i++])

    return sb.toString()
  }

  private fun paint(ts: Pair<Paint, String>): String {
    return when (ts.first) {
      Paint.GREEN -> """<span style="color: green">${ts.second.escapeHTML()}</span>"""
      Paint.GRAY -> """<span style="background-color: gray"><span class="noselect"> </span></span>"""
      Paint.ORANGE -> """<span style="color: orange">${ts.second.escapeHTML()}</span>"""
      Paint.NONE -> ts.second.escapeHTML()
    }
  }
}