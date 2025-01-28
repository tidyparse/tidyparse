import ai.hypergraph.kaliningraph.image.escapeHTML
import com.strumenta.antlrkotlin.parsers.generated.Python3Lexer
import org.antlr.v4.kotlinruntime.CharStreams
import org.antlr.v4.kotlinruntime.Token

data class PyCodeSnippet(val rawCode: String) {
  // Keep all tokens, including whitespace/comments. The hidden channel is included by default.
  val tokens: List<Token> =
    Python3Lexer(CharStreams.fromString(rawCode)).run {
      allTokens
        .filter { it.text?.isNotBlank() == true }
        .takeWhile { it.type != Token.EOF }
        .toList()
    }

//  val essentialTokens: List<Token> =
//    Python3Lexer(CharStreams.fromString(rawCode)).run {
//      allTokens
//        .filter { it.text?.isNotBlank() == true }
//        .takeWhile { it.type != Token.EOF }
//        .toList()
//    }

  /**
   * Returns just the ANTLR "names" of the lexed tokens, e.g. "NAME", "FOR", "IN", etc.
   * (matching the Python3Lexer vocabulary).
   */
  fun lexedTokens(): String =
   tokens.joinToString(" ") { Python3Lexer.VOCABULARY.getDisplayName(it.type) }
      .replace("'", "").replace("NEWLINE", "") + " NEWLINE"

  /**
   * Paints a Levenshtein-aligned patch onto the original code, highlighting:
   *  - inserted tokens in green,
   *  - deleted tokens as a gray 'blank' of the same length,
   *  - substituted tokens in orange,
   *  - identical tokens as plain text.
   *
   * The patch is given as a list of (oldTokenType?, newTokenType?) pairs.
   */
  fun paintDiff(levAlignedPatch: List<Pair<String?, String?>>): String {
    println("TOKENS: ${tokens.map { Python3Lexer.VOCABULARY.getDisplayName(it.type) }}")
    val sb = StringBuilder()
    var indexInOriginal = 0

    for ((oldToken, newToken) in levAlignedPatch) {
      when {
        // (1) Insertions (oldToken == null)
        // Paint the new token in green
        oldToken == null && newToken != null -> {
          sb.append(
            """ <span style="color: green">${newToken.escapeHTML()}</span> """
          )
        }

        // (2) Deletions (newToken == null)
        // Show a gray “blank” of the same length as the original token
        oldToken != null && newToken == null -> {
          if (indexInOriginal < tokens.size) {
            val blank = " "
            sb.append("""<span style="background-color: gray"><span class="noselect">$blank</span></span>""")
            indexInOriginal++
          }
        }

        // (4) Substitution (oldToken != null && newToken != null && oldToken != newToken)
        oldToken != null && newToken != null && oldToken != newToken -> {
          if (indexInOriginal < tokens.size) {
            // Paint the new token in orange
            sb.append(
              """ <span style="color: orange">${newToken.escapeHTML()}</span> """
            )
            indexInOriginal++
          }
        }

        // (5) Match (oldToken == newToken)
        else -> {
          // Output the exact original text, uncolored
          if (indexInOriginal < tokens.size) {
            val tk = if (Python3Lexer.VOCABULARY.getDisplayName(tokens[indexInOriginal].type) == "NEWLINE") " "
            else " " + tokens[indexInOriginal].text!!.escapeHTML() + " "
            sb.append(tk)
            indexInOriginal++
          }
        }
      }
    }

    // Append any leftover original tokens if the patch ended early
    while (indexInOriginal < tokens.size) {
      sb.append(tokens[indexInOriginal].text!!.escapeHTML())
      indexInOriginal++
    }

    return sb.toString().replace(Regex("\\s+"), " ").trim()
  }
}