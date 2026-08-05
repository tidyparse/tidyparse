package ai.hypergraph.tidyparse

import com.strumenta.antlrkotlin.parsers.generated.CPP14Lexer
import org.antlr.v4.kotlinruntime.CharStreams
import org.antlr.v4.kotlinruntime.Token

/** A C++ token and its exact half-open source range. */
data class CppLexicalToken(
  val text: String,
  val type: String,
  val startIndex: Int,
  val endIndexExclusive: Int
)

/** Lexes C++ source while retaining the ranges reported by the ANTLR token stream. */
fun lexCppTokenSpans(source: String): List<CppLexicalToken> {
  val lexer = CPP14Lexer(CharStreams.fromString(source))
  return lexer.allTokens
    .takeWhile { it.type != Token.EOF }
    .mapNotNull { token ->
      token.text?.let { text ->
        CppLexicalToken(
          text = text,
          type = lexer.vocabulary.getSymbolicName(token.type)
            ?: lexer.vocabulary.getDisplayName(token.type),
          startIndex = token.startIndex,
          endIndexExclusive = token.stopIndex + 1
        )
      }
    }
    .toList()
}

/** Splits C++ source into the lexical token texts defined by grammars-v4's C++14 lexer. */
fun lexCppTokens(source: String): List<String> = lexCppTokenSpans(source).map(CppLexicalToken::text)

/** Convenient form for completion code that already has a source line. */
fun String.cppLexicalTokens(): List<String> = lexCppTokens(this)
