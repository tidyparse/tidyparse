package ai.hypergraph.tidyparse

import kotlin.test.Test
import kotlin.test.assertEquals

class CppLexerTest {
  @Test
  fun tokenizesQualifiedTemplatesPointersReferencesAndCalls() {
    assertEquals(
      listOf(
        "const", "std", "::", "vector", "<", "std", "::", "unique_ptr", "<", "Dog", ">", ">",
        "*", "animals", "=", "&", "owner", "->", "animals", ";"
      ),
      lexCppTokens("const std::vector<std::unique_ptr<Dog>>* animals = &owner->animals;")
    )
  }

  @Test
  fun tokenizesLiteralsOperatorsAndComments() {
    assertEquals(
      listOf("routes", ".", "push_back", "(", "Route", "{", "\"/dogs\"", ",", "'\\n'", ",", "42u", "}", ")", ";"),
      "routes.push_back(Route{\"/dogs\", '\\n', 42u}); // register it".cppLexicalTokens()
    )
  }

  @Test
  fun skipsWhitespaceNewlinesAndBlockComments() {
    assertEquals(
      listOf("value", "+=", "step", "*", "2", ";"),
      lexCppTokens("  value /* measured */ +=\n step * 2;  ")
    )
  }

  @Test
  fun neverExposesTheEofToken() {
    assertEquals(emptyList(), lexCppTokens(""))
    assertEquals(listOf("value"), lexCppTokens("value"))
  }

  @Test
  fun reportsAntlrRangesWithoutSearchingSkippedText() {
    assertEquals(
      listOf(
        CppLexicalToken("dog", "Identifier", 10, 13),
        CppLexicalToken("->", "Arrow", 13, 15),
        CppLexicalToken("dog", "Identifier", 15, 18),
        CppLexicalToken("(", "LeftParen", 18, 19),
        CppLexicalToken(")", "RightParen", 19, 20),
        CppLexicalToken(";", "Semi", 20, 21)
      ),
      lexCppTokenSpans("/* dog */ dog->dog();")
    )
  }
}
