import com.strumenta.antlrkotlin.parsers.generated.SimpleExprLexer
import com.strumenta.antlrkotlin.parsers.generated.SimpleExprParser
import org.antlr.v4.kotlinruntime.BaseErrorListener
import org.antlr.v4.kotlinruntime.CharStreams
import org.antlr.v4.kotlinruntime.CommonTokenStream
import org.antlr.v4.kotlinruntime.RecognitionException
import org.antlr.v4.kotlinruntime.Recognizer
import kotlin.test.Test
import kotlin.test.assertEquals

class TestANTLR {
  @Test
  fun testSimpleExprParser() {
    val input = "4 * (2 + 3)"
    val lexer = SimpleExprLexer(CharStreams.fromString(input))
    val tokens = CommonTokenStream(lexer)
    val parser = SimpleExprParser(tokens)

    parser.removeErrorListeners()
    parser.addErrorListener(object : BaseErrorListener() {
      override fun syntaxError(
        recognizer: Recognizer<*, *>,
        offendingSymbol: Any?,
        line: Int,
        charPositionInLine: Int,
        msg: String,
        e: RecognitionException?
      ) {
        throw RuntimeException("Syntax error at line $line:$charPositionInLine - $msg")
      }
    })

    val tree = parser.eval()

    assertEquals(20.0, tree.value)
  }
}