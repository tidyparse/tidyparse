package ai.hypergraph.tidyparse.template

import org.junit.Test
import java.io.File

class ExamplesTest : BaseTest() {
  @Test
  fun testAllExamplesWork() {
    File("../examples/").walkTopDown()
      .filter { it.isFile && it.extension == "tidy" }
      .forEach { println("Testing: ${it.path}"); it.readText().invokeOnAllLines() }
  }
}