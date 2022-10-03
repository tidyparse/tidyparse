package ai.hypergraph.tidyparse.template

import org.junit.Test
import java.io.File

class MyPluginTest : BaseTest() {
  @Test
  fun testAllExamplesWork() {
    File("examples/").walkTopDown()
      .filter { it.isFile && it.extension == "tidy" }
      .forEach { it.readText().testAllLines() }
  }
}
