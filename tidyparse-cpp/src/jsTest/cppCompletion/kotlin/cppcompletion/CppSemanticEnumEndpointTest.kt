package cppcompletion

import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticEnumEndpointTest {
  @Test
  fun scopedAndUnscopedEnumeratorsKeepTheirCppLookupSpellings(): Promise<Unit> =
    MainScope().promise {
      val source = """
        namespace demo {
        enum class Tone { Warm, Cool };
        enum Plain { Alpha, Beta };
        }

        int main() {
          demo::Tone tone = demo::Tone::Warm;
          demo::Plain plain = demo::Alpha;
          to
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { it.trim() == "to" }
      val response = CppBrowserClangdClient().semanticResponse(
        source = source,
        line = line,
        character = lines[line].length,
        graphLimit = 1_024,
        graphDepth = 2,
        operationLimit = 1_024,
        operationDepth = 2
      )

      val operations = assertNotNull(response.operations)
      val enumerators = (operations.nodes as Array<dynamic>).toList()
        .filter { it.role == "enumerator" }
        .mapNotNull { it.name as? String }
        .toSet()

      assertTrue("demo::Tone::Warm" in enumerators)
      assertTrue("demo::Tone::Cool" in enumerators)
      assertTrue("demo::Alpha" in enumerators)
      assertTrue("demo::Beta" in enumerators)
      assertFalse("demo::Plain::Alpha" in enumerators)
      assertFalse("demo::Plain::Beta" in enumerators)
    }
}
