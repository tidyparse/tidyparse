package cppcompletion

import cppCompletionContextFromDto
import cppEditorStatementSnapshot
import cppSemanticCompletionContextDto
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticExactGraphRouteEndpointTest {
  @Test
  fun graphPublishesOnlyDeclarationsFoundByItsExactQualifiedSpelling(): Promise<Unit> =
    MainScope().promise {
      val source = """
        namespace public_api {
        inline int public_value = 1;

        struct Container {
          static int hidden_value;
          static int hidden_call();
        };
        int Container::hidden_value = 2;
        int Container::hidden_call() { return hidden_value; }

        namespace nested {
        inline int deep_value = 3;
        }

        inline namespace abi {
        inline int inline_value = 4;
        }

        enum Plain { injected_value = 5 };

        extern "C" {
        inline int linked_value = 6;
        }
        }

        namespace public_alias = public_api;

        int main() {
          cursor
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { it.trim() == "cursor" }
      val response = CppBrowserClangdClient().semanticResponse(
        source = source,
        line = line,
        character = lines[line].length,
        graphLimit = 512,
        graphDepth = 4,
        operationLimit = 128,
        operationDepth = 1,
        callWitnessLimit = 0,
        callWitnessMaxArity = 0
      )
      val graph = assertNotNull(response.graph)
      val names = (graph.nodes as Array<dynamic>).mapNotNull {
        it.name as? String
      }.toSet()

      assertTrue("public_api::public_value" in names)
      assertTrue("public_api::nested::deep_value" in names)
      assertTrue("public_api::inline_value" in names)
      assertTrue("public_api::injected_value" in names)
      assertTrue("public_api::linked_value" in names)
      assertTrue("public_alias::public_value" in names)
      assertTrue("public_alias::inline_value" in names)
      assertTrue("public_alias::injected_value" in names)

      assertTrue("public_api::hidden_value" !in names)
      assertTrue("public_api::hidden_call" !in names)
      assertTrue("public_api::deep_value" !in names)
      assertTrue("public_alias::hidden_value" !in names)
      assertTrue("public_alias::hidden_call" !in names)
      assertTrue("public_alias::deep_value" !in names)

      val snapshot = assertNotNull(cppEditorStatementSnapshot(source, line, lines[line].length))
      val context = cppCompletionContextFromDto(
        cppSemanticCompletionContextDto(response, snapshot)
      )
      assertTrue(context.values.any { it.name == "public_api::public_value" })
      assertTrue(context.completions.any { it.name == "public_api::public_value" })
      listOf("public_api::hidden_value", "public_api::hidden_call").forEach { invalidRoute ->
        assertTrue(context.values.none { it.name == invalidRoute })
        assertTrue(context.functions.none { it.name == invalidRoute })
        assertTrue(context.completions.none { it.name == invalidRoute })
      }
    }
}
