package cppcompletion

import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.js.jsTypeOf
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticTemplateEndpointTest {
  @Test
  fun primaryMemberTemplatesAndTypeViabilityRemainCorrelated(): Promise<Unit> =
    MainScope().promise {
      val source = """
        #include <map>
        #include <string>

        struct Pipeline {
          Pipeline() = default;
          explicit Pipeline(int);
        };
        struct Forward;

        int main() {
          std::map<int, std::string> records;
          std::string label;
          Pipeline pipeline;
          Forward *opaque = nullptr;
          pip
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { it.trim() == "pip" }
      val response = CppBrowserClangdClient().semanticResponse(
        source = source,
        line = line,
        character = lines[line].length,
        graphLimit = 4_096,
        graphDepth = 2,
        operationLimit = 1_024,
        operationDepth = 2
      )

      val operations = assertNotNull(response.operations)
      val templates = (operations.templates as Array<dynamic>).toList()
      val emplace = assertNotNull(templates.firstOrNull { schema ->
        schema.name == "emplace" &&
          ((schema.pattern.ownerType as? String)?.contains("map<") == true)
      })
      println(
        "SEMANTIC_TEMPLATE_RAW name=${emplace.name} role=${emplace.role} " +
          "pack=${emplace.hasFunctionParameterPack} " +
          "forwarding=${emplace.hasForwardingReferencePack} " +
          "allForwarding=${emplace.allParametersAreForwardingReferencePacks} " +
          "ownerComplete=${emplace.pattern.ownerTypeInfo.isComplete} " +
          "returnDependent=${emplace.pattern.returnTypeInfo.isDependent}"
      )
      assertEquals("member", emplace.role)
      assertEquals(true, emplace.requiresCompilerSubstitution)
      assertEquals(true, emplace.hasFunctionParameterPack)
      assertEquals(true, emplace.hasForwardingReferencePack)
      assertEquals(true, emplace.allParametersAreForwardingReferencePacks)
      assertEquals(true, emplace.pattern.ownerTypeInfo.isComplete)
      assertEquals(false, emplace.pattern.ownerTypeInfo.isDependent)
      assertEquals(false, emplace.pattern.returnTypeInfo.isDependent)

      val parameters = (emplace.pattern.parameters as Array<dynamic>).toList()
      val pack = assertNotNull(parameters.singleOrNull())
      assertEquals(true, pack.isPack)
      assertEquals(true, pack.isForwardingReference)
      assertTrue((pack.templateOccurrences as Array<dynamic>).isNotEmpty())

      val graph = assertNotNull(response.graph)
      val nodes = (graph.nodes as Array<dynamic>).toList()
      val pipeline = assertNotNull(nodes.firstOrNull { node ->
        node.name == "Pipeline" || node.qualifiedName == "Pipeline"
      })
      println(
        "SEMANTIC_TYPE_RAW pipelineComplete=${pipeline.typeInfo.isComplete} " +
          "pipelineDefaultConstructible=${pipeline.typeInfo.isDefaultConstructible}"
      )
      assertEquals(true, pipeline.typeInfo.isComplete)
      assertEquals(true, pipeline.typeInfo.isDefaultConstructible)

      val operationNodes = (operations.nodes as Array<dynamic>).toList()
      println(
        "SEMANTIC_PIPELINE_OPERATIONS " + operationNodes.filter { node ->
          node.ownerType == "::Pipeline"
        }.map { node ->
          val parameterTypes = (node.parameters as? Array<dynamic>)
            ?.map { parameter -> parameter.type as? String }
          "${node.role}:${node.name}$parameterTypes:${node.isExplicit}"
        }
      )
      val explicitPipelineConstructor = assertNotNull(operationNodes.firstOrNull { node ->
        node.role == "constructor" && node.ownerType == "::Pipeline" &&
          (node.parameters as Array<dynamic>).singleOrNull()?.type == "int"
      }, "explicit Pipeline(int) operation was truncated")
      assertEquals(true, explicitPipelineConstructor.isExplicit)

      val conversions = (operations.conversions as Array<dynamic>).toList()
      val cStringTargets = conversions.filter { edge ->
        edge.kind == "constructor" &&
          ((edge.fromType as? String)?.contains("const char *") == true)
      }.mapNotNull { it.toType as? String }
      println("SEMANTIC_CSTRING_TARGETS $cStringTargets")
      val cStringToString = assertNotNull(conversions.firstOrNull { edge ->
        edge.kind == "constructor" &&
          ((edge.fromType as? String)?.contains("const char *") == true) &&
          ((edge.toType as? String)?.let { to ->
            to == "::std::string" ||
              (to.startsWith("::std::basic_string<") && to.endsWith('>'))
          } == true)
      })
      val stringConstructor = assertNotNull(operationNodes.firstOrNull { node ->
        node.role == "constructor" &&
          ((node.ownerType as? String)?.let { owner ->
            owner == "::std::string" || owner.startsWith("::std::basic_string<")
          } == true) &&
          ((node.parameters as Array<dynamic>).firstOrNull()?.type as? String)
            ?.contains("const char *") == true
      })
      assertEquals(false, stringConstructor.isExplicit)

      val forward = assertNotNull(nodes.firstOrNull { node ->
        node.name == "Forward" || node.qualifiedName == "Forward"
      })
      assertEquals(false, forward.typeInfo.isComplete)
      assertEquals("undefined", jsTypeOf(forward.typeInfo.isDefaultConstructible))

      println(
        "SEMANTIC_TEMPLATE_ENDPOINT templates=${templates.size} " +
          "emplaceOwner=${emplace.pattern.ownerType} " +
          "returnType=${emplace.pattern.returnType} " +
          "pipelineComplete=${pipeline.typeInfo.isComplete} " +
          "pipelineDefaultConstructible=${pipeline.typeInfo.isDefaultConstructible} " +
          "explicitPipelineCtor=${explicitPipelineConstructor.isExplicit} " +
          "cstringConversion=${cStringToString.fromType}->${cStringToString.toType} " +
          "canonicalTarget=${cStringToString.canonicalToType} " +
          "forwardComplete=${forward.typeInfo.isComplete}"
      )
    }
}
