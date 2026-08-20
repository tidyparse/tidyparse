package cppcompletion

import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticOperationRootEndpointTest {
  @Test
  fun fieldNodesReportExactBitFieldObjectKind(): Promise<Unit> =
    MainScope().promise {
      val source = """
        struct State {
          unsigned int bits : 1;
          long value;
        };

        int main() {
          State state;
          cursor
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { it.trim() == "cursor" }
      val response = CppBrowserClangdClient().semanticResponse(
        source = source,
        line = line,
        character = lines[line].length,
        graphLimit = 256,
        graphDepth = 1,
        operationLimit = 128,
        operationDepth = 1,
        callWitnessLimit = 0,
        callWitnessMaxArity = 0
      )
      val operations = assertNotNull(response.operations)
      val fields = (operations.nodes as Array<dynamic>).toList()
        .filter { it.role == "member" && (it.name == "bits" || it.name == "value") }
        .associateBy { it.name as String }

      assertEquals(true, assertNotNull(fields["bits"]).isBitField as Boolean)
      assertEquals(false, assertNotNull(fields["value"]).isBitField as Boolean)
    }

  @Test
  fun graphAliasAndValueAtomsSeedTheirConcreteOperationOwners(): Promise<Unit> =
    MainScope().promise {
      val source = """
        #include <iostream>
        #include <string>

        int main() {
          st
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { it.trim() == "st" }
      val response = CppBrowserClangdClient().semanticResponse(
        source = source,
        line = line,
        character = lines[line].length,
        graphLimit = 1_024,
        graphDepth = 2,
        operationLimit = 512,
        operationDepth = 2,
        callWitnessLimit = 0,
        callWitnessMaxArity = 0
      )
      val graph = assertNotNull(response.graph)
      val graphNodes = (graph.nodes as Array<dynamic>).toList()
      val operations = assertNotNull(response.operations)
      val operationNodes = (operations.nodes as Array<dynamic>).toList()
      val conversions = (operations.conversions as Array<dynamic>).toList()
      val stdAtoms = graphNodes.mapNotNull {
        val name = it.name as? String
        val qualified = it.qualifiedName as? String
        if (name?.contains("string") == true || name?.contains("cout") == true ||
          qualified?.contains("string") == true || qualified?.contains("cout") == true
        ) "$name => $qualified" else null
      }
      println(
        "SEMANTIC_OPERATION_ROOT graph=${graphNodes.size}/${graph.isIncomplete} " +
          "stdAtoms=$stdAtoms " +
          "nodes=${operationNodes.size}/${operations.nodeDiscoveryCount} " +
          "conversions=${conversions.size}/${operations.conversionDiscoveryCount} " +
          "incomplete=${operations.nodesIncomplete}/" +
          "${operations.templatesIncomplete}/${operations.conversionsIncomplete}"
      )

      val stringAlias = assertNotNull(graphNodes.firstOrNull {
        it.qualifiedName == "std::string"
      }, "std::string graph alias was truncated")
      val stringTypeId = assertNotNull(
        stringAlias.typeInfo?.valueCanonicalId as? String
      )
      val stringOwnerNodes = operationNodes.filter {
        it.ownerTypeInfo?.valueCanonicalId == stringTypeId
      }.map {
        val parameters = (it.parameters as? Array<dynamic>)
          ?.map { parameter -> parameter.type as? String }
        "${it.role}:${it.name}$parameters"
      }
      val stringConversions = conversions.filter {
        it.toTypeInfo?.valueCanonicalId == stringTypeId
      }.map { "${it.kind}:${it.fromType}->${it.toType}" }
      println(
        "SEMANTIC_OPERATION_STRING owners=$stringOwnerNodes " +
          "conversions=$stringConversions canonical=$stringTypeId"
      )
      assertTrue(conversions.any {
        it.toTypeInfo?.valueCanonicalId == stringTypeId &&
          ((it.fromType as? String)?.contains("char") == true)
      }, "std::string concrete constructor conversion was not discovered")

      val cout = assertNotNull(graphNodes.firstOrNull {
        it.qualifiedName == "std::cout"
      }, "std::cout graph value was truncated")
      val coutTypeId = assertNotNull(cout.typeInfo?.valueCanonicalId as? String)
      assertTrue(operationNodes.any {
        it.role == "member" &&
          it.ownerTypeInfo?.valueCanonicalId == coutTypeId
      }, "std::cout concrete owner members were not discovered")

      assertTrue((operations.nodeDiscoveryCount as Number).toInt() >= operationNodes.size)
      assertTrue(
        (operations.conversionDiscoveryCount as Number).toInt() >= conversions.size
      )
    }
}
