package cppcompletion

import kotlinx.browser.window
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticCallWitnessEndpointTest {
  @Test
  fun standardLibraryMemberTemplateWitnessBudgetIsObservableAndBounded(): Promise<Unit> =
    MainScope().promise {
      val source = """
        #include <map>
        #include <optional>
        #include <string>
        #include <tuple>

        int main() {
          using Record = std::tuple<int, std::string, double>;
          std::map<int, Record> records;
          std::optional<std::string> nickname;
          rec
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { it.trim() == "rec" }
      data class Case(
        val graphLimit: Int,
        val operationLimit: Int,
        val witnessLimit: Int
      )
      val cases = listOf(
        Case(512, 256, 0),
        Case(1_024, 512, 0),
        Case(4_096, 1_024, 0),
        Case(1_024, 512, 2),
        Case(1_024, 512, 4),
        Case(1_024, 512, 8),
        Case(1_024, 512, 16)
      )
      val client = CppBrowserClangdClient()
      var sawAuthoritativeWitness = false
      cases.forEach { case ->
        val started = window.performance.now()
        val response = client.semanticResponse(
          source = source,
          line = line,
          character = lines[line].length,
          graphLimit = case.graphLimit,
          graphDepth = 2,
          operationLimit = case.operationLimit,
          operationDepth = 2,
          callWitnessLimit = case.witnessLimit,
          callWitnessMaxArity = 4
        )
        val elapsed = window.performance.now() - started
        val graph = assertNotNull(response.graph)
        val operations = assertNotNull(response.operations)
        val graphNodes = (graph.nodes as Array<dynamic>).size
        val nodes = (operations.nodes as Array<dynamic>).size
        val templates = (operations.templates as Array<dynamic>).size
        val conversions = (operations.conversions as Array<dynamic>).size
        val witnesses = (operations.callWitnesses as Array<dynamic>).toList()
        val counts = witnesses.groupingBy {
          it.name as? String ?: "<anonymous>"
        }.eachCount()
        val arities = witnesses.groupingBy {
          (it.arguments as? Array<dynamic>)?.size ?: -1
        }.eachCount()
        val owners = witnesses.groupingBy {
          it.receiver?.type as? String ?: it.callable?.ownerType as? String ?: "<none>"
        }.eachCount()
        val probeCount = (operations.callWitnessProbeCount as? Number)?.toInt() ?: 0
        println(
          "SEMANTIC_ENDPOINT_MATRIX graphLimit=${case.graphLimit} " +
            "operationLimit=${case.operationLimit} witnessLimit=${case.witnessLimit} " +
            "elapsedMillis=$elapsed graph=$graphNodes/${graph.isIncomplete} " +
            "nodes=$nodes/${operations.nodeDiscoveryCount}/${operations.nodesIncomplete} " +
            "templates=$templates/${operations.templateDiscoveryCount}/" +
            "${operations.templatesIncomplete} conversions=$conversions/" +
            "${operations.conversionDiscoveryCount}/${operations.conversionsIncomplete} " +
            "probes=$probeCount witnesses=${witnesses.size}/" +
            "${operations.callWitnessDiscoveryCount}/" +
            "${operations.callWitnessesIncomplete} names=$counts arities=$arities owners=$owners"
        )

        assertTrue(witnesses.all { it.authoritative == true })
        assertTrue(witnesses.all {
          val owner = (it.receiver?.type as? String) ?:
            (it.callable?.ownerType as? String) ?: ""
          "::__" !in owner
        }, "reserved closure owners must not consume witness probes")
        assertTrue(witnesses.size <= case.witnessLimit)
        if (case.witnessLimit > 0) {
          // Member/construction and free calls reserve independent turns in a
          // shared prefix that scales to, but never exceeds, 64 Sema probes.
          assertTrue(probeCount <= minOf(case.witnessLimit * 2, 64))
          sawAuthoritativeWitness = sawAuthoritativeWitness || witnesses.isNotEmpty()
        }
      }
      assertTrue(sawAuthoritativeWitness)
    }

  @Test
  fun witnessProbeRoundsCannotBeMonopolizedByOnePrimary(): Promise<Unit> =
    MainScope().promise {
      val source = """
        template<class Tag>
        struct Sink {
          template<class... Args>
          int accept(Args&&...) { return sizeof...(Args); }
        };

        int main() {
          Sink<int> first;
          Sink<double> second;
          fir
        }
      """.trimIndent()
      val lines = source.lines()
      val line = lines.indexOfFirst { it.trim() == "fir" }
      val response = CppBrowserClangdClient().semanticResponse(
        source = source,
        line = line,
        character = lines[line].length,
        graphLimit = 256,
        graphDepth = 1,
        operationLimit = 128,
        operationDepth = 1,
        callWitnessLimit = 8,
        callWitnessMaxArity = 4
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.callWitnesses as Array<dynamic>).toList()
        .filter { it.name == "accept" }
      val receivers = witnesses.mapNotNull {
        it.receiver?.typeInfo?.valueCanonicalId as? String
      }.toSet()

      assertTrue(witnesses.any { (it.arguments as Array<dynamic>).size >= 2 })
      assertTrue(receivers.size >= 2, "one receiver primary monopolized the probe budget")
      assertTrue(witnesses.all {
        ((it.receiver?.type as? String) ?: "").contains("::__").not()
      })
      assertTrue(operations.callWitnessesIncomplete == true)
    }

  @Test
  fun dependentSentinelPrefixIsDeferredWithoutBlockingUsefulPackWitnesses(): Promise<Unit> =
    MainScope().promise {
      val source = """
        #include <type_traits>

        struct SecretTag {};

        template<class Value>
        struct Vessel {
          Vessel() = default;

          template<class Tag, class Fn, class... Args,
                   std::enable_if_t<std::is_same_v<Tag, SecretTag>, int> = 0>
          explicit Vessel(Tag, Fn&&, Args&&...) {}

          template<class... Args>
          int emplace(Args&&...) { return sizeof...(Args); }
        };

        int main() {
          Vessel<int> vessel;
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
        graphDepth = 2,
        operationLimit = 128,
        operationDepth = 2,
        callWitnessLimit = 4,
        callWitnessMaxArity = 2
      )
      val operations = assertNotNull(response.operations)
      val templates = (operations.templates as Array<dynamic>).toList()
      val witnesses = (operations.callWitnesses as Array<dynamic>).toList()
      val deferredConstructor = templates.firstOrNull { schema ->
        schema.name == "Vessel" && schema.role == "constructor" &&
          (schema.minExplicitArguments as? Number)?.toInt() == 2 &&
          (schema.maxExplicitArguments as? Number) == null
      }

      assertNotNull(deferredConstructor, "dependent sentinel constructor schema disappeared")
      assertTrue(witnesses.any { witness ->
        witness.name == "emplace" && witness.authoritative == true &&
          (((witness.receiver?.type as? String) ?: "").contains("Vessel<int>"))
      }, "safe emplace witness missing; names=${witnesses.map { it.name }}")
      assertTrue(
        witnesses.none { it.name == "Vessel" },
        "deferred sentinel constructor produced a witness"
      )
      assertTrue(
        operations.callWitnessesIncomplete == true,
        "deferred sentinel work must keep the witness set incomplete"
      )
      val probeCount = (operations.callWitnessProbeCount as? Number)?.toInt() ?: 0
      assertTrue(probeCount <= 8, "member-only sentinel probe count=$probeCount exceeds 8")
    }

  @Test
  fun riskySameNameMemberOverloadDoesNotPoisonProvablySafeGroup(): Promise<Unit> =
    MainScope().promise {
      val source = """
        #include <type_traits>

        struct Router {
          template<class T>
          int route(T&&) { return 1; }

          template<class T,
                   std::enable_if_t<std::is_pointer_v<T>, int> = 0>
          int route(T*) { return 2; }
        };

        int main() {
          Router router;
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
        graphDepth = 2,
        operationLimit = 128,
        operationDepth = 2,
        callWitnessLimit = 8,
        callWitnessMaxArity = 2
      )
      val operations = assertNotNull(response.operations)
      val schemas = (operations.templates as Array<dynamic>).toList()
        .filter { it.name == "route" }
      val riskyIds = schemas.filter { schema ->
        (schema.pattern.parameters as Array<dynamic>).any { parameter ->
          ((parameter.canonicalType as? String) ?: "").contains("*")
        }
      }.mapNotNull { it.pattern.id as? String }.toSet()
      assertTrue(riskyIds.isNotEmpty())

      val safe = assertNotNull(
        (operations.callWitnesses as Array<dynamic>).firstOrNull { witness ->
          witness.name == "route" && witness.syntax == "memberCall" &&
            (witness.arguments as Array<dynamic>).singleOrNull()?.kind == "integerZero"
        }
      )
      assertTrue(safe.authoritative == true)
      assertTrue((safe.primaryTemplateId as String) !in riskyIds)
      assertEquals(safe.primaryTemplateId as String, safe.callable.primaryTemplateId as String)
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 16)
    }

  @Test
  fun onlySourceRootedTypeClosureIsEligibleForWitnessProbes(): Promise<Unit> =
    MainScope().promise {
      suspend fun probe(source: String): Pair<List<dynamic>, List<dynamic>> {
        val lines = source.lines()
        val line = lines.indexOfFirst { it.trim() == "cursor" }
        val response = CppBrowserClangdClient().semanticResponse(
          source = source,
          line = line,
          character = lines[line].length,
          graphLimit = 256,
          graphDepth = 2,
          operationLimit = 128,
          operationDepth = 2,
          callWitnessLimit = 8,
          callWitnessMaxArity = 2
        )
        val operations = assertNotNull(response.operations)
        return Pair(
          (operations.templates as Array<dynamic>).toList(),
          (operations.callWitnesses as Array<dynamic>).toList()
        )
      }

      val declarations = """
        namespace library {
        template<class Tag>
        struct Box {
          template<class... Args>
          int accept(Args&&...) { return sizeof...(Args); }
        };
        }
        template struct library::Box<int>;
      """.trimIndent()
      val (graphOnlyTemplates, graphOnlyWitnesses) = probe(
        "$declarations\nint main() {\n  cursor\n}"
      )
      assertTrue(graphOnlyTemplates.any { schema ->
        schema.name == "accept" &&
          (((schema.pattern.ownerType as? String) ?: "").contains("Box<"))
      }, "the explicit type specialization must be present in graph closure")
      assertTrue(graphOnlyWitnesses.none { witness ->
        ((witness.receiver?.type as? String) ?: "").contains("Box<")
      }, "namespace-only graph types must not consume witness probes")

      val (_, sourceRootedWitnesses) = probe(
        "$declarations\nint main() {\n  library::Box<int> local;\n  cursor\n}"
      )
      assertTrue(sourceRootedWitnesses.any { witness ->
        witness.name == "accept" &&
          (((witness.receiver?.type as? String) ?: "").contains("Box<"))
      }, "a main-file value must promote its type closure to witness-eligible")
    }

  @Test
  fun inaccessibleNestedConstructorOwnersCannotBypassTypeLookup(): Promise<Unit> =
    MainScope().promise {
      val source = """
        class Vault {
          struct Secret {
            template<class T>
            Secret(T&&) {}
          };
          inline static Secret hidden{0};

        public:
          struct Visible {
            template<class T>
            Visible(T&&) {}
          };
        };

        int main() {
          Vault::Visible visible{0};
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
        graphDepth = 2,
        operationLimit = 256,
        operationDepth = 2,
        callWitnessLimit = 12,
        callWitnessMaxArity = 1
      )
      val operations = assertNotNull(response.operations)
      val constructions = (operations.callWitnesses as Array<dynamic>).toList()
        .filter { it.syntax == "parenConstruction" || it.syntax == "listConstruction" }

      assertTrue(constructions.any {
        ((it.callable?.ownerType as? String) ?: "").contains("Visible")
      }, "the public nested constructor control witness was lost")
      assertTrue(constructions.none {
        ((it.callable?.ownerType as? String) ?: "").contains("Secret")
      }, "a private nested construction target bypassed source type lookup")
    }

  @Test
  fun inaccessibleNestedMemberOwnersCannotBypassTypeLookup(): Promise<Unit> =
    MainScope().promise {
      val source = """
        class Vault {
          struct Secret {
            template<class T>
            T echo(T value) { return value; }
          };
          inline static Secret hidden{};

        public:
          struct Visible {
            template<class T>
            T echo(T value) { return value; }
          };
        };

        int main() {
          Vault::Visible visible;
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
        graphDepth = 2,
        operationLimit = 256,
        operationDepth = 2,
        callWitnessLimit = 16,
        callWitnessMaxArity = 1
      )
      val operations = assertNotNull(response.operations)
      val members = (operations.callWitnesses as Array<dynamic>).toList()
        .filter { it.syntax == "memberCall" && it.name == "echo" }

      assertTrue(members.any {
        ((it.callable?.ownerType as? String) ?: "").contains("Visible")
      }, "the public nested member-template control witness was lost")
      assertTrue(members.none {
        ((it.callable?.ownerType as? String) ?: "").contains("Secret")
      }, "a private nested member owner bypassed source type lookup")
    }
}
