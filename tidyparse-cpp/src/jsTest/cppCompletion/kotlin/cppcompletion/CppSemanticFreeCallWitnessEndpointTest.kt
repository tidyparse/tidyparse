package cppcompletion

import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticFreeCallWitnessEndpointTest {
  private suspend fun query(
    source: String,
    witnessLimit: Int = 8,
    maxArity: Int = 3
  ): dynamic {
    val lines = source.lines()
    val line = lines.indexOfFirst { it.trim() == "cursor" }
    require(line >= 0)
    return CppBrowserClangdClient().semanticResponse(
      source = source,
      line = line,
      character = lines[line].length,
      graphLimit = 512,
      graphDepth = 3,
      operationLimit = 256,
      operationDepth = 2,
      callWitnessLimit = witnessLimit,
      callWitnessMaxArity = maxArity
    )
  }

  private fun freeWitnesses(response: dynamic): List<dynamic> {
    val operations = assertNotNull(response.operations)
    return (operations.callWitnesses as Array<dynamic>).toList()
      .filter { it.syntax == "freeCall" }
  }

  private fun assertAuthoritativeIdentity(witness: dynamic) {
    assertTrue(witness.authoritative == true)
    assertEquals("recursiveDefinitionInstantiation", witness.validation as String)
    assertEquals(witness.primaryTemplateId as String, witness.targetId as String)
    assertEquals(witness.primaryTemplateId as String, witness.callable.primaryTemplateId as String)
    assertTrue((witness.primaryTemplateId as String).isNotEmpty())
    assertTrue(witness.receiver == null)
    assertNotNull(witness.explicitTemplateArguments as? Array<dynamic>)
  }

  private fun assertOrdinaryAuthoritativeIdentity(witness: dynamic) {
    assertTrue(witness.authoritative == true)
    assertEquals("freeCall", witness.syntax as String)
    assertEquals("semaCallExpression", witness.validation as String)
    val targetId = witness.targetId as String
    assertTrue(targetId.isNotEmpty())
    assertEquals("", witness.primaryTemplateId as String)
    assertEquals(targetId, witness.callable.id as String)
    assertEquals("", (witness.callable.primaryTemplateId as? String) ?: "")
    assertTrue(witness.receiver == null)
    assertTrue((witness.explicitTemplateArguments as Array<dynamic>).isEmpty())
  }

  @Test
  fun deducedQualifiedFreeCallPreservesItsLookupRouteAndExactArguments(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          namespace api {
          template<class T>
          T identity(T value) { return value; }
          }

          int main() {
            int local = 1;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val witness = assertNotNull(
        freeWitnesses(response).firstOrNull { candidate ->
          candidate.name == "::api::identity" &&
            (candidate.arguments as Array<dynamic>).singleOrNull()?.kind ==
              "integerZero"
        }
      )
      assertAuthoritativeIdentity(witness)
      assertTrue((witness.explicitTemplateArguments as Array<dynamic>).isEmpty())
      val arguments = witness.arguments as Array<dynamic>
      assertEquals(1, arguments.size)
      assertEquals("integerZero", arguments[0].kind as String)
      assertEquals("0", arguments[0].spelling as String)
      assertEquals("prvalue", arguments[0].valueCategory as String)
      assertEquals("ordinary", arguments[0].objectKind as String)
    }

  @Test
  fun explicitFactoryReachesArityTwoWithoutDependingOnFailedLowerArities(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          struct Widget {
            Widget() = delete;
            Widget(int) = delete;
            Widget(int, int) {}
          };

          template<class T, class... Args>
          T build(Args&&... args) { return T(args...); }

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 2
      )
      val operations = assertNotNull(response.operations)
      val witness = assertNotNull(freeWitnesses(response).firstOrNull { candidate ->
        candidate.name == "build" &&
          (candidate.arguments as Array<dynamic>).size == 2 &&
          (candidate.explicitTemplateArguments as Array<dynamic>).let { explicit ->
            explicit.size == 1 &&
              ((explicit[0].type.type as? String) ?: "").contains("Widget")
          }
      })
      assertAuthoritativeIdentity(witness)
      assertTrue((witness.arguments as Array<dynamic>).all {
        it.kind == "integerZero" && it.spelling == "0" && it.objectKind == "ordinary"
      })
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 16)
    }

  @Test
  fun knownSpecializationTypeVectorsAndArrayShapeMetadataArePreserved(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          struct Alpha {};
          struct Beta {};
          using IncompleteBuffer = const int[];
          using BoundedBuffer = volatile int[7];

          template<class A, class B>
          int ordered_types() { return 1; }
          template int ordered_types<Alpha, Beta>();

          template<class T>
          int incomplete_shape() { return sizeof(T*); }
          template int incomplete_shape<IncompleteBuffer>();

          template<class T>
          int bounded_shape() { return sizeof(T*); }
          template int bounded_shape<BoundedBuffer>();

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 12,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response)
      val ordered = assertNotNull(witnesses.firstOrNull { it.name == "ordered_types" })
      assertAuthoritativeIdentity(ordered)
      val orderedTypes = (ordered.explicitTemplateArguments as Array<dynamic>).map {
        it.kind as String to (it.type.type as String)
      }
      assertEquals(2, orderedTypes.size)
      assertEquals("type", orderedTypes[0].first)
      assertEquals("type", orderedTypes[1].first)
      assertTrue(orderedTypes[0].second.contains("Alpha"))
      assertTrue(orderedTypes[1].second.contains("Beta"))

      val incomplete = assertNotNull(
        witnesses.firstOrNull { it.name == "incomplete_shape" }
      )
      val incompleteInfo =
        (incomplete.explicitTemplateArguments as Array<dynamic>).single().type.typeInfo
      val incompleteSummary =
        "kind=${incompleteInfo.kind} const=${incompleteInfo.isConst} " +
          "volatile=${incompleteInfo.isVolatile} element=${incompleteInfo.elementCanonicalId} " +
          "elementConst=${incompleteInfo.elementIsConst} " +
          "elementVolatile=${incompleteInfo.elementIsVolatile} " +
          "incomplete=${incompleteInfo.isIncompleteArray} bound=${incompleteInfo.arrayBound}"
      assertEquals("array", incompleteInfo.kind as String, incompleteSummary)
      assertTrue(
        (incompleteInfo.elementCanonicalId as String).isNotEmpty(),
        incompleteSummary
      )
      assertTrue(incompleteInfo.elementIsConst == true, incompleteSummary)
      assertTrue(incompleteInfo.elementIsVolatile == false, incompleteSummary)
      assertTrue(incompleteInfo.isIncompleteArray == true, incompleteSummary)

      val bounded = assertNotNull(witnesses.firstOrNull { it.name == "bounded_shape" })
      val boundedInfo =
        (bounded.explicitTemplateArguments as Array<dynamic>).single().type.typeInfo
      val boundedSummary =
        "kind=${boundedInfo.kind} const=${boundedInfo.isConst} " +
          "volatile=${boundedInfo.isVolatile} element=${boundedInfo.elementCanonicalId} " +
          "elementConst=${boundedInfo.elementIsConst} " +
          "elementVolatile=${boundedInfo.elementIsVolatile} " +
          "incomplete=${boundedInfo.isIncompleteArray} bound=${boundedInfo.arrayBound}"
      assertEquals("array", boundedInfo.kind as String, boundedSummary)
      assertTrue((boundedInfo.elementCanonicalId as String).isNotEmpty(), boundedSummary)
      assertTrue(boundedInfo.elementIsConst == false, boundedSummary)
      assertTrue(boundedInfo.elementIsVolatile == true, boundedSummary)
      assertTrue(boundedInfo.isIncompleteArray == false, boundedSummary)
      assertEquals("7", boundedInfo.arrayBound as String, boundedSummary)
    }

  @Test
  fun unsupportedEnumNttpIsIncompleteAndNeverBecomesAFakeIntegerWitness(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          enum class Token : int { first = 1 };

          template<Token Value>
          int unsupported_enum(int value) { return value; }

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 2
      )
      val operations = assertNotNull(response.operations)
      assertTrue(freeWitnesses(response).none { it.name == "unsupported_enum" })
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 16)
    }

  @Test
  fun explicitTypeArgumentsMustBeNameableAtTheCompletionScope(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          template<class T>
          int measured_size() { return sizeof(T); }

          class Vault {
            struct Secret {};
          public:
            static int prime() { return measured_size<Secret>(); }
          };

          int main() {
            {
              struct Expired {};
              using Gone = Expired;
              (void)measured_size<Gone>();
            }
            cursor
          }
        """.trimIndent(),
        witnessLimit = 12,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response).filter { it.name == "measured_size" }
      assertTrue(witnesses.isNotEmpty(), "accessible baseline type seeds were lost")
      assertTrue(witnesses.none { witness ->
        (witness.explicitTemplateArguments as Array<dynamic>).any { argument ->
          val spelling = argument.type?.type as? String ?: ""
          spelling == "Gone" || spelling == "Expired"
            || spelling.contains("Vault::Secret")
        }
      }, "an expired/private known specialization bypassed final type lookup")
    }

  @Test
  fun hybridTypeAndIntegralArgumentsRoundTripInDeclaredOrder(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          using size_type = decltype(sizeof(0));

          struct IndexedValue {
            int value;
            IndexedValue(int input) : value(input) {}
          };

          template<class T, size_type I>
          T indexed_get(int input) {
            static_assert(I == 1, "only the indexed specialization is valid");
            return T(input);
          }
          template IndexedValue indexed_get<IndexedValue, 1>(int);

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val operations = assertNotNull(response.operations)
      val indexed = assertNotNull(freeWitnesses(response).firstOrNull { candidate ->
        if (candidate.name != "indexed_get") return@firstOrNull false
        val explicit = candidate.explicitTemplateArguments as Array<dynamic>
        explicit.size == 2 && explicit[0].kind == "type" &&
          ((explicit[0].type.type as? String) ?: "").contains("IndexedValue") &&
          explicit[1].kind == "exactIntegerLiteral" &&
          explicit[1].canonicalValue == "1"
      })
      assertAuthoritativeIdentity(indexed)
      val explicit = indexed.explicitTemplateArguments as Array<dynamic>
      assertEquals("type", explicit[0].kind as String)
      assertEquals("exactIntegerLiteral", explicit[1].kind as String)
      assertEquals("1", explicit[1].spelling as String)
      assertEquals("1", explicit[1].canonicalValue as String)
      assertEquals("builtin", explicit[1].type.typeInfo.kind as String)
      assertTrue(explicit[1].type.typeInfo.isSourceSpellable == true)
      assertTrue((explicit[1].type.typeInfo.canonicalId as String).isNotEmpty())
      assertTrue(freeWitnesses(response).none { candidate ->
        if (candidate.name != "indexed_get") return@none false
        (candidate.explicitTemplateArguments as Array<dynamic>).any {
          it.kind == "exactIntegerLiteral" && it.canonicalValue == "0"
        }
      }, "recursive validation must reject the mismatched integral specialization")
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 16)
    }

  @Test
  fun riskySameNameOverloadCannotPoisonAProvablySafeFreeCallGroup(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          #include <type_traits>

          template<class T>
          int route(T&&) { return 1; }

          template<class T,
                   std::enable_if_t<std::is_pointer_v<T>, int> = 0>
          int route(T*) { return 2; }

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 2
      )
      val operations = assertNotNull(response.operations)
      val safe = assertNotNull(freeWitnesses(response).firstOrNull { candidate ->
        candidate.name == "route" &&
          (candidate.arguments as Array<dynamic>).singleOrNull()?.kind == "integerZero"
      })
      assertAuthoritativeIdentity(safe)
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 16)
    }

  @Test
  fun definitionErrorsCannotAuthenticateAnOtherwiseViableExplicitCall(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          template<class T>
          int valid_body() { return sizeof(T); }

          template<class T>
          int invalid_body() {
            static_assert(sizeof(T) == 0, "instantiated body must be rejected");
            return 0;
          }

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val operations = assertNotNull(response.operations)
      val witnesses = freeWitnesses(response)
      assertTrue(witnesses.any { it.name == "valid_body" })
      assertTrue(
        witnesses.none { it.name == "invalid_body" },
        "a selected specialization with a failed instantiated body was emitted"
      )
      assertTrue(operations.callWitnessesIncomplete == true)
    }

  @Test
  fun unevaluatedTemplateProbeCannotAuthenticateAnImmediateInvocation(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          template<class T>
          consteval T immediate_identity(T value) { return value; }

          template<class T>
          T ordinary_identity(T value) { return value; }

          int main() {
            int runtime = 1;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response)
      assertTrue(witnesses.any { it.name == "ordinary_identity" })
      assertTrue(
        witnesses.none { it.name == "immediate_identity" },
        "an unevaluated probe published a consteval call without proving an immediate invocation"
      )
    }

  @Test
  fun rootedAndCursorAuthenticatedUsingRoutesShareBoundedFairLanes(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          namespace library {
          template<class T>
          T echo(T value) { return value; }
          }
          namespace library_alias = library;
          namespace facade {
          using library::echo;
          }

          template<class... Args>
          int alpha(Args&&...) { return sizeof...(Args); }
          template<class... Args>
          int beta(Args&&...) { return sizeof...(Args); }

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 12,
        maxArity = 2
      )
      val operations = assertNotNull(response.operations)
      val witnesses = freeWitnesses(response)
      assertTrue(witnesses.any { it.name == "::library::echo" })
      assertTrue(witnesses.any { it.name == "::facade::echo" })
      val exactRelativeRoutes = witnesses.filter {
        (it.name as String) in setOf(
          "library::echo", "library_alias::echo", "facade::echo"
        )
      }
      assertTrue(
        exactRelativeRoutes.isNotEmpty(),
        "exact lookup-authenticated relative routes were discarded"
      )
      exactRelativeRoutes.forEach(::assertAuthoritativeIdentity)

      listOf("alpha", "beta").forEach { name ->
        val lane = witnesses.filter { it.name == name }
        assertTrue(lane.isNotEmpty(), "$name was starved by another free-call primary")
        lane.forEach(::assertAuthoritativeIdentity)
        val arities = lane.map { (it.arguments as Array<dynamic>).size }
        assertEquals(0, arities.first())
        assertEquals(arities.sorted(), arities)
      }
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 24)
    }

  @Test
  fun requiredArityBucketsAndFamiliesCannotStarveEachOther(): Promise<Unit> =
    MainScope().promise {
      val freeNullaries = (0 until 20).joinToString("\n") { index ->
        "template<class T = int> int free_zero_$index() { return $index; }"
      }
      val memberLocals = (0 until 20).joinToString("\n") { index ->
        "  MemberNoise<$index> member_$index;"
      }
      val response = query(
        """
          $freeNullaries

          template<class T>
          T free_unary(T value) { return value; }

          template<int Tag>
          struct MemberNoise {
            template<class T = int>
            int member_zero() { return Tag; }
          };

          struct MemberTarget {
            template<class T>
            T member_unary(T value) { return value; }
          };

          int main() {
          $memberLocals
            MemberTarget target;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.callWitnesses as Array<dynamic>).toList()
      assertTrue(witnesses.any { witness ->
        witness.syntax == "freeCall" && witness.name == "free_unary" &&
          (witness.arguments as Array<dynamic>).size == 1
      }, "twenty nullary free primaries starved the viable unary free call")
      assertTrue(witnesses.any { witness ->
        witness.syntax == "memberCall" && witness.name == "member_unary" &&
          (witness.arguments as Array<dynamic>).size == 1
      }, "twenty nullary member lanes starved the viable unary member call")
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 16)
    }

  @Test
  fun saturatedIntegralSeedInventoryStillRetainsLanguageBaseline(): Promise<Unit> =
    MainScope().promise {
      val constants = (2 until 42).joinToString("\n") { value ->
        "constexpr size_type seed_$value = $value;"
      }
      val response = query(
        """
          using size_type = decltype(sizeof(0));
          $constants

          template<size_type I>
          int baseline_index() {
            static_assert(I == 1, "the retained one-literal is required");
            return 1;
          }

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val operations = assertNotNull(response.operations)
      val baseline = assertNotNull(freeWitnesses(response).firstOrNull { candidate ->
        if (candidate.name != "baseline_index") return@firstOrNull false
        val argument: dynamic =
          (candidate.explicitTemplateArguments as Array<dynamic>).singleOrNull()
        argument != null && argument.kind == "exactIntegerLiteral" &&
          argument.canonicalValue == "1"
      })
      assertAuthoritativeIdentity(baseline)
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue(((operations.callWitnessProbeCount as Number).toInt()) <= 16)
    }

  @Test
  fun declarationOnlyOrdinaryFunctionIsAuthenticatedByItsCallExpression(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          struct Token {};
          Token declared_only(const Token& value);

          int main() {
            Token value;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val witness = assertNotNull(
        freeWitnesses(response).firstOrNull { it.name == "declared_only" }
      )
      assertOrdinaryAuthoritativeIdentity(witness)
      assertEquals(1, (witness.arguments as Array<dynamic>).size)
      assertTrue((witness.callable.returnType as String).contains("Token"))
    }

  @Test
  fun ordinaryOverloadsRetainTheExactSelectedCanonicalDeclaration(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          int choose(int value);
          double choose(double value);

          int main() {
            int value = 1;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 12,
        maxArity = 1
      )
      val overloads = freeWitnesses(response).filter { it.name == "choose" }
      val integer = assertNotNull(overloads.firstOrNull { candidate ->
        (candidate.arguments as Array<dynamic>).singleOrNull()?.kind ==
          "integerZero"
      })
      val floating = assertNotNull(overloads.firstOrNull { candidate ->
        (candidate.arguments as Array<dynamic>).singleOrNull()?.kind ==
          "floatingZero"
      })
      assertOrdinaryAuthoritativeIdentity(integer)
      assertOrdinaryAuthoritativeIdentity(floating)
      assertEquals("int", integer.callable.returnType as String)
      assertEquals("double", floating.callable.returnType as String)
      assertTrue(
        (integer.targetId as String) != (floating.targetId as String),
        "distinct overload declarations collapsed to one target identity"
      )
    }

  @Test
  fun deletedAndInvalidOrdinaryRoutesFailClosedWithoutPoisoningValidWork(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          int available(int value);
          int deleted_route(int value) = delete;
          int invalid_route(UnknownType value);

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 12,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response)
      val available = assertNotNull(witnesses.firstOrNull {
        it.name == "available"
      })
      assertOrdinaryAuthoritativeIdentity(available)
      assertTrue(witnesses.none { it.name == "deleted_route" })
      assertTrue(witnesses.none { it.name == "invalid_route" })
    }

  @Test
  fun benchmarkShapedDeclarationIsReachableThroughQualifiedAndAdlRoutes(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          namespace benchmark_api {
          struct Document {};
          struct Node {};
          using size_type = decltype(sizeof(0));
          bool inspect(const Document&, const Node*, size_type, bool);
          }

          int main() {
            benchmark_api::Document document;
            benchmark_api::Node node;
            benchmark_api::size_type index = 0;
            bool verbose = true;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 16,
        maxArity = 4
      )
      val witnesses = freeWitnesses(response)
      val witnessSummary = witnesses.joinToString { candidate ->
        "${candidate.name}/${(candidate.arguments as Array<dynamic>).size}"
      }
      val qualified = assertNotNull(witnesses.firstOrNull {
        it.name == "::benchmark_api::inspect" &&
          (it.arguments as Array<dynamic>).size == 4
      }, "missing rooted benchmark-shaped route from: $witnessSummary")
      val adl = assertNotNull(witnesses.firstOrNull {
        it.name == "inspect" && (it.arguments as Array<dynamic>).size == 4
      }, "missing exact ADL route from: $witnessSummary")
      assertOrdinaryAuthoritativeIdentity(qualified)
      assertOrdinaryAuthoritativeIdentity(adl)
      assertEquals(qualified.targetId as String, adl.targetId as String)
      assertEquals("bool", qualified.callable.returnType as String)
    }

  @Test
  fun globallyRootedRouteSurvivesAlongsideAnExactRelativeRoute(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          namespace service {
          int evaluate(int value);
          }

          int main() {
            int service = 0;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response)
      val rooted = assertNotNull(witnesses.firstOrNull {
        it.name == "::service::evaluate"
      })
      assertOrdinaryAuthoritativeIdentity(rooted)
      val relative = witnesses.firstOrNull { it.name == "service::evaluate" }
      if (relative != null) {
        assertOrdinaryAuthoritativeIdentity(relative)
        assertEquals(rooted.targetId as String, relative.targetId as String)
      }
    }

  @Test
  fun activeMacrosRejectCalleeAndFixedLiteralSpellingsAtTheCursor(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          int macro_target(int value);
          int bool_target(bool value);
          int pointer_target(void *value);
          int safe_target(int value);

          #define macro_target replacement
          #define true false
          #define nullptr 0

          int main() {
            cursor
          }
        """.trimIndent(),
        witnessLimit = 16,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response)
      assertTrue(witnesses.any { it.name == "safe_target" || it.name == "::safe_target" })
      assertTrue(witnesses.none {
        (it.name as String).removePrefix("::") == "macro_target"
      })
      assertTrue(witnesses.flatMap {
        (it.arguments as Array<dynamic>).toList()
      }.none {
        it.kind == "booleanTrue" || it.kind == "nullptr"
      }, "a macro-active fixed literal spelling escaped the profile filter")
    }

  @Test
  fun macroActiveOperandIsRemovedWhileSameTypedSafeLvalueRemainsUsable(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          struct Token {};
          int consume(Token& value);

          int main() {
            Token unsafe_value;
            Token safe_value;
          #define unsafe_value replacement
            cursor
          }
        """.trimIndent(),
        witnessLimit = 16,
        maxArity = 1
      )
      fun terminal(spelling: String): String = spelling.split("::").last()
      fun names(raw: dynamic): List<String> =
        ((raw as? Array<dynamic>) ?: emptyArray<dynamic>()).mapNotNull {
          (it.insertText as? String) ?: (it.name as? String)
        }

      val itemNames = names(response.items) + names(response.scopeItems)
      val graphNames = names(response.graph?.nodes)
      val operationValueNames =
        ((response.operations?.nodes as? Array<dynamic>) ?: emptyArray<dynamic>())
          .filter { it.isValue == true }
          .mapNotNull { it.name as? String }
      listOf(itemNames, graphNames, operationValueNames).forEach { inventory ->
        assertTrue(inventory.none { terminal(it) == "unsafe_value" },
          "a macro-active operand survived a semantic reference ingress")
      }
      assertTrue((itemNames + graphNames + operationValueNames).any {
        terminal(it) == "safe_value"
      }, "the same-typed safe operand was removed with the macro-active one")

      val witness = assertNotNull(freeWitnesses(response).firstOrNull { candidate ->
        if ((candidate.name as String).removePrefix("::") != "consume")
          return@firstOrNull false
        val argument: dynamic =
          (candidate.arguments as Array<dynamic>).singleOrNull()
        argument != null && argument.kind == "opaque" &&
          argument.valueCategory == "lvalue"
      })
      assertOrdinaryAuthoritativeIdentity(witness)
    }

  @Test
  fun exactNamespaceAliasRouteSurvivesAReservedPhysicalNamespace(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          namespace _implementation {
          struct Token {};
          int route(Token value);
          }
          namespace public_api = _implementation;

          int main() {
            public_api::Token token;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 16,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response)
      val aliased = assertNotNull(witnesses.firstOrNull {
        it.name == "public_api::route"
      })
      assertOrdinaryAuthoritativeIdentity(aliased)
      assertTrue(witnesses.none {
        ((it.name as? String) ?: "").contains("_implementation::route")
      }, "the reserved physical namespace escaped through a source route")
    }

  @Test
  fun bareGlobalRouteIsPublishedOnlyAfterAdlSelectsItsExactTarget(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          struct Token {};
          long adl_route(Token value);

          namespace inner {
          int adl_route(int value);
          void probe() {
            Token token;
            cursor
          }
          }
        """.trimIndent(),
        witnessLimit = 16,
        maxArity = 1
      )
      val witness = assertNotNull(freeWitnesses(response).firstOrNull {
        it.name == "adl_route" && it.callable?.returnType == "long" &&
          (((it.callable?.parameters as? Array<dynamic>)?.singleOrNull()?.type
            as? String) ?: "").contains("Token")
      })
      assertOrdinaryAuthoritativeIdentity(witness)
    }

  @Test
  fun hiddenFriendIsPublishedOnlyThroughAdlWithItsExactTarget(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          struct Token {
            friend long hidden_route(Token& value);
          };

          int hidden_route(int value);

          int main() {
            Token token;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 8,
        maxArity = 1
      )
      val witnesses = freeWitnesses(response)
      val hidden = assertNotNull(witnesses.firstOrNull { candidate ->
        val argument = (candidate.arguments as? Array<dynamic>)?.singleOrNull()
          ?: return@firstOrNull false
        candidate.name == "hidden_route" && candidate.callable?.returnType == "long" &&
          argument.kind == "opaque" && argument.valueCategory == "lvalue" &&
          ((argument.type as? String) ?: "").contains("Token")
      })
      assertOrdinaryAuthoritativeIdentity(hidden)
      val targetId = hidden.targetId as String
      assertTrue(witnesses.none { candidate ->
        (candidate.targetId as? String) == targetId &&
          (candidate.name as? String) != "hidden_route"
      }, "a hidden friend escaped through a qualified or rooted spelling")
    }

  @Test
  fun qualifiedLibraryTemplateCanUseAnExactSourceLvalueProfile(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          #include <memory>

          struct AbstractValue {
            virtual ~AbstractValue() = default;
            virtual int read() const = 0;
          };

          int main() {
            AbstractValue* pointer = nullptr;
            AbstractValue& reference = *pointer;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 256,
        maxArity = 1
      )
      val operations = assertNotNull(response.operations)
      val witness = assertNotNull(freeWitnesses(response).firstOrNull { candidate ->
        val name = candidate.name as? String ?: return@firstOrNull false
        val argument = (candidate.arguments as? Array<dynamic>)?.singleOrNull()
          ?: return@firstOrNull false
        name.removePrefix("::") == "std::addressof" &&
          argument.kind == "opaque" && argument.valueCategory == "lvalue" &&
          ((argument.type as? String) ?: "").contains("AbstractValue")
      }, "the bounded free-call scheduler starved a viable qualified template/lvalue lane")
      assertAuthoritativeIdentity(witness)
      assertTrue(operations.callWitnessesIncomplete == true)
      assertTrue((operations.callWitnessProbeCount as Number).toInt() in 1..64)
    }
}
