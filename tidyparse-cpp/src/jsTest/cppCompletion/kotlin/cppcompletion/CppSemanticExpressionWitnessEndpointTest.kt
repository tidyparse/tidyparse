package cppcompletion

import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticExpressionWitnessEndpointTest {
  private suspend fun response(source: String, witnessLimit: Int = 64): dynamic {
    val lines = source.lines()
    val line = lines.indexOfFirst { it.trim() == "completion_marker" }
    require(line >= 0)
    return CppBrowserClangdClient().semanticResponse(
      source = source,
      line = line,
      character = lines[line].length,
      graphLimit = 512,
      graphDepth = 2,
      operationLimit = 512,
      operationDepth = 2,
      callWitnessLimit = 0,
      callWitnessMaxArity = 0,
      expressionWitnessLimit = witnessLimit
    )
  }

  @Test
  fun exactPolymorphicDowncastSurvivesStandardLibraryNoise(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          #include <iostream>
          #include <memory>

          class Shape {
          public:
            virtual ~Shape() = default;
            virtual double area() const = 0;
          };
          class Circle final : public Shape {
          public:
            explicit Circle(double) {}
            double area() const override { return 1.0; }
          };

          int main() {
            Circle circle{3.0};
            Shape *base = &circle;
            Shape &alias = *base;
            completion_marker
          }
        """.trimIndent(),
        witnessLimit = 128
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()
      val dynamicCasts = witnesses.filter { it.syntax == "dynamicCast" }
      val exact = dynamicCasts.firstOrNull {
        it.typeOperand?.type == "::Circle *" &&
          it.expressionOperand?.type == "::Shape *" &&
          it.expressionOperand?.valueCategory == "lvalue"
      }
      val dynamicSummary = dynamicCasts.joinToString { witness ->
        "${witness.typeOperand?.type}<-${witness.expressionOperand?.type}/" +
          "${witness.expressionOperand?.valueCategory}"
      }
      val targetSummary = witnesses.mapNotNull {
        it.typeOperand?.type as? String
      }.distinct().joinToString()
      assertNotNull(exact, "$dynamicSummary; all targets=[$targetSummary]")
    }

  @Test
  fun semaBuildsAllFourExpressionFamiliesAndQueuesTypeInfoMembers(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          #include <typeinfo>

          struct Root { virtual ~Root() = default; };
          struct Leaf : Root {};
          struct Plain {};

          int main() {
            Root *root = nullptr;
            Leaf *leaf = nullptr;
            Plain *plain = nullptr;
            Root &view = *root;
            completion_marker
          }
        """.trimIndent()
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()
      val syntax = witnesses.mapNotNull { it.syntax as? String }.toSet()

      assertEquals(
        setOf("dynamicCast", "reinterpretCast", "typeidExpression", "typeidType"),
        syntax
      )
      assertTrue(witnesses.all {
        it.authoritative == true && it.validation == "semaExpressionBuild"
      })
      assertTrue(witnesses.size <= 64)
      assertTrue((operations.expressionWitnessProbeCount as Number).toInt() <= 128)
      assertTrue(
        (operations.expressionWitnessDiscoveryCount as Number).toInt() >= witnesses.size
      )
      assertTrue(operations.expressionWitnessesIncomplete == true)

      val dynamicCast = assertNotNull(witnesses.firstOrNull {
        it.syntax == "dynamicCast" && it.typeOperand?.typeInfo?.kind == "pointer" &&
          it.expressionOperand?.typeInfo?.kind == "pointer"
      })
      assertEquals(
        dynamicCast.typeOperand.typeInfo.canonicalId,
        dynamicCast.result.typeInfo.canonicalId
      )
      assertEquals("prvalue", dynamicCast.result.valueCategory)

      val reinterpretCast = assertNotNull(witnesses.firstOrNull {
        it.syntax == "reinterpretCast" && it.typeOperand?.typeInfo?.kind == "pointer" &&
          it.expressionOperand?.typeInfo?.kind == "pointer"
      })
      assertEquals(
        reinterpretCast.typeOperand.typeInfo.canonicalId,
        reinterpretCast.result.typeInfo.canonicalId
      )

      val typeidExpression = assertNotNull(witnesses.firstOrNull {
        it.syntax == "typeidExpression" && it.expressionOperand?.type == "::Root" &&
          it.expressionOperand?.valueCategory == "lvalue"
      })
      assertEquals("lvalue", typeidExpression.result.valueCategory)
      assertEquals(true, typeidExpression.result.typeInfo.isConst)
      assertEquals("record", typeidExpression.result.typeInfo.kind)

      val typeidType = assertNotNull(witnesses.firstOrNull {
        it.syntax == "typeidType" && it.typeOperand?.type == "int"
      })
      val typeInfoId = assertNotNull(typeidType.result.typeInfo.valueCanonicalId as? String)
      val nodes = (operations.nodes as Array<dynamic>).toList()
      assertTrue(nodes.any {
        it.name == "name" && it.role == "member" &&
          it.ownerTypeInfo?.valueCanonicalId == typeInfoId
      }, "the early typeid(int) result did not seed std::type_info::name()")
    }

  @Test
  fun reinterpretCastProbesRecordGlvaluesAgainstReferenceTargets(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          struct Source {};
          struct Target {};

          int main() {
            Source source;
            completion_marker
          }
        """.trimIndent()
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()
      val reinterpretReference = assertNotNull(witnesses.firstOrNull {
        it.syntax == "reinterpretCast" && it.typeOperand?.type == "::Target &" &&
          it.typeOperand?.typeInfo?.kind == "lvalueReference" &&
          it.expressionOperand?.type == "::Source" &&
          it.expressionOperand?.valueCategory == "lvalue"
      })

      assertEquals("lvalue", reinterpretReference.result.valueCategory)
      assertEquals(
        reinterpretReference.typeOperand.typeInfo.valueCanonicalId,
        reinterpretReference.result.typeInfo.canonicalId
      )
    }

  @Test
  fun typeidFailsClosedWithoutItsRequiredDeclarationWhileCastsRemain(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          struct Root { virtual ~Root() = default; };
          struct Leaf : Root {};

          int main() {
            Root *root = nullptr;
            Leaf *leaf = nullptr;
            completion_marker
          }
        """.trimIndent()
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()

      assertTrue(witnesses.any { it.syntax == "dynamicCast" })
      assertTrue(witnesses.any { it.syntax == "reinterpretCast" })
      assertTrue(witnesses.none {
        it.syntax == "typeidExpression" || it.syntax == "typeidType"
      })
      assertTrue(operations.expressionWitnessesIncomplete == true)
    }

  @Test
  fun privateNestedTypeIdsAreNeverSynthesizedPastAccessControl(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          #include <typeinfo>

          struct Root { virtual ~Root() = default; };
          class Vault {
            struct Secret {};
            class PrivateOuter {
            public:
              struct PublicInner {};
            };
          public:
            struct Visible {};
          };
          class AliasVault {
            struct Impl {};
          public:
            using Public = Impl;
          };
          void hidden_scope() {
            struct Hidden {};
          }

          int main() {
            Root *root = nullptr;
            completion_marker
          }
        """.trimIndent()
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()
      val targetSpellings = witnesses.mapNotNull { it.typeOperand?.type as? String }

      assertTrue(targetSpellings.any { it.contains("Vault::Visible") })
      assertTrue(targetSpellings.none { it.contains("Vault::Secret") })
      assertTrue(targetSpellings.none {
        it.contains("Vault::PrivateOuter::PublicInner")
      })
      assertTrue(targetSpellings.none { it.contains("Hidden") })
      assertTrue(targetSpellings.any { it.contains("AliasVault::Public") })
    }

  @Test
  fun localTypeIdsRequireTheExactDeclarationVisibleAtTheCursor(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          #include <typeinfo>

          int main() {
            {
              struct Expired {};
              using Gone = Expired;
            }

            struct Outer {};
            using Retained = Outer;
            using Choice = Outer;
            {
              struct Inner {};
              using Choice = Inner;
              completion_marker
            }
          }
        """.trimIndent()
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()
      val targetSpellings = witnesses.mapNotNull {
        it.typeOperand?.type as? String
      }

      assertTrue(targetSpellings.none { it == "Gone" || it == "Expired" })
      fun targetIds(spelling: String) = witnesses.filter {
        it.typeOperand?.type == spelling
      }.mapNotNull { it.typeOperand?.typeInfo?.canonicalId as? String }.toSet()
      val outerId = assertNotNull(targetIds("Outer").singleOrNull())
      val innerId = assertNotNull(targetIds("Inner").singleOrNull())
      assertEquals(
        setOf(outerId),
        targetIds("Retained"),
        "a visible alias from an enclosing active block was dropped"
      )
      assertEquals(
        setOf(innerId),
        targetIds("Choice"),
        "a shadowed same-name local alias contributed the wrong type-id"
      )
    }

  @Test
  fun namespaceTypesAreGloballyRootedAndAnonymousImplementationsNeedAPublicAlias(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          #include <typeinfo>

          namespace library { struct Value {}; }
          namespace {
          struct Hidden {};
          }
          using PublicHidden = Hidden;

          namespace client {
          namespace library { struct Value {}; }
          void inspect() {
            completion_marker
          }
          }
        """.trimIndent(),
        witnessLimit = 96
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()
      val targets = witnesses.mapNotNull { witness ->
        val type = witness.typeOperand?.type as? String ?: return@mapNotNull null
        val id = witness.typeOperand?.typeInfo?.canonicalId as? String
          ?: return@mapNotNull null
        type to id
      }

      val globalValue = assertNotNull(targets.firstOrNull {
        it.first == "::library::Value"
      })
      val clientValue = assertNotNull(targets.firstOrNull {
        it.first == "::client::library::Value"
      })
      assertTrue(globalValue.second != clientValue.second)
      assertTrue(targets.none {
        it.first == "library::Value" || "anonymous namespace" in it.first
      })
      assertTrue(targets.any { it.first == "::PublicHidden" })
    }

  @Test
  fun activeMacroIdentifiersCannotEnterReferenceInventories(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          int main() {
            int x = 1;
            int safe = 2;
          #define restored hidden
          #undef restored
            int restored = 3;
          #define x replacement
            completion_marker
          }
        """.trimIndent(),
        witnessLimit = 8
      )
      val itemNames = sequenceOf(response.items, response.scopeItems)
        .flatMap { raw -> ((raw as? Array<dynamic>) ?: emptyArray()).asSequence() }
        .mapNotNull { it.insertText as? String }
      val graphNames = ((response.graph?.nodes as? Array<dynamic>) ?: emptyArray())
        .asSequence().mapNotNull { it.name as? String }
      val operationNames = ((response.operations?.nodes as? Array<dynamic>) ?: emptyArray())
        .asSequence().mapNotNull { it.name as? String }
      val names = (itemNames + graphNames + operationNames).toList()
      fun terminal(spelling: String): String = spelling.split("::").last()

      assertTrue(names.none { terminal(it) == "x" },
        "a macro-active local declaration remained a grammar reference")
      assertTrue(names.any { terminal(it) == "safe" })
      assertTrue(names.any { terminal(it) == "restored" },
        "location-sensitive lookup treated an already-undefined macro as active")
    }

  @Test
  fun activeTypeidMacroRejectsSyntheticTypeidWitnesses(): Promise<Unit> =
    MainScope().promise {
      val response = response(
        """
          #include <typeinfo>
          struct Value {};
          #define typeid forbidden

          int main() {
            Value value;
            completion_marker
          }
        """.trimIndent()
      )
      val operations = assertNotNull(response.operations)
      val witnesses = (operations.expressionWitnesses as Array<dynamic>).toList()
      assertTrue(witnesses.none {
        it.syntax == "typeidExpression" || it.syntax == "typeidType"
      })
      assertTrue(witnesses.any { it.syntax == "reinterpretCast" },
        "the macro guard accidentally removed unrelated expression witnesses")
    }
}
