package cppcompletion

import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppSemanticOperatorWitnessEndpointTest {
  private suspend fun query(source: String, witnessLimit: Int = 32): dynamic {
    val lines = source.lines()
    val line = lines.indexOfFirst { it.trim() == "cursor" }
    require(line >= 0)
    return CppBrowserClangdClient().semanticResponse(
      source = source,
      line = line,
      character = lines[line].length,
      graphLimit = 512,
      graphDepth = 3,
      operationLimit = 512,
      operationDepth = 3,
      callWitnessLimit = witnessLimit,
      callWitnessMaxArity = 2
    )
  }

  private fun operatorWitnesses(response: dynamic): List<dynamic> {
    val operations = assertNotNull(response.operations)
    return (operations.callWitnesses as Array<dynamic>).toList()
      .filter { it.syntax == "binaryOperator" }
  }

  private fun assertWholeBinarySchema(witness: dynamic, spelling: String) {
    assertTrue(witness.authoritative == true)
    assertEquals("binaryOperator", witness.syntax as String)
    assertEquals(spelling, witness.operatorSpelling as String)
    assertEquals("operator$spelling", witness.name as String)
    assertNotNull(witness.receiver)
    assertEquals(1, (witness.arguments as Array<dynamic>).size)
    assertTrue((witness.explicitTemplateArguments as Array<dynamic>).isEmpty())
    assertTrue((witness.targetId as String).isNotEmpty())
    assertTrue((witness.callable.id as String).isNotEmpty())
    assertNotNull(witness.result)
  }

  @Test
  fun ordinaryMemberAndDeducedFreeOperatorsCarryExactWholeRelations(): Promise<Unit> =
    MainScope().promise {
      val response = query(
        """
          struct Glyph {};
          template<class T> struct Box {};

          struct Emitter {
            Emitter& operator<<(const Glyph&) { return *this; }
          };

          template<class T>
          Emitter& operator<<(Emitter& out, const Box<T>&) { return out; }

          int main() {
            Emitter out;
            Glyph glyph;
            Box<int> box;
            cursor
          }
        """.trimIndent(),
        witnessLimit = 32
      )
      val operations = assertNotNull(response.operations)
      val insertions = operatorWitnesses(response).filter { it.operatorSpelling == "<<" }
      val member = assertNotNull(insertions.firstOrNull { witness ->
        (witness.receiver?.type as? String)?.contains("Emitter") == true &&
          (witness.arguments as Array<dynamic>).single().let { argument ->
            (argument.type as? String)?.contains("Glyph") == true
          }
      })
      assertWholeBinarySchema(member, "<<")
      assertEquals("semaBinaryOperatorExpression", member.validation as String)
      assertEquals(member.targetId as String, member.callable.id as String)
      assertEquals("", (member.primaryTemplateId as? String) ?: "")
      assertEquals("lvalue", member.receiver.valueCategory as String)
      assertEquals("lvalue", member.result.valueCategory as String)

      val deduced = assertNotNull(insertions.firstOrNull { witness ->
        (witness.arguments as Array<dynamic>).single().let { argument ->
          (argument.type as? String)?.contains("Box<int>") == true
        }
      })
      assertWholeBinarySchema(deduced, "<<")
      assertEquals("recursiveDefinitionInstantiation", deduced.validation as String)
      assertEquals(deduced.targetId as String, deduced.primaryTemplateId as String)
      assertEquals(deduced.targetId as String, deduced.callable.primaryTemplateId as String)

      val probes = (operations.binaryOperatorWitnessProbeCount as Number).toInt()
      assertTrue(probes in 1..32)
      assertTrue(
        (operations.binaryOperatorWitnessDiscoveryCount as Number).toInt() >= insertions.size
      )
      assertTrue(operations.binaryOperatorWitnessesIncomplete == true)
    }

  @Test
  fun buildBinOpRejectsAmbiguityAndDeletedCopyButKeepsReferenceControl(): Promise<Unit> =
    MainScope().promise {
      val witnesses = operatorWitnesses(
        query(
          """
            struct Ambiguous {
              int operator+(int) const { return 1; }
            };
            int operator+(const Ambiguous&, int) { return 2; }

            struct Blocked {
              Blocked() = default;
              Blocked(const Blocked&) = delete;
            };
            Blocked operator<<(Blocked, int) { return {}; }

            struct Safe {};
            Safe& operator<<(Safe& value, int) { return value; }

            int main() {
              Ambiguous ambiguous;
              Blocked blocked;
              Safe safe;
              cursor
            }
          """.trimIndent(),
          witnessLimit = 64
        )
      )
      fun exactLvalue(witness: dynamic, spelling: String, type: String): Boolean =
        witness.operatorSpelling == spelling &&
          witness.receiver?.valueCategory == "lvalue" &&
          ((witness.receiver?.type as? String)?.contains(type) == true) &&
          (witness.arguments as Array<dynamic>).single().type == "int"

      assertTrue(
        witnesses.none { exactLvalue(it, "+", "Ambiguous") },
        "an ambiguous member/nonmember overload set produced authority"
      )
      assertTrue(
        witnesses.none { exactLvalue(it, "<<", "Blocked") },
        "a deleted lvalue copy produced a by-value operator witness"
      )
      val safe = assertNotNull(witnesses.firstOrNull { exactLvalue(it, "<<", "Safe") })
      assertWholeBinarySchema(safe, "<<")
    }

  @Test
  fun rewrittenComparisonsRetainSurfaceOperatorAndSelectedTargetIdentity(): Promise<Unit> =
    MainScope().promise {
      val witnesses = operatorWitnesses(
        query(
          """
            #include <compare>

            struct Ordered {
              int value;
              friend auto operator<=>(const Ordered&, const Ordered&) = default;
              friend bool operator==(const Ordered&, const Ordered&) = default;
            };

            int main() {
              Ordered left{1};
              Ordered right{2};
              cursor
            }
          """.trimIndent(),
          // The observable result budget is shared fairly with ordinary
          // member/free witnesses from <compare>. Leave enough room for one
          // result from every independently probed comparison surface lane.
          witnessLimit = 128
        )
      )
      val less = assertNotNull(witnesses.firstOrNull { witness ->
        witness.operatorSpelling == "<" && witness.receiver?.valueCategory == "lvalue" &&
          ((witness.receiver?.type as? String)?.contains("Ordered") == true)
      })
      assertWholeBinarySchema(less, "<")
      assertEquals("semaDefaultedDefinition", less.validation as String)
      assertTrue(((less.callable.qualifiedName as? String) ?: "").contains("operator<=>"))
      assertEquals("bool", less.result.type as String)

      val notEqual = assertNotNull(witnesses.firstOrNull { witness ->
        witness.operatorSpelling == "!=" && witness.receiver?.valueCategory == "lvalue" &&
          ((witness.receiver?.type as? String)?.contains("Ordered") == true)
      })
      assertWholeBinarySchema(notEqual, "!=")
      assertEquals("semaDefaultedDefinition", notEqual.validation as String)
      assertTrue(((notEqual.callable.qualifiedName as? String) ?: "").contains("operator=="))
      assertEquals(notEqual.targetId as String, notEqual.callable.id as String)
      assertEquals("bool", notEqual.result.type as String)
    }

  @Test
  fun dependentScalarLeftAndExplicitObjectOperatorsReachBuildBinOp(): Promise<Unit> =
    MainScope().promise {
      val witnesses = operatorWitnesses(
        query(
          """
            struct Right {};
            template<class T>
            long operator+(T, const Right&) { return 1; }

            struct ExplicitObject {
              int operator+(this const ExplicitObject&, int value) {
                return value;
              }
            };

            int main() {
              int scalar = 1;
              Right right;
              const ExplicitObject object;
              cursor
            }
          """.trimIndent(),
          witnessLimit = 64
        )
      )
      val scalarLeft = assertNotNull(witnesses.firstOrNull { witness ->
        witness.operatorSpelling == "+" && witness.receiver?.type == "int" &&
          ((witness.arguments as Array<dynamic>).singleOrNull()?.type as? String)
            ?.contains("Right") == true
      })
      assertWholeBinarySchema(scalarLeft, "+")
      assertEquals("long", scalarLeft.result.type as String)

      val explicitObject = assertNotNull(witnesses.firstOrNull { witness ->
        witness.operatorSpelling == "+" && witness.receiver?.valueCategory == "lvalue" &&
          ((witness.receiver?.type as? String)?.contains("ExplicitObject") == true) &&
          (witness.arguments as Array<dynamic>).singleOrNull()?.type == "int"
      })
      assertWholeBinarySchema(explicitObject, "+")
      assertEquals("int", explicitObject.result.type as String)
    }

  @Test
  fun rewrittenOuterTemplateBodyIsPartOfTheAuthenticatedExpression(): Promise<Unit> =
    MainScope().promise {
      val witnesses = operatorWitnesses(
        query(
          """
            struct GoodTag { using available = int; };
            struct BadTag {};

            template<class T> struct Category {};
            template<class T> struct Subject {};

            template<class T>
            Category<T> operator<=>(const Subject<T>&, const Subject<T>&) {
              return {};
            }

            template<class T>
            bool operator<(Category<T>, int) {
              return sizeof(typename T::available) != 0;
            }

            int main() {
              Subject<GoodTag> good;
              Subject<BadTag> bad;
              cursor
            }
          """.trimIndent(),
          witnessLimit = 128
        )
      )
      fun rewrittenLessFor(type: String): dynamic = witnesses.firstOrNull { witness ->
        witness.operatorSpelling == "<" && witness.receiver?.valueCategory == "lvalue" &&
          ((witness.receiver?.type as? String)?.contains(type) == true)
      }

      val good = assertNotNull(rewrittenLessFor("Subject<GoodTag>"))
      assertWholeBinarySchema(good, "<")
      assertEquals("recursiveDefinitionInstantiation", good.validation as String)
      assertTrue(rewrittenLessFor("Subject<BadTag>") == null)
    }
}
