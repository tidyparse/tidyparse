package cppcompletion

import cppEditorStatementSnapshot
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class CppBrowserClangdClientTest {
  @Test
  fun scopeRequestCarriesTheBoundedSemanticGraphContract() {
    val source = "  std"
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 0, source.length))
    val params = cppBrowserSemanticCompletionParams(
      snapshot = snapshot,
      line = 0,
      character = source.length,
      graphLimit = 321,
      graphDepth = 2
    )

    assertEquals(321, (params.graphLimit as Number).toInt())
    assertEquals(2, (params.graphDepth as Number).toInt())
    assertTrue((params.operationLimit as Number).toInt() > 0)
    assertEquals(2, (params.operationDepth as Number).toInt())
    assertTrue((params.expressionWitnessLimit as Number).toInt() > 0)
    assertTrue((params.callWitnessLimit as Number).toInt() > 0)
    assertTrue((params.callWitnessMaxArity as Number).toInt() > 0)
    assertEquals(2, (params.scopePosition.character as Number).toInt())
    assertEquals(source.length, (params.position.character as Number).toInt())
    assertEquals(true, params.allScopes as Boolean)
  }

  @Test
  fun receiverRequestKeepsItsMemberTriggerAndAlsoRequestsTheAmbientGraph() {
    val source = "value."
    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, 0, source.length))
    val params = cppBrowserSemanticCompletionParams(snapshot, 0, source.length)

    assertEquals(2, (params.context.triggerKind as Number).toInt())
    assertEquals(".", params.context.triggerCharacter as String)
    assertTrue((params.graphLimit as Number).toInt() > 0)
    assertEquals(0, (params.scopePosition.character as Number).toInt())
  }
}
