import ai.hypergraph.kaliningraph.parsing.parseCFG
import ai.hypergraph.kaliningraph.parsing.tmMap
import kotlin.test.Test
import kotlin.test.assertEquals

class WGPUAlignmentTest {
  private fun packet(vararg values: Int): JSIntArray =
    JSIntArray(values.size).apply { set(values.toTypedArray(), 0) }

  private fun packed(token: Int, editTag: Int): Int = token or (editTag shl PACKED_EDIT_SHIFT)

  @Test
  fun packetDecoderExpandsEditAnnotations() {
    val packet = packet(
      4, 17,
      packed(1, PACKED_SUBSTITUTION_TAG),
      packed(2, PACKED_SUBSTITUTION_TAG + 2),
      0
    )

    val decoded = packet.decodePacket(0, terminalCount = 2, pktLen = packet.length)!!
    val results = IntersectionResults(listOf(decoded), listOf("x", "c"))
    assertEquals(4, results.editDistanceAt(0))
    assertEquals("x c", results[0])
    assertEquals(
      listOf(LEV_EDIT_SUBSTITUTE, LEV_EDIT_DELETE, LEV_EDIT_DELETE, LEV_EDIT_MATCH, LEV_EDIT_DELETE),
      results.editScriptAt(0)
    )
    assertEquals(17u, results.scoreAt(0))
    assertEquals(
      "<sub>x</sub> <del></del> <del></del> c <del></del>",
      results.htmlAt(0)
    )
  }

  @Test
  fun packetDedupUsesLowestDistanceAndLevenshteinTieBreak() {
    val pairCfg = "START -> x y".parseCFG()
    fun pairToken(token: String) = pairCfg.tmMap.getValue(token) + 1
    val differentDistances = packet(
      2, 0,
      packed(pairToken("x"), PACKED_SUBSTITUTION_TAG),
      packed(pairToken("y"), PACKED_INSERTION_TAG),
      0,
      1, 0,
      packed(pairToken("x"), PACKED_INSERTION_TAG),
      packed(pairToken("y"), 0),
      0
    )
    val lowestDistance = decodePackets(differentDistances, pairCfg, maxRepairLen = 5)

    assertEquals(listOf("x y"), lowestDistance)
    assertEquals(listOf(LEV_EDIT_INSERT, LEV_EDIT_MATCH), lowestDistance.editScriptAt(0))

    val repeatedCfg = "START -> aa".parseCFG()
    val aa = repeatedCfg.tmMap.getValue("aa") + 1
    val tiedDistance = packet(
      1, 0, packed(aa, PACKED_FIRST_DELETION_TAG), 0,
      1, 0, packed(aa, 0), 0
    )
    val canonicalTie = decodePackets(tiedDistance, repeatedCfg, maxRepairLen = 4)

    assertEquals(listOf("aa"), canonicalTie)
    assertEquals(listOf(LEV_EDIT_MATCH, LEV_EDIT_DELETE), canonicalTie.editScriptAt(0))
    assertEquals("aa <del></del>", canonicalTie.htmlAt(0))

    val insertionCfg = "START -> a a".parseCFG()
    val a = insertionCfg.tmMap.getValue("a") + 1
    val insertionTie = packet(
      1, 0, packed(a, PACKED_INSERTION_TAG), packed(a, 0), 0,
      1, 0, packed(a, 0), packed(a, PACKED_INSERTION_TAG), 0
    )
    val canonicalInsertion = decodePackets(insertionTie, insertionCfg, maxRepairLen = 5)

    assertEquals(listOf(LEV_EDIT_MATCH, LEV_EDIT_INSERT), canonicalInsertion.editScriptAt(0))
  }

  @Test
  fun tokenHashCollisionsCompareActualTokens() {
    val first = intArrayOf(0, 0, 1, 32)
    val equal = intArrayOf(0, 9, 1, 32)
    val collision = intArrayOf(0, 0, 2, 1)

    assertEquals(first.tokenHash(), collision.tokenHash())
    assertEquals(true, first.sameTokens(equal))
    assertEquals(false, first.sameTokens(collision))
  }

  @Test
  fun textNormalizationKeepsMatchMetadataAligned() {
    val results = IntersectionResults(
      listOf(intArrayOf(0, 0, 1), intArrayOf(0, 0, 2), intArrayOf(0, 0, 3)),
      listOf("[START]", "X[START]", "XSTART")
    ).mapTerminals { if (it == "[START]") "START" else it }

    assertEquals(listOf("START", "X[START]", "XSTART"), results)
    assertEquals("START", results.htmlAt(0))
    assertEquals(0, results.editDistanceAt(0))
  }

  @Test
  fun annotationEscapesTerminals() {
    val results = IntersectionResults(
      listOf(intArrayOf(
        3, 0,
        packed(1, PACKED_INSERTION_TAG),
        packed(2, 0),
        packed(3, PACKED_SUBSTITUTION_TAG)
      )),
      listOf("<new>", "a&b", "x>y")
    )

    assertEquals(
      "<ins>&lt;new&gt;</ins> a&amp;b <sub>x&gt;y</sub> <del></del>",
      results.htmlAt(0)
    )
  }
}
