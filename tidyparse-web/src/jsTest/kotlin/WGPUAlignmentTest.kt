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

    val decoded = packet.decodePacket(0, listOf("x", "c"), packet.length)!!
    assertEquals(4, decoded.distance)
    assertEquals("x c", decoded.plainResult)
    assertEquals(
      listOf(LEV_EDIT_SUBSTITUTE, LEV_EDIT_DELETE, LEV_EDIT_DELETE, LEV_EDIT_MATCH, LEV_EDIT_DELETE),
      decoded.editScript
    )
    assertEquals(17f, decoded.score)
    assertEquals(
      "<sub>x</sub> <del></del> <del></del> c <del></del>",
      IntersectionResults(listOf(decoded.plainResult), listOf(decoded.editScript), listOf(decoded.score))
        .annotatedResults.single()
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

    assertEquals(listOf("x y"), lowestDistance.plainResults)
    assertEquals(listOf(LEV_EDIT_INSERT, LEV_EDIT_MATCH), lowestDistance.editScript.single())

    val repeatedCfg = "START -> aa".parseCFG()
    val aa = repeatedCfg.tmMap.getValue("aa") + 1
    val tiedDistance = packet(
      1, 0, packed(aa, PACKED_FIRST_DELETION_TAG), 0,
      1, 0, packed(aa, 0), 0
    )
    val canonicalTie = decodePackets(tiedDistance, repeatedCfg, maxRepairLen = 4)

    assertEquals(listOf("aa"), canonicalTie.plainResults)
    assertEquals(listOf(LEV_EDIT_MATCH, LEV_EDIT_DELETE), canonicalTie.editScript.single())
    assertEquals("aa <del></del>", canonicalTie.annotatedResults.single())
  }

  @Test
  fun textNormalizationKeepsMatchMetadataAligned() {
    val results = IntersectionResults(
      plainResults = listOf("[START]"),
      editScript = listOf(listOf(LEV_EDIT_MATCH)),
      scores = listOf(0f)
    ).mapPlainResults { it.replace("[START]", "START") }

    assertEquals("START", results.annotatedResults.single())
    assertEquals(0, results.editDistanceAt(0))
  }

  @Test
  fun annotationEscapesTerminals() {
    val results = IntersectionResults(
      plainResults = listOf("<new> a&b x>y"),
      editScript = listOf(listOf(
        LEV_EDIT_INSERT, LEV_EDIT_MATCH, LEV_EDIT_SUBSTITUTE, LEV_EDIT_DELETE
      )),
      scores = listOf(0f)
    )

    assertEquals(
      "<ins>&lt;new&gt;</ins> a&amp;b <sub>x&gt;y</sub> <del></del>",
      results.annotatedResults.single()
    )
  }
}
