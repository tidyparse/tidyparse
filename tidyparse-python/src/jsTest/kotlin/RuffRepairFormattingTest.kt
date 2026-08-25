import kotlin.test.Test
import kotlin.test.assertEquals

class RuffRepairFormattingTest {
  @Test
  fun stripsRuffFinalNewline() {
    assertEquals("items[index]", normalizeRuffRepair("items [ index ]", "items[index]\n"))
  }

  @Test
  fun preservesWhitespaceInsideStrings() {
    assertEquals(
      "message = \"two  spaces\"",
      normalizeRuffRepair("message=\"two  spaces\"", "message = \"two  spaces\"\n")
    )
  }

  @Test
  fun exposesUnsafeStatementFlatteningToTheFullFileSemanticRecheck() {
    assertEquals("x = 1 y = 2", normalizeRuffRepair("x=1;y=2", "x = 1\ny = 2\n"))
  }

  @Test
  fun compactsMultilineDelimiterLayoutToOnePhysicalLine() {
    assertEquals(
      "points = [Point(3.0, 4.0),]",
      normalizeRuffRepair(
        "points = [ Point ( 3.0 , 4.0 ) , ]",
        "points = [\n    Point(3.0, 4.0),\n]\n"
      )
    )
  }

  @Test
  fun keepsRawRepairWhenRuffReturnsNoEdit() {
    assertEquals("value [ key ]", normalizeRuffRepair(" value [ key ] ", null))
  }

  @Test
  fun keepsRawRepairWhenFormatterOutputIsBlank() {
    assertEquals("value [ key ]", normalizeRuffRepair("value [ key ]", " \n\t"))
  }
}
