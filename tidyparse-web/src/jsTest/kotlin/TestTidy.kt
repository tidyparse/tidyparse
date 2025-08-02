import kotlin.test.Test
import kotlin.test.assertEquals

class TestTidy {
  @Test
  fun thingsShouldWork() {
    assertEquals(listOf(1,2,3).reversed(), listOf(3,2,1))
  }

  @Test
  fun thingsShouldFail() {
    assertEquals(listOf(1,2,3).reversed(), listOf(1,2,3))
  }
}