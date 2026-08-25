import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class PythonRepairAdmissibilityTest {
  @Test
  fun checksEveryRawCandidateBeforeFormattingSemanticSurvivors() {
    val events = mutableListOf<String>()
    val repairs = semanticallyAdmissibleRepairs(
      candidates = listOf("bad raw", "good one", "good two"),
      originalLine = "    broken",
      completionLimit = 10,
      isCurrent = { true },
      sourceWithLine = { "document<$it>" },
      isSemanticallyAdmissible = { source ->
        events += "check:$source"
        "bad raw" !in source && "formatted two" !in source
      },
      formatCandidate = { candidate ->
        events += "format:$candidate"
        when (candidate) {
          "good one" -> "formatted one"
          "good two" -> "formatted two"
          else -> error("A rejected candidate must not be formatted")
        }
      }
    )

    assertEquals(listOf("    formatted one", "    good two"), repairs)
    assertEquals(
      listOf(
        "check:document<    bad raw>",
        "check:document<    good one>",
        "check:document<    good two>",
        "format:good one",
        "check:document<    formatted one>",
        "format:good two",
        "check:document<    formatted two>"
      ),
      events
    )
    assertFalse(events.any { it == "format:bad raw" })
  }

  @Test
  fun exhaustsSemanticAndFormattingPassesBeforeApplyingDisplayLimit() {
    val rawChecks = mutableListOf<String>()
    val formattedChecks = mutableListOf<String>()
    val formatted = mutableListOf<String>()
    val candidates = (1..5).map { "raw-$it" }

    val repairs = semanticallyAdmissibleRepairs(
      candidates = candidates,
      originalLine = "broken",
      completionLimit = 2,
      isCurrent = { true },
      sourceWithLine = { it },
      isSemanticallyAdmissible = { source ->
        if (source.startsWith("raw-")) rawChecks += source else formattedChecks += source
        true
      },
      formatCandidate = { candidate ->
        formatted += candidate
        "formatted-${candidate.removePrefix("raw-")}"
      }
    )

    assertEquals(listOf("formatted-1", "formatted-2"), repairs)
    assertEquals(candidates, rawChecks)
    assertEquals(candidates, formatted)
    assertEquals((1..5).map { "formatted-$it" }, formattedChecks)
  }

  @Test
  fun formattedDuplicatesDoNotSuppressLaterUniqueRepairs() {
    val repairs = semanticallyAdmissibleRepairs(
      candidates = listOf("first", "first", "second", "third"),
      originalLine = "broken",
      completionLimit = 10,
      isCurrent = { true },
      sourceWithLine = { it },
      isSemanticallyAdmissible = { true },
      formatCandidate = {
        when (it) {
          "first", "second" -> "canonical"
          else -> "unique"
        }
      }
    )

    assertEquals(listOf("canonical", "unique"), repairs)
  }

  @Test
  fun cancellationDiscardsPartialResults() {
    var currentChecks = 0
    val repairs = semanticallyAdmissibleRepairs(
      candidates = listOf("first", "second"),
      originalLine = "broken",
      completionLimit = 10,
      isCurrent = { ++currentChecks < 4 },
      sourceWithLine = { it },
      isSemanticallyAdmissible = { true },
      formatCandidate = { error("Cancellation before formatting must stop the second pass") }
    )

    assertEquals(emptyList(), repairs)
  }
}
