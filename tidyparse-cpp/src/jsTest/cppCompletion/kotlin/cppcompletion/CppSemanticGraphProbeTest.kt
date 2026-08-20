package cppcompletion

import cppEditorStatementSnapshot
import kotlinx.browser.window
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test

class CppSemanticGraphProbeTest {
  @Test
  fun measureGraphCoverage(): Promise<Unit> = MainScope().promise {
    data class Case(val name: String, val source: String, val expected: Set<String>)
    val cases = listOf(
      Case(
        "containers",
        """#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
int main() {
  st
}
""",
        setOf("std::cout", "std::map", "std::set", "std::string", "std::tuple")
      ),
      Case(
        "vocabulary",
        """#include <optional>
#include <string>
#include <variant>
int main() {
  st
}
""",
        setOf(
          "std::optional", "std::string", "std::variant", "std::visit",
          "std::get_if", "std::holds_alternative"
        )
      ),
      Case(
        "ranges-memory",
        """#include <algorithm>
#include <memory>
#include <ranges>
#include <string>
#include <utility>
#include <vector>
int main() {
  st
}
""",
        setOf(
          "std::make_unique", "std::move", "std::string", "std::unique_ptr",
          "std::vector", "std::ranges::transform"
        )
      )
    )
    val client = CppBrowserClangdClient()
    cases.forEach { case ->
      val line = case.source.lineSequence().indexOfFirst { it.trim() == "st" }
      val character = case.source.lineSequence().elementAt(line).length
      val snapshot = requireNotNull(cppEditorStatementSnapshot(case.source, line, character))
      listOf("cold", "warm").forEach { run ->
        val started = window.performance.now()
        val context = client.context(case.source, line, character, 4_096, 2)
        val elapsed = window.performance.now() - started
        val names = context.completions.mapTo(hashSetOf()) { it.name }
        val covered = case.expected.intersect(names)
        val provenance = context.completions.groupingBy { it.provenance ?: "none" }.eachCount()
        val prepareStarted = window.performance.now()
        val prepared = CppCompletionGrammar().prepare(context, snapshot.stableTokens)
        val prepareMillis = window.performance.now() - prepareStarted
        val residualStarted = window.performance.now()
        val residual = prepared.generate(snapshot.stableTokens)
        val residualMillis = window.performance.now() - residualStarted
        println(
          "GRAPH_PROBE case=${case.name} run=$run limit=4096 depth=2 " +
            "millis=${elapsed.toInt()} nodes=${context.semanticGraphNodeCount} " +
            "incomplete=${context.semanticGraphIsIncomplete} " +
            "coverage=${covered.size}/${case.expected.size} missing=${case.expected - covered} " +
            "provenance=$provenance prepareMillis=${prepareMillis.toInt()} " +
            "residualMillis=${residualMillis.toInt()} rules=${residual.sourceSyntax.size}"
        )
      }
    }
  }
}
