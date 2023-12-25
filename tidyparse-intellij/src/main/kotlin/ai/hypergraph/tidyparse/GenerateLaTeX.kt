package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.levenshteinAlign

fun main() {
  println(generateDiff("1 + 22 + 4", listOf("1 + 3 + 4", "1 + 33 + 4", "1 + 22 + 4")))
}

fun generateDiff(original: String, variants: List<String>): String = """
  \begin{tcolorbox}[left skip=0.7cm,
  top=0.1cm,
  middle=0mm,
  boxsep=0mm,
  underlay unbroken and first={%
  \path[draw=none] (interior.north west) rectangle node[white]{\includegraphics[width=4mm]{../figures/tidyparse_logo.png}} ([xshift=-10mm,yshift=-9mm]interior.north west);
  }]
  \begin{lstlisting} [language=tidy, basicstyle=\ttfamily\small, escapeinside={(*@}{@*)}]
  $original
  \end{lstlisting}
  \tcblower
  \begin{lstlisting} [language=tidy, basicstyle=\ttfamily\small, escapeinside={(*@}{@*)}]
  """.trimIndent() +
  variants.joinToString("\n", "\n") { diffAsLatex(original.map { "$it" }, it.map { "$it" }) } +
  """
  \end{lstlisting}
  \end{tcolorbox}
  """.trimIndent()

fun diffAsLatex(l1: List<String>, l2: List<String>): String =
  levenshteinAlign(l1, l2).joinToString("") { (a, b) ->
    when {
      a == null -> "(*@\\hlgreen{$b}@*)"
      b == null -> "(*@\\hlred{$a}@*)"
      a != b -> "(*@\\hlorange{$b}@*)"
      else -> b
    }
  }.replace("&gt;", ">").replace("&lt;", "<")