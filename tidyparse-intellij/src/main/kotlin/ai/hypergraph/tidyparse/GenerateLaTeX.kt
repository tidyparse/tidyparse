package ai.hypergraph.tidyparse

import com.github.difflib.text.*

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

private val latexDiffGenerator: DiffRowGenerator =
  DiffRowGenerator.create()
    .showInlineDiffs(true)
    .inlineDiffByWord(true)
    .newTag { f: Boolean -> if (f) "(*@\\hl{" else "}@*)" }
    .oldTag { _ -> "" }
    .build()

fun diffAsLatex(l1: List<String>, l2: List<String>): String =
  latexDiffGenerator.generateDiffRows(l1, l2).joinToString("") {
    when (it.tag) {
      DiffRow.Tag.INSERT -> it.newLine.replace("\\hl", "\\hlgreen")
        .replace("}@*)", "}\\hlgreen@*)")
      DiffRow.Tag.CHANGE -> it.newLine.replace("\\hl", "\\hlorange")
        .replace("}@*)", "}\\hlorange@*)")
      DiffRow.Tag.DELETE -> "(*@\\hlred{${it.oldLine}}@*)"
      else -> it.newLine
    }
  }.replace("&gt;", ">").replace("&lt;", "<")
   .replace("}\\hlorange@*) (*@\\hlorange{", " ")
   .replace("}\\hlgreen@*) (*@\\hlgreen{", " ")
   .replace("}\\hlorange@*)(*@\\hlorange{", "")
   .replace("}\\hlgreen@*)(*@\\hlgreen{", "")
   .replace("}\\hlorange@*)", "}@*)")
   .replace("}\\hlgreen@*)", "}@*)")