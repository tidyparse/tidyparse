fun IntArray.toLaTeX(numStates: Int, numNTs: Int): String {
  val tikzCommands = if (numStates == 0) "" else {
    (0 until numStates).flatMap { q1_rowIndex ->
      (0 until numStates).map { q2_colIndex ->
        val isActive = if (numNTs > 0) {
          val baseFlatIndex = q1_rowIndex * numStates * numNTs + q2_colIndex * numNTs
          (0 until numNTs).any { ntIdxInSlice ->
            val currentFlatIndex = baseFlatIndex + ntIdxInSlice
            (currentFlatIndex < size && this[currentFlatIndex] != 0)
          }
        } else { false }

        val tikzX = q2_colIndex
        val tikzY = numStates - 1 - q1_rowIndex

        val fillColor = if (isActive) "black" else "white"
        "  \\path[fill=${fillColor}] (${tikzX},${tikzY}) rectangle ++(1,1);"
      }
    }.joinToString("\n")
  }

  val squareUnitSize = "0.3cm"
  return """
  \begin{tikzpicture}[x=${squareUnitSize}, y=${squareUnitSize}, draw=gray, very thin]
  ${tikzCommands.ifBlank { "% Empty grid" }}
  \end{tikzpicture}
  """.trimIndent()
}