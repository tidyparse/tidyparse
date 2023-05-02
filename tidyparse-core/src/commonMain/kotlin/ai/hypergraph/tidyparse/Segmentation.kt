package ai.hypergraph.tidyparse


data class Segmentation(
  val valid: List<Int> = emptyList(),
  val invalid: List<Int> = emptyList(),
  val illegal: List<Int> = emptyList(),
  val line: String = ""
) {
  val parseableRegions = valid.map { it..it }.mergeContiguousRanges().map { it.charIndicesOfWordsInString(line) }
  val unparseableRegions = invalid.filter { it !in illegal }.map { it..it }.mergeContiguousRanges().map { it.charIndicesOfWordsInString(line) }
  val illegalRegions = illegal.map { it..it }.map { it.charIndicesOfWordsInString(line) }

  fun List<IntRange>.mergeContiguousRanges(): List<IntRange> =
    sortedBy { it.first }.fold(mutableListOf<IntRange>()) { acc, range ->
      if (acc.isEmpty()) acc.add(range)
      else if (acc.last().last + 1 >= range.first) acc[acc.lastIndex] = acc.last().first..range.last
      else acc.add(range)
      acc
    }

// Takes an IntRange of word indices and a String of words delimited by one or more whitespaces,
// and returns the corresponding IntRange of character indices in the original string.
// For example, if the input is (1..2, "a__bb___ca d e f"), the output is 3..10
  fun IntRange.charIndicesOfWordsInString(str: String): IntRange {
    // All tokens, including whitespaces
    val wordTokens = str.split("\\s+".toRegex()).filter { it.isNotEmpty() }
    val whitespaceTokens = str.split("\\S+".toRegex())

    val allTokens = wordTokens.zip(whitespaceTokens)
    val polarity = str.startsWith(wordTokens.first())
    val interwoven = allTokens.flatMap {
      if (polarity) listOf(it.first, it.second)
      else listOf(it.second, it.first)
    }

    val s = start * 2
    val l = last * 2
    val (startIdx, endIdx) = (s) to (l + 1)

    val adjust = if (startIdx == 0) 0 else 1

    val startOffset = interwoven.subList(0, startIdx).sumOf { it.length } + adjust
    val endOffset = interwoven.subList(0, endIdx + 1).sumOf { it.length }
    return startOffset..endOffset
  }

}
