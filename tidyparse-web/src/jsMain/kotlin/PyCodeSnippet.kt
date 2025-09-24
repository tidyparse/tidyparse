import ai.hypergraph.kaliningraph.image.escapeHTML
import com.strumenta.antlrkotlin.parsers.generated.Python3Lexer
import org.antlr.v4.kotlinruntime.*

data class PyCodeSnippet(val rawCode: String) {
  // Keep all tokens, including whitespace/comments. The hidden channel is included by default.
  val tokens: List<Token> =
    Python3Lexer(CharStreams.fromString(rawCode)).allTokens
      .filter { it.text?.isNotBlank() == true }
      .takeWhile { it.type != Token.EOF }
      .toList()

  /**
   * Returns just the ANTLR "names" of the lexed tokens, e.g. "NAME", "FOR", "IN", etc.
   * (matching the Python3Lexer vocabulary).
   */
  fun lexedTokens(): String =
    tokens.joinToString(" ") { Python3Lexer.VOCABULARY.getDisplayName(it.type) }
      .replace("'", "").replace("NEWLINE", "") + " NEWLINE"

  private enum class Paint { NONE, GREEN, ORANGE }

  /**
   * Paints a Levenshtein-aligned patch onto the original code, highlighting:
   *  - inserted tokens in green,
   *  - deleted tokens as a gray 'blank' of the same length,
   *  - substituted tokens in orange,
   *  - identical tokens as plain text.
   *
   * The patch is given as a list of (oldTokenType?, newTokenType?) pairs.
   */
  fun paintDiff(levAlignedPatch: List<Pair<String?, String?>>, format: (String) -> String): String {
    data class Seg(val text: String, val paint: Paint, val delBefore: Int)

    // Build display segments (B-side) + how many deletions precede each segment.
    val segs = ArrayList<Seg>()
    var pendingDel = 0
    var origIdx = 0

    for ((oldTok, newTok) in levAlignedPatch) {
      when {
        // Deletion: count it, consume an original token
        oldTok != null && newTok == null -> {
          pendingDel += 1
          origIdx += 1
        }

        // Insertion / Substitution / Match
        newTok != null -> {
          val paint = when {
            oldTok == null || oldTok == "_" -> Paint.GREEN
            oldTok != newTok                -> Paint.ORANGE
            else                            -> Paint.NONE
          }

          // Preserve the exact lexeme from rawCode for matches
          val text = if (oldTok != null && oldTok == newTok) {
            tokens.getOrNull(origIdx)?.text ?: newTok
          } else newTok

          segs += Seg(text = text, paint = paint, delBefore = pendingDel)
          pendingDel = 0
          if (oldTok != null) origIdx += 1
        }
      }
    }

    val trailingDel = pendingDel

    // Let external formatter adjust whitespace freely.
    val newJoined = segs.joinToString(" ") { it.text }
    val formatted = format(newJoined)

    // Build mask over the concatenated non-WS chars.
    val segNonWS = segs.map { s -> s.text.count { !it.isWhitespace() } }
    val mask = IntArray(segNonWS.sum())
    run {
      var k = 0
      for (seg in segs) {
        val v = when (seg.paint) { Paint.NONE -> 0; Paint.GREEN -> 1; Paint.ORANGE -> 2 }
        for (ch in seg.text) if (!ch.isWhitespace()) mask[k++] = v
      }
    }

    // Stream formatted text, painting only non-WS and injecting red spaces for deletions.
    fun openTag(kind: Int, out: StringBuilder) {
      when (kind) {
        1 -> out.append("""<span style="color: green">""")
        2 -> out.append("""<span style="color: orange">""")
      }
    }
    fun closeTag(out: StringBuilder) { out.append("</span>") }
    fun appendDeletionSpaces(n: Int, out: StringBuilder) =
      repeat(n) { out.append("""<span style="background-color: gray"><span class="noselect"> </span></span>""") }

    val out = StringBuilder(formatted.length + 64)
    var open = 0        // 0/1/2 per mask
    var k = 0           // index into mask
    var segIdx = 0      // which segment we are in (by non-WS count)
    var posInSeg = 0    // non-WS chars consumed of current segment
    var delInsertedForCurrentSeg = false

    for (ch in formatted) {
      if (ch.isWhitespace()) {
        if (open != 0) { closeTag(out); open = 0 }
        out.append(ch.toString().escapeHTML())
        // whitespace does not advance mask/segment counters
        continue
      }

      // Before the first non-WS char of a segment, inject deletion markers that precede it.
      if (segIdx < segs.size && posInSeg == 0 && !delInsertedForCurrentSeg) {
        val delCount = segs[segIdx].delBefore
        if (delCount > 0) {
          if (open != 0) { closeTag(out); open = 0 } // ensure deletions are not inside a color span
          appendDeletionSpaces(delCount, out)
        }
        delInsertedForCurrentSeg = true
      }

      // Paint current non-WS char according to mask.
      val want = if (k < mask.size) mask[k] else 0
      if (want != open) {
        if (open != 0) closeTag(out)
        if (want != 0) openTag(want, out)
        open = want
      }
      out.append(ch.toString().escapeHTML())
      k += 1
      posInSeg += 1

      // End of this segment's non-WS stream?
      if (segIdx < segs.size && posInSeg == segNonWS[segIdx]) {
        segIdx += 1
        posInSeg = 0
        delInsertedForCurrentSeg = false
      }
    }
    if (open != 0) closeTag(out)

    // Trailing deletions (after the last segment)
    if (trailingDel > 0) appendDeletionSpaces(trailingDel, out)

    return out.toString().replace("\n", " ")
  }
}