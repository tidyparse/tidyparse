package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.kaliningraph.image.escapeHTML

const val LEV_EDIT_MATCH = 0
const val LEV_EDIT_INSERT = 1
const val LEV_EDIT_DELETE = 2
const val LEV_EDIT_SUBSTITUTE = 3

/**
 * Integer-backed intersection results. Each row retains the GPU packet header
 * followed by packed terminal IDs, so text and edit markup are created only
 * when requested. The List facade is for string-only consumers; collection
 * operations on it necessarily render text and discard the packet metadata.
 */
class IntersectionResults(
  private val rows: List<IntArray>,
  private val terminals: List<String>
) : AbstractList<String>() {
  private val terminalLengths = IntArray(terminals.size) { terminal ->
    terminals[terminal].count { !it.isWhitespace() }
  }

  override val size: Int get() = rows.size
  override fun get(index: Int): String = buildString {
    val row = rows[index]
    for (i in PKT_HDR_LEN until row.size) {
      if (i > PKT_HDR_LEN) append(' ')
      append(terminals[row[i].terminalId()])
    }
  }

  fun editDistanceAt(index: Int): Int = rows[index][0]
  fun scoreAt(index: Int): UInt = rows[index][1].toUInt()

  /** Number of non-whitespace characters in the corresponding plain result. */
  fun characterLengthAt(index: Int): Int {
    val row = rows[index]
    var length = 0
    for (i in PKT_HDR_LEN until row.size) length += terminalLengths[row[i].terminalId()]
    return length
  }

  fun editScriptAt(index: Int): List<Int> = rows[index].unpackEdits().toList()

  /** Escaped, rendered HTML generated directly from the packed row. */
  fun htmlAt(index: Int): String {
    val row = rows[index]
    var encodedEdits = 0
    return buildString {
      fun appendPart(part: String) {
        if (isNotEmpty()) append(' ')
        append(part)
      }

      for (i in PKT_HDR_LEN until row.size) {
        val packed = row[i]
        val terminal = terminals[packed.terminalId()].escapeHTML()
        when (val tag = packed.editTag()) {
          PACKED_INSERTION_TAG -> {
            encodedEdits++
            appendPart("<ins>$terminal</ins>")
          }
          PACKED_SUBSTITUTION_TAG -> {
            encodedEdits++
            appendPart("<sub>$terminal</sub>")
          }
          in PACKED_FIRST_DELETION_TAG..PACKED_LAST_DELETION_TAG -> {
            val deletions = tag - PACKED_SUBSTITUTION_TAG
            encodedEdits += deletions
            repeat(deletions) { appendPart("<del></del>") }
            appendPart(terminal)
          }
          else -> appendPart(terminal)
        }
      }

      repeat((row[0] - encodedEdits).coerceAtLeast(0)) { appendPart("<del></del>") }
    }
  }

  fun mapTerminals(transform: (String) -> String): IntersectionResults =
    IntersectionResults(rows, terminals.map(transform))

  internal fun takeResults(count: Int): IntersectionResults = selectResults(rows.indices.take(count))

  internal fun selectResults(indices: List<Int>): IntersectionResults =
    IntersectionResults(indices.map(rows::get), terminals)

  internal fun sameTokens(index: Int, other: IntersectionResults, otherIndex: Int): Boolean =
    rows[index].sameTokens(other.rows[otherIndex])

  fun terminalCountAt(index: Int): Int = rows[index].size - PKT_HDR_LEN

  fun terminalTextAt(row: Int, position: Int): String =
    terminals[rows[row][position + PKT_HDR_LEN].terminalId()]

  internal operator fun plus(other: IntersectionResults): IntersectionResults = when {
    isEmpty() -> other
    other.isEmpty() -> this
    else -> {
      require(terminals == other.terminals) { "Cannot merge results from different grammars" }
      IntersectionResults(rows + other.rows, terminals)
    }
  }

  companion object { val EMPTY = IntersectionResults(emptyList(), emptyList()) }
}

internal fun Int.terminalId(): Int = ((toUInt() and PACKED_TOKEN_MASK.toUInt()).toInt() - 1)

internal fun Int.editTag(): Int = (toUInt() shr PACKED_EDIT_SHIFT).toInt()

internal fun IntArray.tokenHash(): Int {
  var hash = 1
  for (i in PKT_HDR_LEN until size) hash = 31 * hash + this[i].terminalId()
  return hash
}

internal fun IntArray.sameTokens(other: IntArray): Boolean {
  if (size != other.size) return false
  for (i in PKT_HDR_LEN until size) {
    if (this[i].terminalId() != other[i].terminalId()) return false
  }
  return true
}

internal fun IntArray.unpackEdits(): IntArray {
  val distance = this[0]
  val edits = IntArray(size - PKT_HDR_LEN + distance)
  var editIndex = 0
  var encodedEdits = 0

  for (i in PKT_HDR_LEN until size) {
    when (val tag = this[i].editTag()) {
      PACKED_INSERTION_TAG -> {
        encodedEdits++
        edits[editIndex++] = LEV_EDIT_INSERT
      }
      PACKED_SUBSTITUTION_TAG -> {
        encodedEdits++
        edits[editIndex++] = LEV_EDIT_SUBSTITUTE
      }
      in PACKED_FIRST_DELETION_TAG..PACKED_LAST_DELETION_TAG -> {
        val deletions = tag - PACKED_SUBSTITUTION_TAG
        encodedEdits += deletions
        repeat(deletions) { edits[editIndex++] = LEV_EDIT_DELETE }
        edits[editIndex++] = LEV_EDIT_MATCH
      }
      else -> edits[editIndex++] = LEV_EDIT_MATCH
    }
  }

  repeat((distance - encodedEdits).coerceAtLeast(0)) { edits[editIndex++] = LEV_EDIT_DELETE }
  return edits.copyOf(editIndex)
}
