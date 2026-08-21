import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.image.escapeHTML

const val LEV_EDIT_MATCH = 0
const val LEV_EDIT_INSERT = 1
const val LEV_EDIT_DELETE = 2
const val LEV_EDIT_SUBSTITUTE = 3

/**
 * Immutable CPU snapshot of one intersection query.
 *
 * [editScript] uses the public operation constants above rather than the packed
 * GPU token tags. MATCH, INSERT and SUBSTITUTE consume one token from the
 * corresponding plain result; DELETE consumes no result token.
 *
 * The List facade exists for compatibility with string-only consumers. Normal
 * collection operators such as `map` and `filter` return plain collections and
 * therefore intentionally discard this object's metadata.
 */
class IntersectionResults internal constructor(
  val plainResults: List<String>,
  val editScript: List<List<Int>>,
  /** Opaque lower-is-better decoder costs copied from the GPU packet header. */
  val scores: List<Float>
) : AbstractList<String>() {
  override val size: Int get() = plainResults.size
  override fun get(index: Int): String = plainResults[index]

  private val annotatedCache: Array<String?> by lazy { arrayOfNulls(size) }
  /** Escaped, render-ready HTML using semantic edit tags. */
  val annotatedResults: List<String> = object : AbstractList<String>() {
    override val size: Int get() = this@IntersectionResults.size
    override fun get(index: Int): String = annotatedCache[index]
      ?: annotate(plainResults[index], editScript[index])
        .also { annotatedCache[index] = it }
  }

  /** Applies a text-only normalization while keeping row metadata aligned. */
  internal fun mapPlainResults(transform: (String) -> String): IntersectionResults =
    IntersectionResults(plainResults.map(transform), editScript, scores)

  internal fun takeResults(count: Int): IntersectionResults = selectIndices(plainResults.indices.take(count))

  internal operator fun plus(other: IntersectionResults): IntersectionResults = IntersectionResults(
    plainResults + other.plainResults,
    editScript + other.editScript,
    scores + other.scores
  )

  internal fun selectResults(indices: List<Int>): IntersectionResults = selectIndices(indices)

  internal fun reorderedLike(orderedPlainResults: List<String>): IntersectionResults {
    val available = linkedMapOf<String, ArrayDeque<Int>>()
    plainResults.forEachIndexed { i, result ->
      available.getOrPut(result) { ArrayDeque() }.addLast(i)
    }
    val order = orderedPlainResults.map { result ->
      available[result]?.removeFirstOrNull()
        ?: error("Reranker returned an unknown intersection result: $result")
    }
    return selectIndices(order)
  }

  fun editDistanceAt(index: Int): Int = editScript[index].count { it != LEV_EDIT_MATCH }

  private fun selectIndices(indices: List<Int>): IntersectionResults = IntersectionResults(
    indices.map(plainResults::get),
    indices.map(editScript::get),
    indices.map(scores::get)
  )

  companion object {
    val EMPTY = IntersectionResults(emptyList(), emptyList(), emptyList())

    private fun annotate(plainResult: String, script: List<Int>): String {
      val tokens = plainResult.tokenizeByWhitespace()
      var tokenIndex = 0

      val annotated = script.joinToString(" ") { op ->
        when (op) {
          LEV_EDIT_MATCH -> tokens[tokenIndex++].escapeHTML()
          LEV_EDIT_INSERT -> "<ins>${tokens[tokenIndex++].escapeHTML()}</ins>"
          LEV_EDIT_DELETE -> "<del></del>"
          LEV_EDIT_SUBSTITUTE -> "<sub>${tokens[tokenIndex++].escapeHTML()}</sub>"
          else -> error("Unknown edit operation: $op")
        }
      }

      require(tokenIndex == tokens.size) { "Edit script does not consume the repair" }
      return annotated
    }
  }
}
