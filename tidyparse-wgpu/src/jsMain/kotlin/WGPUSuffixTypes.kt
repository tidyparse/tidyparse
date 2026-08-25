package ai.hypergraph.tidyparse.wgpu

/** Shared input types for conditioned WebGPU suffix enumeration. */
data class SuffixSlice(val terminal: String, val length: Int)

data class SuffixBatch(
  val prefix: List<String>,
  val slices: List<SuffixSlice>,
  val completeWords: List<String> = emptyList()
)
