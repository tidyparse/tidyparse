package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.kaliningraph.parsing.PTree
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.readIndices
import web.gpu.GPUBuffer

/** See [PTree.sampleStrWithoutReplacement] for CPU version. */
internal suspend fun uniformIndex(
  numStates: Int, numNTs: Int,
  dp: GPUBuffer, counts: GPUBuffer, offsets: GPUBuffer, storage: GPUBuffer, cdf: GPUBuffer,
  terminals: GPUBuffer, indices: GPUBuffer
): GPUBuffer = GPUBufferScope().use { owned ->
  val nodes = numStates.toLong() * (numStates - 1) / 2 * numNTs + 1 // Node zero is epsilon.
  val roots = (indices.size.toLong() / 4 - 8) / 2
  val choiceCounts = owned.newBuffer((nodes + 1) * 4)
  uniform_choice_counts(dp, counts, terminals, indices, choiceCounts)
    .dispatchFlat(((dp.size.toLong() / 4 + 255) / 256).toInt())
  val choiceOffsets = owned.own(Shader.prefixSumGPU(choiceCounts, (nodes + 1).toInt()))
  val choices = choiceOffsets.readIndices(listOf(nodes.toInt()))[0].toUInt().toLong()
  val bytes = (5 + 2 * nodes + 4 * choices + 3 * roots) * 4
  val limit = minOf(gpu.limits.maxBufferSize.toLong(), gpu.limits.maxStorageBufferBindingSize.toLong())
  require(bytes <= limit) { "The uniform bijection needs $bytes bytes, exceeding the WebGPU buffer limit of $limit" }
  val result = owned.newBuffer(bytes)
  gpu.queue.writeBuffer(result, 0.0, JSIntArray(4).apply {
    set(arrayOf(nodes.toInt(), choices.toInt(), roots.toInt(), 0), 0)
  })
  val encoder = gpu.createCommandEncoder()
  encoder.copyBufferToBuffer(choiceOffsets, 0.0, result, 16.0, choiceOffsets.size)
  gpu.queue.submit(arrayOf(encoder.finish()))
  uniform_index(dp, counts, offsets, storage, cdf, terminals, indices, result)
    .dispatchFlat(((dp.size.toLong() / 4 + 63) / 64).toInt())
  owned.detach(result)
}

//language=wgsl
private const val UNIFORM_CHART_ROW = """
fn bij_row(cell: u32) -> u32 {
  let nt = idx_uni.numNonterminals;
  let n = idx_uni.numStates;
  let pair = cell / nt;
  let r = pair / n;
  return (r * (2u * n - r - 1u) / 2u + pair % n - r - 1u) * nt + cell % nt;
}
"""

//language=wgsl
internal val uniform_choice_counts by Shader("""$TERM_STRUCT $UNIFORM_CHART_ROW
@group(0) @binding(0) var<storage, read> dp: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> terminals: Terminals;
@group(0) @binding(3) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x + gid.y * ${DISPATCH_GROUP_SIZE_X}u * 256u;
  if (cell >= arrayLength(&dp)) { return; }
  let pair = cell / idx_uni.numNonterminals;
  if (pair / idx_uni.numStates >= pair % idx_uni.numStates) { return; }
  var count = 0u;
  if (dp[cell] != 0u) { count = counts[cell] + count_tms(dp[cell], cell % idx_uni.numNonterminals); }
  output[bij_row(cell) + 1u] = count;
}
""")

//language=wgsl
internal val uniform_index by Shader("""$TERM_STRUCT $WORD_BIJECTION_HELPERS
$UNIFORM_CHART_ROW $PACK_EDIT_TOKEN_HELPER
@group(0) @binding(0) var<storage, read> dp: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> children: array<u32>;
@group(0) @binding(4) var<storage, read> cdf: array<u32>;
@group(0) @binding(5) var<storage, read> terminals: Terminals;
@group(0) @binding(6) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(7) var<storage, read_write> bijection: WordBijection;

fn tree_count(cell: u32) -> u32 {
  if (counts[cell] != 0u) { return cdf[offsets[cell] + counts[cell] - 1u]; }
  return count_tms(dp[cell], cell % idx_uni.numNonterminals);
}

@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let cell = gid.x + gid.y * groups.x * 64u;
  if (cell == 0u) {
    bijection.data[bij_count_base()] = 1u;
    var total = 0u;
    for (var root = 0u; root < bijection.roots; root++) {
      let start = idx_uni.startIndices[2u * root];
      total = bij_add(total, tree_count(start));
      let out = bij_roots_base() + 3u * root;
      bijection.data[out] = total;
      bijection.data[out + 1u] = bij_row(start) + 1u;
      bijection.data[out + 2u] = idx_uni.startIndices[2u * root + 1u];
    }
    bijection.rankLimit = total;
  }
  if (cell >= arrayLength(&dp) || dp[cell] == 0u) { return; }
  let nt = cell % idx_uni.numNonterminals;
  let pair = cell / idx_uni.numNonterminals;
  if (pair / idx_uni.numStates >= pair % idx_uni.numStates) { return; }
  let node = bij_row(cell) + 1u;
  bijection.data[bij_count_base() + node] = tree_count(cell);
  let first = bij_first(node);
  let literals = count_tms(dp[cell], nt);
  let predicate = dp[cell] & PREDICATE_MASK;
  for (var i = 0u; i < literals; i++) {
    var terminal = i;
    if (predicate != LIT_ALL) {
      let excluded = ((predicate >> 1u) & 0x03ffffffu) - 1u;
      if ((predicate & NEG_BIT) == 0u) { terminal = excluded; }
      else if (i >= excluded) { terminal = i + 1u; }
    }
    let out = bij_choice_base() + 4u * (first + i);
    bijection.data[out] = i + 1u;
    bijection.data[out + 1u] = BIJ_LEAF;
    bijection.data[out + 2u] = packEditToken(get_all_tms(get_offsets(nt) + terminal) + 1u, dp[cell]);
  }
  for (var i = 0u; i < counts[cell]; i++) {
    let edge = offsets[cell] + i;
    let out = bij_choice_base() + 4u * (first + literals + i);
    bijection.data[out] = cdf[edge];
    bijection.data[out + 1u] = bij_row(children[2u * edge]) + 1u;
    bijection.data[out + 2u] = bij_row(children[2u * edge + 1u]) + 1u;
  }
}
""")