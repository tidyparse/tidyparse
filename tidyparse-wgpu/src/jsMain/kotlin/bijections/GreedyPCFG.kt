package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.tidyparse.wgpu.Shader.Companion.readIndices
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.writeU32
import web.gpu.GPUBuffer
import kotlin.math.ln
import kotlin.math.roundToInt

/**
 * Build the historical seeded, rule-weighted branch order in the shared rank-to-word format.
 * The weights choose which half of each branch range comes first; they do not sort rule costs.
 * Ordinary sum/product alternatives retain each rule's original PCFG cost.
 */
internal suspend fun greedyPCFGIndex(
  numStates: Int, numNTs: Int,
  dp: GPUBuffer, counts: GPUBuffer, offsets: GPUBuffer, storage: GPUBuffer, cdf: GPUBuffer,
  terminals: GPUBuffer, indices: GPUBuffer, pcfg: GPUBuffer?
): GPUBuffer = GPUBufferScope().use { buffers ->
  val (bijection, rowMap, choices) =
    allocateGreedyBijection(numStates, numNTs, dp, counts, terminals, indices, buffers)
  val grammarBytes = pcfg?.size?.toLong() ?: 0L
  val weights = buffers.newBuffer(maxOf(4L, grammarBytes + choices * 8))
  bijection.writeU32(3, (grammarBytes / 4).toInt())
  if (pcfg != null) {
    val encoder = gpu.createCommandEncoder()
    encoder.copyBufferToBuffer(pcfg, 0.0, weights, 0.0, pcfg.size)
    gpu.queue.submit(arrayOf(encoder.finish()))
  }
  greedy_pcfg_order(dp, counts, offsets, storage, cdf, terminals, indices, weights, bijection, rowMap)
    .dispatchFlat(((dp.size.toLong() / 4 + 63) / 64).toInt())
  bijection.writeU32(3, 0) // The build temporarily uses rankLimit for the grammar size.
  buffers.detach(bijection)
}

private suspend fun allocateGreedyBijection(
  numStates: Int, numNTs: Int, dp: GPUBuffer, counts: GPUBuffer,
  terminals: GPUBuffer, indices: GPUBuffer, owned: GPUBufferScope
): Triple<GPUBuffer, GPUBuffer, Long> {
  val rows = numStates.toLong() * (numStates - 1) / 2 * numNTs
  val roots = (indices.size.toLong() / 4 - 8) / 2
  val groups = ((dp.size.toLong() / 4 + 255) / 256).toInt()
  val activeRows = owned.newBuffer((rows + 1) * 4)
  greedy_active_rows(dp, indices, activeRows).dispatchFlat(groups)
  val rowMap = owned.own(Shader.prefixSumGPU(activeRows, (rows + 1).toInt()))
  val nodes = rowMap.readIndices(listOf(rows.toInt()))[0].toUInt().toLong() + 1 // Node zero is epsilon.
  owned.release(activeRows)
  val choiceCounts = owned.newBuffer((nodes + 1) * 4)
  greedy_choice_counts(dp, counts, terminals, indices, choiceCounts, rowMap).dispatchFlat(groups)
  val offsets = owned.own(Shader.prefixSumGPU(choiceCounts, (nodes + 1).toInt()))
  val choices = offsets.readIndices(listOf(nodes.toInt()))[0].toUInt().toLong()
  val bytes = (5 + 2 * nodes + 4 * choices + 3 * roots) * 4
  val limit = minOf(gpu.limits.maxBufferSize.toLong(), gpu.limits.maxStorageBufferBindingSize.toLong())
  require(bytes <= limit) { "The greedy bijection needs $bytes bytes, exceeding the WebGPU buffer limit of $limit" }
  val result = owned.newBuffer(bytes)
  gpu.queue.writeBuffer(result, 0.0, JSIntArray(4).apply {
    set(arrayOf(nodes.toInt(), choices.toInt(), roots.toInt(), 0), 0)
  })
  val encoder = gpu.createCommandEncoder()
  encoder.copyBufferToBuffer(offsets, 0.0, result, 16.0, offsets.size)
  gpu.queue.submit(arrayOf(encoder.finish()))
  owned.release(choiceCounts, offsets)
  log("Greedy PCFG: $rows chart rows, ${nodes - 1} active nodes, $choices choices, ${bytes / 1024} KiB")
  return Triple(result, rowMap, choices)
}

//language=wgsl
private const val GREEDY_CHART_ROW = """
fn bij_row(cell: u32) -> u32 {
  let nt = idx_uni.numNonterminals;
  let n = idx_uni.numStates;
  let pair = cell / nt;
  let r = pair / n;
  return (r * (2u * n - r - 1u) / 2u + pair % n - r - 1u) * nt + cell % nt;
}
"""

//language=wgsl
internal val greedy_active_rows by Shader("""$IDX_UNIFORM_STRUCT $GREEDY_CHART_ROW
@group(0) @binding(0) var<storage, read> dp: array<u32>;
@group(0) @binding(1) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(2) var<storage, read_write> active_flags: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x + gid.y * ${DISPATCH_GROUP_SIZE_X}u * 256u;
  if (cell >= arrayLength(&dp) || dp[cell] == 0u) { return; }
  let pair = cell / idx_uni.numNonterminals;
  if (pair / idx_uni.numStates >= pair % idx_uni.numStates) { return; }
  active_flags[bij_row(cell)] = 1u;
}
""")

//language=wgsl
internal val greedy_choice_counts by Shader("""$TERM_STRUCT $GREEDY_CHART_ROW
@group(0) @binding(0) var<storage, read> dp: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> terminals: Terminals;
@group(0) @binding(3) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<storage, read> row_map: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x + gid.y * ${DISPATCH_GROUP_SIZE_X}u * 256u;
  if (cell >= arrayLength(&dp) || dp[cell] == 0u) { return; }
  let pair = cell / idx_uni.numNonterminals;
  if (pair / idx_uni.numStates >= pair % idx_uni.numStates) { return; }
  output[row_map[bij_row(cell)] + 1u] = counts[cell] + count_tms(dp[cell], cell % idx_uni.numNonterminals);
}
""")

// Each invocation owns one row and its scratch interval. Roots use the completed language-size
// CDF, so their initialization needs no synchronization with the newly ordered alternatives.
//language=wgsl
internal val greedy_pcfg_order by Shader("""$TERM_STRUCT
$WORD_BIJECTION_HELPERS
$GREEDY_CHART_ROW
$PACK_EDIT_TOKEN_HELPER
$WGSL_MIX32
@group(0) @binding(0) var<storage, read> dp_in: array<u32>;
@group(0) @binding(1) var<storage, read> bp_count: array<u32>;
@group(0) @binding(2) var<storage, read> bp_offset: array<u32>;
@group(0) @binding(3) var<storage, read> bp_storage: array<u32>;
@group(0) @binding(4) var<storage, read> ls_sparse: array<u32>;
@group(0) @binding(5) var<storage, read> terminals: Terminals;
@group(0) @binding(6) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(7) var<storage, read_write> weights: array<u32>;
@group(0) @binding(8) var<storage, read_write> bijection: WordBijection;
@group(0) @binding(9) var<storage, read> row_map: array<u32>;

fn greedy_node(cell: u32) -> u32 { return row_map[bij_row(cell)] + 1u; }

fn greedy_size(cell: u32) -> u32 {
  let count = bp_count[cell];
  if (count != 0u) { return ls_sparse[bp_offset[cell] + count - 1u]; }
  return count_tms(dp_in[cell], cell % idx_uni.numNonterminals);
}

fn greedy_token(cell: u32, variant: u32) -> u32 {
  let nt = cell % idx_uni.numNonterminals;
  let predicate = dp_in[cell] & PREDICATE_MASK;
  var terminal = variant;
  if (predicate != LIT_ALL) {
    let excluded = ((predicate >> 1u) & 0x03ffffffu) - 1u;
    if ((predicate & NEG_BIT) == 0u) { terminal = excluded; }
    else if (variant >= excluded) { terminal = variant + 1u; }
  }
  return get_all_tms(get_offsets(nt) + terminal);
}

// Grammar records are (left, right, weight, cost); null PCFG uses equal weights and zero costs.
fn greedy_rule(nt: u32, left: u32, right: u32) -> vec2<u32> {
  if (bijection.rankLimit == 0u) { return vec2<u32>(1u, 0u); }
  let start = weights[nt];
  var lo = 0u;
  var hi = (weights[nt + 1u] - start) / 4u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    let pos = start + 4u * mid;
    let l = weights[pos];
    let r = weights[pos + 1u];
    if (l == left && r == right) { return vec2<u32>(weights[pos + 2u], weights[pos + 3u]); }
    if (l < left || (l == left && r < right)) { lo = mid + 1u; }
    else { hi = mid; }
  }
  return vec2<u32>(1u, ${(-ln(0.001) * SCALE).roundToInt()}u);
}

fn greedy_branch_rule(cell: u32, literals: u32, branch: u32) -> vec2<u32> {
  let nt = cell % idx_uni.numNonterminals;
  if (branch < literals) { return greedy_rule(nt, greedy_token(cell, branch), 0xffffffffu); }
  let edge = 2u * (bp_offset[cell] + branch - literals);
  return greedy_rule(nt, bp_storage[edge] % idx_uni.numNonterminals,
    bp_storage[edge + 1u] % idx_uni.numNonterminals);
}

fn greedy_branch_size(cell: u32, literals: u32, branch: u32) -> u32 {
  if (branch < literals) { return 1u; }
  let edge = 2u * (bp_offset[cell] + branch - literals);
  return bij_mul(greedy_size(bp_storage[edge]), greedy_size(bp_storage[edge + 1u]));
}

fn greedy_write_choice(cell: u32, literals: u32, branch: u32, choice: u32, end: u32, cost: u32) {
  let base = bij_choice_base() + 4u * choice;
  bijection.data[base] = end;
  if (branch < literals) {
    bijection.data[base + 1u] = 0xffffffffu;
    bijection.data[base + 2u] = packEditToken(greedy_token(cell, branch) + 1u, dp_in[cell]);
  } else {
    let edge = 2u * (bp_offset[cell] + branch - literals);
    bijection.data[base + 1u] = greedy_node(bp_storage[edge]);
    bijection.data[base + 2u] = greedy_node(bp_storage[edge + 1u]);
  }
  bijection.data[base + 3u] = cost;
}

fn greedy_prefix(base: u32, end: u32) -> u32 {
  if (end == 0u) { return 0u; }
  return weights[base + 2u * (end - 1u)];
}

fn greedy_roots() {
  bijection.data[bij_count_base()] = 1u; // Node zero is the empty word.
  var cumulative = 0u;
  for (var root = 0u; root < bijection.roots; root++) {
    let cell = idx_uni.startIndices[2u * root];
    let base = bij_roots_base() + 3u * root;
    cumulative = bij_add(cumulative, greedy_size(cell));
    bijection.data[base] = cumulative;
    bijection.data[base + 1u] = greedy_node(cell);
    bijection.data[base + 2u] = idx_uni.startIndices[2u * root + 1u];
  }
}

@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let cell = gid.x + gid.y * groups.x * 64u;
  if (cell == 0u) { greedy_roots(); }
  if (cell >= arrayLength(&dp_in) || dp_in[cell] == 0u) { return; }
  let pair = cell / idx_uni.numNonterminals;
  if (pair % idx_uni.numStates <= pair / idx_uni.numStates) { return; }
  let node = greedy_node(cell);
  let first = bij_first(node);
  let count = bij_end(node) - first;
  if (count == 0u) { return; }
  let literals = count_tms(dp_in[cell], cell % idx_uni.numNonterminals);
  if (count == 1u) {
    let size = greedy_branch_size(cell, literals, 0u);
    greedy_write_choice(cell, literals, 0u, first, size, greedy_branch_rule(cell, literals, 0u).y);
    bijection.data[bij_count_base() + node] = size;
    return;
  }

  let prefix = bijection.rankLimit + 2u * first;
  var totalWeight = 0u;
  var previousWeight = 0u;
  var allEqual = true;
  for (var branch = 0u; branch < count; branch++) {
    var weight = 0u;
    if (greedy_branch_size(cell, literals, branch) != 0u) {
      let rule = greedy_branch_rule(cell, literals, branch);
      weight = rule.x;
      weights[prefix + 2u * branch + 1u] = rule.y;
    }
    allEqual = allEqual && (branch == 0u || weight == previousWeight);
    previousWeight = weight;
    totalWeight = bij_add(totalWeight, weight);
    weights[prefix + 2u * branch] = totalWeight;
  }

  let seed = atomicLoad(&idx_uni.targetCnt) ^ 0xA511E9B3u;
  var stack: array<vec2<u32>, 32>;
  stack[0] = vec2<u32>(0u, count);
  var top = 1u;
  var output = first;
  var cumulative = 0u;
  while (top != 0u) {
    top--;
    let range = stack[top];
    let lo = range.x;
    let hi = range.y;
    if (hi - lo == 1u) {
      cumulative = bij_add(cumulative, greedy_branch_size(cell, literals, lo));
      greedy_write_choice(cell, literals, lo, output, cumulative, weights[prefix + 2u * lo + 1u]);
      output++;
      continue;
    }
    let mid = (lo + hi) >> 1u;
    let random = mix32(seed ^ mix32(cell) ^ mix32(lo) ^ mix32(hi));
    var leftFirst: bool;
    if (allEqual) { leftFirst = random % (hi - lo) < mid - lo; }
    else {
      let splitWeight = greedy_prefix(prefix, mid);
      var mass = vec2<u32>(splitWeight - greedy_prefix(prefix, lo), greedy_prefix(prefix, hi) - splitWeight);
      // A saturated prefix cannot be subtracted reliably; recompute the two masses in that case.
      if (totalWeight == 0xffffffffu) {
        mass = vec2<u32>(0u);
        for (var branch = lo; branch < hi; branch++) {
          if (greedy_branch_size(cell, literals, branch) != 0u) {
            let side = select(0u, 1u, branch >= mid);
            mass[side] = bij_add(mass[side], greedy_branch_rule(cell, literals, branch).x);
          }
        }
      }
      let total = bij_add(mass.x, mass.y);
      let q = random % 1000u;
      let needle = (total / 1000u) * q + ((total % 1000u) * q) / 1000u;
      leftFirst = needle < mass.x;
    }
    let left = vec2<u32>(lo, mid);
    let right = vec2<u32>(mid, hi);
    stack[top] = select(left, right, leftFirst); top++;
    stack[top] = select(right, left, leftFirst); top++;
  }
  bijection.data[bij_count_base() + node] = cumulative;
}
""")
