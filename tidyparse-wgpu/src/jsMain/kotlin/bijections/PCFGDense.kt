package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.tidyparse.wgpu.Shader.Companion.readIndices
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.toGPUBuffer
import web.gpu.GPUBuffer
import kotlin.math.ln
import kotlin.math.roundToInt

private const val PCFG_DENSE_WORKGROUP_SIZE = 256

/** Build a histogram-ordered rank bijection. Counts, choices and packet payloads stay on the GPU. */
suspend fun histogramSemiringIndex(
  numStates: Int, numNTs: Int,
  dp: GPUBuffer, counts: GPUBuffer, offsets: GPUBuffer, storage: GPUBuffer, cdf: GPUBuffer,
  terminals: GPUBuffer, indices: GPUBuffer, pcfg: GPUBuffer?,
  bucketLimit: Int = PCFG_DENSE_WORKGROUP_SIZE
): GPUBuffer {
  if (pcfg == null) return greedyPCFGIndex(numStates, numNTs,
    dp, counts, offsets, storage, cdf, terminals, indices, null)
  require(numStates > 1 && numNTs > 0 && bucketLimit >= 3)
  val targetBytes = minOf(gpu.limits.maxBufferSize.toLong(), gpu.limits.maxStorageBufferBindingSize.toLong()) / 4
  var capacity = bucketLimit
  while (true) {
    try {
      return GPUBufferScope().use { buffers ->
        val forest = buffers.own(buildHistogramForest(numStates, numNTs,
          dp, counts, offsets, storage, cdf, terminals, indices, pcfg, capacity))
        compileHistogramBijection(forest, targetBytes)
      }
    } catch (tooLarge: HistogramBijectionCapacity) {
      if (tooLarge.bins <= 3) throw tooLarge
      capacity = maxOf(3, tooLarge.bins / 2)
      log("Histogram DAG needs ${tooLarge.bytes / 1024} KiB; rebuilding with at most $capacity buckets")
    }
  }
}

/** Construction-only histograms; [compileHistogramBijection] removes this layout before decoding. */
internal suspend fun buildHistogramForest(
  numStates: Int, numNTs: Int,
  dp: GPUBuffer, counts: GPUBuffer, offsets: GPUBuffer, storage: GPUBuffer, cdf: GPUBuffer,
  terminals: GPUBuffer, indices: GPUBuffer, pcfg: GPUBuffer,
  bucketLimit: Int = PCFG_DENSE_WORKGROUP_SIZE
): GPUBuffer {
  require(numStates > 1 && numNTs > 0 && bucketLimit >= 3)
  return GPUBufferScope().use { buffers ->
    val bijection = buffers.own(allocateHistogramForest(numStates, numNTs, dp, counts, terminals, indices, bucketLimit))
    val rows = numStates.toLong() * (numStates - 1) / 2 * numNTs
    val minima = buffers.newBuffer(rows * 4)
    val spans = (1 until numStates).map { span ->
      buffers.own(intArrayOf(span, numStates, numNTs)
        .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST))
    }
    spans.forEachIndexed { i, params ->
      pcfg_min_cost(dp, counts, offsets, storage, terminals, indices, pcfg, bijection, minima, params)
        .dispatchFlat(((numStates - i - 1) * numNTs + 63) / 64)
    }
    pcfg_scale(indices, bijection, minima)(1)
    pcfg_choice_labels(bijection, minima).dispatchFlat(((rows + 63) / 64).toInt())
    spans.forEachIndexed { i, params ->
      pcfg_dense_convolve(bijection, params).dispatchFlat((numStates - i - 1) * numNTs)
    }
    pcfg_build_root_cdf(bijection)(1)
    buffers.detach(bijection)
  }
}

// Construction-only parse-chart storage. None of these fields are part of WordBijection.
private suspend fun allocateHistogramForest(
  numStates: Int, numNTs: Int, dp: GPUBuffer, counts: GPUBuffer,
  terminals: GPUBuffer, indices: GPUBuffer, bucketLimit: Int = PCFG_DENSE_WORKGROUP_SIZE
): GPUBuffer = GPUBufferScope().use { owned ->
  require(numStates > 1 && numNTs > 0 && bucketLimit >= 3)
  val rows = numStates.toLong() * (numStates - 1) / 2 * numNTs
  val roots = (indices.size.toLong() / 4 - 8) / 2
  require(roots > 0)
  val choiceCounts = owned.newBuffer((rows + 1) * 4)
  histogram_choice_counts(dp, counts, terminals, indices, choiceCounts)
    .dispatchFlat((dp.size.toLong() / 4 + 255).toInt() / 256)
  val offsets = owned.own(Shader.prefixSumGPU(choiceCounts, (rows + 1).toInt()))
  val choices = offsets.readIndices(listOf(rows.toInt()))[0].toUInt().toLong()
  val limits = gpu.limits
  val bindingLimit = minOf(limits.maxStorageBufferBindingSize.toLong(), limits.maxBufferSize.toLong())
  fun bytes(bins: Int) = 4 * (8 + rows + 1 + 4 * choices + 3 * roots +
    (rows + choices + roots) * bins)
  val minimum = 3
  require(bytes(minimum) <= bindingLimit) {
    "The minimum word bijection needs ${bytes(minimum)} bytes, exceeding the WebGPU buffer limit of $bindingLimit"
  }
  val target = maxOf(bindingLimit / 4, bytes(minimum))
  val bins = (minOf(bucketLimit, PCFG_DENSE_WORKGROUP_SIZE) downTo minimum).first { bytes(it) <= target }
  val result = owned.newBuffer(bytes(bins))
  gpu.queue.writeBuffer(result, 0.0, JSIntArray(8).apply {
    set(arrayOf(bins, rows.toInt(), choices.toInt(), roots.toInt(), -1, 1, -1, 0), 0)
  })
  val encoder = gpu.createCommandEncoder()
  encoder.copyBufferToBuffer(offsets, 0.0, result, 32.0, offsets.size)
  gpu.queue.submit(arrayOf(encoder.finish()))
  log("PCFG histogram: $rows chart rows, $bins buckets, $choices choices, ${bytes(bins) / 1024} KiB")
  owned.detach(result)
}

private class HistogramBijectionCapacity(val bytes: Long, val bins: Int, budget: Long) :
  IllegalStateException("The compiled histogram bijection needs $bytes bytes with $bins buckets; allocation limit is $budget bytes")

/**
 * Compile construction-only histograms to ordinary sum/product nodes. Each active constituent
 * gets its bucket nodes, a shared chain of finite suffix sums, and one sum of all its words.
 * Only positive alternatives in the represented u32 prefix are materialized. The count pass
 * determines the exact allocation before any alternative records are allocated.
 */
internal suspend fun compileHistogramBijection(
  forest: GPUBuffer,
  targetBytes: Long = minOf(gpu.limits.maxBufferSize.toLong(), gpu.limits.maxStorageBufferBindingSize.toLong())
): GPUBuffer = GPUBufferScope().use { owned ->
  val header = forest.readIndices(listOf(0, 1, 3))
  val bins = header[0]
  val rows = header[1]
  val roots = header[2].toLong() * bins
  val hardLimit = minOf(gpu.limits.maxBufferSize.toLong(), gpu.limits.maxStorageBufferBindingSize.toLong())
  val budget = minOf(targetBytes, hardLimit)
  fun checkSize(bytes: Long) {
    // At the minimum resolution, allow the full per-buffer limit rather than dropping words.
    if (bytes > hardLimit || (bins > 3 && bytes > budget)) {
      throw HistogramBijectionCapacity(bytes, bins, if (bins > 3) budget else hardLimit)
    }
  }

  val activeFlags = owned.newBuffer((rows.toLong() + 1) * 4)
  histogram_compile_row_counts(forest, activeFlags).dispatchFlat((rows + 63) / 64)
  val rowMap = owned.own(Shader.prefixSumGPU(activeFlags, rows + 1))
  val active = rowMap.readIndices(listOf(rows))[0].toUInt().toLong()
  val nodes = 1 + active * 2 * bins // Node zero is epsilon.
  checkSize((5 + 2 * nodes + 3 * roots) * 4)
  val activeRows = owned.newBuffer(maxOf(4L, active * 4))
  val nodeCounts = owned.newBuffer(nodes * 4)
  histogram_compile_row_map(forest, rowMap, activeRows, nodeCounts).dispatchFlat((rows + 63) / 64)
  owned.release(activeFlags)

  val alternativeCounts = owned.newBuffer((nodes + 1) * 4)
  val dummyOffsets = owned.newBuffer(4)
  val dummyOutput = owned.newBuffer(4)
  val countParams = owned.own(intArrayOf(0, nodes.toInt(), 0, 0)
    .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST))
  histogram_compile_nodes(forest, rowMap, activeRows, nodeCounts,
    alternativeCounts, dummyOffsets, dummyOutput, countParams).dispatchFlat(((nodes + 63) / 64).toInt())
  val alternativeOffsets = owned.own(Shader.prefixSumGPU(alternativeCounts, (nodes + 1).toInt()))
  val alternatives = alternativeOffsets.readIndices(listOf(nodes.toInt()))[0].toUInt().toLong()
  val bytes = (5 + 2 * nodes + 4 * alternatives + 3 * roots) * 4
  checkSize(bytes)
  val result = owned.newBuffer(bytes)
  gpu.queue.writeBuffer(result, 0.0, JSIntArray(4).apply {
    set(arrayOf(nodes.toInt(), alternatives.toInt(), roots.toInt(), 0), 0)
  })
  val encoder = gpu.createCommandEncoder()
  encoder.copyBufferToBuffer(alternativeOffsets, 0.0, result, 16.0, alternativeOffsets.size)
  encoder.copyBufferToBuffer(nodeCounts, 0.0, result, ((5 + nodes) * 4).toDouble(), nodeCounts.size)
  gpu.queue.submit(arrayOf(encoder.finish()))

  val writeParams = owned.own(intArrayOf(1, nodes.toInt(), 0, 0)
    .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST))
  histogram_compile_nodes(forest, rowMap, activeRows, nodeCounts,
    alternativeCounts, alternativeOffsets, result, writeParams).dispatchFlat(((nodes + 63) / 64).toInt())
  histogram_compile_roots(forest, rowMap, result).dispatchFlat(((roots + 63) / 64).toInt())
  log("Compiled histogram: $nodes nodes, $alternatives choices, ${bytes / 1024} KiB")
  owned.detach(result)
}

//language=wgsl
private const val HISTOGRAM_FOREST_STRUCT = """
struct HistogramForest {
  bins: u32, rows: u32, choices: u32, roots: u32,
  limit: u32, quantum: u32, best: u32, reserved: u32,
  data: array<u32>
}
"""

//language=wgsl
private const val HISTOGRAM_CHART_ROW = """
fn bij_row(cell: u32) -> u32 {
  let nt = idx_uni.numNonterminals;
  let n = idx_uni.numStates;
  let pair = cell / nt;
  let r = pair / n;
  return (r * (2u * n - r - 1u) / 2u + pair % n - r - 1u) * nt + cell % nt;
}
"""

//language=wgsl
private const val HISTOGRAM_FOREST_HELPERS = """$HISTOGRAM_FOREST_STRUCT
const BIJ_LEAF: u32 = 0xffffffffu;
fn bij_add(a: u32, b: u32) -> u32 { return a + min(b, bijection.limit - a); }
fn bij_mul(a: u32, b: u32) -> u32 {
  if (a == 0u || b == 0u) { return 0u; }
  if (a > bijection.limit / b) { return bijection.limit; }
  return a * b;
}
fn bij_choice_base() -> u32 { return bijection.rows + 1u; }
fn bij_hist_base() -> u32 { return bij_choice_base() + 4u * bijection.choices; }
fn bij_cdf_base() -> u32 { return bij_hist_base() + bijection.rows * bijection.bins; }
fn bij_roots_base() -> u32 { return bij_cdf_base() + bijection.choices * bijection.bins; }
fn bij_root_cdf_base() -> u32 { return bij_roots_base() + 3u * bijection.roots; }
fn bij_first(row: u32) -> u32 { return bijection.data[row]; }
fn bij_end(row: u32) -> u32 { return bijection.data[row + 1u]; }
fn bij_choice(choice: u32) -> vec4<u32> {
  let p = bij_choice_base() + 4u * choice;
  return vec4<u32>(bijection.data[p], bijection.data[p+1u], bijection.data[p+2u], bijection.data[p+3u]);
}
fn bij_hist(row: u32, bin: u32) -> u32 {
  // An empty word is the unit of concatenation, used for the unary root alternatives.
  if (row == BIJ_LEAF) { return select(0u, 1u, bin == 0u); }
  return bijection.data[bij_hist_base() + row * bijection.bins + bin];
}
fn bij_cdf(choice: u32, bin: u32) -> u32 {
  return bijection.data[bij_cdf_base() + choice * bijection.bins + bin];
}
fn bij_root(root: u32) -> vec3<u32> {
  let p = bij_roots_base() + 3u * root;
  return vec3<u32>(bijection.data[p], bijection.data[p+1u], bijection.data[p+2u]);
}
fn bij_root_cdf(entry: u32) -> u32 { return bijection.data[bij_root_cdf_base() + entry]; }
"""

//language=wgsl
internal val histogram_choice_counts by Shader("""$TERM_STRUCT $HISTOGRAM_CHART_ROW
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
  output[bij_row(cell)] = count;
}
""")

// Sparse PCFG records contain (left, right, weight, original cost).
//language=wgsl
private val PCFG_COST_HELPERS = """
fn pcfg_rule_cost(nt: u32, left: u32, right: u32) -> u32 {
  let start = pcfg[nt];
  var lo = 0u;
  var hi = (pcfg[nt + 1u] - start) / 4u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    let pos = start + 4u * mid;
    let l = pcfg[pos];
    let r = pcfg[pos + 1u];
    if (l == left && r == right) { return pcfg[pos + 3u]; }
    if (l < left || (l == left && r < right)) { lo = mid + 1u; }
    else { hi = mid; }
  }
  return ${(-ln(0.001) * SCALE).roundToInt()}u;
}

// variant is the branch ordinal among count_tms(value, nt) permitted terminals.
fn pcfg_literal_token(nt: u32, value: u32, variant: u32) -> u32 {
  let predicate = value & PREDICATE_MASK;
  var terminal = variant;
  if (predicate != LIT_ALL) {
    let excluded = ((predicate >> 1u) & 0x03ffffffu) - 1u;
    if ((predicate & NEG_BIT) == 0u) { terminal = excluded; }
    else if (variant >= excluded) { terminal = variant + 1u; }
  }
  return get_all_tms(get_offsets(nt) + terminal);
}
"""

//language=wgsl
private const val PCFG_HISTOGRAM_HELPERS = """
const PCFG_INF: u32 = 0xffffffffu;
fn pcfg_add_cost(a: u32, b: u32) -> u32 { return a + min(b, PCFG_INF - a); }
fn pcfg_cost_bin(cost: u32, base: u32) -> u32 {
  if (cost == PCFG_INF || base == PCFG_INF) { return bijection.bins - 1u; }
  let excess = cost - base;
  return min(bijection.bins - 2u,
    excess / bijection.quantum + select(0u, 1u, excess % bijection.quantum != 0u));
}
"""

// The minima pass also stores each leaf's final token/edit payload and each product's original cost.
//language=wgsl
internal val pcfg_min_cost by Shader("""$TERM_STRUCT
$HISTOGRAM_FOREST_HELPERS $HISTOGRAM_CHART_ROW
$PCFG_HISTOGRAM_HELPERS $PCFG_COST_HELPERS $PACK_EDIT_TOKEN_HELPER
@group(0) @binding(0) var<storage, read> dp: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> bp_storage: array<u32>;
@group(0) @binding(4) var<storage, read> terminals: Terminals;
@group(0) @binding(5) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(6) var<storage, read> pcfg: array<u32>;
@group(0) @binding(7) var<storage, read_write> bijection: HistogramForest;
@group(0) @binding(8) var<storage, read_write> minima: array<u32>;
struct PCFGSpan { span: u32, numStates: u32, numNTs: u32 }
@group(0) @binding(9) var<uniform> params: PCFGSpan;

@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let flat = gid.x + gid.y * groups.x * 64u;
  let nt = flat % params.numNTs;
  let r = flat / params.numNTs;
  let c = r + params.span;
  if (c >= params.numStates) { return; }
  let cell = (r * params.numStates + c) * params.numNTs + nt;
  let row = bij_row(cell);
  let value = dp[cell];
  var best = PCFG_INF;
  if (value != 0u) {
    let literals = count_tms(value, nt);
    let first = bij_first(row);
    for (var literal = 0u; literal < literals; literal++) {
      let terminal = pcfg_literal_token(nt, value, literal);
      let cost = pcfg_rule_cost(nt, terminal, PCFG_INF);
      let out = bij_choice_base() + 4u * (first + literal);
      bijection.data[out] = PCFG_INF;
      bijection.data[out + 1u] = packEditToken(terminal + 1u, value);
      bijection.data[out + 2u] = 0u;
      bijection.data[out + 3u] = cost;
      best = min(best, cost);
    }
    for (var branch = 0u; branch < counts[cell]; branch++) {
      let edge = offsets[cell] + branch;
      let left = bp_storage[2u * edge];
      let right = bp_storage[2u * edge + 1u];
      let lrow = bij_row(left);
      let rrow = bij_row(right);
      let cost = pcfg_rule_cost(nt, left % params.numNTs, right % params.numNTs);
      let out = bij_choice_base() + 4u * (first + literals + branch);
      bijection.data[out] = lrow;
      bijection.data[out + 1u] = rrow;
      bijection.data[out + 2u] = 0u;
      bijection.data[out + 3u] = cost;
      best = min(best, pcfg_add_cost(cost, pcfg_add_cost(minima[lrow], minima[rrow])));
    }
  }
  minima[row] = best;
}
""")

// The resolution depends on the buffer capacity, not on the requested prefix length.
//language=wgsl
internal val pcfg_scale by Shader("""$IDX_UNIFORM_STRUCT
$HISTOGRAM_FOREST_HELPERS $HISTOGRAM_CHART_ROW $PCFG_HISTOGRAM_HELPERS
@group(0) @binding(0) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(1) var<storage, read_write> bijection: HistogramForest;
@group(0) @binding(2) var<storage, read> minima: array<u32>;

@compute @workgroup_size(1) fn main() {
  var best = PCFG_INF;
  for (var root = 0u; root < bijection.roots; root++) {
    best = min(best, minima[bij_row(idx_uni.startIndices[2u * root])]);
  }
  bijection.best = best;
  let window = ${SCALE} * log(${MAX_SAMPLES}.0);
  bijection.quantum = max(1u, u32(ceil(window / f32(bijection.bins - 2u))));
  for (var root = 0u; root < bijection.roots; root++) {
    let row = bij_row(idx_uni.startIndices[2u * root]);
    let out = bij_roots_base() + 3u * root;
    bijection.data[out] = row;
    bijection.data[out + 1u] = pcfg_cost_bin(minima[row], best);
    bijection.data[out + 2u] = idx_uni.startIndices[2u * root + 1u];
  }
}
""")

// Finalize immutable labels before the span wavefront reads any choices.
//language=wgsl
internal val pcfg_choice_labels by Shader("""
$HISTOGRAM_FOREST_HELPERS $PCFG_HISTOGRAM_HELPERS
@group(0) @binding(0) var<storage, read_write> bijection: HistogramForest;
@group(0) @binding(1) var<storage, read> minima: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let row = gid.x + gid.y * groups.x * 64u;
  if (row >= bijection.rows) { return; }
  for (var c = bij_first(row); c < bij_end(row); c++) {
    let choice = bij_choice(c);
    var cost = choice.w;
    if (choice.x != PCFG_INF) {
      cost = pcfg_add_cost(cost, pcfg_add_cost(minima[choice.x], minima[choice.y]));
    }
    bijection.data[bij_choice_base() + 4u * c + 2u] = pcfg_cost_bin(cost, minima[row]);
  }
}
""")

// One workgroup per constituent; each lane owns a coefficient and its choice CDF.
// Overflow combines a growing right-hand tail, preserving O(B) work for that coefficient.
//language=wgsl
internal val pcfg_dense_convolve by Shader("""
$HISTOGRAM_FOREST_HELPERS
@group(0) @binding(0) var<storage, read_write> bijection: HistogramForest;
struct PCFGSpan { span: u32, numStates: u32, numNTs: u32 }
@group(0) @binding(1) var<uniform> params: PCFGSpan;
var<workgroup> choice_count: u32;
var<workgroup> lhs: array<u32, $PCFG_DENSE_WORKGROUP_SIZE>;
var<workgroup> rhs: array<u32, $PCFG_DENSE_WORKGROUP_SIZE>;

@compute @workgroup_size($PCFG_DENSE_WORKGROUP_SIZE) fn main(
  @builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) bin: u32
) {
  let flat = group.x + group.y * ${DISPATCH_GROUP_SIZE_X}u;
  let nt = flat % params.numNTs;
  let r = flat / params.numNTs;
  if (r + params.span >= params.numStates) { return; }
  let row = (r * (2u * params.numStates - r - 1u) / 2u + params.span - 1u) * params.numNTs + nt;
  let width = bijection.bins;
  let overflow = width - 2u;
  let infinity = width - 1u;
  let first = bij_first(row);
  if (bin == 0u) { choice_count = bij_end(row) - first; }
  let count = workgroupUniformLoad(&choice_count);
  var total = 0u;
  for (var branch = 0u; branch < count; branch++) {
    let c = first + branch;
    let choice = bij_choice(c);
    if (bin < width && choice.x != 0xffffffffu) {
      lhs[bin] = bij_hist(choice.x, bin);
      rhs[bin] = bij_hist(choice.y, bin);
    }
    workgroupBarrier();
    if (bin < width && total < bijection.limit) {
      let shift = choice.z;
      if (choice.x == 0xffffffffu) {
        if (bin == shift) { total = bij_add(total, 1u); }
      } else if (bin < overflow && shift <= bin) {
        let excess = bin - shift;
        for (var a = 0u; a <= excess && total < bijection.limit; a++) {
          total = bij_add(total, bij_mul(lhs[a], rhs[excess - a]));
        }
      } else if (bin == overflow && shift != infinity) {
        var tail = 0u;
        var cursor = overflow + 1u;
        for (var a = 0u; a <= overflow && total < bijection.limit; a++) {
          let threshold = overflow - min(overflow, shift + a);
          while (cursor > threshold) { cursor--; tail = bij_add(tail, rhs[cursor]); }
          total = bij_add(total, bij_mul(lhs[a], tail));
        }
      } else if (bin == infinity) {
        var lf = 0u;
        var rf = 0u;
        for (var a = 0u; a < infinity; a++) {
          lf = bij_add(lf, lhs[a]); rf = bij_add(rf, rhs[a]);
        }
        let rt = bij_add(rf, rhs[infinity]);
        var mass = bij_mul(bij_add(lf, lhs[infinity]), rt);
        if (shift != infinity) {
          mass = bij_add(bij_mul(lf, rhs[infinity]), bij_mul(lhs[infinity], rt));
        }
        total = bij_add(total, mass);
      }
    }
    if (bin < width) { bijection.data[bij_cdf_base() + c * width + bin] = total; }
    workgroupBarrier();
  }
  if (bin < width) { bijection.data[bij_hist_base() + row * width + bin] = total; }
}
""")

// Merge all accepting roots by global quantized score, including overflow and infinity.
//language=wgsl
internal val pcfg_build_root_cdf by Shader("""
$HISTOGRAM_FOREST_HELPERS
@group(0) @binding(0) var<storage, read_write> bijection: HistogramForest;
@compute @workgroup_size(1) fn main() {
  let overflow = bijection.bins - 2u;
  let infinity = bijection.bins - 1u;
  var total = 0u;
  for (var bin = 0u; bin < bijection.bins; bin++) {
    for (var root = 0u; root < bijection.roots; root++) {
      let entry = bij_root(root);
      let row = entry.x;
      let shift = entry.y;
      var mass = 0u;
      if (bin < overflow && shift <= bin) { mass = bij_hist(row, bin - shift); }
      else if (bin == overflow && shift != infinity) {
        for (var local = overflow - shift; local <= overflow; local++) {
          mass = bij_add(mass, bij_hist(row, local));
        }
      } else if (bin == infinity) {
        mass = bij_hist(row, infinity);
        if (shift == infinity) {
          for (var local = 0u; local < infinity; local++) { mass = bij_add(mass, bij_hist(row, local)); }
        }
      }
      total = bij_add(total, mass);
      bijection.data[bij_root_cdf_base() + bin * bijection.roots + root] = total;
    }
  }
}
""")

//language=wgsl
internal val histogram_compile_row_counts by Shader("""$HISTOGRAM_FOREST_HELPERS
@group(0) @binding(0) var<storage, read> bijection: HistogramForest;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let row = gid.x + gid.y * groups.x * 64u;
  if (row < bijection.rows) { counts[row] = select(0u, 1u, bij_first(row) != bij_end(row)); }
}
""")

// Each active row owns 2*B nodes: B buckets, B-1 finite suffixes, then all words.
// Counts are computed once here so compiling product alternatives needs only constant-time reads.
//language=wgsl
internal val histogram_compile_row_map by Shader("""$HISTOGRAM_FOREST_HELPERS
@group(0) @binding(0) var<storage, read> bijection: HistogramForest;
@group(0) @binding(1) var<storage, read> row_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> active_rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> node_counts: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let row = gid.x + gid.y * groups.x * 64u;
  if (row == 0u) { node_counts[0] = 1u; }
  if (row >= bijection.rows || bij_first(row) == bij_end(row)) { return; }
  let width = bijection.bins;
  let first = 1u + row_map[row] * 2u * width;
  active_rows[row_map[row]] = row;
  for (var bin = 0u; bin < width; bin++) { node_counts[first + bin] = bij_hist(row, bin); }
  var tail = 0u;
  var bin = width - 1u;
  while (bin != 0u) {
    bin--;
    tail = bij_add(bij_hist(row, bin), tail);
    node_counts[first + width + bin] = tail;
  }
  node_counts[first + 2u * width - 1u] = bij_add(tail, bij_hist(row, width - 1u));
}
""")

// The count and write passes execute identical expansion logic. CDF limits truncate only the
// represented prefix; splitting ranks always uses the right child's own complete capped count.
//language=wgsl
internal val histogram_compile_nodes by Shader("""$HISTOGRAM_FOREST_HELPERS
@group(0) @binding(0) var<storage, read> bijection: HistogramForest;
@group(0) @binding(1) var<storage, read> row_map: array<u32>;
@group(0) @binding(2) var<storage, read> active_rows: array<u32>;
@group(0) @binding(3) var<storage, read> node_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read> offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<u32>;
struct CompileParams { mode: u32, nodes: u32, pad0: u32, pad1: u32 }
@group(0) @binding(7) var<uniform> params: CompileParams;
struct Writer { node: u32, choices: u32, end: u32 }

fn node_for(row: u32, slot: u32) -> u32 { return 1u + row_map[row] * 2u * bijection.bins + slot; }

fn emit(writer: ptr<function, Writer>, mass: u32, left: u32, right: u32, cost: u32, end: u32) {
  let added = min(mass, end - (*writer).end);
  if (added == 0u) { return; }
  (*writer).end += added;
  if (params.mode != 0u) {
    let out = 5u + 2u * params.nodes + 4u * (offsets[(*writer).node] + (*writer).choices);
    output[out] = (*writer).end;
    output[out + 1u] = left;
    output[out + 2u] = right;
    output[out + 3u] = cost;
  }
  (*writer).choices++;
}

fn product(writer: ptr<function, Writer>, left: u32, right: u32, cost: u32, end: u32) {
  emit(writer, bij_mul(node_counts[left], node_counts[right]), left, right, cost, end);
}

@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let node = gid.x + gid.y * groups.x * 64u;
  if (node >= params.nodes) { return; }
  if (node == 0u || node_counts[node] == 0u) {
    if (params.mode == 0u) { counts[node] = 0u; }
    return;
  }
  let width = bijection.bins;
  let infinity = width - 1u;
  let overflow = width - 2u;
  let row = active_rows[(node - 1u) / (2u * width)];
  let slot = (node - 1u) % (2u * width);
  var writer = Writer(node, 0u, 0u);
  let end = node_counts[node];
  if (slot >= width) {
    if (slot == 2u * width - 1u) {
      product(&writer, node_for(row, width), 0u, 0u, end);
      product(&writer, node_for(row, infinity), 0u, 0u, end);
    } else {
      let bin = slot - width;
      product(&writer, node_for(row, bin), 0u, 0u, end);
      if (bin < overflow) { product(&writer, node_for(row, slot + 1u), 0u, 0u, end); }
    }
  } else {
    let bin = slot;
    for (var c = bij_first(row); c < bij_end(row) && writer.end < end; c++) {
      let choice_end = bij_cdf(c, bin);
      if (choice_end <= writer.end) { continue; }
      let choice = bij_choice(c);
      if (choice.x == BIJ_LEAF) {
        emit(&writer, 1u, BIJ_LEAF, choice.y, choice.w, choice_end);
        continue;
      }
      let left = choice.x;
      let right = choice.y;
      let shift = choice.z;
      if (bin < overflow && shift <= bin) {
        let excess = bin - shift;
        for (var a = 0u; a <= excess && writer.end < choice_end; a++) {
          product(&writer, node_for(left, a), node_for(right, excess - a), choice.w, choice_end);
        }
      } else if (bin == overflow && shift != infinity) {
        for (var a = 0u; a <= overflow && writer.end < choice_end; a++) {
          let threshold = overflow - min(overflow, shift + a);
          product(&writer, node_for(left, a), node_for(right, width + threshold), choice.w, choice_end);
        }
      } else if (bin == infinity) {
        if (shift == infinity) {
          product(&writer, node_for(left, 2u * width - 1u), node_for(right, 2u * width - 1u), choice.w, choice_end);
        } else {
          product(&writer, node_for(left, width), node_for(right, infinity), choice.w, choice_end);
          product(&writer, node_for(left, infinity), node_for(right, 2u * width - 1u), choice.w, choice_end);
        }
      }
    }
  }
  if (params.mode == 0u) { counts[node] = writer.choices; }
}
""")

// Root entries retain the established global bucket/root ordering, but reference ordinary nodes.
//language=wgsl
internal val histogram_compile_roots by Shader("""$HISTOGRAM_FOREST_HELPERS
@group(0) @binding(0) var<storage, read> bijection: HistogramForest;
@group(0) @binding(1) var<storage, read> row_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64) fn main(
  @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
  let entry = gid.x + gid.y * groups.x * 64u;
  if (entry >= bijection.roots * bijection.bins) { return; }
  let bin = entry / bijection.roots;
  let root = bij_root(entry % bijection.roots);
  let width = bijection.bins;
  let overflow = width - 2u;
  let infinity = width - 1u;
  var node = 0u;
  if (bij_first(root.x) != bij_end(root.x)) {
    let first = 1u + row_map[root.x] * 2u * width;
    if (bin < overflow && root.y <= bin) { node = first + bin - root.y; }
    else if (bin == overflow && root.y != infinity) { node = first + width + overflow - root.y; }
    else if (bin == infinity) {
      node = first + infinity;
      if (root.y == infinity) { node = first + 2u * width - 1u; }
    }
  }
  let out = 5u + 2u * output[0] + 4u * output[1] + 3u * entry;
  output[out] = bij_root_cdf(entry);
  output[out + 1u] = node;
  output[out + 2u] = root.z;
}
""")