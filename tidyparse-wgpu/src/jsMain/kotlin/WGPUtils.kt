package ai.hypergraph.tidyparse.wgpu

import ai.hypergraph.tidyparse.wgpu.GPUBufferUsage.STCPSD
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.GPUBuffer
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.buildLanguageSizeCDF
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.packMetadata
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.readIndices
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.toGPUBuffer
import ai.hypergraph.tidyparse.wgpu.Shader.Companion.writeU32
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.cache.LRUCache
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.parsing.bindex
import ai.hypergraph.kaliningraph.parsing.leftAdj
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.types.cache
import web.gpu.GPUBuffer
import kotlin.time.TimeMark
import kotlin.time.TimeSource

data class GrammarEncoding(val flat: IntArray, val offsets: IntArray)
// leftAdjGrouped[B] = triples:
//   (C, parentWord, parentMask)
// where parentMask contains all A in parentWord such that A -> B C.
data class GroupedLeftAdjEncoding(val flat: IntArray, val offsets: IntArray)

val CFG.groupedLeftAdjEncoding: GroupedLeftAdjEncoding by cache {
  val ladj = leftAdj
  val W = ladj.size

  // For each B, group scalar leftAdj entries by (C, parentWord).
  // key = (C << 32) | parentWord
  // value = mask of all parent A bits in that parentWord.
  val rows = Array(W) { linkedMapOf<Long, Int>() }

  var B = 0
  while (B < W) {
    val adj = ladj[B]

    if (adj != null) {
      val cs = adj.other
      val asz = adj.aIdx

      var i = 0
      while (i < cs.size) {
        val C = cs[i]
        val A = asz[i]

        val parentWord = A ushr 5
        val parentMask = 1 shl (A and 31)

        val key = (C.toLong() shl 32) or (parentWord.toLong() and 0xffffffffL)

        rows[B][key] = (rows[B][key] ?: 0) or parentMask

        i++
      }
    }

    B++
  }

  val offsets = IntArray(W + 1)
  var total = 0

  B = 0
  while (B < W) {
    offsets[B] = total
    total += rows[B].size * 3
    B++
  }

  offsets[W] = total

  val flat = IntArray(total)
  var out = 0

  B = 0
  while (B < W) {
    for ((key, mask) in rows[B]) {
      val C = (key ushr 32).toInt()
      val parentWord = (key and 0xffffffffL).toInt()

      flat[out++] = C
      flat[out++] = parentWord
      flat[out++] = mask
    }

    B++
  }

  GroupedLeftAdjEncoding(flat, offsets)
}

val CFG.grammarEncoding: GrammarEncoding by cache {
  val W = nonterminals.size
  val ntIdx = bindex.ntIndices   // Map<Σᐩ, Int>

  val counts = IntArray(W)
  for ((lhs, rhs) in this) {
    if (rhs.size != 2) continue
    val a = ntIdx[lhs] ?: continue
    val b = ntIdx[rhs[0]] ?: continue
    val c = ntIdx[rhs[1]] ?: continue
    counts[a] += 2
  }

  val offsets = IntArray(W + 1)
  var acc = 0
  for (i in 0 until W) { offsets[i] = acc; acc += counts[i] }
  offsets[W] = acc

  val flat = IntArray(acc)
  val cur = offsets.copyOf()
  for ((lhs, rhs) in this) {
    if (rhs.size != 2) continue
    val a = ntIdx[lhs] ?: continue
    val b = ntIdx[rhs[0]] ?: continue
    val c = ntIdx[rhs[1]] ?: continue
    val p = cur[a]
    flat[p] = b
    flat[p + 1] = c
    cur[a] = p + 2
  }

  GrammarEncoding(flat, offsets)
}

suspend fun logActiveNTGrid(
  activeBuf: GPUBuffer,
  numStates: Int,
  numNTs: Int,
  limit: Int = minOf(32, numStates)
) {
  val activeWords = (numNTs + 31) ushr 5
  val countsBuf = Shader.GPUBuffer(numStates.toLong() * numStates * 4L, GPUBufferUsage.STCPSD)
  val uniBuf = intArrayOf(numStates, activeWords).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  val groupsX = (numStates + 7) / 8
  val groupsY = (numStates + 7) / 8
  active_nt_count(activeBuf, countsBuf, uniBuf)(groupsX, groupsY, 1)

  val allIndices = (0 until numStates * numStates).toList()
  val allVals = countsBuf.readIndices(allIndices)

  var totalActiveNTs = 0L
  for (i in 0..<allVals.length) totalActiveNTs += allVals[i]
  val totalUTCells = (numStates.toLong() * (numStates - 1)) / 2
  val maxPossibleActive = totalUTCells * numNTs

  val sparsity = if (maxPossibleActive > 0) (totalActiveNTs.toDouble() / maxPossibleActive) * 100.0 else 0.0

  val previewIdxs = ArrayList<Int>(limit * limit)
  for (r in 0 until limit) for (c in 0 until limit) previewIdxs += r * numStates + c
  val previewVals = countsBuf.readIndices(previewIdxs)

  val w = numNTs.toString().length.coerceAtLeast(2)
  val sb = StringBuilder()

  sb.append("--- UT Sparsity: ${sparsity.toString().take(8)}% ")
  sb.append("($totalActiveNTs / $maxPossibleActive active NTs) ---\n")

  sb.append("Active NTs per cell (k/$numNTs), showing ${limit}x$limit (upper triangle):\n")
  for (r in 0 until limit) {
    for (c in 0 until limit) {
      val k = previewVals[r * limit + c]
      if (c <= r) sb.append(" ".repeat(w)).append("  ")
      else sb.append(k.toString().padStart(w, ' ')).append("  ")
    }
    sb.append('\n')
  }
  log(sb.toString())

  uniBuf.destroy()
  countsBuf.destroy()
}

//language=wgsl
val active_nt_count by Shader("""
struct Uni { n: u32, activeWords: u32 };

@group(0) @binding(0) var<storage, read>       active_nts : array<u32>;
@group(0) @binding(1) var<storage, read_write> outCnt     : array<u32>; // length n*n
@group(0) @binding(2) var<uniform>             uni        : Uni;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  let c = gid.y;
  let n = uni.n;
  let aw = uni.activeWords;

  if (r >= n || c >= n) { return; }
  if (c <= r) { outCnt[r*n + c] = 0u; return; }

  let base = (r * n + c) * aw;
  var k: u32 = 0u;
  for (var w: u32 = 0u; w < aw; w = w + 1u) { k += countOneBits(active_nts[base + w]); }
  outCnt[r*n + c] = k;
}""")

// Orders conditioned suffix packets without relying on workgroup scheduling.
// Every next-token group receives the same number of slots, and its packets
// are ordered by suffix length and then by the lexical terminal sequence.
val suffix_group_select by Shader("""
struct Params {
  maxSamples : u32,
  stride     : u32,
  prefixSize : u32,
  groupCount : u32,
  capacity   : u32,
  sortCapacity : u32,
  padding0   : u32,
  padding1   : u32
};

@group(0) @binding(0) var<uniform>             prm         : Params;
@group(0) @binding(1) var<storage, read>       packets     : array<u32>;
@group(0) @binding(2) var<storage, read>       lexicalRank : array<u32>;
@group(0) @binding(3) var<storage, read_write> selected    : array<u32>;

const HEADER_LEN : u32 = ${PKT_HDR_LEN}u;
const TOKEN_MASK : u32 = ${PACKED_TOKEN_MASK}u;
const SENTINEL   : u32 = 0xffffffffu;

fn packetLength(sample: u32) -> u32 {
  let base = sample * prm.stride;
  var length: u32 = 0u;
  loop {
    if (HEADER_LEN + length >= prm.stride) { break; }
    if (packets[base + HEADER_LEN + length] == 0u) { break; }
    length = length + 1u;
  }
  return length;
}

fn tokenLexicalRank(sample: u32, position: u32) -> u32 {
  let packed = packets[sample * prm.stride + HEADER_LEN + position];
  let token = packed & TOKEN_MASK;
  if (token == 0u) { return SENTINEL; }
  return lexicalRank[token - 1u];
}

fn comesBefore(left: u32, right: u32) -> bool {
  if (right == SENTINEL) { return left != SENTINEL; }
  if (left == SENTINEL) { return false; }

  let leftLength = packetLength(left);
  let rightLength = packetLength(right);
  let leftValid = leftLength > prm.prefixSize;
  let rightValid = rightLength > prm.prefixSize;
  if (leftValid != rightValid) { return leftValid; }
  if (!leftValid) { return left < right; }

  let leftSuffixLength = leftLength - prm.prefixSize;
  let rightSuffixLength = rightLength - prm.prefixSize;
  if (leftSuffixLength != rightSuffixLength) { return leftSuffixLength < rightSuffixLength; }

  var position = prm.prefixSize;
  while (position < leftLength) {
    let leftToken = tokenLexicalRank(left, position);
    let rightToken = tokenLexicalRank(right, position);
    if (leftToken != rightToken) { return leftToken < rightToken; }
    position = position + 1u;
  }
  return left < right;
}

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let group = gid.x;
  if (group >= prm.groupCount || prm.groupCount == 0u) { return; }

  let perGroup = prm.maxSamples / prm.groupCount;
  let available = min(prm.capacity, perGroup);
  for (var slot = 0u; slot < prm.sortCapacity; slot = slot + 1u) {
    selected[slot * prm.groupCount + group] = SENTINEL;
  }

  // suffix_enum_words_wor assigns sid % groupCount to this group and emits a
  // dense localSid prefix. Its first rootCount local IDs contain rank zero for
  // every planned nonempty suffix length, so deterministic prefix selection
  // retains those representatives before considering later samples.
  for (var slot = 0u; slot < available; slot = slot + 1u) {
    selected[slot * prm.groupCount + group] = slot * prm.groupCount + group;
  }

  // A group owns one strided selected[] column. A single invocation runs a
  // bitonic network over its small power-of-two column without atomics, so
  // scheduling cannot perturb ordering.
  var width = 2u;
  while (width <= prm.sortCapacity) {
    var stride = width >> 1u;
    while (stride > 0u) {
      for (var leftSlot = 0u; leftSlot < prm.sortCapacity; leftSlot = leftSlot + 1u) {
        let rightSlot = leftSlot ^ stride;
        if (rightSlot > leftSlot) {
          let leftOffset = leftSlot * prm.groupCount + group;
          let rightOffset = rightSlot * prm.groupCount + group;
          let left = selected[leftOffset];
          let right = selected[rightOffset];
          let ascending = (leftSlot & width) == 0u;
          var swap = comesBefore(left, right);
          if (ascending) { swap = comesBefore(right, left); }
          if (swap) {
            selected[leftOffset] = right;
            selected[rightOffset] = left;
          }
        }
      }
      stride = stride >> 1u;
    }
    width = width << 1u;
  }
}""")

private suspend fun selectSuffixPackets(
  packets: GPUBuffer,
  cfg: CFG,
  stride: Int,
  sampleCount: Int,
  batch: SuffixBatch,
  rootCounts: IntArray,
  limit: Int
): IntersectionResults {
  val terminals = batch.slices.map { it.terminal }.distinct()
  if (terminals.isEmpty() || sampleCount == 0) return IntersectionResults.EMPTY
  require(rootCounts.size == terminals.size)
  val lexicalRanks = IntArray(cfg.tmLst.size)
  cfg.tmLst.indices.sortedBy(cfg.tmLst::get).forEachIndexed { rank, terminal ->
    lexicalRanks[terminal] = rank
  }
  val perGroup = sampleCount / terminals.size
  val capacity = minOf(perGroup, (limit * 8).coerceAtLeast(limit))
  require(rootCounts.all { it <= minOf(capacity, perGroup) }) {
    "Conditioned suffix budget must retain every planned length root"
  }
  var sortCapacity = 1
  while (sortCapacity < capacity) sortCapacity = sortCapacity shl 1
  val selectedCount = terminals.size * capacity
  val params = intArrayOf(
    sampleCount, stride, batch.prefix.size, terminals.size,
    capacity, sortCapacity, 0, 0
  )
    .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
  val lexicalRankBuf = lexicalRanks.toGPUBuffer(STCPSD)
  val selectedBuf = IntArray(terminals.size * sortCapacity) { -1 }.toGPUBuffer(STCPSD)

  suffix_group_select(params, packets, lexicalRankBuf, selectedBuf)(terminals.size)

  val gatherParams = intArrayOf(sampleCount, selectedCount, stride, DISPATCH_GROUP_SIZE_X)
    .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
  val compact = GPUBuffer(selectedCount * stride * 4, STCPSD)
  gather_top_k(gatherParams, packets, selectedBuf, compact)(selectedCount)
  val decoded = decodePackets(compact.readJSIntArray(), cfg, stride, selectedCount)

  listOf(params, lexicalRankBuf, selectedBuf, gatherParams, compact).forEach(GPUBuffer::destroy)
  return decoded
}

suspend fun completeCode(cfg: CFG, porous: List<String>, ngrams: GPUBuffer? = null): IntersectionResults {
  timings = linkedMapOf()
  val preprocT = TimeSource.Monotonic.markNow()

  val fsa: FSA = makePorousFSA(porous)
  val codePoints = porousToCodePoints(cfg, porous)

  log("Made porousFSA(|Q|=${fsa.numStates}, width=${fsa.width}) in ${preprocT.elapsedNow()}")
  mark("preprocessing", preprocT)

  return intersectionPipeline(
    cfg = cfg,
    fsa = fsa,
    ledBuffer = Int.MAX_VALUE,
    codePoints = codePoints,
    chartInitializer = init_line_chart
  ).also { log("Received: ${it.size} completions in ${preprocT.elapsedNow()} (round trip)") }
}

private data class GPUSuffixKey(val grammar: CFG, val batch: SuffixBatch, val limit: Int)

private val gpuSuffixCache = LRUCache<GPUSuffixKey, List<String>>(128)

private data class SuffixSamplingGroup(val terminal: Int, val lengths: Set<Int>)

private suspend fun CFG.suffixIntersectionPipeline(
  fsa: FSA,
  codePoints: IntArray,
  batch: SuffixBatch,
  limit: Int,
  timingTrace: MutableMap<String, Int>
): IntersectionResults {
  require(tmLst.size < PACKED_TOKEN_LIMIT)
  val (numStates, numNTs) = fsa.numStates to nonterminals.size
  log("FSA(|Q|=$numStates, |δ|=${fsa.transit.size}), ${calcStats()}")

  val metadataStarted = TimeSource.Monotonic.markNow()
  val metaBuf = packLineMetadata(this, numStates)
  timingTrace.mark("pack metadata", metadataStarted)
  val tmBuf = termBuf
  val wordBuf = codePoints.toGPUBuffer()
  val activeWords = (numNTs + 31) ushr 5
  val dpBuf = Shader.createParseChart(STCPSD, numStates * numStates * numNTs)
  val activeBuf = GPUBuffer((numStates * numStates * activeWords * 4).toLong(), STCPSD)

  timingTrace["init chart"] = timedGPUIsolated("Init suffix chart") {
    init_line_chart(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(
      numStates, numStates, (numNTs + DENSE_NT_WORKGROUP_SIZE - 1) / DENSE_NT_WORKGROUP_SIZE
    )
  }
  val closureStarted = TimeSource.Monotonic.markNow()
  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  timingTrace.mark("matrix closure", closureStarted)

  val rootsStarted = TimeSource.Monotonic.markNow()
  val startNT = bindex[START_SYMBOL]
  val rootQuery = fsa.finalIdxs.map { it * numNTs + startNT }
  val rootReachability = dpBuf.readIndices(rootQuery)
  val roots = rootQuery.filterIndexed { i, _ -> rootReachability[i] != 0 }
  timingTrace.mark("read roots", rootsStarted)
  if (roots.isEmpty()) {
    destroyAll(activeBuf, wordBuf, metaBuf, dpBuf)
    return IntersectionResults.EMPTY
  }

  val backpointersStarted = TimeSource.Monotonic.markNow()
  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) =
    Shader.buildBackpointers(numStates, numNTs, dpBuf, metaBuf)
  timingTrace.mark("build backpointers", backpointersStarted)

  val rootsFiltered = TimeSource.Monotonic.markNow()
  val startIdxs = roots.flatMap { listOf(it, 0) }
  timingTrace.mark("filter roots", rootsFiltered)
  val maxRepairLen = minOf(MAX_WORD_LEN, fsa.width + PKT_HDR_LEN + 1)
  if (fsa.width >= MAX_WORD_LEN - 1) {
    destroyAll(activeBuf, wordBuf, metaBuf, dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf)
    return IntersectionResults.EMPTY
  }

  val cdfBuf = GPUBuffer(bpStorageBuf.size / 2, STCPSD)
  if (PROFILE_WGPU_KERNELS) awaitGPUQueue()
  val sizesStarted = TimeSource.Monotonic.markNow()
  buildLanguageSizeCDF(
    numStates, numNTs, dpBuf, metaBuf, tmBuf,
    bpCountBuf, bpOffsetBuf, bpStorageBuf, cdfBuf
  )
  if (PROFILE_WGPU_KERNELS) awaitGPUQueue()
  timingTrace.mark("build language-size CDF", sizesStarted)

  val groups = batch.slices.groupBy { it.terminal }.map { (terminal, slices) ->
    SuffixSamplingGroup(tmMap.getValue(terminal), slices.mapTo(linkedSetOf()) { it.length })
  }
  val rootLengths = roots.map { dpRoot ->
    val cell = (dpRoot - startNT) / numNTs
    fsa.idsToCoords.getValue(cell).first - batch.prefix.size
  }
  val pairRoots = mutableListOf<Int>()
  val groupOffsets = mutableListOf(0)
  groups.forEach { group ->
    rootLengths.indices
      .filter { root -> rootLengths[root] in group.lengths }
      .sortedBy(rootLengths::get)
      .forEach(pairRoots::add)
    groupOffsets += pairRoots.size
  }
  require(groups.isNotEmpty() && groupOffsets.zipWithNext().all { (a, b) -> a < b })

  val idxUniBuf = packStruct(
    listOf(0, maxRepairLen, numNTs, numStates, DISPATCH_GROUP_SIZE_X, MAX_SAMPLES, batch.prefix.size, groups.size),
    startIdxs.toGPUBuffer(),
    groups.map { it.terminal }.toGPUBuffer(),
    groupOffsets.toGPUBuffer(),
    pairRoots.toGPUBuffer()
  )
  val cutCells = (batch.prefix.size + 1L) * (numStates - batch.prefix.size - 1L)
  val suffixSizes = GPUBuffer(cutCells * numNTs * groups.size * 4L, STCPSD)
  val conditionedStarted = TimeSource.Monotonic.markNow()
  for (span in 1..<numStates) {
    val spanBuf = span.toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
    suffix_ls_dense(
      dpBuf, cdfBuf, suffixSizes, metaBuf, tmBuf, spanBuf,
      bpCountBuf, bpOffsetBuf, bpStorageBuf, idxUniBuf
    )(numStates - span, 1, numNTs * groups.size)
    spanBuf.destroy()
  }
  timingTrace.mark("build suffix sizes", conditionedStarted)

  val perGroup = minOf(40_000 / groups.size, (limit * 8).coerceAtLeast(limit))
  val sampleCount = perGroup * groups.size
  idxUniBuf.writeU32(5, sampleCount)
  val packets = GPUBuffer(sampleCount * maxRepairLen * 4, STCPSD)
  timingTrace["enumerate"] = timedGPUIsolated("Enumerate conditioned suffixes") {
    suffix_enum_words_wor(
      dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf, cdfBuf,
      tmBuf, idxUniBuf, suffixSizes, packets
    ).dispatchFlat(sampleCount)
  }
  log("Conditioned suffix sampling: ${groups.size} groups, lanes=$sampleCount, maxRepairLen=$maxRepairLen")

  val decodeStarted = TimeSource.Monotonic.markNow()
  val rootCounts = IntArray(groups.size) { group -> groupOffsets[group + 1] - groupOffsets[group] }
  val result = selectSuffixPackets(packets, this, maxRepairLen, sampleCount, batch, rootCounts, limit)
  timingTrace.mark("decode", decodeStarted)
  destroyAll(
    packets, suffixSizes, idxUniBuf, cdfBuf,
    bpCountBuf, bpOffsetBuf, bpStorageBuf, activeBuf, wordBuf, metaBuf, dpBuf
  )
  return result
}

/** Builds and decodes one porous GPU forest for all selected suffix slices. */
suspend fun CFG.gpuDiverseSuffixes(
  batch: SuffixBatch,
  limit: Int,
  requestStarted: TimeMark = TimeSource.Monotonic.markNow(),
  recordTimings: (Map<String, Int>) -> Unit = {}
): List<String>? {
  if (limit <= 0) return emptyList()
  val cacheStarted = TimeSource.Monotonic.markNow()
  val cacheKey = GPUSuffixKey(this, batch, limit)
  gpuSuffixCache[cacheKey]?.let {
    log("Suffix CPU preprocessing (GPU handoff): result-cache hit in ${cacheStarted.elapsedNow()}")
    return it
  }
  val cacheElapsed = cacheStarted.elapsedNow()
  if (batch.slices.isEmpty()) {
    log("Suffix CPU preprocessing (GPU handoff): no forest needed; cache lookup $cacheElapsed")
    return batch.completeWords.take(limit)
  }

  val templateStarted = TimeSource.Monotonic.markNow()
  val horizon = batch.slices.maxOf { it.length }
  val porous = batch.prefix + List(horizon) { HOLE_MARKER }
  if (porous.size >= MAX_WORD_LEN - 1) {
    log("Suffix CPU preprocessing (GPU handoff): template length ${porous.size} exceeds GPU limit")
    return null
  }

  val acceptedLengths = batch.slices.mapTo(linkedSetOf()) { batch.prefix.size + it.length }
  val fsa = makePorousFSA(porous, acceptedLengths)
  val codePoints = porousToCodePoints(this, porous)
  log(
    "Suffix CPU preprocessing (GPU handoff): cache lookup $cacheElapsed; " +
      "porous FSA + token encoding ${templateStarted.elapsedNow()} " +
      "(${fsa.numStates} states, ${acceptedLengths.size} roots)"
  )
  val suffixTimings = linkedMapOf<String, Int>()
  suffixTimings.mark("preprocessing", requestStarted)
  val results = suffixIntersectionPipeline(fsa, codePoints, batch, limit, suffixTimings)

  val filterStarted = TimeSource.Monotonic.markNow()
  val allowedLengths = batch.slices.groupBy({ it.terminal }, { it.length })
    .mapValues { (_, lengths) -> lengths.toSet() }
  val groups = linkedMapOf<String, LinkedHashSet<String>>()
  batch.slices.forEach { groups.getOrPut(it.terminal) { linkedSetOf() } }
  results.indices.forEach { row ->
    if (results.terminalCountAt(row) <= batch.prefix.size) return@forEach
    val terminal = results.terminalTextAt(row, batch.prefix.size)
    val suffixLength = results.terminalCountAt(row) - batch.prefix.size
    if (suffixLength in allowedLengths[terminal].orEmpty()) groups[terminal]?.let {
      val word = results[row]
      it += word
    }
  }
  // Every planned group has an exact conditioned root. Keep this guard as a
  // sampler invariant rather than treating sampling coverage probabilistically.
  val missingGroups = groups.filterValues { it.isEmpty() }.keys
  check(missingGroups.isEmpty()) {
    "Conditioned suffix sampler omitted groups: ${missingGroups.joinToString()}"
  }

  val decodedGroups = groups.values.map { it.toList() }
  val fairDecoded = sequence {
    var rank = 0
    while (decodedGroups.any { rank < it.size }) {
      decodedGroups.forEach { words -> words.getOrNull(rank)?.let { yield(it) } }
      rank++
    }
  }
  val completed = sequence {
    yieldAll(batch.completeWords)
    yieldAll(fairDecoded)
  }.distinct().take(limit).toList()
  val guaranteed = minOf(limit, batch.completeWords.size + batch.slices.size)
  check(completed.size >= guaranteed) {
    "Conditioned suffix sampler decoded ${completed.size} of at least $guaranteed guaranteed rows"
  }
  suffixTimings.mark("filter suffixes", filterStarted)
  recordTimings(suffixTimings.toMap())
  gpuSuffixCache.put(cacheKey, completed)
  return completed
}

// Checks whether there is a forward completion in the language of the CFG
suspend fun CFG.checkSuffix(tokens: List<String>, suffixLen: Int = 20): List<Int> {
  val t0 = TimeSource.Monotonic.markNow()

  val porousTks = tokens + List(suffixLen) { "_" }
  val fsa: FSA = makePorousFSA(porousTks)
  val codePoints = porousToCodePoints(this, porousTks)

  log("Made porousFSA(|Q|=${fsa.numStates}, width=${fsa.width}) in ${t0.elapsedNow()}")

  return checkSuffixPipeline(this, fsa, suffixLen, codePoints).also { log("Checked suffix completions in ${t0.elapsedNow()} (round trip)") }
}

suspend fun checkSuffixPipeline(cfg: CFG, fsa: FSA, suffixLen: Int, codePoints: IntArray): List<Int> {
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("Porous FSA(|Q|=$numStates), ${cfg.calcStats()}")

  val metaBuf = packMetadata(cfg, fsa)

  val tmBuf       = cfg.termBuf
  val wordBuf     = codePoints.toGPUBuffer()
  val totalSize   = numStates * numStates * numNTs
  val activeWords = (numNTs + 31) ushr 5

  val dpBuf     = Shader.createParseChart(STCPSD, totalSize)
  val activeBuf = GPUBuffer((numStates * numStates * activeWords * 4).toLong(), STCPSD)

  timings["init chart"] = timedGPUIsolated("Init chart") {
    val ntWorkgroups = (numNTs + DENSE_NT_WORKGROUP_SIZE - 1) / DENSE_NT_WORKGROUP_SIZE
    init_line_chart(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, ntWorkgroups)
  }

  val closureT = TimeSource.Monotonic.markNow()
  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  mark("matrix closure", closureT)
  log("Matrix closure reached in: ${timings["matrix closure"]}ms")

  val startNT = cfg.bindex[START_SYMBOL]

  val baseLen = codePoints.size - suffixLen
  val queryIndices = ArrayList<Int>(suffixLen + 1)
  for (k in 0..suffixLen) {
    val targetState = baseLen + k
    queryIndices += targetState * numNTs + startNT
  }

  val reachability = dpBuf.readIndices(queryIndices)
  val listSuffixes = (0..<reachability.length).filter { reachability[it] != 0 }

  listOf(activeBuf, metaBuf, dpBuf, wordBuf).forEach(GPUBuffer::destroy)
  return listSuffixes
}

fun makePorousFSA(tokens: List<String>, acceptedLengths: Set<Int> = setOf(tokens.size)): FSA {
  val n = tokens.size
  require(acceptedLengths.isNotEmpty() && acceptedLengths.all { it in 1..n })
  val digits = (n + 1).toString().length

  fun pd(i: Int) = i.toString().padStart(digits, '0')
  fun st(i: Int) = "q_${pd(i)}/${pd(0)}"

  val arcs: TSA = (0 until n).map { i ->
    val lbl = tokens[i]
    Triple(st(i), lbl, st(i + 1))
  }.toSet()

  val initialStates = setOf(st(0))
  val finalStates   = acceptedLengths.mapTo(linkedSetOf(), ::st)

  return AFSA(arcs, initialStates, finalStates)
    .also { it.width = n; it.height = 0; it.levString = tokens }
}

private const val HOLE_SENTINEL_INT: Int = -1 // 0xFFFF_FFFFu on GPU

private fun escapeUnknownTokenHtml(token: String): String =
  token
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace("\"", "&quot;")
    .replace("'", "&#39;")

fun unknownTokenHtml(token: String): String = "❌ Unknown token: ${escapeUnknownTokenHtml(token)}"

fun porousToCodePoints(cfg: CFG, porous: List<String>): IntArray =
  IntArray(porous.size) { i ->
    val t = porous[i]
    if (t == "_") HOLE_SENTINEL_INT
    else cfg.tmMap[t] ?: error("Unknown token '$t' (not in cfg.tmMap)")
  }

var timings = linkedMapOf<String, Int>()
fun MutableMap<String, Int>.mark(step: String, started: TimeMark) {
  this[step] = started.elapsedNow().inWholeMilliseconds.toInt()
}
fun mark(step: String, started: TimeMark) = timings.mark(step, started)

fun Map<String, Int>.logTimesheet() {
  val totalMs = this["total"]?.coerceAtLeast(0)
  val bodyRows = entries
    .asSequence()
    .filter { it.key != "total" }
    .map { (k, v) -> k.ifBlank { "<unnamed>" } to v.coerceAtLeast(0) }
    .sortedByDescending { it.second }
    .toList()

  val total = totalMs ?: bodyRows.sumOf { it.second }.coerceAtLeast(1)
  val maxMs = bodyRows.maxOf { it.second }.coerceAtLeast(1)

  val maxLabel = 28
  val barWidth = 36

  fun String.compactLabel(): String = replace(Regex("\\s+"), " ")
      .let { if (it.length <= maxLabel) it else it.take(maxLabel - 1) + "…" }

  fun Int.bar(): String = "#".repeat(((this.toDouble() / maxMs) * barWidth).toInt().coerceIn(1, barWidth))

  val rowTexts = bodyRows.map { (label, ms) ->
    val pct = (100.0 * ms / total.coerceAtLeast(1)).toInt()
    buildString {
      append(label.compactLabel().padEnd(maxLabel))
      append(" |")
      append(ms.bar().padEnd(barWidth))
      append("| ")
      append(ms.toString().padStart(6))
      append(" ms  ")
      append(pct.toString().padStart(3))
      append('%')
    }
  }

  val titleText = "─ Timings (${bodyRows.size} steps, total=${total}ms) "
  val contentWidth = maxOf(titleText.length, rowTexts.maxOf { it.length })

  val plot = buildString {
    appendLine("┌" + titleText.padEnd(contentWidth, '─') + "┐")
    rowTexts.forEach { row -> appendLine("│" + row.padEnd(contentWidth, ' ') + "│") }
    appendLine("└" + "─".repeat(contentWidth) + "┘")
  }

  println(plot)
}

//language=wgsl
val wdfa_score_raw by Shader("""$SAMPLER_PARAMS
$WDFA_STRUCT

@group(0) @binding(0) var<storage, read_write> packets : array<u32>;
@group(0) @binding(1) var<storage, read>       wdfa    : WDFA;
@group(0) @binding(2) var<uniform>             prm     : Params;

const PKT_HDR_LEN : u32 = ${PKT_HDR_LEN}u;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sid = gid.x + gid.y * prm.threads;
  if (sid >= prm.maxSamples) { return; }

  let stride = prm.stride;
  let base = sid * stride;

  var q = wdfa.startState;
  var cost = wdfa.startCost;

  var pos : u32 = 0u;
  loop {
    if (pos >= stride - PKT_HDR_LEN) { break; }

    let tok = packets[base + PKT_HDR_LEN + pos];
    if (tok == 0u) { break; }

    let e = find_edge(q, tok);
    if (e == NO_EDGE) {
      cost = sat_add_wdfa(cost, wdfa.missingCost);
    } else {
      cost = sat_add_wdfa(cost, edge_cost(e));
      q = edge_dst(e);
    }

    pos = pos + 1u;
  }

  let fc = final_cost(q);
  if (fc >= WDFA_INF) {
    cost = WDFA_INF;
  } else {
    cost = sat_add_wdfa(cost, fc);
  }

  // Apples-to-apples debug score: no edit-distance penalty.
  packets[base + 1u] = cost;
}""")

val line_mdpt_write by Shader("""struct Uni { n : u32 };
@group(0) @binding(0) var<storage, read_write> offsets : array<u32>;
@group(0) @binding(1) var<storage, read_write>  flat_mp : array<u32>;
@group(0) @binding(2) var<uniform>                  uni : Uni;

fn choose3(x: u32) -> u32 {
  if (x < 3u) { return 0u; }
  return (x * (x - 1u) * (x - 2u)) / 6u;
}

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let r = gid.y; let c = gid.x; let N = uni.n;
  if (r >= N || c >= N) { return; }

  let rowBase = choose3(N) - choose3(N - r);
  var withinRow = 0u;
  if (c > r + 1u) {
    let d = c - r;
    withinRow = ((d - 2u) * (d - 1u)) / 2u;
  }
  let offset = rowBase + withinRow;
  offsets[r * N + c] = offset;

  if (c <= r + 1u) { return; }
  for (var m = r + 1u; m < c; m = m + 1u) {
    flat_mp[offset + m - r - 1u] = m;
  }
}""")


private fun packLineMetadata(cfg: CFG, states: Int): GPUBuffer {
  require(states > 0)
  val started = TimeSource.Monotonic.markNow()
  val grammar = cfg.grammarEncoding
  val leftAdj = cfg.groupedLeftAdjEncoding
  val activeWords = (cfg.nonterminals.size + 31) ushr 5
  val midpointCount = (
    states.toLong() * (states - 1).coerceAtLeast(0) * (states - 2).coerceAtLeast(0) / 6
  ).also { require(it <= Int.MAX_VALUE) }.toInt()

  val midpoints = GPUBuffer(midpointCount * 4L, STCPSD)
  val midpointOffsets = GPUBuffer(states.toLong() * states * 4, STCPSD)
  val statesBuf = states.toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
  line_mdpt_write(midpointOffsets, midpoints, statesBuf)(states, states)
  statesBuf.destroy()

  return packStruct(
    constants = listOf(states, cfg.nonterminals.size, activeWords),
    midpoints,
    midpointOffsets,
    intArrayOf().toGPUBuffer(STCPSD),
    grammar.flat.toGPUBuffer(STCPSD),
    grammar.offsets.toGPUBuffer(STCPSD),
    leftAdj.flat.toGPUBuffer(STCPSD),
    leftAdj.offsets.toGPUBuffer(STCPSD)
  ).also {
    log("Packed analytic line metadata in ${started.elapsedNow()} ($midpointCount midpoints)")
  }
}


// Suffix sampling owns a dedicated uniform so generic repair kernels remain unaware of it.
//language=wgsl
private const val SUFFIX_IDX_UNIFORM_STRUCT = """
struct SuffixIndexUniforms {
    targetCnt       : atomic<u32>,
    maxWordLen      : u32,
    numNonterminals : u32,
    numStates       : u32,
    threads         : u32,
    max_samples     : u32,
    suffix_prefix   : u32,
    suffix_groups   : u32,

    startIdxOffset  : u32, numStartIndices : u32,
    suffixTokensOffset : u32, suffixTokensSize : u32,
    suffixGroupOffsetsOffset : u32, suffixGroupOffsetsSize : u32,
    suffixPairRootsOffset : u32, suffixPairRootsSize : u32,
    payload         : array<u32>
};
"""

//language=wgsl
private const val SUFFIX_CHART_DECODING_HELPERS = """
fn getStartIdx(i : u32) -> u32 { return idx_uni.payload[idx_uni.startIdxOffset + i * 2u]; }
fn getEditDist(i : u32) -> u32 { return idx_uni.payload[idx_uni.startIdxOffset + i * 2u + 1u]; }
fn getSuffixToken(i : u32) -> u32 { return idx_uni.payload[idx_uni.suffixTokensOffset + i]; }
fn getSuffixGroupOffset(i : u32) -> u32 { return idx_uni.payload[idx_uni.suffixGroupOffsetsOffset + i]; }
fn getSuffixPairRoot(i : u32) -> u32 { return idx_uni.payload[idx_uni.suffixPairRootsOffset + i]; }
"""

//language=wgsl
private const val SUFFIX_TERM_STRUCT = """
struct Terminals {
    nt_tm_lens_offset : u32, nt_tm_lens_size : u32,
    offsets_offset : u32, offsets_size : u32,
    all_tms_offset : u32, all_tms_size : u32,
    payload : array<u32>
};

$SUFFIX_IDX_UNIFORM_STRUCT

$TM_DECODING_HELPERS
"""

//language=wgsl
val suffix_ls_dense by Shader("""$CFL_STRUCT $SUFFIX_TERM_STRUCT
struct SpanUni { span : u32 };
@group(0) @binding(0) var<storage, read>           dp_in : array<u32>;
@group(0) @binding(1) var<storage, read>        ls_sparse : array<u32>;
@group(0) @binding(2) var<storage, read_write> suffix_sizes : array<u32>;
@group(0) @binding(3) var<storage, read>              cs : CFLStruct;
@group(0) @binding(4) var<storage, read>       terminals : Terminals;
@group(0) @binding(5) var<uniform>                    su : SpanUni;
@group(0) @binding(6) var<storage, read>        bp_count : array<u32>;
@group(0) @binding(7) var<storage, read>       bp_offset : array<u32>;
@group(0) @binding(8) var<storage, read>      bp_storage : array<u32>;
@group(0) @binding(9) var<storage, read_write>    idx_uni : SuffixIndexUniforms;

$WGSL_LANG_SIZE

$SUFFIX_CHART_DECODING_HELPERS

fn spansSuffix(dpIdx: u32) -> bool {
  let cell = dpIdx / cs.numNonterminals;
  let r = cell / cs.numStates;
  let c = cell % cs.numStates;
  return r <= idx_uni.suffix_prefix && idx_uni.suffix_prefix < c;
}

fn suffixIndex(group: u32, dpIdx: u32) -> u32 {
  let nt = dpIdx % cs.numNonterminals;
  let cell = dpIdx / cs.numNonterminals;
  let r = cell / cs.numStates;
  let c = cell % cs.numStates;
  let rightWidth = cs.numStates - idx_uni.suffix_prefix - 1u;
  let cellsPerGroup = (idx_uni.suffix_prefix + 1u) * rightWidth;
  let compactCell = r * rightWidth + c - idx_uni.suffix_prefix - 1u;
  return (group * cellsPerGroup + compactCell) * cs.numNonterminals + nt;
}

fn matchingLiteralCount(val: u32, nt: u32, required: u32) -> u32 {
  let ntLen = get_nt_tm_lens(nt);
  let ntOff = get_offsets(nt);
  var present = false;
  for (var i = 0u; i < ntLen; i = i + 1u) {
    if (get_all_tms(ntOff + i) == required) { present = true; break; }
  }
  if (!present) { return 0u; }

  let predicate = val & PREDICATE_MASK;
  if (predicate == LIT_ALL) { return 1u; }
  let litEnc = (predicate >> 1u) & 0x03ffffffu;
  if (litEnc == 0u || litEnc > ntLen) { return 0u; }
  let excluded = get_all_tms(ntOff + litEnc - 1u);
  if ((predicate & $NEG_STR_LIT) != 0u) { return select(1u, 0u, required == excluded); }
  return select(0u, 1u, required == excluded);
}

fn conditionedSize(group: u32, dpIdx: u32) -> u32 {
  if (spansSuffix(dpIdx)) { return suffix_sizes[suffixIndex(group, dpIdx)]; }
  return max(langSize(dpIdx, cs.numNonterminals), 1u);
}

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let r = gid.x;
  let c = r + su.span;
  if (c >= cs.numStates || c <= idx_uni.suffix_prefix || r > idx_uni.suffix_prefix) { return; }
  let group = gid.z / cs.numNonterminals;
  let A = gid.z % cs.numNonterminals;
  if (group >= idx_uni.suffix_groups) { return; }

  let dpIdx = (r * cs.numStates + c) * cs.numNonterminals + A;
  let val = dp_in[dpIdx];
  if (val == 0u) { return; }

  let required = getSuffixToken(group);
  var total = matchingLiteralCount(val, A, required);
  let base = bp_offset[dpIdx];
  let count = bp_count[dpIdx];
  for (var i = 0u; i < count; i = i + 1u) {
    let pairBase = (base + i) * 2u;
    let left = bp_storage[pairBase];
    let right = bp_storage[pairBase + 1u];
    total = sat_add(total, sat_mul(conditionedSize(group, left), conditionedSize(group, right)));
  }
  suffix_sizes[suffixIndex(group, dpIdx)] = total;
}""")


/** Enumerates words conditioned on a required terminal at the suffix cut. */
//language=wgsl
val suffix_enum_words_wor by Shader("""$SUFFIX_TERM_STRUCT
@group(0) @binding(0) var<storage, read>        dp_in       : array<u32>;
@group(0) @binding(1) var<storage, read>        bp_count    : array<u32>;
@group(0) @binding(2) var<storage, read>        bp_offset   : array<u32>;
@group(0) @binding(3) var<storage, read>        bp_storage  : array<u32>;
@group(0) @binding(4) var<storage, read>        ls_sparse   : array<u32>;
@group(0) @binding(5) var<storage, read>        terminals   : Terminals;
@group(0) @binding(6) var<storage, read_write>  idx_uni     : SuffixIndexUniforms;
@group(0) @binding(7) var<storage, read>        root_aux    : array<u32>;   // conditioned sizes
@group(0) @binding(8) var<storage, read_write>  sampled     : array<u32>;   // out packets

$SUFFIX_CHART_DECODING_HELPERS

const PKT_HDR_LEN : u32 = ${PKT_HDR_LEN}u;
const NEG_MASK    : u32 = ${NEG_STR_LIT};
const TOKEN_MASK  : u32 = ${PACKED_TOKEN_MASK}u;
const EDIT_SHIFT  : u32 = ${PACKED_EDIT_SHIFT}u;
const SENTINEL    : u32 = 0xffffffffu;

$WGSL_LANG_SIZE

fn binarySearchCDF(base: u32, len: u32, needle: u32) -> u32 {
  var lo: u32 = 0u;
  var hi: u32 = len;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (needle < ls_sparse[base + mid]) { hi = mid; } else { lo = mid + 1u; }
  }

  return base + lo;
}

fn packEditToken(token: u32, val: u32) -> u32 {
  let editCode = (val & EDIT_DEL_MASK) >> EDIT_DEL_SHIFT;
  var editTag = 0u;
  if (editCode == EDIT_INSERT_CODE) { editTag = ${PACKED_INSERTION_TAG}u; }
  else if (editCode != 0u) { editTag = editCode + ${PACKED_SUBSTITUTION_TAG}u; }
  else if ((val & EDIT_SUB_BIT) != 0u) { editTag = ${PACKED_SUBSTITUTION_TAG}u; }
  return (token & TOKEN_MASK) | (editTag << EDIT_SHIFT);
}

fn spansSuffix(dpIdx: u32) -> bool {
  let cell = dpIdx / idx_uni.numNonterminals;
  let r = cell / idx_uni.numStates;
  let c = cell % idx_uni.numStates;
  return r <= idx_uni.suffix_prefix && idx_uni.suffix_prefix < c;
}

fn suffixIndex(group: u32, dpIdx: u32) -> u32 {
  let nt = dpIdx % idx_uni.numNonterminals;
  let cell = dpIdx / idx_uni.numNonterminals;
  let r = cell / idx_uni.numStates;
  let c = cell % idx_uni.numStates;
  let rightWidth = idx_uni.numStates - idx_uni.suffix_prefix - 1u;
  let cellsPerGroup = (idx_uni.suffix_prefix + 1u) * rightWidth;
  let compactCell = r * rightWidth + c - idx_uni.suffix_prefix - 1u;
  return (group * cellsPerGroup + compactCell) * idx_uni.numNonterminals + nt;
}

fn conditionedSize(group: u32, dpIdx: u32) -> u32 {
  if (spansSuffix(dpIdx)) { return root_aux[suffixIndex(group, dpIdx)]; }
  return langSize(dpIdx, idx_uni.numNonterminals);
}

// Number of derivations in a group's roots through `rank`, enumerating roots
// diagonally by rank. This is a dense union even when root sizes differ, so a
// short root cannot leave permanent holes in another root's sample lanes.
fn suffixRanksThrough(group: u32, pairBegin: u32, pairEnd: u32, rank: u32) -> u32 {
  var total = 0u;
  let ranks = rank + 1u;
  for (var pair = pairBegin; pair < pairEnd; pair = pair + 1u) {
    let root = getSuffixPairRoot(pair);
    let size = conditionedSize(group, getStartIdx(root));
    total = sat_add(total, min(size, ranks));
  }
  return total;
}

fn literalAllows(val: u32, nt: u32, required: u32) -> bool {
  let ntLen = get_nt_tm_lens(nt);
  let ntOff = get_offsets(nt);
  var present = false;
  for (var i = 0u; i < ntLen; i = i + 1u) {
    if (get_all_tms(ntOff + i) == required) { present = true; break; }
  }
  if (!present) { return false; }

  let predicate = val & PREDICATE_MASK;
  if (predicate == LIT_ALL) { return true; }
  let litEnc = (predicate >> 1u) & 0x03ffffffu;
  if (litEnc == 0u || litEnc > ntLen) { return false; }
  let encoded = get_all_tms(ntOff + litEnc - 1u);
  if ((predicate & NEG_MASK) != 0u) { return required != encoded; }
  return required == encoded;
}

fn decodeRequiredLiteral(
  dpIdx: u32,
  val: u32,
  required: u32,
  word: ptr<function, array<u32, ${MAX_WORD_LEN}u>>,
  wLen: ptr<function, u32>
) -> bool {
  let cap = idx_uni.maxWordLen - PKT_HDR_LEN;
  let nt = dpIdx % idx_uni.numNonterminals;
  if (*wLen >= cap || *wLen >= ${MAX_WORD_LEN}u || !literalAllows(val, nt, required)) { return false; }
  (*word)[*wLen] = packEditToken(required + 1u, val);
  *wLen = *wLen + 1u;
  return true;
}

fn decodeLiteral(
  dpIdx: u32,
  val: u32,
  variant: u32, // must be < litCount(dpIdx)
  word: ptr<function, array<u32, ${MAX_WORD_LEN}u>>,
  wLen: ptr<function, u32>
) -> bool {
  let cap = idx_uni.maxWordLen - PKT_HDR_LEN;
  if (*wLen >= cap || *wLen >= ${MAX_WORD_LEN}u) { return false; }

  let nt    = dpIdx % idx_uni.numNonterminals;
  let ntLen = get_nt_tm_lens(nt);
  if (ntLen == 0u) { return false; }
  let ntOff = get_offsets(nt);
  let predicate = val & PREDICATE_MASK;
  var token: u32;

  // wildcard: choose variant mod |Σ_A|
  if (predicate == LIT_ALL) { token = get_all_tms(ntOff + (variant % ntLen)) + 1u; }
  else {
    let negLit = (predicate & NEG_MASK) != 0u;
    let litEnc = (predicate >> 1u) & 0x03ffffffu;
    if (litEnc == 0u || litEnc > ntLen) { return false; }

    if (negLit) {
      if (ntLen <= 1u) { return false; }
      // exclude the (litEnc-1)th terminal from Σ_A
      let excl = litEnc - 1u;
      let v    = variant % (ntLen - 1u);
      let idx  = select(v, v + 1u, v >= excl);
      token = get_all_tms(ntOff + idx) + 1u;
    } else { token = get_all_tms(ntOff + (litEnc - 1u)) + 1u; }
  }
  (*word)[*wLen] = packEditToken(token, val);
  *wLen = *wLen + 1u;
  return true;
}

struct Frame { dp: u32, rk: u32 }

// ---------- Feistel permutation helpers (for WOR-by-rank when total not saturated) ----------
fn ceil_pow2_even(x: u32) -> u32 {
  // returns smallest even k such that 2^k >= x, capped at 32
  var k: u32 = 0u;
  var p: u32 = 1u;
  while (p < x && k < 32u) { p = p << 1u; k = k + 1u; }
  if ((k & 1u) == 1u) { k = k + 1u; }
  return min(k, 32u);
}

// 4-round Feistel permutation over k bits (k even, <= 32)
fn feistel_perm(x: u32, seed: u32, k_even: u32) -> u32 {
  let h: u32 = k_even >> 1u;
  let mask: u32 = (1u << h) - 1u;

  var l: u32 = x & mask;
  var r: u32 = (x >> h) & mask;

  for (var round: u32 = 0u; round < 4u; round = round + 1u) {
    let f: u32 = ((r * 0x9e3779b9u) ^ (seed + round * 0x7f4a7c15u)) & mask;
    let nl: u32 = r;
    let nr: u32 = l ^ f;
    l = nl; r = nr;
  }

  return (r << h) | l;
}

// Permute sid into [0,total) via cycle-walking on [0, 2^k).
// IMPORTANT: handle k=32 without ever computing (1u<<32).
fn permute_in_range(sid: u32, seed: u32, total: u32) -> u32 {
  let k_even: u32 = ceil_pow2_even(total);

  var x: u32 = sid;

  // If k < 32, we can mask into [0,2^k)
  if (k_even < 32u) {
    let m: u32 = 1u << k_even;
    let mask: u32 = m - 1u;
    x = sid & mask;
  }

  var attempts: u32 = 0u;
  while (attempts < 64u) {
    x = feistel_perm(x, seed, k_even);
    if (x < total) { return x; }
    attempts = attempts + 1u;
  }

  return x % total;
}

// ---------- RNG helpers (used when saturation makes rank/CDF meaningless) ----------
fn mix32(x: u32) -> u32 {
  var z = x + 0x9e3779b9u;
  z = (z ^ (z >> 16u)) * 0x85ebca6bu;
  z = (z ^ (z >> 13u)) * 0xc2b2ae35u;
  return z ^ (z >> 16u);
}

fn rng_next(state: ptr<function, u32>) -> u32 {
  var x = *state;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  *state = x;
  return x;
}

fn rand_bounded(state: ptr<function, u32>, bound: u32) -> u32 {
  if (bound == 0u) { return 0u; }
  let threshold = (0xffffffffu - bound + 1u) % bound;
  var attempts: u32 = 0u;
  while (attempts < 32u) {
    let r = rng_next(state);
    if (r >= threshold) { return r % bound; }
    attempts = attempts + 1u;
  }

  return rng_next(state) % bound;
}

fn randomRankForSize(state: ptr<function, u32>, size: u32) -> u32 {
  if (size == 0u) { return 0u; }
  if (size == SAT_MAX) { return rand_bounded(state, SAT_MAX); }
  return rand_bounded(state, size);
}

// ---------- Utility: write an empty packet (so failures don't leave garbage) ----------
fn write_empty_packet(sid: u32, levDist: u32) {
  let stride  = idx_uni.maxWordLen;
  let outBase = sid * stride;
  sampled[outBase + 0u] = levDist;
  sampled[outBase + 1u] = 0u;
  if (PKT_HDR_LEN < stride) { sampled[outBase + PKT_HDR_LEN] = 0u; }
}

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sid = gid.x + gid.y * idx_uni.threads;
  if (sid >= idx_uni.max_samples) { return; }

  let numRoots = idx_uni.numStartIndices / 2u;
  if (numRoots == 0u || idx_uni.suffix_groups == 0u) { return; }

  let runSeed: u32 = atomicLoad(&idx_uni.targetCnt) ^ 0xA511E9B3u;
  var rootIdx: u32 = 0u;
  var prk: u32 = 0u;
  var rk: u32 = 0u;
  let suffixGroup = sid % idx_uni.suffix_groups;
  let localSid = sid / idx_uni.suffix_groups;
  let pairBegin = getSuffixGroupOffset(suffixGroup);
  let pairEnd = getSuffixGroupOffset(suffixGroup + 1u);
  if (pairBegin == pairEnd) { write_empty_packet(sid, 0u); return; }

  // Invert the diagonal rank count to obtain a hole-free (root, rank) pair.
  // Rank zero seeds every selected suffix length before any root gets rank one.
  if (suffixRanksThrough(suffixGroup, pairBegin, pairEnd, localSid) <= localSid) {
    write_empty_packet(sid, 0u); return;
  }
  var rankLo = 0u;
  var rankHi = localSid;
  while (rankLo < rankHi) {
    let mid = (rankLo + rankHi) >> 1u;
    if (suffixRanksThrough(suffixGroup, pairBegin, pairEnd, mid) > localSid) {
      rankHi = mid;
    } else {
      rankLo = mid + 1u;
    }
  }
  let localRank = rankLo;
  var withinRank = localSid;
  if (localRank != 0u) {
    withinRank = localSid - suffixRanksThrough(suffixGroup, pairBegin, pairEnd, localRank - 1u);
  }
  var foundRoot = false;
  for (var pair = pairBegin; pair < pairEnd; pair = pair + 1u) {
    let candidate = getSuffixPairRoot(pair);
    let size = conditionedSize(suffixGroup, getStartIdx(candidate));
    if (size > localRank) {
      if (withinRank == 0u && !foundRoot) {
        rootIdx = candidate;
        foundRoot = true;
      } else {
        if (!foundRoot) { withinRank = withinRank - 1u; }
      }
    }
  }
  if (!foundRoot) { write_empty_packet(sid, 0u); return; }
  let rootSize = conditionedSize(suffixGroup, getStartIdx(rootIdx));
  if (rootSize == 0u) { return; }
  prk = permute_in_range(localRank, runSeed ^ suffixGroup ^ rootIdx, rootSize);
  rk = prk;
  if (rootSize != SAT_MAX && rk >= rootSize) { rk = rk % rootSize; }
  let dpRoot  = getStartIdx(rootIdx);
  let levDist = getEditDist(rootIdx);

  var rng: u32 = mix32(runSeed ^ sid ^ prk);
  // DFS decode by rank (without replacement by construction)
  var stack : array<Frame, ${MAX_WORD_LEN}u>;
  var top   : u32 = 0u;
  stack[top] = Frame(dpRoot, rk); top++;

  var word : array<u32, ${MAX_WORD_LEN}u>;
  var wLen : u32 = 0u;

  loop {
    if (top == 0u) { break; }
    top = top - 1u;

    let fr = stack[top];
    let d  = fr.dp;
    rk     = fr.rk;

    let val      = dp_in[d];
    if (val == 0u) { write_empty_packet(sid, levDist); return; }

    let nt = d % idx_uni.numNonterminals;
    let constrained = idx_uni.suffix_groups != 0u && spansSuffix(d);
    var litCount = count_tms(val, nt);
    if (constrained) { litCount = select(0u, 1u, literalAllows(val, nt, getSuffixToken(suffixGroup))); }

    let expCnt   = bp_count[d];
    let base2    = bp_offset[d];
    var lastCDF  : u32 = 0u;
    if (expCnt != 0u) { lastCDF = ls_sparse[base2 + expCnt - 1u]; }
    let tot      = select(litCount, lastCDF, expCnt != 0u);

    // Decode an exact rank from the language conditioned on the required
    // terminal. Only the unique child spanning suffix_prefix remains
    // conditioned; its sibling uses the ordinary language size.
    if (constrained) {
      let conditionedTotal = conditionedSize(suffixGroup, d);
      if (conditionedTotal == 0u) { write_empty_packet(sid, levDist); return; }
      if (conditionedTotal != SAT_MAX && rk >= conditionedTotal) { rk = rk % conditionedTotal; }

      // Once the conditioned count saturates, a saturated prefix sum would
      // make every later valid expansion unreachable. Choose among all
      // nonempty conditioned alternatives instead, then sample both children
      // within their own (possibly saturated) languages.
      if (conditionedTotal == SAT_MAX) {
        var viable = litCount;
        for (var rel = 0u; rel < expCnt; rel = rel + 1u) {
          let choice = base2 + rel;
          let left = bp_storage[2u * choice + 0u];
          let right = bp_storage[2u * choice + 1u];
          if (conditionedSize(suffixGroup, left) != 0u && conditionedSize(suffixGroup, right) != 0u) {
            viable = viable + 1u;
          }
        }
        if (viable == 0u) { write_empty_packet(sid, levDist); return; }
        var selected = rand_bounded(&rng, viable);
        if (selected < litCount) {
          if (!decodeRequiredLiteral(d, val, getSuffixToken(suffixGroup), &word, &wLen)) {
            write_empty_packet(sid, levDist); return;
          }
          continue;
        }
        selected = selected - litCount;
        var saturatedLeft = SENTINEL;
        var saturatedRight = SENTINEL;
        for (var rel = 0u; rel < expCnt; rel = rel + 1u) {
          let choice = base2 + rel;
          let left = bp_storage[2u * choice + 0u];
          let right = bp_storage[2u * choice + 1u];
          let sizeL = conditionedSize(suffixGroup, left);
          let sizeR = conditionedSize(suffixGroup, right);
          if (sizeL != 0u && sizeR != 0u) {
            if (selected == 0u && saturatedLeft == SENTINEL) {
              saturatedLeft = left;
              saturatedRight = right;
            }
            if (saturatedLeft == SENTINEL) { selected = selected - 1u; }
          }
        }
        if (saturatedLeft == SENTINEL) { write_empty_packet(sid, levDist); return; }
        if (top + 2u > ${MAX_WORD_LEN}u) { write_empty_packet(sid, levDist); return; }
        stack[top] = Frame(saturatedRight, randomRankForSize(&rng, conditionedSize(suffixGroup, saturatedRight))); top++;
        stack[top] = Frame(saturatedLeft, randomRankForSize(&rng, conditionedSize(suffixGroup, saturatedLeft))); top++;
        continue;
      }

      if (rk < litCount) {
        if (!decodeRequiredLiteral(d, val, getSuffixToken(suffixGroup), &word, &wLen)) {
          write_empty_packet(sid, levDist); return;
        }
        continue;
      }

      var previous = litCount;
      var chosen = SENTINEL;
      var inside = 0u;
      var chosenLeft = 0u;
      var chosenRight = 0u;
      var chosenRightSize = 0u;
      for (var rel = 0u; rel < expCnt; rel = rel + 1u) {
        let choice = base2 + rel;
        let left = bp_storage[2u * choice + 0u];
        let right = bp_storage[2u * choice + 1u];
        let sizeL = conditionedSize(suffixGroup, left);
        let sizeR = conditionedSize(suffixGroup, right);
        let next = sat_add(previous, sat_mul(sizeL, sizeR));
        if (chosen == SENTINEL && rk < next) {
          chosen = choice;
          inside = rk - previous;
          chosenLeft = left;
          chosenRight = right;
          chosenRightSize = sizeR;
        }
        previous = next;
      }
      if (chosen == SENTINEL || chosenRightSize == 0u) { write_empty_packet(sid, levDist); return; }

      var rkL = 0u;
      var rkR = inside;
      if (chosenRightSize != SAT_MAX) {
        rkL = inside / chosenRightSize;
        rkR = inside % chosenRightSize;
      }
      if (top + 2u > ${MAX_WORD_LEN}u) { write_empty_packet(sid, levDist); return; }
      stack[top] = Frame(chosenRight, rkR); top++;
      stack[top] = Frame(chosenLeft, rkL); top++;
      continue;
    }

    if (tot == 0u) { write_empty_packet(sid, levDist); return; }

    // ---- Saturation-aware fallback ----
    // If the expansion CDF is saturated, binarySearchCDF collapses to the first production.
    // Switch to RNG-driven choice to restore diversity.
    if (lastCDF == SAT_MAX && expCnt != 0u) {
      // choose literal vs expansion (roughly proportional to litCount vs expCnt)
      let pickLit = (litCount != 0u) && (rand_bounded(&rng, sat_add(litCount, expCnt)) < litCount);

      if (pickLit) {
        let v = rand_bounded(&rng, litCount);
        if (!decodeLiteral(d, val, v, &word, &wLen)) { write_empty_packet(sid, levDist); return; }
        continue;
      }

      // choose expansion uniformly among expCnt
      let rel      = rand_bounded(&rng, expCnt);
      let choiceIx = base2 + rel;

      let left  = bp_storage[2u * choiceIx + 0u];
      let right = bp_storage[2u * choiceIx + 1u];

      let sizeR = langSize(right, idx_uni.numNonterminals);
      let sizeL = langSize(left,  idx_uni.numNonterminals);

      var rkL: u32 = 0u;
      var rkR: u32 = 0u;

      if (sizeR == 0u || sizeL == 0u) { write_empty_packet(sid, levDist); return; }

      if (sizeR == SAT_MAX || sizeL == SAT_MAX) {
        rkL = randomRankForSize(&rng, sizeL);
        rkR = randomRankForSize(&rng, sizeR);
      } else {
        let prod   = sat_mul(sizeL, sizeR);
        let inside = rand_bounded(&rng, prod);
        rkL = inside / sizeR;
        rkR = inside % sizeR;
      }

      if (top + 2u > ${MAX_WORD_LEN}u) { write_empty_packet(sid, levDist); return; }
      stack[top] = Frame(right, rkR); top++;
      stack[top] = Frame(left,  rkL); top++;
      continue;
    }

    // ---- Normal rank/CDF path (non-saturated) ----
    if (rk >= tot) { rk = rk % tot; }

    if (rk < litCount) {
      if (!decodeLiteral(d, val, rk, &word, &wLen)) { write_empty_packet(sid, levDist); return; }
      continue;
    }

    if (expCnt == 0u) { write_empty_packet(sid, levDist); return; }
    let choiceIx = binarySearchCDF(base2, expCnt, rk);

    // if choiceIx == base2+expCnt, rk was out of range; clamp to last
    let cIx = select(choiceIx, base2 + expCnt - 1u, choiceIx >= base2 + expCnt);

    var prevCDF = litCount;
    if (cIx != base2) { prevCDF = ls_sparse[cIx - 1u]; }
    let inside  = rk - prevCDF;

    let left  = bp_storage[2u * cIx + 0u];
    let right = bp_storage[2u * cIx + 1u];

    let sizeR = langSize(right, idx_uni.numNonterminals);
    if (sizeR == 0u) { write_empty_packet(sid, levDist); return; }

    let rkL = inside / sizeR;
    let rkR = inside % sizeR;

    if (top + 2u > ${MAX_WORD_LEN}u) { write_empty_packet(sid, levDist); return; }
    stack[top] = Frame(right, rkR); top++;
    stack[top] = Frame(left,  rkL); top++;
  }

  // Write packet
  let stride  = idx_uni.maxWordLen;
  let outBase = sid * stride;

  sampled[outBase + 0u] = levDist;
  sampled[outBase + 1u] = 0u; // markov_score fills later

  for (var i = 0u; i < wLen && (PKT_HDR_LEN + i) < stride; i = i + 1u) { sampled[outBase + PKT_HDR_LEN + i] = word[i]; }
  // terminator
  if (PKT_HDR_LEN + wLen < stride) { sampled[outBase + PKT_HDR_LEN + wLen] = 0u; }
}""".trimIndent())
