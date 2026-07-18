import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.buildLanguageSizeBuf
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.*
import js.typedarrays.Int32Array
import kotlinx.coroutines.await
import kotlin.js.Promise
import kotlin.time.TimeSource
import web.gpu.GPUBuffer

const val MAX_WDFA_FRONTIER_V2 = 32_768
const val WDFA_META_STRIDE_V2 = 7
const val WDFA_STATUS_ACTIVE_V2 = 0
const val WDFA_STATUS_DONE_V2 = 1
const val WDFA_STATUS_DEAD_V2 = 2
const val MAX_WDFA_TRACE_STEPS_V2 = MAX_WORD_LEN * 2 + 8
const val MAX_WDFA_COMPLETIONS_V2 = 262_144
const val WDFA_RESAMPLE_SCALE_V2 = 4_096u

private data class WdfaFrontierV2( var buf: GPUBuffer, var count: Int ) { fun destroy() = buf.destroy() }

private fun u32ToLongV2(x: Int): Long = x.toUInt().toLong()

private fun satAddLongV2(a: Long, b: Long): Long =
  if (Long.MAX_VALUE - a < b) Long.MAX_VALUE else a + b

private fun sumLangSizesV2(sizes: Iterable<Long>): Long =
  sizes.fold(0L, ::satAddLongV2)

private fun pctV2(n: Long, d: Long): Double =
  n.toDouble() * 100.0 / d.toDouble().coerceAtLeast(1.0)

private fun allocWdfaFrontierV2(count: Int, stride: Int): WdfaFrontierV2 {
  val intsPerState = WDFA_META_STRIDE_V2 + 2 * stride
  return WdfaFrontierV2(
    buf = GPUBuffer(count.toLong() * intsPerState * 4L, STCPSD),
    count = count
  )
}

private suspend fun readExclusivePlusCountV2(exclusive: GPUBuffer, counts: GPUBuffer, n: Int): Long {
  if (n <= 0) return 0L
  val staging = GPUBuffer(8, GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
  val lastOffset = (n - 1).toDouble() * 4.0
  val encoder = gpu.createCommandEncoder()

  encoder.copyBufferToBuffer(
    source = exclusive,
    sourceOffset = lastOffset,
    destination = staging,
    destinationOffset = 0.0,
    size = 4.0
  )
  encoder.copyBufferToBuffer(
    source = counts,
    sourceOffset = lastOffset,
    destination = staging,
    destinationOffset = 4.0,
    size = 4.0
  )

  gpu.queue.submit(arrayOf(encoder.finish()))
  staging.mapAsync(GPUBufferUsage.MAP_READ).unsafeCast<Promise<*>>().await()

  val ints = Int32Array(staging.getMappedRange())
  val total = ints[0].toUInt().toLong() + ints[1].toUInt().toLong()

  staging.unmap()
  staging.destroy()
  return total
}

private suspend fun readU32V2(buf: GPUBuffer, wordIndex: Int = 0): Long {
  val staging = GPUBuffer(4, GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
  val encoder = gpu.createCommandEncoder()

  encoder.copyBufferToBuffer(
    source = buf,
    sourceOffset = wordIndex.toDouble() * 4.0,
    destination = staging,
    destinationOffset = 0.0,
    size = 4.0
  )

  gpu.queue.submit(arrayOf(encoder.finish()))
  staging.mapAsync(GPUBufferUsage.MAP_READ).unsafeCast<Promise<*>>().await()

  val value = Int32Array(staging.getMappedRange())[0].toUInt().toLong()

  staging.unmap()
  staging.destroy()
  return value
}

suspend fun intersectionPipelineV2(
  cfg: CFG,
  fsa: FSA,
  ledBuffer: Int,
  codePoints: IntArray,
  rerankerQuery: List<String>? = null,
  chartInitializer: Shader = init_lev_chart
): List<String> {
  val wdfaBuf = wdfa ?: error("intersectionPipelineV2 requires a WDFA buffer")
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("V2 WDFA FSA(|Q|=$numStates, |delta|=${fsa.transit.size}), ${cfg.calcStats()}")

  val metaT = TimeSource.Monotonic.markNow()
  val metaBuf = packMetadata(cfg, fsa)
  mark("pack metadata", metaT)
  log("Packed metadata in ${timings["pack metadata"]}ms")

  val tmBuf = cfg.termBuf
  val wordBuf = codePoints.toGPUBuffer()
  val totalSize = numStates * numStates * numNTs
  val activeWords = (numNTs + 31) ushr 5

  val dpBuf = Shader.createParseChart(STCPSD, totalSize)
  val activeBuf = GPUBuffer((numStates * numStates * activeWords * 4).toLong(), STCPSD)

  log(
    "V2 buffers: dp=${dpBuf.size}B (${totalSize} cells), " +
        "active=${activeBuf.size}B (${activeWords} words/cell), " +
        "word=${wordBuf.size}B, tm=${tmBuf.size}B"
  )

  timings["init chart"] = timedGPUIsolated("Init chart") {
    chartInitializer(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  }

  val closureT = TimeSource.Monotonic.markNow()
  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  mark("matrix closure", closureT)
  log("Matrix closure reached in: ${timings["matrix closure"]}ms")

  logActiveNTGrid(activeBuf, numStates, numNTs, limit = minOf(48, numStates))

  val rootsT = TimeSource.Monotonic.markNow()
  val startNT = cfg.bindex[START_SYMBOL]
  val rootQuery = fsa.finalIdxs.map { it * numNTs + startNT }
  val allStartIds = rootQuery
    .zip(dpBuf.readIndices(rootQuery))
    .filter { (_, v) -> v != 0 }
    .map { it.first }
  mark("read roots", rootsT)

  if (allStartIds.isEmpty()) {
    listOf(activeBuf, wordBuf, metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList<String>().also { log("V2 WDFA: no valid parse found") }
  }

  log("V2 WDFA valid parse found: dpComplete has ${allStartIds.size} start indices")

  val bpT = TimeSource.Monotonic.markNow()
  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNTs, dpBuf, metaBuf)
  mark("build backpointers", bpT)
  val totalExp = bpStorageBuf.size.toInt() / (2 * 4)
  log("Built backpointers in ${timings["build backpointers"]}ms | expansions=$totalExp")

  val filterT = TimeSource.Monotonic.markNow()
  val statesToDist = allStartIds.map { it to fsa.idsToCoords[(it - startNT) / numNTs]!!.second }
  val led = statesToDist.minOf { it.second }
  val rootEntries = statesToDist
    .filter { it.second in (led..(led + ledBuffer)) }
    .sortedWith(compareBy<Pair<Int, Int>> { it.second }.thenBy { it.first })
    .also { log("V2 start indices: total=${it.size}, roots=${it.size}, LED=$led, window=[${led}, ${led + ledBuffer}]") }
  mark("filter roots", filterT)

  val maxRepairLen = fsa.width + fsa.height + 10
  if (MAX_WORD_LEN < maxRepairLen) {
    listOf(activeBuf, wordBuf, metaBuf, dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf)
      .forEach(GPUBuffer::destroy)
    return emptyList<String>().also {
      log("V2 WDFA max repair length exceeded $MAX_WORD_LEN ($maxRepairLen)")
    }
  }

  val langSizeT = TimeSource.Monotonic.markNow()
  val rootLangSizes =
    if (rootEntries.isEmpty()) emptyList()
    else {
      val lsDense = buildLanguageSizeBuf(numStates, numNTs, dpBuf, metaBuf, tmBuf)
      try {
        lsDense.readIndices(rootEntries.map { it.first }).map(::u32ToLongV2)
      } finally {
        lsDense.destroy()
      }
    }
  mark("build ls dense", langSizeT)

  val rootsByDist = rootEntries.zip(rootLangSizes)
    .groupBy({ it.first.second }, { it.first.first to it.second })
    .entries
    .sortedBy { it.key }
    .map { it.key to it.value }
  val totalLangSize = sumLangSizesV2(rootLangSizes)
  val bucketSummary = rootsByDist.joinToString(", ") { (dist, roots) ->
    "Δ=$dist:${sumLangSizesV2(roots.map { it.second })}/${roots.size} roots"
  }
  log("V2 language size: total=$totalLangSize across ${rootEntries.size} roots, maxRepairLen=$maxRepairLen, buckets=[$bucketSummary]")

  val decodeT = TimeSource.Monotonic.markNow()
  val decoderTopK = if (rerankerQuery != null) RERANKER_TOP_K_SAMP else TOP_K_SAMP
  log("V2 decoding ${rootEntries.size} roots across ${rootsByDist.size} edit-distance bucket(s)")
  val result = mutableListOf<String>()
  val seen = linkedSetOf<String>()
  for ((dist, roots) in rootsByDist) {
    if (result.size >= decoderTopK) break

    val startIdxs = roots.flatMap { listOf(it.first, dist) }
    val bucketLangSize = sumLangSizesV2(roots.map { it.second })
    val idxUniBuf = packStruct(
      listOf(0, maxRepairLen, numNTs, numStates, DISPATCH_GROUP_SIZE_X, MAX_SAMPLES),
      startIdxs.toGPUBuffer()
    )

    try {
      val bucketResult = wdfaRegexFrontierDecoderV2(
        dpBuf = dpBuf,
        bpCountBuf = bpCountBuf,
        bpOffsetBuf = bpOffsetBuf,
        bpStorageBuf = bpStorageBuf,
        tmBuf = tmBuf,
        idxUniBuf = idxUniBuf,
        wdfa = wdfaBuf,
        maxRepairLen = maxRepairLen,
        cfg = cfg,
        numRoots = roots.size,
        k = decoderTopK - result.size,
        expectedLanguageSize = bucketLangSize,
        distanceLabel = dist
      )

      for (word in bucketResult) {
        if (seen.add(word)) result.add(word)
        if (result.size >= decoderTopK) break
      }
    } finally {
      idxUniBuf.destroy()
    }
  }
  mark("decode", decodeT)

  val rankedResult =
    if (rerankerQuery != null) {
      val candidates = result.take(RERANKER_TOP_K_SAMP)
      val rerankT = TimeSource.Monotonic.markNow()
      RepairReranker.rerankOrOriginal(rerankerQuery, candidates)
        .also { mark("rerank", rerankT) }
    } else result

  return rankedResult.also {
    listOf(
      metaBuf, dpBuf, activeBuf, wordBuf,
      bpCountBuf, bpOffsetBuf, bpStorageBuf
    ).forEach(GPUBuffer::destroy)
  }
}

suspend fun wdfaRegexFrontierDecoderV2(
  dpBuf: GPUBuffer,
  bpCountBuf: GPUBuffer,
  bpOffsetBuf: GPUBuffer,
  bpStorageBuf: GPUBuffer,
  tmBuf: GPUBuffer,
  idxUniBuf: GPUBuffer,
  wdfa: GPUBuffer,
  maxRepairLen: Int,
  cfg: CFG,
  numRoots: Int,
  k: Int = TOP_K_SAMP,
  expectedLanguageSize: Long? = null,
  distanceLabel: Int? = null
): MutableList<String> {
  log("Using V2 WDFA regex frontier decoder...")
  val t0 = TimeSource.Monotonic.markNow()

  if (numRoots == 0) return mutableListOf()

  val frontierCap = MAX_WDFA_FRONTIER_V2
  val completionCap = MAX_WDFA_COMPLETIONS_V2
  var exact = numRoots <= frontierCap
  val initialCount = if (exact) numRoots else frontierCap
  val bpMetaBuf = packStructBorrowed(emptyList(), bpCountBuf, bpOffsetBuf, bpStorageBuf)
  var frontierA = allocWdfaFrontierV2(initialCount, maxRepairLen)
  var frontierB = allocWdfaFrontierV2(frontierCap, maxRepairLen)
  val completionBuf = GPUBuffer(completionCap * maxRepairLen * 4L, STCPSD)
  val completionCntBuf = intArrayOf(0).toGPUBuffer(STCPSD)

  try {
    val initPrm = intArrayOf(initialCount, frontierCap, maxRepairLen, MAX_WDFA_TRACE_STEPS_V2, 0, numRoots, 0, 0)
      .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

    timings["wdfa frontier init"] = timedGPUIsolated("WDFA frontier init") {
      wdfa_frontier_init_v2(frontierA.buf, idxUniBuf, initPrm, wdfa)((initialCount + 255) / 256)
    }
    initPrm.destroy()

    val decodeT = TimeSource.Monotonic.markNow()
    for (step in 0 until MAX_WDFA_TRACE_STEPS_V2) {
      val succCountBuf = GPUBuffer(frontierA.count * 4L, STCPSD)
      val countPrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_WDFA_TRACE_STEPS_V2, step, numRoots, 0, 0)
        .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      wdfa_frontier_count_succ_v2(
        dpBuf, tmBuf, bpMetaBuf, frontierA.buf, succCountBuf, idxUniBuf, countPrm
      )((frontierA.count + 255) / 256)

      val succOffBuf = Shader.prefixSumGPU(succCountBuf, frontierA.count)
      val totalSucc = readExclusivePlusCountV2(succOffBuf, succCountBuf, frontierA.count)
      countPrm.destroy()

      if (totalSucc == 0L) {
        succOffBuf.destroy()
        succCountBuf.destroy()
        break
      }

      if (exact && totalSucc <= frontierCap.toLong()) {
        frontierB.destroy()
        frontierB = allocWdfaFrontierV2(totalSucc.toInt(), maxRepairLen)

        val writePrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_WDFA_TRACE_STEPS_V2, step, numRoots, 0, 0)
          .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

        wdfa_frontier_write_exact_v2(
          dpBuf, tmBuf, wdfa, bpMetaBuf,
          frontierA.buf, succOffBuf, succCountBuf,
          frontierB.buf, idxUniBuf, writePrm
        )((frontierA.count + 63) / 64)

        frontierB.count = totalSucc.toInt()
        writePrm.destroy()
      } else {
        exact = false
        frontierB.destroy()
        frontierB = allocWdfaFrontierV2(frontierCap, maxRepairLen)

        val weightBuf = GPUBuffer(frontierA.count * 4L, STCPSD)
        val weightPrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_WDFA_TRACE_STEPS_V2, step, numRoots, 0, 0)
          .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

        wdfa_frontier_parent_weights_v2(frontierA.buf, succCountBuf, weightBuf, wdfa, weightPrm)((frontierA.count + 255) / 256)
        val parentCDF = Shader.prefixSumGPU(weightBuf, frontierA.count)
        val totalWeight = readExclusivePlusCountV2(parentCDF, weightBuf, frontierA.count).coerceAtMost(UInt.MAX_VALUE.toLong()).toInt()

        if (totalWeight <= 0) {
          weightPrm.destroy()
          parentCDF.destroy()
          weightBuf.destroy()
          succOffBuf.destroy()
          succCountBuf.destroy()
          break
        }

        val samplePrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_WDFA_TRACE_STEPS_V2, step, numRoots, totalWeight, 0)
          .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

        wdfa_frontier_sampled_step_v2(
          dpBuf, tmBuf, wdfa, bpMetaBuf,
          frontierA.buf, parentCDF, weightBuf,
          frontierB.buf, idxUniBuf, samplePrm
        )((frontierCap + 255) / 256)

        frontierB.count = frontierCap

        samplePrm.destroy()
        weightPrm.destroy()
        parentCDF.destroy()
        weightBuf.destroy()
      }

      succOffBuf.destroy()
      succCountBuf.destroy()

      val tmp = frontierA
      frontierA = frontierB
      frontierB = tmp

      val emitPrm = intArrayOf(frontierA.count, completionCap, maxRepairLen, MAX_WDFA_TRACE_STEPS_V2, step, numRoots, 0, 0)
        .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
      wdfa_frontier_emit_done_packets_v2(frontierA.buf, completionBuf, completionCntBuf, emitPrm)((frontierA.count + 255) / 256)
      emitPrm.destroy()
    }
    timings["wdfa frontier decode"] = decodeT.elapsedNow().inWholeMilliseconds.toInt()

    val totalCompletions = readU32V2(completionCntBuf)
    val recordedCompletions = totalCompletions.coerceAtMost(completionCap.toLong()).toInt()
    val label = distanceLabel?.let { " Δ=$it" } ?: ""
    if (expectedLanguageSize != null) {
      val emittedPct = pctV2(totalCompletions, expectedLanguageSize)
      val recordedPct = pctV2(recordedCompletions.toLong(), expectedLanguageSize)
      log(
        "V2 language saturation$label: " +
          "emitted=$emittedPct% ($totalCompletions/$expectedLanguageSize), " +
          "recorded=$recordedPct% ($recordedCompletions/$expectedLanguageSize), " +
          "maxRepairLen=$maxRepairLen, frontierExact=$exact"
      )
    }
    if (recordedCompletions == 0) {
      log("V2 WDFA frontier produced no completed packets in ${t0.elapsedNow()} | frontier=${frontierA.count} | frontierExact=$exact")
      return mutableListOf()
    }

    val totalGroups = (recordedCompletions + 255) / 256
    val threads = minOf(DISPATCH_GROUP_SIZE_X, maxOf(1, totalGroups))
    val selGroupsY = (totalGroups + threads - 1) / threads
    val epsilonTok = cfg.tmMap["eps"]?.plus(1) ?: cfg.tmMap["epsilon"]?.plus(1) ?: cfg.tmMap["ε"]?.plus(1) ?: 0
    val prmBuf = intArrayOf(recordedCompletions, k, maxRepairLen, threads, epsilonTok, 0, 0, 0)
      .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
    val idxBuf = IntArray(k) { -1 }.toGPUBuffer(STCPSD)
    val scrBuf = IntArray(k) { Int.MAX_VALUE }.toGPUBuffer(STCPSD)
    val hashBuf = IntArray(k) { 0 }.toGPUBuffer(STCPSD)
    val bestBuf = GPUBuffer(k * maxRepairLen * 4, STCPSD)

    try {
      timings["select_top_k"] = timedGPUIsolated("Select top-k") {
        select_top_k_unique_v2(prmBuf, completionBuf, idxBuf, scrBuf, hashBuf)(threads, selGroupsY)
      }
      timings["gather_top_k"] = timedGPUIsolated("Gather top-k") {
        gather_top_k(prmBuf, completionBuf, idxBuf, bestBuf)(k)
      }

      val topK = bestBuf.readJSIntArray()
      log("V2 WDFA frontier/select/gather read ${topK.length} = ${k}x${maxRepairLen}x4 bytes in ${t0.elapsedNow()} | frontier=${frontierA.count} | completions=$recordedCompletions/$totalCompletions | frontierExact=$exact")
      return decodePackets(topK, cfg, maxRepairLen).toMutableList()
    } finally {
      listOf(prmBuf, idxBuf, scrBuf, hashBuf, bestBuf).forEach(GPUBuffer::destroy)
    }
  } finally {
    bpMetaBuf.destroy()
    frontierA.destroy()
    frontierB.destroy()
    completionBuf.destroy()
    completionCntBuf.destroy()
  }
}

//language=wgsl
const val WDFA_BP_STRUCT_V2 = """
struct WdfaBackpointers {
  bpCountOffset   : u32, bpCountSize   : u32,
  bpOffsetOffset  : u32, bpOffsetSize  : u32,
  bpStorageOffset : u32, bpStorageSize : u32,

  payload : array<u32>
};

fn bp_count_at(i: u32) -> u32 { return bp.payload[bp.bpCountOffset + i]; }
fn bp_offset_at(i: u32) -> u32 { return bp.payload[bp.bpOffsetOffset + i]; }
fn bp_storage_at(i: u32) -> u32 { return bp.payload[bp.bpStorageOffset + i]; }
"""

//language=wgsl
const val WDFA_FRONTIER_STRUCT_V2 = """
struct WdfaFrontierParams {
  frontierCount : u32,
  frontierCap   : u32,
  stride        : u32,
  traceStride   : u32,
  step          : u32,
  numRoots      : u32,
  totalWeight   : u32,
  reserved      : u32,
};
"""

//language=wgsl
const val WDFA_FRONTIER_ROOTS_V2 = """
fn wdfaStartIdx(i : u32) -> u32 { return idx_uni.startIndices[i * 2u]; }
fn wdfaEditDist(i : u32) -> u32 { return idx_uni.startIndices[i * 2u + 1u]; }
"""

//language=wgsl
const val WDFA_FRONTIER_STATE_V2 = """
const WDFA_META_STRIDE_WGSL : u32 = ${WDFA_META_STRIDE_V2}u;

fn wdfaIntsPerState() -> u32 { return WDFA_META_STRIDE_WGSL + 2u * prm.stride; }
fn wdfaStateBase(i: u32) -> u32 { return i * wdfaIntsPerState(); }
fn wdfaMetaBase(i: u32)  -> u32 { return wdfaStateBase(i); }
fn wdfaStackBase(i: u32) -> u32 { return wdfaStateBase(i) + WDFA_META_STRIDE_WGSL; }
fn wdfaWordBase(i: u32)  -> u32 { return wdfaStateBase(i) + WDFA_META_STRIDE_WGSL + prm.stride; }
"""

//language=wgsl
const val WDFA_FRONTIER_COMMON_V2 = """
const WDFA_STATUS_ACTIVE_WGSL : u32 = ${WDFA_STATUS_ACTIVE_V2}u;
const WDFA_STATUS_DONE_WGSL   : u32 = ${WDFA_STATUS_DONE_V2}u;
const WDFA_STATUS_DEAD_WGSL   : u32 = ${WDFA_STATUS_DEAD_V2}u;
const WDFA_PKT_HDR_LEN        : u32 = ${PKT_HDR_LEN}u;
const WDFA_EDIT_COST_STRIDE   : u32 = 10000000u;
const WDFA_INVALID_SCORE      : u32 = 0xffffffffu;

fn wdfa_xorshift32(x: ptr<function, u32>) -> u32 {
  var v = *x;
  v ^= v << 13u;
  v ^= v >> 17u;
  v ^= v << 5u;
  *x = v;
  return v;
}

fn wdfa_rand01(x: ptr<function, u32>) -> f32 {
  let r = wdfa_xorshift32(x);
  return f32(r) / 4294967296.0;
}
"""

//language=wgsl
const val WDFA_FRONTIER_WDFA_COMMON_V2 = """
fn wdfa_step(q: u32, tok: u32) -> vec2<u32> {
  let e = find_edge(q, tok);
  if (e == NO_EDGE) { return vec2<u32>(q, wdfa.missingCost); }
  return vec2<u32>(edge_dst(e), edge_cost(e));
}

fn wdfa_finalized_cost(q: u32, cost: u32) -> u32 {
  let fc = final_cost(q);
  if (fc >= WDFA_INF) { return WDFA_INF; }
  return sat_add_wdfa(cost, fc);
}

fn wdfa_rank_score(levDist: u32, cost: u32) -> u32 {
  if (cost >= WDFA_INF) { return WDFA_INVALID_SCORE; }
  return sat_add_wdfa(cost, (levDist + 1u) * WDFA_EDIT_COST_STRIDE);
}

fn wdfa_step_mass(stepCost: u32) -> f32 {
  if (stepCost >= WDFA_INF) { return 0.0; }
  let scale = f32(max(wdfa.scale, 1u));
  return exp(-f32(stepCost) / scale);
}
"""

//language=wgsl
const val WDFA_FRONTIER_ENUM_V2 = """
const WDFA_LIT_ALL : u32 = 0x7ffffffeu;
const WDFA_NEG_BIT : u32 = $NEG_STR_LIT;

fn wdfa_lex_count(val: u32, nt: u32) -> u32 { return count_tms(val, nt); }

fn wdfa_nth_lex_tok(val: u32, nt: u32, ord: u32) -> u32 {
  let ntLen = get_nt_tm_lens(nt);
  if (ntLen == 0u) { return 0u; }
  let ntOff = get_offsets(nt);

  if (val == WDFA_LIT_ALL) {
    if (ord >= ntLen) { return 0u; }
    return get_all_tms(ntOff + ord) + 1u;
  }

  if ((val >> 1u) == 0u) { return 0u; }

  let negLit = (val & WDFA_NEG_BIT) != 0u;
  let litEnc = (val >> 1u) & 0x1fffffffu;
  if (litEnc == 0u || litEnc > ntLen) { return 0u; }

  if (!negLit) {
    if (ord != 0u) { return 0u; }
    return get_all_tms(ntOff + (litEnc - 1u)) + 1u;
  }

  if (ntLen <= 1u || ord >= ntLen - 1u) { return 0u; }
  let excl = litEnc - 1u;
  let idx = select(ord, ord + 1u, ord >= excl);
  return get_all_tms(ntOff + idx) + 1u;
}

fn wdfa_bin_count(dpIdx: u32) -> u32 { return bp_count_at(dpIdx); }

fn wdfa_nth_bin_pair(dpIdx: u32, ord: u32) -> vec2<u32> {
  let ix = bp_offset_at(dpIdx) + ord;
  return vec2<u32>(
    bp_storage_at(2u * ix + 0u),
    bp_storage_at(2u * ix + 1u)
  );
}
"""

//language=wgsl
val wdfa_frontier_init_v2 by Shader("""
$IDX_UNIFORM_STRUCT
$WDFA_STRUCT
$WDFA_FRONTIER_STRUCT_V2
$WDFA_FRONTIER_ROOTS_V2
$WDFA_FRONTIER_STATE_V2
$WDFA_FRONTIER_COMMON_V2

@group(0) @binding(0) var<storage, read_write> frontier : array<u32>;
@group(0) @binding(1) var<storage, read_write> idx_uni  : IndexUniforms;
@group(0) @binding(2) var<uniform>             prm      : WdfaFrontierParams;
@group(0) @binding(3) var<storage, read>       wdfa     : WDFA;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= prm.frontierCount || prm.numRoots == 0u) { return; }

  var rng = pid * 747796405u + 2891336453u + atomicLoad(&idx_uni.targetCnt);
  var rootIx = pid;
  if (pid >= prm.numRoots) {
    rootIx = wdfa_xorshift32(&rng) % prm.numRoots;
  }

  let rootDp  = wdfaStartIdx(rootIx);
  let levDist = wdfaEditDist(rootIx);

  let mb = wdfaMetaBase(pid);
  let sb = wdfaStackBase(pid);
  let wb = wdfaWordBase(pid);

  frontier[mb + 0u] = rng;
  frontier[mb + 1u] = 1u;
  frontier[mb + 2u] = 0u;
  frontier[mb + 3u] = WDFA_STATUS_ACTIVE_WGSL;
  frontier[mb + 4u] = wdfa.startState;
  frontier[mb + 5u] = levDist;
  frontier[mb + 6u] = wdfa.startCost;

  frontier[sb + 0u] = rootDp;
  if (WDFA_PKT_HDR_LEN < prm.stride) { frontier[wb + WDFA_PKT_HDR_LEN] = 0u; }
}""".trimIndent())

//language=wgsl
val wdfa_frontier_count_succ_v2 by Shader("""
$TERM_STRUCT
$WDFA_BP_STRUCT_V2
$WDFA_FRONTIER_STRUCT_V2
$WDFA_FRONTIER_STATE_V2
$WDFA_FRONTIER_COMMON_V2
$WDFA_FRONTIER_ENUM_V2

@group(0) @binding(0) var<storage, read>       dp_in     : array<u32>;
@group(0) @binding(1) var<storage, read>       terminals : Terminals;
@group(0) @binding(2) var<storage, read>       bp        : WdfaBackpointers;
@group(0) @binding(3) var<storage, read>       frontier  : array<u32>;
@group(0) @binding(4) var<storage, read_write> succCount : array<u32>;
@group(0) @binding(5) var<storage, read_write> idx_uni   : IndexUniforms;
@group(0) @binding(6) var<uniform>             prm       : WdfaFrontierParams;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= prm.frontierCount) { return; }

  let mb = wdfaMetaBase(i);
  let status = frontier[mb + 3u];

  if (status == WDFA_STATUS_DONE_WGSL) { succCount[i] = 0u; return; }
  if (status != WDFA_STATUS_ACTIVE_WGSL) { succCount[i] = 0u; return; }

  let top = frontier[mb + 1u];
  if (top == 0u) { succCount[i] = 1u; return; }

  let wLen = frontier[mb + 2u];
  if (wLen + WDFA_PKT_HDR_LEN >= prm.stride) { succCount[i] = 0u; return; }
  if (top >= prm.stride) { succCount[i] = 0u; return; }

  let sb = wdfaStackBase(i);
  let d = frontier[sb + (top - 1u)];
  let val = dp_in[d];
  if (val == 0u) { succCount[i] = 0u; return; }

  let nt = d % idx_uni.numNonterminals;
  let lc = wdfa_lex_count(val, nt);
  let bc = wdfa_bin_count(d);

  succCount[i] = lc + bc;
}""".trimIndent())

//language=wgsl
val wdfa_frontier_write_exact_v2 by Shader("""
$TERM_STRUCT
$WDFA_STRUCT
$WDFA_BP_STRUCT_V2
$WDFA_FRONTIER_STRUCT_V2
$WDFA_FRONTIER_STATE_V2
$WDFA_FRONTIER_COMMON_V2
$WDFA_FRONTIER_WDFA_COMMON_V2
$WDFA_FRONTIER_ENUM_V2

@group(0) @binding(0) var<storage, read>        dp_in       : array<u32>;
@group(0) @binding(1) var<storage, read>        terminals   : Terminals;
@group(0) @binding(2) var<storage, read>        wdfa        : WDFA;
@group(0) @binding(3) var<storage, read>        bp          : WdfaBackpointers;
@group(0) @binding(4) var<storage, read>        frontierIn  : array<u32>;
@group(0) @binding(5) var<storage, read>        succOff     : array<u32>;
@group(0) @binding(6) var<storage, read>        succCount   : array<u32>;
@group(0) @binding(7) var<storage, read_write>  frontierOut : array<u32>;
@group(0) @binding(8) var<storage, read_write>  idx_uni     : IndexUniforms;
@group(0) @binding(9) var<uniform>              prm         : WdfaFrontierParams;

fn wdfa_copy_state(src: u32, dst: u32) {
  let sS = wdfaStateBase(src);
  let sD = wdfaStateBase(dst);
  let len = wdfaIntsPerState();
  for (var k: u32 = 0u; k < len; k = k + 1u) {
    frontierOut[sD + k] = frontierIn[sS + k];
  }
}

fn wdfa_mark_done(dst: u32) {
  let mb = wdfaMetaBase(dst);
  let q = frontierOut[mb + 4u];
  let cost = frontierOut[mb + 6u];
  frontierOut[mb + 3u] = WDFA_STATUS_DONE_WGSL;
  frontierOut[mb + 6u] = wdfa_finalized_cost(q, cost);
}

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let src = gid.x;
  if (src >= prm.frontierCount) { return; }

  let cnt = succCount[src];
  if (cnt == 0u) { return; }

  let base = succOff[src];
  let mbS = wdfaMetaBase(src);
  let status = frontierIn[mbS + 3u];

  if (status == WDFA_STATUS_DONE_WGSL) {
    wdfa_copy_state(src, base);
    return;
  }

  if (status != WDFA_STATUS_ACTIVE_WGSL) { return; }

  let top = frontierIn[mbS + 1u];
  if (top == 0u) {
    wdfa_copy_state(src, base);
    wdfa_mark_done(base);
    return;
  }

  let sbS  = wdfaStackBase(src);
  let wLen = frontierIn[mbS + 2u];
  let q    = frontierIn[mbS + 4u];
  let cost = frontierIn[mbS + 6u];

  let d   = frontierIn[sbS + (top - 1u)];
  let val = dp_in[d];
  let nt  = d % idx_uni.numNonterminals;

  let lc = wdfa_lex_count(val, nt);
  let bc = wdfa_bin_count(d);

  for (var j: u32 = 0u; j < lc; j = j + 1u) {
    let dst = base + j;
    wdfa_copy_state(src, dst);

    let mbD = wdfaMetaBase(dst);
    let wbD = wdfaWordBase(dst);

    let tok = wdfa_nth_lex_tok(val, nt, j);
    if (tok == 0u || wLen + WDFA_PKT_HDR_LEN >= prm.stride) {
      frontierOut[mbD + 3u] = WDFA_STATUS_DEAD_WGSL;
      frontierOut[mbD + 6u] = WDFA_INF;
      continue;
    }

    let st = wdfa_step(q, tok);
    let nextQ = st.x;
    var nextCost = sat_add_wdfa(cost, st.y);
    let nextTop = top - 1u;

    frontierOut[wbD + WDFA_PKT_HDR_LEN + wLen] = tok;
    if (WDFA_PKT_HDR_LEN + wLen + 1u < prm.stride) {
      frontierOut[wbD + WDFA_PKT_HDR_LEN + wLen + 1u] = 0u;
    }

    frontierOut[mbD + 1u] = nextTop;
    frontierOut[mbD + 2u] = wLen + 1u;
    frontierOut[mbD + 4u] = nextQ;
    frontierOut[mbD + 6u] = nextCost;

    if (nextTop == 0u) {
      frontierOut[mbD + 3u] = WDFA_STATUS_DONE_WGSL;
      frontierOut[mbD + 6u] = wdfa_finalized_cost(nextQ, nextCost);
    }
  }

  for (var j: u32 = 0u; j < bc; j = j + 1u) {
    let dst = base + lc + j;
    wdfa_copy_state(src, dst);

    let mbD = wdfaMetaBase(dst);
    let sbD = wdfaStackBase(dst);

    if (top >= prm.stride) {
      frontierOut[mbD + 3u] = WDFA_STATUS_DEAD_WGSL;
      frontierOut[mbD + 6u] = WDFA_INF;
      continue;
    }

    let pr = wdfa_nth_bin_pair(d, j);
    frontierOut[sbD + (top - 1u)] = pr.y;
    frontierOut[sbD + top]        = pr.x;
    frontierOut[mbD + 1u] = top + 1u;
  }
}""".trimIndent())

//language=wgsl
val wdfa_frontier_parent_weights_v2 by Shader("""
$WDFA_STRUCT
$WDFA_FRONTIER_STRUCT_V2
$WDFA_FRONTIER_STATE_V2
$WDFA_FRONTIER_COMMON_V2

@group(0) @binding(0) var<storage, read>       frontier : array<u32>;
@group(0) @binding(1) var<storage, read>       succCnt  : array<u32>;
@group(0) @binding(2) var<storage, read_write> weights  : array<u32>;
@group(0) @binding(3) var<storage, read>       wdfa     : WDFA;
@group(0) @binding(4) var<uniform>             prm      : WdfaFrontierParams;

fn wdfa_resample_weight(cost: u32) -> u32 {
  if (cost >= WDFA_INVALID_SCORE) { return 0u; }
  let scaled = min(cost, ${Int.MAX_VALUE}u) / max(wdfa.scale, 1u);
  let denom = 1u + scaled;
  return max(1u, ${WDFA_RESAMPLE_SCALE_V2}u / denom);
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= prm.frontierCount) { return; }

  let mb = wdfaMetaBase(i);
  let status = frontier[mb + 3u];
  if (status == WDFA_STATUS_DEAD_WGSL) {
    weights[i] = 0u;
    return;
  }
  if (status == WDFA_STATUS_DONE_WGSL) {
    weights[i] = 0u;
    return;
  }

  let branch = min(max(succCnt[i], 1u), 64u);
  weights[i] = min(65535u, wdfa_resample_weight(frontier[mb + 6u]) * branch);
}""".trimIndent())

//language=wgsl
val wdfa_frontier_sampled_step_v2 by Shader("""
$TERM_STRUCT
$WDFA_STRUCT
$WDFA_BP_STRUCT_V2
$WDFA_FRONTIER_STRUCT_V2
$WDFA_FRONTIER_STATE_V2
$WDFA_FRONTIER_COMMON_V2
$WDFA_FRONTIER_WDFA_COMMON_V2
$WDFA_FRONTIER_ENUM_V2

@group(0) @binding(0) var<storage, read>        dp_in       : array<u32>;
@group(0) @binding(1) var<storage, read>        terminals   : Terminals;
@group(0) @binding(2) var<storage, read>        wdfa        : WDFA;
@group(0) @binding(3) var<storage, read>        bp          : WdfaBackpointers;
@group(0) @binding(4) var<storage, read>        frontierIn  : array<u32>;
@group(0) @binding(5) var<storage, read>        parentCDF   : array<u32>;
@group(0) @binding(6) var<storage, read>        parentW     : array<u32>;
@group(0) @binding(7) var<storage, read_write>  frontierOut : array<u32>;
@group(0) @binding(8) var<storage, read_write>  idx_uni     : IndexUniforms;
@group(0) @binding(9) var<uniform>              prm         : WdfaFrontierParams;

fn wdfa_parent_total() -> u32 {
  if (prm.frontierCount == 0u) { return 0u; }
  let last = prm.frontierCount - 1u;
  return parentCDF[last] + parentW[last];
}

fn wdfa_choose_parent(rng: ptr<function, u32>) -> u32 {
  let tot = wdfa_parent_total();
  if (tot == 0u) { return 0u; }
  let needle = wdfa_xorshift32(rng) % tot;

  var lo: u32 = 0u;
  var hi: u32 = prm.frontierCount;
  while (lo + 1u < hi) {
    let mid = (lo + hi) >> 1u;
    let base = parentCDF[mid];
    let end = base + parentW[mid];
    if (needle < base) {
      hi = mid;
    } else if (needle >= end) {
      lo = mid + 1u;
    } else {
      return mid;
    }
  }

  return min(lo, prm.frontierCount - 1u);
}

fn wdfa_copy_state(src: u32, dst: u32) {
  let sS = wdfaStateBase(src);
  let sD = wdfaStateBase(dst);
  let len = wdfaIntsPerState();
  for (var k: u32 = 0u; k < len; k = k + 1u) {
    frontierOut[sD + k] = frontierIn[sS + k];
  }
}

fn wdfa_emit_token(dst: u32, tok: u32, rng: u32) {
  let mb = wdfaMetaBase(dst);
  let wb = wdfaWordBase(dst);

  let top = frontierOut[mb + 1u];
  let wLen = frontierOut[mb + 2u];
  let q = frontierOut[mb + 4u];
  let cost = frontierOut[mb + 6u];

  if (tok == 0u || wLen + WDFA_PKT_HDR_LEN >= prm.stride || top == 0u) {
    frontierOut[mb + 0u] = rng;
    frontierOut[mb + 3u] = WDFA_STATUS_DEAD_WGSL;
    frontierOut[mb + 6u] = WDFA_INF;
    return;
  }

  let st = wdfa_step(q, tok);
  let nextQ = st.x;
  let nextCost = sat_add_wdfa(cost, st.y);
  let nextTop = top - 1u;

  frontierOut[wb + WDFA_PKT_HDR_LEN + wLen] = tok;
  if (WDFA_PKT_HDR_LEN + wLen + 1u < prm.stride) {
    frontierOut[wb + WDFA_PKT_HDR_LEN + wLen + 1u] = 0u;
  }

  frontierOut[mb + 0u] = rng;
  frontierOut[mb + 1u] = nextTop;
  frontierOut[mb + 2u] = wLen + 1u;
  frontierOut[mb + 4u] = nextQ;
  frontierOut[mb + 6u] = nextCost;

  if (nextTop == 0u) {
    frontierOut[mb + 3u] = WDFA_STATUS_DONE_WGSL;
    frontierOut[mb + 6u] = wdfa_finalized_cost(nextQ, nextCost);
  }
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dst = gid.x;
  if (dst >= prm.frontierCap) { return; }

  var rng = (dst * 1664525u + 1013904223u + atomicLoad(&idx_uni.targetCnt)) ^ (prm.step * 0x9e3779b9u);
  let src = wdfa_choose_parent(&rng);
  wdfa_copy_state(src, dst);

  let mbD = wdfaMetaBase(dst);
  let sbD = wdfaStackBase(dst);

  let status = frontierOut[mbD + 3u];
  if (status == WDFA_STATUS_DONE_WGSL || status == WDFA_STATUS_DEAD_WGSL) {
    frontierOut[mbD + 0u] = rng;
    return;
  }

  let top = frontierOut[mbD + 1u];
  if (top == 0u) {
    frontierOut[mbD + 0u] = rng;
    frontierOut[mbD + 3u] = WDFA_STATUS_DONE_WGSL;
    frontierOut[mbD + 6u] = wdfa_finalized_cost(frontierOut[mbD + 4u], frontierOut[mbD + 6u]);
    return;
  }

  let wLen = frontierOut[mbD + 2u];
  if (wLen + WDFA_PKT_HDR_LEN >= prm.stride || top >= prm.stride) {
    frontierOut[mbD + 0u] = rng;
    frontierOut[mbD + 3u] = WDFA_STATUS_DEAD_WGSL;
    frontierOut[mbD + 6u] = WDFA_INF;
    return;
  }

  let d = frontierOut[sbD + (top - 1u)];
  let val = dp_in[d];
  if (val == 0u) {
    frontierOut[mbD + 0u] = rng;
    frontierOut[mbD + 3u] = WDFA_STATUS_DEAD_WGSL;
    frontierOut[mbD + 6u] = WDFA_INF;
    return;
  }

  let nt = d % idx_uni.numNonterminals;
  let lc = wdfa_lex_count(val, nt);
  let bc = wdfa_bin_count(d);

  if (lc + bc == 0u) {
    frontierOut[mbD + 0u] = rng;
    frontierOut[mbD + 3u] = WDFA_STATUS_DEAD_WGSL;
    frontierOut[mbD + 6u] = WDFA_INF;
    return;
  }

  let q = frontierOut[mbD + 4u];
  var lexMass = 0.0;
  for (var j: u32 = 0u; j < lc; j = j + 1u) {
    let tok = wdfa_nth_lex_tok(val, nt, j);
    if (tok != 0u) {
      let st = wdfa_step(q, tok);
      lexMass = lexMass + wdfa_step_mass(st.y);
    }
  }

  let binMass = f32(bc);
  let total = lexMass + binMass;
  if (total <= 0.0) {
    if (lc != 0u) {
      let ord = wdfa_xorshift32(&rng) % lc;
      wdfa_emit_token(dst, wdfa_nth_lex_tok(val, nt, ord), rng);
      return;
    }
    frontierOut[mbD + 0u] = rng;
    frontierOut[mbD + 3u] = WDFA_STATUS_DEAD_WGSL;
    frontierOut[mbD + 6u] = WDFA_INF;
    return;
  }

  let u = wdfa_rand01(&rng) * total;
  var acc = 0.0;

  for (var j: u32 = 0u; j < lc; j = j + 1u) {
    let tok = wdfa_nth_lex_tok(val, nt, j);
    if (tok == 0u) { continue; }
    let st = wdfa_step(q, tok);
    acc = acc + wdfa_step_mass(st.y);
    if (u <= acc) {
      wdfa_emit_token(dst, tok, rng);
      return;
    }
  }

  if (bc != 0u) {
    let ord = wdfa_xorshift32(&rng) % bc;
    let pr = wdfa_nth_bin_pair(d, ord);
    frontierOut[sbD + (top - 1u)] = pr.y;
    frontierOut[sbD + top] = pr.x;
    frontierOut[mbD + 0u] = rng;
    frontierOut[mbD + 1u] = top + 1u;
    return;
  }

  frontierOut[mbD + 0u] = rng;
  frontierOut[mbD + 3u] = WDFA_STATUS_DEAD_WGSL;
  frontierOut[mbD + 6u] = WDFA_INF;
}""".trimIndent())

//language=wgsl
val wdfa_frontier_emit_done_packets_v2 by Shader("""
$WDFA_FRONTIER_STRUCT_V2
$WDFA_FRONTIER_STATE_V2
$WDFA_FRONTIER_COMMON_V2

@group(0) @binding(0) var<storage, read_write> frontier : array<u32>;
@group(0) @binding(1) var<storage, read_write> outPk    : array<u32>;
@group(0) @binding(2) var<storage, read_write> outCnt   : array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             prm      : WdfaFrontierParams;

const WDFA_EMIT_INF : u32 = 0x3fffffffu;

fn wdfa_emit_sat_add(a: u32, b: u32) -> u32 {
  let c = a + b;
  if (c < a || c > WDFA_EMIT_INF) { return WDFA_EMIT_INF; }
  return c;
}

fn wdfa_emit_rank_score(levDist: u32, cost: u32) -> u32 {
  if (cost >= WDFA_EMIT_INF) { return WDFA_INVALID_SCORE; }
  return wdfa_emit_sat_add(cost, (levDist + 1u) * WDFA_EDIT_COST_STRIDE);
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= prm.frontierCount) { return; }

  let mb = wdfaMetaBase(pid);
  if (frontier[mb + 3u] != WDFA_STATUS_DONE_WGSL) { return; }

  let slot = atomicAdd(&outCnt[0], 1u);
  frontier[mb + 3u] = WDFA_STATUS_DEAD_WGSL;
  if (slot >= prm.frontierCap) { return; }

  let wb = wdfaWordBase(pid);
  let ob = slot * prm.stride;
  let levDist = frontier[mb + 5u];
  let cost = frontier[mb + 6u];
  let wLen = frontier[mb + 2u];

  outPk[ob + 0u] = levDist;
  outPk[ob + 1u] = wdfa_emit_rank_score(levDist, cost);
  for (var i: u32 = 0u; i < wLen && (WDFA_PKT_HDR_LEN + i) < prm.stride; i = i + 1u) {
    outPk[ob + WDFA_PKT_HDR_LEN + i] = frontier[wb + WDFA_PKT_HDR_LEN + i];
  }
  if (WDFA_PKT_HDR_LEN + wLen < prm.stride) { outPk[ob + WDFA_PKT_HDR_LEN + wLen] = 0u; }
}""".trimIndent())

//language=wgsl
val wdfa_frontier_pack_packets_v2 by Shader("""
$WDFA_FRONTIER_STRUCT_V2
$WDFA_FRONTIER_STATE_V2
$WDFA_FRONTIER_COMMON_V2

@group(0) @binding(0) var<storage, read>       frontier : array<u32>;
@group(0) @binding(1) var<storage, read_write> outPk    : array<u32>;
@group(0) @binding(2) var<uniform>             prm      : WdfaFrontierParams;

const WDFA_PACK_INF : u32 = 0x3fffffffu;

fn wdfa_pack_sat_add(a: u32, b: u32) -> u32 {
  let c = a + b;
  if (c < a || c > WDFA_PACK_INF) { return WDFA_PACK_INF; }
  return c;
}

fn wdfa_pack_rank_score(levDist: u32, cost: u32) -> u32 {
  if (cost >= WDFA_PACK_INF) { return WDFA_INVALID_SCORE; }
  return wdfa_pack_sat_add(cost, (levDist + 1u) * WDFA_EDIT_COST_STRIDE);
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= prm.frontierCount) { return; }

  let mb = wdfaMetaBase(pid);
  let wb = wdfaWordBase(pid);
  let ob = pid * prm.stride;

  let status = frontier[mb + 3u];
  let levDist = frontier[mb + 5u];
  let cost = frontier[mb + 6u];
  let wLen = frontier[mb + 2u];

  outPk[ob + 0u] = levDist;

  if (status == WDFA_STATUS_DONE_WGSL) {
    outPk[ob + 1u] = wdfa_pack_rank_score(levDist, cost);
    for (var i: u32 = 0u; i < wLen && (WDFA_PKT_HDR_LEN + i) < prm.stride; i = i + 1u) {
      outPk[ob + WDFA_PKT_HDR_LEN + i] = frontier[wb + WDFA_PKT_HDR_LEN + i];
    }
    if (WDFA_PKT_HDR_LEN + wLen < prm.stride) { outPk[ob + WDFA_PKT_HDR_LEN + wLen] = 0u; }
  } else {
    outPk[ob + 1u] = WDFA_INVALID_SCORE;
    if (WDFA_PKT_HDR_LEN < prm.stride) { outPk[ob + WDFA_PKT_HDR_LEN] = 0u; }
  }
}""".trimIndent())

//language=wgsl
val select_top_k_unique_v2 by Shader("""
struct UniqueParams {
  maxSamples : u32,
  k          : u32,
  stride     : u32,
  threads    : u32,
  epsilonTok : u32,
  reserved0  : u32,
  reserved1  : u32,
  reserved2  : u32,
};

@group(0) @binding(0) var<uniform>                  prm : UniqueParams;
@group(0) @binding(1) var<storage, read>        packets : array<u32>;
@group(0) @binding(2) var<storage, read_write>   topIdx : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> topScore : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write>  topHash : array<atomic<u32>>;

const UNIQUE_TOPK_EMPTY_SCORE : u32 = 0x7fffffffu;
const UNIQUE_PKT_HDR_LEN : u32 = ${PKT_HDR_LEN}u;

fn packet_hash(i: u32) -> u32 {
    let base = i * prm.stride + UNIQUE_PKT_HDR_LEN;
    var h : u32 = 2166136261u;
    var pos : u32 = 0u;

    loop {
        if (UNIQUE_PKT_HDR_LEN + pos >= prm.stride) { break; }
        let tok = packets[base + pos];
        if (tok == 0u) { break; }
        if (tok != prm.epsilonTok) {
            h = (h ^ tok) * 16777619u;
        }
        pos = pos + 1u;
    }

    h = h ^ (h >> 16u);
    if (h == 0u) { return 1u; }
    return h;
}

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id)        workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_id     : vec3<u32>
) {
    let workgroup_linear_id = workgroup_id.x + workgroup_id.y * prm.threads;
    let i = workgroup_linear_id * 256u + local_id.x;
    if (i >= prm.maxSamples || prm.k == 0u) { return; }

    let score : u32 = packets[i * prm.stride + 1u];
    if (score >= UNIQUE_TOPK_EMPTY_SCORE) { return; }
    let hash = packet_hash(i);

    loop {
        for (var j : u32 = 0u; j < prm.k; j = j + 1u) {
            if (atomicLoad(&topHash[j]) == hash) {
                let oldScore = atomicLoad(&topScore[j]);
                if (oldScore <= score) { return; }
                let old = atomicCompareExchangeWeak(&topScore[j], oldScore, score);
                if (old.exchanged) {
                    atomicStore(&topIdx[j], i);
                    return;
                }
            }
        }

        var worstPos : u32 = 0u;
        var worstVal : u32 = atomicLoad(&topScore[0]);
        for (var j : u32 = 1u; j < prm.k; j = j + 1u) {
            let v = atomicLoad(&topScore[j]);
            if (v > worstVal) { worstVal = v; worstPos = j; }
        }

        if (score > worstVal) { return; }
        let old = atomicCompareExchangeWeak(&topScore[worstPos], worstVal, score);
        if (old.exchanged) {
            atomicStore(&topIdx[worstPos], i);
            atomicStore(&topHash[worstPos], hash);
            return;
        }
    }
}""")
