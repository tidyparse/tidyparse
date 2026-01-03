@file:OptIn(ExperimentalUnsignedTypes::class, ExperimentalStdlibApi::class)

import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.buildLanguageSizeBuf
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.readInts
import Shader.Companion.toGPUBuffer
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.types.cache
import ai.hypergraph.tidyparse.MAX_DISP_RESULTS
import js.array.asList
import js.buffer.*
import js.typedarrays.Int32Array
import kotlinx.browser.document
import kotlinx.coroutines.await
import org.w3c.dom.HTMLDivElement
import web.events.*
import web.gpu.*
import kotlin.js.Promise
import kotlin.math.*
import kotlin.reflect.KProperty
import kotlin.time.TimeSource
import kotlin.js.Promise as KPromise


lateinit var gpu: GPUDevice
var gpuAvailable = false
external val navigator: dynamic

/*
  TODO:
    (1) Minimize repairs on GPU
    (2) Use more sensible midpoint filter based on dense adjacency / reachability matrix
    (3) Implement test harness to avoid regressions. Need to measure: (1) perf (2) prec@k
*/

const val largeMem = "{ requiredLimits: { maxBufferSize: 2000000000, maxStorageBufferBindingSize: 2000000000, maxStorageBuffersPerShaderStage: 10 } }"
const val smallMem = "{ requiredLimits: { maxBufferSize: 1073741824, maxStorageBufferBindingSize: 1073741824, maxStorageBuffersPerShaderStage: 10 } }"

suspend fun tryBootstrappingGPU(needsExtraMemory: Boolean = false) {
  val tmpDev = (navigator.gpu as? GPU)?.requestAdapter()?.also {
    gpu = if (needsExtraMemory) it.requestDevice(js(largeMem))
          else it.requestDevice(js(smallMem))
  }

  if (tmpDev != null) {
    gpu.addEventListener(EventType("uncapturederror"), { e: dynamic -> log("Uncaptured: ${e.error.message}") })
    try {
      listOf(
        prefix_sum_p1, prefix_sum_p2,      // ADT storage utils
        sparse_load, sparse_mat_load,      // Matrix loading utils

        init_chart,
        dag_reach, mdpt_count, mdpt_write,       // Graph reachability
        cfl_mul_upper,                           // Matrix exponentiation
        bp_count, bp_write,                      // Backpointer addressing
        ls_dense, ls_cdf,                        // Language size estimation
        build_root_sizes, enum_words_wor,        // Enumeration and decoding
        markov_score, select_top_k, gather_top_k,// Reranking and truncation

//        init_particles_v2,
//        decode_step_v2,
//        pack_particles_v2
      ).forEach { it.bind() }
//      benchmarkWGPU() // TODO: remove for deployment
//      benchmarkWGPURepair()
//      benchmarkReach()
    } catch (e: Exception) { e.printStackTrace(); return }

    log("Bootstrapping GPU successful!")
    gpuAvailable = true

    (document.getElementById("gpuAvail") as? HTMLDivElement)?.appendChild(
    document.createElement("object").apply {
      setAttribute("type", "image/svg+xml")
      setAttribute("data", "/webgpu.svg")
      setAttribute("width", "35")
      setAttribute("height", "35")
    })
  } else print("GPU not detected.")
}


suspend fun repairCode(cfg: CFG, code: List<String>, ledBuffer: Int = Int.MAX_VALUE, ngrams: GPUBuffer? = null): List<String> {
  val t0 = TimeSource.Monotonic.markNow()
  val fsa: FSA = makeLevFSA(code, MAX_LEV_RAD)
  log("Made levFSA in ${t0.elapsedNow()}")

  // TODO: maybe modify the LevFSA to accommodate unknown tokens?
  val codePoints = IntArray(code.size) { cfg.tmMap[code[it]] ?: 0 }

// This is interchangeable with init_chart for Lev automata
//  val dpInSparse = fsa.byteFormat(cfg).toGPUBuffer()
//  log("Initial nonzeros: ${dpIn.count { it != 0 }}")

  log("PREPROCESSING TOOK: ${t0.elapsedNow()}") // ~230ms
//  val words = repairPipelineV2(cfg, fsa, ledBuffer, ngrams, codePoints)
  val words = repairPipeline(cfg, fsa, ledBuffer, ngrams, codePoints)
//  val distinctWords = words.distinct()
//  log("Distinct: ${distinctWords.size} words")

  return words.also { log("Received: ${words.size} words in ${t0.elapsedNow()} (round trip)") }
}

suspend fun repairPipeline(cfg: CFG, fsa: FSA,
//                           dpInSparse: IntArray,
                           ledBuffer: Int, ngrams: GPUBuffer?, codePoints: IntArray): List<String> {
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("FSA(|Q|=${numStates}, |δ|=${fsa.transit.size}), ${cfg.calcStats()}")

//log("Time to load buffer: ${t0.elapsedNow()} (${input.size * 4} bytes)")

  val metaBuf = packMetadata(cfg, fsa)

  val tmBuf     = cfg.termBuf
  val wordBuf   = codePoints.toGPUBuffer()
  val totalSize = numStates * numStates * numNTs
  val dpBuf     = Shader.createParseChart(STCPSD, totalSize)
  init_chart(dpBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)

  log("Chart construction took: ${t0.elapsedNow()}")

//   log(dpBuf.readInts().toLaTeX(numStates, numNTs))

// val rowCoeff = numStates * numNTs
//  val colCoeff = numNTs
//  val dpBuf = dpInSparse.toGPUBufferSparse(GPUBufferUsage.STCPSD, numStates * rowCoeff, rowCoeff, colCoeff)

  cfl_mul_upper.invokeCFLFixpoint(numStates, numNTs, dpBuf, metaBuf)
  log("Matrix closure reached in: ${t0.elapsedNow()}")

//  dpBuf.readInts().filter { it != 0 }.map { it.toString(2) }
//    .groupingBy { it }.eachCount().entries.sortedBy { it.key }.joinToString("\n") { (a, b) -> "$a => $b" }
//    .also { log(it) }

  val t1 = TimeSource.Monotonic.markNow()
  val startNT     = cfg.bindex[START_SYMBOL]
  val allStartIds = fsa.finalIdxs.map { it * numNTs + startNT }
    .let { it.zip(dpBuf.readIndices(it)) }.filter { (_, v) -> v != 0 }.map { it.first }

  if (!allStartIds.isEmpty()) { log("Valid parse found: dpComplete has ${allStartIds.size} start indices") }
  else return emptyList<String>().also { log("No valid parse found: dpComplete has no entries in final states!") }

  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNTs, dpBuf, metaBuf)
  log("Built backpointers in ${t1.elapsedNow()}")

  val t2 = TimeSource.Monotonic.markNow()
  val statesToDist = allStartIds.map { it to fsa.idsToCoords[(it - startNT) / numNTs]!!.second }
  val led = statesToDist.minOf { it.second } // Language edit distance

  val startIdxs = statesToDist.filter { it.second in (led..(led + ledBuffer)) }
    .map { listOf(it.first, it.second) }.sortedBy { it[1] }.also { log("Start indices (LED=$led): $it") }.flatten()

  val maxRepairLen = fsa.width + fsa.height + 10

  if (MAX_WORD_LEN < maxRepairLen) return emptyList<String>()
    .also { log("Max repair length exceeded $MAX_WORD_LEN ($maxRepairLen)") }

  val lsDense  = buildLanguageSizeBuf(numStates, numNTs, dpBuf, metaBuf, tmBuf)
  val totalExp = bpStorageBuf.size.toInt() / (2 * 4)
//  log("Total expansions: $totalExp")
  val cdfBuf   = GPUBuffer(totalExp * 4, STCPSD)

  ls_cdf(dpBuf, lsDense, bpOffsetBuf, cdfBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)

  lsDense.destroy()

  log("Pairing function construction took: ${t2.elapsedNow()}")

  val numRoots = startIdxs.size / 2
  val rootSizes = GPUBuffer(numRoots * 4, STCPSD)

  /** Memory layout: [IDX_UNIFORM_STRUCT] */
  val idxUniBuf = packStruct(listOf(0, maxRepairLen, numNTs, numStates, DISPATCH_GROUP_SIZE_X, MAX_SAMPLES), startIdxs.toGPUBuffer())

  build_root_sizes(dpBuf, bpCountBuf, bpOffsetBuf, cdfBuf, tmBuf, rootSizes, idxUniBuf)((numRoots + 255) / 256)

  val rootCDF = Shader.prefixSumGPU(rootSizes, numRoots)

  val outBuf = GPUBuffer(MAX_SAMPLES * maxRepairLen * 4, STCPSD)
  enum_words_wor(
    dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf,
    cdfBuf /*ls_sparse*/, tmBuf, idxUniBuf, rootSizes, rootCDF, outBuf
  )(DISPATCH_GROUP_SIZE_X, (MAX_SAMPLES + DISPATCH_GROUP_SIZE_X - 1) / DISPATCH_GROUP_SIZE_X)

  return (if (ngrams != null) ngramDecoder(outBuf, ngrams, maxRepairLen, cfg)
  else uniformDeocder(outBuf, rootSizes, rootCDF, numRoots, cfg, maxRepairLen)).also {
    listOf(outBuf, rootSizes, rootCDF, metaBuf, dpBuf, idxUniBuf, cdfBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf)
    .forEach(GPUBuffer::destroy)
  }
}

suspend fun uniformDeocder(outBuf: GPUBuffer, rootSizes: GPUBuffer, rootCDF: GPUBuffer, numRoots: Int, cfg: CFG, maxRepairLen: Int): List<String> {
  val t3 = TimeSource.Monotonic.markNow()
  val res = mutableMapOf<Int, MutableSet<String>>()
  val allResults = outBuf.readInts()
//  log("sample_words_wor invocation took: ${t3.elapsedNow()}")
  val rootSizesI32 = rootSizes.readInts()
  val rootCDFI32 = rootCDF.readInts()
  fun u(x: Int) = x.toUInt().toLong()
  val last = numRoots - 1
  val total = if (numRoots > 0) u(rootCDFI32[last]) + u(rootSizesI32[last]) else 0L
  val decodeN = minOf(MAX_SAMPLES.toLong(), total).toInt()

  for (i in 0 until decodeN) {
    val t = allResults.decodePacket(i, cfg.tmLst, maxRepairLen) ?: continue
    res.getOrPut(t.first) { mutableSetOf() }.add(t.second)
  }
  res.forEach { log("Δ=${it.key} -> |L|=${it.value.size}") }
  log("Sampled WOR into ${outBuf.size}-byte buffer in ${t3.elapsedNow()}")
  return res.map { it.value.toList() }.flatten()
}

suspend fun ngramDecoder(outBuf: GPUBuffer, ngrams: GPUBuffer, maxRepairLen: Int, cfg: CFG): MutableList<String> {
  val k = 10 * MAX_DISP_RESULTS
  val topK = scoreSelectGather(packets = outBuf, ngrams = ngrams, maxSamples = MAX_SAMPLES, stride = maxRepairLen, k = k)

  val t4 = TimeSource.Monotonic.markNow()
  val result = mutableListOf<String>()
  for (i in 0 until k) result.add(topK.decodePacket(i, cfg.tmLst, maxRepairLen)?.second ?: continue)
  log("Decoded ${result.distinct().size} unique words out of ${result.size} in ${t4.elapsedNow()}")
  return result
}

// Returns Levenshtein distance and string repair
fun IntArray.decodePacket(idx: Int, tm: List<String>, pktLen: Int): Pair<Int, String>? {
  var cur: StringBuilder? = null
  val base = idx * pktLen + PKT_HDR_LEN // skip header cells

  for (j in 0 until pktLen - PKT_HDR_LEN) {
    val tok = this[base + j] and 0xFF
    if (tok != 0) {
      if (cur == null) cur = StringBuilder()
      if (cur.isNotEmpty()) cur.append(' ')
      cur.append(tm[tok - 1])
    } else break
  }

  return if (!cur.isNullOrEmpty()) this[base - PKT_HDR_LEN] to cur.toString() else null
}

suspend fun scoreSelectGather(
  packets          : GPUBuffer,
  ngrams           : GPUBuffer,
  maxSamples       : Int,
  stride           : Int,
  k                : Int
): IntArray {
  val t0 = TimeSource.Monotonic.markNow()
  val threads = DISPATCH_GROUP_SIZE_X
  val groupsY = (maxSamples + threads - 1) / threads
  /** Memory layout: [SAMPLER_PARAMS] */
  val prmBuf  = intArrayOf(maxSamples, k, stride, threads).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  markov_score(packets, ngrams, prmBuf)(threads, groupsY)
//  log("Score in ${t0.elapsedNow()}")

//  log(packets.readInts().toList().windowed(stride, stride)
//    .map { it[1] }.groupingBy { it }.eachCount().entries
//    .sortedBy { it.key }.joinToString("\n") { (a, b) -> "$a => $b" })

//  t0 = TimeSource.Monotonic.markNow()
  val totalGroups = (maxSamples + 255) / 256
  val selGroupsY  = (totalGroups + threads - 1) / threads
  val idxBuf      = IntArray(k) { Int.MAX_VALUE }.toGPUBuffer(STCPSD)
  val scrBuf      = IntArray(k) { Int.MAX_VALUE }.toGPUBuffer(STCPSD)

  select_top_k(prmBuf, packets, idxBuf, scrBuf)(threads, selGroupsY)
//  log("Select in ${t0.elapsedNow()}")

//  t0 = TimeSource.Monotonic.markNow()
  val bestBuf   = GPUBuffer(k * stride * 4, STCPSD)

  gather_top_k(prmBuf, packets, idxBuf, bestBuf)(k)
//  log("Gather in ${t0.elapsedNow()}")

//  t0 = TimeSource.Monotonic.markNow()
  val topK = bestBuf.readInts()
  log("Score/select/gather read ${topK.size} = ${k}x${stride}x4 bytes in ${t0.elapsedNow()}")

  listOf(prmBuf, idxBuf, scrBuf, bestBuf).forEach(GPUBuffer::destroy)
  return topK
}

// Maps NTs to terminals for sampling
val CFG.termBuf: GPUBuffer by cache {
//  val packTime = TimeSource.Monotonic.markNow()
  val terminalLists = nonterminals.map { bimap.UNITS[it]?.map { tmMap[it]!! } ?: emptyList() }
  val nt_tm_lens = terminalLists.map { it.size }.toGPUBuffer()
  val nt_tm_offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1).toGPUBuffer()
  val all_tm = terminalLists.flatten().toGPUBuffer()

  /** Memory layout: [TERM_STRUCT] */ packStruct(emptyList(), nt_tm_lens, nt_tm_offsets, all_tm)
//    .also { log("Packing time: ${packTime.elapsedNow()}") }
}

//language=wgsl
const val CHART_DECODING_HELPERS = """
fn getStartIdx(i : u32) -> u32 { return idx_uni.startIndices[i * 2]; }
fn getEditDist(i : u32) -> u32 { return idx_uni.startIndices[i * 2 + 1]; }
"""

//language=wgsl
const val TM_DECODING_HELPERS = """
// Let Σ_A denote the subset of Σ s.t. for all a ∈ Σ_A ⊢ (A -> a) ∈ P
fn get_nt_tm_lens(nt : u32) -> u32 { return terminals.payload[terminals.nt_tm_lens_offset + nt]; } // |Σ_A|
// Offsets of the nonterminal in the following Map<...> structure
fn get_offsets(nt : u32) -> u32    { return terminals.payload[terminals.offsets_offset + nt];    } // offset of Σ_A
// Flattened index of Map<NT, List<TM-id>> values
fn get_all_tms(i : u32) -> u32     { return terminals.payload[terminals.all_tms_offset + i];     } // σ → TM‑id
// Counts the number of terminals directly generated by a unit-nonterminal entry in the parse chart
fn count_tms(val: u32, unit_nt: u32) -> u32 {
    let hasLiteral = ((val >> 1u) != 0u);             // bit‑packed literal present?
    let negLit     = (val & $NEG_STR_LIT) != 0u;      // negative‑literal flag
    let litCount   = select(0u,
                       select(1u,                                     // positive literal ⇒ exactly 1
                               max(1u, get_nt_tm_lens(unit_nt) - 1u), // negative ⇒ |Σ_A|‑1
                               negLit),
                       hasLiteral);
    return litCount;
}
"""

//language=wgsl
const val IDX_UNIFORM_STRUCT = """
struct IndexUniforms {  // Indices of all accepting states in the parse chart
    targetCnt       : atomic<u32>,  // global counter (LFSR advances on host)
    maxWordLen      : u32,
    numNonterminals : u32,
    numStates       : u32,
    threads         : u32,
    max_samples     : u32,
    
    startIdxOffset  : u32, numStartIndices : u32,
    startIndices    : array<u32> // Contains alternating (1) start index and (2) edit distance
};
"""

//language=wgsl
const val TERM_STRUCT = """
struct Terminals { // Mappings from nonterminals to terminals in CFG
    nt_tm_lens_offset : u32,    nt_tm_lens_size : u32,
       offsets_offset : u32,       offsets_size : u32,
       all_tms_offset : u32,       all_tms_size : u32,
       
       payload : array<u32>
};

$IDX_UNIFORM_STRUCT

$TM_DECODING_HELPERS
"""

//language=wgsl
const val CFL_STRUCT = """struct CFLStruct { // Carries metadata about the CFL + NFA intersection
             numStates : u32,      numNonterminals : u32,

           mdptsOffset : u32,            mdptsSize : u32,
    mdptsOffsetsOffset : u32,     mdptsOffsetsSize : u32,
    acceptStatesOffset : u32,     acceptStatesSize : u32,
grammarFlattenedOffset : u32, grammarFlattenedSize : u32,
  grammarOffsetsOffset : u32,   grammarOffsetsSize : u32,

               payload : array<u32>
};

         fn getMdpt(index: u32) -> u32 { return cs.payload[cs.mdptsOffset + index]; }
   fn getMdptOffset(index: u32) -> u32 { return cs.payload[cs.mdptsOffsetsOffset + index]; }
fn getGrammarSymbol(index: u32) -> u32 { return cs.payload[cs.grammarFlattenedOffset + index]; }
fn getGrammarOffset(index: u32) -> u32 { return cs.payload[cs.grammarOffsetsOffset + index]; }"""

//language=wgsl
const val WGSL_LANG_SIZE = """
fn langSize(dpIdx: u32, numNTs: u32) -> u32 {
  let val    = dp_in[dpIdx];
  let nt     = dpIdx % numNTs;

  let litCount = count_tms(val, nt);

  let expCnt = bp_count[dpIdx];
  if (expCnt == 0u) { return litCount; }
  let base   = bp_offset[dpIdx];
  let last   = ls_sparse[base + expCnt - 1u];
  return litCount + last;
}
"""

//language=text
const val SHORT_PREAMBLE = """
let N  = cs.numStates;
let NT = cs.numNonterminals;

let snt     = N * NT;
let dpIdx   = r*snt + c*NT + A;
let startGC = getGrammarOffset(A);
var endGC: u32;
if (A + 1u < NT) { endGC = getGrammarOffset(A + 1u); } else { endGC = cs.grammarFlattenedSize; }
let aoi            = r*N + c + 1u;
let pairOffset     = getMdptOffset(aoi - 1u);
var pairOffsetNext: u32;
if (aoi < cs.mdptsOffsetsSize) { pairOffsetNext = getMdptOffset(aoi); } 
else { pairOffsetNext = cs.mdptsSize; }
"""

//language=text
const val PREAMBLE = """
let r = gid.x;
let c = gid.y;
if (c <= r) { return; }
let A = gid.z;
$SHORT_PREAMBLE"""

//language=wgsl
val init_chart by Shader("""$CFL_STRUCT $TERM_STRUCT
@group(0) @binding(0) var<storage, read_write>     dp_in : array<u32>;
@group(0) @binding(1) var<storage, read>            word : array<u32>;
@group(0) @binding(2) var<storage, read>              cs : CFLStruct;
@group(0) @binding(3) var<storage, read>       terminals : Terminals;

fn pack_rc(row_j: u32, col_i: u32) -> u32 { return (row_j << 16u) | (col_i & 0xffffu); }
fn unpack_row_j(packed: u32) -> u32 { return packed >> 16u; }
fn unpack_col_i(packed: u32) -> u32 { return packed & 0xffffu; }

fn tpl(x: i32) -> i32 { let y: i32 = max(x, 0); return (y * (y + 1)) / 2; }

fn prefix_count(s_sum: i32, num_j_values: i32, num_i_values: i32) -> i32 {
    return tpl(s_sum) - tpl(s_sum - num_j_values) - tpl(s_sum - num_i_values) + tpl(s_sum - num_j_values - num_i_values);
}

fn find_target_sum(rank_k: u32, max_j_idx: u32, max_i_idx: u32) -> i32 {
    var target_s: i32 = 0;
    var low: i32 = 0;
    var high: i32 = i32(max_j_idx + max_i_idx);
    let num_j_vals = i32(max_j_idx + 1u);
    let num_i_vals = i32(max_i_idx + 1u);

    while (low <= high) {
        let mid_s = low + (high - low) / 2;
        if (prefix_count(mid_s, num_j_vals, num_i_vals) <= i32(rank_k)) {
            target_s = mid_s;
            low = mid_s + 1;
        } else {
            high = mid_s - 1;
        }
    }
    return target_s;
}

// unrank_to_coords(rank_idx, max_j_idx, max_i_idx) -> packed_coords(j,i)
// Primary sort: s = i+j (ascending)
// Secondary sort: i (string position, ascending). For fixed s, as i increases, j decreases.
fn unrank_to_coords(rank_idx: u32, max_j_idx: u32, max_i_idx: u32) -> u32 {
    let num_j_vals = i32(max_j_idx + 1u);
    let num_i_vals = i32(max_i_idx + 1u);

    let s_sum: i32 = find_target_sum(rank_idx, max_j_idx, max_i_idx);
    let elements_before_this_sum_group: i32 = prefix_count(s_sum, num_j_vals, num_i_vals);
    let offset_in_sum_group: i32 = i32(rank_idx) - elements_before_this_sum_group;

    let i_start_for_sum: i32 = max(0, s_sum - i32(max_j_idx));
    let i_final: u32 = u32(i_start_for_sum + offset_in_sum_group);
    let j_final: u32 = u32(s_sum - i32(i_final));

    return pack_rc(j_final, i_final);
}

fn letter_at(idx : u32, wd_len : u32) -> u32 { return select(word[idx], 0xffffffffu, idx >= wd_len); }

const LIT_ALL : u32 = 0x7ffffffeu;
const NEG_BIT : u32 = $NEG_STR_LIT;

fn encode_pos_literal(A_nt_idx : u32, sigma_token : u32) -> u32 {
    if (sigma_token == 0xffffffffu) { return 0u; }
    let ntLen = get_nt_tm_lens(A_nt_idx);
    let ntOff = get_offsets(A_nt_idx);
    for (var k : u32 = 0u; k < ntLen; k = k + 1u) {
        if (get_all_tms(ntOff + k) == sigma_token) { return ((k + 1u) << 1u); }
    }
    return 0u;
}

fn encode_neg_literal(A_nt_idx : u32, sigma_token : u32) -> u32 {
    var s_is_in_Sigma_A = false;
    var k_idx_of_s_in_Sigma_A : u32 = 0u;
    if (sigma_token != 0xffffffffu) {
        let ntLen = get_nt_tm_lens(A_nt_idx);
        let ntOff = get_offsets(A_nt_idx);
        for (var k : u32 = 0u; k < ntLen; k = k + 1u) {
            if (get_all_tms(ntOff + k) == sigma_token) {
                s_is_in_Sigma_A = true;
                k_idx_of_s_in_Sigma_A = k;
                break;
            }
        }
    }
    if (s_is_in_Sigma_A) { return NEG_BIT | ((k_idx_of_s_in_Sigma_A + 1u) << 1u); }
    else { return LIT_ALL; }
}

const MAX_J_IDX_CONST : u32 = ${MAX_LEV_RAD}u; // Max index for j (edit distance)

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let q1_rank = gid.x;
    let q2_rank = gid.y;
    let A_idx  = gid.z;
    let nts = cs.numNonterminals;

    let dpIdx = q1_rank * cs.numStates * nts + q2_rank * nts + A_idx;

    let current_word_len = arrayLength(&word); // Max i is current_word_len (0 to N for string of length N)

    let packed_q1_ji = unrank_to_coords(q1_rank, MAX_J_IDX_CONST, current_word_len);
    let j1 = unpack_row_j(packed_q1_ji);
    let i1 = unpack_col_i(packed_q1_ji);

    let packed_q2_ji = unrank_to_coords(q2_rank, MAX_J_IDX_CONST, current_word_len);
    let j2 = unpack_row_j(packed_q2_ji);
    let i2 = unpack_col_i(packed_q2_ji);

    let di = i32(i2) - i32(i1);
    let dj = i32(j2) - i32(j1);

    var encoded_predicate_val : u32 = 0u;
    var should_write_to_dp_in : bool = false;
    let num_prods_for_A = get_nt_tm_lens(A_idx);
    
    // (1) UP ARC (Insertion): q1=(i,j-1) -> q2=(i,j). Predicate: [!=word[i1]]
    if (di == 0 && dj == 1) {
        if (j1 < MAX_J_IDX_CONST && i1 <= current_word_len) {
            let s_char = letter_at(i1, current_word_len);
            encoded_predicate_val = encode_neg_literal(A_idx, s_char);
            if (encoded_predicate_val == LIT_ALL) { should_write_to_dp_in = (num_prods_for_A > 0u); }
            else if ((encoded_predicate_val & NEG_BIT) != 0u) { should_write_to_dp_in = (num_prods_for_A > 1u); }
        }
    }
    // (2) RIGHT ARC (Match): q1=(i-1,j) -> q2=(i,j). Predicate: word[i1]
    else if (di == 1 && dj == 0) {
        if (i1 < current_word_len) {
            let s_char = letter_at(i1, current_word_len);
            encoded_predicate_val = encode_pos_literal(A_idx, s_char);
            if (encoded_predicate_val != 0u) { should_write_to_dp_in = true; }
        }
    }
    // (3) DIAG ARC (Substitution): q1=(i-1,j-1) -> q2=(i,j). Predicate: [!=word[i1]]
    else if (di == 1 && dj == 1) {
        if (i1 < current_word_len && j1 < MAX_J_IDX_CONST) {
            let s_char = letter_at(i1, current_word_len);
            encoded_predicate_val = encode_neg_literal(A_idx, s_char);
            if (encoded_predicate_val == LIT_ALL) { should_write_to_dp_in = (num_prods_for_A > 0u); }
            else if ((encoded_predicate_val & NEG_BIT) != 0u) { should_write_to_dp_in = (num_prods_for_A > 1u); }
        }
    }
    // (4) "KNIGHT" ARC (Deletion): q1=(i,j) -> q2=(i+d+1,j+d). Predicate: word[i1+d]
    else if (dj >= 1 && di == dj + 1) {
        let d_val = u32(dj);
        if (i1 + d_val < current_word_len) {
            if ( (i1 + d_val + 1u <= current_word_len) && (j1 + d_val <= MAX_J_IDX_CONST) ) {
                let s_char = letter_at(i1 + d_val, current_word_len);
                encoded_predicate_val = encode_pos_literal(A_idx, s_char);
                if (encoded_predicate_val != 0u) { should_write_to_dp_in = true; }
            }
        }
    }

    if (should_write_to_dp_in) { dp_in[dpIdx] = encoded_predicate_val; }
}""")

//language=wgsl
val dag_reach by Shader("""struct AtomicChange { count: atomic<u32> };
@group(0) @binding(0) var<storage, read_write>   input : array<u32>;
@group(0) @binding(1) var<storage, read_write> changes : AtomicChange;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= y) { return; }
    let width = u32(sqrt(f32(arrayLength(&input))));
//    if (x == y) { input[x * width + y] = 1u; atomicAdd(&changes.count, 1u); return; }
    if (input[x * width + y] == 1u) { atomicAdd(&changes.count, 1u); return; }

    for (var k = 0u; k < width; k = k + 1u) {
        if (input[x * width + k] == 1u && input[k * width + y] == 1u) {
            input[x * width + y] = 1u;
            atomicAdd(&changes.count, 1u);
            return;
        }
    }
}""")

//language=wgsl
val mdpt_count by Shader("""struct Uni { n : u32 };
@group(0) @binding(0) var<storage, read>        reach : array<u32>;   // N×N upper‑tri (0/1)
@group(0) @binding(1) var<storage, read_write> counts : array<u32>;   // N×N (aoi‑1 → #midpts)
@group(0) @binding(2) var<uniform>                uni : Uni;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let r = gid.y;  let c = gid.x;  let N = uni.n;
    if (r >= N || c >= N || c <= r) { return; }

    let idx = r*N + c;
    if (reach[idx]==0u) { counts[idx]=0u; return; }

    var cnt = 0u;
    for (var v=0u; v<N; v++) { if (reach[r*N+v]==1u && reach[v*N+c]==1u) { cnt++; } }
    counts[idx] = cnt;
}""")

//language=wgsl
val mdpt_write by Shader("""struct Uni { n : u32 };
@group(0) @binding(0) var<storage, read>         reach : array<u32>;
@group(0) @binding(1) var<storage, read>       offsets : array<u32>; // exclusive scan of counts
@group(0) @binding(2) var<storage, read_write> flat_mp : array<u32>; // flattened mid‑points
@group(0) @binding(3) var<uniform>                 uni : Uni;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let r = gid.y;  let c = gid.x;  let N = uni.n;
    if (r >= N || c >= N || c <= r) { return; }

    let idx = r*N + c;
    if (reach[idx]==0u) { return; }

    var out = offsets[idx];
    for (var v=0u; v<N; v++) { if (reach[r*N+v]==1u && reach[v*N+c]==1u) { flat_mp[out] = v; out++; } }
}""")

//language=wgsl
val cfl_mul_upper by Shader("""$CFL_STRUCT
struct AtomicChange { count: atomic<u32> };

@group(0) @binding(0) var<storage, read_write>    dp_in : array<u32>;
@group(0) @binding(1) var<storage, read>             cs : CFLStruct;
@group(0) @binding(2) var<storage, read_write>  changes : AtomicChange;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $PREAMBLE
    
    let dpVal = dp_in[dpIdx];
    if (dpVal != 0) {
        atomicAdd(&changes.count, 1u);
        if ((dpVal & 0x01) != 0) { return; }
    }

    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = getMdpt(pairIdx); for (var g = startGC; g < endGC; g+= 2u) {
            let B = getGrammarSymbol(g); let C = getGrammarSymbol(g + 1u);

            let idxBM = r*snt + mdpt*NT + B;
            let idxMC = mdpt*snt + c*NT + C;

            if ((dp_in[idxBM] != 0) && (dp_in[idxMC] != 0)) {
                dp_in[dpIdx] |= 0x01;
                atomicAdd(&changes.count, 1u);
                return;
            }
        }
    }
}""")

//language=wgsl
val bp_count by Shader("""$CFL_STRUCT
@group(0) @binding(0) var<storage, read>           dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>  bp_count : array<u32>;
@group(0) @binding(2) var<storage, read>              cs : CFLStruct;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $PREAMBLE
    
    if ((dp_in[dpIdx] & 0x01u) == 0u) { bp_count[dpIdx] = 0; return; }
    
    var count = 0u;
    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = getMdpt(pairIdx); for (var g = startGC; g < endGC; g+= 2u) {
            let B = getGrammarSymbol(g);
            let C = getGrammarSymbol(g + 1u);

            let idxBM = r*snt + mdpt*NT + B;
            let idxMC = mdpt*snt + c*NT + C;

            if (dp_in[idxBM] != 0u && dp_in[idxMC] != 0u) { count++; }
        }
    }

    bp_count[dpIdx] = count;
}""")

//language=wgsl
val bp_write by Shader("""$CFL_STRUCT
@group(0) @binding(0) var<storage, read>             dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>   bp_offset : array<u32>;
@group(0) @binding(2) var<storage, read_write>  bp_storage : array<u32>;
@group(0) @binding(3) var<storage, read>                cs : CFLStruct;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    $PREAMBLE
    
    if ((dp_in[dpIdx] & 0x01u) == 0u) { return; }

    var outPos = bp_offset[dpIdx];

    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx = pairIdx + 1u) {
        let mdpt = getMdpt(pairIdx); for (var g = startGC; g < endGC; g += 2u) {
            let B = getGrammarSymbol(g);
            let C = getGrammarSymbol(g + 1u);

            let idxBM = r*snt + mdpt*NT + B;
            let idxMC = mdpt*snt + c*NT + C;

            if (dp_in[idxBM] != 0u && dp_in[idxMC] != 0u) {
                bp_storage[outPos * 2u + 0u] = idxBM;
                bp_storage[outPos * 2u + 1u] = idxMC;
                outPos++;
            }
        }
    }
}""")

//language=wgsl
val ls_dense by Shader("""$CFL_STRUCT $TERM_STRUCT
struct SpanUni { span : u32 };
@group(0) @binding(0) var<storage, read>           dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>  ls_dense : array<u32>;
@group(0) @binding(2) var<storage, read>              cs : CFLStruct;
@group(0) @binding(3) var<storage, read>       terminals : Terminals;
@group(0) @binding(4) var<uniform>                    su : SpanUni;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let r = gid.x;
    let c = r + su.span;
    if (c >= cs.numStates) { return; }
    let A = gid.z;
    
    $SHORT_PREAMBLE
    
    let val = dp_in[dpIdx];
    if (val == 0u) { return; }

    let litCount = count_tms(val, A);

    if ((val & 0x01u) == 0u) { ls_dense[dpIdx] = max(litCount, 1u); return; }

    var total: u32 = litCount;

    for (var p = pairOffset; p < pairOffsetNext; p = p + 1u) {
        let m = getMdpt(p);

        for (var g = startGC; g < endGC; g = g + 2u) {
            let B = getGrammarSymbol(g);
            let C = getGrammarSymbol(g + 1u);

            let idxBM = r*snt + m*NT + B;
            let idxMC = m*snt + c*NT + C;

            // only add if both children are present
            if (dp_in[idxBM] != 0u && dp_in[idxMC] != 0u) { total += ls_dense[idxBM] * ls_dense[idxMC]; }
        }
    }
    ls_dense[dpIdx] = max(total, 1u);  // total==0 should not happen, but guard anyway
}""")

//language=wgsl
val ls_cdf by Shader("""$CFL_STRUCT $TERM_STRUCT
@group(0) @binding(0) var<storage, read>             dp_in : array<u32>;
@group(0) @binding(1) var<storage, read>          ls_dense : array<u32>;
@group(0) @binding(2) var<storage, read>         bp_offset : array<u32>;
@group(0) @binding(3) var<storage, read_write>   ls_sparse : array<u32>;
@group(0) @binding(4) var<storage, read>                cs : CFLStruct;
@group(0) @binding(5) var<storage, read>         terminals : Terminals;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $PREAMBLE

    let val = dp_in[dpIdx];
    if (val == 0u) { return; }

    var acc    : u32 = 0u;
    var outPos : u32 = bp_offset[dpIdx];
    
    let litCount = count_tms(val, A);

    for (var p = pairOffset; p < pairOffsetNext; p = p + 1u) {
        let m = getMdpt(p);

        for (var g = startGC; g < endGC; g = g + 2u) {
            let B = getGrammarSymbol(g);
            let C = getGrammarSymbol(g + 1u);

            let idxBM = r*snt + m*NT + B;
            let idxMC = m*snt + c*NT + C;

            if (dp_in[idxBM] != 0u && dp_in[idxMC] != 0u) {
                acc += ls_dense[idxBM] * ls_dense[idxMC];
                ls_sparse[outPos] = acc + litCount;
                outPos += 1u;
            }
        }
    }
}""")

const val PFX_SUM_PARAMS = """struct PrefixSumUni { n : u32, numBlocks : u32, threads : u32 };"""
// language=wgsl
val prefix_sum_p1 by Shader("""$PFX_SUM_PARAMS
@group(0) @binding(0) var<storage, read>         inputBuf : array<u32>;
@group(0) @binding(1) var<storage, read_write>  outputBuf : array<u32>;
@group(0) @binding(2) var<storage, read_write>  blockSums : array<u32>;
@group(0) @binding(3) var<uniform>              prefixUni : PrefixSumUni;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> tile: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE) fn main(
  @builtin(workgroup_id)        groupId : vec3<u32>,
  @builtin(local_invocation_id) localId : vec3<u32>
) {
    let N     = prefixUni.n;
    let block = groupId.x + groupId.y * prefixUni.threads;
    
    if (block >= prefixUni.numBlocks) { return; }
    
    let lid = localId.x;
    let gid = block * WORKGROUP_SIZE + lid;   // global element index

    // 1) Load data from inputBuf into shared workgroup array `tile`.
    if (gid < N) { tile[lid] = inputBuf[gid]; } else { tile[lid] = 0u; }
    workgroupBarrier();

    // 2) Up-sweep: build partial sums in place.
    //    Offsets go 1, 2, 4, 8, ...
    var offset = 1u;
    while (offset < WORKGROUP_SIZE) {
        // index = (lid+1)*offset*2 - 1
        let idx = ((lid + 1u) * offset * 2u) - 1u;
        if (idx < WORKGROUP_SIZE) { tile[idx] = tile[idx] + tile[idx - offset]; }
        workgroupBarrier();
        offset = offset * 2u;
    }

    // 3) The last element of `tile` now has the total sum of this block.
    //    Save that to blockSums, then zero it out so this becomes an EXCLUSIVE scan.
    if (lid == 0u) {
        blockSums[block] = tile[WORKGROUP_SIZE - 1u];
        tile[WORKGROUP_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    // 4) Down-sweep: push each partial sum back down the tree to build the exclusive scan.
    //    Offsets go (256 >> 1), (256 >> 2), ...
    offset = WORKGROUP_SIZE / 2u;
    while (offset > 0u) {
        let idx = ((lid + 1u) * offset * 2u) - 1u;
        if (idx < WORKGROUP_SIZE) {
            let tmp = tile[idx - offset];
            tile[idx - offset] = tile[idx];
            tile[idx] = tile[idx] + tmp;
        }
        workgroupBarrier();
        offset = offset / 2u;
    }

    // 5) Write the per-element results back out to outputBuf.
    if (gid < N) { outputBuf[gid] = tile[lid]; }
}""")

//language=wgsl
val prefix_sum_p2 by Shader("""$PFX_SUM_PARAMS
@group(0) @binding(0) var<storage, read_write>          dataBuf : array<u32>;
@group(0) @binding(1) var<storage, read>       scannedBlockSums : array<u32>;
@group(0) @binding(2) var<uniform>                    prefixUni : PrefixSumUni;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id)        groupId : vec3<u32>,
    @builtin(local_invocation_id) localId : vec3<u32>
) {
    let block = groupId.x + groupId.y * prefixUni.threads;
    if (block >= prefixUni.numBlocks) { return; }

    let gid = block * 256u + localId.x;
    if (gid >= prefixUni.n) { return; }

    let offset = scannedBlockSums[block];
    dataBuf[gid] = dataBuf[gid] + offset;
}""")

// Longest word WGSL can handle. If ~2^9<MAX_WORD_LEN, pipeline breaks some on architectures
const val MAX_WORD_LEN = 512
const val MAX_LEV_RAD = 5
const val MAX_SAMPLES = 550_000
const val DISPATCH_GROUP_SIZE_X = 65_535
// Length of the packet header in each repair buffer
const val PKT_HDR_LEN = 2 // [levenshtein distance, Markov probability]
const val MAX_SAMPLES_PER_DIST = 30_000
const val SENTINEL = 0xFFFF_FFFFu
const val HASH_MUL = 0x1e35a7bdu

//language=wgsl
val build_root_sizes by Shader("""$TERM_STRUCT
@group(0) @binding(0) var<storage, read>        dp_in      : array<u32>;
@group(0) @binding(1) var<storage, read>        bp_count   : array<u32>;
@group(0) @binding(2) var<storage, read>        bp_offset  : array<u32>;
@group(0) @binding(3) var<storage, read>        ls_sparse  : array<u32>;
@group(0) @binding(4) var<storage, read>        terminals  : Terminals;
@group(0) @binding(5) var<storage, read_write>  root_sizes : array<u32>;
@group(0) @binding(6) var<storage, read_write>  idx_uni    : IndexUniforms;

$CHART_DECODING_HELPERS

const NEG_MASK : u32 = $NEG_STR_LIT;

$WGSL_LANG_SIZE

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let numRoots = idx_uni.numStartIndices / 2u;
  if (i >= numRoots) { return; }
  root_sizes[i] = langSize(getStartIdx(i), idx_uni.numNonterminals);
}""")

/** See [PTree.sampleStrWithoutReplacement] for CPU version. */
//language=wgsl
val enum_words_wor by Shader("""$TERM_STRUCT
@group(0) @binding(0) var<storage, read>        dp_in       : array<u32>;
@group(0) @binding(1) var<storage, read>        bp_count    : array<u32>;
@group(0) @binding(2) var<storage, read>        bp_offset   : array<u32>;
@group(0) @binding(3) var<storage, read>        bp_storage  : array<u32>;
@group(0) @binding(4) var<storage, read>        ls_sparse   : array<u32>;
@group(0) @binding(5) var<storage, read>        terminals   : Terminals;
@group(0) @binding(6) var<storage, read_write>  idx_uni     : IndexUniforms;
@group(0) @binding(7) var<storage, read>        root_sizes  : array<u32>;   // length = numRoots
@group(0) @binding(8) var<storage, read>        root_cdf    : array<u32>;   // exclusive scan
@group(0) @binding(9) var<storage, read_write>  sampled     : array<u32>;   // out packets

$CHART_DECODING_HELPERS

const PKT_HDR_LEN : u32 = ${PKT_HDR_LEN}u;
const NEG_MASK    : u32 = ${NEG_STR_LIT};

$WGSL_LANG_SIZE

fn binarySearchCDF(base: u32, len: u32, needle: u32) -> u32 {
  var lo: u32 = 0u;
  var hi: u32 = len;
  loop {
    if (lo >= hi) { return base + lo; }
    let mid = (lo + hi) >> 1u;
    if (needle < ls_sparse[base + mid]) { hi = mid; } else { lo = mid + 1u; }
  }
}

fn decodeLiteral(
  dpIdx: u32, // gives NT via modulo
  litEnc: u32,
  negLit: bool,
  variant: u32,
  word: ptr<function, array<u32, ${MAX_WORD_LEN}u>>,
  wLen: ptr<function, u32>
) {
  let nt    = dpIdx % idx_uni.numNonterminals;
  let ntLen = get_nt_tm_lens(nt);
  let ntOff = get_offsets(nt);
  if (negLit) {
    if (ntLen == 0u) { return; }
    let excl = litEnc - 1u;
    let idx  = select(variant, variant + 1u, variant >= excl);
    (*word)[*wLen] = get_all_tms(ntOff + idx) + 1u;
  } else { (*word)[*wLen] = get_all_tms(ntOff + (litEnc - 1u)) + 1u; }
  *wLen = *wLen + 1u;
}

struct Frame { dp: u32, rk: u32 }

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sid = gid.x + gid.y * idx_uni.threads;
  if (sid >= idx_uni.max_samples) { return; }

  let numRoots = idx_uni.numStartIndices / 2u;
  if (numRoots == 0u) { return; }

  // total = lastCDF + lastSize
  let last = numRoots - 1u;
  let total = root_cdf[last] + root_sizes[last];
  if (sid >= total) { return; } // strict WOR

  // Global rank → root via binary search over root_cdf/root_sizes
  var rLo: u32 = 0u;
  var rHi: u32 = numRoots;
  loop {
    if (rLo + 1u >= rHi) { break; }
    let mid = (rLo + rHi) >> 1u;
    let base = root_cdf[mid];
    if (sid < base) { rHi = mid; } else {
      let size = root_sizes[mid];
      if (sid < base + size) { rLo = mid; rHi = mid + 1u; break; }
      rLo = mid + 1u;
    }
  }
  let rootIdx = rLo;
  let base    = root_cdf[rootIdx];
  var rk      = sid - base;               // rank within chosen root
  let dpRoot  = getStartIdx(rootIdx);
  let levDist = getEditDist(rootIdx);

  // DFS decode by rank (without replacement by construction)
  var stack : array<Frame, ${MAX_WORD_LEN}u>;
  var top   : u32 = 0u;
  stack[top] = Frame(dpRoot, rk); top++;

  var word  : array<u32, ${MAX_WORD_LEN}u>;
  var wLen  : u32 = 0u;

  loop {
    if (top == 0u) { break; }
    top = top - 1u;
    var fr = stack[top];
    var d  = fr.dp;
    rk     = fr.rk;

    let val    = dp_in[d];
    let nt     = d % idx_uni.numNonterminals;
    let negLit   = (val & $NEG_STR_LIT) != 0u;
    let litCount = count_tms(val, nt);

    let expCnt = bp_count[d];
    let tot    = litCount + select(0u, ls_sparse[bp_offset[d] + expCnt - 1u], expCnt != 0u);
    rk = rk % tot;

    if (rk < litCount) {
      decodeLiteral(d, (val >> 1u) & 0x1fffffffu, negLit, rk, &word, &wLen);
      continue;
    }
    rk = rk - litCount;

    let base2    = bp_offset[d];
    let choiceIx = binarySearchCDF(base2, expCnt, rk);
    let prevCDF  = select(0u, ls_sparse[choiceIx - 1u], choiceIx != base2);
    let inside   = rk - prevCDF;

    let left  = bp_storage[2u * choiceIx + 0u];
    let right = bp_storage[2u * choiceIx + 1u];

    let sizeR = langSize(right, idx_uni.numNonterminals);
    let rkL   = inside / sizeR;
    let rkR   = inside % sizeR;

    // Push right first, then left, to keep left-to-right DFS order
    stack[top] = Frame(right, rkR); top++;
    stack[top] = Frame(left,  rkL); top++;
  }

  // Write packet
  let stride  = idx_uni.maxWordLen;
  let outBase = sid * stride;

  sampled[outBase + 0u] = levDist;
  sampled[outBase + 1u] = 0u; // optional, keep deterministic before markov_score

  for (var i = 0u; i < wLen && i + PKT_HDR_LEN < stride; i = i + 1u) { sampled[outBase + PKT_HDR_LEN + i] = word[i]; }

  // CRITICAL: write a terminator so the CPU decoder stops correctly
  if (PKT_HDR_LEN + wLen < stride) { sampled[outBase + PKT_HDR_LEN + wLen] = 0u; }
//  let stride = idx_uni.maxWordLen;
//  let outBase = sid * stride;
//  sampled[outBase + 0u] = levDist; // distance
  // sampled[outBase + 1u] left for score; will be filled by markov_score
//  for (var i = 0u; i < wLen && i + PKT_HDR_LEN < stride; i = i + 1u) { sampled[outBase + PKT_HDR_LEN + i] = word[i]; }
}""".trimIndent())

//language=wgsl
const val SAMPLER_PARAMS = """struct Params { maxSamples: u32, k: u32, stride: u32, threads: u32 };"""
//language=wgsl
val markov_score by Shader("""$SAMPLER_PARAMS
@group(0) @binding(0) var<storage, read_write>  packets : array<u32>; // sampledWords/outBuf
@group(0) @binding(1) var<storage, read>          ngram : array<u32>; // hash table
@group(0) @binding(2) var<uniform>                  prm : Params;

const PKT_HDR_LEN  : u32 = ${PKT_HDR_LEN}u;
const SENTINEL_KEY : u32 = ${SENTINEL.toHexString(hexFmt)};
const HASH_MUL     : u32 = ${HASH_MUL.toHexString(hexFmt)};      // same multiplier as CPU side
const BOS_ID       : u32 = ${BOS_ID}u;
const NEWLINE_ID   : u32 = ${NEWLINE_ID}u;
const EOS_ID       : u32 = ${EOS_ID}u;

fn packGram(a : u32, b : u32, c : u32,d : u32) -> u32 { return (a<<21u)|(b<<14u)|(c<<7u)|d; }

fn hash32(x : u32, pow : u32) -> u32 { return (x * HASH_MUL) >> (32u - pow); }

fn lookupScore(key: u32) -> u32 {
    let pow   : u32 = ngram[0];
    let mask  : u32 = (1u << pow) - 1u;
    var slot  : u32 = hash32(key, pow) & mask;

    loop {                                       // ≤ 8 probes when load ≤ 0.75
        let idx      = 1u + slot * 2u;           // 1-word header → slot*2
        let stored   = ngram[idx];
        if (stored == key)          { return ngram[idx + 1u]; } // hit
        if (stored == SENTINEL_KEY) { return 1u; }              // empty
        slot = (slot + 1u) & mask;                              // linear probe
    }
}

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let sid = gid.x + gid.y * prm.threads;
    if (sid >= prm.maxSamples) { return; }

    let stride : u32 = prm.stride;
    let base   : u32 = sid * stride;

    var t0 : u32 = BOS_ID     - 1u;
    var t1 : u32 = NEWLINE_ID - 1u;

    // -- pre-fetch the guaranteed first real token -----------
    var pos : u32 = 1u;
    let w1  : u32 = packets[base + PKT_HDR_LEN];
    var t2  : u32 = w1 - 1u;

    var score       : u32 = 0u;
    var doneSuffix  : u32 = 0u;

    loop {
        // ----- next token or synthetic suffix ---------------
        var tok : u32;
        if (pos < stride - PKT_HDR_LEN && packets[base + PKT_HDR_LEN + pos] != 0u) {
            tok = packets[base + PKT_HDR_LEN + pos];
            pos += 1u;
        } else {
            // two‑token suffix: NEWLINE , EOS
            tok = select(EOS_ID, NEWLINE_ID, doneSuffix == 0u);
            doneSuffix += 1u;
            if (doneSuffix > 2u) { break; }
        }

        // ----- accumulate n-gram score ----------------------
        let key = packGram(t0, t1, t2, tok - 1u);
        score  += lookupScore(key);

        t0 = t1; t1 = t2; t2 = tok - 1u;
    }

    packets[base + 1u] = score + (packets[base] + 1u) * 10000000u;
}""")

//language=wgsl
val select_top_k by Shader("""$SAMPLER_PARAMS
@group(0) @binding(0) var<uniform>                  prm : Params;
@group(0) @binding(1) var<storage, read>        packets : array<u32>;
@group(0) @binding(2) var<storage, read_write>   topIdx : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> topScore : array<atomic<u32>>;

const UINT_MAX : u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id)        workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_id     : vec3<u32>
) {
    let workgroup_linear_id = workgroup_id.x + workgroup_id.y * prm.threads;
    let i = workgroup_linear_id * 256u + local_id.x;
    if (i >= prm.maxSamples || prm.k == 0u) { return; }

    let score : u32 = packets[i * prm.stride + 1u];

    loop {
        var worstPos : u32 = 0u;
        var worstVal : u32 = atomicLoad(&topScore[0]);
        var duplicate : bool = (worstVal == score);
        for (var j : u32 = 1u; j < prm.k; j = j + 1u) {
            let v = atomicLoad(&topScore[j]);
            if (v == score) { duplicate = true; }
            if (v > worstVal) { worstVal = v; worstPos = j; }
        }

        if (duplicate || score >= worstVal) { return; }
        let old = atomicCompareExchangeWeak(&topScore[worstPos], worstVal, score);
        if (old.exchanged) { atomicStore(&topIdx[worstPos], i); return; }
    }
}""")

//language=wgsl
val gather_top_k by Shader("""$SAMPLER_PARAMS
@group(0) @binding(0) var<uniform>                prm : Params;
@group(0) @binding(1) var<storage, read>      packets : array<u32>;  // full outBuf
@group(0) @binding(2) var<storage, read>       topIdx : array<u32>;  // k indices
@group(0) @binding(3) var<storage, read_write> bestPk : array<u32>;  // compacted result

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let j : u32 = gid.x;
    if (j >= prm.k) { return; }

    let srcIdx : u32 = topIdx[j];
    if (srcIdx == 0xFFFFFFFFu) { return; } 

    let stride : u32 = prm.stride;
    let srcOff : u32 = srcIdx * stride;
    let dstOff : u32 = j      * stride;

    for (var t: u32 = 0u; t < stride; t = t + 1u) { bestPk[dstOff + t] = packets[srcOff + t]; }
}""")

//language=wgsl
val sparse_load by Shader("""
struct SparseElement { r: u32, c: u32, v: u32, i: u32 };
struct Coeffs { rowCoeff: u32, colCoeff: u32 };

@group(0) @binding(0) var<storage, read>     sparse_elements : array<SparseElement>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<u32>;
@group(0) @binding(2) var<uniform>                    coeffs : Coeffs;

// Define workgroup size (must match constant in Kotlin code)
const WORKGROUP_SIZE: u32 = ${SPARSE_WRITER_WORKGROUP_SIZE}u;

@compute @workgroup_size(WORKGROUP_SIZE) fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let num_elements = arrayLength(&sparse_elements);
    let output_size = arrayLength(&output_buffer);
    if (index >= num_elements) { return; }
    let element = sparse_elements[index];
    let target_index = element.r * coeffs.rowCoeff + element.c * coeffs.colCoeff + element.v;
    if (target_index < output_size) { output_buffer[target_index] = element.i; }
}""")

//language=wgsl
val sparse_mat_load by Shader("""struct SparseElement { r: u32, c: u32 };
@group(0) @binding(0) var<storage, read>     sparse_elements : array<SparseElement>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<u32>;

const WORKGROUP_SIZE: u32 = ${SPARSE_WRITER_WORKGROUP_SIZE}u;

@compute @workgroup_size(WORKGROUP_SIZE) fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let num_elements = arrayLength(&sparse_elements);
    let output_size = arrayLength(&output_buffer);
    if (index >= num_elements) { return; }
    let element = sparse_elements[index];
    let width = u32(sqrt(f32(output_size)));
    let target_index = element.r * width + element.c;
    if (target_index < output_size) { output_buffer[target_index] = 1; }
}""")

const val SPARSE_WRITER_WORKGROUP_SIZE = 256

object GPUBufferUsage {
  const val MAP_READ      = 0x0001
  const val MAP_WRITE     = 0x0002
  const val COPY_SRC      = 0x0004
  const val COPY_DST      = 0x0008
  const val INDEX         = 0x0010
  const val VERTEX        = 0x0020
  const val UNIFORM       = 0x0040
  const val STORAGE       = 0x0080
  const val INDIRECT      = 0x0100
  const val QUERY_RESOLVE = 0x0200
  const val STCPSD = STORAGE or COPY_SRC or COPY_DST
}

class Shader constructor(val src: String) {
  lateinit var name: String
  lateinit var pipeline: GPUComputePipeline

  @OptIn(ExperimentalWasmJsInterop::class)
  suspend fun bind() = gpu.createComputePipelineAsync(
    GPUComputePipelineDescriptor(
      layout = "auto",
      compute = GPUProgrammableStage(
        module = gpu.createShaderModule(GPUShaderModuleDescriptor(label = name, code = src)),
        entryPoint = "main"
      )
    )
  ).unsafeCast<KPromise<GPUComputePipeline>>().await().also { pipeline = it }

  operator fun getValue(tr: Any?, property: KProperty<*>): Shader = this.also { name = property.name }

  companion object {
    fun GPUComputePipeline.bindBuffers(label: String, vararg buffers: GPUBuffer): GPUBindGroup {
      inline fun <T> jsObject(init: dynamic.() -> Unit): T { val o = js("{}"); init(o); return o as T }
      val ent = buffers.mapIndexed { index, buf ->
        GPUBindGroupEntry(binding = index, resource = jsObject { buffer = buf })
      }.toTypedArray()
      return gpu.createBindGroup(GPUBindGroupDescriptor(label = label, layout = getBindGroupLayout(0), entries = ent))
    }

    suspend fun GPUBuffer.readInts(): IntArray {
//      val t0 = TimeSource.Monotonic.markNow()
      val readDst = GPUBuffer(size.toInt(), GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
      val cmd = gpu.createCommandEncoder()
      cmd.copyBufferToBuffer(source = this, sourceOffset = 0.0, destination = readDst, destinationOffset = 0.0, size = size)
      gpu.queue.submit(arrayOf(cmd.finish()))
      readDst.mapAsync(1).unsafeCast<Promise<*>>().await()
      val t = Int32Array(readDst.getMappedRange()).asList().toIntArray()
      readDst.destroy()
//      log("Read ${size.toInt()} bytes in ${t0.elapsedNow()}")
      return t
    }

    suspend fun GPUBuffer.readIndices(indices: List<Int>): List<Int> {
      val t0 = TimeSource.Monotonic.markNow()
      val stagingBuffer = GPUBuffer(indices.size * 4L, GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
      val encoder = gpu.createCommandEncoder()
      indices.forEachIndexed { i, idx ->
        encoder.copyBufferToBuffer(
          source = this,
          sourceOffset = idx.toDouble() * 4,
          destination = stagingBuffer,
          destinationOffset = i.toDouble() * 4,
          size = 4.0
        )
      }
      gpu.queue.submit(arrayOf(encoder.finish()))
      stagingBuffer.mapAsync(1).unsafeCast<Promise<*>>().await()
      val t = Int32Array(stagingBuffer.getMappedRange())
        .asList().toIntArray().toList().also { stagingBuffer.destroy() }
      log("Read ${indices.size}/${size.toInt()} bytes in ${t0.elapsedNow()}")
      return t
    }

    fun createParseChart(usage: Int, totalSizeInInts: Int): GPUBuffer {
      val outputByteSize = totalSizeInInts.toLong() * Int32Array.BYTES_PER_ELEMENT
      return GPUBuffer(outputByteSize, usage or GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)
    }

    fun IntArray.toGPUBufferSparse(usage: Int, totalSizeInInts: Int, rowCoeff: Int, colCoeff: Int): GPUBuffer {
      require(size % 4 == 0) { "Input array size must be a multiple of 4 for sparse data (r,c,v,i)." }
      require(totalSizeInInts > 0) { "totalSizeInInts must be positive." }

      val sparseDataGpuBuffer = toGPUBuffer()
      val outputBuffer = createParseChart(usage, totalSizeInInts)
      val coeffsBuffer = intArrayOf(rowCoeff, colCoeff).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
      val numWorkgroups = ceil(size / 4.0 / SPARSE_WRITER_WORKGROUP_SIZE).toInt()

      sparse_load(sparseDataGpuBuffer, outputBuffer, coeffsBuffer)(numWorkgroups)

      return outputBuffer.also { sparseDataGpuBuffer.destroy(); coeffsBuffer.destroy() }
    }

    fun IntArray.toSquareMatrixSparse(n: Int): GPUBuffer {
      val outputByteSize = n * n * Int32Array.BYTES_PER_ELEMENT
      val outputBuffer = GPUBuffer(outputByteSize, STCPSD)
      val sparseDataBuffer = toGPUBuffer()
      val numWorkgroups = ceil((size / 2.0) / SPARSE_WRITER_WORKGROUP_SIZE).toInt()
      sparse_mat_load(sparseDataBuffer, outputBuffer)(numWorkgroups)
      sparseDataBuffer.destroy()
      return outputBuffer
    }

    fun List<Int>.toGPUBuffer(usage: Int = STCPSD): GPUBuffer = toTypedArray().toGPUBuffer(usage)
    fun List<UInt>.toGPUBuffer(usage: Int = STCPSD): GPUBuffer = map { it.toInt() }.toTypedArray().toGPUBuffer(usage)
    fun IntArray.toGPUBuffer(usage: Int = GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST): GPUBuffer =
      GPUBuffer(size * 4, usage, unsafeCast<Int32Array<ArrayBuffer>>())
    fun Int.toGPUBuffer(usage: Int = STCPSD): GPUBuffer = intArrayOf(this).toGPUBuffer(usage)
    fun Array<Int>.toGPUBuffer(usage: Int = STCPSD): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@toGPUBuffer, 0) }
        .let { GPUBuffer(byteSize = size * 4, us = usage, data = it) }

    // TODO: figure out map/unmap lifetime?
    fun GPUBuffer(byteSize: Number, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
      gpu.createBuffer(descriptor = GPUBufferDescriptor(size = byteSize.toDouble(), usage = us))
        .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }

    private const val WORKGROUP_SIZE = 256

    fun prefixSumGPU(inputBuf: GPUBuffer, length: Int): GPUBuffer {
      val numBlocks = (length + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
      val groupsX   = DISPATCH_GROUP_SIZE_X
      val groupsY   = (numBlocks + groupsX - 1) / groupsX

      val outputBuf     = GPUBuffer(inputBuf.size.toInt(), STCPSD)
      val blockSumsBuf  = GPUBuffer(numBlocks * 4, STCPSD)
      val uniBuf        = intArrayOf(length, numBlocks, groupsX).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      prefix_sum_p1(inputBuf, outputBuf, blockSumsBuf, uniBuf)(groupsX, groupsY)

      if (numBlocks == 1) return outputBuf.also { blockSumsBuf.destroy(); uniBuf.destroy() }

      val scannedBlockSumsBuf = prefixSumGPU(blockSumsBuf, numBlocks)
      prefix_sum_p2(outputBuf, scannedBlockSumsBuf, uniBuf)(groupsX, groupsY)

      return outputBuf.also { scannedBlockSumsBuf.destroy(); blockSumsBuf.destroy(); uniBuf.destroy() }
    }

    suspend fun packMetadata(cfg: CFG, fsa: FSA): GPUBuffer {
      val t0 = TimeSource.Monotonic.markNow()
      val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toGPUBuffer()
      val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toGPUBuffer()
      log("Encoded grammar in ${t0.elapsedNow()}")

      val (reachBuf: GPUBuffer, entries: Int) = dag_reach.invokeDAGFixpoint(fsa)

      log("DAG fixpoint in ${t0.elapsedNow()}")
//    val (allFSAPairsFlattened, allFSAPairsOffsets) = //fsa.midpoints.prefixScan()
//        reachBuf.readInts().sparsifyReachabilityMatrix().prefixScan()
      //  TODO: enforce exact equivalence?
      val (allFSAPairsFlattened, allFSAPairsOffsets) = buildMidpointsGPU(fsa.numStates, reachBuf)
//      log("Flat midpoints in ${t0.elapsedNow()} : ${allFSAPairsFlattened.size} # ${allFSAPairsOffsets.size}")

      log("Sparse reachability took ${t0.elapsedNow()} / (${4 *(allFSAPairsFlattened.size + allFSAPairsOffsets.size)} bytes)")

      /** Memory layout: [CFL_STRUCT] */ val metaBuf = packStruct(
        constants = listOf(fsa.numStates, cfg.nonterminals.size),
        // FSA Encoding
        allFSAPairsFlattened, allFSAPairsOffsets, fsa.finalIdxs.toGPUBuffer(),
        // CFG Encoding
        grammarFlattened, grammarOffsets
      )

      log("Packed metadata in ${t0.elapsedNow()}")
      return metaBuf
    }

    suspend fun buildMidpointsGPU(states: Int, reachBuf: GPUBuffer): Pair<GPUBuffer, GPUBuffer> {
      val totalPairs = states * states
      val cntBuf     = GPUBuffer(totalPairs * 4, STCPSD)
      val uniBuf     = states.toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      mdpt_count(reachBuf, cntBuf, uniBuf)(states, states)
      val offBuf = prefixSumGPU(cntBuf, totalPairs)
      val last   = listOf(totalPairs - 1)
      val totalM = offBuf.readIndices(last)[0] + cntBuf.readIndices(last)[0]
      val flatBuf = GPUBuffer(totalM * 4, STCPSD)
      mdpt_write(reachBuf, offBuf, flatBuf, uniBuf)(states, states)

      return (flatBuf to offBuf).also { uniBuf.destroy(); cntBuf.destroy() }
    }

    suspend fun buildBackpointers(numStates: Int, numNTs: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer): Triple<GPUBuffer, GPUBuffer, GPUBuffer> {
      val totalCells = numStates * numStates * numNTs

      val bpCountBuf = GPUBuffer(totalCells * 4, STCPSD)

      log("Total cells: $totalCells = $numStates^2 * $numNTs")
      bp_count(dpIn, bpCountBuf, metaBuf)(numStates, numStates, numNTs)

//    val bpOffsetBuf = bpCountBuf.readInts().scan(0) { acc, arr -> acc + arr }.dropLast(1).toIntArray().toGPUBuffer(STCPSD)
      val bpOffsetBuf = prefixSumGPU(bpCountBuf, totalCells)

      val lastIdx = listOf(totalCells - 1)
      val totalExpansions = bpOffsetBuf.readIndices(lastIdx)[0] + bpCountBuf.readIndices(lastIdx)[0]
      log("Total expansions: $totalExpansions")

      val bpStorageBuf = GPUBuffer(totalExpansions * 2 * 4, STCPSD)

      bp_write(dpIn, bpOffsetBuf, bpStorageBuf, metaBuf)(numStates, numStates, numNTs)

      return Triple(bpCountBuf, bpOffsetBuf, bpStorageBuf)
    }

    fun buildLanguageSizeBuf(nStates: Int, nNT: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer, tmBuf: GPUBuffer): GPUBuffer {
      val totalCells = nStates * nStates * nNT
      val lsDenseBuf = GPUBuffer(totalCells * 4, STCPSD)

      for (span in 1..<nStates) {
        val spanBuf = span.toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

        ls_dense(dpIn, lsDenseBuf, metaBuf, tmBuf, spanBuf)(nStates - span, 1, nNT)
      }
      return lsDenseBuf
    }
  }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader
  suspend fun invokeCFLFixpoint(numStates: Int, numNTs: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer) {
    var t0 = TimeSource.Monotonic.markNow()

    var prevValue = -1

    for (round in 0..<numStates) {
      val changesBuf = 0.toGPUBuffer()
      cfl_mul_upper(dpIn, metaBuf, changesBuf)(numStates, numStates, numNTs)
//      log(dpIn.readInts().toLaTeX(numStates, numNTs))
      val changesThisRound = changesBuf.readInts()[0]
      changesBuf.destroy()
      if (changesThisRound == prevValue) break
      prevValue = changesThisRound
      log("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}, ⌈log(|Q|*|V|)⌉=${ceil(log2(numStates * numNTs.toDouble()))}")
      t0 = TimeSource.Monotonic.markNow()
    }
  }

  suspend fun invokeDAGFixpoint(fsa: FSA): Pair<GPUBuffer, Int> {
    val adjList = fsa.adjList
    val states = fsa.numStates
    val input = adjList.toSquareMatrixSparse(states)
//    var t0 = TimeSource.Monotonic.markNow()
    var prevValue = -1

    for (round in 0..<states) {
      val changesBuf = 0.toGPUBuffer()
      dag_reach(input, changesBuf)(states, states)
      val changesThisRound = changesBuf.readInts()[0]
      changesBuf.destroy()
      if (changesThisRound == prevValue) break
      prevValue = changesThisRound
//      log("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
//      t0 = TimeSource.Monotonic.markNow()
    }

    return input to prevValue
  }

  class DispatchStrategy(val gce: GPUCommandEncoder, val gcpe: GPUComputePassEncoder) {
    operator fun invoke(x: Int, y: Int = 1, z: Int = 1) {
      gcpe.dispatchWorkgroups(x, y, z)
      gcpe.end()
      gpu.queue.submit(arrayOf(gce.finish()))
    }
  }

  operator fun invoke(vararg inputs: GPUBuffer): DispatchStrategy =
    gpu.createCommandEncoder().let { gce ->
      gce.beginComputePass().let { gcpe ->
        gcpe.setPipeline(pipeline)
        gcpe.setBindGroup(0, pipeline.bindBuffers("$name.buffers", *inputs))
        return DispatchStrategy(gce, gcpe)
      }
    }
}

// constants   = [c0,c1,…]
// buffers[i]  = payload_i   (u32‑packed GPUBuffer)
// result      = [constants | (off0,len0) (off1,len1)… | payload_0 … payload_k ]
//                ^ headerInts.size * 4  bytes
fun packStruct(constants: List<Int> = emptyList(), vararg buffers: GPUBuffer): GPUBuffer {
  if (buffers.isEmpty()) error("At least one payload buffer required")

  // ── lengths & offsets (in *ints*, not bytes) ──────────────────────────────
  val lens     = buffers.map { (it.size / 4).toInt() }
  val offsets  = lens.runningFold(0) { acc, len -> acc + len }.dropLast(1)

  // ── build header ints ─────────────────────────────────────────────────────
  val headerInts = buildList {
    addAll(constants)
    for (i in lens.indices) { add(offsets[i]); add(lens[i]) }
  }

  val headerBytes  = headerInts.size * 4
  val payloadBytes = lens.sum()      * 4
  val totalBytes   = headerBytes + payloadBytes

  // ── allocate destination buffer ───────────────────────────────────────────
  val metaBuf = GPUBuffer(totalBytes, STCPSD)

  // ── upload header (one writeBuffer) ───────────────────────────────────────
  gpu.queue.writeBuffer(metaBuf, 0.0, Int32Array<ArrayBuffer>(headerInts.size).apply { set(headerInts.toTypedArray(), 0) })

  // ── stitch payloads in place with a single CommandEncoder ────────────────
  val enc = gpu.createCommandEncoder()
  for (i in buffers.indices) {
    val dstOffBytes = headerBytes + offsets[i] * 4
    enc.copyBufferToBuffer(buffers[i], 0.0, metaBuf, dstOffBytes.toDouble(), buffers[i].size)
  }

  gpu.queue.submit(arrayOf(enc.finish()))

  return metaBuf.also { buffers.forEach { it.destroy() } }
}

const val NEWLINE_ID = 1
const val BOS_ID     = 2
const val EOS_ID     = 3
const val FIRST_TID  = 4

fun tmToInt(tm: String, cfg: CFG): Int = when (tm) {
  "NEWLINE" -> NEWLINE_ID
  "BOS"     -> BOS_ID
  "EOS"     -> EOS_ID
  else      -> cfg.tmMap[tm]!! + FIRST_TID
}

const val SCALE = 10_000.0
fun Map<List<String>, Double>.toGpuHash(norm: Double = values.sum(), cfg: CFG): Map<List<UInt>, UInt> =
  mapValues { (_, p) -> (-ln(p / norm) * SCALE).roundToInt().coerceAtLeast(0).toUInt() }
    .mapKeys { (gram, _) -> gram.map { tmToInt(it, cfg).toUInt() } }

fun Map<List<UInt>, UInt>.loadToGPUBuffer(loadFactor: Double = 0.75): GPUBuffer {
  require(all { it.key.size == 4 }) { "Only 4-grams are supported" }

  val offset = FIRST_TID.toUInt()

  /** Compresses a 4‑gram (tokens are 1‑based) into one u32: 4 × 7‑bit fields. */
  fun packGram(g: List<UInt>): UInt =
    ((g[0] - offset) shl 21) or ((g[1] - offset) shl 14) or
        ((g[2] - offset) shl 7)  or  (g[3] - offset)

  /* ── pick a power‑of‑two table size ─────────────────────────────────── */
  val nEntries = size.coerceAtLeast(1)
  var pow = 1
  while ((1 shl pow) < (nEntries / loadFactor).roundToInt()) pow++
  val slots = 1u shl pow
  val mask  = slots - 1u

  val table = UIntArray(slots.toInt() * 2) { SENTINEL }

  /* ── insert with linear probing ─────────────────────────────────────── */
  for ((gram, score) in this) {
    val key  = packGram(gram)
    var slot = ((key * HASH_MUL) shr (32 - pow)) and mask

    while (table[(slot * 2u).toInt()] != SENTINEL) { slot = (slot + 1u) and mask }
    val idx = (slot * 2u).toInt()
    table[idx]     = key
    table[idx + 1] = score
  }

  /* ── prepend header (pow) and upload ────────────────────────────────── */
  val flat = UIntArray(1 + table.size)
  flat[0] = pow.toUInt()
  table.copyInto(flat, 1)

  log("Done")

  return flat.asList().toGPUBuffer()
}