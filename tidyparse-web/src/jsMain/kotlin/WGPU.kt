@file:OptIn(ExperimentalUnsignedTypes::class, ExperimentalStdlibApi::class)

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

lateinit var gpu: GPUDevice
var gpuAvailable = false
external val navigator: dynamic

/*
TODO:
  (1) Split words on WGPU
  (2) rescore samples using Markov Chain
  (3) sort repairs by probability on WGPU
  (4) parallelize makeLevFSA/byteFormat
*/

suspend fun tryBootstrappingGPU() {
  val tmpDev = (navigator.gpu as? GPU)?.requestAdapter()?.requestDevice()?.also { gpu = it }

  if (tmpDev != null) {
    gpu.addEventListener(EventType("uncapturederror"), { e: dynamic -> println("Uncaptured: ${e.error.message}") })
    try {
      listOf(
        prefix_sum_p1, prefix_sum_p2,      // ADT storage utils
        sparse_load, sparse_mat_load,      // Matrix loading utils
        dag_reach, mdpt_count, mdpt_write, // Graph reachability
        cfl_mul_upper,                     // Matrix exponentiation
        bp_count, bp_write,                // Backpointer addressing
        ls_dense, ls_cdf,                  // Language size estimation
        sample_words_wor, markov_score,    // Enumeration and reranking
        select_top_k, gather_top_k         // Top-k selection
      ).forEach { it.bind() }
//      benchmarkWGPU() // TODO: remove for deployment
//      benchmarkWGPURepair()
//      benchmarkReach()
    } catch (e: Exception) { e.printStackTrace(); return }

    gpuAvailable = true
    val obj = document.createElement("object").apply {
      setAttribute("type", "image/svg+xml")
      setAttribute("data", "/webgpu.svg")
      setAttribute("width", "35")
      setAttribute("height", "35")
    }

    (document.getElementById("gpuAvail") as HTMLDivElement).appendChild(obj)
  } else { println("not detected.") }
}

suspend fun repairCode(cfg: CFG, code: List<String>, ledBuffer: Int = Int.MAX_VALUE, ngramTensor: GPUBuffer? = null): List<List<String>> {
  val t0 = TimeSource.Monotonic.markNow()
  val fsa: FSA = makeLevFSA(code, 5)
  println("Made levFSA in ${t0.elapsedNow()}")

  val metaBuf = packMetadata(cfg, fsa)

  // Sparse index nonzero entries of the M_0 parse chart
  fun FSA.byteFormat(cfg: CFG): IntArray { // TODO: kernelize
    val t0 = TimeSource.Monotonic.markNow()
    val bindex = cfg.bindex
    val terminalLists = cfg.terminalLists

    // 0 and 1 are reserved for (0) no parse exists and (1) parse exists, but an internal nonterminal node
    // Other byte values are used to denote the presence (+) or absence (-) of a leaf terminal
    fun StrPred.predByte(A: Int): Int = (
      if (arg == "[.*]" || (arg.startsWith("[!=]") && arg.drop(4) !in terminalLists[A])) Int.MAX_VALUE - 1 // All possible terminals
      else if (arg.startsWith("[!=]")) (NEG_LITERAL.toInt() + (terminalLists[A].indexOf(arg.drop(4)) + 1).shl(1)) // Represent negation using sign bit
      else (terminalLists[A].indexOf(arg) + 1).shl(1)
    )

    fun buildSparseChart(cfg: CFG, nominalForm: NOM, stateMap: Map<String, Int>, bindex: Bindex<String>): IntArray {
      val rowCount = cfg.unitProductions.sumOf { (_, σ) -> nominalForm.flattenedTriples.count { arc -> arc.second(σ) } }

      val out = IntArray(rowCount * 4)

      var p = 0
      for ((A, σ) in cfg.unitProductions) {
        val Aidx = bindex[A]
        for ((q0, sp, q1) in nominalForm.flattenedTriples) {
          if (!sp(σ)) continue

          out[p++] = stateMap[q0]!!          // q0
          out[p++] = stateMap[q1]!!          // q1
          out[p++] = Aidx                    // non‑terminal
          out[p++] = sp.predByte(Aidx)   // terminal byte
        }
      }
      return out
    }

    val sparseChart = buildSparseChart(cfg, nominalForm, stateMap, bindex)
    println("Byte format took: ${t0.elapsedNow()}")
    return sparseChart
  }

  val dpInSparse = fsa.byteFormat(cfg)
//  println("Initial nonzeros: ${dpIn.count { it != 0 }}")

  println("PREPROCESSING TOOK: ${t0.elapsedNow()}") // ~230ms
  val words = repairPipeline(cfg, fsa, dpInSparse, metaBuf, ledBuffer, ngramTensor)
  println("Received: ${words.size} words")
//  val distinctWords = words.distinct()
//  println("Distinct: ${distinctWords.size} words")
  println("Round trip repair: ${t0.elapsedNow()}") // ~500ms

  return words
}

suspend fun repairPipeline(cfg: CFG, fsa: FSA, dpInSparse: IntArray, metaBuf: GPUBuffer, ledBuffer: Int, ngramTensor: GPUBuffer?): List<List<String>> {
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  println("FSA(|Q|=${numStates}, |δ|=${fsa.transit.size}), " +
      "CFG(|Σ|=${cfg.terminals.size}, |V|=${numNTs}, |P|=${cfg.nonterminalProductions.size})")
  val dpBuf = cfl_mul_upper.invokeCFLFixpoint(numStates, numNTs, dpInSparse, metaBuf)
  println("Matrix closure reached in: ${t0.elapsedNow()}")

  val t1 = TimeSource.Monotonic.markNow()
  val startNT     = cfg.bindex[START_SYMBOL]
  val allStartIds = fsa.finalIdxs.map { it * numNTs + startNT }
    .let { it.zip(dpBuf.readIndices(it)) }.filter { (_, v) -> v != 0 }.map { it.first }

  if (!allStartIds.isEmpty()) // { println("Valid parse found: dpComplete has ${allStartIds.size} start indices") }
  else { println("No valid parse found: dpComplete has no entries in final states!"); return emptyList() }

  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNTs, dpBuf, metaBuf)
  println("Built backpointers in ${t1.elapsedNow()}")

  val t2 = TimeSource.Monotonic.markNow()
  val statesToDist = allStartIds.map { it to fsa.idsToCoords[(it - startNT) / numNTs]!!.second }
  val led = statesToDist.minOf { it.second } // Language edit distance

  val startIdxs = statesToDist.filter { it.second in (led..(led + ledBuffer)) }
    .map { listOf(it.first, it.second) }.sortedBy { it[1] }.also { println("Start indices: $it") }.flatten()

  val maxRepairLen = fsa.width + fsa.height + 10

  if (MAX_WORD_LEN < maxRepairLen) {
    println("Max repair length exceeded $MAX_WORD_LEN ($maxRepairLen)")
    return emptyList()
  }

  val outBuf = GPUBuffer(MAX_SAMPLES * maxRepairLen * 4, GPUBufferUsage.STCPSD)

  val tmBuf = cfg.termBuf

  val lsDense  = buildLanguageSizeBuf(numStates, numNTs, dpBuf, metaBuf, tmBuf)
  val totalExp = bpStorageBuf.size.toInt() / (2 * 4)
  val cdfBuf = GPUBuffer(totalExp * 4, GPUBufferUsage.STCPSD)

  ls_cdf(dpBuf, lsDense, bpOffsetBuf, cdfBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)

  lsDense.destroy()

  val header = intArrayOf(0, maxRepairLen, numNTs, numStates)

  /** [TERM_STRUCT] */ val indexUniformsBuf = packStruct(constants = header.toList(), startIdxs.toGPUBuffer())
  println("Pairing function construction took: ${t2.elapsedNow()}")

  val t3 = TimeSource.Monotonic.markNow()
  sample_words_wor(dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf, outBuf, tmBuf, indexUniformsBuf, cdfBuf)(MAX_SAMPLES)

  val k = 10 * MAX_DISP_RESULTS
  val winnerTokens = scoreSelectGather(
    packets          = outBuf,
    ngramTensor      = ngramTensor ?: emptyMap<List<UInt>, UInt>().loadToGPUBuffer(),
    indexUniformsBuf = indexUniformsBuf,
    maxSamples       = MAX_SAMPLES,
    stride           = maxRepairLen,
    k                = k
  )

  listOf(outBuf, metaBuf, dpBuf, indexUniformsBuf, cdfBuf,
    bpCountBuf, bpOffsetBuf, bpStorageBuf).forEach(GPUBuffer::destroy)

  val t4 = TimeSource.Monotonic.markNow()
  val result = (0 until k).map { i -> winnerTokens.decodePacket(i, cfg.tmLst, maxRepairLen) }

  println("Decoded ${result.distinct().size} unique words out of ${result.size} in ${t4.elapsedNow()}")
  println("Sampling took ${t3.elapsedNow()}")
  return result
}

suspend fun scoreSelectGather(
  packets          : GPUBuffer,
  ngramTensor      : GPUBuffer,
  indexUniformsBuf : GPUBuffer,
  maxSamples       : Int,
  stride           : Int,
  k                : Int
): IntArray {
  var t0 = TimeSource.Monotonic.markNow()
  markov_score(packets, ngramTensor, indexUniformsBuf)(maxSamples)
  println("Score in ${t0.elapsedNow()}")

//  println(packets.readInts().toList().windowed(stride, stride)
//    .take(10).joinToString("\n") { it.joinToString(" ")})

  t0 = TimeSource.Monotonic.markNow()
  val prmBuf   = intArrayOf(maxSamples, k, stride).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
  val idxBuf   = IntArray(k) { Int.Companion.MAX_VALUE }.toGPUBuffer(GPUBufferUsage.STCPSD)
  val scrBuf   = IntArray(k) { Int.Companion.MAX_VALUE }.toGPUBuffer(GPUBufferUsage.STCPSD)

  val groups   = (maxSamples + 255) / 256
  select_top_k(prmBuf, packets, idxBuf, scrBuf)(groups)
  println("Select in ${t0.elapsedNow()}")

  t0 = TimeSource.Monotonic.markNow()
  val gatherPrm = intArrayOf(stride, k).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
  val bestBuf   = GPUBuffer(k * stride * 4, GPUBufferUsage.STCPSD)

  gather_top_k(gatherPrm, packets, idxBuf, bestBuf)(k)
  println("Gather in ${t0.elapsedNow()}")

  t0 = TimeSource.Monotonic.markNow()
  val topK = bestBuf.readInts()
  println("Read ${topK.size} = ${k}x${stride}x4 bytes in ${t0.elapsedNow()}")

  listOf(prmBuf, idxBuf, scrBuf, gatherPrm, bestBuf).forEach(GPUBuffer::destroy)
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
//    .also { println("Packing time: ${packTime.elapsedNow()}") }
}

//language=wgsl
const val TERM_STRUCT = """
struct Terminals { // Mappings from nonterminals to terminals in CFG
    nt_tm_lens_offset : u32,    nt_tm_lens_size : u32,
       offsets_offset : u32,       offsets_size : u32,
       all_tms_offset : u32,       all_tms_size : u32,
       
       payload : array<u32>
};

struct IndexUniforms {  // Indices of all accepting states in the parse chart
    targetCnt       : atomic<u32>,  // global counter (LFSR advances on host)
    maxWordLen      : u32,
    numNonterminals : u32,
    numStates       : u32,
    
    startIdxOffset  : u32, numStartIndices : u32,
    startIndices    : array<u32> // Contains alternating (1) start index and (2) edit distance
};"""

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
else { pairOffsetNext = cs.mdptsSize; }"""

//language=text
const val PREAMBLE = """
let r = gid.x;
let c = gid.y;
if (c <= r) { return; }
let A = gid.z;
$SHORT_PREAMBLE"""

//language=wgsl
val dag_reach by Shader("""
struct AtomicChange { count: atomic<u32> };
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
val mdpt_count by Shader("""
struct Uni { n : u32 };

@group(0) @binding(0) var<storage, read>       reach  : array<u32>;   // N×N upper‑tri (0/1)
@group(0) @binding(1) var<storage, read_write> counts : array<u32>;   // N×N  (aoi‑1 → #midpts)
@group(0) @binding(2) var<uniform>             uni    : Uni;

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
val mdpt_write by Shader("""
struct Uni { n : u32 };

@group(0) @binding(0) var<storage, read>       reach   : array<u32>;
@group(0) @binding(1) var<storage, read>       offsets : array<u32>; // exclusive scan of counts
@group(0) @binding(2) var<storage, read_write> flat_mp : array<u32>; // flattened mid‑points
@group(0) @binding(3) var<uniform>             uni     : Uni;

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

fn get_nt_tm_lens(index: u32) -> u32 { return terminals.payload[terminals.nt_tm_lens_offset + index]; }

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let r = gid.x;
    let c = r + su.span;
    if (c >= cs.numStates) { return; }
    let A = gid.z;
    
    $SHORT_PREAMBLE
    
    let val = dp_in[dpIdx];
    if (val == 0u) { return; }

    let hasLiteral = ((val >> 1u) != 0u);           // bit‑packed literal present?
    let negLit     = (val & ${NEG_LITERAL}u) != 0u;     // negative‑literal flag
    let litCount   =
        select(0u,
            select(1u,                                // positive literal ⇒ exactly 1
                    max(1u, get_nt_tm_lens(A) - 1u),  // negative ⇒ |Σ_A|‑1
                    negLit),
            hasLiteral);

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

fn get_nt_tm_lens(index: u32) -> u32 { return terminals.payload[terminals.nt_tm_lens_offset + index]; }

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $PREAMBLE

    let val = dp_in[dpIdx];
    if (val == 0u) { return; }

    var acc    : u32 = 0u;
    var outPos : u32 = bp_offset[dpIdx];
    
    let hasLiteral = ((val >> 1u) != 0u);           // bit‑packed literal present?
    let negLit     = (val & ${NEG_LITERAL}u) != 0u;     // negative‑literal flag 
    let litCount   = select(0u,
                            select(1u,                               // positive literal ⇒ exactly 1
                                    max(1u, get_nt_tm_lens(A) - 1u), // negative ⇒ |Σ_A|‑1
                                    negLit),
                            hasLiteral);

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

// language=wgsl
val prefix_sum_p1 by Shader("""
struct PrefixSumUni { n : u32 };

@group(0) @binding(0) var<storage, read>         inputBuf : array<u32>;
@group(0) @binding(1) var<storage, read_write>  outputBuf : array<u32>;
@group(0) @binding(2) var<storage, read_write>  blockSums : array<u32>;
@group(0) @binding(3) var<uniform>              prefixUni : PrefixSumUni;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> tile: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE) fn main(
    @builtin(global_invocation_id) globalId : vec3<u32>,
    @builtin(workgroup_id)          groupId : vec3<u32>,
    @builtin(local_invocation_id)   localId : vec3<u32>
) {
    let N       = prefixUni.n;
    let gid     = globalId.x;
    let lid     = localId.x;
    let grpId   = groupId.x;

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
        blockSums[grpId] = tile[WORKGROUP_SIZE - 1u];
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
val prefix_sum_p2 by Shader("""
struct PrefixSumUni { n: u32 };

@group(0) @binding(0) var<storage, read_write>          dataBuf : array<u32>;
@group(0) @binding(1) var<storage, read>       scannedBlockSums : array<u32>;
@group(0) @binding(2) var<uniform>                    prefixUni : PrefixSumUni;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id)         groupId  : vec3<u32>,
    @builtin(global_invocation_id) globalId : vec3<u32>,
) {
    let grpId = groupId.x;
    let gid   = globalId.x;
    let N     = prefixUni.n;

    // For each block `grpId`, the offset is scannedBlockSums[grpId].
    let offsetVal = scannedBlockSums[grpId];
    if (gid < N) { dataBuf[gid] = dataBuf[gid] + offsetVal; }
}""")

// Longest word WGSL can handle. If ~2^9<MAX_WORD_LEN, pipeline breaks some on architectures
const val MAX_WORD_LEN = 512
// Maximum threads WGSL allows in a single dispatch. If ~2^16<MAX_SAMPLES, this always fails
const val MAX_SAMPLES = 65_535
const val NEG_LITERAL = 0x40000000u //=1.shl(30)
// Length of the packet header in each repair buffer
const val PKT_HDR_LEN = 2 // [levenshtein distance, Markov probability]
const val MAX_SAMPLES_PER_DIST = 30_000
val SENTINEL = 0xFFFF_FFFFu
val HASH_MUL = 0x1e35a7bdu

/** See [PTree.sampleStrWithoutReplacement] for CPU version. */
//language=wgsl
val sample_words_wor by Shader("""$TERM_STRUCT
@group(0) @binding(0) var<storage, read>              dp_in : array<u32>;
@group(0) @binding(1) var<storage, read>           bp_count : array<u32>;
@group(0) @binding(2) var<storage, read>          bp_offset : array<u32>;
@group(0) @binding(3) var<storage, read>         bp_storage : array<u32>;
@group(0) @binding(4) var<storage, read_write> sampledWords : array<u32>;
@group(0) @binding(5) var<storage, read>          terminals : Terminals;
@group(0) @binding(6) var<storage, read_write>      idx_uni : IndexUniforms;
@group(0) @binding(7) var<storage, read>          ls_sparse : array<u32>;

/* ----------------------------- helpers ------------------------------------------ */
fn getStartIdx(i : u32) -> u32 { return idx_uni.startIndices[i * 2]; }
fn getEditDist(i : u32) -> u32 { return idx_uni.startIndices[i * 2 + 1]; }
fn get_nt_tm_lens(nt : u32) -> u32 { return terminals.payload[terminals.nt_tm_lens_offset + nt]; } // |Σ_A|
fn get_offsets(nt : u32) -> u32 { return terminals.payload[terminals.offsets_offset + nt]; } // offset of Σ_A
fn get_all_tms(i : u32) -> u32 { return terminals.payload[terminals.all_tms_offset + i]; }   // σ → TM‑id

fn binarySearchCDF(base: u32, len: u32, needle: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = len;
    loop {
        if (lo >= hi) { return base + lo; }
        let mid = (lo + hi) >> 1u;
        if (needle < ls_sparse[base + mid]) { hi = mid; } else { lo = mid + 1u; }
    }
}

/* ---------- size of the language rooted at any DP‑cell ------------------------- */
fn langSize(dpIdx : u32) -> u32 {
    /* literal domain */
    let val    = dp_in[dpIdx];
    let hasLit = ((val >> 1u) != 0u);
    let negLit = (val & ${NEG_LITERAL}u) != 0u;
    let litCnt =
        select(0u,
               select(1u,
                      max(1u, get_nt_tm_lens(dpIdx % idx_uni.numNonterminals) - 1u),
                      negLit),
               hasLit);

    /* expansion domain */
    let expCnt = bp_count[dpIdx];
    if (expCnt == 0u) { return litCnt; }

    let base   = bp_offset[dpIdx];
    let cdfLast = ls_sparse[base + expCnt - 1u];   // inclusive CDF
    return litCnt + cdfLast;
}

/* ---------- literal decoder ----------------------------------------------------- */
fn decodeLiteral(
    nt          : u32,   // non‑terminal
    litEnc      : u32,   // encoded literal (1‑based)
    negLit      : bool,  // negative‑literal flag
    variant     : u32,   // rank inside the literal domain
    word        : ptr<function, array<u32, $MAX_WORD_LEN>>,
    wordLen     : ptr<function, u32>
) {
    let numTms = get_nt_tm_lens(nt);
    let ntOff  = get_offsets(nt);

    if (negLit) { // choose any terminal ≠ (litEnc‑1)
        let excl = litEnc - 1u;
        let idx  = select(variant, variant + 1u, variant >= excl);
        (*word)[*wordLen] = get_all_tms(ntOff + idx) + 1u;
    } else {      // positive literal → single choice
        (*word)[*wordLen] = get_all_tms(ntOff + (litEnc - 1u)) + 1u;
    }
    *wordLen = *wordLen + 1u;
}

/* ---------- stack frame --------------------------------------------------------- */
struct Frame { dp : u32, rk : u32 };

fn lcg_permute(x : u32) -> u32 { return 1664525u * x + 1013904223u; }

fn lcg_rand(stateRef: ptr<function, u32>, range: u32) -> u32 { 
  let newVal = (1664525u * (*stateRef)) + 1013904223u;
  *stateRef = newVal;
  return select(newVal % range, 0u, range == 0u); 
}

fn min_u32(a: u32, b: u32) -> u32 { return select(a, b, a > b); }

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
//    /* ---- unique global rank ---------------------------------------------------- */
//    let seqId : u32 = atomicAdd(&idx_uni.targetCnt, 1u);
//    let gRank : u32 = lcg_permute(seqId + 0x9E3779B9u * gid.x);
//
//    let numStartIdxs = idx_uni.numStartIndices / 2u;
//    /* ---- total language size over all accepting states ------------------------- */
//    var total : u32 = 0u;
//    for (var i = 0u; i < numStartIdxs; i = i + 1u) { total = total + langSize(getStartIdx(i)); }
//    var rank  : u32 = gRank % total;
//
//    /* ---- pick a root in proportion to its language size ------------------------ */
//    var rootIdx : u32 = 0u;
//    var levDist : u32 = 0u;
//    for (var i = 0u; i < numStartIdxs; i = i + 1u) {
//        let sz = langSize(getStartIdx(i));
//        if (rank < sz) { rootIdx = getStartIdx(i); levDist = getEditDist(i); break; }
//        rank = rank - sz;
//    }

    /* ------------------------------------------------------------------ */
    /* 1. pre-scan to compute the *total* cross-bucket budget             */
    /*    (we need it to wrap gid.x when there are more threads than      */
    /*     total allowed samples).                                        */

    let numRoots = idx_uni.numStartIndices / 2u;

    var totalBudget : u32 = 0u;
    var cursor      : u32 = 0u;
    loop {
        if (cursor >= numRoots) { break; }

        let dist  = getEditDist(cursor);

        /* accumulate Σ|L(root)| for this distance bucket */
        var langSum : u32 = 0u;
        var j       : u32 = cursor;
        loop {
            langSum = langSum + langSize(getStartIdx(j));
            j       = j + 1u;
            if (j >= numRoots || getEditDist(j) != dist) { break; }
        }

        totalBudget = totalBudget + min_u32(langSum, ${MAX_SAMPLES_PER_DIST}u);
        cursor      = j;
    }

    /* If we launched more work-items than the global budget,              */
    /* just wrap around deterministically.                                 */
    let sampleId : u32 = gid.x % totalBudget;

    /* ------------------------------------------------------------------ */
    /* 2. second pass – locate the distance bucket that owns sampleId     */

    var prefix : u32 = 0u;       // sum of budgets of previous buckets
    var pos    : u32 = 0u;       // iterator over startIndices

    var rootIdx : u32 = 0u;
    var levDist : u32 = 0u;      // chosen edit distance
    var rank    : u32 = 0u;      // rank inside the chosen root’s language

    loop {
        /* ----- gather statistics for the current bucket ---------------- */
        let curDist = getEditDist(pos);

        var langSum  : u32 = 0u;
        var bucketEnd: u32 = pos;
        loop {  // walk until distance changes or we run out of roots
            langSum   = langSum + langSize(getStartIdx(bucketEnd));
            bucketEnd = bucketEnd + 1u;
            if (bucketEnd >= numRoots || getEditDist(bucketEnd) != curDist) { break; }
        }

        let bucketBudget : u32 = min_u32(langSum, ${MAX_SAMPLES_PER_DIST}u);

        /* ----- does sampleId fall into this bucket? -------------------- */
        if (sampleId < prefix + bucketBudget) {
            /* local index inside the current bucket                       */
            var localId : u32 = sampleId - prefix;          // 0 ≤ localId < bucketBudget

            /* map localId → rank ∈ [0 , langSum) without 64-bit math      */
            var rngState : u32 = lcg_permute(localId ^ curDist);
            var bucketRank : u32 = rngState % langSum;

            /* walk roots of this distance until we hit bucketRank         */
            var acc   : u32 = 0u;
            var p     : u32 = pos;
            loop {
                let sizeR = langSize(getStartIdx(p));
                if (bucketRank < acc + sizeR) {
                    rootIdx = getStartIdx(p);
                    levDist = curDist;
                    rank    = bucketRank - acc;
                    break;
                }
                acc = acc + sizeR;
                p   = p + 1u;
            }
            break;          // we’re done – exit outer loop
        }

        /* otherwise skip this bucket and continue                         */
        prefix = prefix + bucketBudget;
        pos    = bucketEnd;
    }
    
    /* ---- DFS stack ------------------------------------------------------------- */
    var stack : array<Frame, $MAX_WORD_LEN>;
    var top   : u32 = 0u;
    stack[top] = Frame(rootIdx, rank);   top++;

    var word  : array<u32, $MAX_WORD_LEN>;
    var wLen  : u32 = 0u;

    /* ---------------- depth‑first enumeration without replacement --------------- */
    loop {
        if (top == 0u) { break; }
        top = top - 1u;
        var fr     = stack[top];
        var dpIdx  = fr.dp;
        var rk     = fr.rk;

        /* --- literal vs expansion ---------------------------------------------- */
        let val      = dp_in[dpIdx];
        let hasLit   = ((val >> 1u) != 0u);
        let negLit   = (val & ${NEG_LITERAL}u) != 0u;
        let litCnt   =
            select(0u,
                   select(1u,                     // positive
                          max(1u, get_nt_tm_lens(dpIdx % idx_uni.numNonterminals) - 1u),
                          negLit),
                   hasLit);

        let expCnt   = bp_count[dpIdx];
        let base     = bp_offset[dpIdx];
        let totSize  = litCnt + select(0u, ls_sparse[base + expCnt - 1u], expCnt != 0u);

        rk = rk % totSize;                     // residual rank at this node

        /* --- literal branch ----------------------------------------------------- */
        if (rk < litCnt) {
            decodeLiteral(dpIdx % idx_uni.numNonterminals, (val >> 1u) & 0x1fffffffu, negLit, rk, &word, &wLen);
            continue;
        }
        rk = rk - litCnt;                      // shift into expansion domain

        /* --- expansion branch --------------------------------------------------- */
        let choiceIdx   = binarySearchCDF(base, expCnt, rk);
        let prevCDF     = select(0u, ls_sparse[choiceIdx - 1u], choiceIdx != base);
        let insidePair  = rk - prevCDF;

        let idxBM = bp_storage[2u*choiceIdx + 0u];
        let idxMC = bp_storage[2u*choiceIdx + 1u];

        let sizeC = langSize(idxMC);           // |L(C)|
        let rkB   = insidePair / sizeC;        // quotient  → rank for B
        let rkC   = insidePair % sizeC;        // remainder → rank for C

        /* push right child first so left child is processed first (DFS order) */
        stack[top] = Frame(idxMC, rkC); top++;
        stack[top] = Frame(idxBM, rkB); top++;
    }

    /* ---- write the resulting word to the output buffer ------------------------- */
    let outBase = gid.x * idx_uni.maxWordLen;
    sampledWords[outBase] = levDist;
    for (var i = 0u; i < wLen; i++) { sampledWords[outBase + i + ${PKT_HDR_LEN}u] = word[i]; }
}""")

//language=wgsl
val markov_score by Shader("""$TERM_STRUCT
@group(0) @binding(0) var<storage, read_write>  packets : array<u32>; // sampledWords/outBuf
@group(0) @binding(1) var<storage, read>          ngram : array<u32>; // hash table
@group(0) @binding(2) var<storage, read_write>  idx_uni : IndexUniforms;

const PKT_HDR_LEN  : u32 = ${PKT_HDR_LEN}u;
const SENTINEL_KEY : u32 = 0x${SENTINEL.toHexString()}u;
const HASH_MUL     : u32 = 0x${HASH_MUL.toHexString()}u;      // same multiplier as CPU side
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

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let sid     : u32 = gid.x;
    let stride  : u32 = idx_uni.maxWordLen;     // real packet length
    let base    : u32 = sid * stride;

    // prefix window <BOS><NEWLINE><BOS>  (order = 4 → need 3 prefixes)
    var t0 = BOS_ID - 1u;
    var t1 = NEWLINE_ID - 1u;
    var t2 = BOS_ID - 1u;

    var pos   : u32 = 0u;
    var score : u32 = 0u;
    var doneSuffix : u32 = 0u;

    loop {
        // ---------- fetch next token or synthesize suffix ----------------
        var tok : u32;
        if (pos < stride - PKT_HDR_LEN && packets[base + PKT_HDR_LEN + pos] != 0u) {
            tok = packets[base + PKT_HDR_LEN + pos];
            pos += 1u;
        } else {
            // two‑token suffix: NEWLINE , EOS
            tok = select(EOS_ID, NEWLINE_ID, doneSuffix == 0u);
            doneSuffix += 1u;
            if (doneSuffix > 2u) { break; }        // finished the suffix
        }

        // ---------- accumulate penalty -----------------------------------
        let key = packGram(t0, t1, t2, tok - 1u);
        score  += lookupScore(key);

        t0 = t1;  t1 = t2;  t2 = tok - 1u;
    }

    packets[base + 1u] = score * 100 + packets[base] * 1000;
}""")

//language=wgsl
val select_top_k by Shader("""
struct Params { n: u32, k: u32, stride: u32 };

@group(0) @binding(0) var<uniform>                  prm : Params;
@group(0) @binding(1) var<storage, read>        packets : array<u32>;
@group(0) @binding(2) var<storage, read_write>   topIdx : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> topScore : array<atomic<u32>>;

const UINT_MAX : u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= prm.n || prm.k == 0u) { return; }

    let score : u32 = packets[i * prm.stride + 1u];

    loop {
        var worstPos : u32 = 0u;
        var worstVal : u32 = atomicLoad(&topScore[0]);
        for (var j : u32 = 1u; j < prm.k; j = j + 1u) {
            let v = atomicLoad(&topScore[j]);
            if (v > worstVal) { worstVal = v; worstPos = j; }
        }

        if (score >= worstVal) { break; }
        let old = atomicCompareExchangeWeak(&topScore[worstPos], worstVal, score);
        if (old.exchanged) { atomicStore(&topIdx[worstPos], i); break; }
    }
}""")

//language=wgsl
val gather_top_k by Shader("""
struct Gather { stride: u32, k: u32 };

@group(0) @binding(0) var<uniform>                  g : Gather;
@group(0) @binding(1) var<storage, read>      packets : array<u32>;  // full outBuf
@group(0) @binding(2) var<storage, read>       topIdx : array<u32>;  // k indices
@group(0) @binding(3) var<storage, read_write> bestPk : array<u32>;  // compacted result

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let j : u32 = gid.x;
    if (j >= g.k) { return; }

    let srcIdx : u32 = topIdx[j];
    if (srcIdx == 0xFFFFFFFFu) { return; } 

    let stride : u32 = g.stride;
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
val sparse_mat_load by Shader("""
struct SparseElement { r: u32, c: u32 };

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

  suspend fun bind() {
    pipeline = try {
      gpu.createComputePipelineAsync(
        GPUComputePipelineDescriptor(
          layout = "auto",
          compute = GPUProgrammableStage(
            module = gpu.createShaderModule(GPUShaderModuleDescriptor(code = src)),
            entryPoint = "main"
          )
        )
      ).await()
    } catch (e: Throwable) { e.printStackTrace(); throw e }
  }

  operator fun getValue(tr: Any?, property: KProperty<*>): Shader = this.also { name = property.name }

  companion object {
    fun GPUComputePipeline.bindBuffers(vararg buffers: GPUBuffer): GPUBindGroup {
      inline fun <T> jsObject(init: dynamic.() -> Unit): T { val o = js("{}"); init(o); return o as T }
      val ent = buffers.mapIndexed { index, buf ->
        GPUBindGroupEntry(binding = index, resource = jsObject { buffer = buf })
      }.toTypedArray()
      return gpu.createBindGroup(GPUBindGroupDescriptor(layout = getBindGroupLayout(0), entries = ent))
    }

    suspend fun GPUBuffer.readInts(): IntArray {
//      val t0 = TimeSource.Monotonic.markNow()
      val readDst = GPUBuffer(size.toInt(), GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
      val cmd = gpu.createCommandEncoder()
      cmd.copyBufferToBuffer(source = this, sourceOffset = 0.0, destination = readDst, destinationOffset = 0.0, size = size)
      gpu.queue.submit(arrayOf(cmd.finish()))
      (readDst.mapAsync(1) as Promise<*>).await()
      val t = Int32Array(readDst.getMappedRange()).asList().toIntArray()
      readDst.destroy()
//      println("Read ${size.toInt()} bytes in ${t0.elapsedNow()}")
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
      (stagingBuffer.mapAsync(1) as Promise<*>).await()
      val t = Int32Array(stagingBuffer.getMappedRange())
        .asList().toIntArray().toList().also { stagingBuffer.destroy() }
      println("Read ${indices.size}/${size.toInt()} bytes in ${t0.elapsedNow()}")
      return t
    }

    fun IntArray.toGPUBufferSparse(usage: Int, totalSizeInInts: Int, rowCoeff: Int, colCoeff: Int): GPUBuffer {
      require(size % 4 == 0) { "Input array size must be a multiple of 4 for sparse data (r,c,v,i)." }
      require(totalSizeInInts > 0) { "totalSizeInInts must be positive." }

      val sparseDataGpuBuffer = toGPUBuffer()
      val outputByteSize = totalSizeInInts.toLong() * Int32Array.BYTES_PER_ELEMENT
      val outputBuffer = GPUBuffer(outputByteSize, usage or GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)
      val coeffsBuffer = intArrayOf(rowCoeff, colCoeff).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
      val numWorkgroups = ceil(size / 4.0 / SPARSE_WRITER_WORKGROUP_SIZE).toInt()

      sparse_load(sparseDataGpuBuffer, outputBuffer, coeffsBuffer)(numWorkgroups)

      sparseDataGpuBuffer.destroy()
      coeffsBuffer.destroy()
      return outputBuffer
    }

    fun IntArray.toSquareMatrixSparse(n: Int): GPUBuffer {
      val outputByteSize = n * n * Int32Array.BYTES_PER_ELEMENT
      val outputBuffer = GPUBuffer(outputByteSize, GPUBufferUsage.STCPSD)
      val sparseDataBuffer = toGPUBuffer()
      val numWorkgroups = ceil((size / 2.0) / SPARSE_WRITER_WORKGROUP_SIZE).toInt()
      sparse_mat_load(sparseDataBuffer, outputBuffer)(numWorkgroups)
      sparseDataBuffer.destroy()
      return outputBuffer
    }

    fun List<Int>.toGPUBuffer(usage: Int = GPUBufferUsage.STCPSD): GPUBuffer = toTypedArray().toGPUBuffer(usage)
    fun List<UInt>.toGPUBuffer(usage: Int = GPUBufferUsage.STCPSD): GPUBuffer = map { it.toInt() }.toTypedArray().toGPUBuffer(usage)
    fun IntArray.toGPUBuffer(usage: Int = GPUBufferUsage.STCPSD): GPUBuffer = toTypedArray().toGPUBuffer(usage)
    fun Int.toGPUBuffer(usage: Int = GPUBufferUsage.STCPSD): GPUBuffer = intArrayOf(this).toGPUBuffer(usage)
    fun Array<Int>.toGPUBuffer(usage: Int = GPUBufferUsage.STCPSD): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@toGPUBuffer, 0) }
        .let { GPUBuffer(byteSize = size * 4, us = usage, data = it) }

    // TODO: figure out map/unmap lifetime?
    fun GPUBuffer(byteSize: Number, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
      gpu.createBuffer(descriptor = GPUBufferDescriptor(size = byteSize.toDouble(), usage = us))
        .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }

    // Define the workgroup size consistently (must match WGSL)
    const val PREFIX_SUM_WORKGROUP_SIZE = 256

    suspend fun prefixSumGPU(inputBuf: GPUBuffer, length: Int): GPUBuffer {
      val numGroups = (length + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE

      val outputBuf = GPUBuffer(inputBuf.size.toInt(), GPUBufferUsage.STCPSD)
      val blockSumsBuf = GPUBuffer(numGroups * 4, GPUBufferUsage.STCPSD)
      val uniBuf = intArrayOf(length).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      prefix_sum_p1(inputBuf, outputBuf, blockSumsBuf, uniBuf)(numGroups)

      if (numGroups > 1) {
        val scannedBlockSumsBuf = blockSumsBuf.readInts().scan(0) { acc, it -> acc + it }.toGPUBuffer()
        prefix_sum_p2(outputBuf, scannedBlockSumsBuf, uniBuf)(numGroups)
        scannedBlockSumsBuf.destroy()
      }

      uniBuf.destroy()
      return outputBuf
    }

    suspend fun packMetadata(cfg: CFG, fsa: FSA): GPUBuffer {
      val t0 = TimeSource.Monotonic.markNow()
      val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toGPUBuffer()
      val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toGPUBuffer()
      println("Encoded grammar in ${t0.elapsedNow()}")

      val (reachBuf: GPUBuffer, entries: Int) = dag_reach.invokeDAGFixpoint(fsa)

      println("DAG fixpoint in ${t0.elapsedNow()}")
//    val (allFSAPairsFlattened, allFSAPairsOffsets) = //fsa.midpoints.prefixScan()
//        reachBuf.readInts().sparsifyReachabilityMatrix().prefixScan()
      //  TODO: enforce exact equivalence?
      val (allFSAPairsFlattened, allFSAPairsOffsets) = buildMidpointsGPU(fsa.numStates, reachBuf)
//      println("Flat midpoints in ${t0.elapsedNow()} : ${allFSAPairsFlattened.size} # ${allFSAPairsOffsets.size}")

      println("Sparse reachability took ${t0.elapsedNow()} / (${4 *(allFSAPairsFlattened.size + allFSAPairsOffsets.size)} bytes)")

      /** Memory layout: [CFL_STRUCT] */ val metaBuf = packStruct(
        constants = listOf(fsa.numStates, cfg.nonterminals.size),
        // FSA Encoding
        allFSAPairsFlattened, allFSAPairsOffsets, fsa.finalIdxs.toGPUBuffer(),
        // CFG Encoding
        grammarFlattened, grammarOffsets
      )

      println("Packed metadata in ${t0.elapsedNow()}")
      return metaBuf
    }

    suspend fun buildMidpointsGPU(states: Int, reachBuf: GPUBuffer): Pair<GPUBuffer, GPUBuffer> {
      val totalPairs = states * states
      val cntBuf     = GPUBuffer(totalPairs * 4, GPUBufferUsage.STCPSD)
      val uniBuf     = states.toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      mdpt_count(reachBuf, cntBuf, uniBuf)(states, states)
      val offBuf = prefixSumGPU(cntBuf, totalPairs)
      val last   = listOf(totalPairs - 1)
      val totalM = offBuf.readIndices(last)[0] + cntBuf.readIndices(last)[0]
      val flatBuf = GPUBuffer(totalM * 4, GPUBufferUsage.STCPSD)
      mdpt_write(reachBuf, offBuf, flatBuf, uniBuf)(states, states)

      uniBuf.destroy()
      cntBuf.destroy()
      return flatBuf to offBuf
    }

    suspend fun buildBackpointers(numStates: Int, numNTs: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer): Triple<GPUBuffer, GPUBuffer, GPUBuffer> {
      val totalCells = numStates * numStates * numNTs

      val bpCountBuf = GPUBuffer(totalCells * 4, GPUBufferUsage.STCPSD)

      println("Total cells: $totalCells = $numStates^2 * $numNTs")
      bp_count(dpIn, bpCountBuf, metaBuf)(numStates, numStates, numNTs)

//    val bpOffsetBuf = bpCountBuf.readInts().scan(0) { acc, arr -> acc + arr }.dropLast(1).toIntArray().toGPUBuffer(GPUBufferUsage.STCPSD)
      val bpOffsetBuf = prefixSumGPU(bpCountBuf, totalCells)

      val lastIdx = listOf(totalCells - 1)
      val totalExpansions = bpOffsetBuf.readIndices(lastIdx)[0] + bpCountBuf.readIndices(lastIdx)[0]
      println("Total expansions: $totalExpansions")

      val bpStorageBuf = GPUBuffer(totalExpansions * 2 * 4, GPUBufferUsage.STCPSD)

      bp_write(dpIn, bpOffsetBuf, bpStorageBuf, metaBuf)(numStates, numStates, numNTs)

      return Triple(bpCountBuf, bpOffsetBuf, bpStorageBuf)
    }

    fun buildLanguageSizeBuf(nStates: Int, nNT: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer, tmBuf: GPUBuffer): GPUBuffer {
      val totalCells = nStates * nStates * nNT
      val lsDenseBuf = GPUBuffer(totalCells * 4, GPUBufferUsage.STCPSD)

      for (span in 1..<nStates) {
        val spanBuf = span.toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

        ls_dense(dpIn, lsDenseBuf, metaBuf, tmBuf, spanBuf)(nStates - span, 1, nNT)
      }
      return lsDenseBuf
    }
  }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader
  suspend fun invokeCFLFixpoint(numStates: Int, numNTs: Int, input: IntArray, metaBuf: GPUBuffer): GPUBuffer {
//    var t0 = TimeSource.Monotonic.markNow()

    val rowCoeff = numStates * numNTs
    val colCoeff = numNTs
    val dpIn = input.toGPUBufferSparse(GPUBufferUsage.STCPSD, numStates * rowCoeff, rowCoeff, colCoeff)
//    println("Time to load buffer: ${t0.elapsedNow()} (${input.size * 4} bytes)")

    var prevValue = -1

    for (round in 0..<numStates) {
      val changesBuf = 0.toGPUBuffer()
      cfl_mul_upper(dpIn, metaBuf, changesBuf)(numStates, numStates, numNTs)
      val changesThisRound = changesBuf.readInts()[0]
      changesBuf.destroy()
      if (changesThisRound == prevValue) break
      prevValue = changesThisRound
//      println("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
//      t0 = TimeSource.Monotonic.markNow()
    }

    return dpIn
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
//      println("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
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
        gcpe.setBindGroup(0, pipeline.bindBuffers(*inputs))
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
  val metaBuf = GPUBuffer(totalBytes, GPUBufferUsage.STCPSD)

  // ── upload header (one writeBuffer) ───────────────────────────────────────
  gpu.queue.writeBuffer(metaBuf, 0.0, Int32Array<ArrayBuffer>(headerInts.size).apply { set(headerInts.toTypedArray(), 0) })

  // ── stitch payloads in place with a single CommandEncoder ────────────────
  val enc = gpu.createCommandEncoder()
  for (i in buffers.indices) {
    val dstOffBytes = headerBytes + offsets[i] * 4
    enc.copyBufferToBuffer(buffers[i], 0.0, metaBuf, dstOffBytes.toDouble(), buffers[i].size)
  }

  gpu.queue.submit(arrayOf(enc.finish()))
  buffers.forEach { it.destroy() }

  return metaBuf
}

fun Map<List<UInt>, UInt>.loadToGPUBuffer(loadFactor: Double = 0.75): GPUBuffer {
  require(all { it.key.size == 4 }) { "Only 4-grams are supported" }

  /** Compresses a 4‑gram (tokens are 1‑based) into one u32: 4 × 7‑bit fields. */
  fun packGram(g: List<UInt>): UInt =
    ((g[0] - 1u) shl 21) or ((g[1] - 1u) shl 14) or
        ((g[2] - 1u) shl 7)  or  (g[3] - 1u)

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
    val idx = (slot * 2u).toInt()            // correct slot found
    table[idx]     = key
    table[idx + 1] = score
  }

  /* ── prepend header (pow) and upload ─────────────────────────────────── */
  val flat = UIntArray(1 + table.size)
  flat[0] = pow.toUInt()
  table.copyInto(flat, 1)

  println("Done")

  return flat.asList().toGPUBuffer()         // unchanged helper
}

private fun IntArray.decodePacket(sampleIdx: Int, tm: List<String>, wordLen: Int): List<String> {
  val words = mutableListOf<StringBuilder>()
  var cur: StringBuilder? = null
  val base = sampleIdx * wordLen + PKT_HDR_LEN   // skip BOTH header cells

  for (j in 0 until wordLen - PKT_HDR_LEN) {
    val tok = this[base + j] and 0xFF
    if (tok == 0) {
      if (cur != null && cur.isNotEmpty()) { words += cur; cur = null }
    } else {
      if (cur == null) cur = StringBuilder()
      if (cur.isNotEmpty()) cur.append(' ')
      cur.append(tm[tok - 1])
    }
  }
  if (cur != null && cur.isNotEmpty()) words += cur
  return words.map(StringBuilder::toString)
}