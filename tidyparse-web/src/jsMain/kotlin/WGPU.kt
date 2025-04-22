@file:OptIn(ExperimentalUnsignedTypes::class)

import Shader.Companion.GPUBuffer
import Shader.Companion.buildLanguageSizeBuf
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.readInts
import Shader.Companion.toGPUBuffer
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.types.cache
import js.array.asList
import js.buffer.*
import js.typedarrays.Int32Array
import kotlinx.browser.document
import kotlinx.coroutines.await
import kotlinx.dom.appendText
import org.w3c.dom.HTMLDivElement
import web.events.*
import web.gpu.*
import kotlin.js.Promise
import kotlin.math.ceil
import kotlin.reflect.KProperty
import kotlin.time.TimeSource

lateinit var gpu: GPUDevice
var gpuAvailable = false
external val navigator: dynamic

/*
TODO:
  (1) rescore samples using Markov Chain
  (2) parallelize makeLevFSA/byteFormat
 */

suspend fun tryBootstrappingGPU() {
  print("Checking GPU availability... ")
  val tmpDev = (navigator.gpu as? GPU)?.requestAdapter()?.requestDevice()?.also { gpu = it }
  val gpuAvailDiv = document.getElementById("gpuAvail") as HTMLDivElement

  if (tmpDev != null) {
    println("detected.")
    val obj = document.createElement("object").apply {
      setAttribute("type", "image/svg+xml")
      setAttribute("data", "/webgpu.svg")
      setAttribute("width", "35")
      setAttribute("height", "35")
    }
    gpuAvailDiv.appendChild(obj)
    gpu.addEventListener(EventType("uncapturederror"), { e: dynamic -> println("Uncaptured: ${e.error.message}") })
    try {
      listOf(
        sparse_load, sparse_mat_load,
        dag_reach, mdpt_count, mdpt_write,
        cfl_mul_upper, prefix_sum_p1, prefix_sum_p2,
        bp_count, bp_write, ls_dense, ls_cdf, sample_words
      ).forEach { it.bind() }
//      benchmarkWGPU() // TODO: remove for deployment
      benchmarkWGPURepair()
//      benchmarkReach()
    } catch (e: Exception) { e.printStackTrace(); return }

    gpuAvailable = true
  } else {
    println("not detected.")
    gpuAvailDiv.appendText("WebGPU is NOT available.")
  }
}

suspend fun repairCode(cfg: CFG, code: List<String>, ledBuffer: Int = Int.MAX_VALUE): List<List<String>> {
  val t0 = TimeSource.Monotonic.markNow()
  val levFSA: FSA = makeLevFSA(code, 5)
  println("Made levFSA in ${t0.elapsedNow()}")

  val metadata = packMetadata(cfg, levFSA)

  // Sparse index nonzero entries of the M_0 parse chart
  fun FSA.byteFormat(cfg: CFG): IntArray { // TODO: kernelize
    val t0 = TimeSource.Monotonic.markNow()
    val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it] ?: emptySet() }
    // 0 and 1 are reserved for (0) no parse exists and (1) parse exists, but an internal nonterminal node
    // Other byte values are used to denote the presence (+) or absence (-) of a leaf terminal
    fun StrPred.predByte(A: Int): Int = (
      if (arg == "[.*]" || (arg.startsWith("[!=]") && arg.drop(4) !in terminalLists[A])) Int.MAX_VALUE - 1 // All possible terminals
      else if (arg.startsWith("[!=]")) (1.shl(30) + (terminalLists[A].indexOf(arg.drop(4)) + 1).shl(1)) // Represent negation using sign bit
      else (terminalLists[A].indexOf(arg) + 1).shl(1)
    )

    val sparseChart = cfg.unitProductions.flatMap { (A, σ) ->
      nominalForm.flattenedTriples.filter { arc -> arc.second(σ) }.map { (q0, sp, q1) ->
        val Aidx = cfg.bindex[A]
        // row, col, Aidx, terminal
        listOf(stateMap[q0]!!, stateMap[q1]!!, Aidx, sp.predByte(Aidx))
//          .also { println("${it[0]}, ${it[1]}, ${it[2]}, ${it[3].toString(2)}") }
      }
    }

    val q = sparseChart.flatten().toIntArray()
    println("Byte format took: ${t0.elapsedNow()}")
    return q
  }

  val dpIn = levFSA.byteFormat(cfg)
//  println("Initial nonzeros: ${dpIn.count { it != 0 }}")

  println("PREPROCESSING TOOK: ${t0.elapsedNow()}") // ~230ms
  val words = repairPipeline(cfg, levFSA, dpIn, metadata, ledBuffer)
  println("Received: ${words.size} words")
  println("Round trip repair: ${t0.elapsedNow()}") // ~500ms

  return words
}

suspend fun repairPipeline(cfg: CFG, fsa: FSA, dpInSparse: IntArray, metaBuf: GPUBuffer, ledBuffer: Int): List<List<String>> {
//  println("FSA(|Q|=${fsa.numStates}, |δ|=${fsa.transit.size}), " +
//      "CFG(|Σ|=${cfg.terminals.size}, |V|=${cfg.nonterminals.size}, |P|=${cfg.nonterminalProductions.size})")
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNonterminals) = fsa.numStates to cfg.nonterminals.size
  val dpBuf = cfl_mul_upper.invokeCFLFixpoint(numStates, numNonterminals, dpInSparse, metaBuf)
  println("Matrix closure reached in: ${t0.elapsedNow()}")

  val t1 = TimeSource.Monotonic.markNow()
  val startNT     = cfg.bindex[START_SYMBOL]
  val allStartIds = fsa.finalIdxs.map { it * numNonterminals + startNT }
    .let { it.zip(dpBuf.readIndices(it)) }.filter { (_, v) -> v != 0 }.map { it.first }

  if (!allStartIds.isEmpty()) { println("Valid parse found: dpComplete has ${allStartIds.size} start indices") }
  else { println("No valid parse found: dpComplete has no entries in final states!"); return emptyList() }

  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNonterminals, dpBuf, metaBuf)
  println("Built backpointers in ${t1.elapsedNow()}")

  val t2 = TimeSource.Monotonic.markNow()
  val distToStates = allStartIds.map { it to fsa.idsToCoords[(it - startNT) / numNonterminals]!!.second }
  val led = distToStates.minOf { it.second } // Language edit distance
  val startIdxs = distToStates.filter { it.second in (led..(led + ledBuffer)) }
    .also { println("Start indices: $it") }.map { it.first }.toIntArray()

  val maxSamples = 1000
  val maxWordLen = fsa.width + fsa.height + 10

  val outBuf = GPUBuffer(maxSamples * maxWordLen * 4, GPUBufferUsage.STCPSD)

  val tmBuf = cfg.termBuf

  /* Phase 1 – language size + CDF  */
  val lsDense  = buildLanguageSizeBuf(fsa.numStates, cfg.nonterminals.size, dpBuf, metaBuf, tmBuf)
  val totalExp = bpStorageBuf.size.toInt() / (2 * 4)
  val cdfBuf   = GPUBuffer(totalExp * 4, GPUBufferUsage.STCPSD)

  ls_cdf.invoke3d(fsa.numStates,cfg.nonterminals.size, dpBuf, lsDense, bpOffsetBuf, cdfBuf, metaBuf, tmBuf)

  val header = intArrayOf(0, maxWordLen, cfg.nonterminals.size, startIdxs.size)

  /** [TERM_STRUCT] */
  val indexUniformsBuf = packStruct(constants = header.toList(), startIdxs.toGPUBuffer())
  println("Pairing function construction took : ${t2.elapsedNow()}")

  /* Phase 3 – launch |maxSamples| single‑thread workgroups */
  sample_words.invoke1d(
    maxSamples,
    dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf, outBuf, tmBuf, indexUniformsBuf, cdfBuf
  )

  val tokens = outBuf.readInts()
  listOf(outBuf, metaBuf, dpBuf, indexUniformsBuf, cdfBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf).forEach { it.destroy() }
  return tokens.splitIntoWords(cfg, maxSamples, maxWordLen)
}

val CFG.termBuf by cache {
  val packTime = TimeSource.Monotonic.markNow()
  val terminalLists = nonterminals.map { bimap.UNITS[it]?.map { tmMap[it]!! } ?: emptyList() }
  val nt_tm_lens = terminalLists.map { it.size }.toGPUBuffer()
  val nt_tm_offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1).toGPUBuffer()
  val all_tm = terminalLists.flatten().toGPUBuffer()

  /** Memory layout: [TERM_STRUCT] */
  packStruct(emptyList(), nt_tm_lens, nt_tm_offsets, all_tm)
    .also { println("Packing time: ${packTime.elapsedNow()}") }
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
    numStartIndices : u32,
    payload         : array<u32>
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
struct Uni { N : u32 };

@group(0) @binding(0) var<storage, read>       reach  : array<u32>;   // N×N upper‑tri (0/1)
@group(0) @binding(1) var<storage, read_write> counts : array<u32>;   // N×N  (aoi‑1 → #midpts)
@group(0) @binding(2) var<uniform>             uni    : Uni;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let r = gid.y;  let c = gid.x;  let N = uni.N;
    if (r >= N || c >= N || c <= r) { return; }

    let idx = r*N + c;
    if (reach[idx]==0u) { counts[idx]=0u; return; }

    var cnt = 0u;
    for (var v=0u; v<N; v++) { if (reach[r*N+v]==1u && reach[v*N+c]==1u) { cnt++; } }
  counts[idx] = cnt;
}""")

//language=wgsl
val mdpt_write by Shader("""
struct Uni { N : u32 };

@group(0) @binding(0) var<storage, read>       reach   : array<u32>;
@group(0) @binding(1) var<storage, read>       offsets : array<u32>; // exclusive scan of counts
@group(0) @binding(2) var<storage, read_write> flat_mp : array<u32>; // flattened mid‑points
@group(0) @binding(3) var<uniform>             uni     : Uni;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let r = gid.y;  let c = gid.x;  let N = uni.N;
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
    let negLit     = (val & 0x40000000u) != 0u;     // negative‑literal flag
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
    let negLit     = (val & 0x40000000u) != 0u;     // negative‑literal flag 
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
    
    ls_sparse[outPos] = acc + litCount;
}""")

// language=wgsl
val prefix_sum_p1 by Shader("""
struct PrefixSumUni { N: u32 };

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
    let N       = prefixUni.N;
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
struct PrefixSumUni { N: u32 };

@group(0) @binding(0) var<storage, read_write>          dataBuf : array<u32>;
@group(0) @binding(1) var<storage, read>       scannedBlockSums : array<u32>;
@group(0) @binding(2) var<uniform>                    prefixUni : PrefixSumUni;

@compute @workgroup_size(256) fn main(
    @builtin(workgroup_id)         groupId  : vec3<u32>,
    @builtin(global_invocation_id) globalId : vec3<u32>,
) {
    let grpId = groupId.x;
    let gid   = globalId.x;
    let N     = prefixUni.N;

    // For each block `grpId`, the offset is scannedBlockSums[grpId].
    let offsetVal = scannedBlockSums[grpId];
    if (gid < N) { dataBuf[gid] = dataBuf[gid] + offsetVal; }
}""")

//language=wgsl
val sample_words by Shader("""$TERM_STRUCT
@group(0) @binding(0) var<storage, read>              dp_in : array<u32>;
@group(0) @binding(1) var<storage, read>           bp_count : array<u32>;
@group(0) @binding(2) var<storage, read>          bp_offset : array<u32>;
@group(0) @binding(3) var<storage, read>         bp_storage : array<u32>;
@group(0) @binding(4) var<storage, read_write> sampledWords : array<u32>;
@group(0) @binding(5) var<storage, read>          terminals : Terminals;
@group(0) @binding(6) var<storage, read_write>      idx_uni : IndexUniforms;
@group(0) @binding(7) var<storage, read>          ls_sparse : array<u32>;

fn binarySearchCDF(base: u32, len: u32, needle: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = len;
    loop {
        let mid = (lo+hi) >> 1u;
        if (mid == hi || mid == lo) { return base + mid; }
        let v = ls_sparse[base + mid];
        if (needle < v) { hi = mid; } else { lo = mid; }
    }
}

fn getStartIdx(i : u32) -> u32 { return idx_uni.payload[i]; }

fn get_nt_tm_lens(index: u32) -> u32 { return terminals.payload[terminals.nt_tm_lens_offset + index]; }
fn get_offsets(index: u32) -> u32 { return terminals.payload[terminals.offsets_offset + index]; }
fn get_all_tms(index: u32) -> u32 { return terminals.payload[terminals.all_tms_offset + index]; }

fn lcg_rand(stateRef: ptr<function, u32>, range: u32) -> u32 { 
  let newVal = (1664525u * (*stateRef)) + 1013904223u;
  *stateRef = newVal;
  return select(newVal % range, 0u, range == 0u); 
}

fn decodeLiteral(
    nonterminal    : u32,
    literalEncoded : u32,
    isNegative     : bool,
    variant        : u32,      // 0‑based selector inside literal domain
    localWord      : ptr<function, array<u32, 1024>>,
    wordLen        : ptr<function, u32>
) {
    let numTms  = get_nt_tm_lens(nonterminal);
    let ntOff   = get_offsets(nonterminal);

    if (isNegative) {                 // choose any terminal ≠ literalEncoded‑1
        let excl = literalEncoded - 1u;
        let idx  = select(variant, variant + 1u, variant >= excl);
        (*localWord)[*wordLen] = get_all_tms(ntOff + idx) + 1u;
    } else {                          // positive literal (only one variant)
        (*localWord)[*wordLen] = get_all_tms(ntOff + (literalEncoded - 1u)) + 1u;
    }
    *wordLen = *wordLen + 1u;
}

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let globalIdx   = atomicAdd(&idx_uni.targetCnt, 1u);
//  let totLangSize  = getCdf(idx_uni.numStartIndices - 1u);
//  var localRank    = globalIdx % totLangSize;
//
//  /* binary search the inclusive CDF */
//  var lo : u32 = 0u;
//  var hi : u32 = idx_uni.numStartIndices;
//  loop {
//      let mid = (lo + hi) >> 1u;
//      if (mid == hi || mid == lo) { break; }
//      if (localRank < getCdf(mid)) { hi = mid; } else { lo = mid; }
//  }
//
//  let rootIdx   = getStartIdx(lo);
//  let prevCdf   = select(0u, getCdf(lo - 1u), lo != 0u);
//  localRank     = localRank - prevCdf;       // 0‑based rank **inside** that root

    let rootIdx = getStartIdx(globalIdx % idx_uni.numStartIndices);

    var stack  : array<u32, 1024>;
    var top    : u32 = 0u;
    var word   : array<u32, 1024>;
    var wLen   : u32 = 0u;

    stack[top] = rootIdx;      top++;

    while (top > 0u) {
        top -= 1u;
        let dpIdx = stack[top];
        let val   = dp_in[dpIdx];
        let hasLit  = ((val >> 1u) != 0u);
        let negLit  = (val & 0x40000000u) != 0u;
        let litCnt  =
            select(0u,
                   select(1u,                     // positive
                          max(1u, get_nt_tm_lens(dpIdx % idx_uni.numNonterminals) - 1u),
                          negLit),
                   hasLit);

        // -------- local index inside this node's language -------------
        // compute the *total* language size of this node
        let base     = bp_offset[dpIdx];
        let expCnt   = bp_count[dpIdx];
        let totSize  = select(litCnt,                        // no expansions? use litCnt
                              ls_sparse[base + expCnt - 1u], // otherwise last CDF entry
                              expCnt != 0u);

        // turn the global counter into a local index
        var localIdx = globalIdx % totSize;

        // --- literal branch?
        if (localIdx < litCnt) {
            decodeLiteral(dpIdx % idx_uni.numNonterminals, (val >> 1u) & 0x1fffffffu, negLit, localIdx, &word, &wLen);
            continue;
        }
        localIdx = localIdx - litCnt;       // shift into expansion domain

        // --- expansion branch (binary search the pre‑shifted CDF) ------
        let choice = binarySearchCDF(base, bp_count[dpIdx], localIdx);

        let idxBM = bp_storage[2u*choice + 0u];
        let idxMC = bp_storage[2u*choice + 1u];

        stack[top] = idxMC;  top++;
        stack[top] = idxBM;  top++;
    }

    let outBase = gid.x * idx_uni.maxWordLen;
    for (var i=0u; i<wLen; i++) { sampledWords[outBase+i] = word[i]; }
}

fn sampleTopDown(
    dp_in_ptr       : ptr<storage, array<u32>, read>,
    bp_count_ptr    : ptr<storage, array<u32>, read>,
    bp_offset_ptr   : ptr<storage, array<u32>, read>,
    bp_storage_ptr  : ptr<storage, array<u32>, read>,
    startDPIdx      : u32,
    rngStateRef     : ptr<function, u32>,
    localWord       : ptr<function, array<u32, 1024>>,
    maxWordLen      : u32,
    nnt             : u32
) {
    let MAX_STACK = 1024u;
    var stack: array<u32, 1024>;
    var top = 0u;
    var wordLen = 0u;
    stack[top] = startDPIdx; top++;

    // Safety limit: "maxWordLen * 98" matches the Metal logic
    let ITER_MAX = maxWordLen * 98u;

    // Loop while we have items on the stack and haven't overflowed localWord
    for (var iter = 0u; (iter < ITER_MAX) && (top > 0u) && (wordLen < (maxWordLen - 5u)); iter++) {
        top -= 1u; let dpIdx = stack[top];

        let expCount = (*bp_count_ptr)[dpIdx];
        let dpVal    = (*dp_in_ptr)[dpIdx];

        if (((dpVal >> 1) != 0u) && (((dpVal & 0x01u) == 0u) || ((lcg_rand(rngStateRef, expCount) % 2u) == 0u))) {
            let nonterminal       = dpIdx % nnt;
            let isNegativeLiteral = (dpVal & 0x40000000u) != 0u;
            let literal           = (dpVal >> 1) & 0x1FFFFFFFu;

            // Look up how many terminals exist for this nonterminal, etc.
            let numTms    = get_nt_tm_lens(nonterminal);
            let ntOffset  = get_offsets(nonterminal);

            // Negative literal: pick from all possible terminals except the chosen literal
            if (isNegativeLiteral) {
                var possibleTms: array<u32, 100>;
                var tmCount = 0u;
                let limit = select(100u, numTms, numTms < 100u);
                for (var i = 0u; i < limit; i++) {
                    if (i != (literal - 1u)) { possibleTms[tmCount] = get_all_tms(ntOffset + i); tmCount++; }
                }
                let tmChoice = possibleTms[lcg_rand(rngStateRef, tmCount)];
                (*localWord)[wordLen] = tmChoice + 1u; wordLen++;
            } else {
                // Positive literal: either pick it or fall back to 99 if no terminals exist
                if (numTms != 0u) {
                    let tmVal = get_all_tms(ntOffset + (literal - 1u));
                    (*localWord)[wordLen] = tmVal + 1u; wordLen++;
                } else {
                    (*localWord)[wordLen] = 99u; wordLen++;
                }
            }
        } else if ((top + 2u) < MAX_STACK) {
            let randIdx = (*bp_offset_ptr)[dpIdx] + lcg_rand(rngStateRef, expCount);

            let idxBM = (*bp_storage_ptr)[2u * randIdx + 0u];
            let idxMC = (*bp_storage_ptr)[2u * randIdx + 1u];

            stack[top] = idxMC; top++;
            stack[top] = idxBM; top++;
        }
    }
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

  suspend fun bind() { pipeline = makePipeline(src) }

  operator fun getValue(tr: Any?, property: KProperty<*>): Shader = this.also { name = property.name }

  companion object {
    private suspend fun makePipeline(wgsl: String, entryPoint: String = "main"): GPUComputePipeline =
      try {
        gpu.createComputePipelineAsync(
          GPUComputePipelineDescriptor(
            layout = "auto",
            compute = GPUProgrammableStage(
              module = gpu.createShaderModule(GPUShaderModuleDescriptor(code = wgsl)),
              entryPoint = entryPoint
            )
          )
        ).await()
      } catch (e: Throwable) { e.printStackTrace(); throw e }

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

      sparse_load.invoke1d(numWorkgroups, sparseDataGpuBuffer, outputBuffer, coeffsBuffer)

      sparseDataGpuBuffer.destroy()
      coeffsBuffer.destroy()
      return outputBuffer
    }

    fun IntArray.toSquareMatrixSparse(n: Int): GPUBuffer {
      val outputByteSize = n * n * Int32Array.BYTES_PER_ELEMENT
      val outputBuffer = GPUBuffer(outputByteSize, GPUBufferUsage.STCPSD)
      val sparseDataBuffer = toGPUBuffer()
      val numWorkgroups = ceil((size / 2.0) / SPARSE_WRITER_WORKGROUP_SIZE).toInt()
      sparse_mat_load.invoke1d(numWorkgroups, sparseDataBuffer, outputBuffer)
      sparseDataBuffer.destroy()
      return outputBuffer
    }

    fun List<Int>.toGPUBuffer(usage: Int = GPUBufferUsage.STCPSD): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@toGPUBuffer.toTypedArray(), 0) }
        .let { GPUBuffer(sz = size * 4, us = usage, data = it) }

    fun IntArray.toGPUBuffer(usage: Int = GPUBufferUsage.STCPSD): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@toGPUBuffer.toTypedArray(), 0) }
        .let { GPUBuffer(sz = size * 4, us = usage, data = it) }

    // TODO: figure out map/unmap lifetime
    fun GPUBuffer(sz: Number, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
      gpu.createBuffer(descriptor = GPUBufferDescriptor(size = sz.toDouble(), usage = us))
        .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }

    // Define the workgroup size consistently (must match WGSL)
    const val PREFIX_SUM_WORKGROUP_SIZE = 256

    suspend fun prefixSumGPU(inputBuf: GPUBuffer, length: Int): GPUBuffer {
      val numGroups = (length + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE

      val outputBuf = GPUBuffer(inputBuf.size.toInt(), GPUBufferUsage.STCPSD)
      val blockSumsBuf = GPUBuffer(numGroups * 4, GPUBufferUsage.STCPSD)
      val uniBuf = intArrayOf(length).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      prefix_sum_p1.invoke1d(numGroups, inputBuf, outputBuf, blockSumsBuf, uniBuf)

      if (numGroups > 1) {
        val scannedBlockSumsBuf = blockSumsBuf.readInts().scan(0) { acc, it -> acc + it }.toGPUBuffer()
        prefix_sum_p2.invoke1d(numGroups, outputBuf, scannedBlockSumsBuf, uniBuf)
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
//      val (allFSAPairsFlattened, allFSAPairsOffsets) = //fsa.midpoints.prefixScan()
//        reachBuf.readInts().sparsifyReachabilityMatrix().prefixScan()
      //  TODO: enforce exact equivalence?
      val (allFSAPairsFlattened, allFSAPairsOffsets) = buildMidpointsGPU(fsa.numStates, reachBuf)
      println("Flat midpoints in ${t0.elapsedNow()} : ${allFSAPairsFlattened.size} # ${allFSAPairsOffsets.size}")

      println("Sparse reachability took ${t0.elapsedNow()} / (${4 *(allFSAPairsFlattened.size + allFSAPairsOffsets.size)} bytes)")

      /** Memory layout: [CFL_STRUCT] */
      val metaBuf = packStruct(
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
      val uniBuf     = intArrayOf(states).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      mdpt_count.invoke2d(states, reachBuf, cntBuf, uniBuf)
      val offBuf = prefixSumGPU(cntBuf, totalPairs)
      val last   = listOf(totalPairs - 1)
      val totalM = offBuf.readIndices(last)[0] + cntBuf.readIndices(last)[0]
      val flatBuf = GPUBuffer(totalM * 4, GPUBufferUsage.STCPSD)
      mdpt_write.invoke2d(states, reachBuf, offBuf, flatBuf, uniBuf)

      uniBuf.destroy()
      cntBuf.destroy()
      return flatBuf to offBuf
    }

    suspend fun buildBackpointers(numStates: Int, numNTs: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer): Triple<GPUBuffer, GPUBuffer, GPUBuffer> {
      val totalCells = numStates * numStates * numNTs

      val bpCountBuf = GPUBuffer(totalCells * 4, GPUBufferUsage.STCPSD)

      println("Total cells: $totalCells = $numStates^2 * $numNTs")
      bp_count.invoke3d(numStates, numNTs, dpIn, bpCountBuf, metaBuf)

//      val bpOffsetBuf = bpCountBuf.readInts().scan(0) { acc, arr -> acc + arr }.dropLast(1).toIntArray().toGPUBuffer(GPUBufferUsage.STCPSD)
      val bpOffsetBuf = prefixSumGPU(bpCountBuf, totalCells)

      val lastIdx = listOf(totalCells - 1)
      val totalExpansions = bpOffsetBuf.readIndices(lastIdx)[0] + bpCountBuf.readIndices(lastIdx)[0]
      println("Total expansions: $totalExpansions")

      val bpStorageBuf = GPUBuffer(totalExpansions * 2 * 4, GPUBufferUsage.STCPSD)

      bp_write.invoke3d(numStates, numNTs, dpIn, bpOffsetBuf, bpStorageBuf, metaBuf)

      return Triple(bpCountBuf, bpOffsetBuf, bpStorageBuf)
    }

    fun buildLanguageSizeBuf(nStates: Int, nNT: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer, tmBuf: GPUBuffer): GPUBuffer {
      val totalCells = nStates * nStates * nNT
      val lsDenseBuf = GPUBuffer(totalCells * 4, GPUBufferUsage.STCPSD)

      for (span in 1..<nStates) {
        val spanBuf = intArrayOf(span).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

        val pass = gpu.createCommandEncoder()
        pass.beginComputePass().apply {
          setPipeline(ls_dense.pipeline)
          setBindGroup(0, ls_dense.pipeline.bindBuffers(dpIn, lsDenseBuf, metaBuf, tmBuf, spanBuf))
          dispatchWorkgroups(nStates - span, 1, nNT)
          end()
        }
        gpu.queue.submit(arrayOf(pass.finish()))
      }
      return lsDenseBuf
    }
  }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader
  suspend fun invokeCFLFixpoint(numStates: Int, numNonterminals: Int, input: IntArray, metaBuf: GPUBuffer): GPUBuffer {
//    var t0 = TimeSource.Monotonic.markNow()

    val rowCoeff = numStates * numNonterminals
    val colCoeff = numNonterminals
    val dpIn = input.toGPUBufferSparse(GPUBufferUsage.STCPSD, numStates * rowCoeff, rowCoeff, colCoeff)
//    println("Time to load buffer: ${t0.elapsedNow()} (${input.size * 4} bytes)")

    var prevValue = -1

    for (round in 0..<numStates) {
      val changesBuf = intArrayOf(0).toGPUBuffer()
      cfl_mul_upper.invoke3d(numStates, numNonterminals, dpIn, metaBuf, changesBuf)
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
    var t0 = TimeSource.Monotonic.markNow()
    var prevValue = -1

    for (round in 0..<states) {
      val changesBuf = intArrayOf(0).toGPUBuffer()
      dag_reach.invoke2d(states, input, changesBuf)
      val changesThisRound = changesBuf.readInts()[0]
      changesBuf.destroy()
      if (changesThisRound == prevValue) break
      prevValue = changesThisRound
      println("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
      t0 = TimeSource.Monotonic.markNow()
    }

    return input to prevValue
  }

  fun invoke1d(threads: Int, vararg inputs: GPUBuffer) =
    gpu.createCommandEncoder().run {
      beginComputePass().apply {
        setPipeline(pipeline)
        setBindGroup(0, pipeline.bindBuffers(*inputs))
        dispatchWorkgroups(threads)
        end()
      }
      gpu.queue.submit(arrayOf(finish()))
    }

  fun invoke2d(t1: Int, vararg inputs: GPUBuffer) =
    gpu.createCommandEncoder().run {
      beginComputePass().apply {
        setPipeline(pipeline)
        setBindGroup(0, pipeline.bindBuffers(*inputs))
        dispatchWorkgroups(t1, t1)
        end()
      }
      gpu.queue.submit(arrayOf(finish()))
    }

  fun invoke3d(t1: Int, t2: Int, vararg inputs: GPUBuffer) =
    gpu.createCommandEncoder().run {
      beginComputePass().apply {
        setPipeline(pipeline)
        setBindGroup(0, pipeline.bindBuffers(*inputs))
        dispatchWorkgroups(t1, t1, t2)
        end()
      }
      gpu.queue.submit(arrayOf(finish()))
    }
}

// constants   = [c0,c1,…]
// buffers[i]  = payload_i   (u32‑packed GPUBuffer)
// result      =  [constants | (off0,len0) (off1,len1)… | payload_0 … payload_k ]
//               ^ headerInts.size * 4  bytes
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

// TODO: kernelize?
fun IntArray.splitIntoWords(cfg: CFG, maxSamples: Int, maxWordLen: Int) =
  (0..<maxSamples).map { sampleIdx ->
    val offset = sampleIdx * maxWordLen
    slice(offset..<(offset + maxWordLen)).map { it and 0xFF }
      .fold(mutableListOf<MutableList<Int>>()) { slices, token ->
        if (token == 0) {
          if (slices.lastOrNull()?.isNotEmpty() == true) slices.add(mutableListOf())
        } else {
          if (slices.isEmpty() || slices.last().isEmpty()) slices.add(mutableListOf())
          slices.last().add(token)
        }
        slices
      }.filter { it.isNotEmpty() }
      .map { slice -> slice.joinToString(" ") { c -> if (c == 0) "0" else cfg.tmLst[c - 1] } }
  }.filter { it.isNotEmpty() }