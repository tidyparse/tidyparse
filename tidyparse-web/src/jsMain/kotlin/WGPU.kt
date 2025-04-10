@file:OptIn(ExperimentalUnsignedTypes::class)

import Shader.Companion.GPUBuffer
import Shader.Companion.readIndices
import Shader.Companion.readInts
import Shader.Companion.toGPUBuffer
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import js.array.asList
import js.buffer.*
import js.typedarrays.Int32Array
import kotlinx.browser.document
import kotlinx.coroutines.*
import kotlinx.dom.appendText
import org.w3c.dom.HTMLDivElement
import web.events.*
import web.gpu.*
import web.performance.performance
import kotlin.js.Promise
import kotlin.math.*
import kotlin.random.Random
import kotlin.reflect.KProperty
import kotlin.time.*

lateinit var gpu: GPUDevice
var gpuAvailable = false
external val navigator: dynamic

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
      listOf(sparse_load,
        cfl_mul_upper, prefix_sum_p1, prefix_sum_p2,
        bp_count, bp_write, sample_words).forEach { it.bind() }
      benchmarkWGPU()
      benchmarkWGPURepair()
    } catch (e: Exception) { e.printStackTrace(); return }

    gpuAvailable = true
  } else {
    println("not detected.")
    gpuAvailDiv.appendText("WebGPU is NOT available.")
  }
}

suspend fun repairCode(cfg: CFG, code: List<String>, levRadius: Int): List<List<String>> {
  val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
  val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()

  val t0 = TimeSource.Monotonic.markNow()
  val levFSA = makeLevFSA(code, levRadius)
  println("Made levFSA in ${t0.elapsedNow()}")

  val (allFSAPairsFlattened, allFSAPairsOffsets) = levFSA.midpoints.prefixScan()
  println("Midpoints took ${t0.elapsedNow()}")

  // Sparse index nonzero entries of the M_0 parse chart
  fun FSA.byteFormat(cfg: CFG): IntArray {
    val t0 = TimeSource.Monotonic.markNow()
    val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it] ?: emptyList() }
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

  /** Memory layout: [CFL_STRUCT] */
  val metadata: IntArray = packStruct(
    constants = listOf(levFSA.numStates, cfg.nonterminals.size),
    // FSA Encoding
    allFSAPairsFlattened, allFSAPairsOffsets, levFSA.finalIdxs.toIntArray(),
    // CFG Encoding
    grammarFlattened, grammarOffsets
  )

  val dpIn = levFSA.byteFormat(cfg)
//  println("Initial nonzeros: ${dpIn.count { it != 0 }}")

  println("PREPROCESSING TOOK: ${t0.elapsedNow()}") // ~230ms
  val words = repairPipeline(cfg, levFSA, dpIn, metadata)
  println("Received: ${words.size} words")
  println("Round trip repair: ${t0.elapsedNow()}") // ~500ms

  return words
}

suspend fun benchmarkWGPURepair() {
  val cfg = pythonStatementCNFAllProds
  val code = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE".tokenizeByWhitespace()
  val words = repairCode(cfg, code, 5).distinct().map { it.joinToString(" ") }
  println("Distinct words: ${words.size}")
//  words.take(10).forEachIndexed { i, it -> println("$i.) ${it.joinToString(" ")}") }

  val (valid, invalid) = words.shuffled().take(10).partition { it in cfg.language }
  println("\nValid samples:\n")
  valid.forEachIndexed { i, it -> println("$i.) ${it.trim()}") }
  println("\nInvalid samples:\n")
  invalid.forEachIndexed { i, it -> println("$i.) ${it.trim()}") }
}

suspend fun repairPipeline(cfg: CFG, fsa: FSA, dpInSparse: IntArray, metadata: IntArray): List<List<String>> {
//  println("FSA(|Q|=${fsa.numStates}, |δ|=${fsa.transit.size}), " +
//      "CFG(|Σ|=${cfg.terminals.size}, |V|=${cfg.nonterminals.size}, |P|=${cfg.nonterminalProductions.size})")
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNonterminals) = fsa.numStates to cfg.nonterminals.size
  val metaBuf = metadata.toGPUBuffer(GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)
  val dpBuf = cfl_mul_upper.invokeCFLFixpoint(numStates, numNonterminals, dpInSparse, metaBuf)
  println("TIME TO FILL PARSE CHART: ${t0.elapsedNow()}")

  val startNT     = cfg.bindex[START_SYMBOL]
  val finalStates = fsa.finalIdxs
  val startIdxs   = finalStates.map { it * numNonterminals + startNT }
    .let { it.zip(dpBuf.readIndices(it)) }.filter { (_, v) -> v != 0 }.map { it.first }
//    .onEach { println("STIDX: $it") }

  if (!startIdxs.isEmpty()) { println("Valid parse found: dpComplete has ${startIdxs.size} start indices") }
  else { println("No valid parse found: dpComplete has no entries in final states!"); return emptyList() }

  val t2 = TimeSource.Monotonic.markNow()
  println("Time to copy metadata: ${t2.elapsedNow()}")

  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNonterminals, dpBuf, metaBuf)
  println("Built backpointers in ${t2.elapsedNow()}")

//  println("bpCountBufSum: ${bpCountBuf.readInts().sum()}")
//  println("bpOffsetBufSum: ${bpOffsetBuf.readInts().sumOf { it.toLong() }}")
//  println("bpStorageBufSize: ${bpStorageBuf.readInts().size}")

  val startIndicesBuf = startIdxs.toIntArray()
  val numStartIndices = startIdxs.size

  val maxSamples = 1000
  val maxWordLen = fsa.width + 10

  val outBuf = GPUBuffer(maxSamples * maxWordLen * 4, GPUBufferUsage.STCPSD)
  val uniforms = (intArrayOf(numStartIndices, numStates, maxWordLen, numNonterminals) + startIndicesBuf)
    .toGPUBuffer(GPUBufferUsage.STCPSD)

  val packTime = TimeSource.Monotonic.markNow()
  val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it]?.map { cfg.tmMap[it]!! } ?: emptyList() }
  val nt_tm_lens = terminalLists.map { it.size }.toIntArray()
  val nt_tm_offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1).toIntArray()
  val all_tm = terminalLists.flatMap { it }.toIntArray()

  /** Memory layout: [TERM_STRUCT] */
  val terminals = packStruct(emptyList(), nt_tm_lens, nt_tm_offsets, all_tm)
    .toGPUBuffer(GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)
  println("Packing time: ${packTime.elapsedNow()}")

  println("Invoking sampler...")
  val t3 = TimeSource.Monotonic.markNow()
  val threads = (maxSamples + 63) / 64
  sample_words.invoke1d(threads, dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf, outBuf, uniforms, terminals)
  val rawTokens: IntArray = outBuf.readInts()
  println("Sampled words in ${t3.elapsedNow()}")

  val t4 = TimeSource.Monotonic.markNow()
  val wordsPerSample = rawTokens.splitIntoWords(cfg, maxSamples, maxWordLen)
  println("Decoded tokens in ${t4.elapsedNow()}")

  outBuf.destroy()
  metaBuf.destroy()
  dpBuf.destroy()
  terminals.destroy()

  return wordsPerSample
}

// TODO: kernelize?
fun IntArray.splitIntoWords(cfg: CFG, maxSamples: Int, maxWordLen: Int) =
  (0 until maxSamples).map { sampleIdx ->
    val offset = sampleIdx * maxWordLen
    slice(offset until offset + maxWordLen).map { it and 0xFF }
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
const val PREAMBLE = """
let r = gid.x;
let c = gid.y;
if (c <= r) { return; }
let A = gid.z;

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

//language=wgsl
val cfl_mul_upper by Shader(CFL_STRUCT + """
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
val bp_count by Shader(CFL_STRUCT + """
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
val bp_write by Shader(CFL_STRUCT + """
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
    @builtin(workgroup_id)         groupId  : vec3<u32>,
    @builtin(local_invocation_id)  localId  : vec3<u32>
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

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE) fn main(
    @builtin(global_invocation_id) globalId : vec3<u32>,
    @builtin(workgroup_id)         groupId  : vec3<u32>,
    @builtin(local_invocation_id)  localId  : vec3<u32>
) {
    let N     = prefixUni.N;
    let gid   = globalId.x;
    let grpId = groupId.x;

    if (grpId > 0u) {
        let blockOffsetVal = scannedBlockSums[grpId - 1u];
        if (gid < N) { dataBuf[gid] = dataBuf[gid] + blockOffsetVal; }
    }
}""")

//language=wgsl
const val TERM_STRUCT = """
struct Uniforms {
    numStartIndices : u32,
    numStates       : u32,
    maxWordLen      : u32,
    numNonterminals : u32,
    startIndices    : array<u32>
};

struct Terminals {
    nt_tm_lens_offset : u32,    nt_tm_lens_size : u32,
       offsets_offset : u32,       offsets_size : u32,
       all_tms_offset : u32,       all_tms_size : u32,
       
       payload : array<u32>
}"""

//language=wgsl
val sample_words by Shader(TERM_STRUCT + """
@group(0) @binding(0) var<storage, read>              dp_in : array<u32>;
@group(0) @binding(1) var<storage, read>           bp_count : array<u32>;
@group(0) @binding(2) var<storage, read>          bp_offset : array<u32>;
@group(0) @binding(3) var<storage, read>         bp_storage : array<u32>;
@group(0) @binding(4) var<storage, read_write> sampledWords : array<u32>;
@group(0) @binding(5) var<storage, read>           uniforms : Uniforms;
@group(0) @binding(6) var<storage, read>          terminals : Terminals;
  
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var tid = gid.x;

    var localWord: array<u32, 1024>;
    for (var i = 0u; i < uniforms.maxWordLen; i++) { localWord[i] = 0u; }
    let q = terminals.offsets_size;

    let rIndex    = lcg_rand(&tid, uniforms.numStartIndices);
    let dpIndex   = uniforms.startIndices[rIndex];

    let nnt       = uniforms.numNonterminals;
    let numStates = uniforms.numStates;
    let A         = dpIndex % nnt;
    let rowCol    = dpIndex / nnt;
    let c         = rowCol % numStates;
    let r         = rowCol / numStates;
    let snt       = numStates * nnt;
    let startIdx  = r * snt + c * nnt + A;

    sampleTopDown(&dp_in, &bp_count, &bp_offset, &bp_storage, startIdx, &tid, &localWord, uniforms.maxWordLen, nnt);

    let baseIdx = gid.x * uniforms.maxWordLen;
    for (var i = 0u; i < uniforms.maxWordLen; i++) { sampledWords[baseIdx + i] = localWord[i]; }
}

fn get_nt_tm_lens(index: u32) -> u32 { return terminals.payload[terminals.nt_tm_lens_offset + index]; }
fn get_offsets(index: u32) -> u32 { return terminals.payload[terminals.offsets_offset + index]; }
fn get_all_tms(index: u32) -> u32 { return terminals.payload[terminals.all_tms_offset + index]; }

fn lcg_rand(stateRef: ptr<function, u32>, range: u32) -> u32 { 
  let newVal = (1664525u * (*stateRef)) + 1013904223u;
  *stateRef = newVal;
  return select(newVal % range, 0u, range == 0u); 
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
struct SparseElement { r: u32, c: u32, v: u32, i: i32 };
struct Coeffs { rowCoeff: u32, colCoeff: u32, };

@group(0) @binding(0) var<storage, read> sparse_elements: array<SparseElement>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<i32>;
@group(0) @binding(2) var<uniform> coeffs: Coeffs;

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

class Shader(val src: String) {
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
        GPUBindGroupEntry(binding = index, resource = jsObject<GPUBindingResource> { buffer = buf })
      }.toTypedArray()
      return gpu.createBindGroup(GPUBindGroupDescriptor(layout = getBindGroupLayout(0), entries = ent))
    }

    suspend fun GPUBuffer.readInts(): IntArray {
      val t0 = TimeSource.Monotonic.markNow()
      val readDst = GPUBuffer(size.toInt(), GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
      val cmd = gpu.createCommandEncoder()
      cmd.copyBufferToBuffer(this, 0.0, readDst, 0.0, size)
      gpu.queue.submit(arrayOf(cmd.finish()))
      (readDst.mapAsync(1) as Promise<*>).await()
      val t = Int32Array(readDst.getMappedRange()).asList().toIntArray()
      readDst.destroy()
      println("Read ${size.toInt()} bytes in ${t0.elapsedNow()}")
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

      val sparseDataGpuBuffer = toGPUBuffer(GPUBufferUsage.STCPSD)
      val outputByteSize = totalSizeInInts.toLong() * Int32Array.BYTES_PER_ELEMENT.toLong()
      val outputBuffer = GPUBuffer(outputByteSize, usage or GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)
      val coeffsBuffer = intArrayOf(rowCoeff, colCoeff).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
      val numWorkgroups = ceil(size / 4.0 / SPARSE_WRITER_WORKGROUP_SIZE).toInt()

      sparse_load.invoke1d(numWorkgroups, sparseDataGpuBuffer, outputBuffer, coeffsBuffer)

      sparseDataGpuBuffer.destroy()
      coeffsBuffer.destroy()
      return outputBuffer
    }

    fun IntArray.toGPUBuffer(usage: Int): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@toGPUBuffer.toTypedArray(), 0) }
        .let { GPUBuffer(sz = size * 4, us = usage, data = it) }

    // TODO: figure out map/unmap lifetime
    fun GPUBuffer(sz: Number, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
      gpu.createBuffer(GPUBufferDescriptor(size = sz.toDouble(), usage = us))
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
        val scannedBlockSumsBuf = blockSumsBuf.readInts()
          .scan(0) { acc, it -> acc + it }.toIntArray().toGPUBuffer(GPUBufferUsage.STCPSD)
        prefix_sum_p2.invoke1d(numGroups, outputBuf, scannedBlockSumsBuf, uniBuf)
        scannedBlockSumsBuf.destroy()
      }

      uniBuf.destroy()
      return outputBuf
    }

    suspend fun buildBackpointers(numStates: Int, numNonterminals: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer): Triple<GPUBuffer, GPUBuffer, GPUBuffer> {
      val totalCells = numStates * numStates * numNonterminals

      val bpCountBuf = GPUBuffer(totalCells * 4, GPUBufferUsage.STCPSD)

      println("Total cells: $totalCells = $numStates^2 * $numNonterminals")
      bp_count.invoke3d(numStates, numNonterminals, dpIn, bpCountBuf, metaBuf)

      val bpOffsetBuf = prefixSumGPU(bpCountBuf, totalCells)

      val lastIdx = listOf(totalCells - 1)
      val totalExpansions = bpOffsetBuf.readIndices(lastIdx)[0] + bpCountBuf.readIndices(lastIdx)[0]
      println("Total expansions: $totalExpansions")

      val bpStorageBuf = GPUBuffer(totalExpansions * 2 * 4, GPUBufferUsage.STCPSD)

      bp_write.invoke3d(numStates, numNonterminals, dpIn, bpOffsetBuf, bpStorageBuf, metaBuf)

      return Triple(bpCountBuf, bpOffsetBuf, bpStorageBuf)
    }
  }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader
  suspend fun invokeCFLFixpoint(numStates: Int, numNonterminals: Int, input: IntArray, metaBuf: GPUBuffer): GPUBuffer {
    var t0 = TimeSource.Monotonic.markNow()

    val rowCoeff = numStates * numNonterminals
    val colCoeff = numNonterminals
    val dpIn = input.toGPUBufferSparse(GPUBufferUsage.STCPSD, numStates * rowCoeff, rowCoeff, colCoeff)
    println("Time to load buffer: ${t0.elapsedNow()} (${input.size * 4} bytes)")

    var prevValue = -1

    for(round in 0 until numStates) {
      val changesBuf = intArrayOf(0).toGPUBuffer(GPUBufferUsage.STCPSD)
      invoke3d(numStates, numNonterminals, dpIn, metaBuf, changesBuf)
      val changesThisRound = changesBuf.readInts()[0]
      changesBuf.destroy()
      if (changesThisRound == prevValue) break
      prevValue = changesThisRound
//      println("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
      t0 = TimeSource.Monotonic.markNow()
    }

    return dpIn
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

  suspend fun invokeExp(vararg inputs: GPUBuffer, threads: Int, iterations: Int = 1): IntArray {
    val encoder: GPUCommandEncoder = gpu.createCommandEncoder()
    val buf1 = inputs[1]
    val buf2 = GPUBuffer(buf1.size.toInt(), buf1.usage)

    for (step in 1..iterations) {
//      val t0 = TimeSource.Monotonic.markNow()
//      val encoder: GPUCommandEncoder = gpu.createCommandEncoder()
      val (currentM, currentOut) = if (step % 2 == 1) buf1 to buf2 else buf2 to buf1
      encoder.beginComputePass().apply {
        setPipeline(pipeline)
        setBindGroup(0, pipeline.bindBuffers(currentM, currentOut, inputs[0]))
        dispatchWorkgroups(threads, threads)
        end()
      }
//      println("Read: ${inputs[0].readInts()}")
//      gpu.queue.submit(arrayOf(encoder.finish()))
//      println("Elapsed: ${t0.elapsedNow()}")
    }

    gpu.queue.submit(arrayOf(encoder.finish()))
    return (if (iterations % 2 == 1) buf2 else buf1).readInts().also {
      buf1.destroy()
      buf2.destroy()
    }
  }
}

fun packStruct(constants: List<Int> = emptyList(), vararg arrays: IntArray): IntArray {
  val offsets = arrays.scan(0) { acc, arr -> acc + arr.size }.dropLast(1)

  val header = buildList {
    addAll(0, constants)
    arrays.forEachIndexed { i, arr -> add(offsets[i]); add(arr.size) }
  }

  return (header + arrays.flatMap { it.asIterable() }).toIntArray()
}

suspend fun benchmarkWGPU() {
  val N = 300
  val P = 20
  val M = IntArray(N * N) { i ->
    val r = i / N // Row index
    val c = i % N // Column index
    if (c > r) Random.nextInt(2, 10) else 0
  }

  WGSL_GEMX_ITERATE.bind()
  val t2 = performance.now()
  val gSum = WGSL_GEMX_ITERATE.invokeExp(
    intArrayOf(N).toGPUBuffer(GPUBufferUsage.STCPSD), M.toGPUBuffer(140),
    threads = N, iterations = P
  ).asList().hashCode()
  val t3 = performance.now()
  println("GPU hash=$gSum in ${t3 - t2} ms (N=$N, P=$P)")

  val t0 = performance.now()

  fun iterateCPU(a: IntArray, P: Int): Int {
    val n = sqrt(a.size.toDouble()).toInt()
    var current = a.copyOf()
    for (step in 1..P) {
      val next = IntArray(n * n)
      for (r in 0 until n) for (c in 0 until n)
        next[r * n + c] = (0 until n).sumOf { k -> current[r * n + k] * current[k * n + c] }
      current = next
    }
    return current.toList().hashCode()
  }

  val cSum = iterateCPU(M, P)
  val t1 = performance.now()
  println("CPU hash=$cSum in ${t1 - t0} ms (N=$N, P=$P)")
}

//language=wgsl
val WGSL_GEMX_ITERATE by Shader("""
struct Params { N: u32 };

@group(0) @binding(0) var<storage, read>       M:   array<i32>;
@group(0) @binding(1) var<storage, read_write> Out: array<i32>;
@group(0) @binding(2) var<storage, read_write> param: Params;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let N = param.N;
    
    if (col <= row) { return; }

    let rowOffset = row * N;
    var acc = 0;
    for (var k = 0u; k < N; k = k + 1u) {
        let a = M[rowOffset + k];
        let b = M[k * N + col];
        acc = acc + (a * b);
    }
    
    Out[rowOffset + col] = acc;
}""")

fun List<List<List<Int>>>.prefixScan(): Pair<IntArray, IntArray> = // TODO: move into shader kernel?
  measureTimedValue {
    fun IntArray.prefixSumSizes(): IntArray =
      IntArray(size + 1).also { prefix ->
        copyInto(prefix, destinationOffset = 1)
        for (i in 1..size) prefix[i] += prefix[i - 1]
      }

    val filtered = mapIndexed { p, outer ->
      outer.mapIndexed { q, inner -> inner.filter { it in (p + 1) until q } }
    }.flatten()

    val flattened = filtered.flatten().toIntArray()
    val offsets = filtered.map { it.size }.toIntArray().prefixSumSizes()

    flattened to offsets
  }.also { println("Completed prefix scan in: ${it.duration}") }.value