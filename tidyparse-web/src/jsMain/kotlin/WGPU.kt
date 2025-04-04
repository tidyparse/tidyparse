@file:OptIn(ExperimentalUnsignedTypes::class)

import Shader.Companion.GPUBuffer
import Shader.Companion.toGPUBuffer
import Shader.Companion.toGPUBufferFast
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

fun tryBootstrapingGPU() = MainScope().async {
  checkWebGPUAvailability()
  if (gpuAvailable) {
    listOf(WGSL_GEMX_ITERATE, sparse_load,
      cfl_mul_upper, prefix_sum_p1, prefix_sum_p2,
      bp_count, bp_write, sample_words).forEach { it.bind() }
    benchmarkWGPURepair()
    benchmarkWGPU()
  }
}

suspend fun checkWebGPUAvailability() {
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
    gpuAvailable = true
    gpu.addEventListener(EventType("uncapturederror"), { e: dynamic -> println("Uncaptured: ${e.error.message}") })
  } else {
    println("not detected.")
    gpuAvailDiv.appendText("WebGPU is NOT available.")
  }
}

suspend fun benchmarkWGPURepair() {
  val cfg = pythonStatementCNFAllProds
  val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
  val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()

  val t0 = TimeSource.Monotonic.markNow()
  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE".tokenizeByWhitespace()
  val radius = 5
  val levFSA = makeLevFSA(pythonCode, radius)
  println("Made levFSA in ${t0.elapsedNow()}")

  val (allFSAPairsFlattened, allFSAPairsOffsets) = levFSA.midpoints.prefixScan()
  println("Midpoints took ${t0.elapsedNow()}")

  // Sparse index nonzero entries of the M_0 parse chart
  fun FSA.byteFormat(cfg: CFG): IntArray {
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
      }
    }

    return sparseChart.flatten().toIntArray()
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
  println("Initial nonzeros: ${dpIn.count { it != 0 }}")

  println("Preprocessing took ${t0.elapsedNow()}")
  val words = repairPipeline(cfg, levFSA, dpIn, metadata)
  println("Received: ${words.size} words")
  words.take(10).map { println(it.joinToString(" ")) }
  println("Round trip repair: ${t0.elapsedNow()}")
}

suspend fun repairPipeline(cfg: CFG, fsa: FSA, dpIn: IntArray, metadata: IntArray): List<List<String>> {
  val (numStates, numNonterminals) = fsa.numStates to cfg.nonterminals.size
  val dpComplete = cfl_mul_upper.invokeCFLFixpoint(numStates, numNonterminals, dpIn, metadata)

  val startNT     = cfg.bindex[START_SYMBOL]
  val finalStates = fsa.finalIdxs
  val startIdxs   = finalStates.map { it * numNonterminals + startNT }.filter { dpComplete[it] != 0 }

  if (!startIdxs.isEmpty()) { println("Valid parse found: dpComplete has ${startIdxs.size} start indices") }
  else { println("No valid parse found: dpComplete has no entries in final states!"); return emptyList() }

  val t1 = TimeSource.Monotonic.markNow()
  val dpBuf = dpComplete.toGPUBufferFast(GPUBufferUsage.STCPSD)
  println("Time to copy back dpBuf: ${t1.elapsedNow()}")
  val t2 = TimeSource.Monotonic.markNow()
  val metaBuf = metadata.toGPUBuffer(GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)
  println("Time to copy metadata: ${t2.elapsedNow()}")

  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNonterminals, dpBuf, metaBuf)
  println("Built backpointers!")

//  println("bpCountBuf: ${bpCountBuf.readInts().sum()}")
//  println("bpOffsetBufSize: ${bpOffsetBuf.size}")
//  println("bpStorageBufSize: ${bpStorageBuf.readInts().size}")

  val startIndicesBuf = startIdxs.toIntArray()
  val numStartIndices = startIdxs.size

  val maxSamples = 1000
  val maxWordLen = fsa.width + 10

  val outBuf = GPUBuffer(maxSamples * maxWordLen * 4, GPUBufferUsage.STCPSD)
  val uniforms  = (intArrayOf(numStartIndices, numStates, maxWordLen, numNonterminals) + startIndicesBuf)
    .toGPUBuffer(GPUBufferUsage.STCPSD)

  val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it]?.map { cfg.tmMap[it]!! } ?: emptyList() }
  val nt_tm_lens = terminalLists.map { it.size }.toIntArray()
  val nt_tm_offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1).toIntArray()
  val all_tm = terminalLists.flatMap { it }.toIntArray()

  val terminals = packStruct(emptyList(), nt_tm_lens, nt_tm_offsets, all_tm)
    .toGPUBuffer(GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)

  println("Invoking sampler...")
  val t3 = TimeSource.Monotonic.markNow()
  val rawTokens: IntArray = sample_words.invoke(
    dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf, outBuf, uniforms, terminals,
    readFrom = outBuf, threads = (maxSamples + 64 - 1) / 64
  )
  println("Sampled words in ${t3.elapsedNow()}")

  val t4 = TimeSource.Monotonic.markNow()
  val wordsPerSample = rawTokens.splitIntoWords(cfg, maxSamples, maxWordLen)
  println("Decoded tokens in ${t4.elapsedNow()}")

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

@group(0) @binding(0) var<storage, read>          dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>   dp_out : array<u32>;
@group(0) @binding(2) var<storage, read>             cs : CFLStruct;
@group(0) @binding(3) var<storage, read_write>  changes : AtomicChange;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    $PREAMBLE
    
    let dpVal = dp_in[dpIdx];
    if (dpVal != 0) {
        dp_out[dpIdx] = dpVal;
        atomicAdd(&changes.count, 1u);
        if ((dpVal & 0x01) != 0) { return; }
    }

    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = getMdpt(pairIdx); for (var g = startGC; g < endGC; g+= 2u) {
            let B = getGrammarSymbol(g); let C = getGrammarSymbol(g + 1u);

            let idxBM = r*snt + mdpt*NT + B;
            let idxMC = mdpt*snt + c*NT + C;

            if ((dp_in[idxBM] != 0) && (dp_in[idxMC] != 0)) {
                dp_out[dpIdx] |= 0x01;
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

    let tileIdx = lid;
    if (gid < N) { tile[tileIdx] = inputBuf[gid]; } else { tile[tileIdx] = 0u; }

    workgroupBarrier();

    var offset = 1u;
    while (offset < WORKGROUP_SIZE) {
        if (lid >= offset) { tile[tileIdx] += tile[tileIdx - offset]; }
        offset = offset * 2u;
        workgroupBarrier();
    }

    if (lid == WORKGROUP_SIZE - 1u) {
        blockSums[grpId] = tile[tileIdx];
        tile[tileIdx] = 0u;
    }

    workgroupBarrier();

    offset = WORKGROUP_SIZE / 2u;
    while (offset > 0u) {
        if (lid < offset) {
            let leftChildIdx  = tileIdx + offset;
            let rightChildIdx = tileIdx;

            let leftVal = tile[leftChildIdx];
            let rightVal = tile[rightChildIdx];

            tile[rightChildIdx] = leftVal;
            tile[leftChildIdx] = leftVal + rightVal;
        }
        offset = offset / 2u;
        workgroupBarrier();
    }

    if (gid < N) { outputBuf[gid] = tile[tileIdx]; }
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
val sample_words by Shader("""
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

    let rIndex    = lcg_randomRange(&tid, uniforms.numStartIndices);
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
}

fn get_nt_tm_lens(index: u32) -> u32 { return terminals.payload[terminals.nt_tm_lens_offset + index]; }
fn get_offsets(index: u32) -> u32 { return terminals.payload[terminals.offsets_offset + index]; }
fn get_all_tms(index: u32) -> u32 { return terminals.payload[terminals.all_tms_offset + index]; }

fn lcg_random(stateRef: ptr<function, u32>) -> u32 {
    let newVal = (1664525u * (*stateRef)) + 1013904223u;
    *stateRef = newVal;
    return newVal;
}

fn lcg_randomRange(stateRef: ptr<function, u32>, range: u32) -> u32 {
    if (range == 0u) { return 0u; }
    return lcg_random(stateRef) % range;
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
    var stack: array<u32, 1024>;
    var top     = 0;
    var wordLen = 0u;

    if (top < 1024) { stack[top] = startDPIdx; top++; }

    var iter = 0u; loop {
        if (top <= 0 || wordLen >= maxWordLen || iter >= (maxWordLen * 98u)) { break; }
        iter++; top--;
        let dpIdx = stack[top];

        let expCount = bp_count_ptr[dpIdx];
        let predicate = dp_in_ptr[dpIdx];
        let canRecurse = (predicate & 0x01u) != 0u;
        // Check if any literal bits are set (anything other than the recursion bit 0)
        let hasLiteral = (predicate & ~0x01u) != 0u; // More robust check for any literal info

        // Simplified random choice logic - Metal version uses expCount which might be more correct
        // if expCount is the total number of *expansions* (recursive + literal).
        // The current WGSL bp_count only counts recursive expansions.
        // Let's assume for now the choice is between literal (if available) and recursion (if available).
        // If both, pick randomly. If only one, pick that one.
        // This needs careful comparison with how bp_count is truly intended to be used.
        // Sticking closer to the Metal version's apparent logic for now:
        let totalOptions = expCount + u32(hasLiteral); // Approx total choices
        let rVal = lcg_randomRange(rngStateRef, totalOptions); // Use total options for range

        // Decision logic: prioritize literal if chosen randomly or if no recursion possible
        // If hasLiteral and (!canRecurse or random choice favors literal)
        // Metal used: if (hasLiteral && (!canRecurse || ((rVal % 2u) == 0u)))
        // Let's refine the random choice based on totalOptions:
        // If rVal < expCount, recurse (if possible). Otherwise, take literal (if possible).
        let chooseLiteral = hasLiteral && (!canRecurse || rVal >= expCount);

        // if (hasLiteral && (!canRecurse || ((rVal % 2u) == 0u))) { // Original WGSL logic based on Metal snippet
        if (chooseLiteral) { // Use refined logic
            let nonterminal = dpIdx % nnt;
            // Decode the u32 predicate from Kotlin
            let isNegative  = (predicate & 0x40000000u) != 0u; // Check bit 30 (1 << 30)
            // Mask out recursion (bit 0) and negative (bit 30) bits to get shifted literal
            let literalShifted = predicate & 0x3FFFFFFEu;
            let literal = literalShifted >> 1u; // Get the original 1-based index

            // Check for wildcard case (Int.MAX_VALUE - 1 == 0x7FFFFFFE)
            if (predicate == 0x7FFFFFFEu) {
               // Handle wildcard ([.*]) - sample any terminal for this nonterminal?
               // This case wasn't fully specified in the Metal snippet's handling.
               // Let's add a placeholder or choose randomly if possible.
               let numTms = get_nt_tm_lens(nonterminal);
               let ntOffset = get_offsets(nonterminal);
               if (numTms > 0u) {
                   let choiceIndex = lcg_randomRange(rngStateRef, numTms);
                   let tmIndex = ntOffset + choiceIndex;
                   if (wordLen < 1024u) {
                       (*localWord)[wordLen] = get_all_tms(tmIndex) + 1u; // Add 1 like others
                       wordLen++;
                   } else { break; }
               } else { /* Cannot sample wildcard if nonterminal has no terminals */ }
               continue; // Skip rest of literal logic
            }

            if (nonterminal >= nnt) { continue; } // Bounds check

            let numTms   = get_nt_tm_lens(nonterminal);
            let ntOffset = get_offsets(nonterminal);

            if (isNegative) {
                // choose from among all possible except “literal - 1”
                var possibleTms: array<u32, 100>; // Max 100 terminals per NT assumed
                var tmCount = 0u;
                // Iterate through terminals for this nonterminal
                for (var i = 0u; i < min(numTms, 100u); i = i + 1u) {
                   let tmIndex = ntOffset + i;
                   // Check if this terminal index corresponds to the negated literal
                   if (i != (literal - 1u)) { // literal is 1-based index
                       if (tmCount < 100u) { // Bounds check for possibleTms
                           possibleTms[tmCount] = get_all_tms(tmIndex);
                           tmCount++;
                       } else { break; } // Stop if possibleTms is full
                   }
                }

                if (tmCount > 0u) { // Only choose if there are options
                    let choiceIndex = lcg_randomRange(rngStateRef, tmCount);
                    let tmChoice    = possibleTms[choiceIndex];
                    if (wordLen < 1024u) { // Bounds check for localWord write
                       (*localWord)[wordLen] = tmChoice + 1u; // Add 1 like positive case
                       wordLen++;
                    } else { break; } // Stop if localWord is full
                } // else: No alternative terminals to sample, word remains unchanged?
            } else { // Positive literal
                if (literal > 0u && literal <= numTms) { // Ensure literal is valid 1-based index
                   let tmIndex = ntOffset + literal - 1u;
                   if (wordLen < 1024u) { // Bounds check
                       let tmVal = get_all_tms(tmIndex);
                       (*localWord)[wordLen] = tmVal + 1u; // Add 1 to make it non-zero
                       wordLen++;
                   } else { break; } // Stop if localWord is full
                } else { // Invalid positive literal index (e.g., literal=0 or > numTms) or numTms=0
                   // Error case? Metal used 99. Let's use 0xFFFFFFFF perhaps? Or stick to 99?
                   if (wordLen < 1024u) {
                       (*localWord)[wordLen] = 99u; // Default/error value
                       wordLen++;
                   } else { break; } // Stop if localWord is full
                }
            }
        // } else if (canRecurse) { // Original WGSL logic
        } else if (canRecurse && expCount > 0u) { // Use refined logic: ensure expansions exist
           // Check if there's room on the stack *before* calculating indices
            if (top + 2 <= 1024) { // Check <= 1024 allows filling stack
                // Select which specific expansion to follow using rVal
                let expansionChoice = rVal; // Since rVal < expCount here
                let bpBaseIndex = bp_offset_ptr[dpIdx] + expansionChoice;

                // Bounds check for bp_storage access (IMPORTANT)
                let randIdxTimes2 = bpBaseIndex * 2u;
                // Need bp_storage size uniform or derived calculation if possible
                // Assuming bp_storage is large enough for now, but this is risky

                let idxBM   = bp_storage_ptr[randIdxTimes2 + 0u];
                let idxMC   = bp_storage_ptr[randIdxTimes2 + 1u];

                // Push onto stack (already checked bounds)
                stack[top]  = idxMC; top++;
                stack[top]  = idxBM; top++;
            } // else: stack is full, cannot recurse further down this path
        }
        // else: Neither literal nor recursion possible/chosen. Frame is dropped.
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
      val t = Int32Array(readDst.getMappedRange()).asList().toIntArray().also { readDst.unmap() }
      println("Read ${size.toInt()} bytes in ${t0.elapsedNow()}")
      return t
    }

    suspend fun IntArray.toGPUBufferSparse(usage: Int, totalSizeInInts: Int, rowCoeff: Int, colCoeff: Int): GPUBuffer {
      require(this.size % 4 == 0) { "Input array size must be a multiple of 4 for sparse data (r,c,v,i)." }
      require(totalSizeInInts > 0) { "totalSizeInInts must be positive." }

      val sparseDataNumElements = this.size / 4
      val packedSparseData = Int32Array<ArrayBuffer>(this.size).also { dest -> for (i in this.indices) { dest[i] = this[i] } }

      // Helper to upload Int32Array efficiently via staging
      suspend fun Int32Array<ArrayBuffer>.uploadInt32ArrayToGPUBuffer(usage: Int): GPUBuffer {
        val byteSize = this.byteLength.toLong()
        val destinationUsage = usage or GPUBufferUsage.COPY_DST
        val destinationBuffer = gpu.createBuffer(GPUBufferDescriptor(
          size = byteSize.toDouble(),
          usage = destinationUsage
        ))

        val stagingBuffer = gpu.createBuffer(GPUBufferDescriptor(
          size = byteSize.toDouble(),
          usage = GPUBufferUsage.MAP_WRITE or GPUBufferUsage.COPY_SRC
        ))

        try {
          stagingBuffer.mapAsync(GPUBufferUsage.MAP_WRITE).await() // Standard API call

          val mappedRange: ArrayBuffer = stagingBuffer.getMappedRange() // Map entire buffer
          Int32Array(mappedRange).set(this, 0) // Copy this Int32Array into the mapped buffer view

          stagingBuffer.unmap()

          val commandEncoder = gpu.createCommandEncoder()
          commandEncoder.copyBufferToBuffer(
            source = stagingBuffer,
            sourceOffset = 0.0,
            destination = destinationBuffer,
            destinationOffset = 0.0,
            size = byteSize.toDouble()
          )
          gpu.queue.submit(arrayOf(commandEncoder.finish()))
        } catch (e: Throwable) {
          stagingBuffer.destroy()
          destinationBuffer.destroy()
          throw e
        } finally { stagingBuffer.destroy() }
        return destinationBuffer
      }

      // --- 2. Upload Sparse Data List to GPU ---
      // This buffer only needs STORAGE usage for the compute shader to read it.
      // uploadInt32ArrayToGPUBuffer adds COPY_DST automatically.
      val sparseDataGpuBuffer = packedSparseData.uploadInt32ArrayToGPUBuffer(usage = GPUBufferUsage.STORAGE)

      // --- 3. Create Final Destination Buffer ---
      val outputByteSize = totalSizeInInts.toLong() * Int32Array.BYTES_PER_ELEMENT.toLong()
      // Ensure the destination buffer has STORAGE for shader write and COPY_DST
      val outputBuffer = gpu.createBuffer(GPUBufferDescriptor(
        size = outputByteSize.toDouble(),
        usage = usage or GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST
      ))
      // Optional: Clear buffer if initial zeros needed?
      // gpu.createCommandEncoder().apply { clearBuffer(outputBuffer, 0, outputByteSize) }
      //    .let { gpu.queue.submit(arrayOf(it.finish())) }
      // Consider if waiting for clear is necessary: gpu.queue.onSubmittedWorkDone().await()

      // --- 4. Create Uniform Buffer for Coefficients ---
      val coeffsBuffer = gpu.createBuffer(GPUBufferDescriptor(
        size = (2 * 4).toDouble(),
        usage = GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST
      ))
      // Write rowCoeff and colCoeff into the uniform buffer
      gpu.queue.writeBuffer(coeffsBuffer, 0.0, Int32Array<ArrayBuffer>(arrayOf(rowCoeff, colCoeff)))

      // --- 6. Dispatch Compute Shader ---
      val commandEncoder = gpu.createCommandEncoder()
      val passEncoder = commandEncoder.beginComputePass()

      passEncoder.setPipeline(sparse_load.pipeline)
      passEncoder.setBindGroup(0, sparse_load.pipeline.bindBuffers(sparseDataGpuBuffer, outputBuffer, coeffsBuffer))

      val numWorkgroups = ceil(sparseDataNumElements.toDouble() / SPARSE_WRITER_WORKGROUP_SIZE).toInt()
      if (numWorkgroups > 0) { passEncoder.dispatchWorkgroups(numWorkgroups) }

      passEncoder.end()
      gpu.queue.submit(arrayOf(commandEncoder.finish()))

      sparseDataGpuBuffer.destroy()
      coeffsBuffer.destroy()
      return outputBuffer
    }

    suspend fun IntArray.toGPUBufferFast(usage: Int): GPUBuffer {
      val dataSize = size
      val byteSize = dataSize * Int32Array.BYTES_PER_ELEMENT.toLong() // Use Long for byteSize

      // 1. Create the final destination buffer
      val destinationBufferDescriptor = GPUBufferDescriptor(
        size = byteSize.toDouble(),
        usage = usage or GPUBufferUsage.COPY_DST
      )
      val destinationBuffer = gpu.createBuffer(destinationBufferDescriptor)

      // 2. Create the temporary CPU-side staging buffer
      val stagingBufferDescriptor = GPUBufferDescriptor(
        size = byteSize.toDouble(),
        usage = GPUBufferUsage.MAP_WRITE or GPUBufferUsage.COPY_SRC
      )
      val stagingBuffer = gpu.createBuffer(stagingBufferDescriptor)

      try {
        stagingBuffer.mapAsync(GPUBufferUsage.MAP_WRITE, 0.0, byteSize.toDouble()).await()

        val mappedRange: ArrayBuffer = stagingBuffer.getMappedRange(0.0, byteSize.toDouble())
        val bufferView = Int32Array(mappedRange)

        // --- CRITICAL SECTION: Efficient Copy ---
        // Direct loop is often the most straightforward in JS environments
        // Need to Avoid creating intermediate Kotlin/JS collections here.
        for (i in 0 until dataSize) bufferView[i] = this[i]
        // -----------------------------------------

        // 5. Unmap the staging buffer (gives data ownership back to GPU)
        stagingBuffer.unmap()

        // 6. Create command encoder and issue GPU-side copy
        val commandEncoder = gpu.createCommandEncoder()
        commandEncoder.copyBufferToBuffer(
          source = stagingBuffer,
          sourceOffset = 0.0,
          destination = destinationBuffer,
          destinationOffset = 0.0,
          size = byteSize.toDouble()
        )

        // 7. Submit the command
        val commandBuffer = commandEncoder.finish()
        gpu.queue.submit(arrayOf(commandBuffer))

      } finally { stagingBuffer.destroy() }

      return destinationBuffer
    }

    fun IntArray.toGPUBuffer(usage: Int): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@toGPUBuffer.toTypedArray(), 0) }
        .let { GPUBuffer(sz = size * 4, us = usage, data = it) }

    fun GPUBuffer(sz: Int, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
      gpu.createBuffer(GPUBufferDescriptor(size = sz.toDouble(), usage = us))
        .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }

    // Define the workgroup size consistently (must match WGSL)
    const val PREFIX_SUM_WORKGROUP_SIZE = 256

    suspend fun prefixSumGPU(inputBuf: GPUBuffer, length: Int): GPUBuffer {
      val tpg = PREFIX_SUM_WORKGROUP_SIZE
      val numGroups = (length + tpg - 1) / tpg

      val outputBuf = GPUBuffer(inputBuf.size.toInt(), GPUBufferUsage.STCPSD)
      val blockSumsBuf = GPUBuffer(numGroups * 4, GPUBufferUsage.STCPSD)
      val scannedBlockSumsBuf = GPUBuffer(numGroups * 4, GPUBufferUsage.STCPSD) // For exclusive scan results
      val uniBuf = GPUBuffer(4, GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      val uniIntData = Int32Array<ArrayBuffer>(1)
      uniIntData[0] = length
      gpu.queue.writeBuffer(uniBuf, 0.0, uniIntData)

      fun Shader.invoke(numGroups: Int, vararg buffs: GPUBuffer) {
        val cmdEncoder = gpu.createCommandEncoder()
        val passEncoder = cmdEncoder.beginComputePass()
        passEncoder.setPipeline(pipeline)
        passEncoder.setBindGroup(0, pipeline.bindBuffers(*buffs))
        passEncoder.dispatchWorkgroups(numGroups)
        passEncoder.end()
        gpu.queue.submit(arrayOf(cmdEncoder.finish()))
      }

      prefix_sum_p1.invoke(numGroups = numGroups, inputBuf, outputBuf, blockSumsBuf, uniBuf)

      val blockSumsCPU = blockSumsBuf.readInts()

      val scannedBlockSumsCPU = IntArray(numGroups)
      var accumulatedSum = 0L
      for (i in 0 until numGroups) {
        scannedBlockSumsCPU[i] = accumulatedSum.toInt()
        accumulatedSum += blockSumsCPU[i].toUInt().toLong()
      }

      val scannedSumsData = Int32Array<ArrayBuffer>(scannedBlockSumsCPU.toTypedArray())
      gpu.queue.writeBuffer(scannedBlockSumsBuf, 0.0, scannedSumsData)

      prefix_sum_p2.invoke(numGroups = numGroups, outputBuf, scannedBlockSumsBuf, uniBuf)

      return outputBuf
    }

    suspend fun buildBackpointers(numStates: Int, numNonterminals: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer): Triple<GPUBuffer, GPUBuffer, GPUBuffer> {
      val totalCells = numStates * numStates * numNonterminals

      val bpCountBuf = GPUBuffer(totalCells * 4, GPUBufferUsage.STCPSD)

      println("Total cells: $totalCells = $numStates^2 * $numNonterminals")
      bp_count.invoke3d(numStates, numNonterminals, dpIn, bpCountBuf, metaBuf)

      val bpOffsetBuf = prefixSumGPU(bpCountBuf, totalCells)

      val lastIdx = totalCells - 1
      val readCmd = gpu.createCommandEncoder()
      val readLen = 2 * 4 // 2 integers
      val readBack = GPUBuffer(readLen, GPUBufferUsage.MAP_READ or GPUBufferUsage.COPY_DST)
      readCmd.copyBufferToBuffer(bpOffsetBuf, (lastIdx * 4).toDouble(), readBack, 0.0, 4.0)
      readCmd.copyBufferToBuffer(bpCountBuf,  (lastIdx * 4).toDouble(), readBack, 4.0, 4.0)
      gpu.queue.submit(arrayOf(readCmd.finish()))

      (readBack.mapAsync(1) as Promise<*>).await()
      val readArray = Int32Array(readBack.getMappedRange()) // 2 elements
      val offsetVal = readArray[0]
      val countVal  = readArray[1]
      val totalExpansions = offsetVal + countVal

      val expansionsBufSize = totalExpansions * 2 * 4
      val bpStorageBuf = GPUBuffer(expansionsBufSize, GPUBufferUsage.STCPSD)

      bp_write.invoke3d(numStates, numNonterminals, dpIn, bpOffsetBuf, bpStorageBuf, metaBuf)
      println("Writing expansions back to GPU...")

      return Triple(bpCountBuf, bpOffsetBuf, bpStorageBuf)
    }
  }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader

  suspend fun invokeCFLFixpoint(numStates: Int, numNonterminals: Int, vararg inputs: IntArray): IntArray {
    var t0 = TimeSource.Monotonic.markNow()
    require(inputs.size >= 2) { "Expected at least dpIn + metadata, got ${inputs.size} buffers." }

    val rowCoeff = numStates * numNonterminals
    val colCoeff = numNonterminals
    val dpIn = inputs[0].toGPUBufferSparse(GPUBufferUsage.STCPSD, numStates * rowCoeff, rowCoeff, colCoeff)
    println("Time to load buffer: ${t0.elapsedNow()} (${inputs[0].size * 4} bytes)")

    val metaBuf = inputs[1].toGPUBuffer(usage = GPUBufferUsage.STORAGE + GPUBufferUsage.COPY_DST)

    val dpOut = GPUBuffer(dpIn.size.toInt(), dpIn.usage)
    val changesBuf = GPUBuffer(sz = 4, us = GPUBufferUsage.STCPSD)

    var prevValue = -1

//    val cmdEnc = gpu.createCommandEncoder()
    repeat(numStates) { round ->
      gpu.queue.writeBuffer(changesBuf, 0.0, data = Int32Array<ArrayBuffer>(1).apply { set(arrayOf(0), 0) })
      val (inBuf, outBuf) = if (round % 2 == 0) dpIn to dpOut else dpOut to dpIn

      invoke3d(numStates, numNonterminals, inBuf, outBuf, metaBuf, changesBuf)
//      cmdEnc.beginComputePass().apply {
//        setPipeline(pipeline)
//        setBindGroup(0, bindBuffers(pipeline, inBuf, outBuf, metaBuf, changesBuf))
//        dispatchWorkgroups(numStates, numStates, numNonterminals)
//        end()
//      }

      val changesThisRound = changesBuf.readInts()[0]

      if (changesThisRound == prevValue) {
        println("Fixpoint reached at round=$round")
        val t1 = TimeSource.Monotonic.markNow()
        val t = outBuf.readInts()
        println("Time to read dpComplete: ${t1.elapsedNow()}")
        return t
      }
      prevValue = changesThisRound
      println("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
    }

    return (if (numStates % 2 == 0) dpIn else dpOut).readInts()
  }

  fun invoke3d(t1: Int, t2: Int, vararg inputs: GPUBuffer) {
    val cmdEnc = gpu.createCommandEncoder()
    cmdEnc.beginComputePass().apply {
      setPipeline(pipeline)
      setBindGroup(0, pipeline.bindBuffers(*inputs))
      dispatchWorkgroups(t1, t1, t2)
      end()
    }
    gpu.queue.submit(arrayOf(cmdEnc.finish()))
  }

  suspend operator fun invoke(vararg inputs: GPUBuffer, threads: Int, iterations: Int = 1): ArrayBuffer {
    val encoder: GPUCommandEncoder = gpu.createCommandEncoder()

    val buf1 = inputs[0]

    val buf2 = GPUBuffer(buf1.size.toInt(), buf1.usage)

    for (step in 1..iterations) {
      val (currentM, currentOut) = if (step % 2 == 1) buf1 to buf2 else buf2 to buf1
      encoder.beginComputePass().apply {
        setPipeline(pipeline)
        setBindGroup(0, pipeline.bindBuffers(currentM, currentOut, inputs[1]))
        dispatchWorkgroups(threads, threads)
        end()
      }
    }

    val finalOut = if (iterations % 2 == 1) buf2 else buf1

    val output = GPUBuffer(finalOut.size.toInt(), GPUBufferUsage.MAP_READ or GPUBufferUsage.COPY_DST)
    encoder.copyBufferToBuffer(finalOut, 0.0, output, 0.0, output.size)
    gpu.queue.submit(arrayOf(encoder.finish()))

    (output.mapAsync(1) as Promise<*>).await()
    return output.getMappedRange()
  }

  suspend operator fun invoke(vararg inputs: GPUBuffer, readFrom: GPUBuffer, threads: Int): IntArray {
    val encoder = gpu.createCommandEncoder()

    encoder.beginComputePass().apply {
      setPipeline(pipeline)
      setBindGroup(0, pipeline.bindBuffers(*inputs))
      dispatchWorkgroups(threads, 1, 1)
      end()
    }

    gpu.queue.submit(arrayOf(encoder.finish()))

    return readFrom.readInts()
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

suspend fun iterateGPU(input: Array<Int>, P: Int): Int {
  val n = sqrt(input.size.toDouble()).toInt()
  val s = input.size
  val bytes = s * 4

  val bufM = GPUBuffer(bytes, 140, Int32Array<ArrayBuffer>(s).apply { set(input, 0) })
  val bufP = GPUBuffer(16, 72, Int32Array<ArrayBuffer>(4).apply { set(arrayOf(n), 0) })

  val bufS = WGSL_GEMX_ITERATE.invoke(bufM, bufP, threads = n, iterations = P)

  return Int32Array(bufS).asList().hashCode()
}

suspend fun benchmarkWGPU() {
  val N = 302
  val P = 5
  val M = IntArray(N * N) { i ->
    val r = i / N // Row index
    val c = i % N // Column index
    if (c > r) Random.nextInt(2, 10) else 0
  }
  val t0 = performance.now()
  val cSum = iterateCPU(M, P)
  val t1 = performance.now()
  println("CPU hash=$cSum in ${t1 - t0} ms (N=$N, P=$P)")
  val t2 = performance.now()
  val gSum = iterateGPU(M.toTypedArray(), P)
  val t3 = performance.now()
  println("GPU hash=$gSum in ${t3 - t2} ms (N=$N, P=$P)")
}

suspend fun iterateCPU(a: IntArray, P: Int): Int {
  val n = sqrt(a.size.toDouble()).toInt()
  var current = a.copyOf()
  for (step in 1..P) {
    val next = IntArray(n * n)
    for (r in 0 until n) {
      for (c in 0 until n) {
        var sum = 0
        for (k in 0 until n) {
          sum += current[r * n + k] * current[k * n + c]
        }
        next[r * n + c] = sum
      }
    }
    current = next
  }
  return current.toList().hashCode()
}

//language=wgsl
val WGSL_GEMX_ITERATE by Shader("""
struct Params { N: u32 };

@group(0) @binding(0) var<storage, read>       M:   array<i32>;
@group(0) @binding(1) var<storage, read_write> Out: array<i32>;
@group(0) @binding(2) var<uniform>             param: Params;

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