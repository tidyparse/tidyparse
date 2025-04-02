@file:OptIn(ExperimentalUnsignedTypes::class)

import Shader.Companion.makeBuffer
import Shader.Companion.readInts
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
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.reflect.KProperty
import kotlin.time.*

lateinit var gpu: GPUDevice
var gpuAvailable = false
external val navigator: dynamic

fun tryBootstrapingGPU() = MainScope().async {
  checkWebGPUAvailability()
  if (gpuAvailable) {
    listOf(WGSL_GEMX_ITERATE, cfl_mul_upper, prefix_sum_p1, prefix_sum_p2,
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

  val t0 = TimeSource.Monotonic.markNow()
  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE".tokenizeByWhitespace()
  val radius = 5
  val levFSA = makeLevFSA(pythonCode, radius)

  // Initializes entries of M_0 parse chart
  fun FSA.byteFormat(cfg: CFG): IntArray {
    val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it] ?: emptyList() }
    // 0 and 1 are reserved for (0) no parse exists and (1) parse exists, but an internal nonterminal node
    // Other byte values are used to denote the presence (+) or absence (-) of a leaf terminal
    fun StrPred.predByte(A: Int): Int = (
      if (arg == "[.*]" || (arg.startsWith("[!=]") && arg.drop(4) !in terminalLists[A])) Int.MAX_VALUE - 1 // All possible terminals
      else if (arg.startsWith("[!=]")) (1.shl(30) + (terminalLists[A].indexOf(arg.drop(4)) + 1).shl(1)) // Represent negation using sign bit
      else (terminalLists[A].indexOf(arg) + 1).shl(1)
    )

    val rowCoeff = numStates * cfg.nonterminals.size  // Precomputed for row index
    val colCoeff = cfg.nonterminals.size              // Precomputed for column index
    val parseChart = IntArray(rowCoeff * numStates) { 0 }

    cfg.unitProductions.flatMap { (A, σ) ->
      nominalForm.flattenedTriples.filter { arc -> arc.second(σ) }.map { (q0, sp, q1) ->
        val Aidx = cfg.bindex[A]
        // row, col, Aidx, terminal
        listOf(stateMap[q0]!!, stateMap[q1]!!, Aidx, sp.predByte(Aidx))
      }
    }.forEach { (r, c, v, i) -> parseChart[r * rowCoeff + c * colCoeff + v] = i }

    return parseChart
  }

  val (allFSAPairsFlattened, allFSAPairsOffsets) = levFSA.midpoints.prefixScan()

  /** Memory layout: [CFL_STRUCT] */
  val metadata: IntArray = packStruct(
    constants = listOf(levFSA.numStates, cfg.nonterminals.size),

    // FSA Encoding
    allFSAPairsFlattened,
    allFSAPairsOffsets,
    levFSA.finalIdxs.toIntArray(),

    // CFG Encoding
    cfg.vindex.map { it.toList() }.flatten().toIntArray(),
    cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()
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
  val dpComplete = cfl_mul_upper.invokeCFLFixpoint(dpIn, metadata)

  val numStates       = metadata[0]
  val numNonterminals = metadata[1]

  println("Built backpointers!")

  val startNT     = cfg.bindex[START_SYMBOL]
  val finalStates = fsa.finalIdxs
  val startIdxs   = mutableListOf<Int>()
  for (q in finalStates) {
    val dpIndex = q * numNonterminals + startNT
    if (dpIndex in dpComplete.indices && dpComplete[dpIndex] != 0) { startIdxs.add(dpIndex) }
  }

  if (!startIdxs.isEmpty()) { println("Valid parse found: dpComplete has ${startIdxs.size} start indices") }
  else { println("No valid parse found: dpComplete has no entries in final states!"); return emptyList() }

  val dpBuf   = dpComplete.makeBuffer(GPUBufferUsage.STCPSD)
  val metaBuf = metadata.makeBuffer(GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)

  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNonterminals, dpBuf, metaBuf)

  // Create a GPU buffer with these valid start indices
  val startIndicesBuf = startIdxs.toIntArray()
  val numStartIndices = startIdxs.size

  val maxSamples = 1000
  val maxWordLen = 30

  val outBuf = makeBuffer(maxSamples * maxWordLen * 4, GPUBufferUsage.STCPSD)
  val uniforms  = (intArrayOf(numStartIndices, numStates, maxWordLen, numNonterminals) + startIndicesBuf)
    .makeBuffer(GPUBufferUsage.STCPSD)

  val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it]?.map { cfg.tmMap[it]!! } ?: emptyList() }
  val all_tm = terminalLists.flatMap { it }.toIntArray()
  val nt_tm_offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1).toIntArray()
  val nt_tm_lens = terminalLists.map { it.size }.toIntArray()

  val terminals = packStruct(emptyList(), all_tm, nt_tm_offsets, nt_tm_lens)
    .makeBuffer(GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST)

  println("Invoking sampler...")
  val rawTokens = sample_words.invoke(
    dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf, outBuf, uniforms, terminals,
    readFrom = outBuf, threads = (maxSamples + 64 - 1) / 64
  )

  println("Sum: ${rawTokens.sum()}")

  val wordsPerSample = mutableListOf<List<String>>()
  var offset = 0
  for (sampleIdx in 0 until maxSamples) {
    // Extract that row
    val row = rawTokens.slice(offset until offset + maxWordLen)
    offset += maxWordLen

    val tokens = row.map { it and 0xFF }

    val slices = mutableListOf<List<Int>>()
    var current = mutableListOf<Int>()
    for (t in tokens) {
      if (t == 0) {
        // zero => delimiter
        if (current.isNotEmpty()) {
          slices.add(current.toList())
          current.clear()
        }
      } else { current.add(t) }
    }

    if (current.isNotEmpty()) { slices.add(current) }

    val sampleWords = slices.map { codeList -> codeList.joinToString("") { c -> c.toChar().toString() } }
    if (sampleWords.isNotEmpty()) { wordsPerSample.add(sampleWords) }
  }

  return wordsPerSample
}

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
   fn getMdptOffset(index: u32) -> u32 { return u32(cs.payload[cs.mdptsOffsetsOffset + index]); }
fn getGrammarSymbol(index: u32) -> u32 { return u32(cs.payload[cs.grammarFlattenedOffset + index]); }
fn getGrammarOffset(index: u32) -> u32 { return u32(cs.payload[cs.grammarOffsetsOffset + index]); }"""

//language=text
val PREAMBLE = """
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
if (aoi < cs.mdptsOffsetsSize) { pairOffsetNext = getMdptOffset(aoi); } else { pairOffsetNext = cs.mdptsSize; }
"""

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
        let mdpt = u32(getMdpt(pairIdx)); for (var g = startGC; g < endGC; g+= 2u) {
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
                // Each backpointer record is 2 ints
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

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
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
        if (lid >= offset) {
             let leftVal = tile[tileIdx - offset];
             tile[tileIdx] = tile[tileIdx] + leftVal;
        }
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

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
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
}""".trimIndent())

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
    let tid = gid.x;

    if (tid >= uniforms.numStartIndices) { return; }

    var localWord: array<u32, 1024>;
    for (var i = 0u; i < uniforms.maxWordLen; i++) { localWord[i] = 0u; }
    let q = terminals.offsets_size;

    var rngState = tid;

    let rIndex    = lcg_randomRange(&rngState, uniforms.numStartIndices);
    let dpIndex   = uniforms.startIndices[rIndex];

    let nnt       = uniforms.numNonterminals;
    let numStates = uniforms.numStates;
    let A         = dpIndex % nnt;
    let rowCol    = dpIndex / nnt;
    let c         = rowCol % numStates;
    let r         = rowCol / numStates;
    let snt       = numStates * nnt;
    let startIdx  = r * snt + c * nnt + A;

    sampleTopDown(&dp_in, &bp_count, &bp_offset, &bp_storage, startIdx, &rngState, &localWord, uniforms.maxWordLen, nnt);

    let baseIndex = tid * uniforms.maxWordLen;
    for (var i = 0u; i < uniforms.maxWordLen; i = i + 1u) {
        if (i >= 1024u) { break; }
        sampledWords[u32(tid) * u32(uniforms.maxWordLen) + u32(i)] = u32(localWord[i] & 0x00FFu);
    }
}

struct Uniforms {
    numStartIndices : u32,
    numStates       : u32,
    maxWordLen      : u32,
    numNonterminals : u32,
    startIndices    : array<u32>
};

fn get_nt_tm_lens(index: u32) -> u32 { return u32(terminals.payload[terminals.nt_tm_lens_offset + index]); }
fn get_offsets(index: u32) -> u32 { return u32(terminals.payload[terminals.offsets_offset + index]); }
fn get_all_tms(index: u32) -> u32 { return u32(terminals.payload[terminals.all_tms_offset + index]); }

struct Terminals {
    nt_tm_lens_offset : u32,    nt_tm_lens_size : u32,
       offsets_offset : u32,       offsets_size : u32,
       all_tms_offset : u32,       all_tms_size : u32,
       
       payload : array<u32>
}

fn lcg_random(stateRef: ptr<function, u32>) -> u32 {
    let newVal = (1664525u * (*stateRef)) + 1013904223u;
    *stateRef = newVal;
    return newVal;
}

fn lcg_randomRange(stateRef: ptr<function, u32>, range: u32) -> u32 {
    if (range == 0u) { return 0u; } // Avoid modulo by zero
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
        let hasLiteral = (predicate >> 1u) != 0u;

        let rVal = lcg_randomRange(rngStateRef, expCount);

        if (hasLiteral && (!canRecurse || ((rVal % 2u) == 0u))) {
            let nonterminal = dpIdx % nnt;
            let isNegative  = (predicate & 0x8000u) != 0u;
            let literal     = (predicate >> 1u) & 0x7FFFu;

            if (nonterminal >= nnt) { continue; }

            let numTms   = get_nt_tm_lens(nonterminal);
            let ntOffset = get_offsets(nonterminal);

            if (isNegative) {
                // choose from among all possible except “literal - 1”
                var possibleTms: array<u32, 100>; // Fixed-size array, OK
                var tmCount = 0u;
                for (var i = 0u; i < min(numTms, 100u); i = i + 1u) {
                    // Bounds check for all_tm access
                    let tmIndex = ntOffset + i;
                    // if (tmIndex >= arrayLength(all_tm_ptr)) { continue; } // Example check

                    if (i != (literal - 1u)) { // Check if literal > 0?
                        if (tmCount < 100u) { // Bounds check for possibleTms
                            possibleTms[tmCount] = get_all_tms(tmIndex);
                            tmCount++;
                        } else { break; } // Stop if possibleTms is full
                    }
                }

                if (tmCount > 0u) { // Only choose if there are options
                    let choiceIndex = lcg_randomRange(rngStateRef, tmCount);
                    let tmChoice    = possibleTms[choiceIndex]; // choiceIndex is already < tmCount
                     // Bounds check for localWord write
                    if (wordLen < 1024u) {
                       (*localWord)[wordLen] = tmChoice + 1u;
                       wordLen++;
                    } else { break; } // Stop if localWord is full
                } // else: what happens if tmCount is 0? Word remains unchanged?
            } else {
                // positive literal
                if (numTms != 0u && literal > 0u) { // Ensure literal is valid index (1-based)
                    let tmIndex = ntOffset + literal - 1u;
                    // Bounds checks
                    // if (tmIndex >= arrayLength(all_tm_ptr)) { continue; }
                    if (wordLen < 1024u) {
                        let tmVal = get_all_tms(tmIndex);
                        (*localWord)[wordLen] = tmVal + 1u;
                        wordLen++;
                    } else { break; } // Stop if localWord is full
                } else {
                     // Bounds check
                    if (wordLen < 1024u) {
                        (*localWord)[wordLen] = 99u; // Default/error value?
                        wordLen++;
                    } else { break; } // Stop if localWord is full
                }
            }
        } else if (canRecurse) { // Only expand if allowed and stack has room
            // Check if there's room on the stack *before* calculating indices
            if (top + 2 <= 1024) { // Check <= 1024 because top is index of *next* empty slot
                 // Bounds check for bp_offset access
                // if (dpIdx >= arrayLength(bp_offset_ptr)) { continue; }
                let bpBaseIndex = bp_offset_ptr[dpIdx] + rVal;

                // Bounds check for bp_storage access
                let randIdxTimes2 = bpBaseIndex * 2u;
                // if ((randIdxTimes2 + 1u) >= arrayLength(bp_storage_ptr)) { continue; }

                let idxBM   = bp_storage_ptr[randIdxTimes2 + 0u];
                let idxMC   = bp_storage_ptr[randIdxTimes2 + 1u];

                // Push onto stack (already checked bounds)
                stack[top]  = idxMC; top++;
                stack[top]  = idxBM; top++;
            }
            // else: stack is full, cannot recurse further down this path
        }
        // Add safety break for excessively long loops without progress?
    }
    // Ensure remaining unused part of localWord is zeroed if needed by subsequent logic?
    // The loop in main handles writing based on wordLen implicitly.
}""")

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

    fun bindBuffers(pipeline: GPUComputePipeline, vararg buffers: GPUBuffer): GPUBindGroup {
      val lay = pipeline.getBindGroupLayout(0)
      inline fun <T> jsObject(init: dynamic.() -> Unit): T {
        val o = js("{}")
        init(o)
        return o as T
      }
      val ent = buffers.mapIndexed { index, buf ->
        GPUBindGroupEntry(binding = index, resource = jsObject<GPUBindingResource> { buffer = buf })
      }.toTypedArray()
      return gpu.createBindGroup(GPUBindGroupDescriptor(layout = lay, entries = ent))
    }

    suspend fun GPUBuffer.readInts(): IntArray {
      val readDst = makeBuffer(size.toInt(), GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
      val cmd = gpu.createCommandEncoder()
      cmd.copyBufferToBuffer(this, 0.0, readDst, 0.0, size)
      gpu.queue.submit(arrayOf(cmd.finish()))
      (readDst.mapAsync(1) as Promise<*>).await()
      return Int32Array(readDst.getMappedRange()).asList().toIntArray().also { readDst.unmap() }
    }

    fun IntArray.makeBuffer(usage: Int): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@makeBuffer.toTypedArray(), 0) }
        .let { makeBuffer(sz = size * 4, us = usage, data = it) }

    fun makeBuffer(sz: Int, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
      gpu.createBuffer(GPUBufferDescriptor(size = sz.toDouble(), usage = us))
        .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }

    // Define the workgroup size consistently (must match WGSL)
    const val PREFIX_SUM_WORKGROUP_SIZE = 256

    suspend fun prefixSumGPU(inputBuf: GPUBuffer, length: Int): GPUBuffer {
      val tpg = PREFIX_SUM_WORKGROUP_SIZE
      val numGroups = (length + tpg - 1) / tpg

      val outputBuf = makeBuffer(inputBuf.size.toInt(), GPUBufferUsage.STCPSD)
      val blockSumsBuf = makeBuffer(numGroups * 4, GPUBufferUsage.STCPSD)
      val scannedBlockSumsBuf = makeBuffer(numGroups * 4, GPUBufferUsage.STCPSD) // For exclusive scan results
      val uniBuf = makeBuffer(4, GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      val uniIntData = Int32Array<ArrayBuffer>(1)
      uniIntData[0] = length
      gpu.queue.writeBuffer(uniBuf, 0.0, uniIntData)

      fun Shader.invoke(numGroups: Int, vararg buffs: GPUBuffer) {
        val cmdEncoder = gpu.createCommandEncoder()
        val passEncoder = cmdEncoder.beginComputePass()
        passEncoder.setPipeline(pipeline)
        passEncoder.setBindGroup(0, bindBuffers(pipeline, *buffs))
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

      val bpCountBuf = makeBuffer(totalCells * 4, GPUBufferUsage.STCPSD)

      println("Total cells: $totalCells = $numStates^2 * $numNonterminals")
      bp_count.invoke3d(numStates, numNonterminals, dpIn, bpCountBuf, metaBuf)

      val bpOffsetBuf = prefixSumGPU(bpCountBuf, totalCells)

      val lastIdx = totalCells - 1
      val readCmd = gpu.createCommandEncoder()
      val readLen = 2 * 4 // 2 integers
      val readBack = makeBuffer(readLen, GPUBufferUsage.MAP_READ or GPUBufferUsage.COPY_DST)
      readCmd.copyBufferToBuffer(bpOffsetBuf, (lastIdx * 4).toDouble(), readBack, 0.0, 4.0)
      readCmd.copyBufferToBuffer(bpCountBuf,  (lastIdx * 4).toDouble(), readBack, 4.0, 4.0)
      gpu.queue.submit(arrayOf(readCmd.finish()))

      (readBack.mapAsync(1) as Promise<*>).await()
      val readArray = Int32Array(readBack.getMappedRange()) // 2 elements
      val offsetVal = readArray[0]
      val countVal  = readArray[1]
      val totalExpansions = offsetVal + countVal

      val expansionsBufSize = totalExpansions * 2 * 4
      val bpStorageBuf = makeBuffer(expansionsBufSize, GPUBufferUsage.STCPSD)

      bp_write.invoke3d(numStates, numNonterminals, dpIn, bpOffsetBuf, bpStorageBuf, metaBuf)
      println("Writing expansions back to GPU...")

      return Triple(bpCountBuf, bpOffsetBuf, bpStorageBuf)
    }
  }

  suspend fun bind() { pipeline = makePipeline(src) }

  operator fun getValue(tr: Any?, property: KProperty<*>): Shader = this.also { name = property.name }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader
  suspend fun invokeCFLFixpoint(vararg inputs: IntArray): IntArray {
    var t0 = TimeSource.Monotonic.markNow()
    require(inputs.size >= 2) { "Expected at least dpIn + metadata, got ${inputs.size} buffers." }

    val dpIn = inputs[0].makeBuffer(usage = GPUBufferUsage.STCPSD)
    println("Time to load buffer: ${t0.elapsedNow()}")

    val metaBuf = inputs[1].makeBuffer(usage = GPUBufferUsage.STORAGE + GPUBufferUsage.COPY_DST)
    val (numStates, numNonterminals) = inputs[1]

    val dpOut = makeBuffer(dpIn.size.toInt(), dpIn.usage)
    val changesBuf = makeBuffer(sz = 4, us = GPUBufferUsage.STCPSD)

    var prevValue = -1
    val maxRounds = numStates

    repeat(maxRounds) { round ->
      gpu.queue.writeBuffer(changesBuf, 0.0, data = Int32Array<ArrayBuffer>(1).apply { set(arrayOf(0), 0) })
      val (inBuf, outBuf) = if (round % 2 == 0) dpIn to dpOut else dpOut to dpIn

      invoke3d(numStates, numNonterminals, inBuf, outBuf, metaBuf, changesBuf)

      val changesThisRound = changesBuf.readInts()[0]

      if (changesThisRound == prevValue) {
        println("Fixpoint reached at round=$round")
        return outBuf.readInts()
      }
      prevValue = changesThisRound
      println("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
    }

    return (if (maxRounds % 2 == 0) dpIn else dpOut).readInts()
  }

  fun invoke3d(t1: Int, t2: Int, vararg inputs: GPUBuffer) {
    val cmdEnc = gpu.createCommandEncoder()
    val pass = cmdEnc.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindBuffers(pipeline, *inputs))
    pass.dispatchWorkgroups(t1, t1, t2)
    pass.end()
    gpu.queue.submit(arrayOf(cmdEnc.finish()))
  }

  suspend operator fun invoke(vararg inputs: GPUBuffer, threads: Int, iterations: Int = 1): ArrayBuffer {
    val encoder: GPUCommandEncoder = gpu.createCommandEncoder()

    val buf1 = inputs[0] // Initial input buffer
    val param = inputs[1]

    val buf2 = makeBuffer(buf1.size.toInt(), buf1.usage)

    for (step in 1..iterations) {
      val (currentM, currentOut) = if (step % 2 == 1) buf1 to buf2 else buf2 to buf1
      val bindGroup = bindBuffers(pipeline, currentM, currentOut, param)
      encoder.beginComputePass().apply {
        setPipeline(pipeline)
        setBindGroup(0, bindGroup)
        dispatchWorkgroups(threads, threads)
        end()
      }
    }

    val finalOut = if (iterations % 2 == 1) buf2 else buf1

    val output = makeBuffer(finalOut.size.toInt(), GPUBufferUsage.MAP_READ or GPUBufferUsage.COPY_DST)
    encoder.copyBufferToBuffer(finalOut, 0.0, output, 0.0, output.size)
    gpu.queue.submit(arrayOf(encoder.finish()))

    (output.mapAsync(1) as Promise<*>).await()
    return output.getMappedRange()
  }

  suspend operator fun invoke(vararg inputs: GPUBuffer, readFrom: GPUBuffer, threads: Int): IntArray {
    gpu.createCommandEncoder().beginComputePass().apply {
      setPipeline(pipeline)
      setBindGroup(0, bindBuffers(pipeline, *inputs))
      dispatchWorkgroups(threads, 1, 1)
      end()
    }

    return readFrom.readInts()
  }
}

fun packStruct(constants: List<Int> = emptyList(), vararg arrays: IntArray): IntArray {
  val offsets = arrays.scan(0) { acc, arr -> acc + arr.size }.dropLast(1)

  val header = buildList {
    addAll(0, constants)
    arrays.forEachIndexed { index, arr -> add(offsets[index]); add(arr.size) }
  }

  return (header + arrays.flatMap { it.asIterable() }).toIntArray()
}

suspend fun iterateGPU(input: Array<Int>, P: Int): Int {
  val n = sqrt(input.size.toDouble()).toInt()
  val s = input.size
  val bytes = s * 4

  val bufM = makeBuffer(bytes, 140, Int32Array<ArrayBuffer>(s).apply { set(input, 0) })
  val bufP = makeBuffer(16, 72, Int32Array<ArrayBuffer>(4).apply { set(arrayOf(n), 0) })

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