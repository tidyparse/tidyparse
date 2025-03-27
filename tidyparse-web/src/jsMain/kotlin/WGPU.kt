@file:OptIn(ExperimentalUnsignedTypes::class)

import Shader.Companion.makeBuffer
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
    WGSL_GEMX_ITERATE.bind()
    cfl_mul_upper.bind()
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
    gpu.addEventListener(EventType("uncapturederror"),
      { e: dynamic -> println("Uncaptured GPU error: ${e.error.message}") })
  } else {
    println("not detected.")
    gpuAvailDiv.appendText("WebGPU is NOT available.")
  }
}

suspend fun benchmarkWGPURepair() {
  val cfg = pythonStatementCNFAllProds

  val t0 = TimeSource.Monotonic.markNow()
  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE".tokenizeByWhitespace()
  val radius = 3
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

  val dpIn = levFSA.byteFormat(cfg).let { dpi -> IntArray(dpi.size) { dpi[it].toInt() } }

  println("Preprocessing took ${t0.elapsedNow()}")
  val dpComplete = cfl_mul_upper.invokeCFLFixpoint(dpIn, metadata)

  println("Round trip repair: ${t0.elapsedNow()}")
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

//language=wgsl
val cfl_mul_upper by Shader(CFL_STRUCT + """
struct AtomicChange { count: atomic<u32> };

@group(0) @binding(0) var<storage, read>          dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>   dp_out : array<u32>;
@group(0) @binding(2) var<storage, read>             cs : CFLStruct;
@group(0) @binding(3) var<storage, read_write>  changes : AtomicChange;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let r = gid.x;      // row index
    let c = gid.y;      // column index
    let A = gid.z;      // nonterminal index

    let N  = cs.numStates;
    let NT = cs.numNonterminals;

    if (c <= r) { return; }
    
    let snt     = N * NT;
    let dpIdx   = r*snt + c*NT + A;
    let startGC = getGrammarOffset(A);
    let endGC   = select(cs.grammarFlattenedSize, getGrammarOffset(A + 1u), A + 1u < NT);
    let aoi     = r*N + c + 1u;
    let pairOffset     = getMdptOffset(aoi - 1u);
    let pairOffsetNext = select(cs.mdptsSize, getMdptOffset(aoi), aoi < cs.mdptsOffsetsSize);

    let dpVal = dp_in[dpIdx];
    if (dpVal != 0) {
        dp_out[dpIdx] = dpVal;
        atomicAdd(&changes.count, 1u);
        if ((dpVal & 0x01) != 0) { return; }
    }

    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx = pairIdx + 1u) {
        let mdpt = u32(getMdpt(pairIdx)); // mdpt is a state index

        var g = startGC;
        loop {
            if (g >= endGC) { break; }
            let B = getGrammarSymbol(g);
            let C = getGrammarSymbol(g + 1u);

            let idxBM = r*snt + mdpt*NT + B;
            let idxMC = mdpt*snt + c*NT + C;

            if ((dp_in[idxBM] != 0) && (dp_in[idxMC] != 0)) {
                dp_out[dpIdx] |= 0x01;
                atomicAdd(&changes.count, 1u);
                return;
            }

            g += 2;
        }
    }
}""".trimIndent())

//language=wgsl
val bp_count by Shader(CFL_STRUCT + """
@group(0) @binding(0) var<storage, read>           dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>  bp_count : array<u32>;
@group(0) @binding(2) var<storage, read>              cs : CFLStruct;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let r = gid.x;      // row index
    let c = gid.y;      // column index
    let A = gid.z;      // nonterminal index

    let N  = cs.numStates;
    let NT = cs.numNonterminals;

    if (c <= r) { return; }
    
    let snt     = N * NT;
    let dpIdx   = r*snt + c*NT + A;
    let startGC = getGrammarOffset(A);
    let endGC   = select(cs.grammarFlattenedSize, getGrammarOffset(A + 1u), A + 1u < NT);
    let aoi     = r*N + c + 1u;
    let pairOffset     = getMdptOffset(aoi - 1u);
    let pairOffsetNext = select(cs.mdptsSize, getMdptOffset(aoi), aoi < cs.mdptsOffsetsSize);

    // If dp_in[dpIdx] does NOT have bit 0 set, then bp_count = 0
    if ((dp_in[dpIdx] & 0x01u) == 0u) { bp_count[dpIdx] = 0; return; }

    // Otherwise accumulate the possible backpointers
    var count = 0;
    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = allFSAPairs[pairIdx];

        var g = startGC;
        while (g < endGC) {
            let B = vidx[g];
            let C = vidx[g + 1];

            let idxBM = r      * snt + mdpt * nnt + B;
            let idxMC = mdpt   * snt + c    * nnt + C;
            if ((dp_in[idxBM] != 0u) && (dp_in[idxMC] != 0u)) { count++; }
            g += 2;
        }
    }

    bp_count[dpIdx] = count;
}""".trimIndent())

//language=wgsl
val bp_write by Shader(CFL_STRUCT + """
@group(0) @binding(0) var<storage, read>             dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>    bp_count : array<u32>;
@group(0) @binding(1) var<storage, read_write>   bp_offset : array<u32>;
@group(0) @binding(1) var<storage, read_write>  bp_storage : array<u32>;
@group(0) @binding(2) var<storage, read>                cs : CFLStruct;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;      // row index
    let c = gid.y;      // column index
    let A = gid.z;      // nonterminal index

    let N  = cs.numStates;
    let NT = cs.numNonterminals;

    if (c <= r) { return; }
    
    let snt     = N * NT;
    let dpIdx   = r*snt + c*NT + A;
    let startGC = getGrammarOffset(A);
    let endGC   = select(cs.grammarFlattenedSize, getGrammarOffset(A + 1u), A + 1u < NT);
    let aoi     = r*N + c + 1u;
    let pairOffset     = getMdptOffset(aoi - 1u);
    let pairOffsetNext = select(cs.mdptsSize, getMdptOffset(aoi), aoi < cs.mdptsOffsetsSize);

    // If dp_in[dpIdx] does NOT have bit 0 set, then bp_count = 0
    if ((dp_in[dpIdx] & 0x01u) == 0u) { return; }

    var outPos = bp_offset[dpIdx];

    // Similar to bp_count, but we record expansions in bp_storage
    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = allFSAPairs[pairIdx];

        var poff = startGC;
        while (poff < endGC) {
            let B = vidx[poff];
            let C = vidx[poff + 1];

            let idxBM = r    * snt + mdpt * nnt + B;
            let idxMC = mdpt * snt + c    * nnt + C;

            if ((dp_in[idxBM] != 0u) && (dp_in[idxMC] != 0u)) {
                // Each record is 2 ints
                bp_storage[outPos * 2 + 0] = idxBM;
                bp_storage[outPos * 2 + 1] = idxMC;
                outPos++;
            }
            poff += 2;
        }
    }
}""".trimIndent())

//language=wgsl
val prefix_sum_p1 by Shader("""
@compute @workgroup_size(1024) fn main(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(workgroup_id)         groupId : vec3<u32>,
    @builtin(local_invocation_id)  localId : vec3<u32>
) {
    let N     = prefixUni.N; // total length
    let gid   = globalId.x;
    let grpId = groupId.x;
    let lid   = localId.x;
    let tpg   = 1024u; // fixed to match @workgroup_size(1024)

    var<workgroup> tile: array<u32, 1024>;

    let base = grpId * tpg;
    let idx  = base + lid;

    let v = select(0, outBuf[idx], idx < N);
    tile[lid] = v;
    workgroupBarrier();

    var offset = 1u;
    while (offset < tpg) {
        let n = select(0, tile[lid - offset], lid >= offset);
        workgroupBarrier();
        tile[lid] = tile[lid] + n;
        workgroupBarrier();

        offset = offset << 1;
    }

    let inclusive = tile[lid];
    tile[lid] = inclusive - v;

    // If last thread in this group or last element in entire array
    if ((idx + 1u == N) || (lid == tpg - 1u)) { blkSum[grpId] = inclusive; }

    if (idx < N) { outBuf[idx] = tile[lid]; }
}""")

//language=wgsl
val prefix_sum_p2 by Shader("""
@compute @workgroup_size(1024) fn main(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(workgroup_id)         groupId : vec3<u32>,
    @builtin(local_invocation_id)  localId : vec3<u32>
) {
    let N     = prefixUni.N;
    let grpId = groupId.x;
    let lid   = localId.x;
    let tpg   = 1024u;

    if (grpId == 0u) { return; }

    let offsetVal = blkSum[grpId - 1u];
    let idx = grpId * tpg + lid;

    if (idx < N) { outBuf[idx] = outBuf[idx] + offsetVal; }
}""".trimIndent())

//language=wgsl
val sample_words by Shader("""
// Example uniform struct for the small constant values:
struct Uniforms {
    numStartIndices : u32,
    numStates       : u32,
    maxWordLen      : u32,
};

@group(0) @binding(0) var<storage, read> dp_in: array<u32>;
@group(0) @binding(1) var<storage, read> bp_count: array<u32>;
@group(0) @binding(2) var<storage, read> bp_offset: array<u32>;
@group(0) @binding(3) var<storage, read> bp_storage: array<u32>;
@group(0) @binding(4) var<storage, read> startIndices: array<u32>;
@group(0) @binding(5) var<storage, read> seeds: array<u32>;
@group(0) @binding(6) var<storage, read_write> sampledWords: array<u32>;
@group(0) @binding(7) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;

    // If needed, clamp to numStartIndices:
    if (tid >= u32(samplerUni.numStartIndices)) { return; }

    // We'll create a local array in function scope
    var localWord: array<u32, 1024>;
    // Zero-initialize
    for (var i = 0; i < samplerUni.maxWordLen; i = i + 1) { localWord[i] = 0u; }

    // Grab rngState from seeds[tid]
    var rngState = seeds[tid];

    // Choose a random index in [0..numStartIndices-1]
    let rIndex   = lcg_randomRange(&rngState, u32(samplerUni.numStartIndices));
    let dpIndex  = startIndices[u32(rIndex)];

    // decode row & col if needed:
    //    A = dpIndex % numNonterminals
    //    rowCol = dpIndex / numNonterminals
    //    c = rowCol % numStates
    //    r = rowCol / numStates
    //    then startIdx = r*snt + c*numNonterminals + A
    let nnt      = uni.numNonterminals;
    let A        = dpIndex % nnt;
    let rowCol   = dpIndex / nnt;
    let c        = rowCol % uni.numStates;
    let r        = rowCol / uni.numStates;
    let snt      = uni.numStates * nnt;
    let startIdx = r * snt + c * nnt + A;

    // Now call sampleTopDown
    sampleTopDown(dp_in, bp_count, bp_offset, bp_storage, startIdx, &rngState, &localWord, samplerUni.maxWordLen);

    // Copy to sampledWords buffer
    // dp_in was 16 bits, but in original code we wrote into a “char”
    // so we do a cast from u32 -> u8 (truncation).
    for (var i = 0; i < samplerUni.maxWordLen; i = i + 1) {
        sampledWords[u32(tid) * u32(samplerUni.maxWordLen) + u32(i)] = u8(localWord[i] & 0x00FFu);
    }
}

fn lcg_random(stateRef: ptr<function, u32>) -> u32 {
    let newVal = (1664525u * (*stateRef)) + 1013904223u;
    *stateRef = newVal;
    return newVal;
}

fn lcg_randomRange(stateRef: ptr<function, u32>, range: u32) -> u32 { return lcg_random(stateRef) % range; }

fn sampleTopDown(
    dp_in       : array<u32>,
    bp_count     : array<u32>,
    bp_sffset    : array<u32>,
    bp_storage   : array<u32>,
    startDPIdx  : u32,
    rngStateRef : ptr<function, u32>,
    localWord   : ptr<function, array<u32, 1024>>,
    maxWordLen  : u32
) {
    // We'll create a stack up to 1024
    var stack: array<u32, 1024>;
    var top     = 0;
    var wordLen = 0;

    stack[top] = startDPIdx; top++;

    // We interpret uni.numNonterminals, etc., as global from `uni`.
    let nnt = uni.numNonterminals;

    // main loop
    var iter = 0;
    loop {
        if ((iter >= maxWordLen * 98) || (top <= 0) || (wordLen >= maxWordLen - 5)) { break; }
        iter++;

        top = top - 1;
        let dpIdx = abs(stack[top]);

        let expCount = bp_count[dpIdx];
        // dp_in[dpIdx] >> 1 => checks high bits
        // dp_in[dpIdx] & 0x01 => checks the binary expansion bit
        let predicate = dp_in[dpIdx];
        let canRecurse = ((predicate & 0x01u) != 0u);

        // If ( (predicate >> 1u) != 0 ) => the node has some “literal” bits set
        // Or we randomly choose to treat it as a leaf ...
        let hasLiteral = ((predicate >> 1u) != 0u);

        // Random check:
        let rVal = u32(lcg_randomRange(rngStateRef, u32(expCount))); 
        // Original logic:
        // if ( (dp_in[dpIdx] >> 1) && ( ! (dp_in[dpIdx] & 0x01) || (rVal % 2 == 0) ) ) { ... }
        if (hasLiteral && ( !canRecurse || ((rVal % 2) == 0) )) {
            // treat as leaf/terminal
            let nonterminal = dpIdx % nnt;
            let isNegative  = ((predicate & 0x8000u) != 0u);
            let literal     = (predicate >> 1u) & 0x7FFFu;
            let numTms      = nt_tm_lens[nonterminal];
            let ntOffset    = offsets[nonterminal];

            if (isNegative) {
                // choose from among all possible except “literal - 1”
                var possibleTms: array<u32, 100>;
                var tmCount = 0;
                for (var i = 0; i < min(numTms, 100); i = i + 1) {
                    if (i != (literal - 1u)) {
                        possibleTms[tmCount] = all_tm[u32(ntOffset) + i]; tmCount++;
                    }
                }
                let choiceIndex = u32(lcg_randomRange(rngStateRef, u32(tmCount)));
                let tmChoice    = possibleTms[choiceIndex];
                (*localWord)[wordLen] = tmChoice + 1u; wordLen++;
            } else {
                // positive literal
                if (numTms != 0) {
                    let tmVal = all_tm[u32(ntOffset) + u32(literal) - 1];
                    (*localWord)[wordLen] = tmVal + 1u;
                } else { (*localWord)[wordLen] = 99u; }
                wordLen++;
            }
        } else {
            // do a binary expansion if there's room on the stack
            if (top + 2 < 1024) {
                let randIdx = bp_offset[dpIdx] + rVal;
                let idxBM   = bp_storage[randIdx * 2 + 0];
                let idxMC   = bp_storage[randIdx * 2 + 1];
                stack[top]  = idxMC; top++;
                stack[top]  = idxBM; top++;
            }
        }
    }
}""".trimIndent())

class Shader(val src: String) {
  lateinit var name: String
  lateinit var pipeline: GPUComputePipeline

  companion object {
    private suspend fun makePipeline(wgsl: String, entryPoint: String = "main"): GPUComputePipeline =
      gpu.createComputePipeline(
        GPUComputePipelineDescriptor(
          layout = "auto",
          compute = GPUProgrammableStage(
            module = gpu.createShaderModule(GPUShaderModuleDescriptor(code = wgsl)),
            entryPoint = entryPoint
          )
        )
      )

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

    fun IntArray.makeBuffer(usage: Int): GPUBuffer =
      Int32Array<ArrayBuffer>(size).apply { set(this@makeBuffer.toTypedArray(), 0) }
        .let { makeBuffer(sz = size * 4, us = usage, data = it) }

    fun makeBuffer(sz: Int, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
      gpu.createBuffer(GPUBufferDescriptor(size = sz.toDouble(), usage = us))
        .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }

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
  }

  suspend fun bind() { pipeline = makePipeline(src) }

  operator fun getValue(tr: Any?, property: KProperty<*>): Shader = this.also { name = property.name }

  /** Copies [count] 32-bit integers from this GPU buffer back to CPU as a List<Int>. */
  suspend fun GPUBuffer.readInts(count: Int): IntArray {
    val bytesNeeded = (count * 4)
    val readDst = makeBuffer(bytesNeeded, GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
    val cmd = gpu.createCommandEncoder()
    cmd.copyBufferToBuffer(this, 0.0, readDst, 0.0, bytesNeeded.toDouble())
    gpu.queue.submit(arrayOf(cmd.finish()))
    (readDst.mapAsync(1) as Promise<*>).await()
    return Int32Array(readDst.getMappedRange()).asList().toIntArray()
  }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader
  suspend fun invokeCFLFixpoint(vararg inputs: IntArray): IntArray {
    var t0 = TimeSource.Monotonic.markNow()
    require(inputs.size >= 2) { "Expected at least dpIn + metadata, got ${inputs.size} buffers." }

    val dpIn = inputs[0].makeBuffer(usage = GPUBufferUsage.STCPSD)
    val metaBuf = inputs[1].makeBuffer(usage = GPUBufferUsage.STORAGE + GPUBufferUsage.COPY_DST)

    val (numStates, numNonterminals) = inputs[1]
    val dpOut = makeBuffer(dpIn.size.toInt(), dpIn.usage)
    val changesBuf = makeBuffer(sz = 4, us = GPUBufferUsage.STCPSD)

    var prevValue = -1
    val maxRounds = numStates

    println("Time to iter: ${t0.elapsedNow()}")
    repeat(maxRounds) { round ->
      gpu.queue.writeBuffer(changesBuf, 0.0, data = Int32Array<ArrayBuffer>(1).apply { set(arrayOf(0), 0) })
      val (inBuf, outBuf) = if (round % 2 == 0) dpIn to dpOut else dpOut to dpIn

      val cmdEnc = gpu.createCommandEncoder()
      cmdEnc.beginComputePass().apply {
        setPipeline(pipeline)
        setBindGroup(0, bindBuffers(pipeline, inBuf, outBuf, metaBuf, changesBuf))
        // The kernel uses (r, c, A) => [numStates, numStates, numNonterminals]
        dispatchWorkgroups(numStates, numStates, numNonterminals)
        end()
      }
      gpu.queue.submit(arrayOf(cmdEnc.finish()))

      val changesThisRound = changesBuf.readInts(1)[0]

      if (changesThisRound == prevValue) {
        println("Fixpoint reached at round=$round")
        return outBuf.readInts(dpIn.size.toInt() / 4)
      }
      prevValue = changesThisRound
      println("Round=$round, changes=$changesThisRound, time=${t0.elapsedNow()}")
    }

    return (if (maxRounds % 2 == 0) dpIn else dpOut).readInts(dpIn.size.toInt() / 4)
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
      dispatchWorkgroups(threads, threads)
      end()
    }

    return readFrom.readInts(readFrom.size.toInt())
  }
}

fun packStruct(constants: List<Int>, vararg arrays: IntArray): IntArray {
  val offsets = arrays.scan(constants.size + arrays.size * 2) { acc, arr -> acc + arr.size }.dropLast(1)

  val header = buildList {
    addAll(0, constants)
    arrays.forEachIndexed { index, arr ->
      add(offsets[index]) // Offset for this array
      add(arr.size)       // Length of this array
    }
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