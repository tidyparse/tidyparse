@file:OptIn(ExperimentalUnsignedTypes::class)

import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import js.array.asList
import js.buffer.*
import js.typedarrays.Int32Array
import kotlinx.browser.document
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.async
import kotlinx.coroutines.await
import kotlinx.dom.appendText
import org.w3c.dom.HTMLDivElement
import web.events.EventType
import web.events.addEventListener
import web.gpu.*
import web.performance.performance
import kotlin.apply
import kotlin.js.Promise
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.reflect.KProperty
import kotlin.time.*

lateinit var gpu: GPUDevice
var gpuAvailable = false

fun tryBootstrapGPU() {
  MainScope().async {
    checkWebGPUAvailability()
    if (gpuAvailable) {
      WGSL_GEMX_ITERATE.bind()
      cfl_mul_upper.bind()
      benchmarkWGPURepair()
    }
  }
}

suspend fun checkWebGPUAvailability() {
  val tmpDev = (navigator.gpu as? GPU)?.requestAdapter()?.requestDevice()?.also { gpu = it }
  val gpuAvailDiv = document.getElementById("gpuAvail") as HTMLDivElement

  if (tmpDev != null) {
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
    gpuAvailDiv.appendText("WebGPU is NOT available.")
  }
}

suspend fun benchmarkWGPURepair() {
  val cfg = pythonStatementCNFAllProds

  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE"
    .tokenizeByWhitespace()
  val radius = 5
  val levFSA = makeLevFSA(pythonCode, radius)

  fun List<List<List<Int>>>.prefixScan(): Pair<IntArray, IntArray> = // TODO: move into shader kernel
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

  val (allFSAPairsFlattened, allFSAPairsOffsets) = levFSA.midpoints.prefixScan()
  val clock = TimeSource.Monotonic.markNow()

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

  val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
  val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()

  val dpIn = levFSA.byteFormat(cfg).let { dpi -> IntArray(dpi.size) { dpi[it].toInt() } }

  /** Memory layout: [WGSL_STRUCT] */
  val metadata: Int32Array<ArrayBuffer> = packStruct(
    constants = listOf(levFSA.numStates, cfg.nonterminals.size),

    // FSA Encoding
    allFSAPairsFlattened,
    allFSAPairsOffsets,
    levFSA.finalIdxs.toIntArray(),

    // CFG Encoding
    grammarFlattened,
    grammarOffsets,
  )

  println("Starting repair...")
  val dpComplete = iterateFixpoint(metadata, dpIn)

  println("Round trip repair: ${clock.elapsedNow()}")
}

suspend fun iterateFixpoint(
  packedMetadata: Int32Array<ArrayBuffer>,
  dpInitial: IntArray
): IntArray {
  gpu.pushErrorScope(GPUErrorFilter.validation)

  // 1) Make the metadata buffer
  val metaSizeBytes = packedMetadata.length * 4
  val metaBuf = makeBuffer(
    sz = metaSizeBytes,
    us = GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST,
    data = packedMetadata
  )

  // 2) Make two DP chart buffers for ping-pong:
  val dpBuf1 = makeBuffer(
    sz = dpInitial.size * 4,
    us = GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_SRC or GPUBufferUsage.COPY_DST,
    data = Int32Array<ArrayBuffer>(dpInitial.size).apply { set(dpInitial.toTypedArray(), 0) }
  )
  // dpBuf2 is initialized to the same data, or zeros, or doesn’t matter if you overwrite it fully anyway:
  val dpBuf2 = makeBuffer(
    sz = dpInitial.size * 4,
    us = GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_SRC or GPUBufferUsage.COPY_DST
  )

  // 3) Atomic changes buffer: 4 bytes
  val changesBuf = makeBuffer(
    sz = 4,
    us = GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_SRC or GPUBufferUsage.COPY_DST
  )

  val numStates       = packedMetadata[0]
  val numNonterminals = packedMetadata[1]
  val totalThreads    = (numStates*(numStates -1))/2 * numNonterminals

  println("Total threads: $totalThreads")
  println("Maxwork: ${gpu.limits.maxComputeWorkgroupsPerDimension}")

  var prevValue = -1

  // For each round, we select which buffer is "dp_in" and which is "dp_out":
  for (round in 0 until numStates) {
    println("Round: $round")

    // (a) Zero out changesBuf
    gpu.queue.writeBuffer(changesBuf, 0.0, Int32Array<ArrayBuffer>(arrayOf(0)))

    // Pick dpIn/dpOut by round
    val dpIn  = if (round % 2 == 0) dpBuf1 else dpBuf2
    val dpOut = if (round % 2 == 0) dpBuf2 else dpBuf1

    // (b) Dispatch the kernel
    val encoder = gpu.createCommandEncoder()
    val pass = encoder.beginComputePass()
    pass.setPipeline(cfl_mul_upper.pipeline)

    // dpIn is read-only (binding=0), dpOut is read-write (binding=1)
    pass.setBindGroup(
      0,
      Shader.bindBuffers(
        cfl_mul_upper.pipeline,
        dpIn,       // dp_in  (read)
        dpOut,      // dp_out (write)
        metaBuf,    // cfl_struct
        changesBuf  // atomicChange
      )
    )

    pass.dispatchWorkgroups(numStates, numStates, numNonterminals)
    pass.end()

    gpu.queue.submit(arrayOf(encoder.finish()))

    // (c) Copy just the 4 bytes of changesBuf back to CPU
    val readEncoder = gpu.createCommandEncoder()
    val readOut = makeBuffer(4, GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
    readEncoder.copyBufferToBuffer(changesBuf, 0.0, readOut, 0.0, 4.0)
    gpu.queue.submit(arrayOf(readEncoder.finish()))

    (readOut.mapAsync(1) as Promise<*>).await()
    val arr   = Int32Array(readOut.getMappedRange())
    val count = arr[0]
    println("Count: $count")
    if (count == prevValue) {
      println("Fixpoint reached at round=$round changes=$count")
      break
    }
    prevValue = count
  }

  // 4) We have converged or used up all rounds -> read final DP
  // The "final" buffer is whichever was used as the "dp_out" in the last iteration
  val finalBuf = if (numStates % 2 == 0) dpBuf1 else dpBuf2

  val outEncoder = gpu.createCommandEncoder()
  val outBuf = makeBuffer(dpInitial.size * 4, GPUBufferUsage.COPY_DST or GPUBufferUsage.MAP_READ)
  outEncoder.copyBufferToBuffer(
    source            = finalBuf,
    sourceOffset      = 0.0,
    destination       = outBuf,
    destinationOffset = 0.0,
    size              = (dpInitial.size * 4).toDouble()
  )
  gpu.queue.submit(arrayOf(outEncoder.finish()))

  (outBuf.mapAsync(1) as Promise<*>).await()
  val finalArray = Int32Array(outBuf.getMappedRange())

  val error = gpu.popErrorScopeAsync().await()
  if (error != null) {
    console.error("GPU validation/usage error:", error.message)
  } else {
    console.log("No GPU errors, got result of size = ${finalArray.asList().size}")
  }

  return finalArray.asList().toIntArray()
}

fun packStruct(constants: List<Int>, vararg arrays: IntArray): Int32Array<ArrayBuffer> {
  val offsets = arrays.scan(constants.size + arrays.size * 2) { acc, arr -> acc + arr.size }.dropLast(1)

  val header = buildList {
    addAll(0, constants)
    arrays.forEachIndexed { index, arr ->
      add(offsets[index]) // Offset for this array
      add(arr.size)       // Length of this array
    }
  }

  val buffer = header + arrays.flatMap { it.asIterable() }

  return Int32Array<ArrayBuffer>(buffer.size).apply { set(buffer.toTypedArray(), 0) }
}

//language=wgsl
val cfl_mul_upper by Shader("""
struct CFLStruct {
             numStates : u32,      numNonterminals : u32,
             
           mdptsOffset : u32,            mdptsSize : u32,
    mdptsOffsetsOffset : u32,     mdptsOffsetsSize : u32,
    acceptStatesOffset : u32,     acceptStatesSize : u32,
grammarFlattenedOffset : u32, grammarFlattenedSize : u32,
  grammarOffsetsOffset : u32,   grammarOffsetsSize : u32,
  
               payload : array<u32>
};

struct AtomicChange { count: atomic<u32> };

         fn getMdpt(index: u32) -> u32 { return cs.payload[cs.mdptsOffset + index]; }
   fn getMdptOffset(index: u32) -> u32 { return u32(cs.payload[cs.mdptsOffsetsOffset + index]); }
fn getGrammarSymbol(index: u32) -> u32 { return u32(cs.payload[cs.grammarFlattenedOffset + index]); }
fn getGrammarOffset(index: u32) -> u32 { return u32(cs.payload[cs.grammarOffsetsOffset + index]); }

@group(0) @binding(0) var<storage, read>         dp_in : array<u32>;
@group(0) @binding(1) var<storage, read_write>  dp_out : array<u32>;
@group(0) @binding(2) var<storage, read>            cs : CFLStruct;
@group(0) @binding(3) var<storage, read_write> changes : AtomicChange;

@compute @workgroup_size(1, 1, 1) fn cfl_mul_upper(@builtin(global_invocation_id) gid : vec3<u32>) {
    let r = gid.x;      // row index
    let c = gid.y;      // column index
    let A = gid.z;      // nonterminal index

    let N  = cs.numStates;
    let NT = cs.numNonterminals;

    // If c <= r, skip to ensure upper triangle
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

            g = g + 2u;
        }
    }
}""".trimIndent())

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

@compute @workgroup_size(1,1,1)
fn WGSL_GEMX_ITERATE(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        GPUBindGroupEntry(
          binding = index,
          resource = jsObject<GPUBindingResource> { buffer = buf }
        )
      }.toTypedArray()
      return gpu.createBindGroup(GPUBindGroupDescriptor(layout = lay, entries = ent))
    }
  }

  suspend fun bind() { pipeline = makePipeline(src, name) }

  operator fun getValue(tr: Any?, property: KProperty<*>): Shader = this.also { name = property.name }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader
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

    val output = makeBuffer(finalOut.size.toInt(), 9) // MAP_READ + COPY_DST
    encoder.copyBufferToBuffer(finalOut, 0.0, output, 0.0, output.size)
    gpu.queue.submit(arrayOf(encoder.finish()))

    (output.mapAsync(1) as Promise<*>).await()
    return output.getMappedRange()
  }

  suspend operator fun invoke(vararg inputs: GPUBuffer, readFrom: GPUBuffer, threads: Int): ArrayBuffer {
    val encoder: GPUCommandEncoder = gpu.createCommandEncoder()
    encoder.beginComputePass().apply {
      setPipeline(pipeline)
      setBindGroup(0, bindBuffers(pipeline, *inputs))
      dispatchWorkgroups(threads, threads)
      end()
    }

    val output = makeBuffer(readFrom.size.toInt(), 9)
    encoder.copyBufferToBuffer(readFrom, 0.0, output, 0.0, output.size)
    gpu.queue.submit(arrayOf(encoder.finish()))

    (output.mapAsync(1) as Promise<*>).await()
    return output.getMappedRange()
  }
}

fun makeBuffer(sz: Int, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
  gpu.createBuffer(GPUBufferDescriptor(size = sz.toDouble(), usage = us))
    .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }