@file:OptIn(ExperimentalUnsignedTypes::class)

import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import js.array.asList
import js.buffer.*
import js.typedarrays.Int32Array
import kotlinx.coroutines.await
import web.gpu.*
import web.performance.performance
import kotlin.js.Promise
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.reflect.KProperty
import kotlin.time.*

suspend fun webGPURepair() {
  val cfg = pythonStatementCNFAllProds

  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE"
    .tokenizeByWhitespace()
  val radius = 5
  val levFSA = makeLevFSA(pythonCode, radius)
  val maxWordLen = pythonCode.size + radius + 10

  val numStates = levFSA.numStates

  fun List<List<List<Int>>>.prefixScan(): Pair<IntArray, IntArray> =
    measureTimedValue {
      fun IntArray.prefixSumSizes(): IntArray =
        IntArray(size + 1).also { prefix ->
          copyInto(prefix, destinationOffset = 1)
          for (i in 1..size) prefix[i] += prefix[i - 1]
        }

      // 1) Filter each triple-nested list
      val filtered = mapIndexed { p, outer ->
        outer.mapIndexed { q, inner -> inner.filter { it in (p + 1) until q } }
      }.flatten()

      // 2) Flatten all integers
      val flattened = filtered.flatten().toIntArray()

      // 3) Build an offsets array with leading 0 and cumulative sums
      val offsets = filtered.map { it.size }.toIntArray().prefixSumSizes()

      flattened to offsets
    }.also { println("Completed prefix scan in: ${it.duration}") }.value

  val (allFSAPairsFlattened, allFSAPairsOffsets) = levFSA.midpoints.prefixScan()
  val clock = TimeSource.Monotonic.markNow()

  fun FSA.byteFormat(cfg: CFG): UIntArray {
    val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it] ?: emptyList() }
    // 0 and 1 are reserved for (0) no parse exists and (1) parse exists, but an internal nonterminal node
    // Other byte values are used to denote the presence (+) or absence (-) of a leaf terminal
    fun StrPred.predByte(A: Int): UInt = (
      if (arg == "[.*]" || (arg.startsWith("[!=]") && arg.drop(4) !in terminalLists[A])) Int.MAX_VALUE - 1 // All possible terminals
      else if (arg.startsWith("[!=]")) (1.shl(32) + (terminalLists[A].indexOf(arg.drop(4)) + 1).shl(1)) // Represent negation using sign bit
      else (terminalLists[A].indexOf(arg) + 1).shl(1)
    ).toUInt() // Otherwise positive sign bit

    return cfg.unitProductions.flatMap { (A, σ) ->
      nominalForm.flattenedTriples.filter { arc -> arc.second(σ) }.map { (q0, sp, q1) ->
        val Aidx = cfg.bindex[A]
        // row, col, Aidx, terminal encoding
        val pb = sp.predByte(Aidx)
        listOf(stateMap[q0]!!, stateMap[q1]!!, Aidx).map { it.toUInt() } + pb
      }
    }.distinct().flatten().toUIntArray()
  } // Maybe create dense chart directly instead since no FFI transfer cost?

  // TM layout (for decoding)
//  val terminalLists = cfg.nonterminals.map { cfg.bimap.UNITS[it]?.map { cfg.tmMap[it]!! } ?: emptyList() }
//  val all_tm = terminalLists.flatMap { it }.toIntArray()
//  val nt_tm_offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1).toIntArray()
//  val nt_tm_lens = terminalLists.map { it.size }.toIntArray()
//  val numNonterminals = cfg.nonterminals.size

  val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
  val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()

  packCFLStruct(
    levFSA.byteFormat(cfg),

    // FSA Encoding
    allFSAPairsFlattened,
    allFSAPairsOffsets,
    levFSA.finalIdxs.toIntArray(),
    numStates,

    // CFG Encoding
    grammarFlattened,
    grammarOffsets,
  )

  println("Round trip repair: ${clock.elapsedNow()}")
}

/**
 * Packs the following data into a single Int32Array matching the WGSL struct:
 *
 *   dpIn:               UIntArray (converted to Int32)
 *   mdpts:              IntArray
 *   mdptsOffsets:       IntArray
 *   acceptStates:       IntArray
 *   acceptStatesSize:   Int
 *   numStates:          Int
 *   grammarFlattened:   IntArray
 *   grammarOffsets:     IntArray
 *   vilen:              Int
 *   volen:              Int
 */

fun packCFLStruct(
  dpIn: UIntArray,
  mdpts: IntArray,
  mdptsOffsets: IntArray,
  acceptStates: IntArray,
  numStates: Int,
  grammarFlattened: IntArray,
  grammarOffsets: IntArray,
): Int32Array<ArrayBuffer> {
  // -- Convert dpIn to signed ints (if needed) --
  val dpInInt = IntArray(dpIn.size) { dpIn[it].toInt() }

  // -- Compute sizes --
  val dpInSize             = dpInInt.size
  val mdptsSize            = mdpts.size
  val mdptsOffsetsSize     = mdptsOffsets.size
  val acceptStatesLen      = acceptStates.size
  val grammarFlattenedSize = grammarFlattened.size
  val grammarOffsetsSize   = grammarOffsets.size

  // WGSL struct has 13 fields in the header (each 1 x i32/u32 = 4 bytes).
  //   1) dpInOffset
  //   2) dpInSize
  //   3) mdptsOffset
  //   4) mdptsSize
  //   5) mdptsOffsetsOffset
  //   6) mdptsOffsetsSize
  //   7) acceptStatesOffset
  //   8) acceptStatesSize
  //   9) numStates
  //   10) grammarFlattenedOffset
  //   11) grammarFlattenedSize
  //   12) grammarOffsetsOffset
  //   13) grammarOffsetsSize
  val HEADER_COUNT = 13

  // sum of all array lengths
  val tailCount = dpInSize + mdptsSize + mdptsOffsetsSize +
      acceptStatesLen + grammarFlattenedSize + grammarOffsetsSize

  // total number of 32-bit elements:
  val totalInts = HEADER_COUNT + tailCount

  val bufferData = IntArray(totalInts)

  var offset = HEADER_COUNT

  // 1) dpIn
  val dpInOffset = offset
  for ((i, v) in dpInInt.withIndex()) { bufferData[offset + i] = v }
  offset += dpInSize

  // 2) mdpts
  val mdptsOffset = offset
  for ((i, v) in mdpts.withIndex()) { bufferData[offset + i] = v }
  offset += mdptsSize

  // 3) mdptsOffsets
  val mdptsOffsetsOffset = offset
  for ((i, v) in mdptsOffsets.withIndex()) { bufferData[offset + i] = v }
  offset += mdptsOffsetsSize

  // 4) acceptStates
  val acceptStatesOffset = offset
  for ((i, v) in acceptStates.withIndex()) { bufferData[offset + i] = v }
  offset += acceptStatesLen

  // 5) grammarFlattened
  val grammarFlattenedOffset = offset
  for ((i, v) in grammarFlattened.withIndex()) { bufferData[offset + i] = v }
  offset += grammarFlattenedSize

  // 6) grammarOffsets
  val grammarOffsetsOffset = offset
  for ((i, v) in grammarOffsets.withIndex()) { bufferData[offset + i] = v }
  offset += grammarOffsetsSize

   bufferData[0] = dpInOffset
   bufferData[1] = dpInSize
   bufferData[2] = mdptsOffset
   bufferData[3] = mdptsSize
   bufferData[4] = mdptsOffsetsOffset
   bufferData[5] = mdptsOffsetsSize
   bufferData[6] = acceptStatesOffset
   bufferData[7] = acceptStatesLen
   bufferData[8] = numStates
   bufferData[9] = grammarFlattenedOffset
  bufferData[10] = grammarFlattenedSize
  bufferData[11] = grammarOffsetsOffset
  bufferData[12] = grammarOffsetsSize

  // Done! Return the Int32Array with everything packed.
  return Int32Array<ArrayBuffer>(totalInts).apply { set(bufferData.toTypedArray(), 0) }
}

//language=wgsl
val WGSL_STRUCT = """struct CFLStruct {
    // dpIn
    dpInOffset        : u32,
    dpInSize          : u32,

    // mdpts
    mdptsOffset       : u32,
    mdptsSize         : u32,

    // mdptsOffsets
    mdptsOffsetsOffset: u32,
    mdptsOffsetsSize  : u32,

    // acceptStates
    acceptStatesOffset: u32,
    acceptStatesSize  : u32,

    // numStates
    numStates         : u32,

    // grammarFlattened
    grammarFlattenedOffset: u32,
    grammarFlattenedSize  : u32,

    // grammarOffsets
    grammarOffsetsOffset  : u32,
    grammarOffsetsSize    : u32,

    tail : array<i32>,
};""".trimIndent()

//language=wgsl
val CFL_ITER by Shader("""$WGSL_STRUCT

  @group(0) @binding(0) var<storage, read> cflStruct: CFLStruct;

  fn getDpIn(index: u32) -> u32 { return fsaStruct.tail[cflStruct.dpInOffset + index]; }

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      //...
  }
""".trimIndent())

/*
private func iterateFixpoint(
bufA: MTLBuffer,
pairsBuf: MTLBuffer,
pairsOffBuf: MTLBuffer,
numStates: Int,
allPairsSize: Int,
allPairsOffsetsSize: Int
) -> MTLBuffer {
  var N = numStates, ap = allPairsSize, ao = allPairsOffsetsSize
  let totalCount = N * N * numNonterminals
      let dpSizeBytes = totalCount * MemoryLayout<UInt16>.stride
  let stride = MemoryLayout<Int32>.stride
  let totalPairs = N * (N - 1) / 2
  let totalThreads = totalPairs * numNonterminals

      let numNonzero = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
  var prevValue: UInt32 = 0

  for r in 0..<numStates {
    var zero: UInt32 = 0
    memcpy(numNonzero.contents(), &zero, MemoryLayout<UInt32>.size)

    runKernel("cfl_mul_upper", totalThreads) { enc in
      enc.setBuffer(bufA,        offset: 0,      index: 0)
      enc.setBuffer(bufA,        offset: 0,      index: 1)
      enc.setBuffer(pairsBuf,    offset: 0,      index: 2)
      enc.setBuffer(pairsOffBuf, offset: 0,      index: 3)
      enc.setBytes(&N,           length: stride, index: 4)
      enc.setBytes(&ap,          length: stride, index: 5)
      enc.setBytes(&ao,          length: stride, index: 6)
      enc.setBuffer(numNonzero,  offset: 0,      index: 7)
    }

    let currentValue = numNonzero.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
    if (currentValue == prevValue) {
      print("Fixpoint escape at round: \(r)/\(numStates), total=\(currentValue), size: \(dpSizeBytes)"); fflush(stdout)
      break
    } else { prevValue = currentValue }
  }

  return bufA
}*/

suspend fun iterateCFL(
  bufA: GPUBuffer,
  pairsBuf: GPUBuffer,
  pairsOffBuff: GPUBuffer,
  numStates: Int,
  allPairsSize: Int,
  allPairsOffsetsSize: Int
): GPUBuffer { TODO() }

suspend fun benchmarkWGPU() {
  val N = 1024
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

suspend fun iterateGPU(input: Array<Int>, P: Int): Int {
  val n = sqrt(input.size.toDouble()).toInt()
  val s = input.size
  val bytes = s * 4

  val bufM = makeBuffer(bytes, 140, Int32Array<ArrayBuffer>(s).apply { set(input, 0) })
  val bufP = makeBuffer(16, 72, Int32Array<ArrayBuffer>(4).apply { set(arrayOf(n), 0) })

  val bufS = WGSL_GEMX_ITERATE.invoke(bufM, bufP, threads = n, iterations = P)

  return Int32Array(bufS).asList().hashCode()
}

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

    private fun bindBuffers(pipeline: GPUComputePipeline, vararg buffers: GPUBuffer): GPUBindGroup {
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

  suspend fun bind() { pipeline = makePipeline(src) }

  operator fun getValue(tr: Any?, property: KProperty<*>): Shader = this.also { name = property.name }

  // Invocation strategies: eliminates some of the ceremony of calling a GSL shader

  suspend operator fun invoke(vararg inputs: GPUBuffer, threads: Int, iterations: Int = 1): ArrayBuffer {
    val encoder: GPUCommandEncoder = gpu.createCommandEncoder()

    val buf1 = inputs[0] // Initial input buffer
    val param = inputs[1] // Uniform buffer

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