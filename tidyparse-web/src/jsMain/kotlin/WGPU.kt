@file:OptIn(ExperimentalUnsignedTypes::class)

import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import js.array.asList
import js.buffer.*
import js.json.parse
import js.typedarrays.Int32Array
import kotlinx.coroutines.await
import web.gpu.*
import web.performance.performance
import kotlin.js.Promise
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.reflect.KProperty
import kotlin.time.*

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

  /** Memory layout: [WGSL_STRUCT] */
  val metadata: Int32Array<ArrayBuffer> = packStruct(
    constants = listOf(levFSA.numStates, cfg.nonterminals.size),

    levFSA.byteFormat(cfg).let { dpIn -> IntArray(dpIn.size) { dpIn[it].toInt() } },

    // FSA Encoding
    allFSAPairsFlattened,
    allFSAPairsOffsets,
    levFSA.finalIdxs.toIntArray(),

    // CFG Encoding
    grammarFlattened,
    grammarOffsets,
  )

  cflIter(metadata)

  println("Round trip repair: ${clock.elapsedNow()}")
}

fun packStruct(constants: List<Int>, vararg arrays: IntArray): Int32Array<ArrayBuffer> {
  val offsets = arrays.scan(arrays.size * 2 + constants.size) { acc, arr -> acc + arr.size }.dropLast(1)

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
val WGSL_STRUCT = """struct CFLStruct {
             numStates : u32,
       numNonterminals : u32,
 
            dpInOffset : u32,
              dpInSize : u32,

           mdptsOffset : u32,
             mdptsSize : u32,

    mdptsOffsetsOffset : u32,
      mdptsOffsetsSize : u32,

    acceptStatesOffset : u32,
      acceptStatesSize : u32,

grammarFlattenedOffset : u32,
  grammarFlattenedSize : u32,

  grammarOffsetsOffset : u32,
    grammarOffsetsSize : u32,

               payload : array<u32>
};""".trimIndent()

//language=wgsl
val cfl_mul_upper by Shader(WGSL_STRUCT + """
fn decodeUpperTriangle(i : u32, N : u32) -> vec2<u32> {
    // same formula as your Metal kernel
    let d     = f32( (2u*N - 1u) * (2u*N - 1u) ) - f32(8u * i);
    let root  = sqrt(d);
    let rFloat= (f32(2u*N - 1u) - root) * 0.5;
    let rInt  = u32(floor(rFloat));
    // rowStart = r*(N-1) - (r*(r-1))/2
    let rowStart = rInt*(N - 1u) - (rInt*(rInt - 1u))/2u;
    let offset   = i - rowStart;
    let cInt     = rInt + 1u + offset;
    return vec2<u32>(rInt, cInt);
}

fn getDpIn(index: u32) -> u32 { return cs.payload[cs.dpInOffset + index]; }
fn setDpOut(index: u32, val: u32) { cs.payload[cs.dpOutOffset + index] = val; }
fn getDpOut(index: u32) -> u32 { return cs.payload[cs.dpOutOffset + index]; }
fn getMdpt(index: u32) -> u32 { return cs.payload[cs.mdptsOffset + index]; }
fn getMdptOffset(index: u32) -> u32 { return u32(cs.payload[cs.mdptsOffsetsOffset + index]); }
fn getGrammarSymbol(index: u32) -> u32 { return u32(cs.payload[cs.grammarFlattenedOffset + index]); }
fn getGrammarOffset(index: u32) -> u32 { return u32(cs.payload[cs.grammarOffsetsOffset + index]); }

  @group(0) @binding(0) var<storage, read_write> cs: CFLStruct;

  @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let tid = gid.x;

      let N  = cs.numStates;
      let NT = cs.numNonterminals;

      // totalCells = (numStates*(numStates-1)/2)*numNonterminals
      let half = (N * (N - 1u)) / 2u;
      let totalCells = half * NT;
      if (tid >= totalCells) { return; }

      // triIndex = tid / numNonterminals
      let triIndex = tid / NT;
      // Nonterminal = tid % numNonterminals
      let A        = tid % NT;

      // decode (r, c) from triIndex
      let rc = decodeUpperTriangle(triIndex, N);
      let r  = rc.x;
      let c  = rc.y;

      // snt = numStates * numNonterminals
      let snt    = N * NT;
      let dpIdx  = r*snt + c*NT + A;

      // grammar offsets for A
      //   startGC = grammarOffsets[A]
      //   endGC   = (A+1 < numNonterminals) ? grammarOffsets[A+1] : grammarFlattenedSize
      let startGC = getGrammarOffset(A);
      let endGC   = select(cs.grammarFlattenedSize, getGrammarOffset(A + 1u), A + 1u < NT);

      // aoi = r*numStates + c + 1
      let aoi = r*N + c + 1u;

      // pairOffset = allFSAPairsOffsets[aoi - 1]
      // pairOffsetNext = (aoi < allFSAPairsOffsetsSize) ? allFSAPairsOffsets[aoi] : allFSAPairsSize
      let pairOffset = getMdptOffset(aoi - 1u);
      let pairOffsetNext = select(cs.mdptsSize, getMdptOffset(aoi), aoi < cs.mdptsOffsetsSize);

      // If dp_in[dpIdx] != 0, copy and check the last bit
      let dpVal = getDpIn(dpIdx);
      if (dpVal != 0) {
          // Copy to dpOut
          setDpOut(dpIdx, dpVal);

          // If the last bit is set, return
          if ((dpVal & 0x01) != 0) { return; }
      }

      // Now loop over midpoints & grammar pairs
      for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx = pairIdx + 1u) {
          // mdpt = allFSAPairs[pairIdx]
          let mdpt = u32(getMdpt(pairIdx)); // mdpt is a state index

          // For each pair of grammar symbols in [startGC..endGC) stepping by 2
          var g = startGC;
          loop {
              if (g >= endGC) { break; }
              let B = getGrammarSymbol(g);
              let C = getGrammarSymbol(g + 1u);

              let idxBM = r*snt + mdpt*NT + B;
              let idxMC = mdpt*snt + c*NT + C;

              // If dp_in[idxBM] && dp_in[idxMC], set last bit in dp_out and return
              if ((getDpIn(idxBM) != 0) && (getDpIn(idxMC) != 0)) {
                  // set the 0x01 bit
                  let oldVal = getDpOut(dpIdx);
                  setDpOut(dpIdx, oldVal | 0x01);
                  return;
              }

              g = g + 2u;
          }
      }
  }""".trimIndent())

fun cflIter(packedMetadata: Int32Array<ArrayBuffer>) {
  // 1) Make a GPUBuffer from the packed metadata
  val metaSizeBytes = packedMetadata.length * 4
  val metaBuf = makeBuffer(
    sz = metaSizeBytes,
    us = 140, //GPUBufferUsage.STORAGE or GPUBufferUsage.COPY_DST or GPUBufferUsage.COPY_SRC,
    data = packedMetadata
  )

  // Let’s read them from the first few fields in packedMetadata:
  val numStates = packedMetadata[0]
  val numNonterminals = packedMetadata[1]
  val totalCells = (numStates * (numStates - 1)) / 2 * numNonterminals

  // 4) Prepare command encoder
  val encoder = gpu.createCommandEncoder()
  val pass = encoder.beginComputePass()
  pass.setPipeline(cfl_mul_upper.pipeline)
  // The “bindBuffers” style in your code might do:
  pass.setBindGroup(0, Shader.bindBuffers(cfl_mul_upper.pipeline, metaBuf))

  // 5) The kernel has a workgroup size of 64 in x, so we do:
  //    pass.dispatchWorkgroups(ceilDivide(totalCells, 64u), 1, 1)
  val blocks = (totalCells + 63) / 64
  pass.dispatchWorkgroups(blocks, 1, 1)

  pass.end()
  gpu.queue.submit(arrayOf(encoder.finish()))
}

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