import Shader.Companion.GPUBuffer
import Shader.Companion.bindBuffers
import Shader.Companion.readInts
import Shader.Companion.toGPUBuffer
import ai.hypergraph.kaliningraph.parsing.contains
import ai.hypergraph.kaliningraph.parsing.language
import ai.hypergraph.kaliningraph.parsing.makeLevFSA
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import web.gpu.GPUBuffer
import web.gpu.GPUCommandEncoder
import web.performance.performance
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.time.TimeSource
import kotlin.time.measureTimedValue

suspend fun benchmarkReach() {
  val code = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE"
  val fsa = makeLevFSA(code, 5)

  val t0 = TimeSource.Monotonic.markNow()
  val midpoints = fsa.midpoints
  val (tdata, toffset) = midpoints.prefixScan()
  println("Sparse CPU reachability took ${t0.elapsedNow()} / sum: ${midpoints.flatten().size}")

  val t1 = TimeSource.Monotonic.markNow()
  val (reachBuf, _) = dag_reach.invokeDAGFixpoint(fsa)
  val (data, offset) = reachBuf.readInts()
    .also { println("Fixpoint reached in ${t1.elapsedNow()} / sum: ${it.sum()}") }
    .sparsifyReachabilityMatrix()
    .also { println("Sparse GPU reachability in ${t1.elapsedNow()} / sum: ${it.flatten().size}") }
    .prefixScan()

  println("Full sparse GPU reachability took ${t1.elapsedNow()}")

  println("Reference (${tdata.size}, ${toffset.size}) / Actual (${data.size}, ${offset.size})")
}

// TODO: move into shader kernel?
fun List<List<List<Int>>>.prefixScan(): Pair<IntArray, IntArray> =
  measureTimedValue {
    fun IntArray.prefixSumSizes(): IntArray =
      IntArray(size + 1).also { prefix ->
        copyInto(prefix, destinationOffset = 1)
        for (i in 1..size) prefix[i] += prefix[i - 1]
      }

    val filtered = mapIndexed { p, outer ->
      outer.mapIndexed { q, inner -> inner.filter { it in (p + 1)..<q } }
    }.flatten()

    val flattened = filtered.flatten().toIntArray()
    val offsets = filtered.map { it.size }.toIntArray().prefixSumSizes()

    flattened to offsets
  }.also { println("Completed prefix scan in: ${it.duration}") }.value

fun IntArray.sparsifyReachabilityMatrix(n: Int = sqrt(size.toDouble()).toInt()): List<List<List<Int>>> =
  List(n) { i ->
    List(n) { j ->
      if (j <= i || this[i*n + j] == 0) emptyList()
      else (0..<n).filter { v -> this[i*n + v] == 1 && this[v*n + j] == 1 }
    }
  }

suspend fun benchmarkWGPURepair() {
  val cfg = pythonStatementCNFAllProds
  val code = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE".tokenizeByWhitespace()
  val words = repairCode(cfg, code, 5).distinct()
  println("Distinct words: ${words.size}")

  val (valid, invalid) = words.shuffled().take(3).partition { it in cfg.language }
  println("\nValid samples (${valid.size})\n")
  valid.forEachIndexed { i, it -> println("$i.) ${it.trim()}") }
  println("\nInvalid samples (${invalid.size})\n")
  invalid.forEachIndexed { i, it -> println("$i.) ${it.trim()}") }
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
  val gSum = WGSL_GEMX_ITERATE.invokeExp(intArrayOf(N).toGPUBuffer(), M.toGPUBuffer(140),
    threads = N, iterations = P).asList().hashCode()
  val t3 = performance.now()
  println("GPU hash=$gSum in ${t3 - t2} ms (N=$N, P=$P)")

  val t0 = performance.now()

  fun iterateCPU(a: IntArray, P: Int): Int {
    val n = sqrt(a.size.toDouble()).toInt()
    var current = a.copyOf()
    for (step in 1..P) {
      val next = IntArray(n * n)
      for (r in 0..<n) for (c in 0..<n)
        next[r * n + c] = (0..<n).sumOf { k -> current[r * n + k] * current[k * n + c] }
      current = next
    }
    return current.toList().hashCode()
  }

  val cSum = iterateCPU(M, P)
  val t1 = performance.now()
  println("CPU hash=$cSum in ${t1 - t0} ms (N=$N, P=$P)")
}

suspend fun Shader.invokeExp(vararg inputs: GPUBuffer, threads: Int, iterations: Int = 1): IntArray {
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