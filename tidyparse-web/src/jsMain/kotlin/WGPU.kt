import js.buffer.AllowSharedBufferSource
import js.buffer.ArrayBuffer
import js.typedarrays.Int32Array
import kotlinx.coroutines.await
import web.gpu.*
import web.performance.performance
import kotlin.js.Promise
import kotlin.random.Random
import kotlin.reflect.KProperty

suspend fun benchmarkWGPU() {
  val N = 1024
  val M = IntArray(N * N) { Random.nextInt(2, 10) }
  val t0 = performance.now()
  var cSum = squareAndSumCpu(M, N)
  val t1 = performance.now()
  println("CPU sum=$cSum in ${t1 - t0} ms (N=$N)")
  val t2 = performance.now()
  val gSum = squareAndSumGPU(M.toTypedArray(), N)
  val t3 = performance.now()
  println("GPU sum=$gSum in ${t3 - t2} ms (N=$N)")
}

fun squareAndSumCpu(a: IntArray, n: Int): Long {
  val o = IntArray(n * n)
  for (r in 0 until n)
    for (c in 0 until n) {
      var s = 0
      for (k in 0 until n) s += a[r * n + k] * a[k * n + c]
      o[r * n + c] = s
    }
  return o.fold(0L) { x, y -> x + y }
}

//language=wgsl
val WGSL_MAT_MUL by Shader("""
struct Params { N: u32 };

@group(0) @binding(0) var<storage, read>       M:   array<i32>;
@group(0) @binding(1) var<storage, read_write> Out: array<i32>;
@group(0) @binding(2) var<uniform>             param: Params;

// We'll launch one thread per cell => dispatchWorkgroups(N, N)
// at @workgroup_size(1,1,1).  That means we have N*N threads total.
@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let N = param.N;

    if (row >= N || col >= N) { return; }

    let rowOffset = row * N;
    var acc = 0;
    for (var k = 0u; k < N; k = k + 1u) {
        let a = M[rowOffset + k];
        let b = M[k * N + col];
        acc = acc + (a * b);
    }
    Out[rowOffset + col] = acc;
}""")

suspend fun squareAndSumGPU(a: Array<Int>, n: Int): Long {
  // 1) Convert input to Int32Array
  val s = a.size         // total elements
  val bytes = s * 4      // size in bytes
  val data = Int32Array<ArrayBuffer>(s).apply { set(a, 0) }

  // 2) Make & fill usage bits:
  // https://www.w3.org/TR/webgpu/#buffer-usage
  //   STORAGE (128) + COPY_DST (8) = 136
  //   STORAGE (128) + COPY_SRC (4) = 132
  //   MAP_READ (1) + COPY_DST (8) = 9
  //   UNIFORM (64) + COPY_DST (8) = 72
  val bufM: GPUBuffer = makeBuffer(bytes, 136, data)
  val bufO: GPUBuffer = makeBuffer(bytes, 132)
  val bufP: GPUBuffer = makeBuffer(16, 72, Int32Array<ArrayBuffer>(4).apply { set(arrayOf(n), 0) })

  // 3) Invoke kernel
  val bufS = WGSL_MAT_MUL(bufM, bufO, bufP, readFrom = bufO, threads = n)

  // 4) Map & sum on CPU
  val i32 = Int32Array(bufS)
  var sum = 0L
  for (i in 0 until s) sum += i32[i]

  return sum
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

  suspend operator fun invoke(vararg inputs: GPUBuffer, readFrom: GPUBuffer, threads: Int): ArrayBuffer {
    val encoder: GPUCommandEncoder = gpu.createCommandEncoder()
    val output = makeBuffer(readFrom.size.toInt(), 9)
    encoder.beginComputePass().apply {
      setPipeline(pipeline)
      setBindGroup(0, bindBuffers(pipeline, *inputs))
      dispatchWorkgroups(threads, threads)
      end()
    }

    encoder.copyBufferToBuffer(readFrom, 0.0, output, 0.0, output.size)
    gpu.queue.submit(arrayOf(encoder.finish()))

    (output.mapAsync(1) as Promise<*>).await()
    return output.getMappedRange()
  }
}

fun makeBuffer(sz: Int, us: Int, data: AllowSharedBufferSource? = null): GPUBuffer =
  gpu.createBuffer(GPUBufferDescriptor(size = sz.toDouble(), usage = us))
    .also { if (data != null) { gpu.queue.writeBuffer(it, 0.0, data) } }