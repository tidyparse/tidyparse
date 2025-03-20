import js.buffer.ArrayBuffer
import js.typedarrays.Int32Array
import kotlinx.coroutines.await
import web.gpu.*
import web.performance.performance
import kotlin.js.Promise
import kotlin.random.Random

//language=wgsl
private const val WGSL_MAT_MUL_SRC: String = """
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
}"""

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

inline fun <T> jsObject(init: dynamic.() -> Unit): T {
  val o = js("{}")
  init(o)
  return o as T
}

suspend fun squareAndSumGPU(a: Array<Int>, n: Int): Long {
  val device = (navigator.gpu as? GPU)?.requestAdapter()?.requestDevice() ?: return 0

  // 2) Convert input to Int32Array
  val s = a.size         // total elements
  val bytes = s * 4      // size in bytes
  val data = Int32Array<ArrayBuffer>(s)
  data.set(a, 0)

  // 3) Make & fill buffers
  // usage bits:
  //   STORAGE (128) + COPY_DST (8) = 136
  //   STORAGE (128) + COPY_SRC (4) = 132
  //   MAP_READ (1) + COPY_DST (8) = 9
  //   UNIFORM (64) + COPY_DST (8) = 72
  val bufM: GPUBuffer = device.makeBuffer(bytes, 136)
  val bufO: GPUBuffer = device.makeBuffer(bytes, 132)
  val bufS: GPUBuffer = device.makeBuffer(bytes, 9)
  val bufP: GPUBuffer = device.makeBuffer(16, 72)

  device.queueWrite(bufM, data)
  // write N into a 16-byte uniform buffer
  device.queueWrite(bufP, Int32Array<ArrayBuffer>(4).apply { set(arrayOf(n), 0) })

  // 4) Create pipeline & bind group
  val pipeline = device.makePipeline(WGSL_MAT_MUL_SRC)
  val bindGroup = device.bindBuffers(pipeline, bufM, bufO, bufP)

  // 5) Encode commands
  val encoder: GPUCommandEncoder = device.createCommandEncoder()
  encoder.beginComputePass().apply {
    setPipeline(pipeline)
    setBindGroup(0, bindGroup)
    dispatchWorkgroups(n, n) // naive NxN threads
    end()
  }

  // copy the “out” buffer to staging, so we can read it on CPU
  encoder.copyBufferToBuffer(bufO, 0.0, bufS, 0.0, bytes.toDouble())
  device.queue.submit(arrayOf(encoder.finish()))

  // 6) Map & sum
  (bufS.mapAsync(1) as Promise<*>).await()
  val i32 = Int32Array(bufS.getMappedRange())

  var sum = 0L
  for (i in 0 until s) sum += i32[i]

  return sum
}

fun GPUDevice.makeBuffer(sz: Int, us: Int): GPUBuffer =
  createBuffer(GPUBufferDescriptor(size = sz.toDouble(), usage = us))

suspend fun GPUDevice.makePipeline(wgsl: String, entryPoint: String = "main"): GPUComputePipeline =
  createComputePipeline(
    GPUComputePipelineDescriptor(
      layout = "auto",
      compute = GPUProgrammableStage(
        module = createShaderModule(GPUShaderModuleDescriptor(code = wgsl)),
        entryPoint = entryPoint
      )
    )
  )

fun GPUDevice.bindBuffers(pipeline: GPUComputePipeline, vararg buffers: GPUBuffer): GPUBindGroup {
  val lay = pipeline.getBindGroupLayout(0)
  val ent = buffers.mapIndexed { index, buf ->
    GPUBindGroupEntry(binding = index, resource = jsObject<GPUBindingResource> { buffer = buf })
  }.toTypedArray()

  return createBindGroup(GPUBindGroupDescriptor(layout = lay, entries = ent))
}

fun GPUDevice.queueWrite(buf: GPUBuffer, data: Int32Array<ArrayBuffer>) =
  queue.writeBuffer(buf, 0.0, data)