import kotlin.random.Random

/**
 * WGSL code for naive matrix multiplication:
 *
 * We have:
 *   - M: the input matrix (N*N) in row-major order, integer (i32).
 *   - Out: the output matrix (N*N).
 *   - param: a uniform with 'N' (the dimension).
 *
 * Each GPU thread corresponds to (row = gid.y, col = gid.x).
 * For the cell (row, col), we do a row/column multiply:
 *
 *    out[row*N + col] = sum(M[row*N + k] * M[k*N + col]), for k in [0..N-1]
 *
 */
private const val WGSL_MAT_MUL = """
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
}
"""

/**
 * Demonstrates square matrix multiplication in Kotlin/JS:
 *  1) CPU: naive triple-nested loop
 *  2) GPU: naive row*col multiply in a compute shader
 * and compares total sums of the result.
 */
fun benchmarkWGPU() {
  // 1) Pick dimension N (try 256 or so for a quick test)
  val N = 1024
  val size = N * N

  // 2) Build a random integer matrix (in row-major)
  //    Range is small to avoid 32-bit overflow (like 2..9).
  val matrix = IntArray(size) { Random.nextInt(2, 10) }

  // ----- CPU: M^2 + sum of all entries -----
  val tCpuStart = now()
  val cpuSum = multiplyAndSumCpu(matrix, N)
  val tCpuEnd = now()
  println("CPU sum=$cpuSum in ${tCpuEnd - tCpuStart} ms (N=$N)")

  // ----- GPU: M^2 in a compute shader, sum on CPU -----
  val tGpuStart = now()
  squareAndSumGPU(matrix, N).then { gpuSum: Long ->
    val tGpuEnd = now()
    println("GPU sum=$gpuSum in ${tGpuEnd - tGpuStart} ms (N=$N)")
  }
}

/** CPU naive approach: compute M^2, then sum all entries. */
fun multiplyAndSumCpu(mat: IntArray, N: Int): Long {
  val out = IntArray(N * N)
  for (r in 0 until N) {
    for (c in 0 until N) {
      var acc = 0
      for (k in 0 until N)
        acc += mat[r * N + k] * mat[k * N + c]
      out[r * N + c] = acc
    }
  }
  var sum = 0L
  for (elem in out) sum += elem.toLong()
  return sum
}

/**
 * GPU approach: naive matrix multiply in a WGSL compute shader,
 * then sum the result on the CPU.
 */
fun squareAndSumGPU(mat: IntArray, N: Int): dynamic {
  val gpu = navigator.gpu ?: return js("Promise.resolve(0)")

  val size = mat.size          // N*N
  val bytes = size * 4         // each i32 is 4 bytes
  val paramBytes = 16          // uniform buffer must be 16-byte aligned

  // Convert Kotlin IntArray -> JS Int32Array
  val jsMat = js("new Int32Array(size)")
  for (i in mat.indices) { jsMat[i] = mat[i] }

  // 1) Acquire device
  return gpu.requestAdapter().then { ad: dynamic -> ad.requestDevice() }.then { device: dynamic ->
    // 2) Create buffers
    //    USAGE bits:
    //      STORAGE=128, COPY_SRC=4, COPY_DST=8, MAP_READ=1, UNIFORM=64
    //    M => STORAGE + COPY_DST = 136
    //    Out => STORAGE + COPY_SRC = 132
    //    Staging => MAP_READ + COPY_DST = 9
    //    Param => UNIFORM + COPY_DST = 72
    val bufM = createBuffer(device, bytes, 136)
    val bufOut = createBuffer(device, bytes, 132)
    val bufStaging = createBuffer(device, bytes, 9)
    val bufParam = createBuffer(device, paramBytes, 72)

    // 3) Write data into GPU buffers
    device.queue.writeBuffer(bufM, 0, jsMat)
    // Write N into a 16-byte uniform buffer
    val paramData = js("new Uint32Array(4)") // 16 bytes total
    paramData[0] = N
    device.queue.writeBuffer(bufParam, 0, paramData)

    // 4) Compile WGSL
    val modDesc = js("{}")
    modDesc.code = WGSL_MAT_MUL
    val shaderModule = device.createShaderModule(modDesc)

    // 5) Pipeline
    val pipeDesc = js("{}")
    pipeDesc.layout = "auto"
    val compStage = js("{}")
    compStage.module = shaderModule
    compStage.entryPoint = "main"
    pipeDesc.compute = compStage
    val pipeline = device.createComputePipeline(pipeDesc)

    // 6) Bind group (M, Out, param)
    val bgDesc = js("{}")
    bgDesc.layout = pipeline.getBindGroupLayout(0)
    bgDesc.entries = arrayOf<dynamic>(
      js("{\"binding\":0,\"resource\":{\"buffer\":bufM}}"),
      js("{\"binding\":1,\"resource\":{\"buffer\":bufOut}}"),
      js("{\"binding\":2,\"resource\":{\"buffer\":bufParam}}")
    )
    val bindGroup = device.createBindGroup(bgDesc)

    // 7) Encode commands: dispatch NxN threads => O(N^3) work
    val encoder = device.createCommandEncoder()
    val pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(N, N)
    pass.end()

    // copy to staging => CPU can map
    encoder.copyBufferToBuffer(bufOut, 0, bufStaging, 0, bytes)
    device.queue.submit(arrayOf(encoder.finish()))

    // 8) Map staging & sum in Kotlin
    bufStaging.mapAsync(1).then {
      val mapped = bufStaging.getMappedRange()
      val outI32 = js("new Int32Array(mapped)")
      var sum = 0L
      for (i in 0 until size) {
        sum += outI32[i].unsafeCast<Int>().toLong()
      }
      sum
    }
  }
}

/** Simple timing helper. */
fun now(): Double = js("performance.now()").unsafeCast<Double>()
