import kotlinx.browser.window
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import kotlin.js.Promise

@JsName("navigator")
external val navigator: dynamic

/**
 * We store the grammar and allFSA arrays in a flattened form:
 *
 * For grammar:
 *   - We'll create an array whose first 2*V elements are offsets and lengths for each A:
 *       grammarFlat[A*2 + 0] = offsetIntoPairs
 *       grammarFlat[A*2 + 1] = numberOfBCpairsForA
 *   - After that, we store all (B,C) pairs for each A in sequence.
 *
 * For allFSA:
 *   - We'll create an array whose first 2*Q*Q elements are offsets and lengths for (p,q):
 *       allFSAFlat[(p*Q + q)*2 + 0] = offsetIntoMidpoints
 *       allFSAFlat[(p*Q + q)*2 + 1] = numberOfMidpoints
 *   - After that, we store all midpoints r for each (p,q) in sequence.
 *
 * In the WGSL code, `mainSquare` will do:
 *   chartTmp[p,q,A] = OR_{r in midpoints(p,q), (B,C) in grammar(A)} ( chart[p,r,B] && chart[r,q,C] ).
 *
 * Finally we perform exponentiation-by-squaring ~log2(Q) times:
 *   S(M_{i+1}) = S(M_i) + S(M_i)^2,
 * which we approximate with repeated "square + OR" steps for up to ceil(log2(Q)) iterations.
 */

private lateinit var WGSL_BAR_HILLEL: String

/**
 * Perform a Bar-Hillel intersection / "CYK-like" step on GPU using exponentiation-by-squaring.
 * - chart is a Q x Q x V 3D boolean array (dp table).
 * - grammar[A] = list of (B, C) pairs for nonterminal A.
 * - allFSA[p][q] = list of midpoint states r between p and q in the underlying acyclic FSA.
 *
 * Steps:
 *  1) Flatten the 3D chart into a 1D Uint32Array (1 for true, 0 for false).
 *  2) Flatten the grammar data:
 *       grammarFlat[A*2]   = offset into "BC pairs" portion
 *       grammarFlat[A*2+1] = number of (B, C) pairs
 *       // followed by the pairs themselves
 *  3) Flatten the allFSA data:
 *       allFSAFlat[(p*Q+q)*2]   = offset into "midpoints" portion
 *       allFSAFlat[(p*Q+q)*2+1] = number of midpoints
 *       // followed by the midpoints themselves
 *  4) Create compute pipelines for mainSquare (chartTmp = chart^2) and mainOr (chart |= chartTmp).
 *  5) Repeat ~log2(Q) times the sequence of "square" + "or".
 *  6) Copy the final chart back to the CPU and return it as a 3D List<List<MutableList<Boolean>>>.
 */

fun wgpuBarHillelIntersection(
  chart: Array<Array<BooleanArray>>,
  grammar: Array<IntArray>,
  allFSA: List<List<List<Int>>>
): dynamic /* Promise<List<List<List<Boolean>>>> */ {
  val gpu = navigator.gpu ?: run {
    console.warn("WebGPU not available.")
    return null
  }

  val Q = chart.size
  if (Q == 0) return js("Promise.resolve(chart)")
  val V = chart[0][0].size
  val totalChartSize = Q * Q * V

  // Flatten the chart
  val chartFlat = js("new Uint32Array(totalChartSize)")
  var idx = 0
  for (p in 0 until Q) {
    for (q in 0 until Q) {
      for (A in 0 until V) {
        chartFlat[idx++] = if (chart[p][q][A]) 1u else 0u
      }
    }
  }

  /**
   * Flatten the grammar:
   *   grammarFlat[0..2*V-1] contains offsets/lengths (two per A),
   *   then all (B,C) pairs in sequence.
   */
  var totalPairs = 0
  for (A in 0 until V) {
    // Each grammar[A] is a list of B,C repeated, so # of pairs = grammar[A].size / 2
    totalPairs += grammar[A].size
  }

  val grammarFlatSize = 2*V + totalPairs // (offset,len) for each A plus all pairs
  val grammarFlat = js("new Uint32Array(grammarFlatSize)")
  var pairPos = (2 * V) // where pairs start
  for (A in 0 until V) {
    val offset = pairPos
    val pairCount = grammar[A].size / 2
    grammarFlat[A*2 + 0] = offset
    grammarFlat[A*2 + 1] = pairCount
    // copy pairs
    for (bci in grammar[A].indices) {
      grammarFlat[pairPos++] = grammar[A][bci].toUInt()
    }
  }

  /**
   * Flatten the allFSA:
   *   allFSAFlat[0..2*Q*Q - 1] contains offsets/lengths (two per (p,q)),
   *   then all midpoints in sequence.
   */
  var totalMidpoints = 0
  for (p in 0 until Q) {
    for (q in 0 until Q) {
      totalMidpoints += allFSA[p][q].size
    }
  }

  val fsaHeaderSize = 2 * Q * Q
  val allFSAFlatSize = fsaHeaderSize + totalMidpoints
  val allFSAFlat = js("new Uint32Array(allFSAFlatSize)")
  var midPos = fsaHeaderSize
  for (p in 0 until Q) {
    for (q in 0 until Q) {
      val offset = midPos
      val count = allFSA[p][q].size
      val idxPQ = (p*Q + q) * 2
      allFSAFlat[idxPQ + 0] = offset
      allFSAFlat[idxPQ + 1] = count
      for (r in allFSA[p][q]) {
        allFSAFlat[midPos++] = r.toUInt()
      }
    }
  }

  // Byte sizes
  val bytesChart   = totalChartSize * 4
  val bytesTemp    = totalChartSize * 4
  val bytesGrammar = grammarFlatSize * 4
  val bytesAllFSA  = allFSAFlatSize * 4
  val bytesParams  = 8 /* Q & V (2 x uint32) -> 8 bytes, must be multiple of 16 => pad to 16 */
  val paramBytesAligned = 16

  // We'll do ~log2(Q) exponentiation passes
  val steps = kotlin.math.ceil(kotlin.math.log2(Q.coerceAtLeast(1).toDouble())).toInt().coerceAtLeast(1)

  return gpu.requestAdapter().then { adapter: dynamic -> adapter.requestDevice() }
    .then { device: dynamic ->
    // Create buffers
    val bufChart = createBuffer(device, bytesChart, 128+8)  // STORAGE + COPY_DST
    val bufTemp  = createBuffer(device, bytesTemp, 128+8)  // STORAGE + COPY_DST
    val bufGrammar = createBuffer(device, bytesGrammar, 128) // STORAGE
    val bufAllFSA  = createBuffer(device, bytesAllFSA, 128)  // STORAGE
    val bufParam   = createBuffer(device, paramBytesAligned, 64) // UNIFORM

    // Write data
    device.queue.writeBuffer(bufChart, 0, chartFlat)
    device.queue.writeBuffer(bufGrammar, 0, grammarFlat)
    device.queue.writeBuffer(bufAllFSA, 0, allFSAFlat)

    // Create param uniform data
    val paramData = js("new Uint32Array(2)") // We'll upload 2 uints, but the buffer is 16 bytes
    paramData[0] = Q
    paramData[1] = V
    device.queue.writeBuffer(bufParam, 0, paramData)

    // Create shader module
    val modDesc = js("{}")
    modDesc.code = WGSL_BAR_HILLEL
    val shaderModule = device.createShaderModule(modDesc)

    // Pipelines
    val pipeSquareDesc = js("{}")
    pipeSquareDesc.layout = "auto"
    val compSquareStage = js("{}")
    compSquareStage.module = shaderModule
    compSquareStage.entryPoint = "mainSquare"
    pipeSquareDesc.compute = compSquareStage
    val pipelineSquare = device.createComputePipeline(pipeSquareDesc)

    val pipeOrDesc = js("{}")
    pipeOrDesc.layout = "auto"
    val compOrStage = js("{}")
    compOrStage.module = shaderModule
    compOrStage.entryPoint = "mainOr"
    pipeOrDesc.compute = compOrStage
    val pipelineOr = device.createComputePipeline(pipeOrDesc)

    // Bind groups
    val bgSquareDesc = js("{}")
    bgSquareDesc.layout = pipelineSquare.getBindGroupLayout(0)
    bgSquareDesc.entries = arrayOf<dynamic>(
      js("{\"binding\":0,\"resource\":{\"buffer\":bufChart}}"),
      js("{\"binding\":1,\"resource\":{\"buffer\":bufGrammar}}"),
      js("{\"binding\":2,\"resource\":{\"buffer\":bufAllFSA}}"),
      js("{\"binding\":3,\"resource\":{\"buffer\":bufTemp}}"),
      js("{\"binding\":4,\"resource\":{\"buffer\":bufParam}}")
    )
    val bindGroupSquare = device.createBindGroup(bgSquareDesc)

    val bgOrDesc = js("{}")
    bgOrDesc.layout = pipelineOr.getBindGroupLayout(0)
    bgOrDesc.entries = arrayOf<dynamic>(
      js("{\"binding\":0,\"resource\":{\"buffer\":bufChart}}"),
      js("{\"binding\":3,\"resource\":{\"buffer\":bufTemp}}"),
      js("{\"binding\":4,\"resource\":{\"buffer\":bufParam}}")
    )
    val bindGroupOr = device.createBindGroup(bgOrDesc)

    // Repeated "square + or"
    repeat(steps) {
      val encoder = device.createCommandEncoder()

      // Square pass
      val passSquare = encoder.beginComputePass()
      passSquare.setPipeline(pipelineSquare)
      passSquare.setBindGroup(0, bindGroupSquare)
      passSquare.dispatchWorkgroups(Q.toDouble(), Q.toDouble(), V.toDouble())
      passSquare.end()

      // Or pass
      val passOr = encoder.beginComputePass()
      passOr.setPipeline(pipelineOr)
      passOr.setBindGroup(0, bindGroupOr)
      passOr.dispatchWorkgroups(Q.toDouble(), Q.toDouble(), V.toDouble())
      passOr.end()

      device.queue.submit(arrayOf(encoder.finish()))
    }

    // Copy final chart back to CPU
    val bufStaging = createBuffer(device, bytesChart, 1+8) // MAP_READ + COPY_DST
    val cenc = device.createCommandEncoder()
    cenc.copyBufferToBuffer(bufChart, 0, bufStaging, 0, bytesChart)
    device.queue.submit(arrayOf(cenc.finish()))

    // Map & reconstruct the 3D chart
    bufStaging.mapAsync(1).then {
      val mapped = bufStaging.getMappedRange()
      val outU32 = js("new Uint32Array(mapped)")
      val finalChart = List(Q) {
        List(Q) { MutableList(V) { false } }
      }
      var idx2 = 0
      for (p in 0 until Q) {
        for (q in 0 until Q) {
          for (A in 0 until V) {
            finalChart[p][q][A] = (outU32[idx2++] == 1u)
          }
        }
      }

      println("Parseable: " + finalChart[0][finalChart.size-1][0])
      finalChart
    }
  }
}

/** Helper to create a GPU buffer with the specified size & usage flags. */
fun createBuffer(device: dynamic, size: Int, usage: Int): dynamic {
  val desc = js("{}")
  desc.size = size
  desc.usage = usage
  return device.createBuffer(desc)
}