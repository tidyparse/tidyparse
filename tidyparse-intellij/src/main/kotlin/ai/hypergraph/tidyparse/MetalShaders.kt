package ai.hypergraph.tidyparse

import com.sun.jna.*
import org.intellij.lang.annotations.Language
import java.io.File
import kotlin.math.*
import kotlin.random.Random
import kotlin.system.measureTimeMillis
import kotlin.time.*

/*
./gradlew generateDylib
 */
fun main() {
  val t = 1000
  var arr = IntArray(t * t) { Random.nextInt(100) }

  println("Linked GPU in ${measureTimeMillis { GPUBridge }}ms")
  val outGPU = measureTimedValue { GPUBridge.completeMat(arr) }
    .also { println("GPU took ${it.duration.inWholeMilliseconds}ms") }.value

  val outCPU = IntArray(arr.size)
  val cpuMs = measureTimeMillis {
    for (e in 0..<ceil(log(t.toDouble(), 2.0)).toInt()) {
      val temp = arr.copyOf()

      for (r in 0 until t) for (c in 0 until t) {
        if (r <= c) continue
        var s = 0
        for (k in 0 until t) s += temp[r*t + k] * temp[k*t + c]
        outCPU[r*t + c] = s
      }
      arr = outCPU.copyOf()
    }
  }
  println("CPU took ${cpuMs}ms")

  (0..(t * t - 1)).forEach { if (outGPU[it] != outCPU[it]) error("GPU[$it]=${outGPU[it]} != CPU[it]=${outCPU[it]}") }
  println("GPU=CPU")
}

object GPUBridge {
  fun completeMat(arr: IntArray): IntArray {
    val memInp = Memory((arr.size * 4).toLong()).apply { write(0, arr, 0, arr.size) }
    val memOut = Memory((arr.size * 4).toLong())
    nativeBridge.imm(memInp, sqrt(arr.size.toDouble()).toInt(), memOut)
    return memOut.getIntArray(0, arr.size)
  }

  interface NativeBridge : Library {
    fun setup()
    fun imm(a: Pointer, n: Int, out: Pointer)
  }

  private val nativeBridge: NativeBridge =
    if (System.getProperty("os.name").startsWith("Mac")) getMetalBridge() else TODO()

  private fun getMetalBridge(): NativeBridge {
    val directory = "src/main/resources/dlls".also { File(it).mkdirs() }
    val dylib = File("$directory/libMetalBridge.dylib")

    @Language("c++") val mpsSrc = """
  #include <metal_stdlib>
  using namespace metal;
  kernel void mat_mul(
    const device int* A[[buffer(0)]],
    device int* O[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    uint i[[thread_position_in_grid]]
  ) {
    if (i < n * n) {
      uint r = i / n, c = i % n; int s = 0;
      if (r <= c) return;
      for (uint k = 0; k < n; k++) s += A[r * n + k] * A[k * n + c];
      O[i] = s;
    }
  }

  kernel void add_buf(
    const device int* bufferA[[buffer(0)]],
    const device int* bufferB[[buffer(1)]],
    device int* output[[buffer(2)]],
    uint index[[thread_position_in_grid]]
  ) { output[index] = bufferA[index] + bufferB[index]; }

  inline int getBit(int value, uint bitIndex) { return (value >> bitIndex) & 1; }
  """

    @Language("swift") val swiftSrc = """
import Foundation
import Metal
private var dvc: MTLDevice!, mtq: MTLCommandQueue!, cpsmm: MTLComputePipelineState!, cpsab: MTLComputePipelineState!
  
@_cdecl("setup") public func setup() {
  let metalSrc = #""${'"'}$mpsSrc${'"'}""#
  dvc = MTLCreateSystemDefaultDevice()!
  mtq = dvc.makeCommandQueue()!
  let lib = try! dvc.makeLibrary(source: metalSrc, options:nil)
  cpsmm = try! dvc.makeComputePipelineState(function:lib.makeFunction(name:"mat_mul")!)
  cpsab = try! dvc.makeComputePipelineState(function:lib.makeFunction(name:"add_buf")!)
}

@_cdecl("imm") public func imm(_ A: UnsafePointer<CInt>?, _ n: CInt, _ out: UnsafeMutablePointer<CInt>?) {
  let nn = Int(n), sz = nn * nn * 4, reps = Int(ceil(log2(Double(nn))))
  let BA = dvc.makeBuffer(bytes: A!, length: sz, options: [])!,
      BO = dvc.makeBuffer(length: sz, options: [])!
  for _ in 0..<reps {
    let cb = mtq.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(cpsmm)

    enc.setBuffer(BA, offset: 0, index: 0)
    enc.setBuffer(BO, offset: 0, index: 1)
    var cpn = n; enc.setBytes(&cpn, length: MemoryLayout<CInt>.size, index: 2)
    enc.dispatchThreads(MTLSizeMake(nn * nn, 1, 1), threadsPerThreadgroup: MTLSizeMake(1, 1, 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    memcpy(BA.contents(), BO.contents(), sz)
  }
  memcpy(out, BA.contents(), sz)
}""".trimIndent()

    val hash = swiftSrc.hashCode().toString()
    val hashFile = File("$directory/.swiftHash")
    fun needsRebuild() = !dylib.exists() || !hashFile.exists() || hashFile.readText() != hash

    if (needsRebuild()) {
      val clock = TimeSource.Monotonic.markNow()
      val metalBridgePath = File("$directory/MetalBridge.swift").apply { writeText(swiftSrc) }.path
      ("xcrun swiftc -emit-library $metalBridgePath -o ${dylib.absolutePath} -module-name M " +
          "-Xlinker -install_name -Xlinker @rpath/${dylib.path}")
        .run { ProcessBuilder(split(" ")).inheritIO().start().waitFor() }
        .also { if (it != 0) error("Failed to build Swift bridging code!") }

      hashFile.writeText(hash)
      println("Finished rebuild in ${clock.elapsedNow()}")
    }

    return (Native.load(dylib.absolutePath, NativeBridge::class.java) as NativeBridge).also { it.setup() }
  }
}