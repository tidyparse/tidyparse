@file:OptIn(ExperimentalUnsignedTypes::class)

package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNF
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import com.sun.jna.*
import org.intellij.lang.annotations.Language
import java.io.File
import kotlin.system.measureTimeMillis
import kotlin.time.TimeSource

/*
./gradlew generateDylib
 */

var grLen = 0
var goLen = 0
var apLen = 0
var aoLen = 0

fun main() {
  val cfg: CFG = pythonStatementCNF
  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE"
    .tokenizeByWhitespace()
  val levFSA = makeLevFSA(pythonCode, 3)

  val bindex = cfg.bindex
  val width = cfg.nonterminals.size
  val vindex = cfg.vindex
  val ups = cfg.unitProductions

  val dp0 = Array(levFSA.numStates) { Array(levFSA.numStates) { BooleanArray(width) { false } } }
  levFSA.allIndexedTxs0(ups, bindex).forEach { (q0, nt, q1) -> dp0[q0][q1][nt] = true }

  // Example placeholders
  val numStates = levFSA.numStates
  val numNonterminals = cfg.nonterminals.size

  val dpIn: UShortArray = dp0.flatten()
    .flatMap { it.map { if(it) 1.toUShort() else 0.toUShort() } }.toUShortArray()

  val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
  grLen = grammarFlattened.size.also { println("grLen: $it") }
  val grammarOffsets = cfg.vindex.map { it.size }
    .fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()
  goLen = grammarOffsets.size.also { println("goLen: $it") }

  val allFSAPairsFlattened = levFSA.allPairs2.flatten().flatten().toIntArray()
  apLen = allFSAPairsFlattened.size.also { println("apLen: $it") }
  val allFSAPairsOffsets = levFSA.allPairs2.flatten().map { it.size }
    .fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()
  aoLen = allFSAPairsOffsets.size.also { println("aoLen: $it") }

  println("\nLinked GPU in ${measureTimeMillis { GPUBridge }}ms\n")

  println("DP size: ${dpIn.size}")
  println("DP sum: ${dpIn.sum()}\n")

  var (resultGPU: List<Boolean>, resultCPU: List<Boolean>) = listOf<Boolean>() to listOf(true)
  measureTimeMillis {
    val outGPU = GPUBridge.cflClosure(
      dpIn,
      grammarFlattened,
      grammarOffsets,
      allFSAPairsFlattened,
      allFSAPairsOffsets,
      numStates,
      numNonterminals
    ).also { resultGPU = it.map { it == 1.toUShort() } }

    println("GPU size: " + outGPU.size)
    println("GPU sum: " + outGPU.sum())
  }.also { println("GPU time: ${it}ms\n") }

  measureTimeMillis {
    resultCPU = nonemptyLevInt(dp0, levFSA.allPairs, vindex).flatten().map { it.toList() }.flatten()
    println("CPU size: " + resultCPU.size)
    println("CPU sum: " + resultCPU.sumBy { if(it) 1 else 0 })
  }.also { println("CPU time: ${it}ms\n") }

  println("GPU ${if (resultGPU != resultCPU) "!=" else "=="} CPU")
}

fun nonemptyLevInt(dp: Array<Array<BooleanArray>>, ap: List<List<List<Int>?>>, vindex: Array<IntArray>): Array<Array<BooleanArray>> {
  for (dist: Int in 1 until dp.size) {
    for (iP: Int in 0 until dp.size - dist) {
      val p = iP
      val q = iP + dist
      if (ap[p][q] == null) continue
      val appq = ap[p][q]!!
      for ((A: Int, indexArray: IntArray) in vindex.withIndex()) {
        outerloop@for(j: Int in 0..<indexArray.size step 2) {
          val B = indexArray[j]
          val C = indexArray[j + 1]
          for (r in appq) if (dp[p][r][B] && dp[r][q][C]) dp[p][q][A] = true
        }
      }
    }
  }

  return dp
}

object GPUBridge {
  fun cflClosure(
    dpIn: UShortArray,
    grammar: IntArray,
    grammarOffsets: IntArray,
    allFSAPairs: IntArray,
    allFSAPairsOffsets: IntArray,
    numStates: Int,
    numNonterminals: Int,
  ): UShortArray {
    val dpBytes = Memory((dpIn.size * 2).toLong()).apply { write(0, dpIn.toShortArray(), 0, dpIn.size) }
    val outMem = Memory(dpBytes.size())
    val grammarMem = Memory((grammar.size * 4).toLong()).apply { write(0, grammar, 0, grammar.size) }
    val grammarOffsetsMem = Memory((grammarOffsets.size * 4).toLong()).apply { write(0, grammarOffsets, 0, grammarOffsets.size) }
    val allFSAPairsMem = Memory((allFSAPairs.size * 4).toLong()).apply { write(0, allFSAPairs, 0, allFSAPairs.size) }
    val allFSAPairsOffsetsMem = Memory((allFSAPairsOffsets.size * 4).toLong()).apply { write(0, allFSAPairsOffsets, 0, allFSAPairsOffsets.size) }

    val rounds = numStates//ceil(log2(numStates.toDouble())).toInt()

    nativeBridge.cflMultiply(
      dpBytes,                             // buffer(0) dp_in
      outMem,                              // buffer(1) dp_out
      grammarMem,                          // buffer(2)
      grammarOffsetsMem,                   // buffer(3)
      allFSAPairsMem,                      // buffer(4)
      allFSAPairsOffsetsMem,               // buffer(5)
      numStates,
      numNonterminals,
      grLen,
      goLen,
      apLen,
      aoLen,
      rounds
    )

    return outMem.getShortArray(0, dpIn.size).map { it.toUShort() }.toUShortArray()
  }

  interface NativeBridge : Library {
    fun setup()
    fun cflMultiply(
      dp_in: Pointer,   // buffer(0)
      dp_out: Pointer,  // buffer(1)
      grammar: Pointer, // buffer(2)
      grammarOffsets: Pointer,    // buffer(3)
      allFSAPairs: Pointer,       // buffer(4)
      allFSAPairsOffsets: Pointer,// buffer(5)
      numStates: Int,
      numNonterminals: Int,
      grammarSize: Int,
      grammarOffsetsSize: Int,
      allFSAPairsSize: Int,
      allFSAPairsOffsetsSize: Int,
      rounds: Int
    )
  }

  private val nativeBridge: NativeBridge =
    if (System.getProperty("os.name").startsWith("Mac")) getMetalBridge() else TODO()

  private fun getMetalBridge(): NativeBridge {
    val directory = "src/main/resources/dlls".also { File(it).mkdirs() }
    val dylib = File("$directory/libMetalBridge.dylib")

    @Language("c++") val metalSrc = """
#include <metal_stdlib>
using namespace metal;

kernel void cfl_mul(
  const device ushort* dp_in [[buffer(0)]],
  device ushort*       dp_out[[buffer(1)]],
  const device int*    grammar[[buffer(2)]],
  const device int*    grammarOffsets[[buffer(3)]],
  const device int*    allFSAPairs[[buffer(4)]],
  const device int*    allFSAPairsOffsets[[buffer(5)]],
  constant int&        numStates[[buffer(6)]],
  constant int&        numNonterminals[[buffer(7)]],
  constant int&        grammarSize[[buffer(8)]],
  constant int&        grammarOffsetsSize[[buffer(9)]],
  constant int&        allFSAPairsSize[[buffer(10)]],
  constant int&        allFSAPairsOffsetsSize[[buffer(11)]],
  uint tid[[thread_position_in_grid]]
) {
    // Each thread handles one index in [0..numStates*numStates*numNonterminals)
    int total = numStates * numStates * numNonterminals;

    // decode (r, c, A) from tid
    int r = int(tid) / (numStates * numNonterminals);
    int rcOff = int(tid) % (numStates * numNonterminals);
    int c = rcOff / numNonterminals;
    int A = rcOff % numNonterminals;
    if (c <= r) return;

    dp_out[tid] = dp_in[tid];  

    // Each pair = (B, C)
    int startGC = grammarOffsets[A];
    int endGC   = (A+1 < grammarOffsetsSize) ? grammarOffsets[A+1] : grammarSize;

    // midpoints between (r..c)
    int pairOffset = allFSAPairsOffsets[r * numStates + c];
    int pairOffsetNext = ((r * numStates + c) + 1 < allFSAPairsOffsetsSize)
       ? allFSAPairsOffsets[r * numStates + c + 1]
       : allFSAPairsSize;

    // loop over midpoints
    for (int idx = pairOffset; idx < pairOffsetNext; idx++) {
        int m = allFSAPairs[idx];
        
        // For each A -> (B, C):
        //     dp_in[r,m,B] && dp_in[m,c,C] => dp_out[r,c,A].
        for (int g = startGC; g < endGC; g += 2) {
            int B = grammar[g];
            int C = grammar[g+1];
            
            int idxBM = r * (numStates * numNonterminals) + m * numNonterminals + B;
            int idxMC = m * (numStates * numNonterminals) + c * numNonterminals + C;
            if (dp_in[idxBM] != 0 && dp_in[idxMC] != 0) {
                dp_out[tid] = 1;
                return;
            }
        }
    }
}

kernel void or_buf(
  const device ushort* bufA [[buffer(0)]],
  const device ushort* bufB [[buffer(1)]],
  device ushort*       out  [[buffer(2)]],
  uint tid [[thread_position_in_grid]]
) { out[tid] = (bufA[tid] != 0 || bufB[tid] != 0) ? ushort(1) : ushort(0); }
"""

    @Language("swift") val swiftSrc = """
import Foundation
import Metal

private var device: MTLDevice!
private var queue: MTLCommandQueue!
private var psoCflMul: MTLComputePipelineState!
private var psoOrBuf: MTLComputePipelineState!

@_cdecl("setup") public func setup() {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!

    let src = #""${'"'}$metalSrc${'"'}""#
    let library = try! device.makeLibrary(source: src, options: nil)
    psoCflMul = try! device.makeComputePipelineState(function: library.makeFunction(name: "cfl_mul")!)
    psoOrBuf = try! device.makeComputePipelineState(function: library.makeFunction(name: "or_buf")!)
}

@_cdecl("cflMultiply") public func cflMultiply(
    _ dp_in: UnsafePointer<UInt16>?,
    _ dp_out: UnsafeMutablePointer<UInt16>?,
    _ grammar: UnsafePointer<CInt>?,
    _ grammarOffsets: UnsafePointer<CInt>?,
    _ allFSAPairs: UnsafePointer<CInt>?,
    _ allFSAPairsOffsets: UnsafePointer<CInt>?,
    _ numStates: CInt,
    _ numNonterminals: CInt,
    _ grammarSize: CInt,
    _ grammarOffsetsSize: CInt,
    _ allFSAPairsSize: CInt,
    _ allFSAPairsOffsetsSize: CInt,
    _ rounds: CInt
) {
    guard let dp_in = dp_in,
          let dp_out = dp_out,
          let grammar = grammar,
          let grammarOffsets = grammarOffsets,
          let allFSAPairs = allFSAPairs,
          let allFSAPairsOffsets = allFSAPairsOffsets
    else { return }
    
    let totalCount = Int(numStates) * Int(numStates) * Int(numNonterminals)
    let tiling = MTLSizeMake(totalCount,1,1)
    let dpSizeBytes = totalCount * MemoryLayout<UInt16>.stride
    
    let bufA = device.makeBuffer(bytes: dp_in, length: dpSizeBytes, options: [])!
    let bufB = device.makeBuffer(length: dpSizeBytes, options: [])!
    let stride = MemoryLayout<Int32>.stride
    
    var ns = numStates
    var nt = numNonterminals
    var gr = grammarSize
    var go = grammarOffsetsSize
    var ap = allFSAPairsSize
    var ao = allFSAPairsOffsetsSize

    let grammarLen = Int(gr)
    let grammarBuf = device.makeBuffer(bytes: grammar, length: grammarLen*stride, options: [])!
    let grammarOffBuf = device.makeBuffer(bytes: grammarOffsets, length: Int(go)*stride, options: [])!
    let pairsBuf = device.makeBuffer(bytes: allFSAPairs, length: Int(ap)*stride, options: [])!
    let pairsOffBuf = device.makeBuffer(bytes: allFSAPairsOffsets, length: Int(ao)*stride, options: [])!

    var oldBuffer = [UInt16](repeating: 0, count: totalCount)
    var newBuffer = [UInt16](repeating: 0, count: totalCount)
    for r in 0..<rounds {
        memcpy(&oldBuffer, bufA.contents(), dpSizeBytes)

            // cfl_mul( ... )
            let commandBuffer1 = queue.makeCommandBuffer()!
            let encoder1 = commandBuffer1.makeComputeCommandEncoder()!
            encoder1.setComputePipelineState(psoCflMul)
            encoder1.setBuffer(bufA, offset: 0, index: 0)
            encoder1.setBuffer(bufB, offset: 0, index: 1)
            encoder1.setBuffer(grammarBuf, offset: 0, index: 2)
            encoder1.setBuffer(grammarOffBuf, offset: 0, index: 3)
            encoder1.setBuffer(pairsBuf, offset: 0, index: 4)
            encoder1.setBuffer(pairsOffBuf, offset: 0, index: 5)
            encoder1.setBytes(&ns, length: MemoryLayout<Int32>.size, index: 6)
            encoder1.setBytes(&nt, length: MemoryLayout<Int32>.size, index: 7)
            encoder1.setBytes(&gr, length: MemoryLayout<Int32>.size, index: 8)
            encoder1.setBytes(&go, length: MemoryLayout<Int32>.size, index: 9)
            encoder1.setBytes(&ap, length: MemoryLayout<Int32>.size, index: 10)
            encoder1.setBytes(&ao, length: MemoryLayout<Int32>.size, index: 11)

            encoder1.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
            encoder1.endEncoding()
            commandBuffer1.commit()
            commandBuffer1.waitUntilCompleted()

            let commandBuffer2 = queue.makeCommandBuffer()!
            let encoder2 = commandBuffer2.makeComputeCommandEncoder()!
            encoder2.setComputePipelineState(psoOrBuf)
            encoder2.setBuffer(bufA, offset: 0, index: 0)
            encoder2.setBuffer(bufB, offset: 0, index: 1)
            encoder2.setBuffer(bufA, offset: 0, index: 2)
            encoder2.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
            encoder2.endEncoding()
            commandBuffer2.commit()
            commandBuffer2.waitUntilCompleted()

        memcpy(&newBuffer, bufA.contents(), dpSizeBytes)

        if newBuffer == oldBuffer {
            print("Fixpoint escape at round: \(r)/\(rounds)")
            break
        }
    }

    memcpy(dp_out, bufA.contents(), dpSizeBytes)
}"""

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