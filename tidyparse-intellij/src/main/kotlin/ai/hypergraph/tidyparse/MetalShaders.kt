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

fun main() {
  val cfg = pythonStatementCNF
  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE"
    .tokenizeByWhitespace()
  val levFSA = makeLevFSA(pythonCode, 5)

  val bindex = cfg.bindex
  val width = cfg.nonterminals.size
  val vindex = cfg.vindex
  val ups = cfg.unitProductions
  val numStates = levFSA.numStates
  val numNonterminals = cfg.nonterminals.size

  val dp0 = Array(levFSA.numStates) { Array(levFSA.numStates) { BooleanArray(width) { false } } }
  levFSA.allIndexedTxs0(ups, bindex).forEach { (q0, nt, q1) -> dp0[q0][q1][nt] = true }

  val dpIn: ByteArray = dp0.flatten()
    .flatMap { it.map { if(it) 1.toByte() else 0.toByte() } }.toByteArray()

  val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
  val grLen = grammarFlattened.size.also { println("grLen: $it") }
  val grammarOffsets = cfg.vindex.map { it.size }
    .fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()
  val goLen = grammarOffsets.size.also { println("goLen: $it") }

  val allFSAPairsFlattened = levFSA.allPairs2.flatten().flatten().toIntArray()
  val apLen = allFSAPairsFlattened.size.also { println("apLen: $it") }
  val allFSAPairsOffsets = levFSA.allPairs2.flatten().map { it.size }
    .fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()
  val aoLen = allFSAPairsOffsets.size.also { println("aoLen: $it") }

  println("\nLinked GPU in ${measureTimeMillis { GPUBridge }}ms\n")

  println("DP size: ${dpIn.size}")
  println("DP sum: ${dpIn.sumBy { if(it == 1.toByte()) 1 else 0 }}\n")

  var (resultGPU: List<Boolean>, resultCPU: List<Boolean>) = listOf<Boolean>() to listOf(true)
  measureTimeMillis {
    val outGPU = GPUBridge.cflClosure(
      dpIn,
      grammarFlattened,
      grammarOffsets,
      allFSAPairsFlattened,
      allFSAPairsOffsets,
      numStates, numNonterminals,
      grLen, goLen, apLen, aoLen
    ).also { resultGPU = it.map { it == 1.toByte() } }

    println("GPU size: " + outGPU.size)
    println("GPU sum: " + outGPU.sumBy { if(it == 1.toByte()) 1 else 0 })
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
    dpIn: ByteArray,
    grammar: IntArray,
    grammarOffsets: IntArray,
    allFSAPairs: IntArray,
    allFSAPairsOffsets: IntArray,
    numStates: Int, numNonterminals: Int,
    grLen: Int, goLen: Int, apLen: Int, aoLen: Int
  ): ByteArray {
    val dpBytes = Memory((dpIn.size).toLong()).apply { write(0, dpIn, 0, dpIn.size) }
    val outMem = Memory(dpBytes.size())
    val grammarMem = Memory((grammar.size * 4).toLong()).apply { write(0, grammar, 0, grammar.size) }
    val grammarOffsetsMem = Memory((grammarOffsets.size * 4).toLong()).apply { write(0, grammarOffsets, 0, grammarOffsets.size) }
    val allFSAPairsMem = Memory((allFSAPairs.size * 4).toLong()).apply { write(0, allFSAPairs, 0, allFSAPairs.size) }
    val allFSAPairsOffsetsMem = Memory((allFSAPairsOffsets.size * 4).toLong()).apply { write(0, allFSAPairsOffsets, 0, allFSAPairsOffsets.size) }

    nativeBridge.cflMultiply(
      dpBytes,                     // buffer(0) dp_in
      outMem,                      // buffer(1) dp_out
      grammarMem,                  // buffer(2)
      grammarOffsetsMem,           // buffer(3)
      allFSAPairsMem,              // buffer(4)
      allFSAPairsOffsetsMem,       // buffer(5)
      numStates, numNonterminals,
      grLen, goLen, apLen, aoLen
    )

    return outMem.getByteArray(0, dpIn.size)
  }

  interface NativeBridge : Library {
    fun setup()
    fun cflMultiply(
      dp_in: Pointer,              // buffer(0)
      dp_out: Pointer,             // buffer(1)
      grammar: Pointer,            // buffer(2)
      grammarOffsets: Pointer,     // buffer(3)
      allFSAPairs: Pointer,        // buffer(4)
      allFSAPairsOffsets: Pointer, // buffer(5)
      numStates: Int,
      numNonterminals: Int,
      grammarSize: Int,
      grammarOffsetsSize: Int,
      allFSAPairsSize: Int,
      allFSAPairsOffsetsSize: Int
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

// Helper to decode (row,col) in the upper-triangle from tid
inline void decodeUpperTriangle(int i, int N, thread int &r, thread int &c) {
//    int rowLength = N - 1;
//    int rowStart = 0;
//    r = 0;
//    while (true) {
//        int rowEnd = rowStart + rowLength;
//        if (i < rowEnd) {
//            // i belongs in row 'r'
//            int offset = i - rowStart;
//            c = (r + 1) + offset;
//            return;
//        }
//        rowStart = rowEnd;
//        rowLength--;
//        r++;
//    }
    // Closed form solution:
    float d = float((2*N - 1)*(2*N - 1)) - float(8*i);
    float root = sqrt(d);
    float rFloat = ((float)(2*N - 1) - root) * 0.5f;
    r = int(floor(rFloat));
    int rowStart = r*(N-1) - (r*(r-1))/2;
    int offset = i - rowStart;
    c = r + 1 + offset;
}

kernel void cfl_mul_upper(
  const device uchar*  dp_in                   [[buffer(0)]],
  device uchar*        dp_out                  [[buffer(1)]],
  const device int*    grammar                 [[buffer(2)]],
  const device int*    grammarOffsets          [[buffer(3)]],
  const device int*    allFSAPairs             [[buffer(4)]],
  const device int*    allFSAPairsOffsets      [[buffer(5)]],
  constant int&        numStates               [[buffer(6)]],
  constant int&        numNonterminals         [[buffer(7)]],
  constant int&        grammarSize             [[buffer(8)]],
  constant int&        grammarOffsetsSize      [[buffer(9)]],
  constant int&        allFSAPairsSize         [[buffer(10)]],
  constant int&        allFSAPairsOffsetsSize  [[buffer(11)]],
  device atomic_uint&  changedFlag             [[buffer(12)]],
  uint                 tid                     [[thread_position_in_grid]]
) {
    // We are only covering the strict upper triangle (c > r)
    // total # of (r, c) pairs = N*(N-1)/2
    // for each pair, we have numNonterminals possible As
    int N = numStates;
    int totalPairs = (N*(N-1))/2;
    int totalThreads = totalPairs * numNonterminals;

    // Decode A
    int A = int(tid) % numNonterminals;
    // Which upper-triangle pair?
    int pairIndex = int(tid) / numNonterminals;

    // Decode (r, c) from pairIndex
    int r, c;
    decodeUpperTriangle(pairIndex, N, r, c);

    // flatten (r,c,A) into index for dp array
    int dpIndex = r*(N*numNonterminals) + c*numNonterminals + A;

    if (dp_in[dpIndex]) {
        dp_out[dpIndex] = dp_in[dpIndex];
        return;
    }

    // Grammar offsets for A
    int startGC = grammarOffsets[A];
    int endGC   = (A+1 < grammarOffsetsSize) ? grammarOffsets[A+1] : grammarSize;

    // midpoints between (r..c)
    int pairOffset = allFSAPairsOffsets[r * N + c];
    int pairOffsetNext = ((r * N + c) + 1 < allFSAPairsOffsetsSize)
       ? allFSAPairsOffsets[r * N + c + 1]
       : allFSAPairsSize;

    // loop over midpoints
    for (int idx = pairOffset; idx < pairOffsetNext; idx++) {
        int m = allFSAPairs[idx];
        
        // For each A -> (B, C):
        //     dp_in[r,m,B] && dp_in[m,c,C] => dp_out[r,c,A].
        for (int g = startGC; g < endGC; g += 2) {
            int B = grammar[g];
            int C = grammar[g+1];
            
            int idxBM = r * (N*numNonterminals) + m * numNonterminals + B;
            int idxMC = m * (N*numNonterminals) + c * numNonterminals + C;
            if (dp_in[idxBM] != 0 && dp_in[idxMC] != 0) {
                dp_out[dpIndex] = 1;
                atomic_store_explicit(&changedFlag, 1, memory_order_relaxed);
                return;
            }
        }
    }
}
"""

    @Language("swift") val swiftSrc = """
import Foundation
import Metal

private var device: MTLDevice!
private var queue: MTLCommandQueue!
private var psoCflMul: MTLComputePipelineState!

@_cdecl("setup") public func setup() {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!

    let src = #""${'"'}$metalSrc${'"'}""#
    let library = try! device.makeLibrary(source: src, options: nil)
    psoCflMul = try! device.makeComputePipelineState(function: library.makeFunction(name: "cfl_mul_upper")!)
}

@_cdecl("cflMultiply") public func cflMultiply(
    _ dp_in: UnsafePointer<UInt8>?,
    _ dp_out: UnsafeMutablePointer<UInt8>?,
    _ grammar: UnsafePointer<CInt>?,
    _ grammarOffsets: UnsafePointer<CInt>?,
    _ allFSAPairs: UnsafePointer<CInt>?,
    _ allFSAPairsOffsets: UnsafePointer<CInt>?,
    _ numStates: CInt,
    _ numNonterminals: CInt,
    _ grammarSize: CInt,
    _ grammarOffsetsSize: CInt,
    _ allFSAPairsSize: CInt,
    _ allFSAPairsOffsetsSize: CInt
) {
    guard let dp_in = dp_in,
          let dp_out = dp_out,
          let grammar = grammar,
          let grammarOffsets = grammarOffsets,
          let allFSAPairs = allFSAPairs,
          let allFSAPairsOffsets = allFSAPairsOffsets
    else { return }
    
    let totalCount = Int(numStates) * Int(numStates) * Int(numNonterminals)
    let N = Int(numStates)
    let totalPairs = N*(N-1)/2
    let totalThreads = totalPairs * Int(numNonterminals)
    let tiling = MTLSizeMake(totalThreads,1,1)
    let dpSizeBytes = totalCount * MemoryLayout<UInt8>.stride
    
    let bufA = device.makeBuffer(bytes: dp_in, length: dpSizeBytes, options: [])!
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

    let changedFlag = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

    for r in 0..<numStates {
        var zero: UInt32 = 0
        memcpy(changedFlag.contents(), &zero, MemoryLayout<UInt32>.size)

        let commandBuffer1 = queue.makeCommandBuffer()!
        let encoder1 = commandBuffer1.makeComputeCommandEncoder()!
        encoder1.setComputePipelineState(psoCflMul)
        encoder1.setBuffer(bufA, offset: 0, index: 0)
        encoder1.setBuffer(bufA, offset: 0, index: 1)
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
        encoder1.setBuffer(changedFlag, offset: 0, index: 12)

        encoder1.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
        encoder1.endEncoding()
        commandBuffer1.commit()
        commandBuffer1.waitUntilCompleted()

        let changedValue = changedFlag.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        if changedValue == 0 {
            print("Fixpoint escape at round: \(r)/\(numStates)")
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