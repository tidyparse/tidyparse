@file:OptIn(ExperimentalUnsignedTypes::class)

package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNF
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.Π2A
import ai.hypergraph.kaliningraph.types.π1
import ai.hypergraph.kaliningraph.types.π2
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

  var (resultGPU: List<Boolean>, resultCPU: List<Boolean>) = listOf<Boolean>() to listOf(true)

  measureTimeMillis {
    fun FSA.buildParseMatrix(unitProds: Set<Π2A<Σᐩ>>, bindex: Bindex<Σᐩ>): UShortArray =
      unitProds.parallelStream().flatMap { (A, σ) ->
        nominalForm.flattenedTriples.stream().filter { arc -> arc.π2(σ) }
          .map { arc -> Triple(stateMap[arc.π1]!!, bindex[A], stateMap[arc.third]!!) }
      }.flatMap { (q0, nt, q1) -> listOf(q0.toUShort(), q1.toUShort(), nt.toUShort()).stream() }
        .toList().toUShortArray()
    val dpIn = levFSA.buildParseMatrix(ups, bindex)

    val outGPU = GPUBridge.cflClosure(
      dpIn.also { println("Sent ${it.size} shorts / ${numStates*numStates*numNonterminals}") },
      grammarFlattened,
      grammarOffsets,
      allFSAPairsFlattened,
      allFSAPairsOffsets,
      numStates, numNonterminals,
      grLen, goLen, apLen, aoLen
    ).also { resultGPU = it.map { it > 0.toByte() } }

    println("GPU size: " + outGPU.size)
  }.also { println("GPU time: ${it}ms\n") }

  measureTimeMillis {
    val dp0 = Array(levFSA.numStates) { Array(levFSA.numStates) { BooleanArray(width) { false } } }
    levFSA.allIndexedTxs0(ups, bindex).forEach { (q0, nt, q1) -> dp0[q0][q1][nt] = true }

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
  @Language("c++") val metalSrc = """
#include <metal_stdlib>
using namespace metal;

// Helper to decode (row,col) in the upper-triangle from tid
inline void decodeUpperTriangle(int i, int N, thread int &r, thread int &c) {
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

kernel void serialize_dpm(
  device const uint8_t* bufA [[buffer(0)]],
  device uint16_t* dp_out [[buffer(1)]],
  device atomic_uint& tripleCount [[buffer(2)]],
  constant uint& numStates [[buffer(3)]],
  constant uint& numNonterminals [[buffer(4)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= numStates * numStates * numNonterminals) return;
  if (bufA[gid] == 1) {
      uint r = gid / (numStates * numNonterminals);
      uint c = (gid / numNonterminals) % numStates;
      uint A = gid % numNonterminals;
      uint writePos = atomic_fetch_add_explicit(&tripleCount, 1, memory_order_relaxed);
      uint tripleIndex = writePos * 3;
      dp_out[tripleIndex]     = uint16_t(r);
      dp_out[tripleIndex + 1] = uint16_t(c);
      dp_out[tripleIndex + 2] = uint16_t(A);
  }
}
"""

  val swiftImports = "import Foundation\nimport Metal"

  @JvmStatic fun cflClosure(
/**[NativeBridge.cflMultiply] */
    dpIn: UShortArray,
    grammar: IntArray,
    grammarOffsets: IntArray,
    allFSAPairs: IntArray,
    allFSAPairsOffsets: IntArray,
    numStates: Int, numNonterminals: Int, grLen: Int, goLen: Int, apLen: Int, aoLen: Int
  ): ByteArray =
    Memory(numStates.toLong() * numStates * numNonterminals).also { outMem ->
      nativeBridge.cflMultiply(
        dp_in = Memory(2L * dpIn.size.toLong()).apply { write(0, dpIn.toShortArray(), 0, dpIn.size) },
        dp_out = outMem,
//        dp_in = Memory(dpIn.size).apply { write(0, dpIn, 0, dpIn.size) },
//        dp_out = outMem,
        grammar = Memory(4L * grammar.size).apply { write(0, grammar, 0, grammar.size) },
        grammarOffsets = Memory(4L * grammarOffsets.size).apply { write(0, grammarOffsets, 0, grammarOffsets.size) },
        allFSAPairs = Memory(4L * allFSAPairs.size).apply { write(0, allFSAPairs, 0, allFSAPairs.size) },
        allFSAPairsOffsets = Memory(4L * allFSAPairsOffsets.size).apply { write(0, allFSAPairsOffsets, 0, allFSAPairsOffsets.size) },
        dpIn.size, numStates, numNonterminals, grLen, goLen, apLen, aoLen
      )
    }.getByteArray(0, numStates * numStates * numNonterminals)

  interface NativeBridge : Library {
    fun setup() fun cflMultiply(
    /** [GPUBridge.swiftSrc] */
      // Expects a flattened array of triples containing the indices (r, c, A) that are initially set
      dp_in: Pointer,              // buffer(0)
      dp_out: Pointer,             // buffer(1)
      grammar: Pointer,            // buffer(2)
      grammarOffsets: Pointer,     // buffer(3)
      allFSAPairs: Pointer,        // buffer(4)
      allFSAPairsOffsets: Pointer, // buffer(5)
      dpSize: Int,
      numStates: Int,
      numNonterminals: Int,
      grammarSize: Int,
      grammarOffsetsSize: Int,
      allFSAPairsSize: Int,
      allFSAPairsOffsetsSize: Int
    )
  }
  /** Caller: [GPUBridge.cflClosure] */
  @Language("swift") val swiftSrc = """$swiftImports
@_cdecl("cflMultiply") public func cflMultiply(
    _ dp_in: UnsafePointer<UInt16>?,
    _ dp_out: UnsafeMutablePointer<UInt16>?,
    _ grammar: UnsafePointer<CInt>?,
    _ grammarOffsets: UnsafePointer<CInt>?,
    _ allFSAPairs: UnsafePointer<CInt>?,
    _ allFSAPairsOffsets: UnsafePointer<CInt>?,
    _ dpSize: CInt,
    _ numStates: CInt,
    _ numNonterminals: CInt,
    _ grammarSize: CInt,
    _ grammarOffsetsSize: CInt,
    _ allFSAPairsSize: CInt,
    _ allFSAPairsOffsetsSize: CInt
) {
    guard let dp_out, let grammar, let grammarOffsets, let allFSAPairs, let allFSAPairsOffsets else { return }
    
    let N = Int(numStates)
    let totalCount = N*N*Int(numNonterminals)
    let dpSizeBytes = totalCount * MemoryLayout<UInt8>.stride

    let bufA = reconstructDPBuffer(
        dp_in: dp_in,
        dpSize: Int(dpSize),
        numStates: Int(numStates),
        numNonterminals: Int(numNonterminals),
        dpSizeBytes: Int(dpSizeBytes),
        totalCount: totalCount
    )

    let stride = MemoryLayout<Int32>.stride

    var ns = numStates
    var nt = numNonterminals
    var gr = grammarSize
    var go = grammarOffsetsSize
    var ap = allFSAPairsSize
    var ao = allFSAPairsOffsetsSize

    let totalPairs = N*(N-1)/2
    let totalThreads = totalPairs * Int(numNonterminals)
    let tiling = MTLSizeMake(totalThreads,1,1)

    let grammarBuf = device.makeBuffer(bytes: grammar, length: Int(gr)*stride, options: [])!
    let grammarOffBuf = device.makeBuffer(bytes: grammarOffsets, length: Int(go)*stride, options: [])!
    let pairsBuf = device.makeBuffer(bytes: allFSAPairs, length: Int(ap)*stride, options: [])!
    let pairsOffBuf = device.makeBuffer(bytes: allFSAPairsOffsets, length: Int(ao)*stride, options: [])!

    let changedFlag = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

    for r in 0..<numStates {
        var zero: UInt32 = 0
        memcpy(changedFlag.contents(), &zero, MemoryLayout<UInt32>.size)

        let cmb = queue.makeCommandBuffer()!
        let enc = cmb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoCflMul)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufA, offset: 0, index: 1)
        enc.setBuffer(grammarBuf, offset: 0, index: 2)
        enc.setBuffer(grammarOffBuf, offset: 0, index: 3)
        enc.setBuffer(pairsBuf, offset: 0, index: 4)
        enc.setBuffer(pairsOffBuf, offset: 0, index: 5)
        enc.setBytes(&ns, length: stride, index: 6)
        enc.setBytes(&nt, length: stride, index: 7)
        enc.setBytes(&gr, length: stride, index: 8)
        enc.setBytes(&go, length: stride, index: 9)
        enc.setBytes(&ap, length: stride, index: 10)
        enc.setBytes(&ao, length: stride, index: 11)
        enc.setBuffer(changedFlag, offset: 0, index: 12)

        enc.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
        enc.endEncoding()
        cmb.commit()
        cmb.waitUntilCompleted()

        let changedValue = changedFlag.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        if changedValue == 0 {
            print("Fixpoint escape at round: \(r)/\(numStates)")
            break
        }
    }

//var dpo: UnsafeMutablePointer<UInt16>? = nil
//var dpSize: CInt = 0
//
//let start = DispatchTime.now()
//deconstructDPBuffer(
//    bufA: bufA,
//    numStates: numStates,
//    numNonterminals: numNonterminals,
//    dpSize: &dpSize,
//    dp_out: &dpo
//)
//let end = DispatchTime.now()
//
//let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
//let timeMs = Double(nanoTime) / 1_000_000
//print("Deconstruct time: \(timeMs) ms, \(Int(dpSize)) shorts") // ~1ms

    memcpy(dp_out, bufA.contents(), dpSizeBytes)
}

private func reconstructDPBuffer(
    dp_in: UnsafePointer<UInt16>?,
    dpSize: Int,
    numStates: Int,
    numNonterminals: Int,
    dpSizeBytes: Int,
    totalCount: Int
) -> MTLBuffer {
    let rowMultiplier = numStates * numNonterminals  // Precomputed for row index
    let colMultiplier = numNonterminals              // Precomputed for column index

    // Create and initialize bufA
    let bufA = device.makeBuffer(length: dpSizeBytes, options: [])!
    let ptr = bufA.contents().bindMemory(to: UInt8.self, capacity: totalCount)
    memset(ptr, 0, dpSizeBytes)

    // Reconstruct DP array from triples
    if let dp_in = dp_in {
        let triples = UnsafeBufferPointer(start: dp_in, count: Int(dpSize))
        let numTriples = Int(dpSize) / 3  // Number of triples; assumes dpSize % 3 == 0
        
        // Process triples in batches to improve cache locality
        let batchSize = 10  // Adjust this value based on performance testing
        DispatchQueue.concurrentPerform(iterations: (numTriples + batchSize - 1) / batchSize) { batchIdx in
            let batchStart = batchIdx * batchSize
            let batchEnd = min(batchStart + batchSize, numTriples)
            for i in batchStart..<batchEnd {
                let r = Int(triples[3 * i])
                let c = Int(triples[3 * i + 1])
                let A = Int(triples[3 * i + 2])
                let index = r * rowMultiplier + c * colMultiplier + A
                ptr[index] = 2
            }
        }
    }

    return bufA
}

func deconstructDPBuffer(
    bufA: MTLBuffer,
    numStates: CInt,
    numNonterminals: CInt,
    dpSize: inout CInt,
    dp_out: inout UnsafeMutablePointer<UInt16>?
) {
    // Calculate total size of bufA
    let numStatesInt = Int(numStates)
    let numNonterminalsInt = Int(numNonterminals)
    let totalCount = numStatesInt * numNonterminalsInt * numStatesInt
    
    // Allocate dp_out with an upper-bound size (worst case: all elements are 1)
    let maxTriples = totalCount // Maximum possible number of 1s
    let maxDpSizeBytes = maxTriples * 3 * MemoryLayout<UInt16>.stride
    let dpOutBuffer = device.makeBuffer(length: maxDpSizeBytes, options: .storageModeShared)!
    dp_out = dpOutBuffer.contents().bindMemory(to: UInt16.self, capacity: maxTriples * 3)
    
    // Allocate a buffer for the atomic counter (initialized to 0)
    let counterBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    let counterPtr = counterBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
    counterPtr[0] = 0
    
    // Set up command buffer and encoder
    let commandBuffer = queue.makeCommandBuffer()!
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeEncoder.setComputePipelineState(serialize)
    
    // Bind buffers and constants
    computeEncoder.setBuffer(bufA, offset: 0, index: 0)
    computeEncoder.setBuffer(dpOutBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(counterBuffer, offset: 0, index: 2)
    var numStatesUInt = UInt32(numStates)
    var numNonterminalsUInt = UInt32(numNonterminals)
    computeEncoder.setBytes(&numStatesUInt, length: 4, index: 3)
    computeEncoder.setBytes(&numNonterminalsUInt, length: 4, index: 4)
    
    // Dispatch threads
    let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
    let numThreadgroups = MTLSize(width: (totalCount + 255) / 256, height: 1, depth: 1)
    computeEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
    
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    // Update dpSize with the actual number of triples * 3 (number of UInt16 elements)
    dpSize = CInt(counterPtr[0]) * 3
}

private var device: MTLDevice!, queue: MTLCommandQueue!, psoCflMul: MTLComputePipelineState!, serialize: MTLComputePipelineState!

@_cdecl("setup") public func setup() {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!

    let src = #""${'"'}$metalSrc${'"'}""#
    let library = try! device.makeLibrary(source: src, options: nil)
    psoCflMul = try! device.makeComputePipelineState(function: library.makeFunction(name: "cfl_mul_upper")!)
    serialize = try! device.makeComputePipelineState(function: library.makeFunction(name: "serialize_dpm")!)
}"""

  private val nativeBridge: NativeBridge =
    if (System.getProperty("os.name").startsWith("Mac")) getMetalBridge() else TODO()

  private fun getMetalBridge(): NativeBridge {
    val directory = "src/main/resources/dlls".also { File(it).mkdirs() }
    val dylib = File("$directory/libMetalBridge.dylib")

    if (needsRebuild(dylib, swiftSrc, directory)) buildNative(directory, dylib, swiftSrc)
    return (Native.load(dylib.absolutePath, NativeBridge::class.java) as NativeBridge).also { it.setup() }
  }

  private fun needsRebuild(dylib: File, swiftSrc: String, dir: String) =
    !dylib.exists() || !File("$dir/.swiftHash").let { it.exists() && it.readText() == swiftSrc.hashCode().toString() }

  private fun buildNative(dir: String, dylib: File, swiftSrc: String) {
    val clock = TimeSource.Monotonic.markNow()
    val path = File("$dir/MetalBridge.swift").apply { writeText(swiftSrc) }.path
    ("xcrun swiftc -emit-library $path -o ${dylib.absolutePath} -module-name M " +
        "-Xlinker -install_name -Xlinker @rpath/${dylib.path}")
      .run { ProcessBuilder(split(" ")).inheritIO().start().waitFor() }
      .also { if (it != 0) error("Failed to build Swift bridging code!") }
    File("$dir/.swiftHash").writeText(swiftSrc.hashCode().toString())
    println("Finished rebuild in ${clock.elapsedNow()}")
  }
}