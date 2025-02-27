@file:OptIn(ExperimentalUnsignedTypes::class)

package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.automata.StrPred
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
  val radius = 5
  val levFSA = makeLevFSA(pythonCode, radius)
  val maxWordLen = pythonCode.size + radius + 10

  val numStates = levFSA.numStates
  val numNonterminals = cfg.nonterminals.size

  val allPairs = levFSA.allPairs2
//    .mapIndexed { p, l1 -> l1.mapIndexed { q, l2 -> l2.filter { it != p && it != q } } }
    .flatten()
  val allFSAPairsFlattened = allPairs.flatten().toIntArray()
  val allFSAPairsOffsets = allPairs.map { it.size }
    .fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()

  println("\nLinked GPU in ${measureTimeMillis { GPUBridge.setupPython() }}ms\n")

  measureTimeMillis {
    fun FSA.byteFormat(cfg: CFG): UShortArray {
      val tmMap = cfg.tmMap
      fun StrPred.predByte(): UShort =
        (if (arg == "[.*]") -127 // Maximum value means all possible terminals
        else if (arg.startsWith("[!=]")) -(2 + (tmMap[arg.drop(4)] ?: 125)) // Represent negation using negative sign bit
        else 2 + tmMap[arg]!!).toUShort() // Otherwise positive sign bit

      return cfg.unitProductions.flatMap { (A, σ) ->
        nominalForm.flattenedTriples.filter { arc -> arc.second(σ) }.map { (q0, sp, q1) ->
          listOf(stateMap[q0]!!, stateMap[q1]!!, cfg.bindex[A]).map { it.toUShort() } + sp.predByte()
        }.flatten()
      }.toUShortArray()
       .also { println("Encoded instance in ${it.size * 2} bytes / ${numStates*numStates*numNonterminals}") }
    }

    val maxSamples = 10_000
    val samples = GPUBridge.cflClosure(
      levFSA.byteFormat(cfg),
      allFSAPairsFlattened,
      allFSAPairsOffsets,
      levFSA.finalIdxs.toIntArray(),
      levFSA.finalIdxs.size,
      numStates,
      maxWordLen,
      maxSamples
    )

    var admissible = 0
    samples.joinToString(" ") {
      if (it == 0.toByte()) "0"
      else if (it.toInt() - 2 !in cfg.tmLst.indices) "??${it.toInt()}??" // Should never happen
      else if (it - 2 == 0) "" // TODO: why do we need to block NEWLINE?
      else cfg.tmLst[it.toInt() - 2]
    }.split(Regex("( 0)+"))
      .filter { it.isNotBlank() }
      .distinct()
      .onEach { if("$it NEWLINE" in pythonStatementCNF.language) admissible++ }
      .onEachIndexed { i, it -> if (i < 9) println("$i.) ${it.trim()}") }
      .toList().also { println("...\nGPU repairs: $admissible valid / ${it.size} unique / $maxSamples total") }
  }.also { println("GPU time: ${it}ms\n") }
}

object GPUBridge {
  @Language("c++") val metalSrc = """
#include <metal_stdlib>
using namespace metal;

// The following header should be regenerated for each CFG
${encodeGrammarHeader()}

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
  const device int*    allFSAPairs             [[buffer(2)]],
  const device int*    allFSAPairsOffsets      [[buffer(3)]],
  constant int&        numStates               [[buffer(4)]],
  constant int&        allFSAPairsSize         [[buffer(5)]],
  constant int&        allFSAPairsOffsetsSize  [[buffer(6)]],
  device atomic_uint&  numNonzero              [[buffer(7)]],
  uint                 tid                     [[thread_position_in_grid]]
) {
    int r, c;
    decodeUpperTriangle(int(tid) / numNonterminals, numStates, r, c);
    int A = tid % numNonterminals, snt = numStates * numNonterminals, dpIndex = r*snt + c*numNonterminals + A;

    if (dp_in[dpIndex]) {
      dp_out[dpIndex] = dp_in[dpIndex];
      atomic_fetch_add_explicit(&numNonzero, 1, memory_order_relaxed);
      return;
    }

    // Grammar offsets for A
    int startGC = vidx_offsets[A];
    int endGC   = (A+1 < volen) ? vidx_offsets[A+1] : vilen;

    // midpoints between (r..c)
    int aoi = r * numStates + c + 1;
    int pairOffset = allFSAPairsOffsets[aoi - 1];
    int pairOffsetNext = (aoi < allFSAPairsOffsetsSize) ? allFSAPairsOffsets[aoi] : allFSAPairsSize;

    // loop over midpoints
    for (int idx = pairOffset; idx < pairOffsetNext; idx++)
      for (int g = startGC, m = allFSAPairs[idx]; g < endGC; g += 2) {
         int B = vidx[g]; int C = vidx[g+1];
         
         int idxBM = r * snt + m * numNonterminals + B;
         int idxMC = m * snt + c * numNonterminals + C;
         if (dp_in[idxBM] != 0 && dp_in[idxMC] != 0) {
             dp_out[dpIndex] = 1;
             atomic_fetch_add_explicit(&numNonzero, 1, memory_order_relaxed);
             return;
         }
      }
}

kernel void bp_count(
    const device uchar* dp_in                   [[buffer(0)]],
    const device int*   allFSAPairs             [[buffer(1)]],
    const device int*   allFSAPairsOffsets      [[buffer(2)]],
    constant int&       numStates               [[buffer(3)]],
    constant int&       allFSAPairsSize         [[buffer(4)]],
    constant int&       allFSAPairsOffsetsSize  [[buffer(5)]],
    device int*         bpCount                 [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    int N = numStates;
    int totalCells = (N*(N-1))/2 * numNonterminals;
    if (tid >= totalCells) return;

    int r, c;
    decodeUpperTriangle(tid / numNonterminals, N, r, c);
    int A = tid % numNonterminals, snt = N * numNonterminals, dpIndex = r*snt + c*numNonterminals + A;

    if (dp_in[dpIndex] == 0) { bpCount[dpIndex] = 0; return; }

    // Grammar offsets for A
    int startGC = vidx_offsets[A];
    int endGC   = (A+1 < volen) ? vidx_offsets[A+1] : vilen;

    // All possible midpoints
    int aoi = r*N + c + 1;
    int pairOffset = allFSAPairsOffsets[aoi - 1];
    int pairOffsetNext = (aoi < allFSAPairsOffsetsSize) ? allFSAPairsOffsets[aoi] : allFSAPairsSize;

    int count = 0;
    // loop over midpoints
    for (int idx = pairOffset; idx < pairOffsetNext; idx++)
      for (int g = startGC, m = allFSAPairs[idx]; g < endGC; g += 2) {
        int B = vidx[g], C = vidx[g+1];

        int idxBM = r*snt + m*numNonterminals + B;
        int idxMC = m*snt + c*numNonterminals + C;
        // If (r,m,B) and (m,c,C) are both present:
        if (dp_in[idxBM] != 0 && dp_in[idxMC] != 0) count++;
      }

    bpCount[r*snt + c*numNonterminals + A] = max(count, 1);
}

kernel void bp_write(
    const device uchar* dp_in                  [[buffer(0)]],
    const device int*   allFSAPairs            [[buffer(1)]],
    const device int*   allFSAPairsOffsets     [[buffer(2)]],
    constant int&       numStates              [[buffer(3)]],
    constant int&       allFSAPairsSize        [[buffer(4)]],
    constant int&       allFSAPairsOffsetsSize [[buffer(5)]],
    const device int*   bpCount                [[buffer(6)]],
    const device int*   bpOffset               [[buffer(7)]],
    // Suppose each entry in bpStorage is 2 x int. 
    device int*         bpStorage              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    int N = numStates;
    int totalCells = (N*(N-1))/2 * numNonterminals;

    int r, c;
    decodeUpperTriangle(tid / numNonterminals, N, r, c);
    int A = tid % numNonterminals, snt = N * numNonterminals, dpIndex = r*snt + c*numNonterminals + A;

    if (bpCount[dpIndex] == 0) return;

    // Grammar offsets
    int startGC = vidx_offsets[A];
    int endGC   = (A+1 < volen) ? vidx_offsets[A+1] : vilen;

    int aoi = r*N + c + 1;
    int pairOffset = allFSAPairsOffsets[aoi - 1];
    int pairOffsetNext = (aoi < allFSAPairsOffsetsSize) ? allFSAPairsOffsets[aoi] : allFSAPairsSize;
    int outPos = bpOffset[dpIndex];

    // Exactly like bp_count, but now we store expansions
    for (int idx = pairOffset; idx < pairOffsetNext; idx++)
      for (int g = startGC, m = allFSAPairs[idx]; g < endGC; g += 2) {
        int B = vidx[g], C = vidx[g+1];

        int idxBM = r*snt + m*numNonterminals + B;
        int idxMC = m*snt + c*numNonterminals + C;
        if (dp_in[idxBM] != 0 && dp_in[idxMC] != 0) {
          // store (B, m, C) in bpStorage
          // Suppose each triple is 2 consecutive int in bpStorage:
          bpStorage[outPos*2 + 0] = idxBM;
          bpStorage[outPos*2 + 1] = idxMC;
          outPos++;
        }
      }
}

inline uint lcg_random(thread uint& state) { return 1664525u * state + 1013904223u; }
//inline uint lcg_random(thread uint& state) { return state % 10; }
inline uint lcg_randomRange(thread uint& state, uint range) { return lcg_random(state) % range; }

inline void sampleCellIterative(
   const device uchar*    dp_in,
   const device int*      bpCount,
   const device int*      bpOffset,
   const device int*      bpStorage,
   int                    startDPIdx,
   thread uint&           rngState,
   thread uchar*          localWord,    // output buffer
   thread int&            wordLen,
   const int              maxWordLen
) {
    constexpr int MAX_STACK = 1024;
    int stack[MAX_STACK];
    int top = 0;

    // Initialize the stack with "root call"
    stack[top++] = startDPIdx;

    // While we have frames to process, and haven't overflowed localWord
    while (top > 0 && wordLen < maxWordLen - 5) {
      int dpIdx = stack[--top];
      int expCount = bpCount[dpIdx];
      
      if (((dp_in[dpIdx] != 0) && (dp_in[dpIdx] != 1))) { // If we are dealing with a leaf node (i.e., a unit nonterminal/terminal)
        int nonterminal = dpIdx % numNonterminals;
        int predicate = dp_in[dpIdx];
        bool isNegativeLiteral = predicate & 0x80;
        uchar literal = predicate & 0x7F;
        int numTms = nt_tm_lens[nonterminal];
        if (isNegativeLiteral) {
          uchar possibleTms[100];
          uint tmCount = 0; uint offset = offsets[nonterminal];
          for (int i = 0; i < min(100, numTms); i++) {
            uchar this_tm = all_tm[offset + i];
            if (this_tm + 2 != literal) possibleTms[tmCount++] = this_tm + 2;
          }
          uchar tmChoice = possibleTms[int(lcg_randomRange(rngState, uint(tmCount)))];
          localWord[wordLen++] = tmChoice;
        } else { localWord[wordLen++] = (numTms != 0) ? literal : 99; }
      } else if (expCount == 0) { localWord[wordLen++] = 111; continue; }
      else if (top + 2 < MAX_STACK) {
        int randIdx = bpOffset[dpIdx] + int(lcg_randomRange(rngState, uint(expCount)));

        int idxBM = bpStorage[randIdx * 2 + 0];
        int idxMC = bpStorage[randIdx * 2 + 1];
//         if (wordLen > 30) localWord[wordLen++] = dpIdx & 0x7F;
        int visitedBM = false, visitedMC = false;

        int i; for (i = 0; i < 1024; i++) {
            if (idxBM == stack[i]) visitedBM = true;
            if (idxMC == stack[i]) visitedMC = true;
            if (visitedBM || visitedMC) break;
        }
//        // Why is this necessary? Should never have been a loop to begin with...
        if (!visitedMC && !visitedBM) { stack[top++] = idxMC; stack[top++] = idxBM; }
//        else if (!visitedMC) {stack[top++] = idxMC;}
//        else if (!visitedBM) {stack[top++] = idxBM;}
//        else { localWord[wordLen++] = 99; }
//        if (!visitedBM) {  }
//          if (idxMC != dpIdx && idxBM != dpIdx) { stack[top++] = idxMC; stack[top++] = idxBM; }
        
//        if (idxBM != idxMC) { stack[top++] = idxMC; stack[top++] = idxBM; }
//        else { stack[top++] = idxBM; }
      }
    }
    
//    for (int i = wordLen; i < maxWordLen; i++) { localWord[wordLen++] = 0; }
}

kernel void sample_words(
    const device uchar*   dp_in           [[buffer(0)]], // parse chart
    const device int*     bpCount         [[buffer(1)]], // parse chart with counts
    const device int*     bpOffset        [[buffer(2)]], // parse chart with offsets into bpStorage
    const device int*     bpStorage       [[buffer(3)]], // flat array without zeros
    const device int*     startIndices    [[buffer(4)]], // each entry is a dpIndex
    const device uint*    seeds           [[buffer(5)]],
    device char*          sampledWords    [[buffer(6)]],
    constant int&         numStartIndices [[buffer(7)]],
    constant int&         numStates       [[buffer(8)]],
    constant int&         maxWordLen      [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    // local word in thread memory
    thread uchar localWord[1024];
    for (int i=0; i<maxWordLen; i++) { localWord[i] = 0; }

    // pick a random dpIndex from [0..numStartIndices-1]
    uint rngState = seeds[tid];
    int dpIndex = startIndices[lcg_randomRange(rngState, (uint)numStartIndices)];

    int A = dpIndex % numNonterminals; // This should always be 74=START
    int rowCol = dpIndex / numNonterminals;
    int c = rowCol % numStates;
    int r = rowCol / numStates;

    int snt = numStates * numNonterminals;
    int startIdx = r*snt + c*numNonterminals + A;

    int wordLen = 0;
    sampleCellIterative(dp_in, bpCount, bpOffset, bpStorage, startIdx, rngState, localWord, wordLen, maxWordLen);

    for (int i=0; i<maxWordLen; i++) sampledWords[int(tid)*maxWordLen + i] = localWord[i];
}
"""

  val swiftImports = "import Foundation\nimport Metal\nimport Dispatch"

  @JvmStatic fun cflClosure(
/**[NativeBridge.cflMultiply] */
    dpIn: UShortArray,
    allFSAPairs: IntArray,
    allFSAPairsOffsets: IntArray,
    acceptStates: IntArray,
    acceptStatesSize: Int,
    numStates: Int,
    maxWordLen: Int,
    maxSamples: Int
  ): ByteArray =
    Memory(maxWordLen * maxSamples.toLong()).also { outMem ->
      nativeBridge.cflMultiply(
        dp_in = Memory(2L * dpIn.size.toLong()).apply { write(0, dpIn.toShortArray(), 0, dpIn.size) },
        dp_out = outMem,
//        dp_in = Memory(dpIn.size).apply { write(0, dpIn, 0, dpIn.size) },
//        dp_out = outMem,
        allFSAPairs = Memory(4L * allFSAPairs.size).apply { write(0, allFSAPairs, 0, allFSAPairs.size) },
        allFSAPairsOffsets = Memory(4L * allFSAPairsOffsets.size).apply { write(0, allFSAPairsOffsets, 0, allFSAPairsOffsets.size) },
        acceptStates =  Memory(4L * acceptStatesSize).apply { write(0, acceptStates, 0, acceptStates.size) },
        acceptStatesSize = acceptStatesSize,
        dpSize = dpIn.size,
        numStates = numStates,
        allFSAPairsSize = allFSAPairs.size,
        allFSAPairsOffsetsSize = allFSAPairsOffsets.size,
        maxWordLen = maxWordLen,
        maxSamples = maxSamples
      )
    }.getByteArray(0, maxWordLen * maxSamples)

  interface NativeBridge : Library {
    fun setup(numNts: Int, startIdx: Int, s: String)

               fun cflMultiply(
    /** [GPUBridge.swiftSrc] */
      // Expects a flattened array of triples containing the indices (r, c, A) that are initially set
      dp_in: Pointer,              // buffer(0)
      dp_out: Pointer,             // buffer(1)
      allFSAPairs: Pointer,        // buffer(2)
      allFSAPairsOffsets: Pointer, // buffer(3)
      acceptStates: Pointer,       // buffer(4)
      acceptStatesSize: Int,
      dpSize: Int,
      numStates: Int,
      allFSAPairsSize: Int,
      allFSAPairsOffsetsSize: Int,
      maxWordLen: Int,
      maxSamples: Int,
    )
  }
  /** Caller: [GPUBridge.cflClosure] */
  @Language("swift") val swiftSrc = """$swiftImports
@_cdecl("cflMultiply") public func cflMultiply(
    _ dp_in_sparse: UnsafePointer<UInt16>?,
    _ dp_out: UnsafeMutablePointer<UInt8>?,
    _ allFSAPairs: UnsafePointer<CInt>?,
    _ allFSAPairsOffsets: UnsafePointer<CInt>?,
    _ acceptStates: UnsafePointer<CInt>?,
    _ acceptStatesSize: CInt,
    _ dpSize: CInt,
    _ numStates: CInt,
    _ allFSAPairsSize: CInt,
    _ allFSAPairsOffsetsSize: CInt,
    _ maxWordLen: CInt,
    _ maxSamples: CInt
) {
    guard let dp_out, let allFSAPairs, let allFSAPairsOffsets else { return }
    
    let N = Int(numStates)
    let totalCount = N*N*Int(numNonterminals)
    let dpSizeBytes = totalCount * MemoryLayout<UInt8>.stride

    let bufA = reconstructDPBuffer(
        dp_in: dp_in_sparse,
        dpSize: Int(dpSize),
        numStates: Int(numStates),
        numNonterminals: Int(numNonterminals),
        dpSizeBytes: Int(dpSizeBytes),
        totalCount: totalCount
    )

    let stride = MemoryLayout<Int32>.stride

    var ns = numStates
    var ap = allFSAPairsSize
    var ao = allFSAPairsOffsetsSize

    let totalPairs = N*(N-1)/2
    let totalThreads = totalPairs * Int(numNonterminals)
    let tiling = MTLSizeMake(totalThreads,1,1)

    let pairsBuf = device.makeBuffer(bytes: allFSAPairs, length: Int(ap)*stride, options: [])!
    let pairsOffBuf = device.makeBuffer(bytes: allFSAPairsOffsets, length: Int(ao)*stride, options: [])!

    let numNonzero = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    var prevValue: UInt32 = 0

    for r in 0..<numStates {
       var zero: UInt32 = 0
       memcpy(numNonzero.contents(), &zero, MemoryLayout<UInt32>.size)

       let cmb = queue.makeCommandBuffer()!
       let enc = cmb.makeComputeCommandEncoder()!
       enc.setComputePipelineState(psoCflMul)
       enc.setBuffer(bufA,        offset: 0,      index: 0)
       enc.setBuffer(bufA,        offset: 0,      index: 1)
       enc.setBuffer(pairsBuf,    offset: 0,      index: 2)
       enc.setBuffer(pairsOffBuf, offset: 0,      index: 3)
       enc.setBytes(&ns,          length: stride, index: 4)
       enc.setBytes(&ap,          length: stride, index: 5)
       enc.setBytes(&ao,          length: stride, index: 6)
       enc.setBuffer(numNonzero,  offset: 0,      index: 7)

       enc.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
       enc.endEncoding()
       cmb.commit()
       cmb.waitUntilCompleted()

       let currentValue = numNonzero.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
//       print("prev: \(prevValue), curr: \(currentValue)")
       if currentValue == prevValue {
          print("Fixpoint escape at round: \(r)/\(numStates), total=\(currentValue), size: \(dpSizeBytes)"); fflush(stdout)
          break
       }
       prevValue = currentValue
    }

    let (bpCountBuf, bpOffsetBuf, bpStorageBuf) =
        buildBackpointers(
          dp_in: bufA,
          allFSAPairs: pairsBuf,
          allFSAPairsOffsets: pairsOffBuf,
          N: Int(numStates),
          ap: Int(ap),
          ao: Int(ao)
        )

    let start = DispatchTime.now()
    let dpInSize = N * N * numNonterminals
    let dpInPtr = bufA.contents().bindMemory(to: UInt8.self, capacity: dpInSize)

    let acceptStatesSwift: [Int]
    if let acceptStatesPtr = acceptStates {
        acceptStatesSwift = UnsafeBufferPointer(start: acceptStatesPtr, count: Int(acceptStatesSize)).map { Int($0) }
    } else { acceptStatesSwift = [] }

    var startIdxs: [Int32] = []
    for q in acceptStatesSwift {
        let dpIndex = 0 * (N * numNonterminals) + q * numNonterminals + startIdx
        if dpIndex < dpInSize && dpInPtr[dpIndex] != 0 { startIdxs.append(Int32(dpIndex)) }
    }
    print("Number of valid startIndices: \(startIdxs.count)/\(acceptStatesSwift.count)")

    let startIdxBuf = device.makeBuffer(bytes: startIdxs, length: startIdxs.count * MemoryLayout<Int32>.stride, options: [])!

    let totSamples = Int(maxSamples), wordLen = Int(maxWordLen)
    var seeds = [UInt32](repeating: 0, count: totSamples)
    for i in 0..<totSamples { seeds[i] = UInt32.random(in: 0..<UInt32.max) }
    let seedsBuf = device.makeBuffer(bytes: seeds, length: totSamples*MemoryLayout<UInt32>.stride, options: [])!

    let sampledWordsBuf = sampleWords(
        dp_in: bufA,
        bpCount: bpCountBuf,
        bpOffset: bpOffsetBuf,
        bpStorage: bpStorageBuf,
        startIndices: startIdxBuf, 
        seeds: seedsBuf,
        maxWordLen: wordLen,
        maxSamples: totSamples,
        numStartIndices: startIdxs.count,
        numStates: Int(numStates)
    )

    let sampSize = totSamples * wordLen

    // Bind the buffer to UInt8 instead of CChar
    let ptr = sampledWordsBuf.contents().bindMemory(to: UInt8.self, capacity: sampSize)
    let buffer = UnsafeBufferPointer(start: ptr, count: sampSize)

    for sIdx in 0..<min(2, totSamples) {
        let start = sIdx * wordLen
        let wordSlice = buffer[start..<(start + wordLen)]
        print("Sample \(sIdx): \(Array(wordSlice))"); fflush(stdout)
    }

    let psEnd = DispatchTime.now()
    let psTimeMs = Double(psEnd.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
    print("Sampled words in \(psTimeMs) ms"); fflush(stdout)

    memcpy(dp_out, sampledWordsBuf.contents(), sampSize)
}

func sampleWords(
    dp_in: MTLBuffer,
    bpCount: MTLBuffer,
    bpOffset: MTLBuffer,
    bpStorage: MTLBuffer,
    startIndices: MTLBuffer,
    seeds: MTLBuffer,
    maxWordLen: Int,
    maxSamples: Int,
    numStartIndices: Int,
    numStates: Int
) -> MTLBuffer {
    // We'll store each sampled word in a row of length `maxWordLen`.
    let sampledWordsSize = maxSamples * maxWordLen
    let sampledWordsBuf = device.makeBuffer(length: sampledWordsSize, options: [])!
//    memset(sampledWordsBuf.contents(), 0, sampledWordsSize) // Something strange happens here

    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(psoSmpWord)

    encoder.setBuffer(dp_in,           offset: 0, index: 0)
    encoder.setBuffer(bpCount,         offset: 0, index: 1)
    encoder.setBuffer(bpOffset,        offset: 0, index: 2)
    encoder.setBuffer(bpStorage,       offset: 0, index: 3)
    encoder.setBuffer(startIndices,    offset: 0, index: 4)
    encoder.setBuffer(seeds,           offset: 0, index: 5)
    encoder.setBuffer(sampledWordsBuf, offset: 0, index: 6)
    var nsStart = Int32(numStartIndices)
    encoder.setBytes(&nsStart, length: MemoryLayout<Int32>.size, index: 7)
    var nStates = Int32(numStates)
    encoder.setBytes(&nStates, length: MemoryLayout<Int32>.size, index: 8)
    var mwl = Int32(maxWordLen)
    encoder.setBytes(&mwl, length: MemoryLayout<Int32>.size, index: 9)

    encoder.dispatchThreads(MTLSizeMake(maxSamples, 1, 1), threadsPerThreadgroup: MTLSizeMake(1,1,1))

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return sampledWordsBuf
}

func buildBackpointers(
  dp_in: MTLBuffer, // size = N*N*numNonterminals (bytes)
  allFSAPairs: MTLBuffer,
  allFSAPairsOffsets: MTLBuffer,
  N: Int,
  ap: Int,
  ao: Int
) -> (MTLBuffer, MTLBuffer, MTLBuffer) {
    let start = DispatchTime.now()
    // 1) Allocate bpCount
    let countSize = N * N * numNonterminals * MemoryLayout<Int32>.stride
    let bpCountBuf = device.makeBuffer(length: countSize, options: [])!

    // 2) Launch bp_count kernel
    let totalCells = (N*(N-1))/2 * numNonterminals
    let threadsCount = MTLSizeMake(totalCells,1,1)

    // Command buffer
    let commandBuffer1 = queue.makeCommandBuffer()!
    let encoder1 = commandBuffer1.makeComputeCommandEncoder()!
    let stride = MemoryLayout<Int32>.stride
    var ns = N
    var ap = ap
    var ao = ao
    encoder1.setComputePipelineState(psoBpCount)  // create psoBpCount from "bp_count" function
    encoder1.setBuffer(dp_in,               offset: 0,      index: 0)
    encoder1.setBuffer(allFSAPairs,         offset: 0,      index: 1)
    encoder1.setBuffer(allFSAPairsOffsets,  offset: 0,      index: 2)
    encoder1.setBytes(&ns,                  length: stride, index: 3)
    encoder1.setBytes(&ap,                  length: stride, index: 4)
    encoder1.setBytes(&ao,                  length: stride, index: 5)
    encoder1.setBuffer(bpCountBuf,          offset: 0,      index: 6)

    encoder1.dispatchThreads(threadsCount, threadsPerThreadgroup: MTLSizeMake(1,1,1))
    encoder1.endEncoding()
    commandBuffer1.commit()
    commandBuffer1.waitUntilCompleted()

    // 3) Compute prefix sum of bpCount --> bpOffset 
    let bpOffsetBuf = device.makeBuffer(length: countSize, options: [])!
    parallelPrefixSumCPU(countBuf: bpCountBuf, offsetBuf: bpOffsetBuf, totalCells: totalCells)

    // 4) Figure out total expansions = last element's offset + last element's count
    //    (We must read back the prefix sum array, or at least the last element.)
    var totalExpansions: Int32 = 0
    do {
      let ptrOffset = bpOffsetBuf.contents().bindMemory(to: Int32.self, capacity: totalCells)
      let ptrCount  = bpCountBuf.contents().bindMemory(to: Int32.self,  capacity: totalCells)
      let lastOffset = ptrOffset[totalCells - 1]
      let lastCount  = ptrCount[totalCells - 1]
      totalExpansions = lastOffset + lastCount
    }

    // 5) Allocate bpStorage
    //    Each expansion is 2 ints.
    let bpStorageSize = Int(totalExpansions) * 2 * MemoryLayout<Int32>.stride
    let bpStorageBuf = device.makeBuffer(length: bpStorageSize, options: [])!

    // 6) Launch bp_write kernel
    let commandBuffer2 = queue.makeCommandBuffer()!
    let encoder2 = commandBuffer2.makeComputeCommandEncoder()!
    encoder2.setComputePipelineState(psoBpWrite)
    encoder2.setBuffer(dp_in,              offset: 0,      index: 0)
    encoder2.setBuffer(allFSAPairs,        offset: 0,      index: 1)
    encoder2.setBuffer(allFSAPairsOffsets, offset: 0,      index: 2)
    encoder2.setBytes(&ns,                 length: stride, index: 3)
    encoder2.setBytes(&ap,                 length: stride, index: 4)
    encoder2.setBytes(&ao,                 length: stride, index: 5)
    encoder2.setBuffer(bpCountBuf,         offset: 0,      index: 6)
    encoder2.setBuffer(bpOffsetBuf,        offset: 0,      index: 7)
    encoder2.setBuffer(bpStorageBuf,       offset: 0,      index: 8)

    encoder2.dispatchThreads(threadsCount, threadsPerThreadgroup: MTLSizeMake(1,1,1))
    encoder2.endEncoding()
    commandBuffer2.commit()
    commandBuffer2.waitUntilCompleted()

    let end = DispatchTime.now()
    let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
    let timeMs = Double(nanoTime) / 1_000_000
    print("Built backpointers in \(timeMs) ms"); fflush(stdout)

    return (bpCountBuf, bpOffsetBuf, bpStorageBuf)
}

// TODO: GPU parallel prefix sum?
func parallelPrefixSumCPU(countBuf: MTLBuffer, offsetBuf: MTLBuffer, totalCells: Int) {
    let psStart = DispatchTime.now()
    // Bind memory of each buffer to `Int32`.
    let ptrCount  = countBuf.contents().bindMemory(to: Int32.self, capacity: totalCells)
    let ptrOffset = offsetBuf.contents().bindMemory(to: Int32.self, capacity: totalCells)

    // Decide how many chunks (and threads) we want to use.
    // For large arrays, 8 or 16 is typical; for smaller arrays, fewer might be fine.
    let numChunks = 8
    let chunkSize = (totalCells + numChunks - 1) / numChunks

    // This array will hold the total sum of each chunk (to be prefix-summed later).
    var partialSum = [Int32](repeating: 0, count: numChunks)

    // === Phase 1: Parallel local prefix sums ===
    //
    // Each chunk i does:
    //   offset[start]   = 0
    //   offset[start+1] = count[start]
    //   ...
    //   offset[end-1]   = sum of count from start..(end-2)
    // The chunk's total sum is partialSum[i].
    DispatchQueue.concurrentPerform(iterations: numChunks) { chunkIndex in
        let start = chunkIndex * chunkSize
        let end   = min(start + chunkSize, totalCells)
        if start >= end { partialSum[chunkIndex] = 0; return }

        var sum: Int32 = 0
        for i in start..<end { ptrOffset[i] = sum; sum += ptrCount[i] }
        partialSum[chunkIndex] = sum
    }

    // === Phase 2: Prefix sum of partialSum array (single-threaded) ===
    for i in 1..<numChunks { partialSum[i] += partialSum[i - 1] }

    // === Phase 3: Add offsets to each chunk in parallel ===
    //
    // For chunk i > 0, we add partialSum[i-1] to everything in that chunk's offset array.
    DispatchQueue.concurrentPerform(iterations: numChunks) { chunkIndex in
        // chunk 0 doesn't need adjustment because it starts at 0
        guard chunkIndex > 0 else { return }

        let start = chunkIndex * chunkSize
        let end   = min(start + chunkSize, totalCells)
        let offsetAdjustment = partialSum[chunkIndex - 1]

        for i in start..<end { ptrOffset[i] += offsetAdjustment }
    }

    let psEnd = DispatchTime.now()
    let psTimeMs = Double(psEnd.uptimeNanoseconds - psStart.uptimeNanoseconds) / 1_000_000
    print("Computed prefix sum in \(psTimeMs) ms"); fflush(stdout)
}

// takes a flattened array of sparse quadruples and reconstructs a dense matrix
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
        let stride = 4
        let numTriples = Int(dpSize) / stride // Number of triples; assumes dpSize % 4 == 0

        // Process triples in batches to improve cache locality
        let batchSize = 10  // Adjust this value based on performance testing
        DispatchQueue.concurrentPerform(iterations: (numTriples + batchSize - 1) / batchSize) { batchIdx in
            let batchStart = batchIdx * batchSize
            let batchEnd = min(batchStart + batchSize, numTriples)
            for i in batchStart..<batchEnd {
                let r = Int(triples[stride * i])
                let c = Int(triples[stride * i + 1])
                let A = Int(triples[stride * i + 2])
                let s = UInt8(truncatingIfNeeded: triples[stride * i + 3])
                let index = r * rowMultiplier + c * colMultiplier + A
                ptr[index] = s
            }
        }
    }

    return bufA
}

private var device: MTLDevice!, queue: MTLCommandQueue!, numNonterminals: Int = 0, startIdx: Int = 0,
            psoCflMul: MTLComputePipelineState!, serialize: MTLComputePipelineState!,
            psoBpCount: MTLComputePipelineState!, psoBpWrite: MTLComputePipelineState!,
            psoSmpWord: MTLComputePipelineState!

@_cdecl("setup") public func setup(_ nnt: CInt, startIndex: CInt, _ str: UnsafePointer<CChar>?) {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!

    let metalSrc = str != nil ? String(cString: str!) : ""
    numNonterminals = Int(nnt)
    startIdx = Int(startIndex)
    let library = try! device.makeLibrary(source: metalSrc, options: nil)
    psoCflMul   = try! device.makeComputePipelineState(function: library.makeFunction(name: "cfl_mul_upper")!)
    psoBpCount  = try! device.makeComputePipelineState(function: library.makeFunction(name: "bp_count")!)
    psoBpWrite  = try! device.makeComputePipelineState(function: library.makeFunction(name: "bp_write")!)
    psoSmpWord  = try! device.makeComputePipelineState(function: library.makeFunction(name: "sample_words")!)
}"""

  fun encodeGrammarHeader(cfg: CFG = pythonStatementCNF): String =
    genNTTerminalMapping(cfg) + genFlattenedBinaryProds(cfg)

  fun genNTTerminalMapping(cfg: CFG): String {
    val terminalLists = cfg.nonterminals.map {
      if (it !in cfg.unitNonterminals) emptyList<Int>()
      else cfg.bimap.UNITS[it]!!.map { cfg.tmMap[it]!! }
    }
    val allTerminals = terminalLists.flatMap { it }
    val offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1)

    return """constant uint8_t all_tm[] = {${allTerminals.joinToString(",")}};
        constant uint offsets[] = {${offsets.joinToString(",")}};
        constant uint nt_tm_lens[] = {${terminalLists.map { it.size }.joinToString(",")}};
        constant int numNonterminals = ${cfg.nonterminals.size};""".trimIndent()
  }

  fun genFlattenedBinaryProds(cfg: CFG): String {
    val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
    val grLen = grammarFlattened.size.also { println("grLen: $it") }
    val grammarOffsets = cfg.vindex.map { it.size }
      .fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()
    val goLen = grammarOffsets.size.also { println("goLen: $it") }
    return "constant int vilen=$grLen; constant int volen=$goLen;" +
        grammarFlattened.joinToString(",", "constant int constant vidx[]={", "};") +
        grammarOffsets.joinToString(",", "constant int vidx_offsets[]={", "};")
  }

  private val nativeBridge: NativeBridge =
    if (System.getProperty("os.name").startsWith("Mac")) getMetalBridge() else TODO()

  fun setupPython() = pythonStatementCNF.run { nativeBridge.setup(nonterminals.size, bindex[START_SYMBOL], metalSrc) }

  private fun getMetalBridge(): NativeBridge {
    val directory = "src/main/resources/dlls".also { File(it).mkdirs() }
    val dylib = File("$directory/libMetalBridge.dylib")

    if (needsRebuild(dylib, swiftSrc, directory)) buildNative(directory, dylib, swiftSrc)
    return (Native.load(dylib.absolutePath, NativeBridge::class.java) as NativeBridge)
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