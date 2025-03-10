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

//  initiateSerialRepair(pythonCode, cfg).take(100).forEach { println("$it / ${it in cfg.language}") }
//  System.exit(1)

  val numStates = levFSA.numStates
  val numNonterminals = cfg.nonterminals.size

  val allPairs = levFSA.allPairs2
    .mapIndexed { p, l1 -> l1.mapIndexed { q, l2 -> l2.filter { it in (p + 1)..<q } } }
//    .also { s->
//      s.forEachIndexed { i, it -> it.forEachIndexed { j, it -> println("$i:$j -> $it") } }
//    }
    .flatten()
  val allFSAPairsFlattened = allPairs.flatten().toIntArray()
  val allFSAPairsOffsets = allPairs.map { it.size }
    .fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()

  println("\nLinked GPU in ${measureTimeMillis { GPUBridge.setupPython() }}ms\n")
  val clock = TimeSource.Monotonic.markNow()

  measureTimeMillis {
    fun FSA.byteFormat(cfg: CFG): UShortArray {
      val terminalLists = cfg.nonterminals.map {
        if (it !in cfg.unitNonterminals) emptyList<Σᐩ>()
        else cfg.bimap.UNITS[it]!!
      }
      // 0 and 1 are reserved for (0) no parse exists and (1) parse exists, but an internal nonterminal node
      // Other byte values are used to denote the presence (+) or absence (-) of a leaf terminal
      fun StrPred.predByte(A: Int): UShort = (
        if (arg == "[.*]" || (arg.startsWith("[!=]") && arg.drop(4) !in terminalLists[A])) 65_534 // All possible terminals
        else if (arg.startsWith("[!=]")) (1.shl(16) + (terminalLists[A].indexOf(arg.drop(4)) + 1).shl(1)) // Represent negation using sign bit
        else (terminalLists[A].indexOf(arg) + 1).shl(1)
      ).toUShort() // Otherwise positive sign bit

      return cfg.unitProductions.flatMap { (A, σ) ->
        nominalForm.flattenedTriples.filter { arc -> arc.second(σ) }.map { (q0, sp, q1) ->
          val Aidx = cfg.bindex[A]
          listOf(stateMap[q0]!!, stateMap[q1]!!, Aidx).map { it.toUShort() } + sp.predByte(Aidx)
//          .also { if(it != 65534.toUShort()) println("${sp.arg}/$A/$it/${terminalLists[Aidx]}/${terminalLists[Aidx].indexOf(sp.arg.drop(4)) + 1}") }
        }.flatten()
      }.toUShortArray()
       .also { println("Encoded instance in ${it.size * 2} bytes / ${numStates*numStates*numNonterminals}") }
    }

    val maxSamples = 10_000
    val samples = GPUBridge.cflClosure(
      levFSA.byteFormat(cfg),
      allFSAPairsFlattened, allFSAPairsOffsets,
      levFSA.finalIdxs.toIntArray(), levFSA.finalIdxs.size,
      numStates, maxWordLen, maxSamples
    )

    println("Round trip: ${clock.elapsedNow()}")

    val (valid, invalid) = samples.joinToString(" ") {
      if (it + 0 == 0) "0" // Should only happen at the end of the word
      else if (it - 1 !in cfg.tmLst.indices) "??${it.toInt()}??" // Should never happen
      else if (it - 1 == 0) "" // TODO: why do we need to block NEWLINE?
      else cfg.tmLst[it.toInt() - 1]
    }.split(Regex("( 0)+")) // Runs of zeros delimit individual words
      .map { it.replace(" return", " ") }
      .map { it.replace(" ...", " ") } // TODO: remove all postprocessing
      .filter { it.isNotBlank() }
      .map { it.tokenizeByWhitespace().joinToString(" ") }
      .distinct()
      .partition { "$it NEWLINE" in pythonStatementCNF.language }

    val totalSamples = valid.size + invalid.size
    println("\nValid samples:\n")
    valid.take(10).forEachIndexed { i, it -> println("$i.) ${it.trim()}") }
    println("\nInvalid samples:\n")
    invalid.take(10).forEachIndexed { i, it -> println("$i.) ${it.trim()}") }

    println("...\n\nGPU repairs: ${valid.size} valid / $totalSamples unique / $maxSamples total")
  }.also { println("Total time: ${it}ms\n") }
}

object GPUBridge {
  @Language("c++") val metalSrc = """
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
#include <metal_stdlib>
using namespace metal;

// The following header must be regenerated on a per-CFG basis
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
  const device ushort* dp_in                   [[buffer(0)]],
  device ushort*       dp_out                  [[buffer(1)]],
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
      if (dp_in[dpIndex] & 0x01) return; // The last bit will represent a nonterminal
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

          if (dp_in[idxBM] && dp_in[idxMC]) {
//          if ((dp_in[idxBM] & 0x01 && dp_in[idxMC] & 0x01) || ((1<dp_in[idxBM]) && (1<dp_in[idxMC]))) {
             dp_out[dpIndex] |= 0x01;
             atomic_fetch_add_explicit(&numNonzero, 1, memory_order_relaxed);
             return;
          }
       }
}

kernel void bp_count(
  const device ushort* dp_in                   [[buffer(0)]],
  const device int*    allFSAPairs             [[buffer(1)]],
  const device int*    allFSAPairsOffsets      [[buffer(2)]],
  constant int&        numStates               [[buffer(3)]],
  constant int&        allFSAPairsSize         [[buffer(4)]],
  constant int&        allFSAPairsOffsetsSize  [[buffer(5)]],
  device int*          bpCount                 [[buffer(6)]],
  uint tid [[thread_position_in_grid]]
) {
    int N = numStates;
    int totalCells = (N*(N-1))/2 * numNonterminals;
    if (tid >= totalCells) return;

    int r, c;
    decodeUpperTriangle(tid / numNonterminals, N, r, c);
    int A = tid % numNonterminals, snt = N * numNonterminals, dpIndex = r*snt + c*numNonterminals + A;

    if (!(dp_in[dpIndex] & 0x01)) { bpCount[dpIndex] = 0; return; }

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
        if (dp_in[idxBM] && dp_in[idxMC]) count++;
//        if ((dp_in[idxBM] & 0x01 && dp_in[idxMC] & 0x01) || ((1<dp_in[idxBM]) && (1<dp_in[idxMC]))) count++;
      }

    bpCount[r*snt + c*numNonterminals + A] = count;//max(count, 1);
}

kernel void bp_write(
  const device ushort* dp_in                  [[buffer(0)]],
  const device int*    allFSAPairs            [[buffer(1)]],
  const device int*    allFSAPairsOffsets     [[buffer(2)]],
  constant int&        numStates              [[buffer(3)]],
  constant int&        allFSAPairsSize        [[buffer(4)]],
  constant int&        allFSAPairsOffsetsSize [[buffer(5)]],
  const device int*    bpCount                [[buffer(6)]],
  const device int*    bpOffset               [[buffer(7)]],
  // Suppose each entry in bpStorage is 2 x int.
  device int*          bpStorage              [[buffer(8)]],
  uint tid [[thread_position_in_grid]]
) {
    int N = numStates;
    int totalCells = (N*(N-1))/2 * numNonterminals;

    int r, c;
    decodeUpperTriangle(tid / numNonterminals, N, r, c);
    int A = tid % numNonterminals, snt = N * numNonterminals, dpIndex = r*snt + c*numNonterminals + A;

    if (!(dp_in[dpIndex] & 0x01)) return;

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
        if (dp_in[idxBM] && dp_in[idxMC]) {
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

inline bool alreadyVisited( thread const int* visitedList, int visitedCount, int candidate ) {
    for (int i = 0; i < visitedCount; i++) { if (visitedList[i] == candidate) return true; }
    return false;
}

inline void sampleCellIterative(
  const device ushort*   dp_in,
  const device int*      bpCount,
  const device int*      bpOffset,
  const device int*      bpStorage,
  int                    startDPIdx,
  thread uint&           rngState,
  thread ushort*         localWord,    // output buffer
  thread int&            wordLen,
  const int              maxWordLen
) {
    constexpr int MAX_STACK = 1024; int stack[MAX_STACK]; int top = 0; stack[top++] = startDPIdx;
    int visitedList[MAX_STACK]; int visitedCount = 0; visitedList[visitedCount++] = startDPIdx;

    // While frames left to process, and haven't overflowed localWord
    for (int iter = 0; iter < maxWordLen * 2 && top > 0 && wordLen < maxWordLen - 5; iter++) {
      int dpIdx = stack[--top];
      int expCount = bpCount[dpIdx];
      
      if ((dp_in[dpIdx] >> 1) && !(dp_in[dpIdx] & 0x01)) { // If we are dealing with a leaf node (i.e., a unit nonterminal/terminal)
        int nonterminal = dpIdx % numNonterminals;
        int predicate = dp_in[dpIdx];
        bool isNegativeLiteral = predicate & 0x8000;
        ushort literal = (predicate >> 1) & 0x7FFF;
        int numTms = nt_tm_lens[nonterminal];
        uint ntOffset = offsets[nonterminal];
        if (isNegativeLiteral) {
          ushort possibleTms[100];
          uint tmCount = 0; 
          for (int i = 0; i < min(100, numTms); i++) {
            if (i != literal - 1) possibleTms[tmCount++] = all_tm[ntOffset + i];
          }
          ushort tmChoice = possibleTms[int(lcg_randomRange(rngState, uint(tmCount)))];
          localWord[wordLen++] = tmChoice + 1;
        } else { localWord[wordLen++] = (numTms != 0) ? all_tm[ntOffset + literal - 1] + 1 : 99; } // Must have been a positive literal
      } else if (top + 2 < MAX_STACK) {
        int randIdx, idxBM, idxMC; bool visited = true;
        for (int j = 0; j < 10 && visited; j++) {
          randIdx = bpOffset[dpIdx] + int(lcg_randomRange(rngState, uint(expCount)));
          idxBM = bpStorage[randIdx * 2 + 0];
          idxMC = bpStorage[randIdx * 2 + 1];
          visited = alreadyVisited(visitedList, visitedCount, idxBM) && alreadyVisited(visitedList, visitedCount, idxMC);
        }

        if (!visited) { // Restricts overlapping subspans
          visitedList[visitedCount++] = idxBM; visitedList[visitedCount++] = idxMC;
          stack[top++] = idxMC; stack[top++] = idxBM;
        } 
      }
    }
    
//    for (int i = wordLen; i < maxWordLen; i++) { localWord[wordLen++] = 0; }
}

kernel void sample_words(
  const device ushort*  dp_in           [[buffer(0)]], // parse chart
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
    thread ushort localWord[1024];
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
    allFSAPairs: IntArray, allFSAPairsOffsets: IntArray,
    acceptStates: IntArray, acceptStatesSize: Int,
    numStates: Int, maxWordLen: Int, maxSamples: Int
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
      acceptStatesSize: Int, dpSize: Int, numStates: Int,
      allFSAPairsSize: Int, allFSAPairsOffsetsSize: Int,
      maxWordLen: Int, maxSamples: Int,
    )
  }
  /** Caller: [GPUBridge.cflClosure] */
  @Language("swift") val swiftSrc = """$swiftImports
@_cdecl("cflMultiply") public func cflMultiply(
    _ dp_in_sparse: UnsafePointer<UInt16>?,
    _ dp_out: UnsafeMutablePointer<UInt16>?,
    _ allFSAPairs: UnsafePointer<CInt>?,
    _ allFSAPairsOffsets: UnsafePointer<CInt>?,
    _ acceptStates: UnsafePointer<CInt>?,
    _ acceptStatesSize: CInt, _ dpSize: CInt, _ numStates: CInt,
    _ allFSAPairsSize: CInt, _ allFSAPairsOffsetsSize: CInt,
    _ maxWordLen: CInt, _ maxSamples: CInt
) {
    guard let dp_out, let allFSAPairs, let allFSAPairsOffsets else { return }
    
    var N = Int(numStates), ap = Int(allFSAPairsSize), ao = Int(allFSAPairsOffsetsSize)

    // Stage 1: Construct the dense parse chart
    var bufA = reconstructDPBuffer(
        dp_in: dp_in_sparse,
        dpSize: Int(dpSize),
        numStates: Int(numStates)
    )

    let stride = MemoryLayout<Int32>.stride
    let pairsBuf = device.makeBuffer(bytes: allFSAPairs, length: ap * stride, options: [])!
    let pairsOffBuf = device.makeBuffer(bytes: allFSAPairsOffsets, length: ao * stride, options: [])!
    
    // Stage 2: Seek the binary CFL closure
    bufA = iterateFixpoint(
        bufA: bufA, 
        pairsBuf: pairsBuf, 
        pairsOffBuf: pairsOffBuf,
        numStates: N,
        allPairsSize: ap,
        allPairsOffsetsSize: ao
    )

    // Stage 3: Reconstruct parse forest
    let (bpCountBuf, bpOffsetBuf, bpStorageBuf) = buildBackpointers(
        dp_in: bufA,
        allFSAPairs: pairsBuf,
        allFSAPairsOffsets: pairsOffBuf,
        numStates: N,
        allPairsSize: ap,
        allPairsOffsetsSize: ao
    )

    // Stage 4: Decode the datastructure
    let sampledWordsBuf = sampleWords(
        dp_in: bufA,
        bpCount: bpCountBuf,
        bpOffset: bpOffsetBuf,
        bpStorage: bpStorageBuf,
        maxWordLen: Int(maxWordLen),
        maxSamples: Int(maxSamples),
        numStates: N,
        acceptStatesPt: acceptStates,
        acceptStatesSize: Int(acceptStatesSize)
    )

    memcpy(dp_out, sampledWordsBuf.contents(), Int(maxSamples) * Int(maxWordLen))
}

private func iterateFixpoint(
    bufA: MTLBuffer,
    pairsBuf: MTLBuffer,
    pairsOffBuf: MTLBuffer,
    numStates: Int,
    allPairsSize: Int,
    allPairsOffsetsSize: Int
) -> MTLBuffer {
    var N = numStates, ap = allPairsSize, ao = allPairsOffsetsSize
    let totalCount = N*N*numNonterminals
    let dpSizeBytes = totalCount * MemoryLayout<UInt16>.stride
    let stride = MemoryLayout<Int32>.stride
    let totalPairs = N*(N-1)/2
    let totalThreads = totalPairs * numNonterminals
    let tiling = MTLSizeMake(totalThreads,1,1)

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
       enc.setBytes(&N,           length: stride, index: 4)
       enc.setBytes(&ap,          length: stride, index: 5)
       enc.setBytes(&ao,          length: stride, index: 6)
       enc.setBuffer(numNonzero,  offset: 0,      index: 7)

       enc.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
       enc.endEncoding()
       cmb.commit()
       cmb.waitUntilCompleted()

       let currentValue = numNonzero.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
       if (currentValue == prevValue) {
          print("Fixpoint escape at round: \(r)/\(numStates), total=\(currentValue), size: \(dpSizeBytes)"); fflush(stdout)
          break
       } else { prevValue = currentValue }
    }
    return bufA
}

// takes a flattened array of sparse quadruples and reconstructs a dense matrix
private func reconstructDPBuffer(dp_in: UnsafePointer<UInt16>?, dpSize: Int, numStates: Int) -> MTLBuffer {
    let totalCount = numStates*numStates*numNonterminals
    let dpSizeBytes = totalCount * MemoryLayout<UInt16>.stride
    let rowMultiplier = numStates * numNonterminals  // Precomputed for row index
    let colMultiplier = numNonterminals              // Precomputed for column index

    // Create and initialize bufA
    let bufA = device.makeBuffer(length: dpSizeBytes, options: [])!
    let ptr = bufA.contents().bindMemory(to: UInt16.self, capacity: totalCount)
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
                let s = triples[stride * i + 3]
                let index = r * rowMultiplier + c * colMultiplier + A
                ptr[index] = s
            }
        }
    }

    return bufA
}

private func sampleWords(
    dp_in: MTLBuffer,
    bpCount: MTLBuffer,
    bpOffset: MTLBuffer,
    bpStorage: MTLBuffer,
    maxWordLen: Int,
    maxSamples: Int,
    numStates: Int,
    acceptStatesPt: UnsafePointer<CInt>?,
    acceptStatesSize: Int
) -> MTLBuffer {
    print("Sampling words..."); fflush(stdout)

    let startTime = DispatchTime.now()
    let totalCount = numStates*numStates*numNonterminals
    let dpInPtr = dp_in.contents().bindMemory(to: UInt16.self, capacity: totalCount)

    let acceptStatesSwift: [Int]
    if let acceptStatesPtr = acceptStatesPt {
        acceptStatesSwift = UnsafeBufferPointer(start: acceptStatesPtr, count: acceptStatesSize).map { Int($0) }
    } else { acceptStatesSwift = [] }

    var startIdxs: [Int32] = []
    for q in acceptStatesSwift {
        let dpIndex = q * numNonterminals + startIdx
        if dpInPtr[dpIndex] != 0 { startIdxs.append(Int32(dpIndex)) }
    }
    print("Number of valid startIndices: \(startIdxs.count)/\(acceptStatesSwift.count)"); fflush(stdout)

    let stride = MemoryLayout<Int32>.stride
    let startIndices = device.makeBuffer(bytes: startIdxs, length: startIdxs.count * stride, options: [])!

    let totSamples = maxSamples, wordLen = maxWordLen
    var seeds = [UInt32](repeating: 0, count: totSamples)
    for i in 0..<totSamples { seeds[i] = UInt32.random(in: 0..<UInt32.max) }
    let seedsBuf = device.makeBuffer(bytes: seeds, length: totSamples * stride, options: [])!

    // We'll store each sampled word in a row of length `maxWordLen`.
    let sampledWordsSize = maxSamples * maxWordLen
    let sampledWordsBuf = device.makeBuffer(length: sampledWordsSize, options: [])!
    memset(sampledWordsBuf.contents(), 0, sampledWordsSize) // Something strange happens here

    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(psoSmpWord)
    encoder.setBuffer(dp_in,           offset: 0, index: 0)
    encoder.setBuffer(bpCount,         offset: 0, index: 1)
    encoder.setBuffer(bpOffset,        offset: 0, index: 2)
    encoder.setBuffer(bpStorage,       offset: 0, index: 3)
    encoder.setBuffer(startIndices,    offset: 0, index: 4)
    encoder.setBuffer(seedsBuf,        offset: 0, index: 5)
    encoder.setBuffer(sampledWordsBuf, offset: 0, index: 6)
    var nsStart = startIdxs.count, nStates = numStates, mwl = maxWordLen
    encoder.setBytes(&nsStart, length: MemoryLayout<Int32>.size, index: 7)
    encoder.setBytes(&nStates, length: MemoryLayout<Int32>.size, index: 8)
    encoder.setBytes(&mwl, length: MemoryLayout<Int32>.size, index: 9)
    encoder.dispatchThreads(MTLSizeMake(maxSamples, 1, 1), threadsPerThreadgroup: MTLSizeMake(1,1,1))

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let sampSize = totSamples * wordLen

    let ptr = sampledWordsBuf.contents().bindMemory(to: UInt8.self, capacity: sampSize)
    let buffer = UnsafeBufferPointer(start: ptr, count: sampSize)

    for sIdx in 0..<min(5, totSamples) {
        let start = sIdx * wordLen
        let wordSlice = buffer[start..<(start + wordLen)]
        print("Sample \(sIdx): \(Array(wordSlice))"); fflush(stdout)
    }

    let psEnd = DispatchTime.now()
    let psTimeMs = Double(psEnd.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000
    print("...\nSampled words in \(psTimeMs) ms"); fflush(stdout)

    return sampledWordsBuf
}

// Constructs the backpointer references for decoding
private func buildBackpointers(
  dp_in: MTLBuffer, // size = N*N*numNonterminals (bytes)
  allFSAPairs: MTLBuffer,
  allFSAPairsOffsets: MTLBuffer,
  numStates: Int,
  allPairsSize: Int,
  allPairsOffsetsSize: Int
) -> (MTLBuffer, MTLBuffer, MTLBuffer) {
    var N = numStates, ap = allPairsSize, ao = allPairsOffsetsSize
    let start = DispatchTime.now()
    // 1) Allocate bpCount
    let countSize = N * N * numNonterminals * MemoryLayout<Int32>.stride
    let bpCountBuf = device.makeBuffer(length: countSize, options: [])!

    // 2) Launch bp_count kernel
    let totalCells = (N*(N-1))/2 * numNonterminals
    let tiling = MTLSizeMake(totalCells,1,1)

    var commandBuffer = queue.makeCommandBuffer()!
    var encoder = commandBuffer.makeComputeCommandEncoder()!
    let stride = MemoryLayout<Int32>.stride
    encoder.setComputePipelineState(psoBpCount)  // create psoBpCount from "bp_count" function
    encoder.setBuffer(dp_in,               offset: 0,      index: 0)
    encoder.setBuffer(allFSAPairs,         offset: 0,      index: 1)
    encoder.setBuffer(allFSAPairsOffsets,  offset: 0,      index: 2)
    encoder.setBytes(&N,                   length: stride, index: 3)
    encoder.setBytes(&ap,                  length: stride, index: 4)
    encoder.setBytes(&ao,                  length: stride, index: 5)
    encoder.setBuffer(bpCountBuf,          offset: 0,      index: 6)

    encoder.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

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

    // 5) Allocate bpStorage - each expansion is 2 ints.
    let bpStorageSize = Int(totalExpansions) * 2 * MemoryLayout<Int32>.stride
    let bpStorageBuf = device.makeBuffer(length: bpStorageSize, options: [])!

    // 6) Launch bp_write kernel
    commandBuffer = queue.makeCommandBuffer()!
    encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(psoBpWrite)
    encoder.setBuffer(dp_in,              offset: 0,      index: 0)
    encoder.setBuffer(allFSAPairs,        offset: 0,      index: 1)
    encoder.setBuffer(allFSAPairsOffsets, offset: 0,      index: 2)
    encoder.setBytes(&N,                  length: stride, index: 3)
    encoder.setBytes(&ap,                 length: stride, index: 4)
    encoder.setBytes(&ao,                 length: stride, index: 5)
    encoder.setBuffer(bpCountBuf,         offset: 0,      index: 6)
    encoder.setBuffer(bpOffsetBuf,        offset: 0,      index: 7)
    encoder.setBuffer(bpStorageBuf,       offset: 0,      index: 8)

    encoder.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let end = DispatchTime.now()
    let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
    let timeMs = Double(nanoTime) / 1_000_000
    print("Built backpointers in \(timeMs) ms"); fflush(stdout)

    return (bpCountBuf, bpOffsetBuf, bpStorageBuf)
}

// TODO: GPU parallel prefix sum?
private func parallelPrefixSumCPU(countBuf: MTLBuffer, offsetBuf: MTLBuffer, totalCells: Int) {
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

    return """constant uint16_t all_tm[] = {${allTerminals.joinToString(",")}};
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