@file:OptIn(ExperimentalUnsignedTypes::class)

package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.*
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.*
import com.sun.jna.*
import org.intellij.lang.annotations.Language
import java.io.File
import java.util.*
import kotlin.system.measureTimeMillis
import kotlin.time.*

// Tested on MacOS 15.3.2 (24D81) / M4 Max / 128 GB

// Expected output:
//     Encoded instance in 403048 bytes / 12665988 / 72.052250ms
//     Bytes: 201524
//     Constructed 222x222x257 parse chart in 2.787083 ms
//     Fixpoint escape at round: 20/222, total=686247, size: 25331976
//     Prefix sum completed in 656.899084 ms
//     Built backpointers in 682.805792 ms
//     Sampled words in 17.079083 ms
//     Round trip: 899.618916ms
//
//     GPU repairs: 237 valid / 237 unique / 1000 total

/*
./gradlew generateDylib
 */

fun main() {
  val cfg = pythonStatementCNFAllProds
  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE"
    .tokenizeByWhitespace()
  val radius = 5
  val levFSA = makeLevFSA(pythonCode, radius)
  val maxWordLen = pythonCode.size + radius + 10

  val numStates = levFSA.numStates
  val numNonterminals = cfg.nonterminals.size

  fun IntArray.parallelPrefixSizes(): IntArray =
    IntArray(size + 1).also { prefix ->
      System.arraycopy(this, 0, prefix, 1, size)
      Arrays.parallelPrefix(prefix) { a, b -> a + b }
    }

  fun List<List<List<Int>>>.parallelPrefixScan(): Pair<IntArray, IntArray> =
    measureTimedValue {
      mapIndexed { p, outer -> outer.mapIndexed { q, inner -> inner.filter { it in (p + 1)..<q } } }.flatten()
        .let { it.flatten().toIntArray() to it.map { it.size }.toIntArray().parallelPrefixSizes() }
    }.also { println("Completed prefix scan in: ${it.duration}") }.value

  val (allFSAPairsFlattened, allFSAPairsOffsets) = levFSA.midpoints.parallelPrefixScan()

  println("\nLinked GPU in ${measureTimeMillis { GPUBridge.setupCFG(cfg) }}ms\n")
  val clock = TimeSource.Monotonic.markNow()

  measureTimeMillis {
    fun FSA.byteFormat(cfg: CFG): UShortArray {
      val clock =  TimeSource.Monotonic.markNow()
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
          // row, col, Aidx, terminal encoding
          val pb = sp.predByte(Aidx)
          listOf(stateMap[q0]!!, stateMap[q1]!!, Aidx).map { it.toUShort() } + pb
        }
      }.distinct().flatten().toUShortArray()
       .also { println("Encoded instance in ${it.size * 2} bytes / ${numStates*numStates*numNonterminals} / ${clock.elapsedNow()}") }
    }

    val maxSamples = 1_000

    // This should match the CPU reference implementation exactly, but tends to produce many invalid samples
    val samples = GPUBridge.cflClosure(
      levFSA.byteFormat(cfg).also { println("Bytes: ${it.size}") },
      allFSAPairsFlattened, allFSAPairsOffsets,
      levFSA.finalIdxs.toIntArray(), levFSA.finalIdxs.size,
      numStates, maxWordLen, maxSamples
    )

    println("Round trip: ${clock.elapsedNow()}")

    val (valid, invalid) = samples.joinToString(" ") {
      if (it + 0 == 0) "0" // Should only happen at the end of the word
      else if (it - 1 !in cfg.tmLst.indices) "??${it.toInt()}??" // Should never happen
      else cfg.tmLst[it.toInt() - 1]
    }.split(Regex("( 0)+")) // Runs of zeros delimit individual words
      .filter { it.isNotBlank() }
      .map { it.tokenizeByWhitespace().joinToString(" ") }
      .distinct()
//      .partition { true }
      .partition { it in cfg.language && levFSA.recognizes(it) }

    val totalSamples = valid.size + invalid.size
    println("\nValid samples:\n")
    valid.take(10).forEachIndexed { i, it -> println("$i.) ${it.trim()}") }
    println("\nInvalid samples:\n")
    invalid.take(10).forEachIndexed { i, it -> println("$i.) ${it.trim()}") }

    println("...\n\nGPU repairs: ${valid.size} valid / $totalSamples unique / $maxSamples total")
  }.also { println("Total time: ${it}ms\n") }
}

object GPUBridge {
  @Language("c++") fun metalSrc(grammarHeader: String) = """
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
#include <metal_stdlib>
using namespace metal;

// The following header is regenerated on a per-CFG basis
$grammarHeader

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

#define PREAMBLE(tid, numStates, r, c, A, dpIdx, snt, startGC, endGC, aoi, pairOffset, pairOffsetNext) \
    int totalCells = (numStates * (numStates - 1) / 2) * numNonterminals; \
    if (tid >= totalCells) return; \
    int r, c; \
    decodeUpperTriangle(tid / numNonterminals, numStates, r, c); \
    int A = tid % numNonterminals; \
    int snt = numStates * numNonterminals; \
    int dpIdx = r * snt + c * numNonterminals + A; \
    int startGC = vidx_offsets[A]; \
    int endGC = (A + 1 < volen) ? vidx_offsets[A + 1] : vilen; \
    int aoi = r * numStates + c + 1; \
    int pairOffset = allFSAPairsOffsets[aoi - 1]; \
    int pairOffsetNext = (aoi < allFSAPairsOffsetsSize) ? allFSAPairsOffsets[aoi] : allFSAPairsSize;

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
    PREAMBLE(tid, numStates, r, c, A, dpIdx, snt, startGC, endGC, aoi, pairOffset, pairOffsetNext)

    if (dp_in[dpIdx]) {
      dp_out[dpIdx] = dp_in[dpIdx];
      atomic_fetch_add_explicit(&numNonzero, 1, memory_order_relaxed);
      if (dp_in[dpIdx] & 0x01) return; // The last bit will represent a nonterminal
    }

    for (int pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++)
       for (int g = startGC, mdpt = allFSAPairs[pairIdx]; g < endGC; g += 2) {
          int B = vidx[g]; int C = vidx[g+1];
          
          int idxBM = r * snt + mdpt * numNonterminals + B;
          int idxMC = mdpt * snt + c * numNonterminals + C;

          if (dp_in[idxBM] && dp_in[idxMC]) {
             dp_out[dpIdx] |= 0x01;
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
    PREAMBLE(tid, numStates, r, c, A, dpIdx, snt, startGC, endGC, aoi, pairOffset, pairOffsetNext)

    if (!(dp_in[dpIdx] & 0x01)) { bpCount[dpIdx] = 0; return; }

    int count = 0;
    for (int pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++)
      for (int g = startGC, mdpt = allFSAPairs[pairIdx]; g < endGC; g += 2) {
        int B = vidx[g], C = vidx[g+1];

        int idxBM = r*snt + mdpt*numNonterminals + B;
        int idxMC = mdpt*snt + c*numNonterminals + C;
        if (dp_in[idxBM] && dp_in[idxMC]) count++;
      }

    bpCount[dpIdx] = count;
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
    PREAMBLE(tid, numStates, r, c, A, dpIdx, snt, startGC, endGC, aoi, pairOffset, pairOffsetNext)

    if (!(dp_in[dpIdx] & 0x01)) return;

    int outPos = bpOffset[dpIdx];

    // Exactly like bp_count, but now we store expansions
    for (int pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++)
      for (int poff = startGC, mdpt = allFSAPairs[pairIdx]; poff < endGC; poff += 2) {
        int B = vidx[poff], C = vidx[poff+1];

        int idxBM = r*snt + mdpt*numNonterminals + B;
        int idxMC = mdpt*snt + c*numNonterminals + C;
        if (dp_in[idxBM] && dp_in[idxMC]) {
          bpStorage[outPos*2 + 0] = idxBM;
          bpStorage[outPos*2 + 1] = idxMC;
          outPos++;
        }
      }
}

inline uint lcg_random(thread uint& state) { return 1664525u * state + 1013904223u; }
inline uint lcg_randomRange(thread uint& state, uint range) { return lcg_random(state) % range; }

inline void sampleTopDown(
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

    // While frames left to process, and haven't overflowed localWord
    for (int iter = 0; iter < maxWordLen * 98 && top > 0 && wordLen < maxWordLen - 5; iter++) {
      int dpIdx = abs(stack[--top]);
      int expCount = bpCount[dpIdx];
      
      // If we are dealing with a leaf node (i.e., a unit nonterminal/terminal)
      if ((dp_in[dpIdx] >> 1) && (!(dp_in[dpIdx] & 0x01) || (int(lcg_randomRange(rngState, uint(expCount))) % 2 == 0))) {
        int nonterminal = dpIdx % numNonterminals;
        ushort predicate = dp_in[dpIdx];
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
        int randIdx = bpOffset[dpIdx] + int(lcg_randomRange(rngState, uint(expCount)));
        int idxBM = bpStorage[2 * randIdx + 0];
        int idxMC = bpStorage[2 * randIdx + 1];

        stack[top++] = idxMC; stack[top++] = idxBM;
      }
    }
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
    sampleTopDown(dp_in, bpCount, bpOffset, bpStorage, startIdx, rngState, localWord, wordLen, maxWordLen);

    for (int i=0; i<maxWordLen; i++) sampledWords[int(tid)*maxWordLen + i] = localWord[i];
}

kernel void prefix_sum_p1(
    device const int * inBuf  [[buffer(0)]],
    device       int * outBuf [[buffer(1)]],
    device       int * blkSum [[buffer(2)]],
    constant    uint &N       [[buffer(3)]],
                uint gid      [[thread_position_in_grid]],
                uint grpId    [[threadgroup_position_in_grid]],
                uint lid      [[thread_position_in_threadgroup]],
                uint tpg      [[threads_per_threadgroup]]
) {
    threadgroup int tile[1024];
    uint base = grpId * tpg, idx = base + lid;

    int v = (idx < N) ? inBuf[idx] : 0;
    tile[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < tpg; offset <<= 1) {
        int n = (lid >= offset) ? tile[lid - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile[lid] += n;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int inclusive = tile[lid];
    tile[lid] = inclusive - v;

    if ((idx + 1 == N) || (lid == tpg - 1)) { blkSum[grpId] = inclusive; }
    if (idx < N) { outBuf[idx] = tile[lid]; }
}

kernel void prefix_sum_p2(
    device int *       outBuf [[buffer(0)]],
    device const int * blkSum [[buffer(1)]],
    constant uint &N         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]],
    uint grpId               [[threadgroup_position_in_grid]],
    uint lid                 [[thread_position_in_threadgroup]],
    uint tpg                 [[threads_per_threadgroup]]
) {
    if (grpId == 0) return;
    int offset = blkSum[grpId - 1];
    uint idx = grpId * tpg + lid;
    if (idx < N) { outBuf[idx] += offset; }
}
"""

  val swiftImports = "import Foundation\nimport Metal\nimport Dispatch"

  @JvmStatic fun cflClosure(
/**[NativeBridge.cflMultiply] */
    dpIn: UShortArray,
    mdpts: IntArray, mdptsOffsets: IntArray,
    acceptStates: IntArray, acceptStatesSize: Int,
    numStates: Int, maxWordLen: Int, maxSamples: Int
  ): ByteArray =
    Memory(maxWordLen * maxSamples.toLong()).also { outMem ->
      nativeBridge.cflMultiply(
        dp_in = Memory(2L * dpIn.size).apply { write(0, dpIn.toShortArray(), 0, dpIn.size) },
        dp_out = outMem,
        allFSAPairs = Memory(4L * mdpts.size).apply { write(0, mdpts, 0, mdpts.size) },
        allFSAPairsOffsets = Memory(4L * mdptsOffsets.size).apply { write(0, mdptsOffsets, 0, mdptsOffsets.size) },
        acceptStates =  Memory(4L * acceptStatesSize).apply { write(0, acceptStates, 0, acceptStates.size) },
        acceptStatesSize = acceptStatesSize,
        dpSize = dpIn.size,
        numStates = numStates,
        allFSAPairsSize = mdpts.size,
        allFSAPairsOffsetsSize = mdptsOffsets.size,
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
    
    let N = Int(numStates), ap = Int(allFSAPairsSize), ao = Int(allFSAPairsOffsetsSize)

    // Stage 1: Construct the dense parse chart
    var bufA = reconstructDPBuffer(dp_in: dp_in_sparse, dpSize: Int(dpSize), numStates: N)

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
        accsPt: acceptStates,
        accsSize: Int(acceptStatesSize)
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
    let totalCount = N * N * numNonterminals
    let dpSizeBytes = totalCount * MemoryLayout<UInt16>.stride
    let stride = MemoryLayout<Int32>.stride
    let totalPairs = N * (N - 1) / 2
    let totalThreads = totalPairs * numNonterminals

    let numNonzero = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    var prevValue: UInt32 = 0

    for r in 0..<numStates {
       var zero: UInt32 = 0
       memcpy(numNonzero.contents(), &zero, MemoryLayout<UInt32>.size)

       runKernel(cfl_mul_upper, totalThreads) { enc in
          enc.setBuffer(bufA,        offset: 0,      index: 0)
          enc.setBuffer(bufA,        offset: 0,      index: 1)
          enc.setBuffer(pairsBuf,    offset: 0,      index: 2)
          enc.setBuffer(pairsOffBuf, offset: 0,      index: 3)
          enc.setBytes(&N,           length: stride, index: 4)
          enc.setBytes(&ap,          length: stride, index: 5)
          enc.setBytes(&ao,          length: stride, index: 6)
          enc.setBuffer(numNonzero,  offset: 0,      index: 7)
       }

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
    let startTime = DispatchTime.now()
    let totalCount = numStates*numStates*numNonterminals
    let dpSizeBytes = totalCount * MemoryLayout<UInt16>.stride
    let rowMultiplier = numStates * numNonterminals  // Precomputed for row index
    let colMultiplier = numNonterminals              // Precomputed for column index

    // Create and initialize bufA
    let bufA = device.makeBuffer(length: dpSizeBytes, options: [])!
    let ptr = bufA.contents().bindMemory(to: UInt16.self, capacity: totalCount)
    
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
                ptr[r * rowMultiplier + c * colMultiplier + A] = triples[stride * i + 3]
            }
        }
    }

    let psTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000
    print("Constructed \(numStates)x\(numStates)x\(numNonterminals) parse chart in \(psTimeMs) ms"); fflush(stdout)
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
    accsPt: UnsafePointer<CInt>?,
    accsSize: Int
) -> MTLBuffer {
    let t0 = DispatchTime.now()
    let totalCount = numStates * numStates * numNonterminals
    let dpPtr = dp_in.contents().bindMemory(to: UInt16.self, capacity: totalCount)

    let acceptStates = accsPt.map { Array(UnsafeBufferPointer(start: $0, count: accsSize)).map(Int.init) } ?? []
    var startIdxs = [Int32]()
    for q in acceptStates {
        let dpIndex = q * numNonterminals + startIdx
        if dpPtr[dpIndex] != 0 { startIdxs.append(Int32(dpIndex)) }
    }

    let i32stride = MemoryLayout<UInt32>.stride
    let seeds = (0..<maxSamples).map { _ in UInt32.random(in: .min ..< .max) }
    let seedsBuf = device.makeBuffer(bytes: seeds, length: maxSamples * i32stride, options: [])!
    let startBuf = device.makeBuffer(bytes: startIdxs, length: startIdxs.count * i32stride, options: [])!

    // We'll store each sampled word in a row of length `maxWordLen`.
    let resultSize = maxSamples * maxWordLen
    let outBuf = device.makeBuffer(length: resultSize, options: [])!

    runKernel(sample_words, maxSamples) { enc in
        enc.setBuffer(dp_in,       offset: 0, index: 0)
        enc.setBuffer(bpCount,     offset: 0, index: 1)
        enc.setBuffer(bpOffset,    offset: 0, index: 2)
        enc.setBuffer(bpStorage,   offset: 0, index: 3)
        enc.setBuffer(startBuf,    offset: 0, index: 4)
        enc.setBuffer(seedsBuf,    offset: 0, index: 5)
        enc.setBuffer(outBuf,      offset: 0, index: 6)
        var si = Int32(startIdxs.count), n = Int32(numStates), wl = Int32(maxWordLen)
        enc.setBytes(&si,          length: 4, index: 7)
        enc.setBytes(&n,           length: 4, index: 8)
        enc.setBytes(&wl,          length: 4, index: 9)
    }

    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
    print("Sampled words in \(ms) ms"); fflush(stdout)
    return outBuf
}

private func buildBackpointers(
  dp_in: MTLBuffer,
  allFSAPairs: MTLBuffer,
  allFSAPairsOffsets: MTLBuffer,
  numStates: Int,
  allPairsSize: Int,
  allPairsOffsetsSize: Int
) -> (MTLBuffer, MTLBuffer, MTLBuffer) {
    let t0 = DispatchTime.now()

    var N = numStates, ap = allPairsSize, ao = allPairsOffsetsSize
    let stride     = MemoryLayout<Int32>.stride
    let totalThreads = (N * (N - 1) / 2) * numNonterminals
    let totalCells = N * N * numNonterminals
    let countSize  = N * N * numNonterminals * stride

    // 1) Allocate bpCount and run "bp_count" kernel
    let bpCountBuf = device.makeBuffer(length: countSize, options: [])!

    runKernel(bp_count, totalThreads) { enc in
      enc.setBuffer(dp_in,               offset: 0,      index: 0)
      enc.setBuffer(allFSAPairs,         offset: 0,      index: 1)
      enc.setBuffer(allFSAPairsOffsets,  offset: 0,      index: 2)
      enc.setBytes(&N,                   length: stride, index: 3)
      enc.setBytes(&ap,                  length: stride, index: 4)
      enc.setBytes(&ao,                  length: stride, index: 5)
      enc.setBuffer(bpCountBuf,          offset: 0,      index: 6)
    }

    // 2) Prefix sum of bpCount → bpOffset
    let bpOffsetBuf = device.makeBuffer(length: countSize, options: [])!
    parallelPrefixSumGPU(countBuf: bpCountBuf, offsetBuf: bpOffsetBuf, totalCells: totalCells)

    // 3) Find total expansions from last offset + last count
    let offPtr  = bpOffsetBuf.contents().bindMemory(to: Int32.self, capacity: totalCells)

    let cntPtr  = bpCountBuf.contents().bindMemory(to: Int32.self,  capacity: totalCells)
    let totalExpansions = offPtr[totalCells - 1] + cntPtr[totalCells - 1]

    // 4) Allocate bpStorage and run "bp_write" kernel
    let bpStorageBuf = device.makeBuffer(length: Int(totalExpansions) * 2 * stride, options: [])!

    runKernel(bp_write, totalThreads) { enc in
      enc.setBuffer(dp_in,              offset: 0,      index: 0)
      enc.setBuffer(allFSAPairs,        offset: 0,      index: 1)
      enc.setBuffer(allFSAPairsOffsets, offset: 0,      index: 2)
      enc.setBytes(&N,                  length: stride, index: 3)
      enc.setBytes(&ap,                 length: stride, index: 4)
      enc.setBytes(&ao,                 length: stride, index: 5)
      enc.setBuffer(bpCountBuf,         offset: 0,      index: 6)
      enc.setBuffer(bpOffsetBuf,        offset: 0,      index: 7)
      enc.setBuffer(bpStorageBuf,       offset: 0,      index: 8)
    }

    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
    print("Built backpointers in \(ms) ms"); fflush(stdout)
    return (bpCountBuf, bpOffsetBuf, bpStorageBuf)
}

func parallelPrefixSumGPU(countBuf: MTLBuffer, offsetBuf: MTLBuffer, totalCells: Int) {
    let t0 = DispatchTime.now()
    guard totalCells > 1 else { return }
    let tpg = 256
    let numGroups = (totalCells + tpg - 1) / tpg
    let sumsBuf = device.makeBuffer(length: numGroups * MemoryLayout<Int32>.stride, options: [])!
    var N = UInt32(totalCells)

    runKernel(prefix_sum_p1, numGroups * tpg, tpg) { enc in
        enc.setBuffer(countBuf,  offset: 0, index: 0)
        enc.setBuffer(offsetBuf, offset: 0, index: 1)
        enc.setBuffer(sumsBuf,   offset: 0, index: 2)
        enc.setBytes(&N, length: 4, index: 3)
    }

    let ptr = sumsBuf.contents().bindMemory(to: Int32.self, capacity: numGroups)
    for i in 1..<numGroups { ptr[i] += ptr[i - 1] }

    runKernel(prefix_sum_p2, numGroups * tpg, tpg) { enc in
        enc.setBuffer(offsetBuf, offset: 0, index: 0)
        enc.setBuffer(sumsBuf,   offset: 0, index: 1)
        enc.setBytes(&N, length: 4, index: 2)
    }

    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
    print("GPU prefix sum in \(ms) ms"); fflush(stdout)
}

func runKernel(_ pipelineState: MTLComputePipelineState, _ numThreads: Int, _ theadsPerGrp: Int = 1, body: (MTLComputeCommandEncoder) -> Void) {
    let commandBuffer = queue.makeCommandBuffer()!, enc = commandBuffer.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipelineState)
    body(enc)
    enc.dispatchThreads(
        MTLSize(width: numThreads, height: 1, depth: 1), 
        threadsPerThreadgroup: MTLSize(width: theadsPerGrp, height: 1, depth: 1)
    )
    enc.endEncoding(); commandBuffer.commit(); commandBuffer.waitUntilCompleted()
}

private var device: MTLDevice!, queue: MTLCommandQueue!, numNonterminals: Int = 0, startIdx: Int = 0,
            cfl_mul_upper: MTLComputePipelineState!,
            bp_count: MTLComputePipelineState!, 
            bp_write: MTLComputePipelineState!,
            sample_words: MTLComputePipelineState!,
            prefix_sum_p1: MTLComputePipelineState!,
            prefix_sum_p2: MTLComputePipelineState!

@_cdecl("setup") public func setup(_ nnt: CInt, startIndex: CInt, _ str: UnsafePointer<CChar>?) {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!

    let metalSrc = str != nil ? String(cString: str!) : ""
    numNonterminals = Int(nnt)
    startIdx = Int(startIndex)
    let library   = try! device.makeLibrary(source: metalSrc, options: nil)
    cfl_mul_upper = try! device.makeComputePipelineState(function: library.makeFunction(name: "cfl_mul_upper")!)
    bp_count      = try! device.makeComputePipelineState(function: library.makeFunction(name: "bp_count")!)
    bp_write      = try! device.makeComputePipelineState(function: library.makeFunction(name: "bp_write")!)
    sample_words  = try! device.makeComputePipelineState(function: library.makeFunction(name: "sample_words")!)
    prefix_sum_p1 = try! device.makeComputePipelineState(function: library.makeFunction(name: "prefix_sum_p1")!)
    prefix_sum_p2 = try! device.makeComputePipelineState(function: library.makeFunction(name: "prefix_sum_p2")!)
}"""

  private fun grammarHeader(cfg: CFG): String = (genNTTerminalMapForC(cfg) + genFlattenedBinaryProds(cfg))

  private fun genNTTerminalMapForC(cfg: CFG): String {
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

  private fun genFlattenedBinaryProds(cfg: CFG): String {
    val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
    val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()
    return "constant int vilen=${grammarFlattened.size}; constant int volen=${grammarOffsets.size};" +
        grammarFlattened.joinToString(",", "constant int constant vidx[]={", "};") +
        grammarOffsets.joinToString(",", "constant int vidx_offsets[]={", "};")
  }

  private val nativeBridge: NativeBridge = if (System.getProperty("os.name").startsWith("Mac")) metalBridge() else TODO()

  fun setupCFG(cfg: CFG) = nativeBridge.setup(cfg.nonterminals.size, cfg.bindex[START_SYMBOL], metalSrc(grammarHeader(cfg)))

  private fun metalBridge(): NativeBridge {
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