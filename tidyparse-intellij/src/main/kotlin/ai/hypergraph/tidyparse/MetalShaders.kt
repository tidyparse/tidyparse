@file:OptIn(ExperimentalUnsignedTypes::class)

package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.automata.StrPred
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.repair.pythonStatementCNF
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.filter
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
    fun FSA.byteFormat(cfg: CFG): UShortArray {
      val tmMap = cfg.tmMap
      fun StrPred.predByte(): UShort =
        (if (arg == "[.*]") 127 // Maximum value means all possible terminals
        else if (arg.startsWith("[!=]")) -(tmMap[arg.drop(4)] ?: 126) - 1 // Represent negation using negative sign bit
        else tmMap[arg]!! + 1).toUShort() // Otherwise positive sign bit

      return cfg.unitProductions.flatMap { (A, σ) ->
        nominalForm.flattenedTriples.filter { arc -> arc.π2(σ) }.map { (q0, sp, q1) ->
          listOf(stateMap[q0]!!, stateMap[q1]!!, cfg.bindex[A], 127).map { it.toUShort() } //+ 127.toUShort()//sp.predByte()
        }.flatten()
      }.toUShortArray()
       .also { println("Encoded instance in ${it.size * 2} bytes / ${numStates*numStates*numNonterminals}") }
    }

    val outGPU = GPUBridge.cflClosure(
      levFSA.byteFormat(cfg),
      grammarFlattened,
      grammarOffsets,
      allFSAPairsFlattened,
      allFSAPairsOffsets,
      numStates, numNonterminals,
      grLen, goLen, apLen, aoLen
    ).also { resultGPU = it.map { it > 0.toByte() } }

    println("GPU size: " + outGPU.size)
    println("GPU sum: " + resultGPU.sumBy { if(it) 1 else 0 })
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
  device atomic_uint&  numNonzero              [[buffer(12)]],
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
        atomic_fetch_add_explicit(&numNonzero, 1, memory_order_relaxed);
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
                atomic_fetch_add_explicit(&numNonzero, 1, memory_order_relaxed);
                return;
            }
        }
    }
}

kernel void bp_count(
    const device uchar* dp_in                   [[buffer(0)]],
    const device int*   grammar                 [[buffer(1)]],
    const device int*   grammarOffsets          [[buffer(2)]],
    const device int*   allFSAPairs             [[buffer(3)]],
    const device int*   allFSAPairsOffsets      [[buffer(4)]],
    constant int&       numStates               [[buffer(5)]],
    constant int&       numNonterminals         [[buffer(6)]],
    constant int&       grammarSize             [[buffer(7)]],
    constant int&       grammarOffsetsSize      [[buffer(8)]],
    constant int&       allFSAPairsSize         [[buffer(9)]],
    constant int&       allFSAPairsOffsetsSize  [[buffer(10)]],
    device int*         bpCount                 [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    int N = numStates;
    int totalPairs = (N*(N-1))/2;
    int totalCells = totalPairs * numNonterminals;
    if (tid >= totalCells) return;

    // Decode A and (r,c)
    int A = tid % numNonterminals;
    int pairIndex = tid / numNonterminals;

    // decodeUpperTriangle or your usual method:
    int r, c;
    decodeUpperTriangle(pairIndex, N, r, c);

    // If dp_in[r,c,A] == 0, then no expansions. 
    // But be careful: dp_in might only store presence/absence for A. 
    // Even if dp_in[r,c,A] is 1, we want to see how many expansions it *can* have.
    // If dp_in[r,c,A] == 0, skip quickly:
    if (dp_in[r*(N*numNonterminals) + c*numNonterminals + A] == 0) {
        bpCount[r*(N*numNonterminals) + c*numNonterminals + A] = 0;
        return;
    }

    // Grammar offsets for A
    int startGC = grammarOffsets[A];
    int endGC   = (A+1 < grammarOffsetsSize) ? grammarOffsets[A+1] : grammarSize;

    // All possible midpoints
    int pairOffset = allFSAPairsOffsets[r * N + c];
    int pairOffsetNext = ( (r * N + c) + 1 < allFSAPairsOffsetsSize )
                       ? allFSAPairsOffsets[r * N + c + 1]
                       : allFSAPairsSize;

    int count = 0;
    // loop over midpoints
    for (int idx = pairOffset; idx < pairOffsetNext; idx++) {
        int m = allFSAPairs[idx];

        // for each A -> B, C
        for (int g = startGC; g < endGC; g += 2) {
            int B = grammar[g];
            int C = grammar[g+1];

            int idxBM = r*(N*numNonterminals) + m*numNonterminals + B;
            int idxMC = m*(N*numNonterminals) + c*numNonterminals + C;
            // If (r,m,B) and (m,c,C) are both present:
            if (dp_in[idxBM] != 0 && dp_in[idxMC] != 0) {
                count++;
            }
        }
    }

    // Write out the total count for this cell
    bpCount[r*(N*numNonterminals) + c*numNonterminals + A] = count;
}

kernel void bp_write(
    const device uchar* dp_in                  [[buffer(0)]],
    const device int*   grammar                [[buffer(1)]],
    const device int*   grammarOffsets         [[buffer(2)]],
    const device int*   allFSAPairs            [[buffer(3)]],
    const device int*   allFSAPairsOffsets     [[buffer(4)]],
    constant int&       numStates              [[buffer(5)]],
    constant int&       numNonterminals        [[buffer(6)]],
    constant int&       grammarSize            [[buffer(7)]],
    constant int&       grammarOffsetsSize     [[buffer(8)]],
    constant int&       allFSAPairsSize        [[buffer(9)]],
    constant int&       allFSAPairsOffsetsSize [[buffer(10)]],
    const device int*   bpCount                [[buffer(11)]],
    const device int*   bpOffset               [[buffer(12)]],
    // Suppose each entry in bpStorage is 3 x int. 
    // Or you might define a struct for them. We'll assume int for B, M, C in a flat array
    device int*         bpStorage              [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    int N = numStates;
    int totalPairs = (N*(N-1))/2;
    int totalCells = totalPairs * numNonterminals;
    if (tid >= totalCells) return;

    int A = tid % numNonterminals;
    int pairIndex = tid / numNonterminals;

    int r, c;
    decodeUpperTriangle(pairIndex, N, r, c);

    int dpIndex = r*(N*numNonterminals) + c*numNonterminals + A;

    // If dp_in == 0 or bpCount == 0 => skip quickly
    int count = bpCount[dpIndex];
    if (count == 0) { return; }

    // We'll fill from offset start
    int offsetStart = bpOffset[dpIndex];
    int outPos = offsetStart;

    // Grammar offsets
    int startGC = grammarOffsets[A];
    int endGC   = (A+1 < grammarOffsetsSize) ? grammarOffsets[A+1] : grammarSize;

    int pairOffset = allFSAPairsOffsets[r*N + c];
    int pairOffsetNext = ( (r*N + c)+1 < allFSAPairsOffsetsSize )
                       ? allFSAPairsOffsets[r*N + c + 1]
                       : allFSAPairsSize;

    // Exactly like bp_count, but now we store expansions
    for (int idx = pairOffset; idx < pairOffsetNext; idx++) {
        int m = allFSAPairs[idx];

        for (int g = startGC; g < endGC; g += 2) {
            int B = grammar[g];
            int C = grammar[g+1];

            int idxBM = r*(N*numNonterminals) + m*numNonterminals + B;
            int idxMC = m*(N*numNonterminals) + c*numNonterminals + C;
            if (dp_in[idxBM] != 0 && dp_in[idxMC] != 0) {
                // store (B, m, C) in bpStorage
                // Suppose each triple is 3 consecutive int in bpStorage:
                bpStorage[outPos*3 + 0] = B;
                bpStorage[outPos*3 + 1] = m;
                bpStorage[outPos*3 + 2] = C;
                outPos++;
            }
        }
    }
}

inline uint lcg_random(thread uint& state) { return 1664525u * state + 1013904223u; }
inline uint lcg_randomRange(thread uint& state, uint range) { return lcg_random(state) % range; }

inline void sampleCell(
   const device uchar*    dp_in,
   const device int*      bpCount,
   const device int*      bpOffset,
   const device int*      bpStorage,
   int N,
   int numNonterminals,
   int r,
   int c,
   int A,
   thread uint& rngState,
//   thread device char* localWord, // output buffer
   thread int& wordLen,
   const int maxWordLen
) {
    // If we've reached maximum length, just stop to avoid overflow
    if (wordLen >= maxWordLen) { return; }

    // If A is a terminal (leaf), append its "symbol" to localWord
//    if (isTerminal[A] != 0) {
//        localWord[wordLen] = char(A); // Example: store A as the character
//        wordLen++;
//        return;
//    }

    // Otherwise, A is internal. Let's see how many expansions it has:
    int dpIndex = r*(N*numNonterminals) + c*numNonterminals + A;
    int count   = bpCount[dpIndex];
    // No expansions? Then do nothing or treat it as a dead end.
    if (count == 0) { return; }

    // Choose a random expansion among [0..count-1]
    int offsetStart = bpOffset[dpIndex];
    int randIndex   = offsetStart + int(lcg_randomRange(rngState, uint(count)));

    // Read that expansion (B, m, C)
    int B = bpStorage[randIndex*3 + 0];
    int M = bpStorage[randIndex*3 + 1];
    int C_= bpStorage[randIndex*3 + 2];

    // Recurse on (r, M, B) and (M, c, C)
    sampleCell(dp_in, bpCount, bpOffset, bpStorage,
               N, numNonterminals, r, M, B,
               rngState, 
//               localWord, 
               wordLen, maxWordLen);

    // If that appended some terminals, continue with the second child
    sampleCell(dp_in, bpCount, bpOffset, bpStorage,
               N, numNonterminals, M, c, C_,
               rngState, 
//               localWord, 
               wordLen, maxWordLen);
}

kernel void sample_words(
    const device uchar*   dp_in           [[buffer(0)]],
    const device int*     bpCount         [[buffer(1)]],
    const device int*     bpOffset        [[buffer(2)]],
    const device int*     bpStorage       [[buffer(3)]],
    const device int*     startIndices    [[buffer(4)]],   // possible start cells
    const device uint*    seeds           [[buffer(5)]],   // random seeds for each thread
    device  char*         sampledWords    [[buffer(6)]],   // output words
    constant int&         numStartIndices [[buffer(7)]],
    constant int&         N               [[buffer(8)]],
    constant int&         numNonterminals [[buffer(9)]],
    constant int&         maxWordLen      [[buffer(10)]],
    constant int&         maxSamples      [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    // If tid >= maxSamples, do nothing
    if (tid >= (uint)maxSamples) { return; }

    // We'll create a local array for the word in "thread" memory.
    // Metal does not allow dynamic arrays in thread memory, so we fix size at compile time or "const int& maxWordLen".
    // But we can do this with 'threadgroup' memory or a fixed-size array if maxWordLen is known at compile time.
    // For demonstration, let's do a static approach for a maximum length:
    thread uchar localWord[1024]; // or some maximum you can manage

    // initialize localWord to 0
    for (int i = 0; i < maxWordLen; i++) { localWord[i] = 0; }

    // Each thread has a seed
    uint rngState = seeds[tid];

    // 1) Pick a random start index in [0..numStartIndices-1]
    uint randIdx = lcg_randomRange(rngState, (uint)numStartIndices);
    int startCell = startIndices[randIdx];

    // Suppose startCell encodes (r, c, A). 
    // You might store it in 32 bits as: (r << 16) | (c << 8) | A, etc.
    // We'll decode that:
    int A = (startCell & 0xFF);
    int c = (startCell >> 8) & 0xFF;
    int r = (startCell >> 16) & 0xFF;
    // This depends on how you actually packed the triple. Adjust as needed.

    // 2) Build the word
    int wordLen = 0;
    sampleCell(dp_in, bpCount, bpOffset, bpStorage,
               N, numNonterminals, r, c, A,
               rngState, 
//               localWord, 
               wordLen, maxWordLen);

    // 3) Write the word to global output
    // Each thread writes to sampledWords[tid * maxWordLen .. (tid * maxWordLen + maxWordLen-1)]
    int baseIdx = int(tid) * maxWordLen;
    for (int i = 0; i < maxWordLen; i++) { sampledWords[baseIdx + i] = localWord[i]; }
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

  val swiftImports = "import Foundation\nimport Metal\nimport Dispatch"

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
        dpSize = dpIn.size, numStates, numNonterminals, grLen, goLen, apLen, aoLen
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

    let numNonzero = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
    var prevValue = numNonzero.contents().bindMemory(to: UInt32.self, capacity: 1).pointee

    for r in 0..<numStates {
        var zero: UInt32 = 0
        memcpy(numNonzero.contents(), &zero, MemoryLayout<UInt32>.size)

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
        enc.setBuffer(numNonzero, offset: 0, index: 12)

        enc.dispatchThreads(tiling, threadsPerThreadgroup: MTLSizeMake(1,1,1))
        enc.endEncoding()
        cmb.commit()
        cmb.waitUntilCompleted()

        let currentValue = numNonzero.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        if currentValue == prevValue {
            print("Fixpoint escape at round: \(r)/\(numStates), total=\(currentValue), size: \(dpSizeBytes)"); fflush(stdout)
            break
        }
        prevValue = currentValue
    }

//    buildBackpointers(
//      dp_in: bufA,
//      grammar: grammarBuf,
//      grammarOffsets: grammarOffBuf,
//      allFSAPairs: pairsBuf,
//      allFSAPairsOffsets: pairsOffBuf,
//      numNonterminals: Int(numNonterminals),
//      N: Int(ns),
//      nt: Int(nt),
//      gr: Int(gr),
//      go: Int(go),
//      ap: Int(ap),
//      ao: Int(ao)
//    )

    memcpy(dp_out, bufA.contents(), dpSizeBytes)
}

func buildBackpointers(
  dp_in: MTLBuffer, // size = N*N*numNonterminals (bytes)
  grammar: MTLBuffer,
  grammarOffsets: MTLBuffer,
  allFSAPairs: MTLBuffer,
  allFSAPairsOffsets: MTLBuffer,
  numNonterminals: Int,
  N: Int,
  nt: Int,
  gr: Int,
  go: Int,
  ap: Int,
  ao: Int
) -> (MTLBuffer, MTLBuffer, MTLBuffer) {
    let start = DispatchTime.now()
    // 1) Allocate bpCount
    let countSize = Int(N)*Int(N)*Int(numNonterminals) * MemoryLayout<Int32>.stride
    let bpCountBuf = device.makeBuffer(length: countSize, options: [])!
    
    // 2) Launch bp_count kernel
    let totalPairs = (Int(N)*(Int(N)-1))/2
    let totalCells = totalPairs * Int(numNonterminals)
    let threadsCount = MTLSizeMake(totalCells,1,1)

    // Command buffer
    let commandBuffer1 = queue.makeCommandBuffer()!
    let encoder1 = commandBuffer1.makeComputeCommandEncoder()!
    let stride = MemoryLayout<Int32>.stride
    var ns = N
    var nt = nt
    var gr = gr
    var go = go
    var ap = ap
    var ao = ao
    encoder1.setComputePipelineState(psoBpCount)  // create psoBpCount from "bp_count" function
    encoder1.setBuffer(dp_in,               offset: 0, index: 0)
    encoder1.setBuffer(grammar,             offset: 0, index: 1)
    encoder1.setBuffer(grammarOffsets,      offset: 0, index: 2)
    encoder1.setBuffer(allFSAPairs,         offset: 0, index: 3)
    encoder1.setBuffer(allFSAPairsOffsets,  offset: 0, index: 4)
    encoder1.setBytes(&ns,                  length: stride, index: 5)
    encoder1.setBytes(&nt,                  length: stride, index: 6)
    encoder1.setBytes(&gr,                  length: stride, index: 7)
    encoder1.setBytes(&go,                  length: stride, index: 8)
    encoder1.setBytes(&ap,                  length: stride, index: 9)
    encoder1.setBytes(&ao,                  length: stride, index: 10)
    encoder1.setBuffer(bpCountBuf,          offset: 0, index: 11)

    encoder1.dispatchThreads(threadsCount, threadsPerThreadgroup: MTLSizeMake(1,1,1))
    encoder1.endEncoding()
    commandBuffer1.commit()
    commandBuffer1.waitUntilCompleted()

    // 3) Compute prefix sum of bpCount --> bpOffset 
    //    This can be done in GPU or CPU. We'll assume CPU. 
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
    //    Each expansion is 3 ints (B, M, C). So 3 * 4 = 12 bytes per expansion
    let bpStorageSize = Int(totalExpansions) * 3 * MemoryLayout<Int32>.stride
    let bpStorageBuf = device.makeBuffer(length: bpStorageSize, options: [])!

    // 6) Launch bp_write kernel
    let commandBuffer2 = queue.makeCommandBuffer()!
    let encoder2 = commandBuffer2.makeComputeCommandEncoder()!
    encoder2.setComputePipelineState(psoBpWrite)
    encoder2.setBuffer(dp_in,              offset: 0, index: 0)
    encoder2.setBuffer(grammar,            offset: 0, index: 1)
    encoder2.setBuffer(grammarOffsets,     offset: 0, index: 2)
    encoder2.setBuffer(allFSAPairs,        offset: 0, index: 3)
    encoder2.setBuffer(allFSAPairsOffsets, offset: 0, index: 4)
    encoder2.setBytes(&ns,                 length: stride, index: 5)
    encoder2.setBytes(&nt,                 length: stride, index: 6)
    encoder2.setBytes(&gr,                 length: stride, index: 7)
    encoder2.setBytes(&go,                 length: stride, index: 8)
    encoder2.setBytes(&ap,                 length: stride, index: 9)
    encoder2.setBytes(&ao,                 length: stride, index: 10)
    encoder2.setBuffer(bpCountBuf,         offset: 0, index: 11)
    encoder2.setBuffer(bpOffsetBuf,        offset: 0, index: 12)
    encoder2.setBuffer(bpStorageBuf,       offset: 0, index: 13)

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
        if start >= end {
            // No work for this chunk
            partialSum[chunkIndex] = 0
            return
        }

        var sum: Int32 = 0
        for i in start..<end {
            ptrOffset[i] = sum
            sum += ptrCount[i]
        }
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
                let s = UInt8(triples[stride * i + 3])
                let index = r * rowMultiplier + c * colMultiplier + A
                ptr[index] = s
            }
        }
    }

    return bufA
}

private var device: MTLDevice!, queue: MTLCommandQueue!, 
            psoCflMul: MTLComputePipelineState!, serialize: MTLComputePipelineState!,
            psoBpCount: MTLComputePipelineState!, psoBpWrite: MTLComputePipelineState!

@_cdecl("setup") public func setup() {
    device = MTLCreateSystemDefaultDevice()!
    queue = device.makeCommandQueue()!

    let library = try! device.makeLibrary(source: #""${'"'}$metalSrc${'"'}""#, options: nil)
    psoCflMul   = try! device.makeComputePipelineState(function: library.makeFunction(name: "cfl_mul_upper")!)
    serialize   = try! device.makeComputePipelineState(function: library.makeFunction(name: "serialize_dpm")!)
    psoBpCount  = try! device.makeComputePipelineState(function: library.makeFunction(name: "bp_count")!)
    psoBpWrite  = try! device.makeComputePipelineState(function: library.makeFunction(name: "bp_write")!)
}"""

  // Assumes: |Σ| <= 126
  fun encodeGrammarHeader(cfg: CFG = pythonStatementCNF): String =
    cfg.nonterminals.map {
      if (it !in cfg.unitNonterminals) ""
      else cfg.bimap.UNITS[it]!!.map { cfg.tmMap[it]!! + 1 }.joinToString(",")
    }.mapIndexed { i, it -> "constant uint8_t nt$i[]={$it};" }.joinToString("") +
      "constant uint8_t* constant nt[]={${(0..<cfg.nonterminals.size).joinToString(",") {"nt$it"}}};"

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