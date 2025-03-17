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
import kotlin.math.absoluteValue
import kotlin.system.measureTimeMillis
import kotlin.time.*

/*
./gradlew generateDylib
 */

fun main1() {
  val cfg = pythonStatementCNFAllProds
//  Semicolon.Small_Stmts -> In_Keyword Or_Test
   cfg.vindex2.forEachIndexed { i, nts ->
    nts.forEach { rhs ->
      println("${cfg.bindex.indexedNTs[i]} -> ${cfg.bindex.indexedNTs[rhs[0]]} ${cfg.bindex.indexedNTs[rhs[1]]}")
    }
//    println(it.first + " -> " + it.second.first() + " " + it.second.last())
  }
}

fun main() {
  val cfg = pythonStatementCNFAllProds
  val pythonCode = "NAME = [ ( STRING , NAME ) , , ( NAME , NAME ) , ( NAME , NAME ) , ( NAME , NAME ) , , ( NAME , NAME ) ] NEWLINE"
    .tokenizeByWhitespace()
  val radius = 3
  val levFSA = makeLevFSA(pythonCode, radius)
  val maxWordLen = pythonCode.size + radius + 10

  initiateSerialRepair(pythonCode, cfg).take(10).forEach { println("$it / ${it in cfg.language}") }
//  System.exit(1)

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
//          if (!sp.arg.startsWith("[!=]") && sp.arg.toString() !in setOf("NAME", "STRING"))
//          println(A + "->" + sp.arg.toString() + " ::= " + pb.toString(2).padStart(16, '0'))
          val g = listOf(stateMap[q0]!!, stateMap[q1]!!, Aidx).map { it.toUShort() } + pb
//          if (g.take(3) == listOf(140, 145, 65))
          g
//          .also { if(it != 65534.toUShort()) println("${sp.arg}/$A/$it/${terminalLists[Aidx]}/${terminalLists[Aidx].indexOf(sp.arg.drop(4)) + 1}") }
        }
      }
        .distinct()
//        .onEach { (q0, q1, Aidx, pb) -> println("ANOMALY: ${q0}, ${q1}, ${Aidx}: ${pb.toString(2).padStart(16, '0')}") }
        .flatten().toUShortArray()

       .also { println("Encoded instance in ${it.size * 2} bytes / ${numStates*numStates*numNonterminals} / ${clock.elapsedNow()}") }
    }

    val maxSamples = 10_000

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
      else if (it - 1 == 0) "" // TODO: why do we need to block NEWLINE?
      else cfg.tmLst[it.toInt() - 1]
    }.split(Regex("( 0)+")) // Runs of zeros delimit individual words
      .filter { it.isNotBlank() }
      .map { it.tokenizeByWhitespace().joinToString(" ") }
      .distinct()
      .partition { "$it NEWLINE" in cfg.language }

    val totalSamples = valid.size + invalid.size
    println("\nValid samples:\n")
    valid.take(10).forEachIndexed { i, it -> println("$i.) ${it.trim()} / ${levFSA.recognizes(("$it NEWLINE").trim())}") }
    println("\nInvalid samples:\n")
    invalid.take(10).forEachIndexed { i, it -> println("$i.) ${it.trim()}") }

    println("...\n\nGPU repairs: ${valid.size} valid / $totalSamples unique / $maxSamples total")
  }.also { println("Total time: ${it}ms\n") }
}

fun initiateSerialRepair(brokenStr: List<Σᐩ>, cfg: CFG): Sequence<Σᐩ> {
  val upperBound = MAX_RADIUS * 3
//  val monoEditBounds = cfg.maxParsableFragmentB(brokenStr, pad = upperBound)
  val timer = TimeSource.Monotonic.markNow()
  val bindex = cfg.bindex
  val width = cfg.nonterminals.size
  val vindex = cfg.vindex
  val ups = cfg.unitProductions
  val t2vs = cfg.tmToVidx
  val maxBranch = vindex.maxOf { it.size }
  val startIdx = bindex[START_SYMBOL]

  fun nonemptyLevInt(levFSA: FSA): Int? {
    val ap: List<List<List<Int>?>> = levFSA.allPairs
    val dp = Array(levFSA.numStates) { Array(levFSA.numStates) { BooleanArray(width) { false } } }

    levFSA.allIndexedTxs0(ups, bindex).forEach { (q0, nt, q1) -> dp[q0][q1][nt] = true }
    var minRad: Int = Int.MAX_VALUE

    // For pairs (p,q) in topological order
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
            for (r in appq)
              if (dp[p][r][B] && dp[r][q][C]) {
                dp[p][q][A] = true
                break@outerloop
              }
          }

          if (p == 0 && A == startIdx && q in levFSA.finalIdxs && dp[p][q][A]) {
            val (x, y) = levFSA.idsToCoords[q]!!
            /** See final state conditions for [makeExactLevCFL] */
            // The minimum radius such that this final state is included in the L-FSA
            minRad = minOf(minRad, (brokenStr.size - x + y).absoluteValue)
          }
        }
      }
    }

    return if (minRad == Int.MAX_VALUE) null else minRad
  }

  val led = (3 until upperBound)
    .firstNotNullOfOrNull { nonemptyLevInt(makeLevFSA(brokenStr, it)) } ?:
  upperBound.also { println("Hit upper bound") }
  val radius = led + LED_BUFFER

  println("Identified LED=$led, radius=$radius in ${timer.elapsedNow()}")

  val levFSA = makeLevFSA(brokenStr, radius)

  val nStates = levFSA.numStates
  val tml = cfg.tmLst
  val tms = tml.size
  val tmm = cfg.tmMap

  // 1) Create dp array of parse trees
  val dp: Array<Array<Array<GRE?>>> = Array(nStates) { Array(nStates) { Array(width) { null } } }

  // 2) Initialize terminal productions A -> a
  val aitx = levFSA.allIndexedTxs1(ups)
  for ((p, σ, q) in aitx) for (Aidx in t2vs[tmm[σ]!!])
    dp[p][q][Aidx] = ((dp[p][q][Aidx] as? GRE.SET) ?: GRE.SET(tms))
      .apply { s.set(tmm[σ]!!)/*; dq[p][q].set(Aidx)*/ }

  var maxChildren = 0
  var location = -1 to -1

  // 3) CYK + Floyd Warshall parsing
  for (dist in 1 until nStates) {
    for (p in 0 until (nStates - dist)) {
      val q = p + dist
      if (levFSA.allPairs[p][q] == null) continue
      val appq = levFSA.allPairs[p][q]!!

      for ((Aidx, indexArray) in vindex.withIndex()) {
        //      println("${cfg.bindex[Aidx]}(${pm!!.ntLengthBounds[Aidx]}):${levFSA.stateLst[p]}-${levFSA.stateLst[q]}(${levFSA.SPLP(p, q)})")
        val rhsPairs = dp[p][q][Aidx]?.let { mutableListOf(it) } ?: mutableListOf()
        outerLoop@for (j in 0..<indexArray.size step 2) {
          val Bidx = indexArray[j]
          val Cidx = indexArray[j + 1]
          for (r in appq) {
            val left = dp[p][r][Bidx]
            if (left == null) continue
            val right = dp[r][q][Cidx]
            if (right == null) continue
            // Found a parse for A
            rhsPairs += left * right
            //            if (rhsPairs.size > 10) break@outerLoop
          }
        }

        val list = rhsPairs.toTypedArray()
        if (rhsPairs.isNotEmpty()) {
          if (list.size > maxChildren) {
            maxChildren = list.size
            location = p to q
          }
          dp[p][q][Aidx] = GRE.CUP(*list)
        }
      }
    }
  }

  println("Completed parse matrix in: ${timer.elapsedNow()}")

  // 4) Gather final parse trees from dp[0][f][startIdx], for all final states f
  val allParses = levFSA.finalIdxs.mapNotNull { q ->
    println("Accept triples kotlin: 0, $q, $startIdx")
    dp[0][q][startIdx] }

  val clock = TimeSource.Monotonic.markNow()
  // 5) Combine them under a single GRE
  return (
      if (allParses.isEmpty()) sequenceOf()
      else GRE.CUP(*allParses.toTypedArray()).let {
        it.words(tml) { clock.hasTimeLeft() }
//      if ( == null) it.words(tml) { clock.hasTimeLeft() }
//      else it.wordsOrdered(tml, ngrams) { clock.hasTimeLeft() }
      }
  ).also { println("Parsing took ${timer.elapsedNow()} with |σ|=${brokenStr.size}, " +
     "|Q|=$nStates, |G|=${cfg.size}, maxBranch=$maxBranch, |V|=$width, |Σ|=$tms, maxChildren=$maxChildren@$location") }
}

sealed class GRE(open vararg val args: GRE) {
  class SET(val s: KBitSet): GRE() { constructor(size: Int): this(KBitSet(size)) }
  class CUP(override vararg val args: GRE): GRE(*args)
  class CAT(val l: GRE, val r: GRE): GRE(l, r)

  fun words(terminals: List<Σᐩ>, shouldContinue: () -> Boolean = { true }): Sequence<Σᐩ> =
    enumerate(shouldContinue).takeWhile { shouldContinue() }.distinct()
      .map { it.mapNotNull { terminals[it].let { if (it == "ε") null else it } }.joinToString(" ") }

  fun enumerate(shouldContinue: () -> Boolean = { true }): Sequence<List<Int>> = sequence {
    if (!shouldContinue()) emptySequence<List<Int>>()
    else when (this@GRE) {
      is SET -> yieldAll(s.toList().map { listOf(it) })
      is CUP -> for (a in args) yieldAll(a.enumerate(shouldContinue))
//      yieldAll(args.map { it.enumerate().toSet() }.reduce { a, b -> a + b })
      is CAT -> for (lhs in l.enumerate(shouldContinue)) for (rhs in r.enumerate(shouldContinue))
        if (lhs.isEmpty()) {
          if (rhs.isEmpty()) yield(emptyList()) else rhs
        } else {
          if (rhs.isEmpty()) yield(lhs)
          else yield(lhs + rhs)
        }
    }
  }

  operator fun plus(g: GRE): GRE = CUP(this, g)
  operator fun times(g: GRE): GRE = CAT(this, g)
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

    // loop over midpoints
    for (int pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++)
       for (int g = startGC, mdpt = allFSAPairs[pairIdx]; g < endGC; g += 2) {
          int B = vidx[g]; int C = vidx[g+1];
          
          int idxBM = r * snt + mdpt * numNonterminals + B;
          int idxMC = mdpt * snt + c * numNonterminals + C;

          if (dp_in[idxBM] && dp_in[idxMC]
//              && r < mdpt && mdpt < c && r < c - 1 && ((mdpt - r) + (c - mdpt) == (c - r))
//              && idxBM < dpIdx
//              && idxMC > dpIdx
          ) {
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

    if (!dp_in[dpIdx]) { bpCount[dpIdx] = 0; return; }

    int count = 0;
    // loop over midpoints
    for (int pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++)
      for (int g = startGC, mdpt = allFSAPairs[pairIdx]; g < endGC; g += 2) {
        int B = vidx[g], C = vidx[g+1];

        int idxBM = r*snt + mdpt*numNonterminals + B;
        int idxMC = mdpt*snt + c*numNonterminals + C;
        if (dp_in[idxBM] && dp_in[idxMC]
//            && r < mdpt && mdpt < c && r < c - 1
//            && idxBM < dpIdx
//            && idxMC > dpIdx
         ) count++;
//        if ((dp_in[idxBM] & 0x01 && dp_in[idxMC] & 0x01) || ((1<dp_in[idxBM]) && (1<dp_in[idxMC]))) count++;
      }

    bpCount[dpIdx] = count;//max(count, 1);
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

    if (!dp_in[dpIdx]) return;

    int outPos = bpOffset[dpIdx];

    // Exactly like bp_count, but now we store expansions
    for (int pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++)
      for (int poff = startGC, mdpt = allFSAPairs[pairIdx]; poff < endGC; poff += 2) {
        int B = vidx[poff], C = vidx[poff+1];

        int idxBM = r*snt + mdpt*numNonterminals + B;
        int idxMC = mdpt*snt + c*numNonterminals + C;
        if (dp_in[idxBM] && dp_in[idxMC] 
//            && r < mdpt && mdpt < c && r < c - 1 && ((mdpt - r) + (c - mdpt) == (c - r))
//            && idxBM < dpIdx // Why are these necessary? Maybe indexing bug?
//            && idxMC > dpIdx
        ) {
//          if (2*bpCount[dpIdx] < (outPos - bpOffset[dpIdx])) return;
          // store (B, m, C) in bpStorage
          // Suppose each triple is 2 consecutive int in bpStorage:
          bpStorage[outPos*2 + 0] = idxBM;
          bpStorage[outPos*2 + 1] = idxMC;
          outPos++;
        }
      }
}

inline uint lcg_random(thread uint& state) { return 1664525u * state + 1013904223u; }
//inline uint lcg_random(thread uint& state) { return state % 100; }
inline uint lcg_randomRange(thread uint& state, uint range) { return lcg_random(state) % range; }

inline int alreadyVisited(thread const int* visitedList, int visitedCount, int candidate) {
    for (int i = visitedCount; 0 < i; i--) { if (visitedList[i] == candidate) return i + 1; }
    return 0;
}

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
      
      if ((dp_in[dpIdx] >> 1)) { // If we are dealing with a leaf node (i.e., a unit nonterminal/terminal)
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
//        stack[top++] = (idxMC == dpIdx) ? -idxMC : idxMC; stack[top++] = (idxBM == dpIdx) ? -idxBM : idxBM;
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
//    let sampledWordsBuf = sampleWordsCPU(
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

       runKernel(pipelineState: cfl_mul_upper, numThreads: totalThreads) { enc in
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

//func binaryString(from value: UInt16) -> String {
//    String(String(String(value, radix: 2).reversed()).padding(toLength: 16, withPad: "0", startingAt: 0).reversed())
//}

// takes a flattened array of sparse quadruples and reconstructs a dense matrix
private func reconstructDPBuffer(dp_in: UnsafePointer<UInt16>?, dpSize: Int, numStates: Int) -> MTLBuffer {
    let startTime = DispatchTime.now()
    let totalCount = numStates*numStates*numNonterminals
    print("tc = \(totalCount), mlo=\(MemoryLayout<UInt16>.stride)")
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
//                let bs = binaryString(from: triples[stride * i + 3])
//                print("RECV: \(r), \(c), \(A): " + bs)
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
        print("accept triples swift: 0, \(q), \(startIdx)"); fflush(stdout)
        print("Start idx: \(dpIndex)"); fflush(stdout)
        if dpPtr[dpIndex] != 0 { startIdxs.append(Int32(dpIndex)) }
    }


    let i32stride = MemoryLayout<UInt32>.stride
    let seeds = (0..<maxSamples).map { _ in UInt32.random(in: .min ..< .max) }
    let seedsBuf = device.makeBuffer(bytes: seeds, length: maxSamples * i32stride, options: [])!
    let startBuf = device.makeBuffer(bytes: startIdxs, length: startIdxs.count * i32stride, options: [])!

    // We'll store each sampled word in a row of length `maxWordLen`.
    let resultSize = maxSamples * maxWordLen
    let outBuf = device.makeBuffer(length: resultSize, options: [])!

    runKernel(pipelineState: sample_words, numThreads: maxSamples) { enc in
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

    // Show a few samples.
    let ptr = outBuf.contents().bindMemory(to: UInt8.self, capacity: resultSize)
    for i in 0..<min(5, maxSamples) {
        let slice = Array(UnsafeBufferPointer(start: ptr + i*maxWordLen, count: maxWordLen))
        print("Sample \(i): \(slice)"); fflush(stdout)
    }

    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
    print("Sampled words in \(ms) ms"); fflush(stdout)
    return outBuf
}

private func sampleWordsCPU(
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
    print("Sampling words..."); fflush(stdout)
    let t0 = DispatchTime.now()
    let totalCount = numStates * numStates * numNonterminals
    let dp = UnsafeBufferPointer(start: dp_in.contents().assumingMemoryBound(to: UInt16.self), count: totalCount)
    let bpCount = UnsafeBufferPointer(start: bpCount.contents().assumingMemoryBound(to: Int32.self), count: totalCount)
    let bpOffset = UnsafeBufferPointer(start: bpOffset.contents().assumingMemoryBound(to: Int32.self), count: totalCount)
    let bpStore = UnsafeBufferPointer(start: bpStorage.contents().assumingMemoryBound(to: Int32.self), count: bpStorage.length / MemoryLayout<Int32>.stride)

    let acceptStates = accsPt.map { Array(UnsafeBufferPointer(start: $0, count: accsSize)).map(Int.init) } ?? []
    let startIdxs = acceptStates.compactMap { q in
        print("accept triples swift: 0, \(q), \(startIdx)"); fflush(stdout)
        let idx = q * numNonterminals + startIdx // accept states + start indices
        return dp[idx] != 0 ? Int32(idx) : nil
    }

    let seeds = (0..<maxSamples).map { _ in UInt32.random(in: .min ..< .max) }
    let outBuf = device.makeBuffer(length: maxSamples * maxWordLen, options: [])!
    let outPtr = outBuf.contents().assumingMemoryBound(to: UInt8.self)

    func lcgRandom(_ state: inout UInt32) -> UInt32 {
        state = state &* 1664525 &+ 1013904223
        return state
    }

    func sampleTopDown(_ idqx: Int, _ rng: inout UInt32, _ word: inout [UInt8], _ len: inout Int, _ depth: Int) {
        if len >= maxWordLen - 5 { print("Exception: OVERFLOW"); fflush(stdout) }
//        print("size: \(bpCount.count)/\(idqx)"); fflush(stdout)
        let tm_contents = dp[idqx], expCount = Int(bpCount[idqx])
        let ntm = idqx % numNonterminals, numTms = nt_tm_lens[ntm], ntoffset = offsets[ntm]
        if (tm_contents >> 1 != 0) {
//            print("nt_tm: \(nt_tm_lens.count)/\(ntm) offsets: \(offsets.count)/\(ntm) "); fflush(stdout)
            let lit = Int((tm_contents >> 1) & 0x7FFF), isNeg = (tm_contents & 0x8000) != 0
            if isNeg {
                let r = Int(lcgRandom(&rng) % UInt32(numTms))
                let tm = (r != lit - 1 ? r : (r - 1) % (numTms))
                let ix = ntoffset + tm
                print("choice: ix=\(ix)/r=\(r)/c=\(tm)/numTms=\(numTms)/len=\(len)/blocked=\(lit)/\(ntm)/name=\(nt_names[ntm])"); fflush(stdout)
                word[len] = UInt8(truncatingIfNeeded: all_tm[ix] + 1)
            } else {
                let ix = ntoffset + lit - 1
//                print("ix:\(ix)/ntoffset:\(ntoffset)/lit:\(lit)/\(all_tm.count)/\(len)"); fflush(stdout)
                word[len] = numTms != 0 ? UInt8(truncatingIfNeeded: all_tm[ix] + 1) : 99
                print("pos_lit: \(nt_names[ntm])->\(tm_names[Int(word[len]) - 1])->\(word[len])"); fflush(stdout)
            }

            let curWord = word.map { "\($0 == 0 ? "" : tm_names[Int($0) - 1])" }.joined(separator: " ")
            print("current word: \(word)"); fflush(stdout)
            print("current word: \(curWord)"); fflush(stdout)
            len += 1
        } else if expCount > 0 {
            let p = Int(lcgRandom(&rng) % UInt32(expCount))
            let nextId = Int(bpOffset[idqx]) + p
            print("sampled: \(p)/\(expCount)"); fflush(stdout)
            let ntl = Int(bpStore[nextId * 2]) % numNonterminals
            let ntr = Int(bpStore[nextId * 2 + 1]) % numNonterminals
            print("expanding (depth=\(depth)): idqx=\(idqx), nid=\(nextId*2), ntli=\(bpStore[nextId * 2]), ntri=\(bpStore[nextId * 2 + 1]) \(nt_names[ntm]) -> \(nt_names[ntl]) \(nt_names[ntr])"); fflush(stdout)
            let curWord = word.map { "\($0 == 0 ? "" : tm_names[Int($0) - 1])" }.joined(separator: " ")
//            print("current word: \(word)"); fflush(stdout)
//            print("current word: \(curWord)"); fflush(stdout)
            let idxBM = Int(bpStore[nextId * 2]), idxMC = Int(bpStore[nextId * 2 + 1])
            
            sampleTopDown(idxBM, &rng, &word, &len, depth + 1)
            sampleTopDown(idxMC, &rng, &word, &len, depth + 1)
        } //else { print("Exception: ELSE"); fflush(stdout) }
    }

    for i in 0..<20 {
        var rng = seeds[i], word = [UInt8](repeating: 0, count: maxWordLen), len = 0
        sampleTopDown(Int(startIdxs[Int(lcgRandom(&rng) % UInt32(startIdxs.count))]), &rng, &word, &len, 0)
        outPtr.advanced(by: i * maxWordLen).initialize(from: word, count: maxWordLen)
    }

    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
    print("Sampled words in \(ms) ms")
    return outBuf
}

private func writeBackpointersSerial(
    dp_in: MTLBuffer,
    allFSAPairs: MTLBuffer,
    allFSAPairsOffsets: MTLBuffer,
    numStates: Int,
    allPairsSize: Int,
    allPairsOffsetsSize: Int,
    bpOffset: MTLBuffer,
    bpStorage: MTLBuffer,
    bpCount: MTLBuffer
) {
    let dp_in_ptr = dp_in.contents().assumingMemoryBound(to: UInt16.self)
    let allFSAPairs_ptr = allFSAPairs.contents().assumingMemoryBound(to: Int32.self)
    let allFSAPairsOffsets_ptr = allFSAPairsOffsets.contents().assumingMemoryBound(to: Int32.self)
    let bpOffset_ptr = bpOffset.contents().assumingMemoryBound(to: Int32.self)
    let bpStorage_ptr = bpStorage.contents().assumingMemoryBound(to: Int32.self)
    let bpCount_ptr = bpCount.contents().assumingMemoryBound(to: Int32.self)
    
    let N = numStates
    let snt = N * numNonterminals
    
    for r in 0..<N - 1 {
        for c in (r + 1)..<N {
            for A in 0..<numNonterminals {
                let dpIdx = r * snt + c * numNonterminals + A
                
                if dp_in_ptr[dpIdx] == 0 { continue }
                
                let outPos = Int(bpOffset_ptr[dpIdx])
                var currentOutPos = outPos
                
                let aoi = r * N + c + 1
                let pairOffset = Int(allFSAPairsOffsets_ptr[aoi - 1])
                let pairOffsetNext = (aoi < allPairsOffsetsSize) ? Int(allFSAPairsOffsets_ptr[aoi]) : allPairsSize
                
                let startGC = vidx_offsets[A]
                let endGC = (A + 1 < vidx_offsets.count) ? vidx_offsets[A + 1] : vidx.count
                
                for pairIdx in pairOffset..<pairOffsetNext {
                    let mdpt = Int(allFSAPairs_ptr[pairIdx])
                    
                    for poff in stride(from: startGC, to: endGC, by: 2) {
                        let B = vidx[poff]
                        let C = vidx[poff + 1]
                        
                        let idxBM = r * snt + mdpt * numNonterminals + B
                        let idxMC = mdpt * snt + c * numNonterminals + C
                        
                        if (dp_in_ptr[idxBM] != 0 && dp_in_ptr[idxMC] != 0) {
                            
//                            bpStorage_ptr[currentOutPos * 2] = Int32(idxBM)
//                            bpStorage_ptr[currentOutPos * 2 + 1] = Int32(idxMC)
//                            currentOutPos += 1
                            
                            if ((currentOutPos - outPos) > bpCount_ptr[dpIdx]) {
                              print("\(currentOutPos - outPos) - dp[\(r),\(c),\(A)]=\(bpCount_ptr[dpIdx])"); fflush(stdout)
                              print("dp[\(r),\(c),\(A)]=\(bpCount_ptr[dpIdx]) - exceeded: \((currentOutPos - outPos))!")
                            } else {
                            if (nt_names[B] == "In_Keyword" && nt_names[C] == "Or_Test") {
                              print("Wrote FORBIDDEN: \(nt_names[A]) -> \(nt_names[B]) \(nt_names[C]) to idx: \(currentOutPos * 2), dpIdx=\(dpIdx), idxBM=\(idxBM), idxMC=\(idxMC)")
                            }
                              bpStorage_ptr[currentOutPos * 2] = Int32(idxBM)
                              bpStorage_ptr[currentOutPos * 2 + 1] = Int32(idxMC)
                              currentOutPos += 1
//                              print("Under: \((currentOutPos - outPos)) / \(bpCount_ptr[dpIdx])")
                            }
                        }
                    }
                }
            }
        }
    }
}

private func countBackpointersSerial(
    dp_in: MTLBuffer,
    allFSAPairs: MTLBuffer,
    allFSAPairsOffsets: MTLBuffer,
    numStates: Int,
    allFSAPairsSize: Int,
    allFSAPairsOffsetsSize: Int,
    bpCount: MTLBuffer
) {
    // Bind buffer contents to Swift types
    let dp_in_ptr = dp_in.contents().assumingMemoryBound(to: UInt16.self)
    let allFSAPairs_ptr = allFSAPairs.contents().assumingMemoryBound(to: Int32.self)
    let allFSAPairsOffsets_ptr = allFSAPairsOffsets.contents().assumingMemoryBound(to: Int32.self)
    let bpCount_ptr = bpCount.contents().assumingMemoryBound(to: Int32.self)

    let N = numStates
    let snt = N * numNonterminals // Stride for nonterminals

    // Iterate over upper triangle (r < c) and all nonterminals
    for r in 0..<N - 1 {
        for c in (r + 1)..<N {
            for A in 0..<numNonterminals {
                // Compute index into dp_in and bpCount
                let dpIdx = r * snt + c * numNonterminals + A

                // If dp_in[dpIdx] is zero, set bpCount[dpIdx] to zero and skip
                if dp_in_ptr[dpIdx] == 0 { continue }

                // Compute pair offset indices
                let aoi = r * N + c + 1
                let pairOffset = Int(allFSAPairsOffsets_ptr[aoi - 1])
                let pairOffsetNext = (aoi < allFSAPairsOffsetsSize) ? Int(allFSAPairsOffsets_ptr[aoi]) : allFSAPairsSize

                // Compute range for nonterminal pairs
                let startGC = vidx_offsets[A]
                let endGC = (A + 1 < vidx_offsets.count) ? vidx_offsets[A + 1] : vidx.count

                // Initialize count of valid backpointers
                var count = 0

                // Loop over midpoints
                for pairIdx in pairOffset..<pairOffsetNext {
                    let mdpt = Int(allFSAPairs_ptr[pairIdx])

                    // Loop over nonterminal pairs (B, C) in steps of 2
                    for g in stride(from: startGC, to: endGC, by: 2) {
                        let B = vidx[g]
                        let C = vidx[g + 1]

                        // Compute indices for B and C
                        let idxBM = r * snt + mdpt * numNonterminals + B
                        let idxMC = mdpt * snt + c * numNonterminals + C

                        // Increment count if both dp_in values are non-zero
                        if dp_in_ptr[idxBM] != 0 && dp_in_ptr[idxMC] != 0 { count += 1 }
                    }
                }

//                print("dp[\(r),\(c),\(A)] = \(count)"); fflush(stdout)
                // Store the count in bpCount
                bpCount_ptr[dpIdx] = Int32(count)
            }
        }
    }
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

    runKernel(pipelineState: bp_count, numThreads: totalThreads) { enc in
      enc.setBuffer(dp_in,               offset: 0,      index: 0)
      enc.setBuffer(allFSAPairs,         offset: 0,      index: 1)
      enc.setBuffer(allFSAPairsOffsets,  offset: 0,      index: 2)
      enc.setBytes(&N,                   length: stride, index: 3)
      enc.setBytes(&ap,                  length: stride, index: 4)
      enc.setBytes(&ao,                  length: stride, index: 5)
      enc.setBuffer(bpCountBuf,          offset: 0,      index: 6)
    }

//    countBackpointersSerial(
//        dp_in: dp_in,
//        allFSAPairs: allFSAPairs,
//        allFSAPairsOffsets: allFSAPairsOffsets,
//        numStates: numStates,
//        allFSAPairsSize: allPairsSize,
//        allFSAPairsOffsetsSize: allPairsOffsetsSize,
//        bpCount: bpCountBuf
//    )

    // 2) Prefix sum of bpCount → bpOffset
    let bpOffsetBuf = device.makeBuffer(length: countSize, options: [])!
    parallelPrefixSumCPU(countBuf: bpCountBuf, offsetBuf: bpOffsetBuf, totalCells: totalCells)

    // 3) Find total expansions from last offset + last count
    let offPtr  = bpOffsetBuf.contents().bindMemory(to: Int32.self, capacity: totalCells)

    let cntPtr  = bpCountBuf.contents().bindMemory(to: Int32.self,  capacity: totalCells)
    let totalExpansions = offPtr[totalCells - 1] + cntPtr[totalCells - 1]

    // 4) Allocate bpStorage and run "bp_write" kernel
    let bpStorageBuf = device.makeBuffer(length: Int(totalExpansions) * 2 * stride, options: [])!

    runKernel(pipelineState: bp_write, numThreads: totalThreads) { enc in
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

//    writeBackpointersSerial(
//        dp_in: dp_in,
//        allFSAPairs: allFSAPairs,
//        allFSAPairsOffsets: allFSAPairsOffsets,
//        numStates: numStates,
//        allPairsSize: allPairsSize,
//        allPairsOffsetsSize: allPairsOffsetsSize,
//        bpOffset: bpOffsetBuf,
//        bpStorage: bpStorageBuf,
//        bpCount: bpCountBuf
//    )

    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
    print("Built backpointers in \(ms) ms"); fflush(stdout)
    return (bpCountBuf, bpOffsetBuf, bpStorageBuf)
}

func parallelPrefixSumCPU(countBuf: MTLBuffer, offsetBuf: MTLBuffer, totalCells: Int) {
    let t0 = DispatchTime.now()
    let ptrCount  = countBuf.contents().bindMemory(to: Int32.self, capacity: totalCells)
    let ptrOffset = offsetBuf.contents().bindMemory(to: Int32.self, capacity: totalCells)

    let numChunks = 8
    let chunkSize = (totalCells + numChunks - 1) / numChunks
    var partialSum = [Int32](repeating: 0, count: numChunks)

    // Phase 1: per-chunk prefix sums
    DispatchQueue.concurrentPerform(iterations: numChunks) { c in
        let start = c * chunkSize, end = min(start + chunkSize, totalCells)
        var sum: Int32 = 0
        for i in start..<end { ptrOffset[i] = sum; sum += ptrCount[i] }
        partialSum[c] = sum
    }

    // Phase 2: prefix-sum of partialSum (single-threaded)
    for i in 1..<numChunks { partialSum[i] += partialSum[i - 1] }

    // Phase 3: add chunk offsets in parallel
    DispatchQueue.concurrentPerform(iterations: numChunks) { c in
        guard c > 0 else { return }
        let start = c * chunkSize, end = min(start + chunkSize, totalCells)
        let offset = partialSum[c - 1]
        for i in start..<end { ptrOffset[i] += offset }
    }

    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
    print("Prefix sum completed in \(ms) ms"); fflush(stdout)
}

func runKernel(pipelineState: MTLComputePipelineState, numThreads: Int, body: (MTLComputeCommandEncoder) -> Void) {
    let commandBuffer = queue.makeCommandBuffer()!, enc = commandBuffer.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipelineState)
    body(enc)
    enc.dispatchThreads(MTLSize(width: numThreads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding(); commandBuffer.commit(); commandBuffer.waitUntilCompleted()
}

private var device: MTLDevice!, queue: MTLCommandQueue!, numNonterminals: Int = 0, startIdx: Int = 0,
            cfl_mul_upper: MTLComputePipelineState!,
            bp_count: MTLComputePipelineState!, 
            bp_write: MTLComputePipelineState!,
            sample_words: MTLComputePipelineState!

${genNTTerminalMapForSwift(pythonStatementCNFAllProds)}

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

  private fun genNTTerminalMapForSwift(cfg: CFG): String {
    val terminalLists = cfg.nonterminals.map {
      if (it !in cfg.unitNonterminals) emptyList<Int>()
      else cfg.bimap.UNITS[it]!!.map { cfg.tmMap[it]!! }
    }
    val allTerminals = terminalLists.flatMap { it }
    val offsets = terminalLists.scan(0) { acc, list -> acc + list.size }.dropLast(1)
    val grammarFlattened = cfg.vindex.map { it.toList() }.flatten().toIntArray()
    val grammarOffsets = cfg.vindex.map { it.size }.fold(listOf(0)) { acc, it -> acc + (acc.last() + it) }.toIntArray()

    return """let all_tm: [Int] = [${allTerminals.joinToString(",")}];
              let offsets: [Int] = [${offsets.joinToString(",")}];
              let nt_tm_lens: [Int] = [${terminalLists.map { it.size }.joinToString(",")}];
              let nt_names: [String] = [${cfg.nonterminals.joinToString(",") { "\"$it\"" }}]; // For debugging only
              let tm_names: [String] = [${cfg.tmLst.joinToString(",") { "\"$it\"" }}]; // For debugging only
              
              let vilen: Int = ${grammarFlattened.size};
              let volen: Int = ${grammarOffsets.size};
              let vidx: [Int] = [${grammarFlattened.joinToString(",")}];
              let vidx_offsets: [Int] = [${grammarOffsets.joinToString(",")}];
           """.trimIndent()
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