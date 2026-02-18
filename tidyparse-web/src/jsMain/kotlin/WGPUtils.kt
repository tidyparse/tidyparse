import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.buildLanguageSizeBuf
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import Shader.Companion.writeU32
import ai.hypergraph.kaliningraph.automata.AFSA
import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.automata.TSA
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.START_SYMBOL
import ai.hypergraph.kaliningraph.parsing.bindex
import ai.hypergraph.kaliningraph.parsing.calcStats
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.parsing.rankOneDecomposition
import ai.hypergraph.kaliningraph.parsing.tmMap
import ai.hypergraph.kaliningraph.types.cache
import js.buffer.*
import js.typedarrays.Int32Array
import web.gpu.GPUBuffer
import kotlin.time.TimeSource

suspend fun logActiveNTGrid(
  dpBuf: GPUBuffer,
  numStates: Int,
  numNTs: Int,
  limit: Int = minOf(32, numStates)
) {
  val countsBuf = Shader.GPUBuffer(numStates.toLong() * numStates * 4L, GPUBufferUsage.STCPSD)
  val uniBuf = intArrayOf(numStates, numNTs).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  val groupsX = (numStates + 7) / 8
  val groupsY = (numStates + 7) / 8
  active_nt_count(dpBuf, countsBuf, uniBuf)(groupsX, groupsY, 1)

  val allIndices = (0 until numStates * numStates).toList()
  val allVals = countsBuf.readIndices(allIndices)

  val totalActiveNTs = allVals.fold(0L) { acc, i -> acc + i }
  val totalUTCells = (numStates.toLong() * (numStates - 1)) / 2
  val maxPossibleActive = totalUTCells * numNTs

  val sparsity = if (maxPossibleActive > 0) { (totalActiveNTs.toDouble() / maxPossibleActive) * 100 } else 0.0

  val previewIdxs = ArrayList<Int>(limit * limit)
  for (r in 0 until limit) for (c in 0 until limit) previewIdxs.add(r * numStates + c)
  val previewVals = countsBuf.readIndices(previewIdxs)

  val w = numNTs.toString().length.coerceAtLeast(2)
  val sb = StringBuilder()

  sb.append("--- UT Sparsity: ${sparsity.toString().take(8)}% ")
  sb.append("($totalActiveNTs / $maxPossibleActive active NTs) ---\n")

  sb.append("Active NTs per cell (k/$numNTs), showing ${limit}x$limit (upper triangle):\n")
  for (r in 0 until limit) {
    for (c in 0 until limit) {
      val k = previewVals[r * limit + c]
      if (c <= r) sb.append(" ".repeat(w)).append("  ")
      else sb.append(k.toString().padStart(w, ' ')).append("  ")
    }
    sb.append('\n')
  }
  log(sb.toString())

  uniBuf.destroy()
  countsBuf.destroy()
}

//language=wgsl
val active_nt_count by Shader("""
struct Uni { n: u32, nt: u32 };

@group(0) @binding(0) var<storage, read>       dp_in  : array<u32>;
@group(0) @binding(1) var<storage, read_write> outCnt : array<u32>; // length n*n
@group(0) @binding(2) var<uniform>             uni    : Uni;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  let c = gid.y;
  let n = uni.n;
  let nt = uni.nt;
  if (r >= n || c >= n) { return; }
  if (c <= r) { outCnt[r*n + c] = 0u; return; }
  let base = r * n * nt + c * nt;
  var k: u32 = 0u;
  for (var a: u32 = 0u; a < nt; a = a + 1u) { if (dp_in[base + a] != 0u) { k = k + 1u; } }
  outCnt[r*n + c] = k;
}""".trimIndent())

suspend fun completeCode(cfg: CFG, porous: List<String>, ngrams: GPUBuffer? = null): List<String> {
  val t0 = TimeSource.Monotonic.markNow()

  val fsa: FSA = makePorousFSA(porous)
  val codePoints = porousToCodePoints(cfg, porous)

  log("Made porousFSA(|Q|=${fsa.numStates}, width=${fsa.width}) in ${t0.elapsedNow()}")

  return completePipeline(cfg, fsa, ngrams, codePoints)
    .also { log("Received: ${it.size} completions in ${t0.elapsedNow()} (round trip)") }
}

suspend fun completePipeline(cfg: CFG, fsa: FSA, ngrams: GPUBuffer?, codePoints: IntArray): List<String> {
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("Porous FSA(|Q|=$numStates), ${cfg.calcStats()}")

  val metaBuf = packMetadata(cfg, fsa)

  val tmBuf   = cfg.termBuf
  val wordBuf = codePoints.toGPUBuffer()
  val totalSize = numStates * numStates * numNTs

  val dpBuf = Shader.createParseChart(STCPSD, totalSize)

  init_chart_line(dpBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  log("Chart construction took: ${t0.elapsedNow()} / dpBuf: ${dpBuf.size} bytes")

  cfl_mul_upper.invokeCFLFixpoint(numStates, numNTs, dpBuf, metaBuf)
  log("Matrix closure reached in: ${t0.elapsedNow()}")

//  logActiveNTGrid(dpBuf, numStates, numNTs, limit = minOf(48, numStates))

  val startNT = cfg.bindex[START_SYMBOL]

  // For a chain, finalIdxs should contain just the end state (n,0)
  val allStartIds = listOf(fsa.finalIdxs[0] * numNTs + startNT)

  if (allStartIds.isEmpty()) {
    log("No valid completion found: dpComplete has no entries in final states!")
    listOf(metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList()
  }

  val startIdxs = allStartIds + 0
  val maxRepairLen = fsa.width + 10
  if (MAX_WORD_LEN < maxRepairLen) {
    log("Max completion length exceeded $MAX_WORD_LEN ($maxRepairLen)")
    listOf(metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList()
  }

  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNTs, dpBuf, metaBuf)

  val lsDense  = buildLanguageSizeBuf(numStates, numNTs, dpBuf, metaBuf, tmBuf)
  val totalExp = bpStorageBuf.size.toInt() / (2 * 4)
  val cdfBuf   = GPUBuffer(totalExp * 4, STCPSD)
  ls_cdf(dpBuf, lsDense, bpOffsetBuf, cdfBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  lsDense.destroy()

  val numRoots  = startIdxs.size / 2
  val rootSizes = GPUBuffer(numRoots * 4, STCPSD)

  val idxUniBuf = packStruct(
    listOf(0, maxRepairLen, numNTs, numStates, DISPATCH_GROUP_SIZE_X, MAX_SAMPLES),
    startIdxs.toGPUBuffer()
  )

  build_root_sizes(dpBuf, bpCountBuf, bpOffsetBuf, cdfBuf, tmBuf, rootSizes, idxUniBuf)((numRoots + 255) / 256)
  val rootCDF = Shader.prefixSumGPU(rootSizes, numRoots)

  suspend fun langIntSize(): Long {
    fun u(x: Int) = x.toUInt().toLong()
    val last = numRoots - 1
    val lastSize = if (numRoots > 0) rootSizes.readIndices(listOf(last))[0] else 0
    val lastCDF  = if (numRoots > 0) rootCDF.readIndices(listOf(last))[0] else 0
    return if (numRoots > 0) u(lastCDF) + u(lastSize) else 0L
  }

//  val langSize = langIntSize()
  val toDecode = 100_000
  idxUniBuf.writeU32(wordIndex = 5, value = toDecode)

  val outBuf = GPUBuffer(toDecode * maxRepairLen * 4, STCPSD)
  enum_words_wor(
    dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf,
    cdfBuf, tmBuf, idxUniBuf, rootSizes, rootCDF, outBuf
  )(DISPATCH_GROUP_SIZE_X, (toDecode + DISPATCH_GROUP_SIZE_X - 1) / DISPATCH_GROUP_SIZE_X)

  return (
      if (ngrams != null) ngramDecoder(outBuf, ngrams, maxRepairLen, cfg, toDecode)
      else uniformDecoder(outBuf, cfg, maxRepairLen, toDecode)
      ).also {
      listOf(outBuf, rootSizes, rootCDF, metaBuf, dpBuf, idxUniBuf, cdfBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf)
        .forEach(GPUBuffer::destroy)
    }
}

// Checks whether there is a forward completion in the language of the CFG
suspend fun checkSuffix(cfg: CFG, tokens: List<String>, suffixLen: Int = 20): List<Int> {
  val t0 = TimeSource.Monotonic.markNow()

  val porousTks = tokens + List(suffixLen) { "_" }
  val fsa: FSA = makePorousFSA(porousTks)
  val codePoints = porousToCodePoints(cfg, porousTks)

  log("Made porousFSA(|Q|=${fsa.numStates}, width=${fsa.width}) in ${t0.elapsedNow()}")

  return checkSuffixPipeline(cfg, fsa, suffixLen, codePoints).also { log("Checked suffix completions in ${t0.elapsedNow()} (round trip)") }
}

suspend fun checkSuffixPipeline(cfg: CFG, fsa: FSA, suffixLen: Int, codePoints: IntArray): List<Int> {
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("Porous FSA(|Q|=$numStates), ${cfg.calcStats()}")

  val metaBuf = packMetadata(cfg, fsa)

  val tmBuf   = cfg.termBuf
  val wordBuf = codePoints.toGPUBuffer()
  val totalSize = numStates * numStates * numNTs

  val dpBuf = Shader.createParseChart(STCPSD, totalSize)

  init_chart_line(dpBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  log("Chart construction took: ${t0.elapsedNow()} / dpBuf: ${dpBuf.size} bytes")

  cfl_mul_upper.invokeCFLFixpoint(numStates, numNTs, dpBuf, metaBuf)
  log("Matrix closure reached in: ${t0.elapsedNow()}")

  val startNT = cfg.bindex[START_SYMBOL]

  val baseLen = codePoints.size - suffixLen
  val queryIndices = ArrayList<Int>(suffixLen + 1)

  for (k in 0..suffixLen) {
    val targetState = baseLen + k
    val flatIndex = targetState * numNTs + startNT
    queryIndices.add(flatIndex)
  }

  val reachability = dpBuf.readIndices(queryIndices)

  val listSuffixes = reachability.withIndex().filter { it.value != 0 }.map { it.index }

  listOf(metaBuf, dpBuf, wordBuf).forEach(GPUBuffer::destroy)
  return listSuffixes
}

fun makePorousFSA(tokens: List<String>): FSA {
  val n = tokens.size
  val digits = (n + 1).toString().length

  fun pd(i: Int) = i.toString().padStart(digits, '0')
  fun st(i: Int) = "q_${pd(i)}/${pd(0)}"

  val arcs: TSA = (0 until n).map { i ->
    val lbl = tokens[i]
    Triple(st(i), lbl, st(i + 1))
  }.toSet()

  val initialStates = setOf(st(0))
  val finalStates   = setOf(st(n))

  return AFSA(arcs, initialStates, finalStates)
    .also { it.width = n; it.height = 0; it.levString = tokens }
}

private const val HOLE_SENTINEL_INT: Int = -1 // 0xFFFF_FFFFu on GPU

fun porousToCodePoints(cfg: CFG, porous: List<String>): IntArray =
  IntArray(porous.size) { i ->
    val t = porous[i]
    if (t == "_") HOLE_SENTINEL_INT
    else cfg.tmMap[t] ?: error("Unknown token '$t' (not in cfg.tmMap)")
  }

/**
 * LOW-RANK CFG-NFA TENSOR INTERSECTION EXPERIMENT
 */

//language=wgsl
val RANK1_STRUCT = """
struct Rank1Data {
    K: u32, numWords: u32, pad1: u32, pad2: u32,
    u_off: u32, u_len: u32,
    v_off: u32, v_len: u32,
    w_off: u32, w_len: u32,
    payload: array<u32>
};
fn getU(k: u32, w: u32) -> u32 { return r1.payload[r1.u_off + k * r1.numWords + w]; }
fn getV(k: u32, w: u32) -> u32 { return r1.payload[r1.v_off + k * r1.numWords + w]; }
fn getW(k: u32, w: u32) -> u32 { return r1.payload[r1.w_off + k * r1.numWords + w]; }
"""

//language=wgsl
val r1_update_feats by Shader("""
$RANK1_STRUCT
struct Params { N: u32, featWords: u32 }; 
@group(0) @binding(0) var<storage, read>       packed_dp : array<u32>;
@group(0) @binding(1) var<storage, read_write> row_feats : array<atomic<u32>>; // Use atomic for ORing
@group(0) @binding(2) var<storage, read_write> col_feats : array<atomic<u32>>; 
@group(0) @binding(3) var<storage, read>              r1 : Rank1Data;
@group(0) @binding(4) var<uniform>                   uni : Params;

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let q = gid.y;
    if (p >= uni.N || q >= uni.N) { return; }

    let cellIdx = (p * uni.N + q) * r1.numWords;
    
    for (var k = 0u; k < r1.K; k++) {
        // 1. Check U
        var matchU = false;
        for (var w = 0u; w < r1.numWords; w++) {
            if ((packed_dp[cellIdx + w] & getU(k, w)) != 0u) { matchU = true; break; }
        }
        if (matchU) {
            let fw = q / 32u;
            let fb = q % 32u;
            let idx = p * r1.K * uni.featWords + k * uni.featWords + fw;
            atomicOr(&row_feats[idx], (1u << fb));
        }

        // 2. Check V
        var matchV = false;
        for (var w = 0u; w < r1.numWords; w++) {
            if ((packed_dp[cellIdx + w] & getV(k, w)) != 0u) { matchV = true; break; }
        }
        if (matchV) {
            let fw = p / 32u;
            let fb = p % 32u;
            let idx = q * r1.K * uni.featWords + k * uni.featWords + fw;
            atomicOr(&col_feats[idx], (1u << fb));
        }
    }
}
""")

//language=wgsl
val r1_project by Shader("""
$RANK1_STRUCT
struct Params { N: u32, featWords: u32 };
@group(0) @binding(0) var<storage, read>       packed_dp : array<u32>;
@group(0) @binding(1) var<storage, read_write> row_feats : array<u32>; // [K][N][Words]
@group(0) @binding(2) var<storage, read_write> col_feats : array<u32>; // [K][N][Words]
@group(0) @binding(3) var<storage, read>              r1 : Rank1Data;
@group(0) @binding(4) var<uniform>                   uni : Params;

// Thread (idx, k) builds the entire bit-vector for Row 'idx' and Col 'idx' for component 'k'
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; // Represents Row p (for Left) and Col q (for Right)
    let k   = gid.y; // Component index
    if (idx >= uni.N || k >= r1.K) { return; }

    let N = uni.N;
    // Base offset for this (k, idx) pair in the feature matrices
    let featBase = (k * N + idx) * uni.featWords;

    // Loop through the chart words to build the bit-vector locally
    for (var fw = 0u; fw < uni.featWords; fw++) {
        var l_bits = 0u; // Accumulator for RowFeats[idx][k]
        var r_bits = 0u; // Accumulator for ColFeats[idx][k]
        
        let bitOffset = fw * 32u;
        
        for (var b = 0u; b < 32u; b++) {
            let j = bitOffset + b; // The 'scanning' coordinate
            if (j >= N) { break; }

            // 1. Build Row Feature: Does DP[idx, j] match U_k?
            let dpIdx_L = (idx * N + j) * r1.numWords;
            var matchL = false;
            for (var w = 0u; w < r1.numWords; w++) {
                if ((packed_dp[dpIdx_L + w] & getU(k, w)) != 0u) { matchL = true; break; }
            }
            if (matchL) { l_bits |= (1u << b); }

            // 2. Build Col Feature: Does DP[j, idx] match V_k?
            // Note: Accessing DP in column-major fashion (Row j, Col idx)
            let dpIdx_R = (j * N + idx) * r1.numWords;
            var matchR = false;
            for (var w = 0u; w < r1.numWords; w++) {
                if ((packed_dp[dpIdx_R + w] & getV(k, w)) != 0u) { matchR = true; break; }
            }
            if (matchR) { r_bits |= (1u << b); }
        }

        // Write once, non-atomically. Each thread owns this memory slot.
        row_feats[featBase + fw] = l_bits;
        col_feats[featBase + fw] = r_bits;
    }
}
""")

//language=wgsl
val r1_combine by Shader("""
$RANK1_STRUCT
struct Params { N: u32, featWords: u32 };
struct AtomicFlag { changed: atomic<u32> };

@group(0) @binding(0) var<storage, read_write> packed_dp : array<u32>;
@group(0) @binding(1) var<storage, read>       row_feats : array<u32>;
@group(0) @binding(2) var<storage, read>       col_feats : array<u32>;
@group(0) @binding(3) var<storage, read>              r1 : Rank1Data;
@group(0) @binding(4) var<uniform>                   uni : Params;
@group(0) @binding(5) var<storage, read_write>      flag : AtomicFlag;

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let q = gid.y;
    if (p >= uni.N || q >= uni.N) { return; }

    let cellBase = (p * uni.N + q) * r1.numWords;
    var changedLocal = false;

    for (var k = 0u; k < r1.K; k++) {
        // Intersect: row_feats[k][p] AND col_feats[k][q]
        // Layout: [K][N][Words]
        let rfBase = (k * uni.N + p) * uni.featWords;
        let cfBase = (k * uni.N + q) * uni.featWords;
        
        var overlap = false;
        for (var fw = 0u; fw < uni.featWords; fw++) {
            if ((row_feats[rfBase + fw] & col_feats[cfBase + fw]) != 0u) {
                overlap = true; break;
            }
        }

        if (overlap) {
            for (var w = 0u; w < r1.numWords; w++) {
                let mask = getW(k, w);
                // Check if adding new bits
                if ((packed_dp[cellBase + w] & mask) != mask) {
                    packed_dp[cellBase + w] |= mask;
                    changedLocal = true;
                }
            }
        }
    }
    
    if (changedLocal) { atomicStore(&flag.changed, 1u); }
}
""")

//language=wgsl
val compress_chart by Shader("""
struct Params { numStates: u32, numNTs: u32, numWords: u32 };
@group(0) @binding(0) var<storage, read>       exploded : array<u32>;
@group(0) @binding(1) var<storage, read_write>   packed : array<u32>;
@group(0) @binding(2) var<uniform>                  uni : Params;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numCells = uni.numStates * uni.numStates;
    let idx = gid.x + gid.y * 65535u;
    if (idx >= numCells) { return; }

    let r = idx / uni.numStates;
    let c = idx % uni.numStates;

    // Base offset in Exploded: (r * N + c) * NT
    let explBase = idx * uni.numNTs;
    // Base offset in Packed:   (r * N + c) * Words
    let packBase = idx * uni.numWords;

    // Naive bit packing
    for (var w = 0u; w < uni.numWords; w++) {
        var wordVal = 0u;
        for (var b = 0u; b < 32u; b++) {
            let nt = w * 32u + b;
            if (nt < uni.numNTs) {
                // Check if exploded[explBase + nt] is active (non-zero)
                if (exploded[explBase + nt] != 0u) {
                    wordVal |= (1u << b);
                }
            }
        }
        packed[packBase + w] = wordVal;
    }
}
""")

//language=wgsl
val expand_chart by Shader("""
struct Params { numStates: u32, numNTs: u32, numWords: u32 };
@group(0) @binding(0) var<storage, read>         packed : array<u32>;
@group(0) @binding(1) var<storage, read_write> exploded : array<u32>;
@group(0) @binding(2) var<uniform>                  uni : Params;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numCells = uni.numStates * uni.numStates;
    let idx = gid.x + gid.y * 65535u;
    if (idx >= numCells) { return; }

    let explBase = idx * uni.numNTs;
    let packBase = idx * uni.numWords;

    for (var nt = 0u; nt < uni.numNTs; nt++) {
        let w = nt / 32u;
        let b = nt % 32u;
        let isSet = (packed[packBase + w] & (1u << b)) != 0u;
        
        if (isSet) { exploded[explBase + nt] |= 1u; }
    }
}
""")

fun LongArray.toIntArray32(): IntArray {
  val res = IntArray(size * 2)
  for (i in indices) {
    val l = this[i]
    res[i * 2]     = l.toUInt().toInt()         // Low 32 bits
    res[i * 2 + 1] = (l shr 32).toUInt().toInt() // High 32 bits
  }
  return res
}

val CFG.rank1Buf: GPUBuffer by cache {
  val components = rankOneDecomposition
  val K = components.size
  // Calculate 32-bit words per mask.
  val numWords = ((nonterminals.size + 63) / 64) * 2

  val flatU = IntArray(K * numWords)
  val flatV = IntArray(K * numWords)
  val flatW = IntArray(K * numWords)

  for (k in 0 until K) {
    val comp = components[k]
    val u32 = comp.U.toIntArray32()
    val v32 = comp.V.toIntArray32()
    val w32 = comp.W.toIntArray32()

    // Use Kotlin's copyInto
    val len = u32.size.coerceAtMost(numWords)
    u32.copyInto(flatU, destinationOffset = k * numWords, startIndex = 0, endIndex = len)
    v32.copyInto(flatV, destinationOffset = k * numWords, startIndex = 0, endIndex = len)
    w32.copyInto(flatW, destinationOffset = k * numWords, startIndex = 0, endIndex = len)
  }

  /* Memory Layout: [K, numWords, 0, 0 | U... | V... | W... ] */
  val header = listOf(K, numWords, 0, 0)

  packStruct(
    header,
    flatU.toGPUBuffer(STCPSD),
    flatV.toGPUBuffer(STCPSD),
    flatW.toGPUBuffer(STCPSD)
  )
}

suspend fun invokeCFLFixpointFast(cfg: CFG, numStates: Int, numNTs: Int, dpIn: GPUBuffer, metaBuf: GPUBuffer) {
  val t0 = TimeSource.Monotonic.markNow()

  // --- 1. Setup Buffers ---
  val packWords = (numNTs + 31) / 32
  val featWords = (numStates + 31) / 32
  val K         = cfg.rankOneDecomposition.size

  // Compact Chart Buffer
  val packedBuf = GPUBuffer(numStates * numStates * packWords * 4, STCPSD)
  val uniBuf    = intArrayOf(numStates, numNTs, packWords)
    .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  // Compress Initial State (Literals -> Packed Bits)
  compress_chart(dpIn, packedBuf, uniBuf)((numStates * numStates + 255) / 256)

  // Feature Buffers: Size [K][N][featWords]
  // Note: We do NOT need to clear these every round because r1_project overwrites them completely.
  val featBufSize = K * numStates * featWords * 4
  val rowFeats    = GPUBuffer(featBufSize, STCPSD)
  val colFeats    = GPUBuffer(featBufSize, STCPSD)

  val rank1Buf = cfg.rank1Buf
  val r1Uni    = intArrayOf(numStates, featWords)
    .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  val flagBuf  = 0.toGPUBuffer()
  val zeroFlag = Int32Array<ArrayBuffer>(1).apply { set(arrayOf(0), 0) }

  // --- 2. Fixpoint Loop ---
  // Project: Threads (N, K)
  val projX = (numStates + 15) / 16
  val projY = (K + 15) / 16
  // Combine: Threads (N, N)
  val combX = (numStates + 15) / 16
  val combY = (numStates + 15) / 16

  var rounds = 0
  while (rounds < numStates) {
    // Reset convergence flag
    gpu.queue.writeBuffer(flagBuf, 0.0, zeroFlag)

    // Step A: Project (Gather Features) - No Atomics
    r1_project(packedBuf, rowFeats, colFeats, rank1Buf, r1Uni)(projX, projY)

    // Step B: Combine (Gemm-like update)
    r1_combine(packedBuf, rowFeats, colFeats, rank1Buf, r1Uni, flagBuf)(combX, combY)

    // Check convergence
    if (flagBuf.readIndices(listOf(0))[0] == 0) break
    rounds++
  }

  // --- 3. Expand Result ---
  // Ensure we use the FIXED expand_chart kernel that sets LSB |= 1u
  expand_chart(packedBuf, dpIn, uniBuf)((numStates * numStates + 255) / 256)

  // Cleanup
  listOf(packedBuf, rowFeats, colFeats, uniBuf, r1Uni, flagBuf).forEach { it.destroy() }

  log("GPU Rank-1 Fixpoint converged in $rounds rounds, ${t0.elapsedNow()}")
}