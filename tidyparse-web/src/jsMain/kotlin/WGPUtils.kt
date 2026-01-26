import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.buildLanguageSizeBuf
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import Shader.Companion.writeU32
import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.START_SYMBOL
import ai.hypergraph.kaliningraph.parsing.bindex
import ai.hypergraph.kaliningraph.parsing.calcStats
import ai.hypergraph.kaliningraph.parsing.makePorousFSA
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.parsing.porousToCodePoints
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

  val idxs = ArrayList<Int>(limit * limit)
  for (r in 0 until limit) for (c in 0 until limit) idxs.add(r * numStates + c)
  val vals = countsBuf.readIndices(idxs)

  val w = numNTs.toString().length.coerceAtLeast(2)
  val sb = StringBuilder()
  sb.append("Active NTs per cell (k/$numNTs), showing ${limit}x$limit (upper triangle):\n")
  for (r in 0 until limit) {
    for (c in 0 until limit) {
      val k = vals[r * limit + c]
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

//  logActiveNTGrid(dpBuf, numStates, numNTs, limit = minOf(48, numStates)) // pick your window

  val startNT = cfg.bindex[START_SYMBOL]

  // For a chain, finalIdxs should contain just the end state (n,0)
  val allStartIds = listOf(fsa.finalIdxsq[0] * numNTs + startNT)

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

const val MAX_SUFF_LEN = 20

// Checks whether there is a forward completion in the language of the CFG
suspend fun checkSuffix(cfg: CFG, tokens: List<String>): List<Int> {
  val t0 = TimeSource.Monotonic.markNow()

  val porousTks = tokens + List(MAX_SUFF_LEN) { "_" }
  val fsa: FSA = makePorousFSA(porousTks)
  val codePoints = porousToCodePoints(cfg, porousTks)

  log("Made porousFSA(|Q|=${fsa.numStates}, width=${fsa.width}) in ${t0.elapsedNow()}")

  return checkSuffixPipeline(cfg, fsa, codePoints).also { log("Checked suffix completions in ${t0.elapsedNow()} (round trip)") }
}

suspend fun checkSuffixPipeline(cfg: CFG, fsa: FSA, codePoints: IntArray): List<Int> {
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

  val baseLen = codePoints.size - MAX_SUFF_LEN
  val queryIndices = ArrayList<Int>(MAX_SUFF_LEN + 1)

  for (k in 0..MAX_SUFF_LEN) {
    val targetState = baseLen + k
    val flatIndex = targetState * numNTs + startNT
    queryIndices.add(flatIndex)
  }

  val reachability = dpBuf.readIndices(queryIndices)

  val listSuffixes = reachability.withIndex().filter { it.value != 0 }.map { it.index }

  listOf(metaBuf, dpBuf, wordBuf).forEach(GPUBuffer::destroy)
  return listSuffixes
}