import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.buildLanguageSizeBuf
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import Shader.Companion.writeU32
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import web.gpu.GPUBuffer
import kotlin.time.TimeSource

suspend fun logActiveNTGrid(
  activeBuf: GPUBuffer,
  numStates: Int,
  numNTs: Int,
  limit: Int = minOf(32, numStates)
) {
  val activeWords = (numNTs + 31) ushr 5
  val countsBuf = Shader.GPUBuffer(numStates.toLong() * numStates * 4L, GPUBufferUsage.STCPSD)
  val uniBuf = intArrayOf(numStates, activeWords).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  val groupsX = (numStates + 7) / 8
  val groupsY = (numStates + 7) / 8
  active_nt_count(activeBuf, countsBuf, uniBuf)(groupsX, groupsY, 1)

  val allIndices = (0 until numStates * numStates).toList()
  val allVals = countsBuf.readIndices(allIndices)

  val totalActiveNTs = allVals.fold(0L) { acc, i -> acc + i }
  val totalUTCells = (numStates.toLong() * (numStates - 1)) / 2
  val maxPossibleActive = totalUTCells * numNTs

  val sparsity =
    if (maxPossibleActive > 0) (totalActiveNTs.toDouble() / maxPossibleActive) * 100.0
    else 0.0

  val previewIdxs = ArrayList<Int>(limit * limit)
  for (r in 0 until limit) for (c in 0 until limit) previewIdxs += r * numStates + c
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
struct Uni { n: u32, activeWords: u32 };

@group(0) @binding(0) var<storage, read>       active_nts : array<u32>;
@group(0) @binding(1) var<storage, read_write> outCnt     : array<u32>; // length n*n
@group(0) @binding(2) var<uniform>             uni        : Uni;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  let c = gid.y;
  let n = uni.n;
  let aw = uni.activeWords;

  if (r >= n || c >= n) { return; }
  if (c <= r) { outCnt[r*n + c] = 0u; return; }

  let base = (r * n + c) * aw;
  var k: u32 = 0u;
  for (var w: u32 = 0u; w < aw; w = w + 1u) { k += countOneBits(active_nts[base + w]); }
  outCnt[r*n + c] = k;
}""")

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

  val tmBuf       = cfg.termBuf
  val wordBuf     = codePoints.toGPUBuffer()
  val totalSize   = numStates * numStates * numNTs
  val activeWords = (numNTs + 31) ushr 5

  val dpBuf     = Shader.createParseChart(STCPSD, totalSize)
  val activeBuf = GPUBuffer((numStates * numStates * activeWords * 4).toLong(), STCPSD)

  init_chart_line(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  log("Chart construction took: ${t0.elapsedNow()}")

  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  log("Matrix closure reached in: ${t0.elapsedNow()}")

//  logActiveNTGrid(activeBuf, numStates, numNTs, limit = minOf(48, numStates))

  val startNT = cfg.bindex[START_SYMBOL]
  val rootQuery = fsa.finalIdxs.map { it * numNTs + startNT }

  val allStartIds = rootQuery
    .zip(dpBuf.readIndices(rootQuery))
    .filter { (_, v) -> v != 0 }
    .map { it.first }

  if (allStartIds.isEmpty()) {
    log("No valid completion found: dpComplete has no entries in final states!")
    listOf(activeBuf, metaBuf, dpBuf, wordBuf).forEach(GPUBuffer::destroy)
    return emptyList()
  }

  val startIdxs = allStartIds.flatMap { listOf(it, 0) }
  val maxRepairLen = fsa.width + 10

  if (MAX_WORD_LEN < maxRepairLen) {
    log("Max completion length exceeded $MAX_WORD_LEN ($maxRepairLen)")
    listOf(activeBuf, metaBuf, dpBuf, wordBuf).forEach(GPUBuffer::destroy)
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
      listOf(
        outBuf, rootSizes, rootCDF, metaBuf, dpBuf, activeBuf, wordBuf,
        idxUniBuf, cdfBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf
      ).forEach(GPUBuffer::destroy)
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

  val tmBuf       = cfg.termBuf
  val wordBuf     = codePoints.toGPUBuffer()
  val totalSize   = numStates * numStates * numNTs
  val activeWords = (numNTs + 31) ushr 5

  val dpBuf     = Shader.createParseChart(STCPSD, totalSize)
  val activeBuf = GPUBuffer((numStates * numStates * activeWords * 4).toLong(), STCPSD)

  init_chart_line(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  log("Chart construction took: ${t0.elapsedNow()}")

  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  log("Matrix closure reached in: ${t0.elapsedNow()}")

  val startNT = cfg.bindex[START_SYMBOL]

  val baseLen = codePoints.size - suffixLen
  val queryIndices = ArrayList<Int>(suffixLen + 1)
  for (k in 0..suffixLen) {
    val targetState = baseLen + k
    queryIndices += targetState * numNTs + startNT
  }

  val reachability = dpBuf.readIndices(queryIndices)
  val listSuffixes = reachability.withIndex().filter { it.value != 0 }.map { it.index }

  listOf(activeBuf, metaBuf, dpBuf, wordBuf).forEach(GPUBuffer::destroy)
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

fun Map<String, Int>.logTimingsToJSConsole() {
  val totalMs = this["total"]?.coerceAtLeast(0)
  val bodyRows = entries
    .asSequence()
    .filter { it.key != "total" }
    .map { (k, v) -> k.ifBlank { "<unnamed>" } to v.coerceAtLeast(0) }
    .sortedByDescending { it.second }
    .toList()

  val total = totalMs ?: bodyRows.sumOf { it.second }.coerceAtLeast(1)
  val maxMs = bodyRows.maxOf { it.second }.coerceAtLeast(1)

  val maxLabel = 28
  val barWidth = 36

  fun String.compactLabel(): String =
    replace(Regex("\\s+"), " ")
      .let { if (it.length <= maxLabel) it else it.take(maxLabel - 1) + "…" }

  fun Int.bar(): String {
    val n = ((this.toDouble() / maxMs) * barWidth).toInt().coerceIn(1, barWidth)
    return "#".repeat(n)
  }

  val rowTexts = bodyRows.map { (label, ms) ->
    val pct = (100.0 * ms / total.coerceAtLeast(1)).toInt()
    buildString {
      append(label.compactLabel().padEnd(maxLabel))
      append(" |")
      append(ms.bar().padEnd(barWidth))
      append("| ")
      append(ms.toString().padStart(6))
      append(" ms  ")
      append(pct.toString().padStart(3))
      append('%')
    }
  }

  val titleText = "─ Timings (${bodyRows.size} steps, total=${total}ms) "
  val contentWidth = maxOf(titleText.length, rowTexts.maxOf { it.length })

  val plot = buildString {
    appendLine("┌" + titleText.padEnd(contentWidth, '─') + "┐")
    rowTexts.forEach { row ->
      appendLine("│" + row.padEnd(contentWidth, ' ') + "│")
    }
    appendLine("└" + "─".repeat(contentWidth) + "┘")
  }

  println(plot)
}