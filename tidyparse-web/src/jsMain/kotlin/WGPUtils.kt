import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.buildLanguageSizeBuf
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import Shader.Companion.writeU32
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.parsing.bindex
import ai.hypergraph.kaliningraph.parsing.leftAdj
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.repair.pythonStatementCNFAllProds
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import ai.hypergraph.kaliningraph.types.cache
import ai.hypergraph.tidyparse.MAX_DISP_RESULTS
import ai.hypergraph.tidyparse.PyCodeSnippet
import web.gpu.GPUBuffer
import kotlin.time.TimeMark
import kotlin.time.TimeSource

data class GrammarEncoding(val flat: IntArray, val offsets: IntArray)
// leftAdjGrouped[B] = triples:
//   (C, parentWord, parentMask)
// where parentMask contains all A in parentWord such that A -> B C.
data class GroupedLeftAdjEncoding(val flat: IntArray, val offsets: IntArray)

val CFG.groupedLeftAdjEncoding: GroupedLeftAdjEncoding by cache {
  val ladj = leftAdj
  val W = ladj.size

  // For each B, group scalar leftAdj entries by (C, parentWord).
  // key = (C << 32) | parentWord
  // value = mask of all parent A bits in that parentWord.
  val rows = Array(W) { linkedMapOf<Long, Int>() }

  var B = 0
  while (B < W) {
    val adj = ladj[B]

    if (adj != null) {
      val cs = adj.other
      val asz = adj.aIdx

      var i = 0
      while (i < cs.size) {
        val C = cs[i]
        val A = asz[i]

        val parentWord = A ushr 5
        val parentMask = 1 shl (A and 31)

        val key =
          (C.toLong() shl 32) or
              (parentWord.toLong() and 0xffffffffL)

        rows[B][key] = (rows[B][key] ?: 0) or parentMask

        i++
      }
    }

    B++
  }

  val offsets = IntArray(W + 1)
  var total = 0

  B = 0
  while (B < W) {
    offsets[B] = total
    total += rows[B].size * 3
    B++
  }

  offsets[W] = total

  val flat = IntArray(total)
  var out = 0

  B = 0
  while (B < W) {
    for ((key, mask) in rows[B]) {
      val C = (key ushr 32).toInt()
      val parentWord = (key and 0xffffffffL).toInt()

      flat[out++] = C
      flat[out++] = parentWord
      flat[out++] = mask
    }

    B++
  }

  GroupedLeftAdjEncoding(flat, offsets)
}

val CFG.grammarEncoding: GrammarEncoding by cache {
  val W = nonterminals.size
  val ntIdx = bindex.ntIndices   // Map<Σᐩ, Int>

  val counts = IntArray(W)
  for ((lhs, rhs) in this) {
    if (rhs.size != 2) continue
    val a = ntIdx[lhs] ?: continue
    val b = ntIdx[rhs[0]] ?: continue
    val c = ntIdx[rhs[1]] ?: continue
    counts[a] += 2
  }

  val offsets = IntArray(W + 1)
  var acc = 0
  for (i in 0 until W) { offsets[i] = acc; acc += counts[i] }
  offsets[W] = acc

  val flat = IntArray(acc)
  val cur = offsets.copyOf()
  for ((lhs, rhs) in this) {
    if (rhs.size != 2) continue
    val a = ntIdx[lhs] ?: continue
    val b = ntIdx[rhs[0]] ?: continue
    val c = ntIdx[rhs[1]] ?: continue
    val p = cur[a]
    flat[p] = b
    flat[p + 1] = c
    cur[a] = p + 2
  }

  GrammarEncoding(flat, offsets)
}

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

  val sparsity = if (maxPossibleActive > 0) (totalActiveNTs.toDouble() / maxPossibleActive) * 100.0 else 0.0

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
  timings = linkedMapOf()
  val preprocT = TimeSource.Monotonic.markNow()

  val fsa: FSA = makePorousFSA(porous)
  val codePoints = porousToCodePoints(cfg, porous)

  log("Made porousFSA(|Q|=${fsa.numStates}, width=${fsa.width}) in ${preprocT.elapsedNow()}")
  mark("preprocessing", preprocT)

  return completePipeline(cfg, fsa, ngrams, codePoints)
    .also { log("Received: ${it.size} completions in ${preprocT.elapsedNow()} (round trip)") }
}

suspend fun completePipeline(cfg: CFG, fsa: FSA, ngrams: GPUBuffer?, codePoints: IntArray): List<String> {
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("Porous FSA(|Q|=$numStates), ${cfg.calcStats()}")

  val metaT = TimeSource.Monotonic.markNow()
  val metaBuf = packMetadata(cfg, fsa)
  mark("pack metadata", metaT)
  log("Packed metadata in ${timings["pack metadata"]}ms")

  val tmBuf       = cfg.termBuf
  val wordBuf     = codePoints.toGPUBuffer()
  val totalSize   = numStates * numStates * numNTs
  val activeWords = (numNTs + 31) ushr 5

  val dpBuf     = Shader.createParseChart(STCPSD, totalSize)
  val activeBuf = GPUBuffer((numStates * numStates * activeWords * 4).toLong(), STCPSD)

  timings["init chart"] = timedGPUIsolated("Init chart") {
    init_chart_line(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  }

  val closureT = TimeSource.Monotonic.markNow()
  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  mark("matrix closure", closureT)
  log("Matrix closure reached in: ${timings["matrix closure"]}ms")

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

  val bpT = TimeSource.Monotonic.markNow()
  val (bpCountBuf, bpOffsetBuf, bpStorageBuf) = Shader.buildBackpointers(numStates, numNTs, dpBuf, metaBuf)
  mark("build backpointers", bpT)
  val totalExp = bpStorageBuf.size.toInt() / (2 * 4)
  log("Built backpointers in ${timings["build backpointers"]}ms | expansions=$totalExp")

  val lsDenseT = TimeSource.Monotonic.markNow()
  val lsDense = buildLanguageSizeBuf(numStates, numNTs, dpBuf, metaBuf, tmBuf)
  mark("build ls dense", lsDenseT)
  log("Built lsDense in ${timings["build ls dense"]}ms (${lsDense.size}B)")

  val cdfBuf = GPUBuffer(totalExp * 4, STCPSD)
  timings["build cdf"] = timedGPUIsolated("Build CDF") {
    ls_cdf(dpBuf, lsDense, bpOffsetBuf, cdfBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  }
  lsDense.destroy()
  log("Pairing function construction took: ${timings["build cdf"]}ms (${cdfBuf.size}B)")

  val numRoots  = startIdxs.size / 2
  val rootSizes = GPUBuffer(numRoots * 4, STCPSD)

  val idxUniBuf = packStruct(
    listOf(0, maxRepairLen, numNTs, numStates, DISPATCH_GROUP_SIZE_X, MAX_SAMPLES),
    startIdxs.toGPUBuffer()
  )

  timings["build root sizes"] = timedGPUIsolated("Build root sizes (roots=$numRoots)") {
    build_root_sizes(dpBuf, bpCountBuf, bpOffsetBuf, cdfBuf, tmBuf, rootSizes, idxUniBuf)((numRoots + 255) / 256)
  }

  val rootCDFTime = TimeSource.Monotonic.markNow()
  val rootCDF = Shader.prefixSumGPU(rootSizes, numRoots)
  mark("prefix root cdf", rootCDFTime)
  log("Built root CDF in ${timings["prefix root cdf"]}ms (${rootCDF.size}B)")

  val toDecode = MAX_DISP_RESULTS
  idxUniBuf.writeU32(wordIndex = 5, value = toDecode)

  val outBuf = GPUBuffer(toDecode * maxRepairLen * 4, STCPSD)
  timings["enumerate"] = timedGPUIsolated("Enumerate") {
    enum_words_wor(
      dpBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf,
      cdfBuf, tmBuf, idxUniBuf, rootSizes, rootCDF, outBuf
    )(DISPATCH_GROUP_SIZE_X, (toDecode + DISPATCH_GROUP_SIZE_X - 1) / DISPATCH_GROUP_SIZE_X)
  }

  val decodeT = TimeSource.Monotonic.markNow()
  val result = uniformDecoder(outBuf, cfg, maxRepairLen, toDecode)
//  val result =
//    if (wdfa != null) wdfaDecoder(outBuf, wdfa!!, maxRepairLen, cfg, toDecode)
//    else uniformDecoder(outBuf, cfg, maxRepairLen, toDecode)
  mark("decode", decodeT)

  return result.also {
    listOf(
      outBuf, rootSizes, rootCDF, metaBuf, dpBuf, activeBuf, wordBuf,
      idxUniBuf, cdfBuf, bpCountBuf, bpOffsetBuf, bpStorageBuf
    ).forEach(GPUBuffer::destroy)
  }
}

// Checks whether there is a forward completion in the language of the CFG
suspend fun CFG.checkSuffix(tokens: List<String>, suffixLen: Int = 20): List<Int> {
  val t0 = TimeSource.Monotonic.markNow()

  val porousTks = tokens + List(suffixLen) { "_" }
  val fsa: FSA = makePorousFSA(porousTks)
  val codePoints = porousToCodePoints(this, porousTks)

  log("Made porousFSA(|Q|=${fsa.numStates}, width=${fsa.width}) in ${t0.elapsedNow()}")

  return checkSuffixPipeline(this, fsa, suffixLen, codePoints).also { log("Checked suffix completions in ${t0.elapsedNow()} (round trip)") }
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

  timings["init chart"] = timedGPUIsolated("Init chart") {
    init_chart_line(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  }

  val closureT = TimeSource.Monotonic.markNow()
  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  mark("matrix closure", closureT)
  log("Matrix closure reached in: ${timings["matrix closure"]}ms")

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

var timings = linkedMapOf<String, Int>()
fun mark(step: String, started: TimeMark) { timings[step] = started.elapsedNow().inWholeMilliseconds.toInt() }

fun Map<String, Int>.logTimesheet() {
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

  fun String.compactLabel(): String = replace(Regex("\\s+"), " ")
      .let { if (it.length <= maxLabel) it else it.take(maxLabel - 1) + "…" }

  fun Int.bar(): String = "#".repeat(((this.toDouble() / maxMs) * barWidth).toInt().coerceIn(1, barWidth))

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
    rowTexts.forEach { row -> appendLine("│" + row.padEnd(contentWidth, ' ') + "│") }
    appendLine("└" + "─".repeat(contentWidth) + "┘")
  }

  println(plot)
}

//language=wgsl
val wdfa_score_raw by Shader("""$SAMPLER_PARAMS
$WDFA_STRUCT

@group(0) @binding(0) var<storage, read_write> packets : array<u32>;
@group(0) @binding(1) var<storage, read>       wdfa    : WDFA;
@group(0) @binding(2) var<uniform>             prm     : Params;

const PKT_HDR_LEN : u32 = ${PKT_HDR_LEN}u;

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sid = gid.x + gid.y * prm.threads;
  if (sid >= prm.maxSamples) { return; }

  let stride = prm.stride;
  let base = sid * stride;

  var q = wdfa.startState;
  var cost = wdfa.startCost;

  var pos : u32 = 0u;
  loop {
    if (pos >= stride - PKT_HDR_LEN) { break; }

    let tok = packets[base + PKT_HDR_LEN + pos];
    if (tok == 0u) { break; }

    let e = find_edge(q, tok);
    if (e == NO_EDGE) {
      cost = sat_add_wdfa(cost, wdfa.missingCost);
    } else {
      cost = sat_add_wdfa(cost, edge_cost(e));
      q = edge_dst(e);
    }

    pos = pos + 1u;
  }

  let fc = final_cost(q);
  if (fc >= WDFA_INF) {
    cost = WDFA_INF;
  } else {
    cost = sat_add_wdfa(cost, fc);
  }

  // Apples-to-apples debug score: no edit-distance penalty.
  packets[base + 1u] = cost;
}""")

suspend fun debugWDFATokenIndexing(cfg: CFG = pythonStatementCNFAllProds, wdfaBuf: GPUBuffer? = wdfa) {
  log("Debugging WDFA token indexing...")
  if (wdfaBuf == null) return log("WDFA rank testing skipped: buffer not loaded")

  val lines = """
    x = 1 + 1
    [ i for i in j ]
    stripped_lines = lambda f : (l.rstrip("\n") for l in f)
    newlist = [word for word in words if len(word) == 9]
    Keys = [x for x in d if d[x] == 'a']
    gen_fun = lambda num: (x for u in range(num) for x in (u*2, u*10, u*u))
    [2 * x if x > 2 else add_nothing_to_list for x in some_list]
    lines = [[float(x) for x in line] for line in csv.reader(f)]
    filtered = [x for x in common if x in words]
    () + (1, 'a') + (2, 'b') + (3, 'c')
  """.lines().map { it.trim() }.filter { it.isNotBlank() }

  val packets = IntArray(lines.size * MAX_WORD_LEN)
  val decoded = mutableListOf<List<String>>()

  lines.forEachIndexed { i, line ->
    val toks = PyCodeSnippet(line).lexedTokens().tokenizeByWhitespace().filter { it in cfg.terminals }

    decoded += toks

    val base = i * MAX_WORD_LEN
    packets[base + 0] = 0 // edit distance, ignored by wdfa_score_raw
    packets[base + 1] = 0 // score filled by wdfa_score_raw

    toks.forEachIndexed { j, tok ->
      val tid = cfg.tmMap[tok] ?: error("Token not in cfg.tmMap: $tok from $line")

      // Packet convention: 0 = terminator, local terminal i = i + 1.
      packets[base + PKT_HDR_LEN + j] = tid + 1
    }

    packets[base + PKT_HDR_LEN + toks.size] = 0
  }

  val packetBuf = packets.toGPUBuffer(STCPSD)
  val prmBuf = intArrayOf(
    lines.size,              // maxSamples
    lines.size,              // k, unused
    MAX_WORD_LEN,            // stride
    DISPATCH_GROUP_SIZE_X
  ).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  wdfa_score_raw(packetBuf, wdfaBuf, prmBuf)(DISPATCH_GROUP_SIZE_X, 1)

  val scored = packetBuf.readJSIntArray()
  val rows = lines.indices.map { lines[it] to scored[it * MAX_WORD_LEN + 1].toUInt().toLong() }
  val sorted = rows.sortedByDescending { it.second }

  log("GPU WDFA sorted order:")
  sorted.forEachIndexed { rank, row -> log("${rank + 1}. ${row.first} // gpu=${row.second}") }

  val expected = listOf(
    "gen_fun = lambda num: (x for u in range(num) for x in (u*2, u*10, u*u))",
    "() + (1, 'a') + (2, 'b') + (3, 'c')",
    "[2 * x if x > 2 else add_nothing_to_list for x in some_list]",
    "lines = [[float(x) for x in line] for line in csv.reader(f)]",
    """stripped_lines = lambda f : (l.rstrip("\n") for l in f)""",
    "newlist = [word for word in words if len(word) == 9]",
    "Keys = [x for x in d if d[x] == 'a']",
    "filtered = [x for x in common if x in words]",
    "x = 1 + 1",
    "[ i for i in j ]"
  )

  val got = sorted.map { it.first }
  log("GPU WDFA ordering: ${if (got == expected)"PASS" else "FAIL"}")

  packetBuf.destroy()
  prmBuf.destroy()
}
