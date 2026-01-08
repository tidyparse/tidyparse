import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import ai.hypergraph.kaliningraph.automata.FSA
import ai.hypergraph.kaliningraph.automata.NEG_STR_LIT
import ai.hypergraph.kaliningraph.automata.hexFmt
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.START_SYMBOL
import ai.hypergraph.kaliningraph.parsing.bindex
import ai.hypergraph.kaliningraph.parsing.calcStats
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.parsing.tmLst
import ai.hypergraph.tidyparse.MAX_DISP_RESULTS
import web.gpu.GPUBuffer
import kotlin.time.TimeSource

// WIP rewrite of V1 to circumvent backpointer reconstruction stage and decoding LTR directly from dpBuf

const val MAX_PARTICLES_V2 = 8192      // tune: 2k–16k typical
const val META_STRIDE_V2   = 8         // rng, top, wLen, status, t0,t1,t2, levDist
const val STATUS_ACTIVE    = 0
const val STATUS_DONE      = 1
const val STATUS_DEAD      = 2

//language=wgsl
const val V2_COMMON = """
fn xorshift32(x: ptr<function, u32>) -> u32 {
  var v = *x;
  v ^= v << 13u;
  v ^= v >> 17u;
  v ^= v << 5u;
  *x = v;
  return v;
}

fn rand01(x: ptr<function, u32>) -> f32 {
  let r = xorshift32(x);
  return f32(r) / 4294967296.0;
}

fn decode_r(dpIdx: u32, snt: u32) -> u32 { return dpIdx / snt; }
fn decode_c(dpIdx: u32, snt: u32, nt: u32) -> u32 { return (dpIdx - (dpIdx / snt) * snt) / nt; }
fn decode_A(dpIdx: u32, nt: u32) -> u32 { return dpIdx % nt; }
"""

//language=wgsl
val init_particles_v2 by Shader("""
$V2_COMMON

struct IndexUniforms {
  targetCnt: atomic<u32>,
  maxWordLen: u32,
  numNonterminals: u32,
  numStates: u32,
  threads: u32,
  max_samples: u32,
  startIndicesOffset: u32,
  numStartIndices: u32,
  payload: array<u32>,
};

@group(0) @binding(0) var<storage, read_write> metab   : array<u32>; // META_STRIDE_V2 per particle
@group(0) @binding(1) var<storage, read_write> stacks : array<u32>; // maxWordLen per particle
@group(0) @binding(2) var<storage, read_write> words  : array<u32>; // maxWordLen per particle
@group(0) @binding(3) var<storage, read_write> idx_uni: IndexUniforms;

const META_STRIDE : u32 = ${META_STRIDE_V2}u;

fn getStartIdx(i: u32) -> u32 {
  return idx_uni.payload[idx_uni.startIndicesOffset + 2u*i];
}
fn getEditDist(i: u32) -> u32 {
  return idx_uni.payload[idx_uni.startIndicesOffset + 2u*i + 1u];
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= idx_uni.max_samples) { return; }

  let numRoots = idx_uni.numStartIndices / 2u;
  if (numRoots == 0u) { return; }

  // rng seed
  var rng = pid * 747796405u + 2891336453u + atomicLoad(&idx_uni.targetCnt);

  // choose root uniformly
  let rix = xorshift32(&rng) % numRoots;
  let rootDp  = getStartIdx(rix);
  let levDist = getEditDist(rix);

  // metab layout
  let mb = pid * META_STRIDE;
  metab[mb + 0u] = rng;
  metab[mb + 1u] = 1u;                  // top
  metab[mb + 2u] = 0u;                  // wLen
  metab[mb + 3u] = ${STATUS_ACTIVE}u;   // status
  metab[mb + 4u] = ${BOS_ID}u - 1u;     // t0 (0-based for gram keys)
  metab[mb + 5u] = ${NEWLINE_ID}u - 1u; // t1
  metab[mb + 6u] = ${BOS_ID}u - 1u;     // t2
  metab[mb + 7u] = levDist;

  // stack + word storage
  let stride = idx_uni.maxWordLen;
  let sb = pid * stride;
  stacks[sb + 0u] = rootDp;

  let wb = pid * stride;
  words[wb + 0u] = 0u;
}
""".trimIndent())

//language=wgsl
val decode_step_v2 by Shader("""
$CFL_STRUCT
$TERM_STRUCT
$SAMPLER_PARAMS
$V2_COMMON

@group(0) @binding(0) var<storage, read>      dp_in   : array<u32>;
@group(0) @binding(1) var<storage, read>      cs     : CFLStruct;
@group(0) @binding(2) var<storage, read>      terminals : Terminals;
@group(0) @binding(3) var<storage, read>      ngram  : array<u32>;
@group(0) @binding(4) var<storage, read_write> metab   : array<u32>;
@group(0) @binding(5) var<storage, read_write> stacks : array<u32>;
@group(0) @binding(6) var<storage, read_write> words  : array<u32>;
@group(0) @binding(7) var<storage, read>      prm    : Params;        // prm.maxSamples = #particles

const META_STRIDE : u32 = ${META_STRIDE_V2}u;
const PKT_HDR_LEN : u32 = ${PKT_HDR_LEN}u;
const SENTINEL_KEY : u32 = ${SENTINEL.toHexString(hexFmt)};
const HASH_MUL     : u32 = ${HASH_MUL.toHexString(hexFmt)};
const SCALE        : f32 = ${SCALE}f;
const MIN_W        : f32 = 1e-12; // weight floor to avoid underflow -> preserves completeness in float land

fn packGram(a : u32, b : u32, c : u32, d : u32) -> u32 { return (a<<21u)|(b<<14u)|(c<<7u)|d; }
fn hash32(x : u32, pow : u32) -> u32 { return (x * HASH_MUL) >> (32u - pow); }

fn lookupScore(key: u32) -> u32 {
  let pow   : u32 = ngram[0];
  let mask  : u32 = (1u << pow) - 1u;
  var slot  : u32 = hash32(key, pow) & mask;
  loop {
    let idx    = 1u + slot * 2u;
    let stored = ngram[idx];
    if (stored == key)          { return ngram[idx + 1u]; }
    if (stored == SENTINEL_KEY) { return 1u; }
    slot = (slot + 1u) & mask;
  }
}

// Same literal counting as you already use
const LIT_ALL : u32 = 0x7ffffffeu;
const NEG_BIT : u32 = $NEG_STR_LIT;

fn litCountFor(val: u32, nt: u32) -> u32 { return count_tms(val, nt); }

// Sample a token from the literal set, weighted by exp(-cost/SCALE) (single-pass weighted reservoir).
// Returns (hasTok, tokId(1-based), sumW)
fn sampleLex(
  val: u32, nt: u32,
  t0: u32, t1: u32, t2: u32,
  rng: ptr<function, u32>
) -> vec3<u32> {
  let negLit   = (val & NEG_BIT) != 0u;
  let hasLit   = ((val >> 1u) != 0u);
  if (!hasLit) { return vec3<u32>(0u, 0u, 0u); }

  let ntLen = get_nt_tm_lens(nt);
  if (ntLen == 0u) { return vec3<u32>(0u, 0u, 0u); }
  let ntOff = get_offsets(nt);

  // literal encoding: (val>>1)&... gives index+1 (or excluded index+1 for neg)
  let litEnc = (val >> 1u) & 0x1fffffffu;

  var chosenTok : u32 = 0u;
  var bestKey   : f32 = 1e30;
  var sumW      : f32 = 0.0;

  // iterate candidates
  for (var k: u32 = 0u; k < ntLen; k = k + 1u) {
    // apply neg-literal exclusion semantics when needed
    var ok = true;
    if (negLit && litEnc != 0u && val != LIT_ALL) {
      let excl = litEnc - 1u;          // excluded index in Σ_A
      if (k == excl) { ok = false; }
    }
    if (!ok) { continue; }

    // tokId is 1-based in packets (to keep 0 as terminator)
    let tok0 = get_all_tms(ntOff + k); // 0-based terminal id
    let tok  = tok0 + 1u;

    // n-gram weight
    let key   = packGram(t0, t1, t2, tok0);
    let cost  = f32(lookupScore(key));
    let w     = max(MIN_W, exp(-cost / SCALE));
    sumW += w;

    // weighted reservoir: key = -log(u)/w, choose min
    let u = max(1e-12, rand01(rng));
    let rk = -log(u) / w;
    if (rk < bestKey) { bestKey = rk; chosenTok = tok; }
  }

  // pack sumW as u32-ish by scaling; only used for branching decision vs binary
  let sumWU = u32(min(4.0e9, sumW * 1.0e6)); // coarse but monotone
  return vec3<u32>(select(0u, 1u, chosenTok != 0u), chosenTok, sumWU);
}

// Uniformly sample one valid binary expansion for (r,c,A) using reservoir.
// Returns (count, leftDpIdx, rightDpIdx)
fn sampleBin(dpIdx: u32, rng: ptr<function, u32>) -> vec3<u32> {
  let N  = cs.numStates;
  let NT = cs.numNonterminals;
  let snt = N * NT;

  let r = decode_r(dpIdx, snt);
  let c = decode_c(dpIdx, snt, NT);
  let A = decode_A(dpIdx, NT);

  // midpoint range for (r,c) using same addressing scheme as PREAMBLE
  let aoi        = r * N + c + 1u;
  if (c <= r) { return vec3<u32>(0u, 0u, 0u); }

  let pairOffset = getMdptOffset(aoi - 1u);
  var pairNext   : u32;
  if (aoi < cs.mdptsOffsetsSize) { pairNext = getMdptOffset(aoi); }
  else { pairNext = cs.mdptsSize; }

  let startGC = getGrammarOffset(A);
  var endGC: u32;
  if (A + 1u < NT) { endGC = getGrammarOffset(A + 1u); } else { endGC = cs.grammarFlattenedSize; }

  var cnt: u32 = 0u;
  var chosenL: u32 = 0u;
  var chosenR: u32 = 0u;

  for (var p = pairOffset; p < pairNext; p = p + 1u) {
    let m = getMdpt(p);
    for (var g = startGC; g < endGC; g = g + 2u) {
      let B = getGrammarSymbol(g);
      let C = getGrammarSymbol(g + 1u);

      let idxBM = r*snt + m*NT + B;
      let idxMC = m*snt + c*NT + C;

      if (dp_in[idxBM] != 0u && dp_in[idxMC] != 0u) {
        cnt += 1u;
        // reservoir: replace with prob 1/cnt
        let u = xorshift32(rng) % cnt;
        if (u == 0u) { chosenL = idxBM; chosenR = idxMC; }
      }
    }
  }

  return vec3<u32>(cnt, chosenL, chosenR);
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= prm.maxSamples) { return; }

  let mb = pid * META_STRIDE;
  let status = metab[mb + 3u];
  if (status != ${STATUS_ACTIVE}u) { return; }

  var rng = metab[mb + 0u];
  var top = metab[mb + 1u];
  var wLen = metab[mb + 2u];
  var t0 = metab[mb + 4u];
  var t1 = metab[mb + 5u];
  var t2 = metab[mb + 6u];

  let stride = prm.stride;
  let sb = pid * stride;
  let wb = pid * stride;

  // emit at most one token per invocation; allow several internal binary expansions before emitting
  var safety: u32 = 0u;
  loop {
    safety += 1u;
    if (safety > 1024u) { // should never happen in acyclic span forest; acts as a hard guard
      metab[mb + 3u] = ${STATUS_DEAD}u;
      break;
    }

    if (top == 0u) {
      metab[mb + 3u] = ${STATUS_DONE}u;
      break;
    }
    if (wLen + PKT_HDR_LEN >= stride) {
      metab[mb + 3u] = ${STATUS_DEAD}u;
      break;
    }

    let d = stacks[sb + (top - 1u)];
    let val = dp_in[d];
    if (val == 0u) {
      metab[mb + 3u] = ${STATUS_DEAD}u;
      top = 0u;
      break;
    }

    let nt = d % cs.numNonterminals;
    let hasBinFlag = (val & 0x01u) != 0u;
    let litCnt = litCountFor(val, nt);

    // sample one binary expansion (and count how many exist)
    var binCnt: u32 = 0u;
    var leftDp: u32 = 0u;
    var rightDp: u32 = 0u;
    if (hasBinFlag) {
      let b = sampleBin(d, &rng);
      binCnt = b.x; leftDp = b.y; rightDp = b.z;
    }

    // sample a lexical token using n-grams
    var lexHas: u32 = 0u;
    var lexTok: u32 = 0u;
    var lexSumW: u32 = 0u;
    if (litCnt != 0u) {
      let lx = sampleLex(val, nt, t0, t1, t2, &rng);
      lexHas = lx.x; lexTok = lx.y; lexSumW = lx.z;
    }

    // Decide lexical vs binary if both exist.
    // We use a probabilistic mix that gives >0 to both sides whenever both exist.
    if (lexHas != 0u && (binCnt == 0u || binCnt == 0u)) {
      // lexical only
      words[wb + PKT_HDR_LEN + wLen] = lexTok;
      wLen += 1u;
      top -= 1u;

      // advance LM context (tok-1 for packGram)
      t0 = t1; t1 = t2; t2 = lexTok - 1u;
      break;
    }

    if (lexHas == 0u && binCnt != 0u) {
      // binary only: expand (push right then left so left is processed next)
      stacks[sb + (top - 1u)] = rightDp;
      stacks[sb + top] = leftDp;
      top += 1u;
      continue;
    }

    if (lexHas != 0u && binCnt != 0u) {
      // probabilistic choose: p(lex) ~ lexSumW, p(bin) ~ binCnt (scaled)
      let a = f32(lexSumW);
      let b = f32(binCnt) * 1000.0; // tune: makes binary competitive vs lexical mass
      let u = rand01(&rng);
      if (u < (a / (a + b))) {
        // take lexical
        words[wb + PKT_HDR_LEN + wLen] = lexTok;
        wLen += 1u;
        top -= 1u;
        t0 = t1; t1 = t2; t2 = lexTok - 1u;
        break;
      } else {
        // take binary
        stacks[sb + (top - 1u)] = rightDp;
        stacks[sb + top] = leftDp;
        top += 1u;
        continue;
      }
    }

    // neither? dead
    metab[mb + 3u] = ${STATUS_DEAD}u;
    break;
  }

  // store back metab
  metab[mb + 0u] = rng;
  metab[mb + 1u] = top;
  metab[mb + 2u] = wLen;
  metab[mb + 4u] = t0;
  metab[mb + 5u] = t1;
  metab[mb + 6u] = t2;
}
""".trimIndent())

//language=wgsl
val pack_particles_v2 by Shader("""
$SAMPLER_PARAMS

@group(0) @binding(0) var<storage, read>       metab   : array<u32>;
@group(0) @binding(1) var<storage, read>       words  : array<u32>;
@group(0) @binding(2) var<storage, read_write> outPk  : array<u32>;
@group(0) @binding(3) var<storage, read>       prm    : Params;

const META_STRIDE : u32 = ${META_STRIDE_V2}u;
const PKT_HDR_LEN : u32 = ${PKT_HDR_LEN}u;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= prm.maxSamples) { return; }

  let stride = prm.stride;
  let mb = pid * META_STRIDE;
  let wb = pid * stride;
  let ob = pid * stride;

  let status = metab[mb + 3u];
  let levDist = metab[mb + 7u];
  let wLen = metab[mb + 2u];

  // Header: [levDist, scorePlaceholder]
  outPk[ob + 0u] = levDist;
  outPk[ob + 1u] = 0u;

  // Copy tokens
  if (status == ${STATUS_DONE}u) {
    for (var i: u32 = 0u; i < wLen && (PKT_HDR_LEN + i) < stride; i = i + 1u) {
      outPk[ob + PKT_HDR_LEN + i] = words[wb + PKT_HDR_LEN + i];
    }
    // terminator
    if (PKT_HDR_LEN + wLen < stride) { outPk[ob + PKT_HDR_LEN + wLen] = 0u; }
  } else {
    // dead/incomplete => empty packet
    outPk[ob + PKT_HDR_LEN] = 0u;
  }
}
""".trimIndent())

suspend fun repairPipelineV2(
  cfg: CFG,
  fsa: FSA,
  ledBuffer: Int,
  ngrams: GPUBuffer?,
  codePoints: IntArray
): List<String> {
  val t0 = TimeSource.Monotonic.markNow()
  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("FSA(|Q|=$numStates, |δ|=${fsa.transit.size}), ${cfg.calcStats()}")

  val metaBuf = packMetadata(cfg, fsa)
  val tmBuf   = cfg.termBuf
  val wordBuf = codePoints.toGPUBuffer()

  val totalSize = numStates * numStates * numNTs
  val dpBuf     = Shader.createParseChart(STCPSD, totalSize)

  init_chart(dpBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  log("Chart construction took: ${t0.elapsedNow()}")

  cfl_mul_upper.invokeCFLFixpoint(numStates, numNTs, dpBuf, metaBuf)
  log("Matrix closure reached in: ${t0.elapsedNow()}")

  // ---- roots (same logic you already had) ----
  val t1 = TimeSource.Monotonic.markNow()
  val startNT = cfg.bindex[START_SYMBOL]
  val allStartIds = fsa.finalIdxs
    .map { it * numNTs + startNT }
    .let { it.zip(dpBuf.readIndices(it)) }
    .filter { (_, v) -> v != 0 }
    .map { it.first }

  if (allStartIds.isEmpty()) {
    listOf(wordBuf, tmBuf, metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList<String>().also { log("No valid parse found in final states!") }
  }

  val statesToDist = allStartIds.map { it to fsa.idsToCoords[(it - startNT) / numNTs]!!.second }
  val led = statesToDist.minOf { it.second }

  val startIdxs = statesToDist
    .filter { it.second in (led..(led + ledBuffer)) }
    .map { listOf(it.first, it.second) }
    .sortedBy { it[1] }
    .flatten()

  log("V2 roots LED=$led: ${startIdxs.size / 2} roots in ${t1.elapsedNow()}")

  val maxWordLen = (fsa.width + fsa.height + 10).coerceAtMost(MAX_WORD_LEN)
  if (maxWordLen < 8) {
    listOf(wordBuf, tmBuf, metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList<String>()
  }

  // ---- uniforms for V2 ----
  val numRoots = startIdxs.size / 2
  val maxParticles = MAX_PARTICLES_V2.coerceAtMost(65_535 * 256) // keep dispatchable

  // Memory layout: [IDX_UNIFORM_STRUCT]
  val idxUniBuf = packStruct(
    listOf(
      0,
      maxWordLen,
      numNTs,
      numStates,
      DISPATCH_GROUP_SIZE_X,
      maxParticles
    ),
    startIdxs.toGPUBuffer()
  )

  // Particles storage
  val metaSize  = maxParticles * META_STRIDE_V2
  val partMeta  = GPUBuffer(metaSize * 4L, STCPSD)
  val partStack = GPUBuffer(maxParticles * maxWordLen * 4L, STCPSD)
  val partWord  = GPUBuffer(maxParticles * maxWordLen * 4L, STCPSD)

  // Params: maxSamples, k(ignored), stride, threads(ignored here)
  val prmBuf = intArrayOf(maxParticles, 0, maxWordLen, DISPATCH_GROUP_SIZE_X).toGPUBuffer(STCPSD)

  // init particles
  init_particles_v2(partMeta, partStack, partWord, idxUniBuf)((maxParticles + 255) / 256)


  // decode steps (one emitted token per step per particle)
  // If grammar never derives ε, this upper-bounds token count.
  val steps = maxWordLen - PKT_HDR_LEN - 1
  val ngBuf = ngrams ?: run {
    // If you want V2 to also work without LM, you can pass a dummy table or swap lex weights to uniform.
    error("repairPipelineV2 requires ngrams (for now)")
  }

  for (s in 0 until steps) {
    decode_step_v2(dpBuf, metaBuf, tmBuf, ngBuf, partMeta, partStack, partWord, prmBuf)((maxParticles + 255) / 256)
  }

  // pack into packets buffer compatible with decodePacket()
  val outBuf = GPUBuffer(maxParticles * maxWordLen * 4L, STCPSD)
  pack_particles_v2(partMeta, partWord, outBuf, prmBuf)((maxParticles + 255) / 256)

  val ints = outBuf.readInt32Array()
  val res = ArrayList<String>(MAX_DISP_RESULTS)
  for (i in 0 until maxParticles) {
    val pkt = ints.decodePacket(i, cfg.tmLst, maxWordLen) ?: continue
    // pkt.first = levDist; pkt.second = string
    res.add(pkt.second)
    if (res.size >= 10 * MAX_DISP_RESULTS) break
  }

  // cleanup
  listOf(outBuf, partMeta, partStack, partWord, prmBuf, idxUniBuf, wordBuf, tmBuf, metaBuf, dpBuf)
    .forEach(GPUBuffer::destroy)

  return res.distinct().take(MAX_DISP_RESULTS)
}