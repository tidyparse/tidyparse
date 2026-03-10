import GPUBufferUsage.STCPSD
import Shader.Companion.GPUBuffer
import Shader.Companion.packMetadata
import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import ai.hypergraph.kaliningraph.automata.*
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.*
import js.buffer.ArrayBuffer
import js.typedarrays.Int32Array
import web.gpu.GPUBuffer
import kotlin.time.TimeSource

// ============================================================================
// V2: adaptive exact frontier -> weighted particle compression
// ============================================================================

const val MAX_FRONTIER_V2    = 8192
const val META_STRIDE_V2     = 10 // rng,top,wLen,status,t0,t1,t2,levDist,cost,traceLen
const val STATUS_ACTIVE_V2   = 0
const val STATUS_DONE_V2     = 1
const val STATUS_DEAD_V2     = 2
const val MAX_TRACE_STEPS_V2 = MAX_WORD_LEN * 2 + 8
const val RESAMPLE_SCALE_V2  = 1_000_000u

// language=wgsl
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

// language=wgsl
const val V2_FRONTIER_STRUCT = """
struct FrontierParams {
  frontierCount : u32,
  frontierCap   : u32,
  stride        : u32,   // maxWordLen packet stride
  traceStride   : u32,   // reserved
  step          : u32,
  numRoots      : u32,
  totalWeight   : u32,
  reserved      : u32,
};

fn getStartIdx(i : u32, idx_uni: ptr<storage, IndexUniforms, read_write>) -> u32 {
  return (*idx_uni).startIndices[i * 2u];
}
fn getEditDist(i : u32, idx_uni: ptr<storage, IndexUniforms, read_write>) -> u32 {
  return (*idx_uni).startIndices[i * 2u + 1u];
}
"""

// language=wgsl
const val V2_BP_STRUCT = """
struct Backpointers {
  bp_count_offset   : u32, bp_count_size   : u32,
  bp_offset_offset  : u32, bp_offset_size  : u32,
  bp_storage_offset : u32, bp_storage_size : u32,

  payload : array<u32>
};

fn getBpCount(bp: ptr<storage, Backpointers, read>, i: u32) -> u32 { return (*bp).payload[(*bp).bp_count_offset + i]; }
fn getBpOffset(bp: ptr<storage, Backpointers, read>, i: u32) -> u32 { return (*bp).payload[(*bp).bp_offset_offset + i]; }
fn getBpStorage(bp: ptr<storage, Backpointers, read>, i: u32) -> u32 { return (*bp).payload[(*bp).bp_storage_offset + i]; }
"""

// language=wgsl
const val V2_STATE_LAYOUT = """
const META_STRIDE_V2_WGSL : u32 = ${META_STRIDE_V2}u;

fn intsPerState() -> u32 { return META_STRIDE_V2_WGSL + 2u * prm.stride; }
fn stateBase(i: u32) -> u32 { return i * intsPerState(); }
fn metaBase(i: u32)  -> u32 { return stateBase(i); }
fn stackBase(i: u32) -> u32 { return stateBase(i) + META_STRIDE_V2_WGSL; }
fn wordBase(i: u32)  -> u32 { return stateBase(i) + META_STRIDE_V2_WGSL + prm.stride; }
"""

// language=wgsl
val V2_MARKOV_HELPERS = """
const SENTINEL_KEY_V2 : u32 = ${SENTINEL.toHexString(hexFmt)};
const HASH_MUL_V2     : u32 = ${HASH_MUL.toHexString(hexFmt)};
const BOS_ID_V2       : u32 = ${BOS_ID}u;
const NEWLINE_ID_V2   : u32 = ${NEWLINE_ID}u;
const EOS_ID_V2       : u32 = ${EOS_ID}u;
const SCALE_V2        : f32 = ${SCALE}f;

fn packGramV2(a : u32, b : u32, c : u32, d : u32) -> u32 { return (a<<21u)|(b<<14u)|(c<<7u)|d; }
fn hash32V2(x : u32, pow : u32) -> u32 { return (x * HASH_MUL_V2) >> (32u - pow); }

fn lookupScoreV2(ngram: ptr<storage, array<u32>, read>, key: u32) -> u32 {
  let pow   : u32 = (*ngram)[0];
  let mask  : u32 = (1u << pow) - 1u;
  var slot  : u32 = hash32V2(key, pow) & mask;

  loop {
    let idx      = 1u + slot * 2u;
    let stored   = (*ngram)[idx];
    if (stored == key) { return (*ngram)[idx + 1u]; }
    if (stored == SENTINEL_KEY_V2) { return 1u; }
    slot = (slot + 1u) & mask;
  }
}

fn scoreTokV2(
  ngram : ptr<storage, array<u32>, read>,
  t0: u32, t1: u32, t2: u32, tok1: u32
) -> u32 {
  if (tok1 == 0u) { return 0u; }
  return lookupScoreV2(ngram, packGramV2(t0, t1, t2, tok1 - 1u));
}

fn resampleWeightFromCost(cost: u32) -> u32 {
  let denom = 1u + min(cost, ${Int.MAX_VALUE}u);
  return max(1u, ${RESAMPLE_SCALE_V2}u / denom);
}
"""

// language=wgsl
const val V2_ENUM_HELPERS = """
const LIT_ALL_V2 : u32 = 0x7ffffffeu;
const NEG_BIT_V2 : u32 = $NEG_STR_LIT;

fn lexCountFor(val: u32, nt: u32) -> u32 {
  return count_tms(val, nt);
}

fn nthLexTok(val: u32, nt: u32, ord: u32) -> u32 {
  let ntLen = get_nt_tm_lens(nt);
  if (ntLen == 0u) { return 0u; }
  let ntOff = get_offsets(nt);

  if (val == LIT_ALL_V2) {
    if (ord >= ntLen) { return 0u; }
    return get_all_tms(ntOff + ord) + 1u;
  }

  let hasLit = ((val >> 1u) != 0u);
  if (!hasLit) { return 0u; }

  let negLit = (val & NEG_BIT_V2) != 0u;
  let litEnc = (val >> 1u) & 0x1fffffffu;

  if (!negLit) {
    if (ord != 0u) { return 0u; }
    return get_all_tms(ntOff + (litEnc - 1u)) + 1u;
  }

  let excl = litEnc - 1u;
  if (ord >= max(1u, ntLen - 1u)) { return 0u; }
  let idx = select(ord, ord + 1u, ord >= excl);
  return get_all_tms(ntOff + idx) + 1u;
}

fn binCountFor(bp   : ptr<storage, Backpointers, read>, dpIdx : u32) -> u32 { return getBpCount(bp, dpIdx); }

fn nthBinPair(
  bp   : ptr<storage, Backpointers, read>,
  dpIdx : u32,
  ord   : u32
) -> vec2<u32> {
  let ix = getBpOffset(bp, dpIdx) + ord;
  return vec2<u32>(
    getBpStorage(bp, 2u * ix + 0u),
    getBpStorage(bp, 2u * ix + 1u)
  );
}
"""

// ----------------------------------------------------------------------------
// Init frontier from roots
// ----------------------------------------------------------------------------

// language=wgsl
val frontier_init_v2 by Shader("""
$V2_COMMON
$V2_FRONTIER_STRUCT
$IDX_UNIFORM_STRUCT
$V2_STATE_LAYOUT

@group(0) @binding(0) var<storage, read_write> frontier : array<u32>;
@group(0) @binding(1) var<storage, read_write> idx_uni  : IndexUniforms;
@group(0) @binding(2) var<uniform>             prm      : FrontierParams;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= prm.frontierCount) { return; }

  let mb = metaBase(pid);
  let sb = stackBase(pid);
  let wb = wordBase(pid);

  var rng = pid * 747796405u + 2891336453u + atomicLoad(&idx_uni.targetCnt);

  let rootIx  = pid % prm.numRoots;
  let rootDp  = getStartIdx(rootIx, &idx_uni);
  let levDist = getEditDist(rootIx, &idx_uni);

  frontier[mb + 0u] = rng;
  frontier[mb + 1u] = 1u;
  frontier[mb + 2u] = 0u;
  frontier[mb + 3u] = ${STATUS_ACTIVE_V2}u;
  frontier[mb + 4u] = ${BOS_ID}u - 1u;
  frontier[mb + 5u] = ${NEWLINE_ID}u - 1u;
  frontier[mb + 6u] = ${BOS_ID}u - 1u;
  frontier[mb + 7u] = levDist;
  frontier[mb + 8u] = 0u;
  frontier[mb + 9u] = 0u;

  frontier[sb + 0u] = rootDp;
  frontier[wb + 0u] = 0u;
}
""".trimIndent())

// ----------------------------------------------------------------------------
// Count successors per frontier state
// ----------------------------------------------------------------------------

// language=wgsl
val frontier_count_succ_v2 by Shader("""
$TERM_STRUCT
$V2_BP_STRUCT
$V2_COMMON
$V2_FRONTIER_STRUCT
$V2_ENUM_HELPERS
$V2_STATE_LAYOUT

@group(0) @binding(0) var<storage, read>       dp_in     : array<u32>;
@group(0) @binding(1) var<storage, read>       terminals : Terminals;
@group(0) @binding(2) var<storage, read>       bp        : Backpointers;
@group(0) @binding(3) var<storage, read>       frontier  : array<u32>;
@group(0) @binding(4) var<storage, read_write> succCount : array<u32>;
@group(0) @binding(5) var<storage, read_write> idx_uni   : IndexUniforms;
@group(0) @binding(6) var<uniform>             prm       : FrontierParams;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= prm.frontierCount) { return; }

  let mb = metaBase(i);
  let sb = stackBase(i);

  let status = frontier[mb + 3u];
  if (status == ${STATUS_DONE_V2}u) { succCount[i] = 1u; return; }
  if (status != ${STATUS_ACTIVE_V2}u) { succCount[i] = 0u; return; }

  let top  = frontier[mb + 1u];
  let wLen = frontier[mb + 2u];
  if (top == 0u) { succCount[i] = 0u; return; }
  if (wLen + ${PKT_HDR_LEN}u >= prm.stride) { succCount[i] = 0u; return; }

  let d   = frontier[sb + (top - 1u)];
  let val = dp_in[d];
  if (val == 0u) { succCount[i] = 0u; return; }

  let nt = d % idx_uni.numNonterminals;
  let lc = lexCountFor(val, nt);
  let bc = binCountFor(&bp, d);

  succCount[i] = lc + bc;
}
""".trimIndent())

// ----------------------------------------------------------------------------
// Exact successor materialization while frontier fits in budget
// ----------------------------------------------------------------------------

// language=wgsl
val frontier_write_exact_v2 by Shader("""
$TERM_STRUCT
$V2_BP_STRUCT
$V2_COMMON
$V2_FRONTIER_STRUCT
$V2_MARKOV_HELPERS
$V2_ENUM_HELPERS
$V2_STATE_LAYOUT

@group(0) @binding(0) var<storage, read>        dp_in       : array<u32>;
@group(0) @binding(1) var<storage, read>        terminals   : Terminals;
@group(0) @binding(2) var<storage, read>        ngram       : array<u32>;
@group(0) @binding(3) var<storage, read>        bp          : Backpointers;
@group(0) @binding(4) var<storage, read>        frontierIn  : array<u32>;
@group(0) @binding(5) var<storage, read>        succOff     : array<u32>;
@group(0) @binding(6) var<storage, read>        succCount   : array<u32>;
@group(0) @binding(7) var<storage, read_write>  frontierOut : array<u32>;
@group(0) @binding(8) var<storage, read_write>  idx_uni     : IndexUniforms;
@group(0) @binding(9) var<uniform>              prm         : FrontierParams;

fn copyState(src: u32, dst: u32) {
  let sS = stateBase(src);
  let sD = stateBase(dst);
  let len = intsPerState();
  for (var k: u32 = 0u; k < len; k = k + 1u) {
    frontierOut[sD + k] = frontierIn[sS + k];
  }
}

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let src = gid.x;
  if (src >= prm.frontierCount) { return; }

  let cnt = succCount[src];
  if (cnt == 0u) { return; }

  let base = succOff[src];

  let mbS    = metaBase(src);
  let status = frontierIn[mbS + 3u];

  if (status == ${STATUS_DONE_V2}u) {
    copyState(src, base);
    return;
  }
  if (status != ${STATUS_ACTIVE_V2}u) { return; }

  let sbS  = stackBase(src);
  let top  = frontierIn[mbS + 1u];
  let wLen = frontierIn[mbS + 2u];
  let t0   = frontierIn[mbS + 4u];
  let t1   = frontierIn[mbS + 5u];
  let t2   = frontierIn[mbS + 6u];
  let cost = frontierIn[mbS + 8u];

  let d   = frontierIn[sbS + (top - 1u)];
  let val = dp_in[d];
  let nt  = d % idx_uni.numNonterminals;

  let lc = lexCountFor(val, nt);
  let bc = binCountFor(&bp, d);

  for (var j: u32 = 0u; j < lc; j = j + 1u) {
    let dst = base + j;
    copyState(src, dst);

    let mbD = metaBase(dst);
    let wbD = wordBase(dst);

    let tok = nthLexTok(val, nt, j);
    frontierOut[wbD + ${PKT_HDR_LEN}u + wLen] = tok;
    if (${PKT_HDR_LEN}u + wLen + 1u < prm.stride) {
      frontierOut[wbD + ${PKT_HDR_LEN}u + wLen + 1u] = 0u;
    }

    frontierOut[mbD + 2u] = wLen + 1u;
    frontierOut[mbD + 1u] = top - 1u;
    frontierOut[mbD + 8u] = cost + scoreTokV2(&ngram, t0, t1, t2, tok);
    frontierOut[mbD + 4u] = t1;
    frontierOut[mbD + 5u] = t2;
    frontierOut[mbD + 6u] = tok - 1u;
    if (top == 1u) { frontierOut[mbD + 3u] = ${STATUS_DONE_V2}u; }
  }

  for (var j: u32 = 0u; j < bc; j = j + 1u) {
    let dst = base + lc + j;
    copyState(src, dst);

    let mbD = metaBase(dst);
    let sbD = stackBase(dst);

    let pr = nthBinPair(&bp, d, j);
    frontierOut[sbD + (top - 1u)] = pr.y;
    frontierOut[sbD + top]        = pr.x;
    frontierOut[mbD + 1u] = top + 1u;
  }
}
""".trimIndent())

// ----------------------------------------------------------------------------
// Parent resampling weights
// ----------------------------------------------------------------------------

// language=wgsl
val frontier_parent_weights_v2 by Shader("""
$IDX_UNIFORM_STRUCT
$V2_FRONTIER_STRUCT
$V2_STATE_LAYOUT

@group(0) @binding(0) var<storage, read>       frontier : array<u32>;
@group(0) @binding(1) var<storage, read_write> weights  : array<u32>;
@group(0) @binding(2) var<uniform>             prm      : FrontierParams;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= prm.frontierCount) { return; }

  let mb = metaBase(i);
  let status = frontier[mb + 3u];

  if (status == ${STATUS_DONE_V2}u) {
    weights[i] = ${RESAMPLE_SCALE_V2}u;
    return;
  }

  if (status != ${STATUS_ACTIVE_V2}u) {
    weights[i] = 0u;
    return;
  }

  let cost = frontier[mb + 8u];
  let denom = 1u + min(cost, ${Int.MAX_VALUE}u);
  weights[i] = max(1u, ${RESAMPLE_SCALE_V2}u / denom);
}
""".trimIndent())

// ----------------------------------------------------------------------------
// Compressed weighted successor sampling
// ----------------------------------------------------------------------------

// language=wgsl
val frontier_sampled_step_v2 by Shader("""
$TERM_STRUCT
$V2_BP_STRUCT
$V2_COMMON
$V2_FRONTIER_STRUCT
$V2_MARKOV_HELPERS
$V2_ENUM_HELPERS
$V2_STATE_LAYOUT

@group(0) @binding(0) var<storage, read>        dp_in       : array<u32>;
@group(0) @binding(1) var<storage, read>        terminals   : Terminals;
@group(0) @binding(2) var<storage, read>        ngram       : array<u32>;
@group(0) @binding(3) var<storage, read>        bp          : Backpointers;
@group(0) @binding(4) var<storage, read>        frontierIn  : array<u32>;
@group(0) @binding(5) var<storage, read>        parentCDF   : array<u32>;
@group(0) @binding(6) var<storage, read>        parentW     : array<u32>;
@group(0) @binding(7) var<storage, read_write>  frontierOut : array<u32>;
@group(0) @binding(8) var<storage, read_write>  idx_uni     : IndexUniforms;
@group(0) @binding(9) var<uniform>              prm         : FrontierParams;

fn parentTotal() -> u32 {
  if (prm.frontierCount == 0u) { return 0u; }
  let last = prm.frontierCount - 1u;
  return parentCDF[last] + parentW[last];
}

fn chooseParent(rng: ptr<function, u32>) -> u32 {
  let tot = parentTotal();
  if (tot == 0u) { return 0u; }
  let needle = xorshift32(rng) % tot;

  var lo: u32 = 0u;
  var hi: u32 = prm.frontierCount;
  loop {
    if (lo + 1u >= hi) { return min(lo, prm.frontierCount - 1u); }
    let mid = (lo + hi) >> 1u;
    let base = parentCDF[mid];
    let end  = base + parentW[mid];
    if (needle < base) {
      hi = mid;
    } else if (needle >= end) {
      lo = mid + 1u;
    } else {
      return mid;
    }
  }
}

fn copyState(src: u32, dst: u32) {
  let sS = stateBase(src);
  let sD = stateBase(dst);
  let len = intsPerState();
  for (var k: u32 = 0u; k < len; k = k + 1u) {
    frontierOut[sD + k] = frontierIn[sS + k];
  }
}

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dst = gid.x;
  if (dst >= prm.frontierCap) { return; }

  var rng = dst * 1664525u + 1013904223u + atomicLoad(&idx_uni.targetCnt);
  let src = chooseParent(&rng);
  copyState(src, dst);

  let mbD = metaBase(dst);
  let sbD = stackBase(dst);
  let wbD = wordBase(dst);

  let status = frontierOut[mbD + 3u];
  if (status != ${STATUS_ACTIVE_V2}u) { return; }

  let top  = frontierOut[mbD + 1u];
  let wLen = frontierOut[mbD + 2u];
  if (top == 0u) { frontierOut[mbD + 3u] = ${STATUS_DONE_V2}u; return; }
  if (wLen + ${PKT_HDR_LEN}u >= prm.stride) { frontierOut[mbD + 3u] = ${STATUS_DEAD_V2}u; return; }

  let d   = frontierOut[sbD + (top - 1u)];
  let val = dp_in[d];
  if (val == 0u) { frontierOut[mbD + 3u] = ${STATUS_DEAD_V2}u; return; }
  
  let nt  = d % idx_uni.numNonterminals;
  let lc  = lexCountFor(val, nt);
  let bc  = select(0u, binCountFor(&bp, d), (val & 0x01u) != 0u);

  if (lc + bc == 0u) { frontierOut[mbD + 3u] = ${STATUS_DEAD_V2}u; return; }

  let t0 = frontierOut[mbD + 4u];
  let t1 = frontierOut[mbD + 5u];
  let t2 = frontierOut[mbD + 6u];

  var lexMass = 0.0;
  for (var j: u32 = 0u; j < lc; j = j + 1u) {
    let tok = nthLexTok(val, nt, j);
    let sc  = f32(scoreTokV2(&ngram, t0, t1, t2, tok));
    lexMass = lexMass + exp(-sc / SCALE_V2);
  }
  let binMass = f32(bc);
  let total   = lexMass + binMass;

  let u = rand01(&rng) * max(total, 1e-9);
  var acc = 0.0;

  for (var j: u32 = 0u; j < lc; j = j + 1u) {
    let tok = nthLexTok(val, nt, j);
    let scU = scoreTokV2(&ngram, t0, t1, t2, tok);
    let sc  = f32(scU);
    let w   = exp(-sc / SCALE_V2);
    acc = acc + w;

    if (u <= acc) {
      frontierOut[wbD + ${PKT_HDR_LEN}u + wLen] = tok;
      if (${PKT_HDR_LEN}u + wLen + 1u < prm.stride) {
        frontierOut[wbD + ${PKT_HDR_LEN}u + wLen + 1u] = 0u;
      }

      frontierOut[mbD + 2u] = wLen + 1u;
      frontierOut[mbD + 1u] = top - 1u;
      frontierOut[mbD + 8u] = frontierOut[mbD + 8u] + scU;
      frontierOut[mbD + 4u] = t1;
      frontierOut[mbD + 5u] = t2;
      frontierOut[mbD + 6u] = tok - 1u;
      if (top == 1u) { frontierOut[mbD + 3u] = ${STATUS_DONE_V2}u; }
      frontierOut[mbD + 0u] = rng;
      return;
    }
  }

  if (bc != 0u) {
    let ord = xorshift32(&rng) % bc;
    let pr  = nthBinPair(&bp, d, ord);

    frontierOut[sbD + (top - 1u)] = pr.y;
    frontierOut[sbD + top]        = pr.x;
    frontierOut[mbD + 1u] = top + 1u;
    frontierOut[mbD + 0u] = rng;
    return;
  }

  frontierOut[mbD + 3u] = ${STATUS_DEAD_V2}u;
  frontierOut[mbD + 0u] = rng;
}
""".trimIndent())

// ----------------------------------------------------------------------------
// Pack frontier states into V1-compatible packets
// ----------------------------------------------------------------------------

// language=wgsl
val frontier_pack_packets_v2 by Shader("""
$V2_FRONTIER_STRUCT
$V2_STATE_LAYOUT
$IDX_UNIFORM_STRUCT

@group(0) @binding(0) var<storage, read>       frontier : array<u32>;
@group(0) @binding(1) var<storage, read_write> outPk    : array<u32>;
@group(0) @binding(2) var<uniform>             prm      : FrontierParams;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= prm.frontierCount) { return; }

  let mb = metaBase(pid);
  let wb = wordBase(pid);
  let ob = pid * prm.stride;

  let status  = frontier[mb + 3u];
  let levDist = frontier[mb + 7u];
  let wLen    = frontier[mb + 2u];

  outPk[ob + 0u] = levDist;
  outPk[ob + 1u] = 0u;

  if (status == ${STATUS_DONE_V2}u) {
    for (var i: u32 = 0u; i < wLen && (${PKT_HDR_LEN}u + i) < prm.stride; i = i + 1u) {
      outPk[ob + ${PKT_HDR_LEN}u + i] = frontier[wb + ${PKT_HDR_LEN}u + i];
    }
    if (${PKT_HDR_LEN}u + wLen < prm.stride) { outPk[ob + ${PKT_HDR_LEN}u + wLen] = 0u; }
  } else {
    outPk[ob + ${PKT_HDR_LEN}u] = 0u;
  }
}
""".trimIndent())

//language=wgsl
val markov_score_v2 by Shader("""$SAMPLER_PARAMS
@group(0) @binding(0) var<storage, read_write>  packets : array<u32>;
@group(0) @binding(1) var<storage, read>          ngram : array<u32>;
@group(0) @binding(2) var<uniform>                  prm : Params;

const PKT_HDR_LEN  : u32 = ${PKT_HDR_LEN}u;
const SENTINEL_KEY : u32 = ${SENTINEL.toHexString(hexFmt)};
const HASH_MUL     : u32 = ${HASH_MUL.toHexString(hexFmt)};
const BOS_ID       : u32 = ${BOS_ID}u;
const NEWLINE_ID   : u32 = ${NEWLINE_ID}u;
const EOS_ID       : u32 = ${EOS_ID}u;

fn packGram(a : u32, b : u32, c : u32, d : u32) -> u32 { return (a<<21u)|(b<<14u)|(c<<7u)|d; }

fn hash32(x : u32, pow : u32) -> u32 { return (x * HASH_MUL) >> (32u - pow); }

fn lookupScore(key: u32) -> u32 {
    let pow   : u32 = ngram[0];
    let mask  : u32 = (1u << pow) - 1u;
    var slot  : u32 = hash32(key, pow) & mask;

    loop {
        let idx      = 1u + slot * 2u;
        let stored   = ngram[idx];
        if (stored == key)          { return ngram[idx + 1u]; }
        if (stored == SENTINEL_KEY) { return 1u; }
        slot = (slot + 1u) & mask;
    }
}

@compute @workgroup_size(1,1,1) fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let sid = gid.x + gid.y * prm.threads;
    if (sid >= prm.maxSamples) { return; }

    let stride : u32 = prm.stride;
    let base   : u32 = sid * stride;

    let firstTok = packets[base + PKT_HDR_LEN];
    if (firstTok == 0u) {
        packets[base + 1u] = 0xffffffffu;
        return;
    }

    var t0 : u32 = BOS_ID     - 1u;
    var t1 : u32 = NEWLINE_ID - 1u;
    var t2 : u32 = firstTok   - 1u;

    var pos : u32 = 1u;
    var score : u32 = 0u;
    var doneSuffix : u32 = 0u;

    loop {
        var tok : u32;
        if (pos < stride - PKT_HDR_LEN && packets[base + PKT_HDR_LEN + pos] != 0u) {
            tok = packets[base + PKT_HDR_LEN + pos];
            pos += 1u;
        } else {
            tok = select(EOS_ID, NEWLINE_ID, doneSuffix == 0u);
            doneSuffix += 1u;
            if (doneSuffix > 2u) { break; }
        }

        let key = packGram(t0, t1, t2, tok - 1u);
        score += lookupScore(key);

        t0 = t1; t1 = t2; t2 = tok - 1u;
    }

    packets[base + 1u] = score + (packets[base] + 1u) * 10000000u;
}""")

// ----------------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------------

private data class FrontierV2(
  var buf: GPUBuffer,
  var count: Int
) {
  fun destroy() = buf.destroy()
}

private fun allocFrontierV2(count: Int, stride: Int): FrontierV2 {
  val intsPerState = META_STRIDE_V2 + 2 * stride
  return FrontierV2(
    buf = GPUBuffer(count.toLong() * intsPerState * 4L, STCPSD),
    count = count
  )
}

private suspend fun readLastExclusivePlusLastCount(exclusive: GPUBuffer, counts: GPUBuffer, n: Int): Long {
  if (n <= 0) return 0L
  val last = n - 1
  val ex = exclusive.readIndices(listOf(last))[0].toUInt().toLong()
  val ct = counts.readIndices(listOf(last))[0].toUInt().toLong()
  return ex + ct
}

private suspend fun uniqueDecodePackets(
  ints: Int32Array<ArrayBuffer>,
  cfg: CFG,
  stride: Int,
  limit: Int = MAX_DISP_RESULTS
): List<String> {
  val out = LinkedHashSet<String>()
  for (i in 0 until (ints.length / stride)) {
    val pkt = ints.decodePacket(i, cfg.tmLst, stride) ?: continue
    out += pkt.second
    if (out.size >= limit) break
    pause(100_000)
  }
  return out.toList()
}

// ----------------------------------------------------------------------------
// Full repairPipelineV2
// ----------------------------------------------------------------------------

suspend fun repairPipelineV2(
  cfg: CFG,
  fsa: FSA,
  ledBuffer: Int,
  ngrams: GPUBuffer?,
  codePoints: IntArray
): List<String> {
  val t0 = TimeSource.Monotonic.markNow()
  val timings = linkedMapOf<String, Int>()
  fun mark(step: String, started: kotlin.time.TimeMark) {
    timings[step] = started.elapsedNow().inWholeMilliseconds.toInt()
  }

  val (numStates, numNTs) = fsa.numStates to cfg.nonterminals.size
  log("V2 FSA(|Q|=$numStates, |δ|=${fsa.transit.size}), ${cfg.calcStats()}")

  val metaT = TimeSource.Monotonic.markNow()
  val metaBuf = packMetadata(cfg, fsa)
  mark("pack metadata", metaT)

  val tmBuf       = cfg.termBuf
  val wordBuf     = codePoints.toGPUBuffer()
  val totalSize   = numStates * numStates * numNTs
  val activeWords = (numNTs + 31) ushr 5

  val dpBuf     = Shader.createParseChart(STCPSD, totalSize)
  val activeBuf = GPUBuffer((numStates * numStates * activeWords * 4).toLong(), STCPSD)

  val initT = TimeSource.Monotonic.markNow()
  init_chart(dpBuf, activeBuf, wordBuf, metaBuf, tmBuf)(numStates, numStates, numNTs)
  mark("init chart", initT)

  val closureT = TimeSource.Monotonic.markNow()
  cfl_mul_upper.invokeCFLFixpoint(numStates, dpBuf, activeBuf, metaBuf)
  mark("matrix closure", closureT)

  val rootsT = TimeSource.Monotonic.markNow()
  val startNT = cfg.bindex[START_SYMBOL]
  val allStartIds = fsa.levFinalIdxs.map { it * numNTs + startNT }
    .let { it.zip(dpBuf.readIndices(it)) }
    .filter { (_, v) -> v != 0 }
    .map { it.first }
  mark("read roots", rootsT)

  if (allStartIds.isEmpty()) {
    timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
    timings.logTimingsToJSConsole()
    listOf(activeBuf, wordBuf, metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList<String>().also { log("V2: no valid parse found") }
  }

  val filterT = TimeSource.Monotonic.markNow()
  val statesToDist = allStartIds.map { it to fsa.idsToCoords[(it - startNT) / numNTs]!!.second }
  val led = statesToDist.minOf { it.second }
  val startIdxs = statesToDist
    .filter { it.second in (led..(led + ledBuffer)) }
    .map { listOf(it.first, it.second) }
    .sortedBy { it[1] }
    .flatten()
  mark("filter roots", filterT)

  val numRoots = startIdxs.size / 2
  if (numRoots == 0) {
    timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
    timings.logTimingsToJSConsole()
    listOf(activeBuf, wordBuf, metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList()
  }

  val maxRepairLen = fsa.width + fsa.height + 10
  if (MAX_WORD_LEN < maxRepairLen) {
    timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
    timings.logTimingsToJSConsole()
    listOf(activeBuf, wordBuf, metaBuf, dpBuf).forEach(GPUBuffer::destroy)
    return emptyList<String>().also {
      log("V2: max repair length exceeded $MAX_WORD_LEN ($maxRepairLen)")
    }
  }

  val bpT = TimeSource.Monotonic.markNow()
  val bpMetaBuf = run {
    val (bpCountBuf, bpOffsetBuf, bpStorageBuf) =
      Shader.buildBackpointers(numStates, numNTs, dpBuf, metaBuf)
    packStruct(emptyList(), bpCountBuf, bpOffsetBuf, bpStorageBuf)
  }
  mark("build backpointers", bpT)

  val idxUniBuf = packStruct(
    listOf(0, maxRepairLen, numNTs, numStates, DISPATCH_GROUP_SIZE_X, MAX_FRONTIER_V2),
    startIdxs.toGPUBuffer()
  )

  val frontierCap = MAX_FRONTIER_V2
  var exact = numRoots <= frontierCap
  var frontierA = allocFrontierV2(if (exact) numRoots else frontierCap, maxRepairLen)
  var frontierB = allocFrontierV2(frontierCap, maxRepairLen)

  val initFrontierT = TimeSource.Monotonic.markNow()
  run {
    val prm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_TRACE_STEPS_V2, 0, numRoots, 0, 0)
      .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)
    frontier_init_v2(frontierA.buf, idxUniBuf, prm)((frontierA.count + 255) / 256)
    prm.destroy()
  }
  mark("init frontier", initFrontierT)

  val ngBuf = ngrams ?: listOf(1u, SENTINEL, 1u, SENTINEL, 1u).toGPUBuffer(STCPSD)

  val decodeT = TimeSource.Monotonic.markNow()

  for (step in 0 until MAX_TRACE_STEPS_V2) {
    val succCountBuf = GPUBuffer(frontierA.count * 4L, STCPSD)
    val countPrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_TRACE_STEPS_V2, step, numRoots, 0, 0)
      .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

    frontier_count_succ_v2(
      dpBuf, tmBuf, bpMetaBuf, frontierA.buf, succCountBuf, idxUniBuf, countPrm
    )((frontierA.count + 255) / 256)

    val succOffBuf = Shader.prefixSumGPU(succCountBuf, frontierA.count)
    val totalSucc = readLastExclusivePlusLastCount(succOffBuf, succCountBuf, frontierA.count).toInt()

    countPrm.destroy()

    if (totalSucc == 0) {
      succOffBuf.destroy()
      succCountBuf.destroy()
      break
    }

    if (exact && totalSucc <= frontierCap) {
      frontierB.destroy()
      frontierB = allocFrontierV2(totalSucc, maxRepairLen)

      val writePrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_TRACE_STEPS_V2, step, numRoots, 0, 0)
        .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      frontier_write_exact_v2(
        dpBuf, tmBuf, ngBuf, bpMetaBuf,
        frontierA.buf, succOffBuf, succCountBuf,
        frontierB.buf, idxUniBuf, writePrm
      )((frontierA.count + 63) / 64)

      frontierB.count = totalSucc
      writePrm.destroy()
    } else {
      exact = false

      frontierB.destroy()
      frontierB = allocFrontierV2(frontierCap, maxRepairLen)

      val weightBuf = GPUBuffer(frontierA.count * 4L, STCPSD)
      val wPrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_TRACE_STEPS_V2, step, numRoots, 0, 0)
        .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      frontier_parent_weights_v2(frontierA.buf, weightBuf, wPrm)((frontierA.count + 255) / 256)
      val parentCDF = Shader.prefixSumGPU(weightBuf, frontierA.count)
      val totalW = readLastExclusivePlusLastCount(parentCDF, weightBuf, frontierA.count).toInt()

      val samplePrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_TRACE_STEPS_V2, step, numRoots, totalW, 0)
        .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

      frontier_sampled_step_v2(
        dpBuf,      // binding 0
        tmBuf,      // binding 1
        ngBuf,      // binding 2
        bpMetaBuf,  // binding 3
        frontierA.buf, // binding 4
        parentCDF,  // binding 5
        weightBuf,  // binding 6
        frontierB.buf, // binding 7
        idxUniBuf,  // binding 8
        samplePrm   // binding 9
      )((frontierCap + 255) / 256)

      frontierB.count = frontierCap

      samplePrm.destroy()
      wPrm.destroy()
      parentCDF.destroy()
      weightBuf.destroy()
    }

    succOffBuf.destroy()
    succCountBuf.destroy()

    val tmp = frontierA
    frontierA = frontierB
    frontierB = tmp
  }

  mark("decode", decodeT)

  val packT = TimeSource.Monotonic.markNow()
  val outBuf = GPUBuffer(frontierA.count * maxRepairLen * 4L, STCPSD)
  val packPrm = intArrayOf(frontierA.count, frontierCap, maxRepairLen, MAX_TRACE_STEPS_V2, 0, numRoots, 0, 0)
    .toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  frontier_pack_packets_v2(frontierA.buf, outBuf, packPrm)((frontierA.count + 255) / 256)
  packPrm.destroy()
  mark("pack packets", packT)

  val result = if (ngrams != null) {
    val topK = scoreSelectGatherV2(
      packets = outBuf,
      ngrams = ngBuf,
      maxSamples = frontierA.count,
      stride = maxRepairLen,
      k = TOP_K_SAMP
    )
    val t = TimeSource.Monotonic.markNow()
    val r = mutableListOf<String>()
    for (i in 0 until TOP_K_SAMP) {
      val pkt = topK.decodePacket(i, cfg.tmLst, maxRepairLen) ?: continue
      r.add(pkt.second)
    }
    log("V2 decoded ${r.distinct().size} unique words out of ${r.size} in ${t.elapsedNow()}")
    r.distinct().take(MAX_DISP_RESULTS)
  } else uniqueDecodePackets(outBuf.readInt32Array(), cfg, maxRepairLen, MAX_DISP_RESULTS)

  timings["total"] = t0.elapsedNow().inWholeMilliseconds.toInt()
  timings.logTimingsToJSConsole()
  log("repairPipelineV2 completed in ${timings["total"]}ms | frontier=${frontierA.count} | exact=$exact")

  listOf(outBuf, idxUniBuf, wordBuf, metaBuf, bpMetaBuf, dpBuf, activeBuf).forEach(GPUBuffer::destroy)
  frontierA.destroy()
  frontierB.destroy()
  if (ngrams == null) ngBuf.destroy()

  return result
}

suspend fun scoreSelectGatherV2(
  packets    : GPUBuffer,
  ngrams     : GPUBuffer,
  maxSamples : Int,
  stride     : Int,
  k          : Int
): Int32Array<ArrayBuffer> {
  val t0 = TimeSource.Monotonic.markNow()
  val threads = DISPATCH_GROUP_SIZE_X
  val groupsY = (maxSamples + threads - 1) / threads
  /** Memory layout: [SAMPLER_PARAMS] */
  val prmBuf  = intArrayOf(maxSamples, k, stride, threads).toGPUBuffer(GPUBufferUsage.UNIFORM or GPUBufferUsage.COPY_DST)

  markov_score_v2(packets, ngrams, prmBuf)(threads, groupsY)
//  log("Score in ${t0.elapsedNow()}")

//  log(packets.readInts().toList().windowed(stride, stride)
//    .map { it[1] }.groupingBy { it }.eachCount().entries
//    .sortedBy { it.key }.joinToString("\n") { (a, b) -> "$a => $b" })

//  t0 = TimeSource.Monotonic.markNow()
  val totalGroups = (maxSamples + 255) / 256
  val selGroupsY  = (totalGroups + threads - 1) / threads
  val idxBuf      = IntArray(k) { Int.MAX_VALUE }.toGPUBuffer(STCPSD)
  val scrBuf      = IntArray(k) { Int.MAX_VALUE }.toGPUBuffer(STCPSD)

  select_top_k(prmBuf, packets, idxBuf, scrBuf)(threads, selGroupsY)
//  log("Select in ${t0.elapsedNow()}")

//  t0 = TimeSource.Monotonic.markNow()
  val bestBuf   = GPUBuffer(k * stride * 4, STCPSD)

  gather_top_k(prmBuf, packets, idxBuf, bestBuf)(k)
//  log("Gather in ${t0.elapsedNow()}")

//  t0 = TimeSource.Monotonic.markNow()
  val topK = bestBuf.readInt32Array()
  log("Score/select/gather read ${topK.length} = ${k}x${stride}x4 bytes in ${t0.elapsedNow()}")

  listOf(prmBuf, idxBuf, scrBuf, bestBuf).forEach(GPUBuffer::destroy)
  return topK
}