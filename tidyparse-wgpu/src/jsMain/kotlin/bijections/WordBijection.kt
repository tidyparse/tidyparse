package ai.hypergraph.tidyparse.wgpu


/**
 * A finite rank → word map represented by ordered sums of Cartesian products.
 * Each node partitions its rank interval among alternatives. A product splits the selected
 * local rank by the right child's cardinality; a leaf emits its token. No builder-specific
 * ordering, grammar, cost quantization, or construction data is needed to decode a rank.
 *
 * Layout after the four-word header: nodeOffsets[N+1], nodeCounts[N], alternatives[4*A], roots[3*R].
 * Alternative = (inclusive end, left node, right node, local cost).
 * left=UINT_MAX denotes a leaf whose right field is the final packed token.
 * Root = (inclusive end, node, edit distance). Node zero is epsilon: count 1, no alternatives.
 * Counts saturate at UINT_MAX, retaining a deterministic prefix of the derivation order.
 * rankLimit sets the sampling domain; zero uses the requested count. Explicit ranks ignore it.
 */

//language=wgsl
internal const val WORD_BIJECTION_STRUCT = """
struct WordBijection {
  nodes: u32, choices: u32, roots: u32, rankLimit: u32,
  data: array<u32>
}
"""

//language=wgsl
internal const val WORD_BIJECTION_HELPERS = """$WORD_BIJECTION_STRUCT
const BIJ_LEAF: u32 = 0xffffffffu;
fn bij_add(a: u32, b: u32) -> u32 { return a + min(b, BIJ_LEAF - a); }
fn bij_mul(a: u32, b: u32) -> u32 {
  if (a == 0u || b == 0u) { return 0u; }
  if (a > BIJ_LEAF / b) { return BIJ_LEAF; }
  return a * b;
}
fn bij_count_base() -> u32 { return bijection.nodes + 1u; }
fn bij_choice_base() -> u32 { return 2u * bijection.nodes + 1u; }
fn bij_roots_base() -> u32 { return bij_choice_base() + 4u * bijection.choices; }
fn bij_first(node: u32) -> u32 { return bijection.data[node]; }
fn bij_end(node: u32) -> u32 { return bijection.data[node + 1u]; }
fn bij_size(node: u32) -> u32 { return bijection.data[bij_count_base() + node]; }
fn bij_choice(choice: u32) -> vec4<u32> {
  let p = bij_choice_base() + 4u * choice;
  return vec4<u32>(bijection.data[p], bijection.data[p+1u], bijection.data[p+2u], bijection.data[p+3u]);
}
fn bij_root(root: u32) -> vec3<u32> {
  let p = bij_roots_base() + 3u * root;
  return vec3<u32>(bijection.data[p], bijection.data[p+1u], bijection.data[p+2u]);
}
"""

// Deterministic unranking is reusable with arbitrary caller-supplied ranks, independently of RNG.
//language=wgsl
internal const val WORD_BIJECTION_DECODER = """$IDX_UNIFORM_STRUCT $WORD_BIJECTION_HELPERS
@group(0) @binding(0) var<storage, read> bijection: WordBijection;
@group(0) @binding(1) var<storage, read_write> idx_uni: IndexUniforms;
@group(0) @binding(2) var<storage, read_write> sampled: array<u32>;
struct Frame { node: u32, rank: u32 }

fn decode_word(sid: u32, rank: u32) {
  let out = sid * idx_uni.maxWordLen;
  sampled[out] = 0u; sampled[out+1u] = 0u; sampled[out+${PKT_HDR_LEN}u] = 0u;
  if (bijection.roots == 0u || rank >= bij_root(bijection.roots - 1u).x) { return; }
  var lo = 0u;
  var hi = bijection.roots;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (rank < bij_root(mid).x) { hi = mid; } else { lo = mid + 1u; }
  }
  let root = bij_root(lo);
  var previous = 0u;
  if (lo != 0u) { previous = bij_root(lo - 1u).x; }
  sampled[out] = root.z;
  var stack: array<Frame, ${MAX_WORD_LEN}u>;
  stack[0] = Frame(root.y, rank - previous);
  var top = 1u;
  var word: array<u32, ${MAX_WORD_LEN}u>;
  var length = 0u;
  var cost = 0u;
  while (top != 0u) {
    top--;
    let frame = stack[top];
    if (frame.node == 0u) { continue; }
    let first = bij_first(frame.node);
    let end = bij_end(frame.node);
    lo = first; hi = end;
    while (lo < hi) {
      let mid = (lo + hi) >> 1u;
      if (frame.rank < bij_choice(mid).x) { hi = mid; } else { lo = mid + 1u; }
    }
    if (lo == end) { return; }
    previous = 0u;
    if (lo != first) { previous = bij_choice(lo - 1u).x; }
    let choice = bij_choice(lo);
    cost = bij_add(cost, choice.w);
    if (choice.y == BIJ_LEAF) {
      if (length + ${PKT_HDR_LEN}u >= idx_uni.maxWordLen || length >= ${MAX_WORD_LEN}u) { return; }
      word[length] = choice.z; length++;
    } else {
      let inside = frame.rank - previous;
      let rightCount = bij_size(choice.z);
      if (rightCount == 0u) { return; }
      // Epsilon products also implement unions and do not consume a stack frame.
      if (choice.z != 0u) {
        if (top >= ${MAX_WORD_LEN}u) { return; }
        stack[top] = Frame(choice.z, inside % rightCount); top++;
      }
      if (choice.y != 0u) {
        if (top >= ${MAX_WORD_LEN}u) { return; }
        stack[top] = Frame(choice.y, inside / rightCount); top++;
      }
    }
  }
  sampled[out+1u] = cost;
  for (var i = 0u; i < length; i++) { sampled[out+${PKT_HDR_LEN}u+i] = word[i]; }
  if (length + ${PKT_HDR_LEN}u < idx_uni.maxWordLen) { sampled[out+${PKT_HDR_LEN}u+length] = 0u; }
}
"""

// A pointwise permutation of [0,count). Cycle walking preserves uniqueness without a rank buffer.
//language=wgsl
internal const val BIJECTION_PERMUTATION = """$WGSL_MIX32
fn bijection_rank(sid: u32, count: u32, seed: u32) -> u32 {
  if (count <= 1u) { return 0u; }
  let bits = (33u - countLeadingZeros(count - 1u)) & 0xfffffffeu;
  let half = bits >> 1u;
  let mask = (1u << half) - 1u;
  var value = sid;
  loop {
    var left = value & mask;
    var right = value >> half;
    for (var round = 0u; round < 4u; round++) {
      let next = left ^ (mix32(right ^ seed ^ (round * 0x7f4a7c15u)) & mask);
      left = right; right = next;
    }
    value = (right << half) | left;
    if (value < count) { return value; }
  }
}
"""

// One production dispatch selects a rank and decodes it using the same builder-independent algorithm.
//language=wgsl
internal val enum_words_wor by Shader("""$WORD_BIJECTION_DECODER $BIJECTION_PERMUTATION
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let sid = gid.x + gid.y * idx_uni.threads;
  let count = select(idx_uni.max_samples, bijection.rankLimit, bijection.rankLimit != 0u);
  if (sid >= min(idx_uni.max_samples, count)) { return; }
  let seed = atomicLoad(&idx_uni.targetCnt) ^ 0xA511E9B3u;
  decode_word(sid, bijection_rank(sid, count, seed));
}
""")