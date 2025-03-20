external val navigator: dynamic

/**
 * We store the grammar and allFSA arrays in a flattened form:
 *
 * For grammar:
 *   - We'll create an array whose first 2*V elements are offsets and lengths for each A:
 *       grammarFlat[A*2 + 0] = offsetIntoPairs
 *       grammarFlat[A*2 + 1] = numberOfBCpairsForA
 *   - After that, we store all (B,C) pairs for each A in sequence.
 *
 * For allFSA:
 *   - We'll create an array whose first 2*Q*Q elements are offsets and lengths for (p,q):
 *       allFSAFlat[(p*Q + q)*2 + 0] = offsetIntoMidpoints
 *       allFSAFlat[(p*Q + q)*2 + 1] = numberOfMidpoints
 *   - After that, we store all midpoints r for each (p,q) in sequence.
 *
 * In the WGSL code, `mainSquare` will do:
 *   chartTmp[p,q,A] = OR_{r in midpoints(p,q), (B,C) in grammar(A)} ( chart[p,r,B] && chart[r,q,C] ).
 *
 * Finally we perform exponentiation-by-squaring ~log2(Q) times:
 *   S(M_{i+1}) = S(M_i) + S(M_i)^2,
 * which we approximate with repeated "square + OR" steps for up to ceil(log2(Q)) iterations.
 */

//language=wgsl
private val WGSL_BAR_HILLEL: String = """
// -------------------------------------------------------------------------
// Example buffer/struct declarations
// Adjust as appropriate for your own pipeline layout
// -------------------------------------------------------------------------

struct Uniforms {
    numStates               : i32,
    allFSAPairsSize         : i32,
    allFSAPairsOffsetsSize  : i32,
    numNonterminals         : i32,
    volen                   : i32,
    vilen                   : i32,
};

// Device buffers
@group(0) @binding(0) var<storage, read>        dp_in               : array<u16>;
@group(0) @binding(1) var<storage, read_write>  dp_out              : array<u16>;
@group(0) @binding(2) var<storage, read>        allFSAPairs         : array<i32>;
@group(0) @binding(3) var<storage, read>        allFSAPairsOffsets  : array<i32>;

// Uniform buffer holding int scalars
@group(0) @binding(4) var<uniform>              uni                 : Uniforms;

// Atomic counter for number of nonzero entries
@group(0) @binding(5) var<storage, read_write>  numNonzero          : atomic<u32>;

// Grammar header data (example)
@group(0) @binding(6) var<storage, read>        vidx_offsets        : array<i32>;
@group(0) @binding(7) var<storage, read>        vidx                : array<i32>;

// For the bp_count kernel
@group(0) @binding(8) var<storage, read_write>  bpCount             : array<i32>;

// -------------------------------------------------------------------------
// Helper function: decode (row, col) from tid in the upper-triangle
// -------------------------------------------------------------------------
fn decodeUpperTriangle(i: i32, N: i32) -> vec2<i32> {
    let d     = f32( (2 * N - 1) * (2 * N - 1) ) - f32(8 * i);
    let root  = sqrt(d);
    let rF    = (f32(2 * N - 1) - root) * 0.5;
    let r     = i32(floor(rF));
    let rowSt = r * (N - 1) - (r * (r - 1)) / 2;
    let off   = i - rowSt;
    let c     = r + 1 + off;
    return vec2<i32>(r, c);
}

// -------------------------------------------------------------------------
// cfl_mul_upper kernel (Metal -> WGSL)
// -------------------------------------------------------------------------
@compute @workgroup_size(64)
fn cfl_mul_upper(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid         = i32(gid.x);
    let numStates   = uni.numStates;
    let nnt         = uni.numNonterminals;
    // totalCells = (numStates*(numStates-1)/2)*numNonterminals
    let totalCells  = (numStates * (numStates - 1) / 2) * nnt;

    if (tid >= totalCells) { return; }

    // Decode (r, c) for the upper-triangle
    let i       = tid / nnt;
    let rc      = decodeUpperTriangle(i, numStates);
    let r       = rc.x;
    let c       = rc.y;
    let A       = tid % nnt;
    let snt     = numStates * nnt;
    let dpIdx   = r * snt + c * nnt + A;

    // These come from the grammar header
    let startGC = vidx_offsets[A];
    let endGC   = if (A + 1 < uni.volen) { vidx_offsets[A + 1] } else { uni.vilen };

    // aoi = r * numStates + c + 1
    let aoi             = r * numStates + c + 1;
    let pairOffset      = allFSAPairsOffsets[aoi - 1];
    let pairOffsetNext  = if (aoi < uni.allFSAPairsOffsetsSize) {
                              allFSAPairsOffsets[aoi]
                          } else {
                              uni.allFSAPairsSize
                          };

    // Check if dp_in[dpIdx] is already non-zero
    let val = dp_in[dpIdx];
    if (val != 0u) {
        // Mirror original cfl_mul_upper: copy dp_in -> dp_out, increment count
        dp_out[dpIdx] = val;
        atomicAdd(&numNonzero, 1u);

        // If the last bit (0x01) is set, return
        if ((val & 0x01u) != 0u) { return; }
    }

    // Otherwise, search FSA pairs
    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = allFSAPairs[pairIdx];

        var g = startGC;
        while (g < endGC) {
            let B = vidx[g];
            let C = vidx[g + 1];

            let idxBM = r      * snt + mdpt * nnt + B;
            let idxMC = mdpt   * snt + c    * nnt + C;

            if ((dp_in[idxBM] != 0u) && (dp_in[idxMC] != 0u)) {
                // Set dp_out[dpIdx] bit 0 => 'nonterminal found'
                dp_out[dpIdx] = dp_out[dpIdx] | 0x01u;
                atomicAdd(&numNonzero, 1u);
                return;
            }
            g = g + 2;
        }
    }
}

// -------------------------------------------------------------------------
// bp_count kernel (Metal -> WGSL)
// -------------------------------------------------------------------------
@compute @workgroup_size(64)
fn bp_count(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid         = i32(gid.x);
    let numStates   = uni.numStates;
    let nnt         = uni.numNonterminals;
    let totalCells  = (numStates * (numStates - 1) / 2) * nnt;

    if (tid >= totalCells) { return; }

    // Decode (r, c) for the upper-triangle
    let i       = tid / nnt;
    let rc      = decodeUpperTriangle(i, numStates);
    let r       = rc.x;
    let c       = rc.y;
    let A       = tid % nnt;
    let snt     = numStates * nnt;
    let dpIdx   = r * snt + c * nnt + A;

    let startGC = vidx_offsets[A];
    let endGC   = if (A + 1 < uni.volen) { vidx_offsets[A + 1] } else { uni.vilen };

    let aoi             = r * numStates + c + 1;
    let pairOffset      = allFSAPairsOffsets[aoi - 1];
    let pairOffsetNext  = if (aoi < uni.allFSAPairsOffsetsSize) {
                              allFSAPairsOffsets[aoi]
                          } else {
                              uni.allFSAPairsSize
                          };

    // If dp_in[dpIdx] does NOT have bit 0 set, then bpCount = 0
    if ((dp_in[dpIdx] & 0x01u) == 0u) {
        bpCount[dpIdx] = 0;
        return;
    }

    // Otherwise accumulate the possible backpointers
    var count = 0;
    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = allFSAPairs[pairIdx];

        var g = startGC;
        while (g < endGC) {
            let B = vidx[g];
            let C = vidx[g + 1];

            let idxBM = r      * snt + mdpt * nnt + B;
            let idxMC = mdpt   * snt + c    * nnt + C;
            if ((dp_in[idxBM] != 0u) && (dp_in[idxMC] != 0u)) {
                count = count + 1;
            }
            g = g + 2;
        }
    }

    bpCount[dpIdx] = count;
}

@compute @workgroup_size(64)
fn bp_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid         = i32(gid.x);
    let numStates   = uni.numStates;
    let nnt         = uni.numNonterminals;
    let totalCells  = (numStates * (numStates - 1) / 2) * nnt;

    if (tid >= totalCells) { return; }

    // Decode (r, c, A, dpIdx)
    let i        = tid / nnt;
    let rc       = decodeUpperTriangle(i, numStates);
    let r        = rc.x;
    let c        = rc.y;
    let A        = tid % nnt;
    let snt      = numStates * nnt;
    let dpIdx    = r * snt + c * nnt + A;

    // startGC, endGC
    let startGC  = vidx_offsets[A];
    let endGC    = if (A + 1 < uni.volen) { vidx_offsets[A + 1] } else { uni.vilen };

    // aoi, pair offsets
    let aoi             = r * numStates + c + 1;
    let pairOffset      = allFSAPairsOffsets[aoi - 1];
    let pairOffsetNext  = if (aoi < uni.allFSAPairsOffsetsSize) {
                              allFSAPairsOffsets[aoi]
                          } else {
                              uni.allFSAPairsSize
                          };

    // If dp_in[dpIdx] has bit 0 set
    if ((dp_in[dpIdx] & 0x01u) == 0u) { return; }

    var outPos = bpOffset[dpIdx];

    // Similar to bp_count, but we record expansions in bpStorage
    for (var pairIdx = pairOffset; pairIdx < pairOffsetNext; pairIdx++) {
        let mdpt = allFSAPairs[pairIdx];

        var poff = startGC;
        while (poff < endGC) {
            let B = vidx[poff];
            let C = vidx[poff + 1];

            let idxBM = r    * snt + mdpt * nnt + B;
            let idxMC = mdpt * snt + c    * nnt + C;

            if ((dp_in[idxBM] != 0u) && (dp_in[idxMC] != 0u)) {
                // Each record is 2 ints
                bpStorage[outPos * 2 + 0] = idxBM;
                bpStorage[outPos * 2 + 1] = idxMC;
                outPos = outPos + 1;
            }
            poff = poff + 2;
        }
    }
}

fn lcg_random(stateRef: ptr<function, u32>) -> u32 {
    // replicate: state = 1664525u * state + 1013904223u
    let newVal = (1664525u * (*stateRef)) + 1013904223u;
    *stateRef = newVal;
    return newVal;
}

fn lcg_randomRange(stateRef: ptr<function, u32>, range: u32) -> u32 {
    return lcg_random(stateRef) % range;
}

fn sampleTopDown(
    dp_in       : array<u16>,
    bpCount     : array<i32>,
    bpOffset    : array<i32>,
    bpStorage   : array<i32>,
    startDPIdx  : i32,
    rngStateRef : ptr<function, u32>,
    localWord   : ptr<function, array<u16, 1024>>,
    maxWordLen  : i32
) {
    // We'll create a stack up to 1024
    var stack: array<i32, 1024>;
    var top    = 0;
    var wordLen= 0;

    stack[top] = startDPIdx;
    top = top + 1;

    // We interpret uni.numNonterminals, etc., as global from `uni`.
    let nnt = uni.numNonterminals;

    // main loop
    var iter = 0;
    loop {
        if ((iter >= maxWordLen * 98) || (top <= 0) || (wordLen >= maxWordLen - 5)) { break; }
        iter = iter + 1;

        top = top - 1;
        let dpIdx = abs(stack[top]);

        let expCount = bpCount[dpIdx];
        // dp_in[dpIdx] >> 1 => checks high bits
        // dp_in[dpIdx] & 0x01 => checks the binary expansion bit
        let predicate = dp_in[dpIdx];
        let canRecurse = ((predicate & 0x01u) != 0u);

        // If ( (predicate >> 1u) != 0 ) => the node has some “literal” bits set
        // Or we randomly choose to treat it as a leaf ...
        let hasLiteral = ((predicate >> 1u) != 0u);

        // Random check:
        let rVal = i32(lcg_randomRange(rngStateRef, u32(expCount))); 
        // Original logic:
        // if ( (dp_in[dpIdx] >> 1) && ( ! (dp_in[dpIdx] & 0x01) || (rVal % 2 == 0) ) ) { ... }
        if (hasLiteral && ( !canRecurse || ((rVal % 2) == 0) )) {
            // treat as leaf/terminal
            let nonterminal     = dpIdx % nnt;
            let isNegative      = ((predicate & 0x8000u) != 0u);
            let literal         = (predicate >> 1u) & 0x7FFFu;
            let numTms         = nt_tm_lens[nonterminal];
            let ntOffset       = offsets[nonterminal];

            if (isNegative) {
                // choose from among all possible except “literal - 1”
                var possibleTms: array<u16, 100>;
                var tmCount = 0;
                for (var i = 0; i < min(numTms, 100); i = i + 1) {
                    if (i != (literal - 1u)) {
                        possibleTms[tmCount] = all_tm[i32(ntOffset) + i];
                        tmCount = tmCount + 1;
                    }
                }
                let choiceIndex = i32(lcg_randomRange(rngStateRef, u32(tmCount)));
                let tmChoice    = possibleTms[choiceIndex];
                (*localWord)[wordLen] = tmChoice + 1u; // your original code
                wordLen = wordLen + 1;
            } else {
                // positive literal
                if (numTms != 0) {
                    let tmVal = all_tm[i32(ntOffset) + i32(literal) - 1];
                    (*localWord)[wordLen] = tmVal + 1u;
                } else {
                    (*localWord)[wordLen] = 99u; // fallback
                }
                wordLen = wordLen + 1;
            }
        } else {
            // do a binary expansion if there's room on the stack
            if (top + 2 < 1024) {
                let randIdx = bpOffset[dpIdx] + rVal;
                let idxBM   = bpStorage[randIdx * 2 + 0];
                let idxMC   = bpStorage[randIdx * 2 + 1];
                stack[top] = idxMC;
                top = top + 1;
                stack[top] = idxBM;
                top = top + 1;
            }
        }
    }
}

@compute @workgroup_size(64)
fn sample_words(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;

    // If needed, clamp to numStartIndices:
    if (tid >= u32(samplerUni.numStartIndices)) { return; }

    // We'll create a local array in function scope
    var localWord: array<u16, 1024>;
    // Zero-initialize
    for (var i = 0; i < samplerUni.maxWordLen; i = i + 1) { localWord[i] = 0u; }

    // Grab rngState from seeds[tid]
    var rngState = seeds[tid];

    // Choose a random index in [0..numStartIndices-1]
    let rIndex   = lcg_randomRange(&rngState, u32(samplerUni.numStartIndices));
    let dpIndex  = startIndices[i32(rIndex)];

    // decode row & col if needed:
    //    A = dpIndex % numNonterminals
    //    rowCol = dpIndex / numNonterminals
    //    c = rowCol % numStates
    //    r = rowCol / numStates
    //    then startIdx = r*snt + c*numNonterminals + A
    let nnt   = uni.numNonterminals;
    let A     = dpIndex % nnt;
    let rowCol= dpIndex / nnt;
    let c     = rowCol % uni.numStates;
    let r     = rowCol / uni.numStates;
    let snt   = uni.numStates * nnt;
    let startIdx = r * snt + c * nnt + A;

    // Now call sampleTopDown
    sampleTopDown(dp_in, bpCount, bpOffset, bpStorage,
                  startIdx, &rngState, &localWord, samplerUni.maxWordLen);

    // Copy to sampledWords buffer
    // dp_in was 16 bits, but in original code we wrote into a “char”
    // so we do a cast from u16 -> u8 (truncation).
    for (var i = 0; i < samplerUni.maxWordLen; i = i + 1) {
        sampledWords[u32(tid) * u32(samplerUni.maxWordLen) + u32(i)]
            = u8(localWord[i] & 0x00FFu);
    }
}

@compute @workgroup_size(1024)
fn prefix_sum_p1(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(workgroup_id)         groupId : vec3<u32>,
    @builtin(local_invocation_id)  localId : vec3<u32>
) {
    let N       = prefixUni.N; // total length
    let gid     = globalId.x;
    let grpId   = groupId.x;
    let lid     = localId.x;
    let tpg     = 1024u; // fixed to match @workgroup_size(1024)

    var<workgroup> tile: array<i32, 1024>;

    let base = grpId * tpg;
    let idx  = base + lid;

    let v = if (idx < N) { outBuf[idx] } else { 0 };
    tile[lid] = v;
    workgroupBarrier();

    var offset = 1u;
    while (offset < tpg) {
        let n = if (lid >= offset) { tile[lid - offset] } else { 0 };
        workgroupBarrier();
        tile[lid] = tile[lid] + n;
        workgroupBarrier();

        offset = offset << 1;
    }

    let inclusive = tile[lid];
    tile[lid] = inclusive - v;

    // If last thread in this group or last element in entire array
    if ((idx + 1u == N) || (lid == tpg - 1u)) { blkSum[grpId] = inclusive; }

    if (idx < N) { outBuf[idx] = tile[lid]; }
}

@compute @workgroup_size(1024)
fn prefix_sum_p2(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(workgroup_id)         groupId : vec3<u32>,
    @builtin(local_invocation_id)  localId : vec3<u32>
) {
    let N     = prefixUni.N;
    let grpId = groupId.x;
    let lid   = localId.x;
    let tpg   = 1024u;

    if (grpId == 0u) { return; }

    let offsetVal = blkSum[grpId - 1u];
    let idx = grpId * tpg + lid;

    if (idx < N) { outBuf[idx] = outBuf[idx] + offsetVal; }
}"""

/**
 * Perform a Bar-Hillel intersection / "CYK-like" step on GPU using exponentiation-by-squaring.
 * - chart is a Q x Q x V 3D boolean array (dp table).
 * - grammar[A] = list of (B, C) pairs for nonterminal A.
 * - allFSA[p][q] = list of midpoint states r between p and q in the underlying acyclic FSA.
 *
 * Steps:
 *  1) Flatten the 3D chart into a 1D Uint32Array (1 for true, 0 for false).
 *  2) Flatten the grammar data:
 *       grammarFlat[A*2]   = offset into "BC pairs" portion
 *       grammarFlat[A*2+1] = number of (B, C) pairs
 *       // followed by the pairs themselves
 *  3) Flatten the allFSA data:
 *       allFSAFlat[(p*Q+q)*2]   = offset into "midpoints" portion
 *       allFSAFlat[(p*Q+q)*2+1] = number of midpoints
 *       // followed by the midpoints themselves
 *  4) Create compute pipelines for mainSquare (chartTmp = chart^2) and mainOr (chart |= chartTmp).
 *  5) Repeat ~log2(Q) times the sequence of "square" + "or".
 *  6) Copy the final chart back to the CPU and return it as a 3D List<List<MutableList<Boolean>>>.
 */

fun wgpuBarHillelIntersection(
  chart: Array<Array<BooleanArray>>,
  grammar: Array<IntArray>,
  allFSA: List<List<List<Int>>>
): dynamic /* Promise<List<List<List<Boolean>>>> */ {
  val gpu = navigator.gpu ?: run {
    console.warn("WebGPU not available.")
    return null
  }

  val Q = chart.size
  if (Q == 0) return js("Promise.resolve(chart)")
  val V = chart[0][0].size
  val totalChartSize = Q * Q * V

  // Flatten the chart
  val chartFlat = js("new Uint32Array(totalChartSize)")
  var idx = 0
  for (p in 0 until Q) {
    for (q in 0 until Q) {
      for (A in 0 until V) {
        chartFlat[idx++] = if (chart[p][q][A]) 1u else 0u
      }
    }
  }

  /**
   * Flatten the grammar:
   *   grammarFlat[0..2*V-1] contains offsets/lengths (two per A),
   *   then all (B,C) pairs in sequence.
   */
  var totalPairs = 0
  for (A in 0 until V) {
    // Each grammar[A] is a list of B,C repeated, so # of pairs = grammar[A].size / 2
    totalPairs += grammar[A].size
  }

  val grammarFlatSize = 2*V + totalPairs // (offset,len) for each A plus all pairs
  val grammarFlat = js("new Uint32Array(grammarFlatSize)")
  var pairPos = (2 * V) // where pairs start
  for (A in 0 until V) {
    val offset = pairPos
    val pairCount = grammar[A].size / 2
    grammarFlat[A*2 + 0] = offset
    grammarFlat[A*2 + 1] = pairCount
    // copy pairs
    for (bci in grammar[A].indices) {
      grammarFlat[pairPos++] = grammar[A][bci].toUInt()
    }
  }

  /**
   * Flatten the allFSA:
   *   allFSAFlat[0..2*Q*Q - 1] contains offsets/lengths (two per (p,q)),
   *   then all midpoints in sequence.
   */
  var totalMidpoints = 0
  for (p in 0 until Q) {
    for (q in 0 until Q) {
      totalMidpoints += allFSA[p][q].size
    }
  }

  val fsaHeaderSize = 2 * Q * Q
  val allFSAFlatSize = fsaHeaderSize + totalMidpoints
  val allFSAFlat = js("new Uint32Array(allFSAFlatSize)")
  var midPos = fsaHeaderSize
  for (p in 0 until Q) {
    for (q in 0 until Q) {
      val offset = midPos
      val count = allFSA[p][q].size
      val idxPQ = (p*Q + q) * 2
      allFSAFlat[idxPQ + 0] = offset
      allFSAFlat[idxPQ + 1] = count
      for (r in allFSA[p][q]) {
        allFSAFlat[midPos++] = r.toUInt()
      }
    }
  }

  // Byte sizes
  val bytesChart   = totalChartSize * 4
  val bytesTemp    = totalChartSize * 4
  val bytesGrammar = grammarFlatSize * 4
  val bytesAllFSA  = allFSAFlatSize * 4
  val bytesParams  = 8 /* Q & V (2 x uint32) -> 8 bytes, must be multiple of 16 => pad to 16 */
  val paramBytesAligned = 16

  // We'll do ~log2(Q) exponentiation passes
  val steps = kotlin.math.ceil(kotlin.math.log2(Q.coerceAtLeast(1).toDouble())).toInt().coerceAtLeast(1)

  return gpu.requestAdapter().then { adapter: dynamic -> adapter.requestDevice() }
    .then { device: dynamic ->
    // Create buffers
    val bufChart = createBuffer(device, bytesChart, 128+8)  // STORAGE + COPY_DST
    val bufTemp  = createBuffer(device, bytesTemp, 128+8)  // STORAGE + COPY_DST
    val bufGrammar = createBuffer(device, bytesGrammar, 128) // STORAGE
    val bufAllFSA  = createBuffer(device, bytesAllFSA, 128)  // STORAGE
    val bufParam   = createBuffer(device, paramBytesAligned, 64) // UNIFORM

    // Write data
    device.queue.writeBuffer(bufChart, 0, chartFlat)
    device.queue.writeBuffer(bufGrammar, 0, grammarFlat)
    device.queue.writeBuffer(bufAllFSA, 0, allFSAFlat)

    // Create param uniform data
    val paramData = js("new Uint32Array(2)") // We'll upload 2 uints, but the buffer is 16 bytes
    paramData[0] = Q
    paramData[1] = V
    device.queue.writeBuffer(bufParam, 0, paramData)

    // Create shader module
    val modDesc = js("{}")
    modDesc.code = WGSL_BAR_HILLEL
    val shaderModule = device.createShaderModule(modDesc)

    // Pipelines
    val pipeSquareDesc = js("{}")
    pipeSquareDesc.layout = "auto"
    val compSquareStage = js("{}")
    compSquareStage.module = shaderModule
    compSquareStage.entryPoint = "mainSquare"
    pipeSquareDesc.compute = compSquareStage
    val pipelineSquare = device.createComputePipeline(pipeSquareDesc)

    val pipeOrDesc = js("{}")
    pipeOrDesc.layout = "auto"
    val compOrStage = js("{}")
    compOrStage.module = shaderModule
    compOrStage.entryPoint = "mainOr"
    pipeOrDesc.compute = compOrStage
    val pipelineOr = device.createComputePipeline(pipeOrDesc)

    // Bind groups
    val bgSquareDesc = js("{}")
    bgSquareDesc.layout = pipelineSquare.getBindGroupLayout(0)
    bgSquareDesc.entries = arrayOf<dynamic>(
      js("{\"binding\":0,\"resource\":{\"buffer\":bufChart}}"),
      js("{\"binding\":1,\"resource\":{\"buffer\":bufGrammar}}"),
      js("{\"binding\":2,\"resource\":{\"buffer\":bufAllFSA}}"),
      js("{\"binding\":3,\"resource\":{\"buffer\":bufTemp}}"),
      js("{\"binding\":4,\"resource\":{\"buffer\":bufParam}}")
    )
    val bindGroupSquare = device.createBindGroup(bgSquareDesc)

    val bgOrDesc = js("{}")
    bgOrDesc.layout = pipelineOr.getBindGroupLayout(0)
    bgOrDesc.entries = arrayOf<dynamic>(
      js("{\"binding\":0,\"resource\":{\"buffer\":bufChart}}"),
      js("{\"binding\":3,\"resource\":{\"buffer\":bufTemp}}"),
      js("{\"binding\":4,\"resource\":{\"buffer\":bufParam}}")
    )
    val bindGroupOr = device.createBindGroup(bgOrDesc)

    // Repeated "square + or"
    repeat(steps) {
      val encoder = device.createCommandEncoder()

      // Square pass
      val passSquare = encoder.beginComputePass()
      passSquare.setPipeline(pipelineSquare)
      passSquare.setBindGroup(0, bindGroupSquare)
      passSquare.dispatchWorkgroups(Q.toDouble(), Q.toDouble(), V.toDouble())
      passSquare.end()

      // Or pass
      val passOr = encoder.beginComputePass()
      passOr.setPipeline(pipelineOr)
      passOr.setBindGroup(0, bindGroupOr)
      passOr.dispatchWorkgroups(Q.toDouble(), Q.toDouble(), V.toDouble())
      passOr.end()

      device.queue.submit(arrayOf(encoder.finish()))
    }

    // Copy final chart back to CPU
    val bufStaging = createBuffer(device, bytesChart, 1+8) // MAP_READ + COPY_DST
    val cenc = device.createCommandEncoder()
    cenc.copyBufferToBuffer(bufChart, 0, bufStaging, 0, bytesChart)
    device.queue.submit(arrayOf(cenc.finish()))

    // Map & reconstruct the 3D chart
    bufStaging.mapAsync(1).then {
      val mapped = bufStaging.getMappedRange()
      val outU32 = js("new Uint32Array(mapped)")
      val finalChart = List(Q) {
        List(Q) { MutableList(V) { false } }
      }
      var idx2 = 0
      for (p in 0 until Q) {
        for (q in 0 until Q) {
          for (A in 0 until V) {
            finalChart[p][q][A] = (outU32[idx2++] == 1u)
          }
        }
      }

      println("Parseable: " + finalChart[0][finalChart.size-1][0])
      finalChart
    }
  }
}

/** Helper to create a GPU buffer with the specified size & usage flags. */
fun createBuffer(device: dynamic, size: Int, usage: Int): dynamic {
  val desc = js("{}")
  desc.size = size
  desc.usage = usage
  return device.createBuffer(desc)
}