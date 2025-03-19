
struct Params {
    Q : u32,  // number of states
    V : u32,  // number of nonterminals
};

// chart   -> read_write
// grammar -> read
// allFSA  -> read
// chartTmp-> read_write
// param   -> uniform

@group(0) @binding(0) var<storage, read_write> chart : array<u32>;
@group(0) @binding(1) var<storage, read>       grammar : array<u32>;
@group(0) @binding(2) var<storage, read>       allFSA  : array<u32>;
@group(0) @binding(3) var<storage, read_write> chartTmp : array<u32>;
@group(0) @binding(4) var<uniform>             param  : Params;

/**
 * mainSquare:
 *   chartTmp[p,q,A] = OR_{r in allFSA[p,q], (B,C) in grammar(A)} ( chart[p,r,B] && chart[r,q,C] )
 */
@compute @workgroup_size(8, 8, 1)
fn mainSquare(@builtin(global_invocation_id) gid : vec3<u32>) {
    let p = gid.x;
    let q = gid.y;
    let A = gid.z;

    let Q = param.Q;
    let V = param.V;

    // Bounds check
    if (p >= Q || q >= Q || A >= V) { return; }

    // Flatten index for chart
    let idx = (p * Q + q) * V + A;

    // Read offset & count of midpoints for (p,q)
    let pqIndex    = (p * Q + q) * 2u;
    let midOffset  = allFSA[pqIndex + 0u];
    let midCount   = allFSA[pqIndex + 1u];

    // Read offset & count of (B,C) pairs for A
    let gramIndex  = A * 2u;
    let gramOffset = grammar[gramIndex + 0u];
    let gramCount  = grammar[gramIndex + 1u];

    var result : bool = false;

    // For each midpoint r
    for (var i = 0u; i < midCount; i = i + 1u) {
        let r = allFSA[midOffset + i];
        // For each (B, C) pair
        for (var j = 0u; j < gramCount; j = j + 1u) {
            let b = grammar[gramOffset + 2u*j];
            let c = grammar[gramOffset + 2u*j + 1u];

            let idxBR = (p * Q + r) * V + b;
            let idxRC = (r * Q + q) * V + c;

            // If both positions are 1u, we found a match
            if ((chart[idxBR] == 1u) && (chart[idxRC] == 1u)) {
                result = true;
                break;
            }
        }
        if (result) { break; }
    }

    let storeVal = select(0u, 1u, result);
    chartTmp[idx] = storeVal;
}

/**
 * mainOr:
 *   chart[p,q,A] |= chartTmp[p,q,A]
 */
@compute @workgroup_size(8, 8, 1)
fn mainOr(@builtin(global_invocation_id) gid : vec3<u32>) {
    let p = gid.x;
    let q = gid.y;
    let A = gid.z;

    let Q = param.Q;
    let V = param.V;

    // Bounds check
    if (p >= Q || q >= Q || A >= V) { return; }

    let idx = (p * Q + q) * V + A;
    let valC = chart[idx];
    let valT = chartTmp[idx];

    // Boolean or: if either is 1u, result is 1u
    let cond = (valC == 1u || valT == 1u);
    let orVal = select(0u, 1u, cond);

    chart[idx] = orVal;
}