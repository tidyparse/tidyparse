import Shader.Companion.readIndices
import Shader.Companion.toGPUBuffer
import web.gpu.GPUBuffer

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