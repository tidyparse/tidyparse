import ai.hypergraph.kaliningraph.parsing.tmMap
import ai.hypergraph.kaliningraph.repair.s2pg
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import js.buffer.ArrayBuffer
import kotlinx.browser.window
import kotlinx.coroutines.*
import kotlinx.coroutines.await
import kotlin.js.Promise
import kotlin.math.min
import kotlin.time.TimeSource

private const val RERANKER_WEIGHTS = "./reranker_2000.q8.safetensors"
private const val RERANKER_MAX_LEN_Q = 100
private const val RERANKER_MAX_LEN_D = 110
private const val RERANKER_MAX_LEN = RERANKER_MAX_LEN_Q + RERANKER_MAX_LEN_D + 2
private const val RERANKER_DOCS_PER_CALL = 64
// Must match reranker_validation.html/train_reranker.py:
// s2pg terminals occupy ids 0..89, with BOS/EOS encoded as 91/92.
private const val RERANKER_CLS_Q = 90
private const val RERANKER_CLS_D = 91
private const val RERANKER_BOS = 91
private const val RERANKER_EOS = 92
private const val RERANKER_ASCII_OFFSET = 33

var neuralRerankerEnabled = false

fun neuralRerankerQuery(tokens: List<String>): List<String>? =
  if (neuralRerankerEnabled) tokens else null

object RepairReranker {
  private val scope = MainScope()
  private var netReady: Deferred<dynamic>? = null
  private val tokenIds by lazy { s2pg.tmMap }

  fun preload() { if (netReady == null) netReady = scope.async { loadNet() } }

  suspend fun preloadAvailable(): Boolean =
    try {
      net()
      true
    } catch (t: Throwable) {
      netReady = null
      log("Reranker unavailable on page load: ${t.message ?: t}")
      false
    }

  suspend fun rerankOrOriginal(query: List<String>, candidates: List<String>): List<String> =
    try { if (candidates.size <= 1) candidates else rerank(query, candidates) }
    catch (t: Throwable) { log("Reranker unavailable, keeping decoder order: ${t.message ?: t}"); candidates }

  private suspend fun rerank(query: List<String>, candidates: List<String>): List<String> {
    val t0 = TimeSource.Monotonic.markNow()
    val encodedQuery = encodeTokens(query, RERANKER_MAX_LEN_Q)
      ?: return candidates.also { log("Reranker skipped: query has tokens outside s2pg") }

    val encodedDocs = candidates.map { repair ->
      repair.tokenizeByWhitespace().let { encodeTokens(it, RERANKER_MAX_LEN_D) }
    }

    val missingIdx = encodedDocs.indexOfFirst { it == null }
    if (missingIdx >= 0) {
      log("Reranker skipped: candidate $missingIdx has tokens outside s2pg")
      return candidates
    }

    val net = net()
    val scores = scoreEncoded(net, encodedQuery, encodedDocs.filterNotNull())
    val order = candidates.indices.sortedWith { a, b ->
      val byScore = scores[b].compareTo(scores[a])
      if (byScore != 0) byScore else a.compareTo(b)
    }

    return order.map { candidates[it] }.also {
      log("Reranked ${candidates.size} repairs in ${t0.elapsedNow()}")
    }
  }

  private suspend fun net(): dynamic = (netReady ?: scope.async { loadNet() }.also { netReady = it }).await()

  private suspend fun loadNet(): dynamic {
    val t0 = TimeSource.Monotonic.markNow()
    val moduleUrl = moduleUrl()
    val imported = js("Function('url', 'return import(url)')(moduleUrl)")
      .unsafeCast<Promise<dynamic>>()
      .await()
    val setupNet = imported.default.setupNet
      ?: error("Generated reranker module does not export setupNet()")

    val weights = loadWeights()
    val net = setupNet(gpu, weights).unsafeCast<Promise<dynamic>>().await()
    log("Loaded reranker in ${t0.elapsedNow()}")
    return net
  }

  private fun moduleUrl(): String {
    val modelJs = RERANKER_2000_JS
    return js("""URL.createObjectURL(new Blob([modelJs], { type: "text/javascript" }))""") as String
  }

  private suspend fun loadWeights(): dynamic {
    val inlineGzipWeights = window.asDynamic().raw_reranker_weights_gzip_b64 as? String
    if (!inlineGzipWeights.isNullOrBlank()) {
      val buffer = gzipBase64ToArrayBuffer(inlineGzipWeights)
      return materializeF32Safetensors(buffer)
    }

    val inlineWeights = window.asDynamic().raw_reranker_weights_b64 as? String
    if (!inlineWeights.isNullOrBlank()) {
      val buffer = base64ToArrayBuffer(inlineWeights)
      return materializeF32Safetensors(buffer)
    }

    val response = window.fetch(browserResourceUrl(RERANKER_WEIGHTS)).await()
    if (!response.ok) error("Failed to load $RERANKER_WEIGHTS")
    val buffer = response.arrayBuffer().await().unsafeCast<ArrayBuffer>()
    return materializeF32Safetensors(buffer)
  }

  private fun browserResourceUrl(path: String): String =
    js("new URL(path, document.baseURI).href") as String

  private fun base64ToArrayBuffer(b64: String): ArrayBuffer =
    js("""(function(s) {
      const binary = atob(s);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    })(b64)""").unsafeCast<ArrayBuffer>()

  private suspend fun gzipBase64ToArrayBuffer(b64: String): ArrayBuffer =
    js("""(function(s) {
      const binary = atob(s);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).arrayBuffer();
    })(b64)""").unsafeCast<Promise<ArrayBuffer>>().await()

  private fun materializeF32Safetensors(rawSafetensors: ArrayBuffer): dynamic =
    js("""(function(buffer) {
      function fail(message) { throw new Error(message); }
      function assert(condition, message) { if (!condition) fail(message); }
      function tensorElementCount(shape) {
        assert(Array.isArray(shape), "Tensor shape is not an array.");
        return shape.reduce((total, dim) => total * Number(dim), 1);
      }
      function dtypeByteSize(dtype) {
        if (dtype === "F32") return 4;
        if (dtype === "F16") return 2;
        if (dtype === "I8" || dtype === "U8") return 1;
        fail("Unsupported safetensors dtype for browser reranker: " + dtype);
      }
      function float16ToNumber(bits) {
        const sign = bits & 0x8000 ? -1 : 1;
        const exponent = (bits >>> 10) & 0x1f;
        const fraction = bits & 0x03ff;
        if (exponent === 0) return sign * (fraction ? Math.pow(2, -14) * (fraction / 1024) : 0);
        if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
        return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
      }

      assert(buffer.byteLength >= 8, "Safetensors file is shorter than its 8-byte header.");
      const view = new DataView(buffer);
      const headerBig = view.getBigUint64(0, true);
      assert(headerBig <= BigInt(Number.MAX_SAFE_INTEGER), "Safetensors header is too large.");
      const headerLength = Number(headerBig);
      const dataStart = 8 + headerLength;
      assert(dataStart <= buffer.byteLength, "Safetensors header extends past the end of the file.");

      let metadata;
      try {
        metadata = JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(buffer, 8, headerLength)));
      } catch (error) {
        throw new Error("Invalid safetensors metadata JSON: " + error.message);
      }

      const sourceMetadata = metadata.__metadata__ || {};
      const quantization = sourceMetadata["cstk.quantization"] || "";
      const tensorNames = Object.keys(metadata).filter(name => name !== "__metadata__");

      for (const name of tensorNames) {
        const entry = metadata[name];
        const offsets = entry.data_offsets;
        assert(Array.isArray(offsets) && offsets.length === 2, "Bad data offsets for weight: " + name);
        assert(dataStart + offsets[1] <= buffer.byteLength, "Weight extends past EOF: " + name);
        const byteLength = offsets[1] - offsets[0];
        const expectedBytes = tensorElementCount(entry.shape) * dtypeByteSize(entry.dtype);
        assert(byteLength === expectedBytes, name + " has " + byteLength + " data bytes; expected " + expectedBytes + " for " + entry.dtype + ".");

        if (!quantization) {
          assert(entry.dtype === "F32", name + " is " + entry.dtype + "; unquantized generated WebGPU weights must be F32.");
        } else if (quantization === "q8_symmetric_per_tensor") {
          assert(entry.dtype === "I8", name + " is " + entry.dtype + "; q8 transfer weights must be I8.");
          const scale = Number(sourceMetadata["cstk.quant." + name + ".scale"]);
          assert(Number.isFinite(scale) && scale > 0, "Missing or invalid q8 scale for " + name + ".");
        } else if (quantization === "f16") {
          assert(entry.dtype === "F16", name + " is " + entry.dtype + "; f16 transfer weights must be F16.");
        } else {
          fail("Unsupported safetensors quantization format: " + quantization);
        }
      }

      if (!quantization) return new Uint8Array(buffer);

      const outHeader = {"__metadata__": {}};
      for (const key of Object.keys(sourceMetadata)) outHeader.__metadata__[key] = sourceMetadata[key];
      outHeader.__metadata__["cstk.materialized_dtype"] = "F32";
      outHeader.__metadata__["cstk.materialized_from_quantization"] = quantization;
      const chunks = [];
      let cursor = 0;

      for (const name of tensorNames) {
        const entry = metadata[name];
        const count = tensorElementCount(entry.shape);
        const begin = dataStart + entry.data_offsets[0];
        const values = new Float32Array(count);

        if (quantization === "q8_symmetric_per_tensor") {
          const scale = Number(sourceMetadata["cstk.quant." + name + ".scale"]);
          const packed = new Int8Array(buffer, begin, count);
          for (let i = 0; i < count; i++) values[i] = packed[i] * scale;
        } else if (quantization === "f16") {
          const view = new DataView(buffer, begin, count * 2);
          for (let i = 0; i < count; i++) values[i] = float16ToNumber(view.getUint16(i * 2, true));
        }

        const bytes = new Uint8Array(values.buffer);
        outHeader[name] = {
          dtype: "F32",
          shape: entry.shape,
          data_offsets: [cursor, cursor + bytes.byteLength],
        };
        chunks.push(bytes);
        cursor += bytes.byteLength;
      }

      const headerBytes = new TextEncoder().encode(JSON.stringify(outHeader));
      const out = new Uint8Array(8 + headerBytes.byteLength + cursor);
      new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.byteLength), true);
      out.set(headerBytes, 8);
      let offset = 8 + headerBytes.byteLength;
      for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return out;
    })(rawSafetensors)""")

  private fun encodeTokens(tokens: List<String>, maxLength: Int): String? {
    val ids = ArrayList<Int>(tokens.size + 2)
    ids.add(RERANKER_BOS)
    for (token in tokens) ids.add(tokenIds[token] ?: return null)
    ids.add(RERANKER_EOS)

    val end = min(maxLength, ids.size)
    return buildString(end) {
      for (i in 0 until end) append((ids[i] + RERANKER_ASCII_OFFSET).toChar())
    }
  }

  private suspend fun scoreEncoded(net: dynamic, query: String, docs: List<String>): FloatArray {
    val queryInput = buildQueryInput(query)
    val queryAlignment = int32Array(RERANKER_MAX_LEN)
    val allScores = FloatArray(docs.size)
    val docInput = int32Array(RERANKER_DOCS_PER_CALL * RERANKER_MAX_LEN)
    val docAlignment = int32Array(RERANKER_DOCS_PER_CALL * RERANKER_MAX_LEN)

    var base = 0
    while (base < docs.size) {
      val count = min(RERANKER_DOCS_PER_CALL, docs.size - base)
      zero(docInput)
      zero(docAlignment)

      for (row in 0 until RERANKER_DOCS_PER_CALL) {
        writeDocumentRow(
          inputIds = docInput,
          alignmentIds = docAlignment,
          row = row,
          query = query,
          doc = if (row < count) docs[base + row] else ""
        )
      }

      // The generated wrapper currently exports `async (_input2, _input3, _input0, _input1)`.
      val result = net(docInput, docAlignment, queryInput, queryAlignment)
        .unsafeCast<Promise<dynamic>>()
        .await()
      val chunkScores = unpackScores(result)
      for (i in 0 until count) allScores[base + i] = (chunkScores[i] as Number).toFloat()
      base += count
    }

    return allScores
  }

  private fun buildQueryInput(query: String): JSIntArray {
    val inputIds = int32Array(RERANKER_MAX_LEN)
    inputIds[0] = RERANKER_CLS_Q
    writeEncodedString(inputIds, 1, query, RERANKER_MAX_LEN_Q)
    inputIds[RERANKER_MAX_LEN_Q + 1] = RERANKER_CLS_D
    return inputIds
  }

  private fun writeDocumentRow(
    inputIds: JSIntArray,
    alignmentIds: JSIntArray,
    row: Int,
    query: String,
    doc: String
  ) {
    val offset = row * RERANKER_MAX_LEN
    inputIds[offset] = RERANKER_CLS_Q
    writeEncodedString(inputIds, offset + 1, query, RERANKER_MAX_LEN_Q)
    inputIds[offset + RERANKER_MAX_LEN_Q + 1] = RERANKER_CLS_D
    writeEncodedString(inputIds, offset + RERANKER_MAX_LEN_Q + 2, doc, RERANKER_MAX_LEN_D)

    val alignment = levAlign(query.take(RERANKER_MAX_LEN_Q), doc.take(RERANKER_MAX_LEN_D))
    for (i in alignment.indices) alignmentIds[offset + RERANKER_MAX_LEN_Q + 2 + i] = alignment[i]
  }

  private fun writeEncodedString(dst: JSIntArray, offset: Int, encoded: String, maxLength: Int) {
    val end = min(maxLength, encoded.length)
    for (i in 0 until end) dst[offset + i] = encoded[i].code - RERANKER_ASCII_OFFSET
  }

  private fun levAlign(query: String, doc: String): IntArray {
    val m = query.length
    val n = doc.length
    val width = n + 1
    val dp = IntArray((m + 1) * (n + 1))

    for (i in 0..m) dp[i * width] = i
    for (j in 0..n) dp[j] = j

    for (i in 1..m) {
      val row = i * width
      val prev = (i - 1) * width
      val qi = query[i - 1]
      for (j in 1..n) {
        val substitutionCost = if (qi == doc[j - 1]) 0 else 2
        val deletion = dp[prev + j] + 1
        val insertion = dp[row + j - 1] + 1
        val substitution = dp[prev + j - 1] + substitutionCost
        dp[row + j] = minOf(deletion, insertion, substitution)
      }
    }

    val alignment = IntArray(n)
    var i = m
    var j = n
    while (i > 0 || j > 0) {
      val current = dp[i * width + j]
      if (i > 0 && j > 0) {
        val substitutionCost = if (query[i - 1] == doc[j - 1]) 0 else 2
        if (current == dp[(i - 1) * width + j - 1] + substitutionCost) {
          alignment[j - 1] = substitutionCost
          i--
          j--
          continue
        }
      }
      if (j > 0 && (i == 0 || current == dp[i * width + j - 1] + 1)) {
        alignment[j - 1] = 1
        j--
        continue
      }
      i--
    }

    return alignment
  }

  private fun unpackScores(result: dynamic): dynamic {
    if (js("Array.isArray(result)") as Boolean) {
      if (result.length == 1 && (js("ArrayBuffer.isView(result[0])") as Boolean)) return result[0]
      if (result.length > 0 && js("typeof result[0] === 'number'") as Boolean) return result
    }
    if (js("ArrayBuffer.isView(result)") as Boolean) return result
    error("Could not unpack reranker scores")
  }

  private fun zero(array: JSIntArray) { js("array.fill(0)") }

  private fun int32Array(size: Int): JSIntArray =
    js("new Int32Array(size)").unsafeCast<JSIntArray>()
}


//language=js
val RERANKER_2000_JS = """
const model = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const E_2_212_4_8_16_4_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3473408:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_13568:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_24064:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_54272:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_13568:array<i32>;
@group(0) @binding(6)var<storage,read_write>data5_1024:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 212 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (gidx1+(gidx2*6784)+(lidx0*848));
  var val0 = data1_13568[alu0];
  var alu1 = (alu0+212);
  var val1 = data1_13568[alu1];
  var alu2 = (alu0+424);
  var val2 = data1_13568[alu2];
  var alu3 = (alu0+636);
  var val3 = data1_13568[alu3];
  var val4 = data4_13568[alu0];
  var val5 = data4_13568[alu1];
  var val6 = data4_13568[alu2];
  var val7 = data4_13568[alu3];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu4 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu5 = (alu4+bitcast<i32>((bitcast<u32>(val0)<<8u)));
  var alu6 = ((-1<val0)&(val0<94));
  var val8 = select(0.0f, data2_24064[alu5], alu6);
  var alu7 = (alu4+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val9 = data3_54272[alu7];
  var alu8 = (alu4+bitcast<i32>((bitcast<u32>(val4)<<8u)));
  var alu9 = ((-1<val4)&(val4<4));
  var val10 = select(0.0f, data5_1024[alu8], alu9);
  var val11 = select(0.0f, data2_24064[(alu5+1)], alu6);
  var val12 = data3_54272[(alu7+1)];
  var val13 = select(0.0f, data5_1024[(alu8+1)], alu9);
  var val14 = select(0.0f, data2_24064[(alu5+2)], alu6);
  var val15 = data3_54272[(alu7+2)];
  var val16 = select(0.0f, data5_1024[(alu8+2)], alu9);
  var val17 = select(0.0f, data2_24064[(alu5+3)], alu6);
  var val18 = data3_54272[(alu7+3)];
  var alu10 = (alu4+bitcast<i32>((bitcast<u32>(val1)<<8u)));
  var alu11 = ((-1<val1)&(val1<94));
  var val19 = select(0.0f, data2_24064[alu10], alu11);
  var alu12 = (alu4+bitcast<i32>((bitcast<u32>(val5)<<8u)));
  var alu13 = ((-1<val5)&(val5<4));
  var val20 = select(0.0f, data5_1024[alu12], alu13);
  var val21 = select(0.0f, data2_24064[(alu10+2)], alu11);
  var val22 = select(0.0f, data5_1024[(alu12+2)], alu13);
  var val23 = select(0.0f, data2_24064[(alu10+3)], alu11);
  var val24 = select(0.0f, data5_1024[(alu12+3)], alu13);
  var alu14 = (alu4+bitcast<i32>((bitcast<u32>(val2)<<8u)));
  var alu15 = ((-1<val2)&(val2<94));
  var val25 = select(0.0f, data2_24064[alu14], alu15);
  var alu16 = (alu4+bitcast<i32>((bitcast<u32>(val6)<<8u)));
  var alu17 = ((-1<val6)&(val6<4));
  var val26 = select(0.0f, data5_1024[alu16], alu17);
  var val27 = select(0.0f, data5_1024[(alu8+3)], alu9);
  var val28 = select(0.0f, data2_24064[(alu14+1)], alu15);
  var val29 = select(0.0f, data5_1024[(alu16+1)], alu17);
  var alu18 = (alu4+bitcast<i32>((bitcast<u32>(val3)<<8u)));
  var alu19 = ((-1<val3)&(val3<94));
  var val30 = select(0.0f, data2_24064[alu18], alu19);
  var val31 = select(0.0f, data2_24064[(alu10+1)], alu11);
  var val32 = select(0.0f, data2_24064[(alu14+2)], alu15);
  var val33 = select(0.0f, data2_24064[(alu14+3)], alu15);
  var alu20 = (alu4+bitcast<i32>((bitcast<u32>(val7)<<8u)));
  var alu21 = ((-1<val7)&(val7<4));
  var val34 = select(0.0f, data5_1024[alu20], alu21);
  var val35 = select(0.0f, data5_1024[(alu12+1)], alu13);
  var val36 = select(0.0f, data5_1024[(alu16+2)], alu17);
  var val37 = select(0.0f, data5_1024[(alu16+3)], alu17);
  var val38 = select(0.0f, data2_24064[(alu18+1)], alu19);
  var val39 = select(0.0f, data5_1024[(alu20+1)], alu21);
  var val40 = select(0.0f, data2_24064[(alu18+2)], alu19);
  var val41 = select(0.0f, data5_1024[(alu20+2)], alu21);
  var val42 = select(0.0f, data2_24064[(alu18+3)], alu19);
  var val43 = select(0.0f, data5_1024[(alu20+3)], alu21);
  var alu22 = (alu7+(gidx2*1736704)+(lidx0*217088));
  data0_3473408[alu22] = (val8+val9+val10);
  data0_3473408[(alu22+1)] = (val11+val12+val13);
  data0_3473408[(alu22+2)] = (val14+val15+val16);
  data0_3473408[(alu22+3)] = (val17+val18+val27);
  data0_3473408[(alu22+54272)] = (val19+val9+val20);
  data0_3473408[(alu22+54273)] = (val31+val12+val35);
  data0_3473408[(alu22+54274)] = (val21+val15+val22);
  data0_3473408[(alu22+54275)] = (val23+val18+val24);
  data0_3473408[(alu22+108544)] = (val25+val9+val26);
  data0_3473408[(alu22+108545)] = (val28+val12+val29);
  data0_3473408[(alu22+108546)] = (val32+val15+val36);
  data0_3473408[(alu22+108547)] = (val33+val18+val37);
  data0_3473408[(alu22+162816)] = (val30+val9+val34);
  data0_3473408[(alu22+162817)] = (val38+val12+val39);
  data0_3473408[(alu22+162818)] = (val40+val15+val41);
  data0_3473408[(alu22+162819)] = (val42+val18+val43);
}`;

const r_212_53_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_212:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_212:array<i32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 212; Ridx0++) {
    var val0 = data1_212[Ridx0];
    acc0[0] = (acc0[0]+(f32(val0)));
  }
  acc1[0] = 0.0f;
  for (var Ridx1 = 0; Ridx1 < 53; Ridx1++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Ridx1)<<2u));
    var val1 = data2_212[cast0];
    var val2 = data2_212[(cast0+1)];
    var val3 = data2_212[(cast0+2)];
    var val4 = data2_212[(cast0+3)];
    acc1[0] = (acc1[0]+(f32(val1))+(f32(val2))+(f32(val3))+(f32(val4)));
  }
  data0_1[0] = ((acc0[0]+acc1[0])*1e-12f);
}`;

const r_424_16_8_16_3_4_64_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_10420224:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_3473408[alu12];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val1 = data2_196608[(alu13+257)];
    var val2 = data2_196608[(alu13+258)];
    var val3 = data2_196608[(alu13+259)];
    var val4 = data2_196608[alu13];
    var val5 = data1_3473408[(alu12+1)];
    var val6 = data2_196608[(alu13+1)];
    var val7 = data1_3473408[(alu12+2)];
    var val8 = data2_196608[(alu13+2)];
    var val9 = data1_3473408[(alu12+3)];
    var val10 = data2_196608[(alu13+3)];
    var val11 = data1_3473408[(alu12+256)];
    var val12 = data1_3473408[(alu12+257)];
    var val13 = data1_3473408[(alu12+258)];
    var val14 = data1_3473408[(alu12+259)];
    var val15 = data1_3473408[(alu12+512)];
    var val16 = data1_3473408[(alu12+513)];
    var val17 = data1_3473408[(alu12+514)];
    var val18 = data1_3473408[(alu12+515)];
    var val19 = data1_3473408[(alu12+768)];
    var val20 = data1_3473408[(alu12+769)];
    var val21 = data1_3473408[(alu12+770)];
    var val22 = data1_3473408[(alu12+771)];
    var val23 = data2_196608[(alu13+256)];
    var val24 = data2_196608[(alu13+512)];
    var val25 = data2_196608[(alu13+513)];
    var val26 = data2_196608[(alu13+514)];
    var val27 = data2_196608[(alu13+515)];
    acc0[0] = (acc0[0]+(val0*val4)+(val5*val6)+(val7*val8)+(val9*val10));
    acc0[1] = (acc0[1]+(val11*val4)+(val12*val6)+(val13*val8)+(val14*val10));
    acc0[2] = (acc0[2]+(val15*val4)+(val16*val6)+(val17*val8)+(val18*val10));
    acc0[3] = (acc0[3]+(val19*val4)+(val20*val6)+(val21*val8)+(val22*val10));
    acc0[4] = (acc0[4]+(val0*val23)+(val5*val1)+(val7*val2)+(val9*val3));
    acc0[5] = (acc0[5]+(val11*val23)+(val12*val1)+(val13*val2)+(val14*val3));
    acc0[6] = (acc0[6]+(val15*val23)+(val16*val1)+(val17*val2)+(val18*val3));
    acc0[7] = (acc0[7]+(val19*val23)+(val20*val1)+(val21*val2)+(val22*val3));
    acc0[8] = (acc0[8]+(val0*val24)+(val5*val25)+(val7*val26)+(val9*val27));
    acc0[9] = (acc0[9]+(val11*val24)+(val12*val25)+(val13*val26)+(val14*val27));
    acc0[10] = (acc0[10]+(val15*val24)+(val16*val25)+(val17*val26)+(val18*val27));
    acc0[11] = (acc0[11]+(val19*val24)+(val20*val25)+(val21*val26)+(val22*val27));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+2)];
  var val29 = data3_768[alu27];
  var val30 = data3_768[(alu27+1)];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_10420224[(alu28+768)] = (acc0[1]+val29);
  data0_10420224[(alu28+769)] = (acc0[5]+val30);
  data0_10420224[(alu28+770)] = (acc0[9]+val28);
  data0_10420224[(alu28+1536)] = (acc0[2]+val29);
  data0_10420224[(alu28+1537)] = (acc0[6]+val30);
  data0_10420224[(alu28+1538)] = (acc0[10]+val28);
  data0_10420224[(alu28+2304)] = (acc0[3]+val29);
  data0_10420224[(alu28+2305)] = (acc0[7]+val30);
  data0_10420224[(alu28+2306)] = (acc0[11]+val28);
  data0_10420224[(alu28+1)] = (acc0[4]+val30);
  data0_10420224[(alu28+2)] = (acc0[8]+val28);
  data0_10420224[alu28] = (acc0[0]+val29);
}`;

const r_4_53_53_16_8_4_4_32n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_23011328:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_10420224:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 53 */
  var gidx2 = i32(gindex.z); /* 4 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu0 = ((gidx2*2605056)+(lidx0*162816));
  var alu1 = ((gidx0*3072)+cast0+alu0);
  var val0 = data1_10420224[(alu1+256)];
  var val1 = data1_10420224[(alu1+257)];
  var val2 = data1_10420224[(alu1+258)];
  var val3 = data1_10420224[(alu1+259)];
  var val4 = data1_10420224[(alu1+260)];
  var val5 = data1_10420224[(alu1+261)];
  var val6 = data1_10420224[(alu1+262)];
  var val7 = data1_10420224[(alu1+263)];
  var val8 = data1_10420224[(alu1+264)];
  var val9 = data1_10420224[(alu1+265)];
  var val10 = data1_10420224[(alu1+266)];
  var val11 = data1_10420224[(alu1+267)];
  var val12 = data1_10420224[(alu1+268)];
  var val13 = data1_10420224[(alu1+269)];
  var val14 = data1_10420224[(alu1+270)];
  var val15 = data1_10420224[(alu1+271)];
  var val16 = data1_10420224[(alu1+272)];
  var val17 = data1_10420224[(alu1+273)];
  var val18 = data1_10420224[(alu1+274)];
  var val19 = data1_10420224[(alu1+275)];
  var val20 = data1_10420224[(alu1+276)];
  var val21 = data1_10420224[(alu1+277)];
  var val22 = data1_10420224[(alu1+278)];
  var val23 = data1_10420224[(alu1+279)];
  var val24 = data1_10420224[(alu1+280)];
  var val25 = data1_10420224[(alu1+281)];
  var val26 = data1_10420224[(alu1+282)];
  var val27 = data1_10420224[(alu1+283)];
  var val28 = data1_10420224[(alu1+284)];
  var val29 = data1_10420224[(alu1+285)];
  var val30 = data1_10420224[(alu1+286)];
  var val31 = data1_10420224[(alu1+287)];
  var val32 = data1_10420224[(alu1+1024)];
  var val33 = data1_10420224[(alu1+1025)];
  var val34 = data1_10420224[(alu1+1026)];
  var val35 = data1_10420224[(alu1+1027)];
  var val36 = data1_10420224[(alu1+1028)];
  var val37 = data1_10420224[(alu1+1029)];
  var val38 = data1_10420224[(alu1+1030)];
  var val39 = data1_10420224[(alu1+1031)];
  var val40 = data1_10420224[(alu1+1032)];
  var val41 = data1_10420224[(alu1+1033)];
  var val42 = data1_10420224[(alu1+1034)];
  var val43 = data1_10420224[(alu1+1035)];
  var val44 = data1_10420224[(alu1+1036)];
  var val45 = data1_10420224[(alu1+1037)];
  var val46 = data1_10420224[(alu1+1038)];
  var val47 = data1_10420224[(alu1+1039)];
  var val48 = data1_10420224[(alu1+1040)];
  var val49 = data1_10420224[(alu1+1041)];
  var val50 = data1_10420224[(alu1+1042)];
  var val51 = data1_10420224[(alu1+1043)];
  var val52 = data1_10420224[(alu1+1044)];
  var val53 = data1_10420224[(alu1+1045)];
  var val54 = data1_10420224[(alu1+1046)];
  var val55 = data1_10420224[(alu1+1047)];
  var val56 = data1_10420224[(alu1+1048)];
  var val57 = data1_10420224[(alu1+1049)];
  var val58 = data1_10420224[(alu1+1050)];
  var val59 = data1_10420224[(alu1+1051)];
  var val60 = data1_10420224[(alu1+1052)];
  var val61 = data1_10420224[(alu1+1053)];
  var val62 = data1_10420224[(alu1+1054)];
  var val63 = data1_10420224[(alu1+1055)];
  var val64 = data1_10420224[(alu1+1792)];
  var val65 = data1_10420224[(alu1+1793)];
  var val66 = data1_10420224[(alu1+1794)];
  var val67 = data1_10420224[(alu1+1795)];
  var val68 = data1_10420224[(alu1+1796)];
  var val69 = data1_10420224[(alu1+1797)];
  var val70 = data1_10420224[(alu1+1798)];
  var val71 = data1_10420224[(alu1+1799)];
  var val72 = data1_10420224[(alu1+1800)];
  var val73 = data1_10420224[(alu1+1801)];
  var val74 = data1_10420224[(alu1+1802)];
  var val75 = data1_10420224[(alu1+1803)];
  var val76 = data1_10420224[(alu1+1804)];
  var val77 = data1_10420224[(alu1+1805)];
  var val78 = data1_10420224[(alu1+1806)];
  var val79 = data1_10420224[(alu1+1807)];
  var val80 = data1_10420224[(alu1+1808)];
  var val81 = data1_10420224[(alu1+1809)];
  var val82 = data1_10420224[(alu1+1810)];
  var val83 = data1_10420224[(alu1+1811)];
  var val84 = data1_10420224[(alu1+1812)];
  var val85 = data1_10420224[(alu1+1813)];
  var val86 = data1_10420224[(alu1+1814)];
  var val87 = data1_10420224[(alu1+1815)];
  var val88 = data1_10420224[(alu1+1816)];
  var val89 = data1_10420224[(alu1+1817)];
  var val90 = data1_10420224[(alu1+1818)];
  var val91 = data1_10420224[(alu1+1819)];
  var val92 = data1_10420224[(alu1+1820)];
  var val93 = data1_10420224[(alu1+1821)];
  var val94 = data1_10420224[(alu1+1822)];
  var val95 = data1_10420224[(alu1+1823)];
  var val96 = data1_10420224[(alu1+2560)];
  var val97 = data1_10420224[(alu1+2561)];
  var val98 = data1_10420224[(alu1+2562)];
  var val99 = data1_10420224[(alu1+2563)];
  var val100 = data1_10420224[(alu1+2564)];
  var val101 = data1_10420224[(alu1+2565)];
  var val102 = data1_10420224[(alu1+2566)];
  var val103 = data1_10420224[(alu1+2567)];
  var val104 = data1_10420224[(alu1+2568)];
  var val105 = data1_10420224[(alu1+2569)];
  var val106 = data1_10420224[(alu1+2570)];
  var val107 = data1_10420224[(alu1+2571)];
  var val108 = data1_10420224[(alu1+2572)];
  var val109 = data1_10420224[(alu1+2573)];
  var val110 = data1_10420224[(alu1+2574)];
  var val111 = data1_10420224[(alu1+2575)];
  var val112 = data1_10420224[(alu1+2576)];
  var val113 = data1_10420224[(alu1+2577)];
  var val114 = data1_10420224[(alu1+2578)];
  var val115 = data1_10420224[(alu1+2579)];
  var val116 = data1_10420224[(alu1+2580)];
  var val117 = data1_10420224[(alu1+2581)];
  var val118 = data1_10420224[(alu1+2582)];
  var val119 = data1_10420224[(alu1+2583)];
  var val120 = data1_10420224[(alu1+2584)];
  var val121 = data1_10420224[(alu1+2585)];
  var val122 = data1_10420224[(alu1+2586)];
  var val123 = data1_10420224[(alu1+2587)];
  var val124 = data1_10420224[(alu1+2588)];
  var val125 = data1_10420224[(alu1+2589)];
  var val126 = data1_10420224[(alu1+2590)];
  var val127 = data1_10420224[(alu1+2591)];
  var gidx1 = i32(gindex.y); /* 53 */
  var alu2 = ((gidx1*3072)+cast0+alu0);
  var val128 = data1_10420224[(alu2+1)];
  var val129 = data1_10420224[(alu2+2)];
  var val130 = data1_10420224[(alu2+3)];
  var val131 = data1_10420224[(alu2+4)];
  var val132 = data1_10420224[(alu2+5)];
  var val133 = data1_10420224[(alu2+6)];
  var val134 = data1_10420224[(alu2+7)];
  var val135 = data1_10420224[(alu2+8)];
  var val136 = data1_10420224[(alu2+9)];
  var val137 = data1_10420224[(alu2+10)];
  var val138 = data1_10420224[(alu2+11)];
  var val139 = data1_10420224[(alu2+12)];
  var val140 = data1_10420224[(alu2+13)];
  var val141 = data1_10420224[(alu2+14)];
  var val142 = data1_10420224[(alu2+15)];
  var val143 = data1_10420224[(alu2+16)];
  var val144 = data1_10420224[(alu2+17)];
  var val145 = data1_10420224[(alu2+18)];
  var val146 = data1_10420224[(alu2+19)];
  var val147 = data1_10420224[(alu2+20)];
  var val148 = data1_10420224[(alu2+21)];
  var val149 = data1_10420224[(alu2+22)];
  var val150 = data1_10420224[(alu2+23)];
  var val151 = data1_10420224[(alu2+24)];
  var val152 = data1_10420224[(alu2+25)];
  var val153 = data1_10420224[(alu2+26)];
  var val154 = data1_10420224[(alu2+27)];
  var val155 = data1_10420224[(alu2+28)];
  var val156 = data1_10420224[(alu2+29)];
  var val157 = data1_10420224[(alu2+30)];
  var val158 = data1_10420224[(alu2+31)];
  var val159 = data1_10420224[(alu2+768)];
  var val160 = data1_10420224[(alu2+769)];
  var val161 = data1_10420224[(alu2+770)];
  var val162 = data1_10420224[(alu2+771)];
  var val163 = data1_10420224[(alu2+772)];
  var val164 = data1_10420224[(alu2+773)];
  var val165 = data1_10420224[(alu2+774)];
  var val166 = data1_10420224[(alu2+775)];
  var val167 = data1_10420224[(alu2+776)];
  var val168 = data1_10420224[(alu2+777)];
  var val169 = data1_10420224[(alu2+778)];
  var val170 = data1_10420224[(alu2+779)];
  var val171 = data1_10420224[(alu2+780)];
  var val172 = data1_10420224[(alu2+781)];
  var val173 = data1_10420224[(alu2+782)];
  var val174 = data1_10420224[(alu2+783)];
  var val175 = data1_10420224[(alu2+784)];
  var val176 = data1_10420224[(alu2+785)];
  var val177 = data1_10420224[(alu2+786)];
  var val178 = data1_10420224[(alu2+787)];
  var val179 = data1_10420224[(alu2+788)];
  var val180 = data1_10420224[(alu2+789)];
  var val181 = data1_10420224[(alu2+790)];
  var val182 = data1_10420224[(alu2+791)];
  var val183 = data1_10420224[(alu2+792)];
  var val184 = data1_10420224[(alu2+793)];
  var val185 = data1_10420224[(alu2+794)];
  var val186 = data1_10420224[(alu2+795)];
  var val187 = data1_10420224[(alu2+796)];
  var val188 = data1_10420224[(alu2+797)];
  var val189 = data1_10420224[(alu2+798)];
  var val190 = data1_10420224[(alu2+799)];
  var val191 = data1_10420224[(alu2+1536)];
  var val192 = data1_10420224[(alu2+1537)];
  var val193 = data1_10420224[(alu2+1538)];
  var val194 = data1_10420224[(alu2+1539)];
  var val195 = data1_10420224[(alu2+1540)];
  var val196 = data1_10420224[(alu2+1541)];
  var val197 = data1_10420224[(alu2+1542)];
  var val198 = data1_10420224[(alu2+1543)];
  var val199 = data1_10420224[(alu2+1544)];
  var val200 = data1_10420224[(alu2+1545)];
  var val201 = data1_10420224[(alu2+1546)];
  var val202 = data1_10420224[(alu2+1547)];
  var val203 = data1_10420224[(alu2+1548)];
  var val204 = data1_10420224[(alu2+1549)];
  var val205 = data1_10420224[(alu2+1550)];
  var val206 = data1_10420224[(alu2+1551)];
  var val207 = data1_10420224[(alu2+1552)];
  var val208 = data1_10420224[(alu2+1553)];
  var val209 = data1_10420224[(alu2+1554)];
  var val210 = data1_10420224[(alu2+1555)];
  var val211 = data1_10420224[(alu2+1556)];
  var val212 = data1_10420224[(alu2+1557)];
  var val213 = data1_10420224[(alu2+1558)];
  var val214 = data1_10420224[(alu2+1559)];
  var val215 = data1_10420224[(alu2+1560)];
  var val216 = data1_10420224[(alu2+1561)];
  var val217 = data1_10420224[(alu2+1562)];
  var val218 = data1_10420224[(alu2+1563)];
  var val219 = data1_10420224[(alu2+1564)];
  var val220 = data1_10420224[(alu2+1565)];
  var val221 = data1_10420224[(alu2+1566)];
  var val222 = data1_10420224[(alu2+1567)];
  var val223 = data1_10420224[(alu2+2304)];
  var val224 = data1_10420224[(alu2+2305)];
  var val225 = data1_10420224[(alu2+2306)];
  var val226 = data1_10420224[(alu2+2307)];
  var val227 = data1_10420224[(alu2+2308)];
  var val228 = data1_10420224[(alu2+2309)];
  var val229 = data1_10420224[(alu2+2310)];
  var val230 = data1_10420224[(alu2+2311)];
  var val231 = data1_10420224[(alu2+2312)];
  var val232 = data1_10420224[(alu2+2313)];
  var val233 = data1_10420224[(alu2+2314)];
  var val234 = data1_10420224[(alu2+2315)];
  var val235 = data1_10420224[(alu2+2316)];
  var val236 = data1_10420224[(alu2+2317)];
  var val237 = data1_10420224[(alu2+2318)];
  var val238 = data1_10420224[(alu2+2319)];
  var val239 = data1_10420224[(alu2+2320)];
  var val240 = data1_10420224[(alu2+2321)];
  var val241 = data1_10420224[(alu2+2322)];
  var val242 = data1_10420224[(alu2+2323)];
  var val243 = data1_10420224[(alu2+2324)];
  var val244 = data1_10420224[(alu2+2325)];
  var val245 = data1_10420224[(alu2+2326)];
  var val246 = data1_10420224[(alu2+2327)];
  var val247 = data1_10420224[(alu2+2328)];
  var val248 = data1_10420224[(alu2+2329)];
  var val249 = data1_10420224[(alu2+2330)];
  var val250 = data1_10420224[(alu2+2331)];
  var val251 = data1_10420224[(alu2+2332)];
  var val252 = data1_10420224[(alu2+2333)];
  var val253 = data1_10420224[(alu2+2334)];
  var val254 = data1_10420224[(alu2+2335)];
  var val255 = data1_10420224[alu2];
  var alu3 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+(gidx1*848)+(lidx1*44944)+(gidx2*5752832)+(lidx0*359552));
  data0_23011328[alu3] = (((val255*val0)+(val128*val1)+(val129*val2)+(val130*val3)+(val131*val4)+(val132*val5)+(val133*val6)+(val134*val7)+(val135*val8)+(val136*val9)+(val137*val10)+(val138*val11)+(val139*val12)+(val140*val13)+(val141*val14)+(val142*val15)+(val143*val16)+(val144*val17)+(val145*val18)+(val146*val19)+(val147*val20)+(val148*val21)+(val149*val22)+(val150*val23)+(val151*val24)+(val152*val25)+(val153*val26)+(val154*val27)+(val155*val28)+(val156*val29)+(val157*val30)+(val158*val31))*0.17677669529663687f);
  data0_23011328[(alu3+1)] = (((val255*val32)+(val128*val33)+(val129*val34)+(val130*val35)+(val131*val36)+(val132*val37)+(val133*val38)+(val134*val39)+(val135*val40)+(val136*val41)+(val137*val42)+(val138*val43)+(val139*val44)+(val140*val45)+(val141*val46)+(val142*val47)+(val143*val48)+(val144*val49)+(val145*val50)+(val146*val51)+(val147*val52)+(val148*val53)+(val149*val54)+(val150*val55)+(val151*val56)+(val152*val57)+(val153*val58)+(val154*val59)+(val155*val60)+(val156*val61)+(val157*val62)+(val158*val63))*0.17677669529663687f);
  data0_23011328[(alu3+2)] = (((val255*val64)+(val128*val65)+(val129*val66)+(val130*val67)+(val131*val68)+(val132*val69)+(val133*val70)+(val134*val71)+(val135*val72)+(val136*val73)+(val137*val74)+(val138*val75)+(val139*val76)+(val140*val77)+(val141*val78)+(val142*val79)+(val143*val80)+(val144*val81)+(val145*val82)+(val146*val83)+(val147*val84)+(val148*val85)+(val149*val86)+(val150*val87)+(val151*val88)+(val152*val89)+(val153*val90)+(val154*val91)+(val155*val92)+(val156*val93)+(val157*val94)+(val158*val95))*0.17677669529663687f);
  data0_23011328[(alu3+3)] = (((val255*val96)+(val128*val97)+(val129*val98)+(val130*val99)+(val131*val100)+(val132*val101)+(val133*val102)+(val134*val103)+(val135*val104)+(val136*val105)+(val137*val106)+(val138*val107)+(val139*val108)+(val140*val109)+(val141*val110)+(val142*val111)+(val143*val112)+(val144*val113)+(val145*val114)+(val146*val115)+(val147*val116)+(val148*val117)+(val149*val118)+(val150*val119)+(val151*val120)+(val152*val121)+(val153*val122)+(val154*val123)+(val155*val124)+(val156*val125)+(val157*val126)+(val158*val127))*0.17677669529663687f);
  data0_23011328[(alu3+212)] = (((val159*val0)+(val160*val1)+(val161*val2)+(val162*val3)+(val163*val4)+(val164*val5)+(val165*val6)+(val166*val7)+(val167*val8)+(val168*val9)+(val169*val10)+(val170*val11)+(val171*val12)+(val172*val13)+(val173*val14)+(val174*val15)+(val175*val16)+(val176*val17)+(val177*val18)+(val178*val19)+(val179*val20)+(val180*val21)+(val181*val22)+(val182*val23)+(val183*val24)+(val184*val25)+(val185*val26)+(val186*val27)+(val187*val28)+(val188*val29)+(val189*val30)+(val190*val31))*0.17677669529663687f);
  data0_23011328[(alu3+213)] = (((val159*val32)+(val160*val33)+(val161*val34)+(val162*val35)+(val163*val36)+(val164*val37)+(val165*val38)+(val166*val39)+(val167*val40)+(val168*val41)+(val169*val42)+(val170*val43)+(val171*val44)+(val172*val45)+(val173*val46)+(val174*val47)+(val175*val48)+(val176*val49)+(val177*val50)+(val178*val51)+(val179*val52)+(val180*val53)+(val181*val54)+(val182*val55)+(val183*val56)+(val184*val57)+(val185*val58)+(val186*val59)+(val187*val60)+(val188*val61)+(val189*val62)+(val190*val63))*0.17677669529663687f);
  data0_23011328[(alu3+214)] = (((val159*val64)+(val160*val65)+(val161*val66)+(val162*val67)+(val163*val68)+(val164*val69)+(val165*val70)+(val166*val71)+(val167*val72)+(val168*val73)+(val169*val74)+(val170*val75)+(val171*val76)+(val172*val77)+(val173*val78)+(val174*val79)+(val175*val80)+(val176*val81)+(val177*val82)+(val178*val83)+(val179*val84)+(val180*val85)+(val181*val86)+(val182*val87)+(val183*val88)+(val184*val89)+(val185*val90)+(val186*val91)+(val187*val92)+(val188*val93)+(val189*val94)+(val190*val95))*0.17677669529663687f);
  data0_23011328[(alu3+215)] = (((val159*val96)+(val160*val97)+(val161*val98)+(val162*val99)+(val163*val100)+(val164*val101)+(val165*val102)+(val166*val103)+(val167*val104)+(val168*val105)+(val169*val106)+(val170*val107)+(val171*val108)+(val172*val109)+(val173*val110)+(val174*val111)+(val175*val112)+(val176*val113)+(val177*val114)+(val178*val115)+(val179*val116)+(val180*val117)+(val181*val118)+(val182*val119)+(val183*val120)+(val184*val121)+(val185*val122)+(val186*val123)+(val187*val124)+(val188*val125)+(val189*val126)+(val190*val127))*0.17677669529663687f);
  data0_23011328[(alu3+424)] = (((val191*val0)+(val192*val1)+(val193*val2)+(val194*val3)+(val195*val4)+(val196*val5)+(val197*val6)+(val198*val7)+(val199*val8)+(val200*val9)+(val201*val10)+(val202*val11)+(val203*val12)+(val204*val13)+(val205*val14)+(val206*val15)+(val207*val16)+(val208*val17)+(val209*val18)+(val210*val19)+(val211*val20)+(val212*val21)+(val213*val22)+(val214*val23)+(val215*val24)+(val216*val25)+(val217*val26)+(val218*val27)+(val219*val28)+(val220*val29)+(val221*val30)+(val222*val31))*0.17677669529663687f);
  data0_23011328[(alu3+425)] = (((val191*val32)+(val192*val33)+(val193*val34)+(val194*val35)+(val195*val36)+(val196*val37)+(val197*val38)+(val198*val39)+(val199*val40)+(val200*val41)+(val201*val42)+(val202*val43)+(val203*val44)+(val204*val45)+(val205*val46)+(val206*val47)+(val207*val48)+(val208*val49)+(val209*val50)+(val210*val51)+(val211*val52)+(val212*val53)+(val213*val54)+(val214*val55)+(val215*val56)+(val216*val57)+(val217*val58)+(val218*val59)+(val219*val60)+(val220*val61)+(val221*val62)+(val222*val63))*0.17677669529663687f);
  data0_23011328[(alu3+426)] = (((val191*val64)+(val192*val65)+(val193*val66)+(val194*val67)+(val195*val68)+(val196*val69)+(val197*val70)+(val198*val71)+(val199*val72)+(val200*val73)+(val201*val74)+(val202*val75)+(val203*val76)+(val204*val77)+(val205*val78)+(val206*val79)+(val207*val80)+(val208*val81)+(val209*val82)+(val210*val83)+(val211*val84)+(val212*val85)+(val213*val86)+(val214*val87)+(val215*val88)+(val216*val89)+(val217*val90)+(val218*val91)+(val219*val92)+(val220*val93)+(val221*val94)+(val222*val95))*0.17677669529663687f);
  data0_23011328[(alu3+427)] = (((val191*val96)+(val192*val97)+(val193*val98)+(val194*val99)+(val195*val100)+(val196*val101)+(val197*val102)+(val198*val103)+(val199*val104)+(val200*val105)+(val201*val106)+(val202*val107)+(val203*val108)+(val204*val109)+(val205*val110)+(val206*val111)+(val207*val112)+(val208*val113)+(val209*val114)+(val210*val115)+(val211*val116)+(val212*val117)+(val213*val118)+(val214*val119)+(val215*val120)+(val216*val121)+(val217*val122)+(val218*val123)+(val219*val124)+(val220*val125)+(val221*val126)+(val222*val127))*0.17677669529663687f);
  data0_23011328[(alu3+636)] = (((val223*val0)+(val224*val1)+(val225*val2)+(val226*val3)+(val227*val4)+(val228*val5)+(val229*val6)+(val230*val7)+(val231*val8)+(val232*val9)+(val233*val10)+(val234*val11)+(val235*val12)+(val236*val13)+(val237*val14)+(val238*val15)+(val239*val16)+(val240*val17)+(val241*val18)+(val242*val19)+(val243*val20)+(val244*val21)+(val245*val22)+(val246*val23)+(val247*val24)+(val248*val25)+(val249*val26)+(val250*val27)+(val251*val28)+(val252*val29)+(val253*val30)+(val254*val31))*0.17677669529663687f);
  data0_23011328[(alu3+637)] = (((val223*val32)+(val224*val33)+(val225*val34)+(val226*val35)+(val227*val36)+(val228*val37)+(val229*val38)+(val230*val39)+(val231*val40)+(val232*val41)+(val233*val42)+(val234*val43)+(val235*val44)+(val236*val45)+(val237*val46)+(val238*val47)+(val239*val48)+(val240*val49)+(val241*val50)+(val242*val51)+(val243*val52)+(val244*val53)+(val245*val54)+(val246*val55)+(val247*val56)+(val248*val57)+(val249*val58)+(val250*val59)+(val251*val60)+(val252*val61)+(val253*val62)+(val254*val63))*0.17677669529663687f);
  data0_23011328[(alu3+638)] = (((val223*val64)+(val224*val65)+(val225*val66)+(val226*val67)+(val227*val68)+(val228*val69)+(val229*val70)+(val230*val71)+(val231*val72)+(val232*val73)+(val233*val74)+(val234*val75)+(val235*val76)+(val236*val77)+(val237*val78)+(val238*val79)+(val239*val80)+(val240*val81)+(val241*val82)+(val242*val83)+(val243*val84)+(val244*val85)+(val245*val86)+(val246*val87)+(val247*val88)+(val248*val89)+(val249*val90)+(val250*val91)+(val251*val92)+(val252*val93)+(val253*val94)+(val254*val95))*0.17677669529663687f);
  data0_23011328[(alu3+639)] = (((val223*val96)+(val224*val97)+(val225*val98)+(val226*val99)+(val227*val100)+(val228*val101)+(val229*val102)+(val230*val103)+(val231*val104)+(val232*val105)+(val233*val106)+(val234*val107)+(val235*val108)+(val236*val109)+(val237*val110)+(val238*val111)+(val239*val112)+(val240*val113)+(val241*val114)+(val242*val115)+(val243*val116)+(val244*val117)+(val245*val118)+(val246*val119)+(val247*val120)+(val248*val121)+(val249*val122)+(val250*val123)+(val251*val124)+(val252*val125)+(val253*val126)+(val254*val127))*0.17677669529663687f);
}`;

const r_848_32_4_53_4n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_108544:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_23011328:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 848 */
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = (f32(-INFINITY));
  acc0[1] = (f32(-INFINITY));
  acc0[2] = (f32(-INFINITY));
  acc0[3] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu4 = ((gidx0*27136)+(lidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_23011328[(alu4+1)];
    var val1 = data1_23011328[(alu4+2)];
    var val2 = data1_23011328[(alu4+3)];
    var val3 = data1_23011328[(alu4+215)];
    var val4 = data1_23011328[alu4];
    var val5 = data1_23011328[(alu4+212)];
    var val6 = data1_23011328[(alu4+213)];
    var val7 = data1_23011328[(alu4+214)];
    var val8 = data1_23011328[(alu4+424)];
    var val9 = data1_23011328[(alu4+425)];
    var val10 = data1_23011328[(alu4+426)];
    var val11 = data1_23011328[(alu4+427)];
    var val12 = data1_23011328[(alu4+636)];
    var val13 = data1_23011328[(alu4+637)];
    var val14 = data1_23011328[(alu4+638)];
    var val15 = data1_23011328[(alu4+639)];
    var alu5 = select(acc0[0],val4,(acc0[0]<val4));
    var alu6 = select(acc0[1],val5,(acc0[1]<val5));
    var alu7 = select(acc0[2],val8,(acc0[2]<val8));
    var alu8 = select(acc0[3],val12,(acc0[3]<val12));
    var alu9 = select(alu5,val0,(alu5<val0));
    var alu10 = select(alu6,val6,(alu6<val6));
    var alu11 = select(alu7,val9,(alu7<val9));
    var alu12 = select(alu8,val13,(alu8<val13));
    var alu13 = select(alu9,val1,(alu9<val1));
    var alu14 = select(alu10,val7,(alu10<val7));
    var alu15 = select(alu11,val10,(alu11<val10));
    var alu16 = select(alu12,val14,(alu12<val14));
    var alu17 = select(alu13,val2,(alu13<val2));
    var alu18 = select(alu14,val3,(alu14<val3));
    var alu19 = select(alu15,val11,(alu15<val11));
    var alu20 = select(alu16,val15,(alu16<val15));
    acc0[0] = alu17;
    acc0[1] = alu18;
    acc0[2] = alu19;
    acc0[3] = alu20;
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<7u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  data0_108544[alu26] = acc0[0];
  data0_108544[(alu26+1)] = acc0[1];
  data0_108544[(alu26+2)] = acc0[2];
  data0_108544[(alu26+3)] = acc0[3];
}`;

const r_848_32_4_53_4n3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_108544:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_23011328:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_108544:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 848 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<7u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  var val0 = data2_108544[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_108544[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_108544[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_108544[alu3];
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu8 = ((gidx0*27136)+(lidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val4 = data1_23011328[(alu8+1)];
    var val5 = data1_23011328[(alu8+2)];
    var val6 = data1_23011328[(alu8+3)];
    var val7 = data1_23011328[(alu8+212)];
    var val8 = data1_23011328[(alu8+213)];
    var val9 = data1_23011328[(alu8+214)];
    var val10 = data1_23011328[(alu8+215)];
    var val11 = data1_23011328[alu8];
    var val12 = data1_23011328[(alu8+424)];
    var val13 = data1_23011328[(alu8+425)];
    var val14 = data1_23011328[(alu8+426)];
    var val15 = data1_23011328[(alu8+427)];
    var val16 = data1_23011328[(alu8+636)];
    var val17 = data1_23011328[(alu8+637)];
    var val18 = data1_23011328[(alu8+638)];
    var val19 = data1_23011328[(alu8+639)];
    acc0[0] = (acc0[0]+exp2(((val11-val0)*1.4426950408889634f))+exp2(((val4-val0)*1.4426950408889634f))+exp2(((val5-val0)*1.4426950408889634f))+exp2(((val6-val0)*1.4426950408889634f)));
    acc0[1] = (acc0[1]+exp2(((val7-val1)*1.4426950408889634f))+exp2(((val8-val1)*1.4426950408889634f))+exp2(((val9-val1)*1.4426950408889634f))+exp2(((val10-val1)*1.4426950408889634f)));
    acc0[2] = (acc0[2]+exp2(((val12-val2)*1.4426950408889634f))+exp2(((val13-val2)*1.4426950408889634f))+exp2(((val14-val2)*1.4426950408889634f))+exp2(((val15-val2)*1.4426950408889634f)));
    acc0[3] = (acc0[3]+exp2(((val16-val3)*1.4426950408889634f))+exp2(((val17-val3)*1.4426950408889634f))+exp2(((val18-val3)*1.4426950408889634f))+exp2(((val19-val3)*1.4426950408889634f)));
  }
  data0_108544[alu0] = acc0[0];
  data0_108544[alu1] = acc0[1];
  data0_108544[alu2] = acc0[2];
  data0_108544[alu3] = acc0[3];
}`;

const r_32_53_2_8_8_4_4_53_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3473408:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_23011328:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_108544:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_108544:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_10420224:array<f32>;
@compute @workgroup_size(2,8,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 53 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (bitcast<i32>((cast0<<2u))+(lidx1*212)+(gidx1*3392)+(lidx0*1696));
  var val0 = data2_108544[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_108544[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_108544[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_108544[alu3];
  var lidx2 = i32(lindex.z); /* 8 */
  var cast1 = bitcast<i32>((bitcast<u32>(lidx2)<<2u));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu20 = ((gidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u))+(lidx1*44944)+(gidx1*719104)+(lidx0*359552));
    var val4 = data1_23011328[alu20];
    var alu21 = (bitcast<i32>((bitcast<u32>(lidx1)<<5u))+cast1+(Ridx0*3072)+(gidx1*325632)+(lidx0*162816));
    var val5 = data4_10420224[(alu21+512)];
    var val6 = data1_23011328[(alu20+1)];
    var val7 = data4_10420224[(alu21+513)];
    var val8 = data4_10420224[(alu21+1280)];
    var val9 = data1_23011328[(alu20+2)];
    var val10 = data4_10420224[(alu21+1281)];
    var val11 = data4_10420224[(alu21+2048)];
    var val12 = data1_23011328[(alu20+3)];
    var val13 = data4_10420224[(alu21+2049)];
    var val14 = data4_10420224[(alu21+2816)];
    var val15 = data1_23011328[(alu20+212)];
    var val16 = data1_23011328[(alu20+213)];
    var val17 = data1_23011328[(alu20+214)];
    var val18 = data1_23011328[(alu20+215)];
    var val19 = data1_23011328[(alu20+424)];
    var val20 = data1_23011328[(alu20+425)];
    var val21 = data1_23011328[(alu20+426)];
    var val22 = data1_23011328[(alu20+427)];
    var val23 = data1_23011328[(alu20+636)];
    var val24 = data1_23011328[(alu20+637)];
    var val25 = data1_23011328[(alu20+638)];
    var val26 = data1_23011328[(alu20+639)];
    var val27 = data4_10420224[(alu21+2817)];
    var val28 = data4_10420224[(alu21+514)];
    var val29 = data4_10420224[(alu21+2818)];
    var val30 = data4_10420224[(alu21+515)];
    var val31 = data4_10420224[(alu21+1282)];
    var val32 = data4_10420224[(alu21+1283)];
    var val33 = data4_10420224[(alu21+2050)];
    var val34 = data4_10420224[(alu21+2051)];
    var val35 = data4_10420224[(alu21+2819)];
    var alu22 = exp2(((val6-val0)*1.4426950408889634f));
    var alu23 = exp2(((val9-val0)*1.4426950408889634f));
    var alu24 = exp2(((val12-val0)*1.4426950408889634f));
    var alu25 = exp2(((val15-val1)*1.4426950408889634f));
    var alu26 = exp2(((val16-val1)*1.4426950408889634f));
    var alu27 = exp2(((val17-val1)*1.4426950408889634f));
    var alu28 = exp2(((val18-val1)*1.4426950408889634f));
    var alu29 = exp2(((val19-val2)*1.4426950408889634f));
    var alu30 = exp2(((val20-val2)*1.4426950408889634f));
    var alu31 = exp2(((val21-val2)*1.4426950408889634f));
    var alu32 = exp2(((val22-val2)*1.4426950408889634f));
    var alu33 = exp2(((val23-val3)*1.4426950408889634f));
    var alu34 = exp2(((val24-val3)*1.4426950408889634f));
    var alu35 = exp2(((val25-val3)*1.4426950408889634f));
    var alu36 = exp2(((val26-val3)*1.4426950408889634f));
    var alu37 = exp2(((val4-val0)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu37*val5)+(alu22*val8)+(alu23*val11)+(alu24*val14));
    acc0[1] = (acc0[1]+(alu25*val5)+(alu26*val8)+(alu27*val11)+(alu28*val14));
    acc0[2] = (acc0[2]+(alu29*val5)+(alu30*val8)+(alu31*val11)+(alu32*val14));
    acc0[3] = (acc0[3]+(alu33*val5)+(alu34*val8)+(alu35*val11)+(alu36*val14));
    acc0[4] = (acc0[4]+(alu37*val7)+(alu22*val10)+(alu23*val13)+(alu24*val27));
    acc0[5] = (acc0[5]+(alu25*val7)+(alu26*val10)+(alu27*val13)+(alu28*val27));
    acc0[6] = (acc0[6]+(alu29*val7)+(alu30*val10)+(alu31*val13)+(alu32*val27));
    acc0[7] = (acc0[7]+(alu33*val7)+(alu34*val10)+(alu35*val13)+(alu36*val27));
    acc0[8] = (acc0[8]+(alu37*val28)+(alu22*val31)+(alu23*val33)+(alu24*val29));
    acc0[9] = (acc0[9]+(alu25*val28)+(alu26*val31)+(alu27*val33)+(alu28*val29));
    acc0[10] = (acc0[10]+(alu29*val28)+(alu30*val31)+(alu31*val33)+(alu32*val29));
    acc0[11] = (acc0[11]+(alu33*val28)+(alu34*val31)+(alu35*val33)+(alu36*val29));
    acc0[12] = (acc0[12]+(alu37*val30)+(alu22*val32)+(alu23*val34)+(alu24*val35));
    acc0[13] = (acc0[13]+(alu25*val30)+(alu26*val32)+(alu27*val34)+(alu28*val35));
    acc0[14] = (acc0[14]+(alu29*val30)+(alu30*val32)+(alu31*val34)+(alu32*val35));
    acc0[15] = (acc0[15]+(alu33*val30)+(alu34*val32)+(alu35*val34)+(alu36*val35));
  }
  var val36 = data3_108544[alu0];
  var val37 = data3_108544[alu1];
  var val38 = data3_108544[alu2];
  var val39 = data3_108544[alu3];
  var alu55 = (bitcast<i32>((cast0<<7u))+cast1+(lidx1*6784)+(gidx1*108544)+(lidx0*54272));
  var alu56 = (1/val36);
  data0_3473408[alu55] = (acc0[0]*alu56);
  data0_3473408[(alu55+1)] = (acc0[4]*alu56);
  data0_3473408[(alu55+2)] = (acc0[8]*alu56);
  data0_3473408[(alu55+3)] = (acc0[12]*alu56);
  var alu61 = (1/val37);
  data0_3473408[(alu55+32)] = (acc0[1]*alu61);
  data0_3473408[(alu55+33)] = (acc0[5]*alu61);
  data0_3473408[(alu55+34)] = (acc0[9]*alu61);
  data0_3473408[(alu55+35)] = (acc0[13]*alu61);
  var alu66 = (1/val38);
  data0_3473408[(alu55+64)] = (acc0[2]*alu66);
  data0_3473408[(alu55+65)] = (acc0[6]*alu66);
  data0_3473408[(alu55+66)] = (acc0[10]*alu66);
  data0_3473408[(alu55+67)] = (acc0[14]*alu66);
  var alu71 = (1/val39);
  data0_3473408[(alu55+96)] = (acc0[3]*alu71);
  data0_3473408[(alu55+97)] = (acc0[7]*alu71);
  data0_3473408[(alu55+98)] = (acc0[11]*alu71);
  data0_3473408[(alu55+99)] = (acc0[15]*alu71);
}`;

const r_8_53_4_8_16_4_4_8_32n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3473408:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3473408:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 53 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx1);
  var alu0 = ((gidx2*434176)+(lidx0*54272));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0_0 = 0; Ridx0_0 < 8; Ridx0_0++) {
    var alu17 = (bitcast<i32>((cast1<<7u))+(Ridx0_0*6784)+alu0);
    var val0 = data2_3473408[alu17];
    var alu18 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Ridx0_0)<<5u)));
    var val1 = data3_65536[alu18];
    var val2 = data2_3473408[(alu17+1)];
    var val3 = data3_65536[(alu18+1)];
    var val4 = data2_3473408[(alu17+2)];
    var val5 = data3_65536[(alu18+2)];
    var val6 = data2_3473408[(alu17+3)];
    var val7 = data3_65536[(alu18+3)];
    var val8 = data2_3473408[(alu17+4)];
    var val9 = data3_65536[(alu18+4)];
    var val10 = data2_3473408[(alu17+5)];
    var val11 = data3_65536[(alu18+5)];
    var val12 = data2_3473408[(alu17+6)];
    var val13 = data3_65536[(alu18+6)];
    var val14 = data2_3473408[(alu17+7)];
    var val15 = data3_65536[(alu18+7)];
    var val16 = data2_3473408[(alu17+8)];
    var val17 = data3_65536[(alu18+8)];
    var val18 = data2_3473408[(alu17+9)];
    var val19 = data3_65536[(alu18+9)];
    var val20 = data2_3473408[(alu17+10)];
    var val21 = data3_65536[(alu18+10)];
    var val22 = data2_3473408[(alu17+11)];
    var val23 = data3_65536[(alu18+11)];
    var val24 = data2_3473408[(alu17+12)];
    var val25 = data3_65536[(alu18+12)];
    var val26 = data2_3473408[(alu17+13)];
    var val27 = data3_65536[(alu18+13)];
    var val28 = data2_3473408[(alu17+14)];
    var val29 = data3_65536[(alu18+14)];
    var val30 = data2_3473408[(alu17+15)];
    var val31 = data3_65536[(alu18+15)];
    var val32 = data2_3473408[(alu17+16)];
    var val33 = data3_65536[(alu18+16)];
    var val34 = data2_3473408[(alu17+17)];
    var val35 = data3_65536[(alu18+17)];
    var val36 = data2_3473408[(alu17+18)];
    var val37 = data3_65536[(alu18+18)];
    var val38 = data2_3473408[(alu17+19)];
    var val39 = data3_65536[(alu18+19)];
    var val40 = data2_3473408[(alu17+20)];
    var val41 = data3_65536[(alu18+20)];
    var val42 = data2_3473408[(alu17+21)];
    var val43 = data3_65536[(alu18+21)];
    var val44 = data2_3473408[(alu17+22)];
    var val45 = data3_65536[(alu18+22)];
    var val46 = data2_3473408[(alu17+23)];
    var val47 = data3_65536[(alu18+23)];
    var val48 = data2_3473408[(alu17+24)];
    var val49 = data3_65536[(alu18+24)];
    var val50 = data2_3473408[(alu17+25)];
    var val51 = data3_65536[(alu18+25)];
    var val52 = data2_3473408[(alu17+26)];
    var val53 = data3_65536[(alu18+26)];
    var val54 = data2_3473408[(alu17+27)];
    var val55 = data3_65536[(alu18+27)];
    var val56 = data2_3473408[(alu17+28)];
    var val57 = data3_65536[(alu18+28)];
    var val58 = data2_3473408[(alu17+29)];
    var val59 = data3_65536[(alu18+29)];
    var val60 = data2_3473408[(alu17+30)];
    var val61 = data3_65536[(alu18+30)];
    var val62 = data2_3473408[(alu17+31)];
    var val63 = data3_65536[(alu18+31)];
    var val64 = data2_3473408[(alu17+32)];
    var val65 = data2_3473408[(alu17+33)];
    var val66 = data2_3473408[(alu17+34)];
    var val67 = data2_3473408[(alu17+35)];
    var val68 = data2_3473408[(alu17+36)];
    var val69 = data2_3473408[(alu17+37)];
    var val70 = data2_3473408[(alu17+38)];
    var val71 = data2_3473408[(alu17+39)];
    var val72 = data2_3473408[(alu17+40)];
    var val73 = data2_3473408[(alu17+41)];
    var val74 = data2_3473408[(alu17+42)];
    var val75 = data2_3473408[(alu17+43)];
    var val76 = data2_3473408[(alu17+44)];
    var val77 = data2_3473408[(alu17+45)];
    var val78 = data2_3473408[(alu17+46)];
    var val79 = data2_3473408[(alu17+47)];
    var val80 = data2_3473408[(alu17+48)];
    var val81 = data2_3473408[(alu17+49)];
    var val82 = data2_3473408[(alu17+50)];
    var val83 = data2_3473408[(alu17+51)];
    var val84 = data2_3473408[(alu17+52)];
    var val85 = data2_3473408[(alu17+53)];
    var val86 = data2_3473408[(alu17+54)];
    var val87 = data2_3473408[(alu17+55)];
    var val88 = data2_3473408[(alu17+56)];
    var val89 = data2_3473408[(alu17+57)];
    var val90 = data2_3473408[(alu17+58)];
    var val91 = data2_3473408[(alu17+59)];
    var val92 = data2_3473408[(alu17+60)];
    var val93 = data2_3473408[(alu17+61)];
    var val94 = data2_3473408[(alu17+62)];
    var val95 = data2_3473408[(alu17+63)];
    var val96 = data2_3473408[(alu17+64)];
    var val97 = data2_3473408[(alu17+65)];
    var val98 = data2_3473408[(alu17+66)];
    var val99 = data2_3473408[(alu17+67)];
    var val100 = data2_3473408[(alu17+68)];
    var val101 = data2_3473408[(alu17+69)];
    var val102 = data2_3473408[(alu17+70)];
    var val103 = data2_3473408[(alu17+71)];
    var val104 = data2_3473408[(alu17+72)];
    var val105 = data2_3473408[(alu17+73)];
    var val106 = data2_3473408[(alu17+74)];
    var val107 = data2_3473408[(alu17+75)];
    var val108 = data2_3473408[(alu17+76)];
    var val109 = data2_3473408[(alu17+77)];
    var val110 = data2_3473408[(alu17+78)];
    var val111 = data2_3473408[(alu17+79)];
    var val112 = data2_3473408[(alu17+80)];
    var val113 = data2_3473408[(alu17+81)];
    var val114 = data2_3473408[(alu17+82)];
    var val115 = data2_3473408[(alu17+83)];
    var val116 = data2_3473408[(alu17+84)];
    var val117 = data2_3473408[(alu17+85)];
    var val118 = data2_3473408[(alu17+86)];
    var val119 = data2_3473408[(alu17+87)];
    var val120 = data2_3473408[(alu17+88)];
    var val121 = data2_3473408[(alu17+89)];
    var val122 = data2_3473408[(alu17+90)];
    var val123 = data2_3473408[(alu17+91)];
    var val124 = data2_3473408[(alu17+92)];
    var val125 = data2_3473408[(alu17+93)];
    var val126 = data2_3473408[(alu17+94)];
    var val127 = data2_3473408[(alu17+95)];
    var val128 = data2_3473408[(alu17+96)];
    var val129 = data2_3473408[(alu17+97)];
    var val130 = data2_3473408[(alu17+98)];
    var val131 = data2_3473408[(alu17+99)];
    var val132 = data2_3473408[(alu17+100)];
    var val133 = data2_3473408[(alu17+101)];
    var val134 = data2_3473408[(alu17+102)];
    var val135 = data2_3473408[(alu17+103)];
    var val136 = data2_3473408[(alu17+104)];
    var val137 = data2_3473408[(alu17+105)];
    var val138 = data2_3473408[(alu17+106)];
    var val139 = data2_3473408[(alu17+107)];
    var val140 = data2_3473408[(alu17+108)];
    var val141 = data2_3473408[(alu17+109)];
    var val142 = data2_3473408[(alu17+110)];
    var val143 = data2_3473408[(alu17+111)];
    var val144 = data2_3473408[(alu17+112)];
    var val145 = data2_3473408[(alu17+113)];
    var val146 = data2_3473408[(alu17+114)];
    var val147 = data2_3473408[(alu17+115)];
    var val148 = data2_3473408[(alu17+116)];
    var val149 = data2_3473408[(alu17+117)];
    var val150 = data2_3473408[(alu17+118)];
    var val151 = data2_3473408[(alu17+119)];
    var val152 = data2_3473408[(alu17+120)];
    var val153 = data2_3473408[(alu17+121)];
    var val154 = data2_3473408[(alu17+122)];
    var val155 = data2_3473408[(alu17+123)];
    var val156 = data2_3473408[(alu17+124)];
    var val157 = data2_3473408[(alu17+125)];
    var val158 = data2_3473408[(alu17+126)];
    var val159 = data2_3473408[(alu17+127)];
    var val160 = data3_65536[(alu18+256)];
    var val161 = data3_65536[(alu18+257)];
    var val162 = data3_65536[(alu18+258)];
    var val163 = data3_65536[(alu18+259)];
    var val164 = data3_65536[(alu18+260)];
    var val165 = data3_65536[(alu18+261)];
    var val166 = data3_65536[(alu18+262)];
    var val167 = data3_65536[(alu18+263)];
    var val168 = data3_65536[(alu18+264)];
    var val169 = data3_65536[(alu18+265)];
    var val170 = data3_65536[(alu18+266)];
    var val171 = data3_65536[(alu18+267)];
    var val172 = data3_65536[(alu18+268)];
    var val173 = data3_65536[(alu18+269)];
    var val174 = data3_65536[(alu18+270)];
    var val175 = data3_65536[(alu18+271)];
    var val176 = data3_65536[(alu18+272)];
    var val177 = data3_65536[(alu18+273)];
    var val178 = data3_65536[(alu18+274)];
    var val179 = data3_65536[(alu18+275)];
    var val180 = data3_65536[(alu18+276)];
    var val181 = data3_65536[(alu18+277)];
    var val182 = data3_65536[(alu18+278)];
    var val183 = data3_65536[(alu18+279)];
    var val184 = data3_65536[(alu18+280)];
    var val185 = data3_65536[(alu18+281)];
    var val186 = data3_65536[(alu18+282)];
    var val187 = data3_65536[(alu18+283)];
    var val188 = data3_65536[(alu18+284)];
    var val189 = data3_65536[(alu18+285)];
    var val190 = data3_65536[(alu18+286)];
    var val191 = data3_65536[(alu18+287)];
    var val192 = data3_65536[(alu18+512)];
    var val193 = data3_65536[(alu18+513)];
    var val194 = data3_65536[(alu18+514)];
    var val195 = data3_65536[(alu18+515)];
    var val196 = data3_65536[(alu18+516)];
    var val197 = data3_65536[(alu18+517)];
    var val198 = data3_65536[(alu18+518)];
    var val199 = data3_65536[(alu18+519)];
    var val200 = data3_65536[(alu18+520)];
    var val201 = data3_65536[(alu18+521)];
    var val202 = data3_65536[(alu18+522)];
    var val203 = data3_65536[(alu18+523)];
    var val204 = data3_65536[(alu18+524)];
    var val205 = data3_65536[(alu18+525)];
    var val206 = data3_65536[(alu18+526)];
    var val207 = data3_65536[(alu18+527)];
    var val208 = data3_65536[(alu18+528)];
    var val209 = data3_65536[(alu18+529)];
    var val210 = data3_65536[(alu18+530)];
    var val211 = data3_65536[(alu18+531)];
    var val212 = data3_65536[(alu18+532)];
    var val213 = data3_65536[(alu18+533)];
    var val214 = data3_65536[(alu18+534)];
    var val215 = data3_65536[(alu18+535)];
    var val216 = data3_65536[(alu18+536)];
    var val217 = data3_65536[(alu18+537)];
    var val218 = data3_65536[(alu18+538)];
    var val219 = data3_65536[(alu18+539)];
    var val220 = data3_65536[(alu18+540)];
    var val221 = data3_65536[(alu18+541)];
    var val222 = data3_65536[(alu18+542)];
    var val223 = data3_65536[(alu18+543)];
    var val224 = data3_65536[(alu18+768)];
    var val225 = data3_65536[(alu18+769)];
    var val226 = data3_65536[(alu18+770)];
    var val227 = data3_65536[(alu18+771)];
    var val228 = data3_65536[(alu18+772)];
    var val229 = data3_65536[(alu18+773)];
    var val230 = data3_65536[(alu18+774)];
    var val231 = data3_65536[(alu18+775)];
    var val232 = data3_65536[(alu18+776)];
    var val233 = data3_65536[(alu18+777)];
    var val234 = data3_65536[(alu18+778)];
    var val235 = data3_65536[(alu18+779)];
    var val236 = data3_65536[(alu18+780)];
    var val237 = data3_65536[(alu18+781)];
    var val238 = data3_65536[(alu18+782)];
    var val239 = data3_65536[(alu18+783)];
    var val240 = data3_65536[(alu18+784)];
    var val241 = data3_65536[(alu18+785)];
    var val242 = data3_65536[(alu18+786)];
    var val243 = data3_65536[(alu18+787)];
    var val244 = data3_65536[(alu18+788)];
    var val245 = data3_65536[(alu18+789)];
    var val246 = data3_65536[(alu18+790)];
    var val247 = data3_65536[(alu18+791)];
    var val248 = data3_65536[(alu18+792)];
    var val249 = data3_65536[(alu18+793)];
    var val250 = data3_65536[(alu18+794)];
    var val251 = data3_65536[(alu18+795)];
    var val252 = data3_65536[(alu18+796)];
    var val253 = data3_65536[(alu18+797)];
    var val254 = data3_65536[(alu18+798)];
    var val255 = data3_65536[(alu18+799)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17)+(val18*val19)+(val20*val21)+(val22*val23)+(val24*val25)+(val26*val27)+(val28*val29)+(val30*val31)+(val32*val33)+(val34*val35)+(val36*val37)+(val38*val39)+(val40*val41)+(val42*val43)+(val44*val45)+(val46*val47)+(val48*val49)+(val50*val51)+(val52*val53)+(val54*val55)+(val56*val57)+(val58*val59)+(val60*val61)+(val62*val63));
    acc0[1] = (acc0[1]+(val64*val1)+(val65*val3)+(val66*val5)+(val67*val7)+(val68*val9)+(val69*val11)+(val70*val13)+(val71*val15)+(val72*val17)+(val73*val19)+(val74*val21)+(val75*val23)+(val76*val25)+(val77*val27)+(val78*val29)+(val79*val31)+(val80*val33)+(val81*val35)+(val82*val37)+(val83*val39)+(val84*val41)+(val85*val43)+(val86*val45)+(val87*val47)+(val88*val49)+(val89*val51)+(val90*val53)+(val91*val55)+(val92*val57)+(val93*val59)+(val94*val61)+(val95*val63));
    acc0[2] = (acc0[2]+(val96*val1)+(val97*val3)+(val98*val5)+(val99*val7)+(val100*val9)+(val101*val11)+(val102*val13)+(val103*val15)+(val104*val17)+(val105*val19)+(val106*val21)+(val107*val23)+(val108*val25)+(val109*val27)+(val110*val29)+(val111*val31)+(val112*val33)+(val113*val35)+(val114*val37)+(val115*val39)+(val116*val41)+(val117*val43)+(val118*val45)+(val119*val47)+(val120*val49)+(val121*val51)+(val122*val53)+(val123*val55)+(val124*val57)+(val125*val59)+(val126*val61)+(val127*val63));
    acc0[3] = (acc0[3]+(val128*val1)+(val129*val3)+(val130*val5)+(val131*val7)+(val132*val9)+(val133*val11)+(val134*val13)+(val135*val15)+(val136*val17)+(val137*val19)+(val138*val21)+(val139*val23)+(val140*val25)+(val141*val27)+(val142*val29)+(val143*val31)+(val144*val33)+(val145*val35)+(val146*val37)+(val147*val39)+(val148*val41)+(val149*val43)+(val150*val45)+(val151*val47)+(val152*val49)+(val153*val51)+(val154*val53)+(val155*val55)+(val156*val57)+(val157*val59)+(val158*val61)+(val159*val63));
    acc0[4] = (acc0[4]+(val0*val160)+(val2*val161)+(val4*val162)+(val6*val163)+(val8*val164)+(val10*val165)+(val12*val166)+(val14*val167)+(val16*val168)+(val18*val169)+(val20*val170)+(val22*val171)+(val24*val172)+(val26*val173)+(val28*val174)+(val30*val175)+(val32*val176)+(val34*val177)+(val36*val178)+(val38*val179)+(val40*val180)+(val42*val181)+(val44*val182)+(val46*val183)+(val48*val184)+(val50*val185)+(val52*val186)+(val54*val187)+(val56*val188)+(val58*val189)+(val60*val190)+(val62*val191));
    acc0[5] = (acc0[5]+(val64*val160)+(val65*val161)+(val66*val162)+(val67*val163)+(val68*val164)+(val69*val165)+(val70*val166)+(val71*val167)+(val72*val168)+(val73*val169)+(val74*val170)+(val75*val171)+(val76*val172)+(val77*val173)+(val78*val174)+(val79*val175)+(val80*val176)+(val81*val177)+(val82*val178)+(val83*val179)+(val84*val180)+(val85*val181)+(val86*val182)+(val87*val183)+(val88*val184)+(val89*val185)+(val90*val186)+(val91*val187)+(val92*val188)+(val93*val189)+(val94*val190)+(val95*val191));
    acc0[6] = (acc0[6]+(val96*val160)+(val97*val161)+(val98*val162)+(val99*val163)+(val100*val164)+(val101*val165)+(val102*val166)+(val103*val167)+(val104*val168)+(val105*val169)+(val106*val170)+(val107*val171)+(val108*val172)+(val109*val173)+(val110*val174)+(val111*val175)+(val112*val176)+(val113*val177)+(val114*val178)+(val115*val179)+(val116*val180)+(val117*val181)+(val118*val182)+(val119*val183)+(val120*val184)+(val121*val185)+(val122*val186)+(val123*val187)+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191));
    acc0[7] = (acc0[7]+(val128*val160)+(val129*val161)+(val130*val162)+(val131*val163)+(val132*val164)+(val133*val165)+(val134*val166)+(val135*val167)+(val136*val168)+(val137*val169)+(val138*val170)+(val139*val171)+(val140*val172)+(val141*val173)+(val142*val174)+(val143*val175)+(val144*val176)+(val145*val177)+(val146*val178)+(val147*val179)+(val148*val180)+(val149*val181)+(val150*val182)+(val151*val183)+(val152*val184)+(val153*val185)+(val154*val186)+(val155*val187)+(val156*val188)+(val157*val189)+(val158*val190)+(val159*val191));
    acc0[8] = (acc0[8]+(val0*val192)+(val2*val193)+(val4*val194)+(val6*val195)+(val8*val196)+(val10*val197)+(val12*val198)+(val14*val199)+(val16*val200)+(val18*val201)+(val20*val202)+(val22*val203)+(val24*val204)+(val26*val205)+(val28*val206)+(val30*val207)+(val32*val208)+(val34*val209)+(val36*val210)+(val38*val211)+(val40*val212)+(val42*val213)+(val44*val214)+(val46*val215)+(val48*val216)+(val50*val217)+(val52*val218)+(val54*val219)+(val56*val220)+(val58*val221)+(val60*val222)+(val62*val223));
    acc0[9] = (acc0[9]+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val73*val201)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val84*val212)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216)+(val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val95*val223));
    acc0[10] = (acc0[10]+(val96*val192)+(val97*val193)+(val98*val194)+(val99*val195)+(val100*val196)+(val101*val197)+(val102*val198)+(val103*val199)+(val104*val200)+(val105*val201)+(val106*val202)+(val107*val203)+(val108*val204)+(val109*val205)+(val110*val206)+(val111*val207)+(val112*val208)+(val113*val209)+(val114*val210)+(val115*val211)+(val116*val212)+(val117*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219)+(val124*val220)+(val125*val221)+(val126*val222)+(val127*val223));
    acc0[11] = (acc0[11]+(val128*val192)+(val129*val193)+(val130*val194)+(val131*val195)+(val132*val196)+(val133*val197)+(val134*val198)+(val135*val199)+(val136*val200)+(val137*val201)+(val138*val202)+(val139*val203)+(val140*val204)+(val141*val205)+(val142*val206)+(val143*val207)+(val144*val208)+(val145*val209)+(val146*val210)+(val147*val211)+(val148*val212)+(val149*val213)+(val150*val214)+(val151*val215)+(val152*val216)+(val153*val217)+(val154*val218)+(val155*val219)+(val156*val220)+(val157*val221)+(val158*val222)+(val159*val223));
    acc0[12] = (acc0[12]+(val0*val224)+(val2*val225)+(val4*val226)+(val6*val227)+(val8*val228)+(val10*val229)+(val12*val230)+(val14*val231)+(val16*val232)+(val18*val233)+(val20*val234)+(val22*val235)+(val24*val236)+(val26*val237)+(val28*val238)+(val30*val239)+(val32*val240)+(val34*val241)+(val36*val242)+(val38*val243)+(val40*val244)+(val42*val245)+(val44*val246)+(val46*val247)+(val48*val248)+(val50*val249)+(val52*val250)+(val54*val251)+(val56*val252)+(val58*val253)+(val60*val254)+(val62*val255));
    acc0[13] = (acc0[13]+(val64*val224)+(val65*val225)+(val66*val226)+(val67*val227)+(val68*val228)+(val69*val229)+(val70*val230)+(val71*val231)+(val72*val232)+(val73*val233)+(val74*val234)+(val75*val235)+(val76*val236)+(val77*val237)+(val78*val238)+(val79*val239)+(val80*val240)+(val81*val241)+(val82*val242)+(val83*val243)+(val84*val244)+(val85*val245)+(val86*val246)+(val87*val247)+(val88*val248)+(val89*val249)+(val90*val250)+(val91*val251)+(val92*val252)+(val93*val253)+(val94*val254)+(val95*val255));
    acc0[14] = (acc0[14]+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val106*val234)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val117*val245)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251)+(val124*val252)+(val125*val253)+(val126*val254)+(val127*val255));
    acc0[15] = (acc0[15]+(val128*val224)+(val129*val225)+(val130*val226)+(val131*val227)+(val132*val228)+(val133*val229)+(val134*val230)+(val135*val231)+(val136*val232)+(val137*val233)+(val138*val234)+(val139*val235)+(val140*val236)+(val141*val237)+(val142*val238)+(val143*val239)+(val144*val240)+(val145*val241)+(val146*val242)+(val147*val243)+(val148*val244)+(val149*val245)+(val150*val246)+(val151*val247)+(val152*val248)+(val153*val249)+(val154*val250)+(val155*val251)+(val156*val252)+(val157*val253)+(val158*val254)+(val159*val255));
  }
  var alu36 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u)));
  var alu37 = (alu36+bitcast<i32>((cast1<<10u))+alu0);
  var val256 = data1_3473408[alu37];
  var val257 = data4_256[alu36];
  var alu38 = (alu37+1);
  var val258 = data1_3473408[alu38];
  var val259 = data4_256[(alu36+1)];
  var alu39 = (alu37+2);
  var val260 = data1_3473408[alu39];
  var val261 = data4_256[(alu36+2)];
  var alu40 = (alu37+3);
  var val262 = data1_3473408[alu40];
  var val263 = data4_256[(alu36+3)];
  var alu41 = (alu37+256);
  var val264 = data1_3473408[alu41];
  var alu42 = (alu37+257);
  var val265 = data1_3473408[alu42];
  var alu43 = (alu37+258);
  var val266 = data1_3473408[alu43];
  var alu44 = (alu37+259);
  var val267 = data1_3473408[alu44];
  var alu45 = (alu37+512);
  var val268 = data1_3473408[alu45];
  var alu46 = (alu37+513);
  var val269 = data1_3473408[alu46];
  var alu47 = (alu37+514);
  var val270 = data1_3473408[alu47];
  var alu48 = (alu37+515);
  var val271 = data1_3473408[alu48];
  var alu49 = (alu37+768);
  var val272 = data1_3473408[alu49];
  var alu50 = (alu37+769);
  var val273 = data1_3473408[alu50];
  var alu51 = (alu37+770);
  var val274 = data1_3473408[alu51];
  var alu52 = (alu37+771);
  var val275 = data1_3473408[alu52];
  data0_3473408[alu37] = (val256+acc0[0]+val257);
  data0_3473408[alu38] = (val258+acc0[4]+val259);
  data0_3473408[alu39] = (val260+acc0[8]+val261);
  data0_3473408[alu40] = (val262+acc0[12]+val263);
  data0_3473408[alu41] = (val264+acc0[1]+val257);
  data0_3473408[alu42] = (val265+acc0[5]+val259);
  data0_3473408[alu43] = (val266+acc0[9]+val261);
  data0_3473408[alu44] = (val267+acc0[13]+val263);
  data0_3473408[alu45] = (val268+acc0[2]+val257);
  data0_3473408[alu46] = (val269+acc0[6]+val259);
  data0_3473408[alu47] = (val270+acc0[10]+val261);
  data0_3473408[alu48] = (val271+acc0[14]+val263);
  data0_3473408[alu49] = (val272+acc0[3]+val257);
  data0_3473408[alu50] = (val273+acc0[7]+val259);
  data0_3473408[alu51] = (val274+acc0[11]+val261);
  data0_3473408[alu52] = (val275+acc0[15]+val263);
}`;

const r_106_32_4_64_4n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_13568:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 106 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu4 = (bitcast<i32>((cast0<<15u))+bitcast<i32>((cast1<<10u))+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_3473408[alu4];
    var val1 = data1_3473408[(alu4+1)];
    var val2 = data1_3473408[(alu4+2)];
    var val3 = data1_3473408[(alu4+3)];
    var val4 = data1_3473408[(alu4+256)];
    var val5 = data1_3473408[(alu4+257)];
    var val6 = data1_3473408[(alu4+258)];
    var val7 = data1_3473408[(alu4+259)];
    var val8 = data1_3473408[(alu4+512)];
    var val9 = data1_3473408[(alu4+513)];
    var val10 = data1_3473408[(alu4+514)];
    var val11 = data1_3473408[(alu4+515)];
    var val12 = data1_3473408[(alu4+768)];
    var val13 = data1_3473408[(alu4+769)];
    var val14 = data1_3473408[(alu4+770)];
    var val15 = data1_3473408[(alu4+771)];
    acc0[0] = (acc0[0]+val0+val1+val2+val3);
    acc0[1] = (acc0[1]+val4+val5+val6+val7);
    acc0[2] = (acc0[2]+val8+val9+val10+val11);
    acc0[3] = (acc0[3]+val12+val13+val14+val15);
  }
  var alu10 = (bitcast<i32>((cast0<<7u))+bitcast<i32>((cast1<<2u)));
  data0_13568[alu10] = (acc0[0]*0.00390625f);
  data0_13568[(alu10+1)] = (acc0[1]*0.00390625f);
  data0_13568[(alu10+2)] = (acc0[2]*0.00390625f);
  data0_13568[(alu10+3)] = (acc0[3]*0.00390625f);
}`;

const r_106_32_4_64_4n3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_13568:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_13568:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 106 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<7u))+bitcast<i32>((cast1<<2u)));
  var val0 = data2_13568[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_13568[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_13568[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_13568[alu3];
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu8 = (bitcast<i32>((cast0<<15u))+bitcast<i32>((cast1<<10u))+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val4 = data1_3473408[alu8];
    var val5 = data1_3473408[(alu8+1)];
    var val6 = data1_3473408[(alu8+2)];
    var val7 = data1_3473408[(alu8+3)];
    var val8 = data1_3473408[(alu8+256)];
    var val9 = data1_3473408[(alu8+257)];
    var val10 = data1_3473408[(alu8+258)];
    var val11 = data1_3473408[(alu8+259)];
    var val12 = data1_3473408[(alu8+512)];
    var val13 = data1_3473408[(alu8+513)];
    var val14 = data1_3473408[(alu8+514)];
    var val15 = data1_3473408[(alu8+515)];
    var val16 = data1_3473408[(alu8+768)];
    var val17 = data1_3473408[(alu8+769)];
    var val18 = data1_3473408[(alu8+770)];
    var val19 = data1_3473408[(alu8+771)];
    var alu9 = (val4-val0);
    var alu10 = (val8-val1);
    var alu11 = (val12-val2);
    var alu12 = (val16-val3);
    var alu13 = (val5-val0);
    var alu14 = (val9-val1);
    var alu15 = (val13-val2);
    var alu16 = (val17-val3);
    var alu17 = (val6-val0);
    var alu18 = (val10-val1);
    var alu19 = (val14-val2);
    var alu20 = (val18-val3);
    var alu21 = (val7-val0);
    var alu22 = (val11-val1);
    var alu23 = (val15-val2);
    var alu24 = (val19-val3);
    acc0[0] = (acc0[0]+(alu9*alu9)+(alu13*alu13)+(alu17*alu17)+(alu21*alu21));
    acc0[1] = (acc0[1]+(alu10*alu10)+(alu14*alu14)+(alu18*alu18)+(alu22*alu22));
    acc0[2] = (acc0[2]+(alu11*alu11)+(alu15*alu15)+(alu19*alu19)+(alu23*alu23));
    acc0[3] = (acc0[3]+(alu12*alu12)+(alu16*alu16)+(alu20*alu20)+(alu24*alu24));
  }
  data0_13568[alu0] = (1/sqrt(((acc0[0]*0.00390625f)+1e-05f)));
  data0_13568[alu1] = (1/sqrt(((acc0[1]*0.00390625f)+1e-05f)));
  data0_13568[alu2] = (1/sqrt(((acc0[2]*0.00390625f)+1e-05f)));
  data0_13568[alu3] = (1/sqrt(((acc0[3]*0.00390625f)+1e-05f)));
}`;

const E_424_4_8_16_4_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3473408:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_13568:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_13568:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_3473408[alu1];
  var alu2 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val1 = data2_13568[alu2];
  var val2 = data3_13568[alu2];
  var val3 = data4_256[alu0];
  var val4 = data5_256[alu0];
  var alu3 = (alu1+1);
  var val5 = data1_3473408[alu3];
  var alu4 = (alu0+1);
  var val6 = data4_256[alu4];
  var val7 = data5_256[alu4];
  var alu5 = (alu1+2);
  var val8 = data1_3473408[alu5];
  var alu6 = (alu0+2);
  var val9 = data4_256[alu6];
  var alu7 = (alu0+3);
  var val10 = data4_256[alu7];
  var val11 = data5_256[alu6];
  var alu8 = (alu1+3);
  var val12 = data1_3473408[alu8];
  var val13 = data5_256[alu7];
  var alu9 = (alu1+256);
  var val14 = data1_3473408[alu9];
  var alu10 = (alu2+1);
  var val15 = data2_13568[alu10];
  var val16 = data3_13568[alu10];
  var alu11 = (alu1+257);
  var val17 = data1_3473408[alu11];
  var alu12 = (alu1+258);
  var val18 = data1_3473408[alu12];
  var alu13 = (alu1+259);
  var val19 = data1_3473408[alu13];
  var alu14 = (alu1+512);
  var val20 = data1_3473408[alu14];
  var alu15 = (alu2+2);
  var val21 = data2_13568[alu15];
  var val22 = data3_13568[alu15];
  var alu16 = (alu1+513);
  var val23 = data1_3473408[alu16];
  var alu17 = (alu1+514);
  var val24 = data1_3473408[alu17];
  var alu18 = (alu1+515);
  var val25 = data1_3473408[alu18];
  var alu19 = (alu1+768);
  var val26 = data1_3473408[alu19];
  var alu20 = (alu2+3);
  var val27 = data2_13568[alu20];
  var val28 = data3_13568[alu20];
  var alu21 = (alu1+769);
  var val29 = data1_3473408[alu21];
  var alu22 = (alu1+770);
  var val30 = data1_3473408[alu22];
  var alu23 = (alu1+771);
  var val31 = data1_3473408[alu23];
  data0_3473408[alu1] = (((val0-val1)*val2*val3)+val4);
  data0_3473408[alu3] = (((val5-val1)*val2*val6)+val7);
  data0_3473408[alu5] = (((val8-val1)*val2*val9)+val11);
  data0_3473408[alu8] = (((val12-val1)*val2*val10)+val13);
  data0_3473408[alu9] = (((val14-val15)*val16*val3)+val4);
  data0_3473408[alu11] = (((val17-val15)*val16*val6)+val7);
  data0_3473408[alu12] = (((val18-val15)*val16*val9)+val11);
  data0_3473408[alu13] = (((val19-val15)*val16*val10)+val13);
  data0_3473408[alu14] = (((val20-val21)*val22*val3)+val4);
  data0_3473408[alu16] = (((val23-val21)*val22*val6)+val7);
  data0_3473408[alu17] = (((val24-val21)*val22*val9)+val11);
  data0_3473408[alu18] = (((val25-val21)*val22*val10)+val13);
  data0_3473408[alu19] = (((val26-val27)*val28*val3)+val4);
  data0_3473408[alu21] = (((val29-val27)*val28*val6)+val7);
  data0_3473408[alu22] = (((val30-val27)*val28*val9)+val11);
  data0_3473408[alu23] = (((val31-val27)*val28*val10)+val13);
}`;

const r_424_16_8_16_4_4_64_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_13893632:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_262144:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1024:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var cast4 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu16 = (bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u))+cast4);
    var val0 = data1_3473408[alu16];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+cast4);
    var val1 = data2_262144[alu17];
    var val2 = data1_3473408[(alu16+1)];
    var val3 = data2_262144[(alu17+1)];
    var val4 = data1_3473408[(alu16+2)];
    var val5 = data2_262144[(alu17+2)];
    var val6 = data1_3473408[(alu16+3)];
    var val7 = data2_262144[(alu17+3)];
    var val8 = data1_3473408[(alu16+256)];
    var val9 = data1_3473408[(alu16+257)];
    var val10 = data1_3473408[(alu16+258)];
    var val11 = data1_3473408[(alu16+259)];
    var val12 = data1_3473408[(alu16+512)];
    var val13 = data1_3473408[(alu16+513)];
    var val14 = data1_3473408[(alu16+514)];
    var val15 = data1_3473408[(alu16+515)];
    var val16 = data1_3473408[(alu16+768)];
    var val17 = data1_3473408[(alu16+769)];
    var val18 = data1_3473408[(alu16+770)];
    var val19 = data1_3473408[(alu16+771)];
    var val20 = data2_262144[(alu17+256)];
    var val21 = data2_262144[(alu17+257)];
    var val22 = data2_262144[(alu17+258)];
    var val23 = data2_262144[(alu17+259)];
    var val24 = data2_262144[(alu17+512)];
    var val25 = data2_262144[(alu17+513)];
    var val26 = data2_262144[(alu17+514)];
    var val27 = data2_262144[(alu17+515)];
    var val28 = data2_262144[(alu17+768)];
    var val29 = data2_262144[(alu17+769)];
    var val30 = data2_262144[(alu17+770)];
    var val31 = data2_262144[(alu17+771)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val16*val1)+(val17*val3)+(val18*val5)+(val19*val7));
    acc0[4] = (acc0[4]+(val0*val20)+(val2*val21)+(val4*val22)+(val6*val23));
    acc0[5] = (acc0[5]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[6] = (acc0[6]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[7] = (acc0[7]+(val16*val20)+(val17*val21)+(val18*val22)+(val19*val23));
    acc0[8] = (acc0[8]+(val0*val24)+(val2*val25)+(val4*val26)+(val6*val27));
    acc0[9] = (acc0[9]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[10] = (acc0[10]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc0[11] = (acc0[11]+(val16*val24)+(val17*val25)+(val18*val26)+(val19*val27));
    acc0[12] = (acc0[12]+(val0*val28)+(val2*val29)+(val4*val30)+(val6*val31));
    acc0[13] = (acc0[13]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc0[14] = (acc0[14]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    acc0[15] = (acc0[15]+(val16*val28)+(val17*val29)+(val18*val30)+(val19*val31));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var val32 = data3_1024[alu35];
  var val33 = data3_1024[(alu35+1)];
  var val34 = data3_1024[(alu35+2)];
  var val35 = data3_1024[(alu35+3)];
  var alu36 = (alu35+bitcast<i32>((cast1<<15u))+bitcast<i32>((cast2<<12u)));
  var alu37 = (acc0[0]+val32);
  var alu38 = (acc0[4]+val33);
  var alu39 = (acc0[8]+val34);
  var alu40 = (acc0[12]+val35);
  data0_13893632[alu36] = ((1/(1.0f+exp2(((alu37+(0.044715f*alu37*alu37*alu37))*-2.302208198144325f))))*alu37);
  data0_13893632[(alu36+1)] = ((1/(1.0f+exp2(((alu38+(0.044715f*alu38*alu38*alu38))*-2.302208198144325f))))*alu38);
  data0_13893632[(alu36+2)] = ((1/(1.0f+exp2(((alu39+(0.044715f*alu39*alu39*alu39))*-2.302208198144325f))))*alu39);
  data0_13893632[(alu36+3)] = ((1/(1.0f+exp2(((alu40+(0.044715f*alu40*alu40*alu40))*-2.302208198144325f))))*alu40);
  var alu45 = (acc0[1]+val32);
  var alu46 = (acc0[5]+val33);
  var alu47 = (acc0[9]+val34);
  var alu48 = (acc0[13]+val35);
  data0_13893632[(alu36+1024)] = ((1/(1.0f+exp2(((alu45+(0.044715f*alu45*alu45*alu45))*-2.302208198144325f))))*alu45);
  data0_13893632[(alu36+1025)] = ((1/(1.0f+exp2(((alu46+(0.044715f*alu46*alu46*alu46))*-2.302208198144325f))))*alu46);
  data0_13893632[(alu36+1026)] = ((1/(1.0f+exp2(((alu47+(0.044715f*alu47*alu47*alu47))*-2.302208198144325f))))*alu47);
  data0_13893632[(alu36+1027)] = ((1/(1.0f+exp2(((alu48+(0.044715f*alu48*alu48*alu48))*-2.302208198144325f))))*alu48);
  var alu53 = (acc0[2]+val32);
  var alu54 = (acc0[6]+val33);
  var alu55 = (acc0[10]+val34);
  var alu56 = (acc0[14]+val35);
  data0_13893632[(alu36+2048)] = ((1/(1.0f+exp2(((alu53+(0.044715f*alu53*alu53*alu53))*-2.302208198144325f))))*alu53);
  data0_13893632[(alu36+2049)] = ((1/(1.0f+exp2(((alu54+(0.044715f*alu54*alu54*alu54))*-2.302208198144325f))))*alu54);
  data0_13893632[(alu36+2050)] = ((1/(1.0f+exp2(((alu55+(0.044715f*alu55*alu55*alu55))*-2.302208198144325f))))*alu55);
  data0_13893632[(alu36+2051)] = ((1/(1.0f+exp2(((alu56+(0.044715f*alu56*alu56*alu56))*-2.302208198144325f))))*alu56);
  var alu61 = (acc0[3]+val32);
  var alu62 = (acc0[7]+val33);
  var alu63 = (acc0[11]+val34);
  var alu64 = (acc0[15]+val35);
  data0_13893632[(alu36+3072)] = ((1/(1.0f+exp2(((alu61+(0.044715f*alu61*alu61*alu61))*-2.302208198144325f))))*alu61);
  data0_13893632[(alu36+3073)] = ((1/(1.0f+exp2(((alu62+(0.044715f*alu62*alu62*alu62))*-2.302208198144325f))))*alu62);
  data0_13893632[(alu36+3074)] = ((1/(1.0f+exp2(((alu63+(0.044715f*alu63*alu63*alu63))*-2.302208198144325f))))*alu63);
  data0_13893632[(alu36+3075)] = ((1/(1.0f+exp2(((alu64+(0.044715f*alu64*alu64*alu64))*-2.302208198144325f))))*alu64);
}`;

const r_424_4_8_16_4_4_256_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3473408:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_13893632:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_262144:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var cast4 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu16 = (bitcast<i32>((cast1<<15u))+bitcast<i32>((cast2<<12u))+cast4);
    var val0 = data2_13893632[alu16];
    var alu17 = (bitcast<i32>((cast0<<16u))+bitcast<i32>((cast3<<12u))+cast4);
    var val1 = data3_262144[alu17];
    var val2 = data2_13893632[(alu16+1)];
    var val3 = data3_262144[(alu17+1)];
    var val4 = data2_13893632[(alu16+2)];
    var val5 = data3_262144[(alu17+2)];
    var val6 = data2_13893632[(alu16+3)];
    var val7 = data3_262144[(alu17+3)];
    var val8 = data2_13893632[(alu16+1024)];
    var val9 = data2_13893632[(alu16+1025)];
    var val10 = data2_13893632[(alu16+1026)];
    var val11 = data2_13893632[(alu16+1027)];
    var val12 = data2_13893632[(alu16+2048)];
    var val13 = data2_13893632[(alu16+2049)];
    var val14 = data2_13893632[(alu16+2050)];
    var val15 = data2_13893632[(alu16+2051)];
    var val16 = data2_13893632[(alu16+3072)];
    var val17 = data2_13893632[(alu16+3073)];
    var val18 = data2_13893632[(alu16+3074)];
    var val19 = data2_13893632[(alu16+3075)];
    var val20 = data3_262144[(alu17+1024)];
    var val21 = data3_262144[(alu17+1025)];
    var val22 = data3_262144[(alu17+1026)];
    var val23 = data3_262144[(alu17+1027)];
    var val24 = data3_262144[(alu17+2048)];
    var val25 = data3_262144[(alu17+2049)];
    var val26 = data3_262144[(alu17+2050)];
    var val27 = data3_262144[(alu17+2051)];
    var val28 = data3_262144[(alu17+3072)];
    var val29 = data3_262144[(alu17+3073)];
    var val30 = data3_262144[(alu17+3074)];
    var val31 = data3_262144[(alu17+3075)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val16*val1)+(val17*val3)+(val18*val5)+(val19*val7));
    acc0[4] = (acc0[4]+(val0*val20)+(val2*val21)+(val4*val22)+(val6*val23));
    acc0[5] = (acc0[5]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[6] = (acc0[6]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[7] = (acc0[7]+(val16*val20)+(val17*val21)+(val18*val22)+(val19*val23));
    acc0[8] = (acc0[8]+(val0*val24)+(val2*val25)+(val4*val26)+(val6*val27));
    acc0[9] = (acc0[9]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[10] = (acc0[10]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc0[11] = (acc0[11]+(val16*val24)+(val17*val25)+(val18*val26)+(val19*val27));
    acc0[12] = (acc0[12]+(val0*val28)+(val2*val29)+(val4*val30)+(val6*val31));
    acc0[13] = (acc0[13]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc0[14] = (acc0[14]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    acc0[15] = (acc0[15]+(val16*val28)+(val17*val29)+(val18*val30)+(val19*val31));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val32 = data1_3473408[alu36];
  var val33 = data4_256[alu35];
  var alu37 = (alu36+1);
  var val34 = data1_3473408[alu37];
  var val35 = data4_256[(alu35+1)];
  var alu38 = (alu36+2);
  var val36 = data1_3473408[alu38];
  var val37 = data4_256[(alu35+2)];
  var alu39 = (alu36+3);
  var val38 = data1_3473408[alu39];
  var val39 = data4_256[(alu35+3)];
  var alu40 = (alu36+256);
  var val40 = data1_3473408[alu40];
  var alu41 = (alu36+257);
  var val41 = data1_3473408[alu41];
  var alu42 = (alu36+258);
  var val42 = data1_3473408[alu42];
  var alu43 = (alu36+259);
  var val43 = data1_3473408[alu43];
  var alu44 = (alu36+512);
  var val44 = data1_3473408[alu44];
  var alu45 = (alu36+513);
  var val45 = data1_3473408[alu45];
  var alu46 = (alu36+514);
  var val46 = data1_3473408[alu46];
  var alu47 = (alu36+515);
  var val47 = data1_3473408[alu47];
  var alu48 = (alu36+768);
  var val48 = data1_3473408[alu48];
  var alu49 = (alu36+769);
  var val49 = data1_3473408[alu49];
  var alu50 = (alu36+770);
  var val50 = data1_3473408[alu50];
  var alu51 = (alu36+771);
  var val51 = data1_3473408[alu51];
  data0_3473408[alu36] = (val32+acc0[0]+val33);
  data0_3473408[alu37] = (val34+acc0[4]+val35);
  data0_3473408[alu38] = (val36+acc0[8]+val37);
  data0_3473408[alu39] = (val38+acc0[12]+val39);
  data0_3473408[alu40] = (val40+acc0[1]+val33);
  data0_3473408[alu41] = (val41+acc0[5]+val35);
  data0_3473408[alu42] = (val42+acc0[9]+val37);
  data0_3473408[alu43] = (val43+acc0[13]+val39);
  data0_3473408[alu44] = (val44+acc0[2]+val33);
  data0_3473408[alu45] = (val45+acc0[6]+val35);
  data0_3473408[alu46] = (val46+acc0[10]+val37);
  data0_3473408[alu47] = (val47+acc0[14]+val39);
  data0_3473408[alu48] = (val48+acc0[3]+val33);
  data0_3473408[alu49] = (val49+acc0[7]+val35);
  data0_3473408[alu50] = (val50+acc0[11]+val37);
  data0_3473408[alu51] = (val51+acc0[15]+val39);
}`;

const r_64_16_256_8n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_64:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3473408:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_32768:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_128:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_1:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var acc2: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(lidx0);
  acc1[0] = 0.0f;
  for (var Ridx1 = 0; Ridx1 < 8; Ridx1++) {
    acc0[0] = 0.0f;
    for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
      var val0 = data1_3473408[((gidx0*54272)+Ridx0)];
      var val1 = data2_32768[(bitcast<i32>((cast0<<11u))+bitcast<i32>((bitcast<u32>(Ridx1)<<8u))+Ridx0)];
      acc0[0] = (acc0[0]+(val0*val1));
    }
    var alu4 = (bitcast<i32>((cast0<<3u))+Ridx1);
    var val2 = data3_128[alu4];
    var val3 = data4_128[alu4];
    var alu5 = (acc0[0]+val2);
    acc1[0] = (acc1[0]+((1/(1.0f+exp2(((alu5+(0.044715f*alu5*alu5*alu5))*-2.302208198144325f))))*alu5*val3));
  }
  temp0[lidx0] = acc1[0];
  workgroupBarrier();
  acc2[0] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 16; Ridx103++) {
    var val4 = temp0[Ridx103];
    acc2[0] = (acc2[0]+val4);
  }
  var val5 = data5_1[0];
  var val6 = data6_1[0];
  var alu13 = ((bool(lidx0))!=true);
  if (alu13) {
    data0_64[gidx0] = (((acc2[0]+val5)*2.0f)+val6);
  }
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 13893632);;
    const input2 = createEmptyBuf(device, 54272);;
    const buf_1 = createWeightBuf(device, 96256, getTensorBuffer(safetensor, metadata['enc.tok_emb.weight']));
    const buf_2 = createWeightBuf(device, 217088, getTensorBuffer(safetensor, metadata['enc.pos_emb.weight']));
    const input3 = createEmptyBuf(device, 54272);;
    const buf_3 = createWeightBuf(device, 4096, getTensorBuffer(safetensor, metadata['enc.la_emb.weight']));
    const buf_4 = createEmptyBuf(device, 4);;
    const input0 = createEmptyBuf(device, 848);;
    const input1 = createEmptyBuf(device, 848);;
    const buf_5 = createEmptyBuf(device, 41680896);;
    const buf_6 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.in_proj_weight']));
    const buf_7 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.in_proj_bias']));
    const buf_8 = createEmptyBuf(device, 92045312);;
    const buf_9 = createEmptyBuf(device, 434176);;
    const buf_10 = createEmptyBuf(device, 434176);;
    const buf_11 = createEmptyBuf(device, 13893632);;
    const buf_12 = createEmptyBuf(device, 13893632);;
    const buf_13 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.out_proj.weight']));
    const buf_14 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.out_proj.bias']));
    const buf_15 = createEmptyBuf(device, 54272);;
    const buf_16 = createEmptyBuf(device, 54272);;
    const buf_17 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.0.norm1.weight']));
    const buf_18 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.0.norm1.bias']));
    const buf_19 = createEmptyBuf(device, 55574528);;
    const buf_20 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.0.linear1.weight']));
    const buf_21 = createWeightBuf(device, 4096, getTensorBuffer(safetensor, metadata['enc.layers.0.linear1.bias']));
    const buf_22 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.0.linear2.weight']));
    const buf_23 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.0.linear2.bias']));
    const buf_24 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.0.norm2.weight']));
    const buf_25 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.0.norm2.bias']));
    const buf_26 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.in_proj_weight']));
    const buf_27 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.in_proj_bias']));
    const buf_28 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.out_proj.weight']));
    const buf_29 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.out_proj.bias']));
    const buf_30 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.1.norm1.weight']));
    const buf_31 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.1.norm1.bias']));
    const buf_32 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.1.linear1.weight']));
    const buf_33 = createWeightBuf(device, 4096, getTensorBuffer(safetensor, metadata['enc.layers.1.linear1.bias']));
    const buf_34 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.1.linear2.weight']));
    const buf_35 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.1.linear2.bias']));
    const buf_36 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.1.norm2.weight']));
    const buf_37 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.1.norm2.bias']));
    const buf_38 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.in_proj_weight']));
    const buf_39 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.in_proj_bias']));
    const buf_40 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.out_proj.weight']));
    const buf_41 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.out_proj.bias']));
    const buf_42 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.2.norm1.weight']));
    const buf_43 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.2.norm1.bias']));
    const buf_44 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.2.linear1.weight']));
    const buf_45 = createWeightBuf(device, 4096, getTensorBuffer(safetensor, metadata['enc.layers.2.linear1.bias']));
    const buf_46 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.2.linear2.weight']));
    const buf_47 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.2.linear2.bias']));
    const buf_48 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.2.norm2.weight']));
    const buf_49 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.2.norm2.bias']));
    const buf_50 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.in_proj_weight']));
    const buf_51 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.in_proj_bias']));
    const buf_52 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.out_proj.weight']));
    const buf_53 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.out_proj.bias']));
    const buf_54 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.3.norm1.weight']));
    const buf_55 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.3.norm1.bias']));
    const buf_56 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.3.linear1.weight']));
    const buf_57 = createWeightBuf(device, 4096, getTensorBuffer(safetensor, metadata['enc.layers.3.linear1.bias']));
    const buf_58 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['enc.layers.3.linear2.weight']));
    const buf_59 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.3.linear2.bias']));
    const buf_60 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.3.norm2.weight']));
    const buf_61 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['enc.layers.3.norm2.bias']));
    const output0 = createEmptyBuf(device, 256);;
    const buf_62 = createWeightBuf(device, 131072, getTensorBuffer(safetensor, metadata['rer.proj.weight']));
    const buf_63 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.proj.bias']));
    const buf_64 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.sc.weight']));
    const buf_65 = createWeightBuf(device, 4, getTensorBuffer(safetensor, metadata['rer.sc.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input2.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer1 = device.createBuffer({size:input3.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer2 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer3 = device.createBuffer({size:input1.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [E_2_212_4_8_16_4_4n1, r_212_53_4n1, r_424_16_8_16_3_4_64_4n1, r_4_53_53_16_8_4_4_32n1, r_848_32_4_53_4n2, r_848_32_4_53_4n3, r_32_53_2_8_8_4_4_53_4n1, r_8_53_4_8_16_4_4_8_32n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_424_16_8_16_4_4_64_4n1, r_424_4_8_16_4_4_256_4n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_424_16_8_16_3_4_64_4n1, r_4_53_53_16_8_4_4_32n1, r_848_32_4_53_4n2, r_848_32_4_53_4n3, r_32_53_2_8_8_4_4_53_4n1, r_8_53_4_8_16_4_4_8_32n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_424_16_8_16_4_4_64_4n1, r_424_4_8_16_4_4_256_4n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_424_16_8_16_3_4_64_4n1, r_4_53_53_16_8_4_4_32n1, r_848_32_4_53_4n2, r_848_32_4_53_4n3, r_32_53_2_8_8_4_4_53_4n1, r_8_53_4_8_16_4_4_8_32n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_424_16_8_16_4_4_64_4n1, r_424_4_8_16_4_4_256_4n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_424_16_8_16_3_4_64_4n1, r_4_53_53_16_8_4_4_32n1, r_848_32_4_53_4n2, r_848_32_4_53_4n3, r_32_53_2_8_8_4_4_53_4n1, r_8_53_4_8_16_4_4_8_32n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_424_16_8_16_4_4_64_4n1, r_424_4_8_16_4_4_256_4n1, r_106_32_4_64_4n2, r_106_32_4_64_4n3, E_424_4_8_16_4_4n1, r_64_16_256_8n1];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input2,_input3,_input0,_input1) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer0.getMappedRange()).set(_input2);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input2, 0, gpuWriteBuffer0.size);
    await gpuWriteBuffer1.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer1.getMappedRange()).set(_input3);
        gpuWriteBuffer1.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer1, 0, input3, 0, gpuWriteBuffer1.size);
    await gpuWriteBuffer2.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer2.getMappedRange()).set(_input0);
        gpuWriteBuffer2.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer2, 0, input0, 0, gpuWriteBuffer2.size);
    await gpuWriteBuffer3.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer3.getMappedRange()).set(_input1);
        gpuWriteBuffer3.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer3, 0, input1, 0, gpuWriteBuffer3.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input2, buf_1, buf_2, input3, buf_3], [4, 212, 2]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_4, input0, input1], [1, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_5, buf_0, buf_6, buf_7], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_8, buf_5], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_9, buf_8], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_10, buf_8, buf_9], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_11, buf_8, buf_9, buf_10, buf_5], [53, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_12, buf_0, buf_11, buf_13, buf_14], [4, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_15, buf_12], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_16, buf_12, buf_15], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_11, buf_12, buf_15, buf_16, buf_17, buf_18], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_19, buf_11, buf_20, buf_21], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_12, buf_11, buf_19, buf_22, buf_23], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_16, buf_12], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_15, buf_12, buf_16], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_11, buf_12, buf_16, buf_15, buf_24, buf_25], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_5, buf_11, buf_26, buf_27], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_8, buf_5], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_10, buf_8], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_9, buf_8, buf_10], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_12, buf_8, buf_10, buf_9, buf_5], [53, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_0, buf_11, buf_12, buf_28, buf_29], [4, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_15, buf_0], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_16, buf_0, buf_15], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_12, buf_0, buf_15, buf_16, buf_30, buf_31], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_19, buf_12, buf_32, buf_33], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_0, buf_12, buf_19, buf_34, buf_35], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_16, buf_0], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_15, buf_0, buf_16], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_12, buf_0, buf_16, buf_15, buf_36, buf_37], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_5, buf_12, buf_38, buf_39], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_8, buf_5], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_9, buf_8], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_10, buf_8, buf_9], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_0, buf_8, buf_9, buf_10, buf_5], [53, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_11, buf_12, buf_0, buf_40, buf_41], [4, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_15, buf_11], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_16, buf_11, buf_15], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_0, buf_11, buf_15, buf_16, buf_42, buf_43], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_19, buf_0, buf_44, buf_45], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_11, buf_0, buf_19, buf_46, buf_47], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_16, buf_11], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_15, buf_11, buf_16], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_0, buf_11, buf_16, buf_15, buf_48, buf_49], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_5, buf_0, buf_50, buf_51], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_8, buf_5], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_10, buf_8], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_9, buf_8, buf_10], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_11, buf_8, buf_10, buf_9, buf_5], [53, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_12, buf_0, buf_11, buf_52, buf_53], [4, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_15, buf_12], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_16, buf_12, buf_15], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_11, buf_12, buf_15, buf_16, buf_54, buf_55], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_19, buf_11, buf_56, buf_57], [16, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_12, buf_11, buf_19, buf_58, buf_59], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_16, buf_12], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_15, buf_12, buf_16], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_11, buf_12, buf_16, buf_15, buf_60, buf_61], [4, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [output0, buf_11, buf_62, buf_63, buf_64, buf_65, buf_4], [64, 1, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default model;
"""
