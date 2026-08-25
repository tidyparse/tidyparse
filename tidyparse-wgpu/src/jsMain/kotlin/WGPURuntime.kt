package ai.hypergraph.tidyparse.wgpu

import web.gpu.GPUBuffer

/** Consumer-owned hooks and model buffers used by the reusable WebGPU pipelines. */
private var logSink: (String) -> Unit = ::println
private var statusSink: (String, String) -> Unit = { _, _ -> }

var wdfa: GPUBuffer? = null
var ngrams: GPUBuffer? = null
var wdfaNumStates: Int = 0
var wdfaNumEdges: Int = 0

typealias RepairRerankerCallback = suspend (query: List<String>, candidates: IntersectionResults) -> List<Int>

fun configureWgpuRuntime(
  logger: (String) -> Unit = ::println,
  statusReporter: (state: String, message: String) -> Unit = { _, _ -> }
) {
  logSink = logger
  statusSink = statusReporter
}

internal fun log(message: String) = logSink(message)
internal fun reportRuntimeStatus(state: String, message: String) = statusSink(state, message)
