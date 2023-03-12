import ai.hypergraph.tidyparse.*
import kotlinx.coroutines.*
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.w3c.dom.DedicatedWorkerGlobalScope
import org.w3c.dom.url.URLSearchParams
import kotlin.random.Random

fun main() = worker {
  receiveRequest { request ->
    when (request) {
      is PIApproximation -> PIApproximationResult(approximatePI(request.iterations))
      is Sleep -> {
        delay(request.ms)
        SleepResult(request.ms)
      }
    }
  }
}

fun approximatePI(iterations: Int): Double {
  var inner = 0
  var px: Double
  var py: Double
  repeat(iterations) {
    px = Random.nextDouble(-1.0, 1.0)
    py = Random.nextDouble(-1.0, 1.0)
    if (px * px + py * py <= 1) inner++
  }
  return 4 * inner.toDouble() / iterations
}

fun worker(block: WorkerScope.() -> Unit) {
  val isWorkerGlobalScope = js("typeof(WorkerGlobalScope) !== \"undefined\"") as? Boolean
    ?: throw IllegalStateException("Boolean cast went wrong")
  if (!isWorkerGlobalScope) return

  val self = js("self") as? DedicatedWorkerGlobalScope
    ?: throw IllegalStateException("DedicatedWorkerGlobalScope cast went wrong")
  val scope = WorkerScope(self)
  block(scope)
}

class WorkerScope(private val self: DedicatedWorkerGlobalScope) {
  val workerId = URLSearchParams(self.location.search).get("id") ?: "Unknown worker"

  fun receive(block: suspend (String) -> String) {
    self.onmessage = { messageEvent ->
      GlobalScope.launch {
        self.postMessage(block(messageEvent.data.toString()))
      }
    }
  }

  fun receiveRequest(block: suspend (request: Request<*>) -> RequestResult) = receive { data ->
    val message = Json.decodeFromString<Message>(data)
    val response = try {
      val result = block(message as Request<*>)
      Response(workerId = workerId, result = result, error = null)
    } catch (e: Throwable) {
      Response(workerId = workerId, result = null, error = e.message)
    }
    Json.encodeToString(response)
  }
}