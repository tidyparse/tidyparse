import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.*
import kotlinx.coroutines.*
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.w3c.dom.DedicatedWorkerGlobalScope
import org.w3c.dom.url.URLSearchParams

fun main() = worker {
  onRequestReceived { request: Request<*> ->
    when (request) {
      is RepairRequest -> {
        repair(
          prompt = request.string,
          cfg = cfg,
          synthesizer = { it.solve(cfg, takeMoreWhile = { true }) },
          updateProgress = { respondWith(RepairResult(it + "\n" + request.requestId)) }
        )
      }
      is Sleep -> {
        delay(request.ms)
        respondWith(SleepResult(request.ms))
      }
    }
  }
}


fun worker(block: MutliResponseWorkerScope.() -> Unit) {
  val isWorkerGlobalScope = js("typeof(WorkerGlobalScope) !== \"undefined\"") as? Boolean
    ?: throw IllegalStateException("Boolean cast went wrong")
  if (!isWorkerGlobalScope) return

  val self = js("self") as? DedicatedWorkerGlobalScope
    ?: throw IllegalStateException("DedicatedWorkerGlobalScope cast went wrong")
  val scope = MutliResponseWorkerScope(self)
  block(scope)
}

class MutliResponseWorkerScope(private val self: DedicatedWorkerGlobalScope) {
  val workerId = URLSearchParams(self.location.search).get("id") ?: "Unknown worker"

  fun receive(block: suspend (String) -> Unit) {
    self.onmessage = { messageEvent ->
      GlobalScope.launch {
        block(messageEvent.data.toString())
      }
    }
  }

  fun respondWith(response: RequestResult) =
    self.postMessage(Json.encodeToString(Response(workerId = workerId, result = response, error = null)))

  fun onRequestReceived(block: suspend (request: Request<*>) -> Unit) = receive { data ->
    val message = Json.decodeFromString<Message>(data)
    try {
      block(message as Request<*>)
    } catch (e: Throwable) {
      self.postMessage(Json.encodeToString(Response(workerId = workerId, result = null, error = e.message)))
    }
  }
}