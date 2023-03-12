
import ai.hypergraph.tidyparse.*
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.w3c.dom.*
import kotlin.coroutines.Continuation
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine
import kotlin.math.min
import kotlin.random.Random

class WorkerException(message: String?) : Throwable(message)

suspend fun Worker.send(data: String) = suspendCoroutine<MessageEvent> { continuation ->
  this.onmessage = { messageEvent ->
    continuation.resume(messageEvent)
  }
  this.onerror = { event -> continuation.resumeWithException(WorkerException(event.type))}
  this.postMessage(data)
}

@Suppress("UNCHECKED_CAST")
suspend fun <R : RequestResult> Worker.request(request: Request<R>): R {
  val data = Json.encodeToString(request as Request<RequestResult>)
  val messageEvent = send(data)
  val response = Json.decodeFromString<Response>(messageEvent.data.toString())

  if (response.error != null) throw WorkerException(response.error)
  return response.result as R
}


external val self: DedicatedWorkerGlobalScope
fun workerPool() {
  val pool = WorkerPool(10, "./worker.js")
  repeat(20) { i ->
    GlobalScope.launch {
      when {
        i % 2 == 0 -> console.log("PI approximation: ", pool.request(PIApproximation(10000000)).pi)
        else -> console.log("Sleeping for: ", pool.request(Sleep(Random.nextLong(500, 5000))).ms.toString())
      }
    }
  }
}


class WorkerPool(size: Int, private val workerScript: String) {

  data class Job(val data: String, val continuation: Continuation<String>) {
    suspend fun execute(worker: Worker) {
      try {
        val response = worker.send(data)
        continuation.resume(response.data.toString())
      } catch (t: Throwable) {
        continuation.resumeWithException(t)
      }
    }
  }

  private val availableWorkers = ArrayDeque<Worker>()
  private val jobs = ArrayDeque<Job>()

  init {
    repeat(size) { nr ->
      availableWorkers.addLast(Worker("$workerScript?id=Worker-$nr"))
    }
  }

  suspend fun send(data: String) = suspendCoroutine<String> { continuation ->
    jobs.addLast(Job(data, continuation))
    checkAvailableWork()
  }

  @Suppress("UNCHECKED_CAST")
  suspend fun <R : RequestResult> request(request: Request<R>): R {
    val data = Json.encodeToString(request as Request<RequestResult>)
    val response = send(data)
    val deserialized = Json.decodeFromString<Response>(response)

    if (deserialized.error != null) throw WorkerException(deserialized.error)
    return deserialized.result as R
  }

  private fun checkAvailableWork() {
    if (jobs.isEmpty() || availableWorkers.isEmpty()) return
    val noOfMessages = min(jobs.size, availableWorkers.size)
    val work = (0 until noOfMessages).map { jobs.removeFirst() to availableWorkers.removeFirst() }
    work.forEach { (job, worker) ->
      GlobalScope.launch {
        job.execute(worker)
        availableWorkers.addLast(worker)
        checkAvailableWork()
      }
    }
  }
}