
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.*
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.*
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

val mutex = Mutex()
var counter = 0
val results = mutableListOf<RepairResult>()

suspend fun workerPool() {
  val pool = WorkerPool(10, "./tidyparse-web-worker.js")
  val requestCounter = mutex.withLock { ++counter }

  GlobalScope.launch {
    RepairRequest(inputField.getCurrentLine() + "\n" + requestCounter)
      .let { repair -> makeRequestAndWaitForTenRepairs(pool, repair) }
  }
}

suspend fun makeRequestAndWaitForTenRepairs(pool: WorkerPool, repair: RepairRequest) {
  val results = mutableListOf<RepairResult>()
  while (results.size < 10) {
    val result = pool.request(repair)
    results.add(result)
    result.updateTextArea(mutex)
  }
}

suspend fun RepairResult.updateTextArea(mutex: Mutex) {
  mutex.withLock {
    console.log("Received: $requestId / $counter")
    results.removeAll { it.requestId < counter }
    if (requestId == counter) {
      results.add(this)
      outputField.textContent = results.sortedBy { it.requestId }.joinToString("\n") { "${it.requestId}: ${it.message}" }
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

//  fun terminateAll() {
//    availableWorkers.forEach { worker -> worker.terminate() }
//    jobs.forEach { job -> job.continuation.resumeWithException(WorkerException("Worker pool terminated")) }
//  }

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