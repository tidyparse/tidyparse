import kotlinx.coroutines.MainScope
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.promise
import kotlinx.coroutines.yield
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CppCompletionWorkerClientTest {
  @Test
  fun cancellationDuringReadyHandshakeReplacesTheWedgedWorker(): Promise<Unit> =
    MainScope().promise {
      val workers = mutableListOf<dynamic>()
      val client = CppCompletionWorkerClient {
        val created = fakeWorker()
        workers.add(workers.size, created)
        created
      }
      val waiting = async { client.complete(completionRequest()) }
      yield()

      waiting.cancelAndJoin()

      assertEquals(2, workers.size)
      assertTrue(workers.first().terminated as Boolean)
      publishReady(workers.last())
      client.complete(completionRequest())
      client.dispose()
    }

  @Test
  fun nextExplicitActionRetriesAfterAWorkerError(): Promise<Unit> = MainScope().promise {
    val workers = mutableListOf<dynamic>()
    val client = CppCompletionWorkerClient {
      val created = fakeWorker()
      workers.add(workers.size, created)
      created
    }
    val error = js("({ message: 'failed to evaluate completion bundle' })")
    error.preventDefault = {}
    workers.single().onerror(error)

    val retry = async { client.complete(completionRequest()) }
    yield()

    assertEquals(2, workers.size)
    assertTrue(workers.first().terminated as Boolean)
    publishReady(workers.last())
    retry.await()
    client.dispose()
  }

  private fun fakeWorker(): dynamic {
    val worker = js("({ terminated: false })")
    worker.postMessage = { request: dynamic ->
      val reply = js("({ type: 'result', ok: true })")
      reply.id = request.id
      val event = js("({})")
      event.data = reply
      worker.onmessage(event)
    }
    worker.terminate = { worker.terminated = true }
    return worker
  }

  private fun publishReady(worker: dynamic) {
    val reply = js("({ type: 'ready', ok: true })")
    val event = js("({})")
    event.data = reply
    worker.onmessage(event)
  }

  private fun completionRequest(): dynamic = js("({ type: 'complete' })")
}
