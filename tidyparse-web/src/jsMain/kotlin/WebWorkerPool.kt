import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.async

data class PyCompileResult(val output: String)

class WebWorkerPool(private val indexURL: String, size: Int) {
  private class Slot(val worker: dynamic) { val pending = mutableMapOf<Int, CompletableDeferred<dynamic>>() }

  private var nextId = 1
  private var nextSlot = 0
  private val workerURL = createPyodideWorkerURL()

  private fun makeWorker(url: String): dynamic = js("(url) => new Worker(url)")(url)

  private val slots: Array<Slot> = Array(size) {
    val worker = makeWorker(workerURL)

    Slot(worker).also { slot ->
      worker.onmessage = { ev: dynamic ->
        val data = ev.data
        val id = data.id as Int
        val deferred = slot.pending.remove(id)

        if (deferred != null) {
          if ((data.ok as? Boolean) != false) deferred.complete(data)
          else deferred.completeExceptionally(RuntimeException(data.error as String))
        }
      }

      worker.onerror = { ev: dynamic ->
        try { ev.preventDefault() } catch (_: dynamic) {}

        val pending = slot.pending.values.toList()
        slot.pending.clear()

        pending.forEach { deferred ->
          val data = js("{}")
          data.ok = true
          data.output = ""
          data.formatted = ""
          data.infraError = ev.message ?: "Web Worker error"
          deferred.complete(data)
        }

        true
      }
    }
  }

  private suspend fun request(slot: Slot, op: String, src: String? = null): dynamic {
    val id = nextId++
    val deferred = CompletableDeferred<dynamic>()
    slot.pending[id] = deferred

    val msg = js("{}")
    msg.id = id
    msg.op = op
    msg.indexURL = indexURL
    if (src != null) msg.src = src

    slot.worker.postMessage(msg)
    return deferred.await()
  }

  private fun nextWorkerSlot(): Slot {
    val slot = slots[nextSlot]
    nextSlot = (nextSlot + 1) % slots.size
    return slot
  }

  suspend fun init() =
    coroutineScope { slots.map { slot -> async { request(slot, "init") } }.awaitAll() }

  suspend fun compile(src: String): PyCompileResult {
    val data = request(nextWorkerSlot(), "compile", src)
    val infraError = data.infraError as? String
    if (!infraError.isNullOrBlank()) throw RuntimeException(infraError)
    return PyCompileResult(output = (data.output as? String) ?: "")
  }

  private fun normalizeBlackOutput(s: String): String = s.trim().lineSequence().joinToString(" ") { it.trim() }

  suspend fun format(src: String): String {
    val data = request(nextWorkerSlot(), "format", src)
    val infraError = data.infraError as? String
    if (!infraError.isNullOrBlank()) throw RuntimeException(infraError)
    val formatted = (data.formatted as? String) ?: (data.output as? String) ?: src
    return normalizeBlackOutput(formatted)
  }

  fun terminate() {
    slots.forEach { it.worker.terminate() }
    revokePyodideWorkerURL(workerURL)
  }
}
