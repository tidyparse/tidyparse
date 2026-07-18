import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.async

data class PyCompileResult(val output: String)

class WebWorkerPool(private val indexURL: String, private val size: Int) {
  private class Slot(val worker: dynamic) { val pending = mutableMapOf<Int, CompletableDeferred<dynamic>>() }

  private var nextId = 1
  private var nextSlot = 0
  private val workerURL = createPyodideWorkerURL()

  private fun makeWorker(url: String): dynamic = js("(url) => new Worker(url)")(url)

  private fun makeSlot(): Slot {
    val worker = makeWorker(workerURL)

    return Slot(worker).also { slot ->
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

  private fun makeSlots(): Array<Slot> = Array(size) { makeSlot() }

  private var slots: Array<Slot> = makeSlots()

  private fun terminateSlots(slots: Array<Slot>) {
    slots.forEach { it.worker.terminate() }
  }

  private fun resetSlots() {
    terminateSlots(slots)
    slots = makeSlots()
    nextSlot = 0
  }

  private fun throwInfraError(data: dynamic, context: String) {
    val infraError = data.infraError as? String
    if (!infraError.isNullOrBlank()) throw RuntimeException("$context: $infraError")
  }

  private suspend fun request(slot: Slot, op: String, src: String? = null, snapshot: dynamic = null): dynamic {
    val id = nextId++
    val deferred = CompletableDeferred<dynamic>()
    slot.pending[id] = deferred

    val msg = js("{}")
    msg.id = id
    msg.op = op
    msg.indexURL = indexURL
    if (src != null) msg.src = src
    if (snapshot != null) msg.snapshot = snapshot

    slot.worker.postMessage(msg)
    return deferred.await()
  }

  private fun nextWorkerSlot(): Slot {
    val slot = slots[nextSlot]
    nextSlot = (nextSlot + 1) % slots.size
    return slot
  }

  suspend fun init() {
    if (tryInitFromSnapshot()) return
    initCold()
  }

  private suspend fun initCold() =
    coroutineScope {
      slots.map { slot ->
        async { throwInfraError(request(slot, "init"), "Pyodide worker cold init failed") }
      }.awaitAll()
    }

  private suspend fun buildSnapshot(op: String, label: String): dynamic {
    val builder = makeSlot()

    return try {
      log("Building $label Pyodide worker memory snapshot")
      val snapshotData = request(builder, op)
      throwInfraError(snapshotData, "Pyodide $label snapshot build failed")

      if (snapshotData.snapshot == null) throw RuntimeException("Pyodide $label snapshot build returned no snapshot")
      log("Built $label Pyodide worker memory snapshot (${snapshotData.snapshotBytes} bytes)")
      snapshotData
    } finally {
      builder.worker.terminate()
    }
  }

  private fun shouldTryWarmBlackSnapshot(): Boolean =
    try {
      js("(() => { try { return !globalThis.localStorage || globalThis.localStorage.getItem('tidyparseWarmBlackSnapshot') !== '0'; } catch (_) { return true; } })()") as Boolean
    } catch (_: Throwable) { true }

  private suspend fun tryInitFromSnapshot(): Boolean {
    return try {
      val snapshotData =
        if (shouldTryWarmBlackSnapshot()) {
          try {
            buildSnapshot("snapshot", "warm")
          } catch (warmFailure: Throwable) {
            log("Warm Pyodide worker memory snapshot failed; trying base snapshot: ${warmFailure.message ?: warmFailure}")
            buildSnapshot("snapshotBase", "base")
          }
        } else buildSnapshot("snapshotBase", "base")

      val snapshot = snapshotData.snapshot

      coroutineScope {
        slots.map { slot ->
          async { throwInfraError(request(slot, "initSnapshot", snapshot = snapshot), "Pyodide snapshot restore failed") }
        }.awaitAll()
      }

      log("Initialized $size Python Web Workers from ${snapshotData.snapshotKind} snapshot")
      true
    } catch (t: Throwable) {
      log("Pyodide memory snapshot warmup failed; falling back to cold workers: ${t.message ?: t}")
      resetSlots()
      false
    }
  }

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
    terminateSlots(slots)
    revokePyodideWorkerURL(workerURL)
  }
}
