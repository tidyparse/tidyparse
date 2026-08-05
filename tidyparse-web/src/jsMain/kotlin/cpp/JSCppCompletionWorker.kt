import cppcompletion.CPP_MAX_INTERACTIVE_COMPLETIONS
import cppcompletion.CppCompletionGrammar
import cppcompletion.CppToken
import cppcompletion.CppTokenKind
import cppcompletion.PreparedCppCompletionGrammar
import cppcompletion.cppLines
import cppcompletion.completeCppStatement
import kotlinx.browser.window
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred

internal const val CPP_COMPLETION_WORKER_NAME = "tidyparse-cpp-completion"
private const val CPP_COMPLETION_PREPARED_CACHE_SIZE = 3

private val cppCompletionWorkerScope: dynamic
  get() = js("globalThis")

fun isCppCompletionWorkerRuntime(): Boolean =
  js(
    """(name) => typeof document === "undefined" &&
       typeof globalThis.postMessage === "function" &&
       globalThis.name === name"""
  )(CPP_COMPLETION_WORKER_NAME) as Boolean

fun setupCppCompletionWorker() {
  val scope = cppCompletionWorkerScope
  if (scope.__tidyparseCppCompletionStarted == true) return
  scope.__tidyparseCppCompletionStarted = true

  val runtime = CppCompletionWorkerRuntime(scope)
  scope.onmessage = { event: dynamic -> runtime.accept(event.data) }
  scope.onmessageerror = {
    val reply = cppCompletionReply(-1, ok = false)
    reply.error = "The C++ completion worker received an unreadable message"
    scope.postMessage(reply)
  }
  val ready = js("({})")
  ready.type = "ready"
  ready.ok = true
  scope.postMessage(ready)
}

/** All mutable grammar state stays inside the dedicated worker. */
private class CppCompletionWorkerRuntime(private val scope: dynamic) {
  // Kotlin's LinkedHashMap preserves insertion order. Hits are removed and reinserted, giving
  // this tiny map ordinary LRU behavior without retaining a document-sized cache.
  private val prepared = linkedMapOf<String, PreparedCppCompletionGrammar>()
  private val grammar = CppCompletionGrammar()

  fun accept(rawRequest: dynamic) {
    val requestId = cppCompletionInt(rawRequest?.id, -1)
    try {
      require(rawRequest != null && jsTypeOf(rawRequest) != "undefined") { "Missing C++ completion request" }
      require(rawRequest.type as? String == "complete") { "Unsupported C++ completion worker request" }
      require(requestId >= 0) { "A C++ completion request id must be nonnegative" }

      val source = rawRequest.source as? String ?: error("Missing C++ source text")
      val prefixText = rawRequest.statementPrefixText as? String
        ?: error("Missing C++ statement prefix text")
      val semanticPrefixText = rawRequest.semanticPrefixText as? String ?: prefixText
      require('\n' !in prefixText && '\r' !in prefixText) {
        "A C++ completion request must contain one physical statement prefix"
      }
      val prefix = cppCompletionPrefixTokens(rawRequest.prefixTokens, prefixText)
      val limit = cppCompletionInt(rawRequest.limit, CPP_MAX_INTERACTIVE_COMPLETIONS)
      require(limit in 1..CPP_MAX_INTERACTIVE_COMPLETIONS)
      val seed = cppCompletionInt(rawRequest.seed)
      val line = cppCompletionInt(rawRequest.line, -1)
      val character = cppCompletionInt(rawRequest.character, -1)
      require(line >= 0 && character >= 0) { "Invalid C++ completion cursor" }
      val statementStartCharacter = cppCompletionInt(rawRequest.statementStartCharacter)
      val replacementEndCharacter = cppCompletionInt(rawRequest.replacementEndCharacter, character)
      val snapshot = CppEditorStatementSnapshot(
        line = line,
        character = character,
        statementStartCharacter = statementStartCharacter,
        prefixText = prefixText,
        semanticPrefixText = semanticPrefixText,
        tokens = prefix,
        replacementEndCharacter = replacementEndCharacter,
        cacheKey = rawRequest.cacheKey as? String ?: "",
        seed = seed
      )

      val facts = rawRequest.facts
      val completionGroups = cppCompletionDynamicArray(facts?.completionGroups).map { group ->
        CppClangdCompletionGroup(
          result = group.result,
          receiverMember = group.receiverMember as? Boolean ?: false,
          receiverOperator = group.receiverOperator as? String
        )
      }
      val rawAst = facts?.ast
      val ast = if (
        rawAst != null && rawAst != js("undefined") &&
        rawAst[CPP_NORMALIZED_AST_CONTEXT_FIELD] as? Boolean == true
      ) rawAst
      else cppClangdAstContextDto(rawAst, source, line, character)
      val contextDto = cppCompletionContextDto(
        source = source,
        completionGroups = completionGroups,
        signatures = facts?.signatures,
        hover = facts?.hover,
        diagnostics = facts?.diagnostics,
        ast = ast,
        snapshot = snapshot
      )
      val context = cppCompletionContextFromDto(contextDto)
      val query = snapshot.completionQuery(context.identifiers, limit, seed)

      val cacheKey = rawRequest.cacheKey as? String ?: ""
      val exactKey = cppCompletionExactPreparedKey(
        cacheKey,
        contextDto,
        query.prefixText,
        query.prefix
      )
      val cached = exactKey.takeIf(String::isNotEmpty)?.let(prepared::remove)
      val activeGrammar = cached ?: grammar.prepare(context, query.prefix)
      if (exactKey.isNotEmpty()) {
        prepared[exactKey] = activeGrammar
        while (prepared.size > CPP_COMPLETION_PREPARED_CACHE_SIZE) {
          prepared.remove(prepared.keys.first())
        }
      }
      val execution = activeGrammar.completeCppStatement(query)
      val suggestions = execution.suggestions.map { completion ->
        val suggestion = js("({})")
        suggestion.candidateText = completion.candidateText
        suggestion.tokenLength = completion.tokenLength
        suggestion
      }.toTypedArray()

      val reply = cppCompletionReply(requestId, ok = true)
      reply.suggestions = suggestions
      scope.postMessage(reply)
    } catch (failure: Throwable) {
      val reply = cppCompletionReply(requestId, ok = false)
      reply.error = failure.message ?: failure.toString()
      reply.stack = failure.asDynamic().stack as? String
      scope.postMessage(reply)
    }
  }
}

private fun cppCompletionExactPreparedKey(
  cacheKey: String,
  contextDto: dynamic,
  prefixText: String,
  prefix: List<CppToken>
): String {
  if (cacheKey.isBlank()) return ""
  val contextJson = try { JSON.stringify(contextDto) } catch (_: Throwable) { return "" }
  val tokenKey = prefix.joinToString("\u0001") { token ->
    "${token.start}:${token.end}:${token.kind.name}:${token.text}"
  }
  // External keys are only a cache namespace. Exact semantic and lexical inputs remain in the
  // identity, so an accidentally coarse document key can reduce hit rate but never affect output.
  return "$cacheKey\u0000$contextJson\u0000$prefixText\u0000$tokenKey"
}

internal fun cppCompletionPrefixTokens(serialized: dynamic, prefixText: String): List<CppToken> {
  val items = cppCompletionDynamicArray(serialized)
  if (js("Array.isArray(serialized)") as Boolean) {
    return items.mapIndexed { index, raw ->
      // [raw] already has the dynamic type. Calling asDynamic() on a dynamic receiver is emitted
      // as a JavaScript member call (`raw.asDynamic()`), which plain structured-clone objects do
      // not provide. Keep the value dynamic and read its DTO fields directly.
      val item = raw
      val text = item.text as? String ?: error("Missing text for C++ prefix token $index")
      val kindName = item.kind as? String ?: error("Missing kind for C++ prefix token $index")
      val kind = CppTokenKind.values().firstOrNull { it.name == kindName }
        ?: error("Unknown C++ prefix token kind '$kindName'")
      val start = cppCompletionInt(item.start, -1)
      val end = cppCompletionInt(item.end, -1)
      require(start >= 0 && end >= start) { "Invalid source range for C++ prefix token $index" }
      CppToken(text, start, end, kind, item.completeText as? String)
    }
  }

  // This fallback keeps hand-built protocol requests useful while the public DTO helper always
  // sends explicit tokens. cppLines retains ANTLR token kinds and adjacency offsets.
  return cppLines(prefixText).single().tokens
}

private fun cppCompletionDynamicArray(value: dynamic): List<dynamic> {
  if (!(js("Array.isArray(value)") as Boolean)) return emptyList()
  return (0 until cppCompletionInt(value.length)).map { index -> value[index] }
}

private fun cppCompletionReply(requestId: Int, ok: Boolean): dynamic {
  val reply = js("({})")
  reply.type = "result"
  reply.id = requestId
  reply.ok = ok
  return reply
}

/**
 * Creates the plain JavaScript request consumed by [CppCompletionWorkerClient].
 *
 * This is the browser's single completion boundary: the page supplies source, one immutable
 * statement snapshot, and browser-available clangd facts. Context normalization and the complete
 * grammar pipeline both run behind this request in the dedicated worker.
 */
fun cppCompletionWorkerRequest(
  cacheKey: String,
  source: String,
  snapshot: CppEditorStatementSnapshot,
  facts: CppCompletionSemanticFacts = CppCompletionSemanticFacts(),
  limit: Int = CPP_MAX_INTERACTIVE_COMPLETIONS
): dynamic {
  require(limit in 1..CPP_MAX_INTERACTIVE_COMPLETIONS)
  require('\n' !in snapshot.prefixText && '\r' !in snapshot.prefixText) {
    "A C++ completion request must contain one physical statement prefix"
  }

  val request = js("({})")
  request.type = "complete"
  // The client replaces this placeholder immediately before posting the request. Keeping it in
  // the DTO makes the wire shape explicit and lets protocol fixtures use the same helper.
  request.id = 0
  request.cacheKey = cacheKey
  request.source = source
  request.line = snapshot.line
  request.character = snapshot.character
  request.statementStartCharacter = snapshot.statementStartCharacter
  request.replacementEndCharacter = snapshot.replacementEndCharacter
  request.statementPrefixText = snapshot.prefixText
  request.semanticPrefixText = snapshot.semanticPrefixText
  request.prefixTokens = snapshot.tokens.map { token ->
    val serialized = js("({})")
    serialized.text = token.text
    serialized.start = token.start
    serialized.end = token.end
    serialized.kind = token.kind.name
    serialized.completeText = token.completeText
    serialized
  }.toTypedArray()
  request.seed = snapshot.seed
  request.facts = js("({})")
  request.facts.completionGroups = facts.completionGroups.map { group ->
    val serialized = js("({})")
    serialized.result = group.result
    serialized.receiverMember = group.receiverMember
    serialized.receiverOperator = group.receiverOperator
    serialized
  }.toTypedArray()
  request.facts.signatures = facts.signatures
  request.facts.hover = facts.hover
  request.facts.diagnostics = facts.diagnostics
  request.facts.ast = facts.ast
  request.limit = limit
  // Raw LSP values and Kotlin arrays become one owned graph of plain structured-clone fields.
  return cppCompletionJsonClone(request)
}

/**
 * One dedicated, lazily reusable grammar worker for the C++ editor.
 *
 * A newer request makes all older requests stale immediately. The old worker computation may
 * finish (JavaScript cannot interrupt synchronous code inside a worker), but its reply is ignored
 * and can never populate a newer Monaco completion session.
 */
class CppCompletionWorkerClient internal constructor(
  private val workerFactory: () -> dynamic = ::createCppCompletionWorker
) {
  private var worker: dynamic = workerFactory()
  private val pending = mutableMapOf<Int, CompletableDeferred<dynamic>>()
  private var ready = CompletableDeferred<Unit>()
  private var nextRequestId = 1
  private var latestRequestId = 0
  private var disposed = false
  private var failed = false

  init {
    bindWorkerCallbacks()
  }

  /** Completes only after the worker bundle has evaluated and installed its message handler. */
  suspend fun awaitReady() {
    check(!disposed) { "The C++ completion worker client has been disposed" }
    // A script-evaluation/message error poisons only that Worker instance. Retrying on the next
    // explicit action is both bounded by Monaco's outer timeout and avoids retaining a permanently
    // failed client until the entire editor is recreated.
    if (failed) restartWorker("Retrying the C++ completion worker")
    val readiness = ready
    try {
      readiness.await()
    } catch (cancelled: CancellationException) {
      // [complete]'s ordinary cancellation handler starts only after the ready handshake. If the
      // browser never evaluates the worker bundle, the outer timeout cancels this await instead.
      // Replace that wedged instance so the next completion does not repeat the same timeout.
      if (!disposed && ready === readiness && !readiness.isCompleted)
        restartWorker("Cancelled while waiting for the C++ completion worker")
      throw cancelled
    }
  }

  suspend fun complete(request: dynamic): dynamic {
    check(!disposed) { "The C++ completion worker client has been disposed" }
    require(request != null && jsTypeOf(request) != "undefined") {
      "A C++ completion worker request is required"
    }
    awaitReady()

    // Grammar construction is synchronous inside its worker. Terminating a superseded worker is
    // the only way to prevent an obsolete long task from blocking the new request behind it.
    if (pending.isNotEmpty()) restartWorker("Superseded by a newer C++ completion request")
    check(!failed) { "The C++ completion worker is unavailable" }

    val requestId = allocateRequestId()
    latestRequestId = requestId

    val deferred = CompletableDeferred<dynamic>()
    pending[requestId] = deferred
    val payload = cppCompletionJsonClone(request)
    payload.type = "complete"
    payload.id = requestId

    try {
      worker.postMessage(payload)
    } catch (failure: Throwable) {
      pending.remove(requestId)
      deferred.completeExceptionally(failure)
    }
    return try {
      deferred.await()
    } catch (cancelled: CancellationException) {
      val wasActive = pending.remove(requestId) != null && latestRequestId == requestId
      if (wasActive && !disposed) restartWorker("Cancelled C++ completion request $requestId")
      throw cancelled
    }
  }

  fun dispose() {
    if (disposed) return
    disposed = true
    latestRequestId = 0
    ready.completeExceptionally(
      CancellationException("The C++ completion worker client was disposed")
    )
    cancelPending("The C++ completion worker client was disposed")
    terminateWorker()
    worker = null
  }

  private fun bindWorkerCallbacks() {
    worker.onmessage = { event: dynamic ->
      val reply = event.data
      if (reply?.type as? String == "ready") {
        ready.complete(Unit)
      } else {
        val requestId = cppCompletionInt(reply?.id, -1)
        val deferred = pending.remove(requestId)
        if (deferred != null && requestId == latestRequestId) {
          if (reply?.ok as? Boolean == true) {
            deferred.complete(reply)
          } else {
            deferred.completeExceptionally(
              IllegalStateException(reply?.error as? String ?: "C++ completion failed")
            )
          }
        }
      }
    }
    worker.onerror = { event: dynamic ->
      try {
        event.preventDefault()
      } catch (_: Throwable) {
      }
      failWorker(event?.message as? String ?: "The C++ completion worker failed")
      true
    }
    worker.onmessageerror = {
      failWorker("The C++ completion worker returned an unreadable message")
    }
  }

  private fun failWorker(message: String) {
    if (disposed || failed) return
    failed = true
    val failure = IllegalStateException(message)
    ready.completeExceptionally(failure)
    val requests = pending.values.toList()
    pending.clear()
    requests.forEach { it.completeExceptionally(failure) }
    terminateWorker()
  }

  private fun restartWorker(message: String) {
    latestRequestId = 0
    cancelPending(message)
    terminateWorker()
    failed = false
    ready = CompletableDeferred()
    try {
      worker = workerFactory()
      bindWorkerCallbacks()
    } catch (failure: Throwable) {
      failed = true
      worker = null
      throw failure
    }
  }

  private fun terminateWorker() {
    try {
      worker?.onmessage = null
      worker?.onerror = null
      worker?.onmessageerror = null
      worker?.terminate()
    } catch (_: Throwable) {
    }
  }

  private fun cancelPending(message: String) {
    if (pending.isEmpty()) return
    val cancellation = CancellationException(message)
    val requests = pending.values.toList()
    pending.clear()
    requests.forEach { it.completeExceptionally(cancellation) }
  }

  private fun allocateRequestId(): Int {
    val result = nextRequestId
    nextRequestId = if (nextRequestId == Int.MAX_VALUE) 1 else nextRequestId + 1
    return result
  }
}

/**
 * Resolves the same single bundle in an ordinary page and under the exact-page COI service worker.
 * The service worker currently recognizes the compatibility `cpp-worker=clangd` route; the
 * dedicated worker name, not that query key, selects this completion runtime in [main].
 */
private fun createCppCompletionWorker(): dynamic {
  val url = js(
    """(href) => {
      const controller = navigator.serviceWorker && navigator.serviceWorker.controller;
      const controllerUrl = controller && new URL(controller.scriptURL);
      const cppBootstrapControlsPage =
        controllerUrl &&
        controllerUrl.pathname.endsWith("/tidyparse-web.js") &&
        controllerUrl.searchParams.has("cpp-coi");
      const result = cppBootstrapControlsPage
        ? new URL(href)
        : new URL("tidyparse-web.js", href);
      result.search = "";
      result.hash = "";
      if (cppBootstrapControlsPage) result.searchParams.set("cpp-worker", "clangd");
      return result.href;
    }"""
  )(window.location.href) as String
  return js("(url, name) => new Worker(url, { name })")(url, CPP_COMPLETION_WORKER_NAME)
}

internal fun cppCompletionJsonClone(value: dynamic): dynamic =
  js("(value) => JSON.parse(JSON.stringify(value))")(value)

internal fun cppCompletionInt(value: dynamic, fallback: Int = 0): Int = (value as? Number)?.toInt() ?: fallback
