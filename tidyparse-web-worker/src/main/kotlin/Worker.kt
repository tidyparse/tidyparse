import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.tidyparse.*
import kotlinx.coroutines.*
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.w3c.dom.DedicatedWorkerGlobalScope
import org.w3c.dom.url.URLSearchParams
import kotlin.random.*


fun main() = worker {

// It seems like the CFG crashes the browser at runtime??
//  val cfg = """
//  S -> X
//  X -> A | V | ( X , X ) | X X | ( X )
//  A -> FUN | F | LI | M | L
//  FUN -> fun V `->` X
//  F -> if X then X else X
//  M -> match V with Branch
//  Branch -> `|` X `->` X | Branch Branch
//  L -> let V = X
//  L -> let rec V = X
//  LI -> L in X
//
//  V -> Vexp | ( Vexp ) | List | Vexp Vexp
//  Vexp -> Vname | FunName | Vexp VO Vexp | B
//  Vexp -> ( Vname , Vname ) | Vexp Vexp | I
//  List -> [] | V :: V
//  Vname -> a | b | c | d | e | f | g | h | i
//  Vname -> j | k | l | m | n | o | p | q | r
//  Vname -> s | t | u | v | w | x | y | z
//  FunName -> foldright | map | filter
//  FunName -> curry | uncurry | ( VO )
//  VO -> + | - | * | / | >
//  VO -> = | < | `||` | `&&`
//  I -> 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
//  B ->  true | false
//""".trimIndent().parseCFG()

  receiveRequest { request ->
    when (request) {
      is RepairRequest -> {
        console.log("Received: ${request.string}")
//        repair(
//          prompt = request.string,
//          cfg = setOf(),
//          synthesizer = { it.solve(setOf(), takeMoreWhile = { true }) },
//          updateProgress = {         console.log("Repair found: ${it}")
//          }
//        )
        "_ _ _".split(" ").solve(setOf(), takeMoreWhile = { true })
        delay(Random.nextLong(0L until 1000L))
            RepairResult(request.string)
      }
      is Sleep -> {
        delay(request.ms)
        SleepResult(request.ms)
      }
    }
  }
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