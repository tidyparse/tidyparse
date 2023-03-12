package ai.hypergraph.tidyparse

import kotlinx.serialization.Serializable

@Serializable sealed interface Message
@Serializable sealed interface Request<R : RequestResult> : Message
@Serializable sealed interface RequestResult : Message
@Serializable data class Response(val workerId: String, val result: RequestResult? = null, val error: String? = null): Message

@Serializable data class Sleep(val ms: Long): Request<SleepResult>
@Serializable data class SleepResult(val ms: Long): RequestResult

@Serializable data class PIApproximation(val iterations: Int) : Request<PIApproximationResult>
@Serializable data class PIApproximationResult(val pi: Double) : RequestResult