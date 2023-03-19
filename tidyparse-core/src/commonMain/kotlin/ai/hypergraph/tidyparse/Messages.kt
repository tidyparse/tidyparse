package ai.hypergraph.tidyparse

import kotlinx.serialization.Serializable

@Serializable sealed interface Message
@Serializable sealed interface Request<R : RequestResult> : Message
@Serializable sealed interface RequestResult : Message

@Serializable data class Response(val workerId: String, val result: RequestResult? = null, val error: String? = null): Message

@Serializable data class Sleep(val ms: Long): Request<SleepResult>
@Serializable data class SleepResult(val ms: Long): RequestResult

@Serializable data class RepairRequest(val string: String) : Request<RepairResult>
@Serializable data class RepairResult(val string: String) : RequestResult
val RepairResult.requestId get() = string.lines().last().toInt()
val RepairResult.message get() = string.lines().first()
val RepairRequest.requestId get() = string.lines().last().toInt()
val RepairRequest.message get() = string.lines().first()
