package ai.hypergraph.kaliningraph.parsing

import com.ionspin.kotlin.bignum.integer.BigInteger
import kotlin.js.JsName
import kotlin.js.unsafeCast

/** An opaque Kotlin view of JavaScript's primitive `bigint`. */
internal external class NativeExactCount {
  fun toString(radix: Int = definedExternally): String
}

internal actual typealias ExactCount = NativeExactCount

@JsName("BigInt")
private external fun nativeExactCount(value: String): NativeExactCount

internal actual val EXACT_COUNT_ZERO: ExactCount = nativeExactCount("0")
internal actual val EXACT_COUNT_ONE: ExactCount = nativeExactCount("1")
private val NATIVE_HASH_MASK: ExactCount = nativeExactCount("4294967295")

internal actual fun exactCountAdd(left: ExactCount, right: ExactCount): ExactCount {
  val lhs = left
  val rhs = right
  return js("lhs + rhs").unsafeCast<ExactCount>()
}

internal actual fun exactCountSubtract(left: ExactCount, right: ExactCount): ExactCount {
  val lhs = left
  val rhs = right
  return js("lhs - rhs").unsafeCast<ExactCount>()
}

internal actual fun exactCountMultiply(left: ExactCount, right: ExactCount): ExactCount {
  val lhs = left
  val rhs = right
  return js("lhs * rhs").unsafeCast<ExactCount>()
}

internal actual fun exactCountDivide(left: ExactCount, right: ExactCount): ExactCount {
  val lhs = left
  val rhs = right
  return js("lhs / rhs").unsafeCast<ExactCount>()
}

internal actual fun exactCountRemainder(left: ExactCount, right: ExactCount): ExactCount {
  val lhs = left
  val rhs = right
  return js("lhs % rhs").unsafeCast<ExactCount>()
}

internal actual fun exactCountCompare(left: ExactCount, right: ExactCount): Int {
  val lhs = left
  val rhs = right
  return js("lhs < rhs ? -1 : lhs > rhs ? 1 : 0").unsafeCast<Int>()
}

internal actual fun exactCountEquals(left: ExactCount, right: ExactCount): Boolean {
  val lhs = left
  val rhs = right
  return js("lhs === rhs").unsafeCast<Boolean>()
}

internal actual fun exactCountHash(value: ExactCount): Int {
  val bigint = value
  val mask = NATIVE_HASH_MASK
  return js("Number(bigint & mask) | 0").unsafeCast<Int>()
}

internal actual fun exactCountFromInt(value: Int): ExactCount {
  val int = value
  return js("BigInt(int)").unsafeCast<ExactCount>()
}

internal actual fun exactCountShiftLeft(value: ExactCount, bitCount: Int): ExactCount {
  val bigint = value
  val bits = bitCount
  return js("bigint << BigInt(bits)").unsafeCast<ExactCount>()
}

internal actual fun exactCountBitLength(value: ExactCount): Int =
  if (exactCountEquals(value, EXACT_COUNT_ZERO)) 0 else value.toString(2).length

internal actual fun BigInteger.toExactCount(): ExactCount = nativeExactCount(toString())
internal actual fun ExactCount.toPublicBigInteger(): BigInteger = BigInteger.parseString(toString(10))
