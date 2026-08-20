package ai.hypergraph.kaliningraph.parsing

import com.ionspin.kotlin.bignum.integer.BigInteger

internal actual typealias ExactCount = BigInteger

internal actual val EXACT_COUNT_ZERO: ExactCount = BigInteger.ZERO
internal actual val EXACT_COUNT_ONE: ExactCount = BigInteger.ONE

internal actual fun exactCountAdd(left: ExactCount, right: ExactCount): ExactCount = left + right
internal actual fun exactCountSubtract(left: ExactCount, right: ExactCount): ExactCount = left - right
internal actual fun exactCountMultiply(left: ExactCount, right: ExactCount): ExactCount = left * right
internal actual fun exactCountDivide(left: ExactCount, right: ExactCount): ExactCount = left / right
internal actual fun exactCountRemainder(left: ExactCount, right: ExactCount): ExactCount = left % right
internal actual fun exactCountCompare(left: ExactCount, right: ExactCount): Int = left.compareTo(right)
internal actual fun exactCountEquals(left: ExactCount, right: ExactCount): Boolean = left == right
internal actual fun exactCountHash(value: ExactCount): Int = value.hashCode()
internal actual fun exactCountFromInt(value: Int): ExactCount = BigInteger.fromInt(value)
internal actual fun exactCountShiftLeft(value: ExactCount, bitCount: Int): ExactCount = value.shl(bitCount)
internal actual fun exactCountBitLength(value: ExactCount): Int = value.bitLength()
internal actual fun BigInteger.toExactCount(): ExactCount = this
internal actual fun ExactCount.toPublicBigInteger(): BigInteger = this
