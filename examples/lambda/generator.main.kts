#!/usr/bin/env kotlin

val names = ('a'..'z').map { it.toString() }.toSet()
val cfg = generateRelevantLambdaCFG(names)
println(cfg.lines().size)

/**
 * Nonterminal t[S,U] is encoded as: t:<S>/u:<U>
 *   S = in-scope names; U ⊆ S = names still available for one use.
 * Start symbol: t:empty/u:empty.
 * Assumption: no shadowing (λ x requires x ∉ S).
 */
fun generateAffineLambdaCFG(names: Set<String>): String {
  require(names.isNotEmpty()) { "Provide at least one name." }
  val base = names.sorted()

  // All pairs (S, U) with U ⊆ S
  val pairs = mutableListOf<Pair<Set<String>, Set<String>>>()
  for (S in powerSet(base)) {
    val sList = S.toList()
    for (U in powerSet(sList)) pairs += S to U
  }

  // Stable ordering for readability
  val order = compareBy<Pair<Set<String>, Set<String>>>(
    { it.first.size },
    { it.first.sorted().joinToString(",") },
    { it.second.size },
    { it.second.sorted().joinToString(",") })
  pairs.sortWith(order)

  fun id(set: Set<String>) = if (set.isEmpty()) "∅" else set.sorted().joinToString(",")
  fun nt(S: Set<String>, U: Set<String>) = "t:${id(S)}/u:${id(U)}"

  val out = mutableListOf<String>()
  out += "START -> ${nt(emptySet(), emptySet())}"

  for ((S, U) in pairs) {
    val alts = linkedSetOf<String>()

    // Application: split U into disjoint U1, U2 (unused names may be in neither)
    // Enumerate U1 ⊆ U, then U2 ⊆ (U \\ U1)
    for (U1 in powerSet(U.toList())) {
      val remaining = U - U1
      for (U2 in powerSet(remaining.toList())) alts += "( ${nt(S, U1)} ${nt(S, U2)} )"
    }

    // Lambda: introduce fresh x (no shadowing), make it available once
    for (x in base) if (x !in S) {
      val S2 = S + x
      val U2 = U + x
      alts += "λ $x . ${nt(S2, U2)}"
    }

    // Variable use: only if x is currently available
    for (x in U) alts += x

    out += "${nt(S, U)} -> " + alts.joinToString(" | ")
  }

  return out.joinToString("\n")
}

/**
 * Nonterminal t[S,M] is encoded as: t:<S>/m:<M>
 *   S = in-scope names; M ⊆ S = names that must be used at least once.
 * Start symbol is t:empty/m:empty.
 * Assumption: no shadowing (λ x requires x ∉ S).
 */
fun generateRelevantLambdaCFG(names: Set<String>): String {
  require(names.isNotEmpty()) { "Provide at least one name." }
  val base = names.sorted()

  // All pairs (S, M) with M ⊆ S
  val pairs = mutableListOf<Pair<Set<String>, Set<String>>>()
  for (S in powerSet(base)) {
    for (M in powerSet(S.toList())) pairs += S to M
  }
  // Stable order: by |S|, then S lexicographically, then |M|, then M lexicographically
  val pairOrder = compareBy<Pair<Set<String>, Set<String>>>(
    { it.first.size },
    { it.first.sorted().joinToString(",") },
    { it.second.size },
    { it.second.sorted().joinToString(",") })
  pairs.sortWith(pairOrder)

  fun id(set: Set<String>) = if (set.isEmpty()) "∅" else set.sorted().joinToString(",")
  fun nt(S: Set<String>, M: Set<String>) = "t:${id(S)}/m:${id(M)}"

  val lines = mutableListOf<String>()
  lines += "START -> ${nt(emptySet(), emptySet())}"

  for ((S, M) in pairs) {
    val alts = linkedSetOf<String>()

    // Application: partition obligations M into M1 ⊎ M2
    for (M1 in powerSet(M.toList())) {
      val M2 = M - M1
      alts += "( ${nt(S, M1)} ${nt(S, M2)} )"
    }

    // Lambda (no shadowing): introduce x fresh, add usage obligation for x
    for (x in base) if (x !in S) {
      val S2 = S + x
      val M2 = M + x
      alts += "λ $x . ${nt(S2, M2)}"
    }

    // Variables: allowed when obligations are either {} or {x}
    if (M.isEmpty()) for (x in S) alts += x
    if (M.size == 1) {
      val x = M.first()
      if (x in S) alts += x
    }

    lines += "${nt(S, M)} -> " + alts.joinToString(" | ")
  }

  return lines.joinToString("\n")
}

/**
 * Nonterminal t[S,U] is encoded as: t:<S>/u:<U>
 *   S = in-scope names; U ⊆ S = names that must still be used exactly once.
 * Start symbol: t:empty/u:empty (closedness).
 * Assumption: no shadowing (λ x requires x ∉ S).
 */
fun generateLinearLambdaCFG(names: Set<String>): String {
  require(names.isNotEmpty()) { "Provide at least one name." }
  val base = names.sorted()

  // Enumerate all (S,U) with U ⊆ S
  val pairs = mutableListOf<Pair<Set<String>, Set<String>>>()
  for (S in powerSet(base)) {
    val sList = S.toList()
    for (U in powerSet(sList)) pairs += S to U
  }
  // Stable ordering for readability
  val ord = compareBy<Pair<Set<String>, Set<String>>>(
    { it.first.size },
    { it.first.sorted().joinToString(",") },
    { it.second.size },
    { it.second.sorted().joinToString(",") })
  pairs.sortWith(ord)

  fun id(set: Set<String>) = if (set.isEmpty()) "∅" else set.sorted().joinToString(",")
  fun nt(S: Set<String>, U: Set<String>) = "t:${id(S)}/u:${id(U)}"

  val out = mutableListOf<String>()
  out += "START -> ${nt(emptySet(), emptySet())}"

  for ((S, U) in pairs) {
    val alts = linkedSetOf<String>()

    // Application: exact partition U = U1 ⊎ U2
    for (U1 in powerSet(U.toList())) {
      val U2 = U - U1
      alts += "( ${nt(S, U1)} ${nt(S, U2)} )"
    }

    // Lambda: introduce fresh x (no shadowing), obligate its single use
    for (x in base) if (x !in S) {
      val S2 = S + x
      val U2 = U + x
      alts += "λ $x . ${nt(S2, U2)}"
    }

    // Variable leaf: consume the *only* remaining obligation in this branch
    if (U.size == 1) {
      val x = U.first()
      if (x in S) alts += x
    }

    out += "${nt(S, U)} -> " + alts.joinToString(" | ")
  }

  return out.joinToString("\n")
}

private fun <T> powerSet(items: List<T>): List<Set<T>> {
  val out = ArrayList<Set<T>>(1 shl items.size)
  fun go(i: Int, acc: MutableSet<T>) {
    if (i == items.size) { out += acc.toSet(); return }
    go(i + 1, acc)
    acc += items[i]; go(i + 1, acc); acc.remove(items[i])
  }
  go(0, linkedSetOf())
  return out
}