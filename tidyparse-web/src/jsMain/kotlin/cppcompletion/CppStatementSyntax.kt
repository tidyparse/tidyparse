package cppcompletion

import ai.hypergraph.kaliningraph.KBitSet
import ai.hypergraph.kaliningraph.automata.GRE
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.BoundedAcyclicCFG
import ai.hypergraph.kaliningraph.parsing.START_SYMBOL
import ai.hypergraph.kaliningraph.parsing.bindex
import ai.hypergraph.kaliningraph.parsing.bimap
import ai.hypergraph.kaliningraph.parsing.boundedAcyclic
import ai.hypergraph.kaliningraph.parsing.freeze
import ai.hypergraph.kaliningraph.parsing.leftAdj
import ai.hypergraph.kaliningraph.parsing.matches
import ai.hypergraph.kaliningraph.parsing.nonterminals
import ai.hypergraph.kaliningraph.parsing.tmLst
import ai.hypergraph.kaliningraph.parsing.tmMap
import ai.hypergraph.kaliningraph.parsing.tmToVidx
import ai.hypergraph.kaliningraph.parsing.unitNonterminals

private const val CPP_SYNTAX_INFINITY = Int.MAX_VALUE / 4
private const val CPP_SYNTAX_COMPLETION_CACHE_SIZE = 16

/**
 * Context-independent grammar for one physical C++ statement.
 *
 * The build derives this language from the statement-reachable portion of the checksum-pinned
 * grammars-v4 CPP14Parser, plus the explicitly audited modern-C++ overlay in the generator. It is
 * complete with respect to that declared syntax contract, not an unversioned claim about every ISO
 * C++ revision. Identifiers and literals are lexical categories, so clangd facts cannot remove a
 * structural production.
 */
private val cppSingleStatementSyntax: CFG by lazy(::generatedCppStatementSyntax)

private fun generatedCppStatementSyntax(): CFG {
  val generated = GeneratedCpp14StatementGrammar
  check(generated.parserSha256 == "628062e9f75710ba1d1436ced8bd7d9d8f2f08c31a6e962c175e06b28994ff27")
  check(generated.lexerSha256 == "739a8782e05279318dccab76bf05af1ff5e3ff9e43f1b5b0d04e14d91d4fff47")
  check(generated.reachableParserRules == 188)
  check(generated.modernOverlayRevision == 4)
  check(generated.productions.size % 3 == 0) { "Malformed packed C++ statement grammar" }
  val syntax = LinkedHashSet<Pair<String, List<String>>>(generated.productions.size / 3)
  var index = 0
  while (index < generated.productions.size) {
    val lhs = generated.symbols[generated.productions[index++]]
    val first = generated.symbols[generated.productions[index++]]
    val secondIndex = generated.productions[index++]
    syntax += lhs to if (secondIndex < 0) listOf(first)
    else listOf(first, generated.symbols[secondIndex])
  }
  check(syntax.mapTo(linkedSetOf()) { it.first }.size == generated.nonterminalCount) {
    "Packed C++ statement grammar nonterminal count drifted from its generated manifest"
  }
  return syntax.freeze()
}

private val cppSingleStatementSyntaxIndex: CppStatementSyntaxIndex by lazy {
  CppStatementSyntaxIndex(cppSingleStatementSyntax)
}

private data class CachedCppSyntaxCompletion(
  val bounded: BoundedAcyclicCFG?,
  val templateTokens: Int
)

/** Worker-confined LRU. Generic identifier projection lets equivalent prefixes share a forest. */
private val cppSyntaxCompletionCache = linkedMapOf<List<String>, CachedCppSyntaxCompletion>()

internal fun cppSingleStatementSyntaxRecognizes(tokens: List<CppToken>): Boolean =
  projectCppCompletionTokens(tokens, CppProjectionMode.SYNTAX).matches(cppSingleStatementSyntax)

/** Returns a finite CFG containing every shortest syntactic continuation of [prefix]. */
internal fun cppSingleStatementSyntaxCompletion(prefix: List<CppToken>): CppSuffixGrammar? {
  val projectedPrefix = projectCppCompletionTokens(prefix, CppProjectionMode.SYNTAX)
  val cacheKey = projectedPrefix.toList()
  val cached = cppSyntaxCompletionCache.remove(cacheKey) ?: run {
    val minimumSuffixLength = cppSingleStatementSyntaxIndex.minimumSuffixLength(projectedPrefix)
    val bounded = when (minimumSuffixLength) {
      null -> null
      0 -> setOf(START_SYMBOL to emptyList<String>()).freeze().boundedAcyclic(0)
      else -> {
        val suffixForest = checkNotNull(
          cppSingleStatementSyntaxIndex.completeShortestSuffix(projectedPrefix, minimumSuffixLength)
        ) {
          "C++ syntax prefix analysis found a $minimumSuffixLength-token continuation, but its exact forest was empty"
        }
        suffixForest.toAcyclicCfg(cppSingleStatementSyntax.tmLst)
          .boundedAcyclic(minimumSuffixLength)
      }
    }
    CachedCppSyntaxCompletion(bounded, minimumSuffixLength ?: 0)
  }
  cppSyntaxCompletionCache[cacheKey] = cached
  while (cppSyntaxCompletionCache.size > CPP_SYNTAX_COMPLETION_CACHE_SIZE)
    cppSyntaxCompletionCache.remove(cppSyntaxCompletionCache.keys.first())
  val bounded = cached.bounded ?: return null
  return CppSuffixGrammar(
    bounded = bounded,
    rawPrefix = prefix,
    projectedPrefix = projectedPrefix,
    templateTokens = cached.templateTokens,
    sourceSyntax = cppSingleStatementSyntax,
    projectionMode = CppProjectionMode.SYNTAX,
    recognizesCompleteSyntax = true
  )
}

/** Immutable sparse indexes plus exact min-plus prefix completion for a recursive binary CFG. */
internal class CppStatementSyntaxIndex(private val grammar: CFG) {
  private data class BinaryRule(val parent: Int, val left: Int, val right: Int)
  private data class WeightedParent(val parent: Int, val appendedLength: Int)

  init {
    require(START_SYMBOL in grammar.nonterminals) {
      "A C++ statement completion grammar must declare $START_SYMBOL"
    }
    grammar.forEach { (lhs, rhs) ->
      require(
        rhs.size == 1 && rhs.single() !in grammar.nonterminals ||
          rhs.size == 2 && rhs.all(grammar.nonterminals::contains)
      ) {
        "C++ statement completion requires epsilon-free CNF; found $lhs -> ${rhs.joinToString(" ")}"
      }
    }
  }

  private val variableCount = grammar.nonterminals.size
  private val start = grammar.bindex[START_SYMBOL]
  private val terminalMap = grammar.tmMap
  private val terminalParents = grammar.tmToVidx
  private val leftAdjacency = grammar.leftAdj
  private val binaryRules = grammar.mapNotNull { (lhs, rhs) ->
    if (rhs.size != 2) null
    else BinaryRule(grammar.bindex[lhs], grammar.bindex[rhs[0]], grammar.bindex[rhs[1]])
  }
  private val minimumWordLength: IntArray = minimumWordLengths()
  private val weightedParents: Array<MutableList<WeightedParent>> =
    Array(variableCount) { mutableListOf<WeightedParent>() }.also { parents ->
      binaryRules.forEach { rule ->
        val appendedLength = minimumWordLength[rule.right]
        if (appendedLength < CPP_SYNTAX_INFINITY)
          parents[rule.left] += WeightedParent(rule.parent, appendedLength)
      }
  }

  private fun minimumWordLengths(): IntArray {
    val result = IntArray(variableCount) { CPP_SYNTAX_INFINITY }
    grammar.forEach { (lhs, rhs) ->
      if (rhs.size == 1 && rhs[0] !in grammar.nonterminals)
        result[grammar.bindex[lhs]] = 1
    }
    var changed: Boolean
    do {
      changed = false
      binaryRules.forEach { rule ->
        val left = result[rule.left]
        val right = result[rule.right]
        if (left >= CPP_SYNTAX_INFINITY || right >= CPP_SYNTAX_INFINITY) return@forEach
        val candidate = left + right
        if (candidate < result[rule.parent]) {
          result[rule.parent] = candidate
          changed = true
        }
      }
    } while (changed)
    check(result[start] < CPP_SYNTAX_INFINITY) { "The C++ statement syntax grammar is not productive" }
    return result
  }

  /**
   * Computes min |s| such that prefix·s is in the statement language.
   *
   * For A -> B C, a cursor either lies inside B (cost[B] + minWord[C]) or after a complete B
   * (Full[B,i,k] + cost[C,k]). Same-position edges have positive weights, so Dijkstra handles
   * left-recursive grammar cycles without a token horizon.
   */
  fun minimumSuffixLength(prefix: List<String>): Int? {
    if (prefix.isEmpty()) return minimumWordLength[start]
    val size = prefix.size
    val full = Array(size + 1) { Array(size + 1) { KBitSet(variableCount) } }
    prefix.forEachIndexed { index, terminal ->
      val terminalIndex = terminalMap[terminal] ?: return null
      terminalParents[terminalIndex].forEach { parent -> full[index][index + 1].set(parent) }
    }
    for (span in 2..size) for (begin in 0..size - span) {
      val end = begin + span
      val target = full[begin][end]
      for (split in begin + 1 until end) {
        val left = full[begin][split]
        val right = full[split][end]
        if (left.isEmpty() || right.isEmpty()) continue
        for (leftVariable in left.iterator())
          (leftAdjacency[leftVariable] ?: continue).forEachIfIn(right) { _, parent ->
            target.set(parent)
          }
      }
    }

    val completion = Array(size + 1) { IntArray(variableCount) { CPP_SYNTAX_INFINITY } }
    minimumWordLength.copyInto(completion[size])
    val heap = CppSyntaxMinHeap()
    for (begin in size - 1 downTo 0) {
      val distance = completion[begin]
      for (variable in full[begin][size].iterator()) distance[variable] = 0
      for (split in begin + 1 until size) {
        for (leftVariable in full[begin][split].iterator()) {
          val adjacency = leftAdjacency[leftVariable] ?: continue
          for (edge in adjacency.other.indices) {
            val rightCost = completion[split][adjacency.other[edge]]
            if (rightCost < distance[adjacency.aIdx[edge]])
              distance[adjacency.aIdx[edge]] = rightCost
          }
        }
      }
      heap.clear()
      distance.forEachIndexed { variable, cost -> if (cost < CPP_SYNTAX_INFINITY) heap.push(variable, cost) }
      while (heap.isNotEmpty()) {
        val packed = heap.pop()
        val variable = packed.toInt()
        val cost = (packed ushr 32).toInt()
        if (cost != distance[variable]) continue
        weightedParents[variable].forEach { edge ->
          val candidate = cost + edge.appendedLength
          if (candidate < distance[edge.parent]) {
            distance[edge.parent] = candidate
            heap.push(edge.parent, candidate)
          }
        }
      }
    }
    return completion[0][start].takeIf { it < CPP_SYNTAX_INFINITY }
  }

  /** Sparse two-pass CYK forest whose fixed-prefix leaves emit epsilon and holes emit terminals. */
  fun completeShortestSuffix(prefix: List<String>, suffixLength: Int): GRE? {
    val template = prefix + List(suffixLength) { "_" }
    val tokenCount = template.size
    val active = Array(tokenCount + 1) {
      Array(tokenCount + 1) { KBitSet(variableCount) }
    }
    for (index in template.indices) {
      val terminal = template[index]
      val target = active[index][index + 1]
      if (index >= prefix.size) {
        grammar.unitNonterminals.forEach { nonterminal ->
          target.set(grammar.bindex[nonterminal])
        }
      } else {
        val terminalIndex = terminalMap[terminal] ?: return null
        terminalParents[terminalIndex].forEach(target::set)
      }
    }
    for (span in 2..tokenCount) for (begin in 0..tokenCount - span) {
      val end = begin + span
      val target = active[begin][end]
      for (split in begin + 1 until end) {
        val left = active[begin][split]
        val right = active[split][end]
        if (left.isEmpty() || right.isEmpty()) continue
        for (leftVariable in left.iterator())
          (leftAdjacency[leftVariable] ?: continue).forEachIfIn(right) { _, parent ->
            target.set(parent)
          }
      }
    }
    if (!active[0][tokenCount][start]) return null

    val holeLeaves: Array<GRE?> = arrayOfNulls(variableCount)
    grammar.unitNonterminals.forEach { nonterminal ->
      val variable = grammar.bindex[nonterminal]
      val terminals = grammar.bimap.UNITS[nonterminal].orEmpty()
      if (terminals.isEmpty()) return@forEach
      val choices = KBitSet(grammar.tmLst.size)
      terminals.forEach { terminal -> grammar.tmMap[terminal]?.let(choices::set) }
      if (!choices.isEmpty()) holeLeaves[variable] = GRE.SET(choices)
    }
    val forest = Array(tokenCount + 1) {
      Array(tokenCount + 1) { mutableMapOf<Int, GRE>() }
    }
    for (index in template.indices) {
      val target = forest[index][index + 1]
      if (index < prefix.size) {
        for (variable in active[index][index + 1].iterator()) target[variable] = GRE.EPS()
      } else {
        for (variable in active[index][index + 1].iterator())
          holeLeaves[variable]?.let { target[variable] = it }
      }
    }
    for (span in 2..tokenCount) for (begin in 0..tokenCount - span) {
      val end = begin + span
      if (active[begin][end].isEmpty()) continue
      val alternatives = mutableMapOf<Int, MutableList<GRE>>()
      for (split in begin + 1 until end) {
        val leftBits = active[begin][split]
        val rightBits = active[split][end]
        if (leftBits.isEmpty() || rightBits.isEmpty()) continue
        val leftForest = forest[begin][split]
        val rightForest = forest[split][end]
        for (leftVariable in leftBits.iterator()) {
          val left = leftForest[leftVariable] ?: continue
          (leftAdjacency[leftVariable] ?: continue).forEachIfIn(rightBits) { rightVariable, parent ->
            val right = rightForest[rightVariable] ?: return@forEachIfIn
            alternatives.getOrPut(parent) { mutableListOf() } += left * right
          }
        }
      }
      alternatives.forEach { (parent, choices) ->
        forest[begin][end][parent] = if (choices.size == 1) choices.single()
        else GRE.CUP(*choices.toTypedArray())
      }
    }
    return forest[0][tokenCount][start]
  }
}

private class CppSyntaxMinHeap {
  private var variables = IntArray(64)
  private var costs = IntArray(64)
  private var size = 0

  fun clear() { size = 0 }
  fun isNotEmpty(): Boolean = size > 0

  fun push(variable: Int, cost: Int) {
    if (size == variables.size) {
      variables = variables.copyOf(size * 2)
      costs = costs.copyOf(size * 2)
    }
    var index = size++
    while (index > 0) {
      val parent = (index - 1) / 2
      if (costs[parent] <= cost) break
      variables[index] = variables[parent]
      costs[index] = costs[parent]
      index = parent
    }
    variables[index] = variable
    costs[index] = cost
  }

  /** High 32 bits are the cost; low 32 bits are the variable. */
  fun pop(): Long {
    val variable = variables[0]
    val cost = costs[0]
    val lastIndex = --size
    if (lastIndex > 0) {
      val lastVariable = variables[lastIndex]
      val lastCost = costs[lastIndex]
      var index = 0
      while (true) {
        val left = index * 2 + 1
        if (left >= size) break
        val right = left + 1
        val child = if (right < size && costs[right] < costs[left]) right else left
        if (costs[child] >= lastCost) break
        variables[index] = variables[child]
        costs[index] = costs[child]
        index = child
      }
      variables[index] = lastVariable
      costs[index] = lastCost
    }
    return (cost.toLong() shl 32) or (variable.toLong() and 0xffffffffL)
  }
}

internal fun GRE.toAcyclicCfg(terminals: List<String>): CFG {
  val productions = linkedSetOf<Pair<String, List<String>>>()
  val symbols = mutableMapOf<GRE, String>()
  fun visit(node: GRE): String {
    symbols[node]?.let { return it }
    val symbol = "CPP_SYNTAX_GRE_${symbols.size}"
    symbols[node] = symbol
    when (node) {
      is GRE.EPS -> productions += symbol to emptyList()
      is GRE.SET -> node.s.toList().forEach { terminal ->
        productions += symbol to listOf(terminals[terminal])
      }
      is GRE.CUP -> node.args.forEach { child -> productions += symbol to listOf(visit(child)) }
      is GRE.CAT -> productions += symbol to listOf(visit(node.l), visit(node.r))
    }
    return symbol
  }
  productions += START_SYMBOL to listOf(visit(this))
  return finiteAcyclicCnf(productions)
}
