package cppcompletion

import ai.hypergraph.kaliningraph.KBitSet
import ai.hypergraph.kaliningraph.automata.GRE
import ai.hypergraph.kaliningraph.parsing.CFG
import ai.hypergraph.kaliningraph.parsing.CFGCompletionIndex
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

private data class CachedCppSyntaxCompletion(val bounded: BoundedAcyclicCFG?, val templateTokens: Int)

private data class CppSyntaxCompletionKey(
  val prefix: List<String>,
  val allowedFirstTerminals: Set<String>?,
  val identifiers: List<String>
)

/** Worker-confined LRU. Generic prefix projection lets equivalent requests share a forest. */
private val cppSyntaxCompletionCache =
  linkedMapOf<CppSyntaxCompletionKey, CachedCppSyntaxCompletion>()

internal fun cppSingleStatementSyntaxRecognizes(tokens: List<CppToken>): Boolean =
  projectCppCompletionTokens(tokens, CppProjectionMode.SYNTAX).matches(cppSingleStatementSyntax)

/**
 * Returns every shortest syntactic continuation of [prefix]. When [tokenPrefix] is present, the
 * first emitted grammar terminal must have a concrete spelling that starts with it.
 * The returned forest emits that whole terminal; the caller replaces the partial source token.
 */
internal fun cppSingleStatementSyntaxCompletion(
  prefix: List<CppToken>,
  tokenPrefix: CppToken? = null,
  identifierInventory: Set<String> = emptySet()
): CppSuffixGrammar? {
  val identifiers = identifierInventory.sorted()
  val projectedPrefix = projectCppCompletionTokens(prefix, CppProjectionMode.SYNTAX)
  val allowedFirstTerminals = tokenPrefix?.let { token ->
    cppSingleStatementSyntaxIndex.terminalsWithSourcePrefix(token.text) { terminal ->
      if (terminal == CPP_SYNTAX_IDENTIFIER) identifiers.filter { it.startsWith(token.text) }
      else cppCompletionTerminalSpellings(terminal, token)
    }
  }
  val cacheKey = CppSyntaxCompletionKey(projectedPrefix, allowedFirstTerminals, identifiers)
  val cached = cppSyntaxCompletionCache.remove(cacheKey) ?: run {
    val minimumSuffixLength = cppSingleStatementSyntaxIndex.minimumSuffixLength(
      projectedPrefix,
      allowedFirstTerminals
    )
    val bounded = when (minimumSuffixLength) {
      null -> null
      0 -> setOf(START_SYMBOL to emptyList<String>()).freeze().boundedAcyclic(0)
      else -> {
        val suffixForest = checkNotNull(
          cppSingleStatementSyntaxIndex.completeShortestSuffix(
            projectedPrefix,
            minimumSuffixLength,
            allowedFirstTerminals
          )
        ) {
          "C++ syntax prefix analysis found a $minimumSuffixLength-token continuation, but its exact forest was empty"
        }
        suffixForest.toAcyclicCfg(cppSingleStatementSyntax.tmLst)
          .withIdentifierInventory(identifiers)
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
    identifierInventory = identifierInventory,
    recognizesCompleteSyntax = true
  )
}

private fun CFG.withIdentifierInventory(identifiers: List<String>): CFG = flatMapTo(linkedSetOf()) {
  (lhs, rhs) ->
  if (rhs.singleOrNull() == CPP_SYNTAX_IDENTIFIER)
    identifiers.map { lhs to listOf(encodeIdentifier(it)) }
  else listOf(lhs to rhs)
}

/** Immutable sparse indexes plus exact min-plus prefix completion for a recursive binary CFG. */
internal class CppStatementSyntaxIndex(private val grammar: CFG) {
  private val completionIndex = CFGCompletionIndex(grammar)
  private val variableCount = grammar.nonterminals.size
  private val start = grammar.bindex[START_SYMBOL]
  private val terminalMap = grammar.tmMap
  private val terminalParents = grammar.tmToVidx
  private val leftAdjacency = grammar.leftAdj

  /** Concrete grammar terminals whose source spelling can extend [sourcePrefix]. */
  fun terminalsWithSourcePrefix(sourcePrefix: String, spellings: (String) -> Iterable<String> = { listOf(it) }): Set<String> =
    terminalMap.keys.filterTo(linkedSetOf()) { terminal ->
      spellings(terminal).any { spelling -> spelling.startsWith(sourcePrefix) }
    }

  fun minimumSuffixLength(prefix: List<String>, allowedFirstTerminals: Set<String>? = null): Int? =
    completionIndex.minimumSuffixLength(prefix, allowedFirstTerminals)

  /** Sparse two-pass CYK forest whose fixed-prefix leaves emit epsilon and holes emit terminals. */
  fun completeShortestSuffix(
    prefix: List<String>,
    suffixLength: Int,
    allowedFirstTerminals: Set<String>? = null
  ): GRE? {
    if (allowedFirstTerminals?.isEmpty() == true ||
      allowedFirstTerminals != null && suffixLength == 0
    ) return null
    val template = prefix + List(suffixLength) { "_" }
    val tokenCount = template.size
    val active = Array(tokenCount + 1) {
      Array(tokenCount + 1) { KBitSet(variableCount) }
    }
    for (index in template.indices) {
      val terminal = template[index]
      val target = active[index][index + 1]
      if (index >= prefix.size) {
        if (index == prefix.size && allowedFirstTerminals != null) {
          allowedFirstTerminals.forEach { allowed ->
            val terminalIndex = terminalMap[allowed] ?: return@forEach
            terminalParents[terminalIndex].forEach(target::set)
          }
        } else {
          grammar.unitNonterminals.forEach { nonterminal ->
            target.set(grammar.bindex[nonterminal])
          }
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
    val constrainedHoleLeaves: Array<GRE?>? =
      allowedFirstTerminals?.let { arrayOfNulls(variableCount) }
    grammar.unitNonterminals.forEach { nonterminal ->
      val variable = grammar.bindex[nonterminal]
      val terminals = grammar.bimap.UNITS[nonterminal].orEmpty()
      if (terminals.isEmpty()) return@forEach
      val choices = KBitSet(grammar.tmLst.size)
      terminals.forEach { terminal -> grammar.tmMap[terminal]?.let(choices::set) }
      if (!choices.isEmpty()) holeLeaves[variable] = GRE.SET(choices)
      constrainedHoleLeaves?.let { leaves ->
        val constrained = KBitSet(grammar.tmLst.size)
        terminals.filter { terminal -> allowedFirstTerminals?.contains(terminal) == true }
          .forEach { terminal -> grammar.tmMap[terminal]?.let(constrained::set) }
        if (!constrained.isEmpty()) leaves[variable] = GRE.SET(constrained)
      }
    }
    val forest = Array(tokenCount + 1) {
      Array(tokenCount + 1) { mutableMapOf<Int, GRE>() }
    }
    for (index in template.indices) {
      val target = forest[index][index + 1]
      if (index < prefix.size) {
        for (variable in active[index][index + 1].iterator()) target[variable] = GRE.EPS()
      } else {
        for (variable in active[index][index + 1].iterator()) {
          val leaf = if (index != prefix.size || allowedFirstTerminals == null) {
            holeLeaves[variable]
          } else constrainedHoleLeaves?.get(variable)
          leaf?.let { target[variable] = it }
        }
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
