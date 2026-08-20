package buildlogic

import org.gradle.api.DefaultTask
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.CacheableTask
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import java.security.MessageDigest

private const val START_SYMBOL = "START"

/**
 * Converts the pinned grammars-v4 C++14 `statement` rule into an epsilon-free binary CFG.
 *
 * The generated Kotlin contains only packed integer tables. Parsing ANTLR EBNF and normalization
 * therefore happen during the build, never on the browser completion worker's hot path.
 */
@CacheableTask
abstract class GenerateCppStatementGrammar : DefaultTask() {
  @get:InputFile
  @get:PathSensitive(PathSensitivity.RELATIVE)
  abstract val parserGrammar: RegularFileProperty

  @get:InputFile
  @get:PathSensitive(PathSensitivity.RELATIVE)
  abstract val lexerGrammar: RegularFileProperty

  @get:Input
  abstract val expectedParserSha256: Property<String>

  @get:Input
  abstract val expectedLexerSha256: Property<String>

  @get:OutputFile
  abstract val outputFile: RegularFileProperty

  @TaskAction
  fun generate() {
    val parserBytes = parserGrammar.get().asFile.readBytes()
    val lexerBytes = lexerGrammar.get().asFile.readBytes()
    val parserHash = parserBytes.sha256()
    val lexerHash = lexerBytes.sha256()
    check(parserHash == expectedParserSha256.get()) {
      "CPP14Parser.g4 changed: expected ${expectedParserSha256.get()}, found $parserHash"
    }
    check(lexerHash == expectedLexerSha256.get()) {
      "CPP14Lexer.g4 changed: expected ${expectedLexerSha256.get()}, found $lexerHash"
    }

    val generated = CppStatementGrammarGenerator.generate(
      parserSource = parserBytes.toString(Charsets.UTF_8),
      lexerSource = lexerBytes.toString(Charsets.UTF_8),
      parserSha256 = parserHash,
      lexerSha256 = lexerHash
    )
    outputFile.get().asFile.apply {
      parentFile.mkdirs()
      writeText(generated.kotlinSource)
    }
    logger.lifecycle(
      "Generated C++14 statement CNF: ${generated.reachableParserRules} parser rules, " +
        "${generated.nonterminalCount} variables, ${generated.productionCount} productions"
    )
  }
}

internal data class GeneratedCppStatementGrammar(
  val kotlinSource: String,
  val productions: Set<Production>,
  val reachableParserRules: Int,
  val nonterminalCount: Int,
  val productionCount: Int
)

internal data class Production(val lhs: String, val rhs: List<String>)

internal object CppStatementGrammarGenerator {
  private val lexicalCategories = mapOf(
    // `final` and `override` are contextual keywords: the pinned lexer gives them dedicated token
    // kinds, but the C++ grammar must also admit their spellings wherever an identifier is legal.
    "Identifier" to listOf("@identifier", "final", "override"),
    "IntegerLiteral" to listOf("@integer"),
    "FloatingLiteral" to listOf("@floating"),
    "CharacterLiteral" to listOf("@character"),
    "StringLiteral" to listOf("@string"),
    "BooleanLiteral" to listOf("@boolean"),
    "PointerLiteral" to listOf("@nullptr"),
    // UDLs have different syntactic permissions from their ordinary literal counterparts. Keep
    // them distinct all the way into the packed grammar rather than widening ordinary-only slots.
    "UserDefinedLiteral" to listOf("@ud_integer", "@ud_floating", "@ud_character", "@ud_string"),
    "UserDefinedIntegerLiteral" to listOf("@ud_integer"),
    "UserDefinedFloatingLiteral" to listOf("@ud_floating"),
    "UserDefinedCharacterLiteral" to listOf("@ud_character"),
    "UserDefinedStringLiteral" to listOf("@ud_string")
  )

  /** Standard alternative tokens absent from this otherwise checksum-pinned lexer revision. */
  private val auditedFixedTokenAliases = mapOf(
    "And" to listOf("bitand"),
    "Or" to listOf("bitor"),
    "Tilde" to listOf("compl"),
    "Caret" to listOf("xor"),
    "AndAssign" to listOf("and_eq"),
    "OrAssign" to listOf("or_eq"),
    "XorAssign" to listOf("xor_eq"),
    "NotEqual" to listOf("not_eq")
  )

  fun generate(
    parserSource: String,
    lexerSource: String,
    parserSha256: String = "test",
    lexerSha256: String = "test"
  ): GeneratedCppStatementGrammar {
    val parserRules = AntlrParserGrammar(parserSource).rules()
    check("statement" in parserRules) { "CPP14Parser.g4 has no statement rule" }
    val reachable = reachableRules(parserRules, "statement")
    val lexer = AntlrLexerVocabulary(
      source = lexerSource,
      categories = lexicalCategories,
      fixedTokenAliases = auditedFixedTokenAliases
    )
    val builder = EbnfCfgBuilder(parserRules.filterKeys(reachable::contains), lexer)
    val raw = builder.build("statement") + ModernCppSyntaxOverlay.productions
    val cnf = CnfNormalizer.normalize(raw, START_SYMBOL)
    validateCnf(cnf, parserRules.keys)
    return GeneratedCppStatementGrammar(
      kotlinSource = renderKotlin(cnf, parserSha256, lexerSha256, reachable.size),
      productions = cnf,
      reachableParserRules = reachable.size,
      nonterminalCount = cnf.mapTo(linkedSetOf(), Production::lhs).size,
      productionCount = cnf.size
    )
  }

  private fun reachableRules(rules: Map<String, Expr>, root: String): Set<String> {
    val reachable = linkedSetOf(root)
    val queue = mutableListOf(root)
    var next = 0
    while (next < queue.size) {
      val name = queue[next++]
      val expression = checkNotNull(rules[name]) { "Undefined parser rule '$name'" }
      expression.references().filter { it.firstOrNull()?.isLowerCase() == true }.forEach { child ->
        check(child in rules) { "Parser rule '$name' references undefined rule '$child'" }
        if (reachable.add(child)) queue += child
      }
    }
    return reachable
  }

  private fun validateCnf(grammar: Set<Production>, parserRuleNames: Set<String>) {
    val nonterminals = grammar.mapTo(linkedSetOf(), Production::lhs)
    check(START_SYMBOL in nonterminals) { "Generated grammar has no $START_SYMBOL" }
    grammar.forEach { production ->
      check(
        production.rhs.size == 1 && production.rhs.single() !in nonterminals ||
          production.rhs.size == 2 && production.rhs.all(nonterminals::contains)
      ) { "Generated grammar is not epsilon-free CNF: $production" }
      production.rhs.filterNot(nonterminals::contains).forEach { terminal ->
        check(!terminal.startsWith("CPP14_") && terminal !in parserRuleNames) {
          "Generated helper/parser variable leaked into the terminal alphabet: '$terminal'"
        }
      }
    }
  }

  private fun renderKotlin(
    grammar: Set<Production>,
    parserSha256: String,
    lexerSha256: String,
    reachableRules: Int
  ): String {
    val nonterminals = grammar.mapTo(linkedSetOf(), Production::lhs).toList()
    val terminals = grammar.asSequence().flatMap { it.rhs.asSequence() }
      .filter { it !in nonterminals }
      .distinct()
      .sorted()
      .toList()
    val symbols = nonterminals + terminals
    val index = symbols.withIndex().associate { it.value to it.index }
    val packed = grammar.flatMap { production ->
      listOf(
        index.getValue(production.lhs),
        index.getValue(production.rhs[0]),
        production.rhs.getOrNull(1)?.let(index::getValue) ?: -1
      )
    }
    return buildString {
      appendLine("// Generated by GenerateCppStatementGrammar; do not edit.")
      appendLine("package cppcompletion")
      appendLine()
      appendLine("internal object GeneratedCpp14StatementGrammar {")
      appendLine("  const val parserSha256: String = \"$parserSha256\"")
      appendLine("  const val lexerSha256: String = \"$lexerSha256\"")
      appendLine("  const val reachableParserRules: Int = $reachableRules")
      appendLine("  const val modernOverlayRevision: Int = ${ModernCppSyntaxOverlay.REVISION}")
      appendLine("  const val nonterminalCount: Int = ${nonterminals.size}")
      appendLine("  val symbols: Array<String> = arrayOf(")
      symbols.chunked(8).forEach { chunk ->
        append("    ")
        append(chunk.joinToString(", ") { "\"${it.kotlinEscaped()}\"" })
        appendLine(",")
      }
      appendLine("  )")
      appendLine("  val productions: IntArray = intArrayOf(")
      packed.chunked(18).forEach { chunk -> appendLine("    ${chunk.joinToString(", ")},") }
      appendLine("  )")
      appendLine("}")
    }
  }
}

/** Audited C++14 grammar errata and post-C++14 syntax used by the current editor corpus. */
private object ModernCppSyntaxOverlay {
  const val REVISION = 4
  private const val STRUCTURED_BINDING = "CPP_MODERN_STRUCTURED_BINDING"
  private const val STRUCTURED_RANGE = "CPP_MODERN_STRUCTURED_RANGE_DECLARATION"
  private const val STRUCTURED_PREFIX = "CPP_MODERN_STRUCTURED_BINDING_PREFIX"
  private const val IDENTIFIER_LIST = "CPP_MODERN_IDENTIFIER_LIST"
  private const val CO_RETURN = "CPP_MODERN_CO_RETURN"
  private const val CO_YIELD = "CPP_MODERN_CO_YIELD"
  private const val CO_AWAIT = "CPP_MODERN_CO_AWAIT"
  private const val DESIGNATED_LIST = "CPP_MODERN_DESIGNATED_INITIALIZER_LIST"
  private const val DESIGNATED_CLAUSE = "CPP_MODERN_DESIGNATED_INITIALIZER_CLAUSE"
  private const val ABSTRACT_ARRAY_CHAIN = "CPP_AUDITED_ABSTRACT_ARRAY_CHAIN"
  private const val ABSTRACT_ARRAY_SUFFIX = "CPP_AUDITED_ABSTRACT_ARRAY_SUFFIX"

  val productions: Set<Production> = setOf(
    // Coroutines.
    Production("jumpStatement", listOf(CO_RETURN)),
    Production(CO_RETURN, listOf("co_return", ";")),
    Production(CO_RETURN, listOf("co_return", "expression", ";")),
    Production(CO_RETURN, listOf("co_return", "bracedInitList", ";")),
    Production("assignmentExpression", listOf(CO_YIELD)),
    Production(CO_YIELD, listOf("co_yield", "assignmentExpression")),
    Production(CO_YIELD, listOf("co_yield", "bracedInitList")),
    Production("unaryExpression", listOf(CO_AWAIT)),
    Production(CO_AWAIT, listOf("co_await", "castExpression")),

    // C++17 structured bindings, including the range-for declaration form.
    Production("blockDeclaration", listOf(STRUCTURED_BINDING)),
    Production(
      STRUCTURED_BINDING,
      listOf(STRUCTURED_PREFIX, "[", IDENTIFIER_LIST, "]", "initializer", ";")
    ),
    Production("forRangeDeclaration", listOf(STRUCTURED_RANGE)),
    Production(STRUCTURED_RANGE, listOf(STRUCTURED_PREFIX, "[", IDENTIFIER_LIST, "]")),
    Production(STRUCTURED_PREFIX, listOf("declSpecifierSeq")),
    Production(STRUCTURED_PREFIX, listOf("attributeSpecifierSeq", "declSpecifierSeq")),
    Production(STRUCTURED_PREFIX, listOf("declSpecifierSeq", "refQualifier")),
    Production(
      STRUCTURED_PREFIX,
      listOf("attributeSpecifierSeq", "declSpecifierSeq", "refQualifier")
    ),
    Production(IDENTIFIER_LIST, listOf("@identifier")),
    Production(IDENTIFIER_LIST, listOf(IDENTIFIER_LIST, ",", "@identifier")),

    // C++20 core tokens and three-way comparison.
    Production("simpleTypeSpecifier", listOf("char8_t")),
    Production("declSpecifier", listOf("consteval")),
    Production("declSpecifier", listOf("constinit")),
    Production(
      "relationalExpression",
      // Maximal munch in the pinned lexer exposes the C++20 token as LessEqual, Greater.
      listOf("relationalExpression", "<=", ">", "shiftExpression")
    ),

    // C++20 designated initializer lists.
    Production("bracedInitList", listOf("{", DESIGNATED_LIST, "}")),
    Production("bracedInitList", listOf("{", DESIGNATED_LIST, ",", "}")),
    Production(DESIGNATED_LIST, listOf(DESIGNATED_CLAUSE)),
    Production(DESIGNATED_LIST, listOf(DESIGNATED_LIST, ",", DESIGNATED_CLAUSE)),
    Production(DESIGNATED_CLAUSE, listOf(".", "@identifier", "braceOrEqualInitializer")),

    // Audited C++14 correction: upstream requires a parameter group before its repeated suffixes,
    // so noPointerAbstractDeclarator cannot begin with `[ ... ]`. Restore the standard recursive
    // shape: one leading array suffix followed by any mixture of array and parameter suffixes.
    // Linking at noPointerAbstractDeclarator also restores composition through pointerAbstractDeclarator.
    Production("noPointerAbstractDeclarator", listOf(ABSTRACT_ARRAY_CHAIN)),
    Production(ABSTRACT_ARRAY_CHAIN, listOf(ABSTRACT_ARRAY_SUFFIX)),
    Production(ABSTRACT_ARRAY_CHAIN, listOf(ABSTRACT_ARRAY_CHAIN, ABSTRACT_ARRAY_SUFFIX)),
    Production(
      ABSTRACT_ARRAY_CHAIN,
      listOf(ABSTRACT_ARRAY_CHAIN, "parametersAndQualifiers")
    ),
    Production(ABSTRACT_ARRAY_SUFFIX, listOf("[", "]")),
    Production(ABSTRACT_ARRAY_SUFFIX, listOf("[", "constantExpression", "]")),
    Production(ABSTRACT_ARRAY_SUFFIX, listOf("[", "]", "attributeSpecifierSeq")),
    Production(
      ABSTRACT_ARRAY_SUFFIX,
      listOf("[", "constantExpression", "]", "attributeSpecifierSeq")
    )
  )
}

private sealed interface Expr {
  data object Empty : Expr
  data class Ref(val name: String) : Expr
  data class Literal(val text: String) : Expr
  data class Sequence(val parts: List<Expr>) : Expr
  data class Choice(val choices: List<Expr>) : Expr
  data class Optional(val child: Expr) : Expr
  data class ZeroOrMore(val child: Expr) : Expr
  data class OneOrMore(val child: Expr) : Expr
  data class NegatedTokens(val excluded: Set<String>) : Expr
}

private fun Expr.references(): Set<String> = when (this) {
  Expr.Empty, is Expr.Literal, is Expr.NegatedTokens -> emptySet()
  is Expr.Ref -> setOf(name)
  is Expr.Sequence -> parts.flatMapTo(linkedSetOf()) { it.references() }
  is Expr.Choice -> choices.flatMapTo(linkedSetOf()) { it.references() }
  is Expr.Optional -> child.references()
  is Expr.ZeroOrMore -> child.references()
  is Expr.OneOrMore -> child.references()
}

private enum class G4Kind {
  ID, LITERAL, ACTION, COLON, SEMI, PIPE, LPAREN, RPAREN, QUESTION, STAR, PLUS,
  TILDE, ASSIGN, OTHER, EOF
}

private data class G4Token(val kind: G4Kind, val text: String, val offset: Int)

/** Tokenizer for the ANTLR grammar metalanguage, not for C++ source. */
private class G4Tokenizer(private val source: String) {
  private var index = 0

  fun tokenize(): List<G4Token> = buildList {
    while (true) {
      val token = next()
      add(token)
      if (token.kind == G4Kind.EOF) break
    }
  }

  private fun next(): G4Token {
    skipTrivia()
    if (index == source.length) return G4Token(G4Kind.EOF, "", index)
    val start = index
    val char = source[index++]
    return when (char) {
      ':' -> G4Token(G4Kind.COLON, ":", start)
      ';' -> G4Token(G4Kind.SEMI, ";", start)
      '|' -> G4Token(G4Kind.PIPE, "|", start)
      '(' -> G4Token(G4Kind.LPAREN, "(", start)
      ')' -> G4Token(G4Kind.RPAREN, ")", start)
      '?' -> G4Token(G4Kind.QUESTION, "?", start)
      '*' -> G4Token(G4Kind.STAR, "*", start)
      '+' -> G4Token(G4Kind.PLUS, "+", start)
      '~' -> G4Token(G4Kind.TILDE, "~", start)
      '=' -> G4Token(G4Kind.ASSIGN, "=", start)
      '\'' -> G4Token(G4Kind.LITERAL, readQuoted('\'', start), start)
      '{' -> G4Token(G4Kind.ACTION, readBalancedAction(start), start)
      else -> if (char == '_' || char.isLetter()) {
        while (index < source.length && (source[index] == '_' || source[index].isLetterOrDigit())) index++
        G4Token(G4Kind.ID, source.substring(start, index), start)
      } else G4Token(G4Kind.OTHER, char.toString(), start)
    }
  }

  private fun skipTrivia() {
    while (index < source.length) when {
      source[index].isWhitespace() -> index++
      source.startsWith("//", index) -> {
        index += 2
        while (index < source.length && source[index] != '\n') index++
      }
      source.startsWith("/*", index) -> {
        val end = source.indexOf("*/", index + 2)
        require(end >= 0) { "Unterminated block comment at $index" }
        index = end + 2
      }
      else -> return
    }
  }

  private fun readQuoted(quote: Char, start: Int): String {
    val result = StringBuilder()
    while (index < source.length) {
      val char = source[index++]
      when {
        char == quote -> return result.toString()
        char == '\\' && index < source.length -> {
          val escaped = source[index++]
          result.append(when (escaped) {
            'n' -> '\n'
            'r' -> '\r'
            't' -> '\t'
            else -> escaped
          })
        }
        else -> result.append(char)
      }
    }
    error("Unterminated quoted literal at $start")
  }

  private fun readBalancedAction(start: Int): String {
    var depth = 1
    var quote: Char? = null
    var escaped = false
    while (index < source.length) {
      val char = source[index++]
      if (quote != null) {
        if (escaped) escaped = false
        else if (char == '\\') escaped = true
        else if (char == quote) quote = null
        continue
      }
      when (char) {
        '\'', '"' -> quote = char
        '{' -> depth++
        '}' -> if (--depth == 0) return source.substring(start, index)
      }
    }
    error("Unterminated action block at $start")
  }
}

private class AntlrParserGrammar(source: String) {
  private val tokens = G4Tokenizer(source).tokenize()
  private var index = 0

  fun rules(): Map<String, Expr> {
    val result = linkedMapOf<String, Expr>()
    while (peek().kind != G4Kind.EOF) {
      if (peek().kind == G4Kind.ID && peek().text.firstOrNull()?.isLowerCase() == true &&
        peek(1).kind == G4Kind.COLON
      ) {
        val name = take().text
        take(G4Kind.COLON)
        val expression = parseChoice(setOf(G4Kind.SEMI))
        take(G4Kind.SEMI)
        check(result.put(name, expression) == null) { "Duplicate parser rule '$name'" }
      } else index++
    }
    return result
  }

  private fun parseChoice(stops: Set<G4Kind>): Expr {
    val choices = mutableListOf(parseSequence(stops + G4Kind.PIPE))
    while (peek().kind == G4Kind.PIPE) {
      take()
      choices += parseSequence(stops + G4Kind.PIPE)
    }
    return if (choices.size == 1) choices.single() else Expr.Choice(choices)
  }

  private fun parseSequence(stops: Set<G4Kind>): Expr {
    val parts = mutableListOf<Expr>()
    while (peek().kind !in stops && peek().kind != G4Kind.EOF) parts += parsePostfix()
    val consumingParts = parts.filterNot { it == Expr.Empty }
    return when (consumingParts.size) {
      0 -> Expr.Empty
      1 -> consumingParts.single()
      else -> Expr.Sequence(consumingParts)
    }
  }

  private fun parsePostfix(): Expr {
    val expression = parseAtom()
    fun consumeNongreedyMarker() {
      if (peek().kind == G4Kind.QUESTION) take()
    }
    return when (peek().kind) {
      G4Kind.QUESTION -> {
        take()
        consumeNongreedyMarker()
        if (expression == Expr.Empty) Expr.Empty else Expr.Optional(expression)
      }
      G4Kind.STAR -> {
        take()
        consumeNongreedyMarker()
        if (expression == Expr.Empty) Expr.Empty else Expr.ZeroOrMore(expression)
      }
      G4Kind.PLUS -> {
        take()
        consumeNongreedyMarker()
        if (expression == Expr.Empty) Expr.Empty else Expr.OneOrMore(expression)
      }
      else -> expression
    }
  }

  private fun parseAtom(): Expr {
    val token = take()
    return when (token.kind) {
      G4Kind.ID -> if (peek().kind == G4Kind.ASSIGN) {
        take()
        parseAtom()
      } else Expr.Ref(token.text)
      G4Kind.LITERAL -> Expr.Literal(token.text)
      G4Kind.ACTION -> {
        // Embedded actions and semantic predicates consume no input. A predicate's trailing `?` is
        // part of the predicate syntax, not an EBNF optionality operator.
        if (peek().kind == G4Kind.QUESTION) take()
        Expr.Empty
      }
      G4Kind.LPAREN -> parseChoice(setOf(G4Kind.RPAREN)).also { take(G4Kind.RPAREN) }
      G4Kind.TILDE -> negatedTokens(parseAtom())
      else -> error("Unexpected ANTLR token '${token.text}' at ${token.offset}")
    }
  }

  private fun negatedTokens(expression: Expr): Expr.NegatedTokens {
    fun collect(node: Expr): Set<String> = when (node) {
      is Expr.Ref -> setOf(node.name)
      is Expr.Choice -> node.choices.flatMapTo(linkedSetOf(), ::collect)
      else -> error("ANTLR token negation must contain token references, found $node")
    }
    return Expr.NegatedTokens(collect(expression))
  }

  private fun peek(ahead: Int = 0): G4Token = tokens.getOrElse(index + ahead) { tokens.last() }
  private fun take(): G4Token = peek().also { index++ }
  private fun take(kind: G4Kind): G4Token = take().also {
    check(it.kind == kind) { "Expected $kind at ${it.offset}, found '${it.text}'" }
  }
}

private class AntlrLexerVocabulary(
  source: String,
  private val categories: Map<String, List<String>>,
  fixedTokenAliases: Map<String, List<String>> = emptyMap()
) {
  private val fixed: Map<String, List<String>> = readFixedRules(source).mapValues { (name, terminals) ->
    terminals + fixedTokenAliases[name].orEmpty()
  }
  val projectedUniverse: Set<String> = (fixed.values.flatten() + categories.values.flatten())
    .map(::projectFixedTerminal)
    .toCollection(linkedSetOf())

  fun terminals(tokenName: String): List<String> =
    (categories[tokenName] ?: fixed[tokenName]?.map(::projectFixedTerminal))
      ?.distinct()
      ?: error("Parser references lexer token '$tokenName' without a finite terminal mapping")

  private fun readFixedRules(source: String): Map<String, List<String>> {
    val tokens = G4Tokenizer(source).tokenize()
    val result = linkedMapOf<String, List<String>>()
    var index = 0
    while (index + 1 < tokens.size) {
      val name = tokens[index]
      if (name.kind != G4Kind.ID || name.text.firstOrNull()?.isUpperCase() != true ||
        tokens[index + 1].kind != G4Kind.COLON
      ) {
        index++
        continue
      }
      val fragment = tokens.getOrNull(index - 1)?.let { it.kind == G4Kind.ID && it.text == "fragment" } == true
      index += 2
      val body = mutableListOf<G4Token>()
      while (tokens[index].kind != G4Kind.SEMI && tokens[index].kind != G4Kind.EOF) body += tokens[index++]
      check(tokens[index].kind == G4Kind.SEMI) { "Unterminated lexer rule '${name.text}'" }
      index++
      if (fragment) continue
      val alternatives = mutableListOf<String>()
      var expectLiteral = true
      var finite = body.isNotEmpty()
      body.forEach { token ->
        if (expectLiteral && token.kind == G4Kind.LITERAL) {
          alternatives += token.text
          expectLiteral = false
        } else if (!expectLiteral && token.kind == G4Kind.PIPE) {
          expectLiteral = true
        } else finite = false
      }
      if (expectLiteral) finite = false
      if (finite) result[name.text] = alternatives
    }
    return result
  }

  private fun projectFixedTerminal(terminal: String): String = when (terminal) {
    "true", "false" -> "@boolean"
    "nullptr" -> "@nullptr"
    else -> terminal
  }
}

private class EbnfCfgBuilder(
  private val rules: Map<String, Expr>,
  private val lexer: AntlrLexerVocabulary
) {
  private val productions = linkedSetOf<Production>()
  private val tokenSymbols = mutableMapOf<String, String>()
  private var helperCount = 0
  private val rightShiftSymbol by lazy {
    fresh("RIGHT_SHIFT").also { symbol ->
      productions += Production(symbol, listOf(tokenSymbol("Greater"), tokenSymbol("Greater")))
      productions += Production(symbol, listOf(">>"))
    }
  }

  fun build(root: String): Set<Production> {
    productions += Production(START_SYMBOL, listOf(root))
    rules.forEach { (name, expression) -> emit(name, expression) }
    return productions
  }

  private fun emit(lhs: String, expression: Expr) {
    when (expression) {
      Expr.Empty -> productions += Production(lhs, emptyList())
      is Expr.Choice -> expression.choices.forEach { emit(lhs, it) }
      is Expr.Sequence -> productions += Production(lhs, sequenceSymbols(expression.parts))
      else -> productions += Production(lhs, listOf(symbol(expression)))
    }
  }

  private fun sequenceSymbols(parts: List<Expr>): List<String> {
    val result = mutableListOf<String>()
    var index = 0
    while (index < parts.size) {
      val left = parts[index]
      val right = parts.getOrNull(index + 1)
      // projectCppTokens collapses the explicit `Greater Greater` pairs in shiftOperator and
      // theOperator outside templates. Nested-template closers arise across other productions and
      // therefore retain their two independent `>` terminals.
      if (left is Expr.Ref && left.name == "Greater" && right is Expr.Ref && right.name == "Greater") {
        result += rightShiftSymbol
        index += 2
      } else {
        result += symbol(left)
        index++
      }
    }
    return result
  }

  private fun symbol(expression: Expr): String = when (expression) {
    Expr.Empty -> error("An empty-only EBNF node must be erased before it is referenced")
    is Expr.Ref -> when {
      expression.name.firstOrNull()?.isLowerCase() == true -> {
        check(expression.name in rules) { "Reference to unreachable or undefined parser rule '${expression.name}'" }
        expression.name
      }
      else -> tokenSymbol(expression.name)
    }
    is Expr.Literal -> fresh("LITERAL").also { productions += Production(it, listOf(expression.text)) }
    is Expr.NegatedTokens -> fresh("NOT_SET").also { helper ->
      val excluded = expression.excluded.flatMapTo(linkedSetOf(), lexer::terminals)
      (lexer.projectedUniverse - excluded).forEach { terminal ->
        productions += Production(helper, listOf(terminal))
      }
      check(productions.any { it.lhs == helper }) { "Negated lexer token set removed the entire alphabet" }
    }
    is Expr.Sequence -> fresh("SEQUENCE").also { helper -> emit(helper, expression) }
    is Expr.Choice -> fresh("CHOICE").also { helper -> emit(helper, expression) }
    is Expr.Optional -> fresh("OPTIONAL").also { helper ->
      productions += Production(helper, emptyList())
      productions += Production(helper, listOf(symbol(expression.child)))
    }
    is Expr.ZeroOrMore -> fresh("STAR").also { helper ->
      val child = symbol(expression.child)
      productions += Production(helper, emptyList())
      productions += Production(helper, listOf(helper, child))
    }
    is Expr.OneOrMore -> fresh("PLUS").also { helper ->
      val child = symbol(expression.child)
      productions += Production(helper, listOf(child))
      productions += Production(helper, listOf(helper, child))
    }
  }

  private fun tokenSymbol(tokenName: String): String = tokenSymbols.getOrPut(tokenName) {
    "CPP14_TOKEN_$tokenName".also { helper ->
      lexer.terminals(tokenName).forEach { terminal -> productions += Production(helper, listOf(terminal)) }
    }
  }

  private fun fresh(label: String): String = "CPP14_EBNF_${helperCount++}_$label"
}

private object CnfNormalizer {
  fun normalize(input: Set<Production>, start: String): Set<Production> {
    val lifted = liftTerminals(input)
    val binary = binarize(lifted)
    val declaredNonterminals = binary.mapTo(linkedSetOf(), Production::lhs)
    val epsilonFree = eliminateEpsilon(binary, start)
    val unitFree = eliminateVariableUnits(epsilonFree, declaredNonterminals)
    return prune(unitFree, start, declaredNonterminals)
  }

  private fun liftTerminals(grammar: Set<Production>): Set<Production> {
    val nonterminals = grammar.mapTo(linkedSetOf(), Production::lhs)
    val terminalSymbols = linkedMapOf<String, String>()
    val result = linkedSetOf<Production>()
    grammar.forEach { production ->
      if (production.rhs.size < 2) result += production
      else result += production.copy(rhs = production.rhs.map { symbol ->
        if (symbol in nonterminals) symbol
        else terminalSymbols.getOrPut(symbol) { "CPP14_CNF_TOKEN_${terminalSymbols.size}" }
      })
    }
    terminalSymbols.forEach { (terminal, symbol) -> result += Production(symbol, listOf(terminal)) }
    return result
  }

  private fun binarize(grammar: Set<Production>): Set<Production> {
    val result = linkedSetOf<Production>()
    val suffixes = linkedMapOf<List<String>, String>()
    var suffixCount = 0
    fun suffixSymbol(rhs: List<String>): String = suffixes.getOrPut(rhs) {
      "CPP14_CNF_SUFFIX_${suffixCount++}".also { symbol ->
        result += Production(
          symbol,
          if (rhs.size == 2) rhs else listOf(rhs.first(), suffixSymbol(rhs.drop(1)))
        )
      }
    }
    grammar.forEach { production ->
      result += when {
        production.rhs.size <= 2 -> production
        else -> production.copy(rhs = listOf(production.rhs.first(), suffixSymbol(production.rhs.drop(1))))
      }
    }
    return result
  }

  private fun eliminateEpsilon(grammar: Set<Production>, start: String): Set<Production> {
    val nonterminals = grammar.mapTo(linkedSetOf(), Production::lhs)
    val nullable = linkedSetOf<String>()
    var changed: Boolean
    do {
      changed = false
      grammar.forEach { production ->
        if (production.lhs !in nullable &&
          (production.rhs.isEmpty() || production.rhs.all(nullable::contains))
        ) changed = nullable.add(production.lhs) || changed
      }
    } while (changed)
    check(start !in nullable) { "The C++ statement rule unexpectedly accepts an empty token sequence" }

    val result = linkedSetOf<Production>()
    grammar.forEach { production -> when (production.rhs.size) {
      0 -> Unit
      1 -> result += production
      2 -> {
        val (left, right) = production.rhs
        result += production
        if (left in nullable) result += Production(production.lhs, listOf(right))
        if (right in nullable) result += Production(production.lhs, listOf(left))
      }
      else -> error("Epsilon elimination requires a binary grammar: $production")
    } }
    return result
  }

  private fun eliminateVariableUnits(
    grammar: Set<Production>,
    nonterminals: Set<String>
  ): Set<Production> {
    val byLhs = grammar.groupBy(Production::lhs)
    val unitEdges = grammar.filter { it.rhs.size == 1 && it.rhs.single() in nonterminals }
      .groupBy({ it.lhs }, { it.rhs.single() })
    val result = linkedSetOf<Production>()
    nonterminals.forEach { source ->
      val closure = linkedSetOf(source)
      val queue = mutableListOf(source)
      var next = 0
      while (next < queue.size) unitEdges[queue[next++]].orEmpty().forEach { target ->
        if (closure.add(target)) queue += target
      }
      closure.forEach { target ->
        byLhs[target].orEmpty().forEach { production ->
          if (production.rhs.size != 1 || production.rhs.single() !in nonterminals) {
            result += production.copy(lhs = source)
          }
        }
      }
    }
    return result
  }

  private fun prune(
    grammar: Set<Production>,
    start: String,
    nonterminals: Set<String>
  ): Set<Production> {
    val generating = linkedSetOf<String>()
    var changed: Boolean
    do {
      changed = false
      grammar.forEach { production ->
        if (production.lhs !in generating && production.rhs.all { symbol ->
            symbol !in nonterminals || symbol in generating
          }
        ) changed = generating.add(production.lhs) || changed
      }
    } while (changed)
    check(start in generating) { "The generated C++ statement grammar is not productive" }
    val productive = grammar.filterTo(linkedSetOf()) { production ->
      production.lhs in generating && production.rhs.all { it !in nonterminals || it in generating }
    }
    val byLhs = productive.groupBy(Production::lhs)
    val reachable = linkedSetOf(start)
    val queue = mutableListOf(start)
    var next = 0
    while (next < queue.size) byLhs[queue[next++]].orEmpty().forEach { production ->
      production.rhs.filter(nonterminals::contains).forEach { child ->
        if (reachable.add(child)) queue += child
      }
    }
    return productive.filterTo(linkedSetOf()) { it.lhs in reachable }
  }
}

private fun ByteArray.sha256(): String =
  MessageDigest.getInstance("SHA-256").digest(this).joinToString("") { "%02x".format(it) }

private fun String.kotlinEscaped(): String = buildString {
  this@kotlinEscaped.forEach { char -> append(when (char) {
    '\\' -> "\\\\"
    '"' -> "\\\""
    '\n' -> "\\n"
    '\r' -> "\\r"
    '\t' -> "\\t"
    '$' -> "\\${'$'}"
    else -> char
  }) }
}
