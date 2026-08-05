import ai.hypergraph.tidyparse.lexCppTokenSpans
import cppcompletion.CppCompletionContext
import cppcompletion.CppCompletionQuery
import cppcompletion.CppConversion
import cppcompletion.CppParameter
import cppcompletion.CppReceiver
import cppcompletion.CppReference
import cppcompletion.CppSignature
import cppcompletion.CppToken
import cppcompletion.CppTokenKind
import cppcompletion.CppTypeMembers
import cppcompletion.CPP_MAX_INTERACTIVE_COMPLETIONS
import cppcompletion.cppLines

/** Immutable, editor-independent input for one full-statement completion request. */
data class CppEditorStatementSnapshot(
  val line: Int,
  val character: Int,
  val statementStartCharacter: Int,
  val prefixText: String,
  val semanticPrefixText: String,
  val tokens: List<CppToken>,
  val replacementEndCharacter: Int,
  val cacheKey: String,
  val seed: Int
) {
  /** The lexer token touching the caret, shortened to the typed fragment when necessary. */
  val activeFragment: CppToken?
    get() = tokens.lastOrNull()?.takeIf { it.end == prefixText.length }

  /** Fully committed tokens before [activeFragment]. */
  val stableTokens: List<CppToken>
    get() = if (activeFragment == null) tokens else tokens.dropLast(1)

  /** Exact source preceding [activeFragment], including the original statement indentation. */
  val stablePrefixText: String
    get() = prefixText.substring(0, activeFragment?.start ?: prefixText.length)
}

fun CppEditorStatementSnapshot.completionQuery(
  identifiers: Set<String>,
  limit: Int = CPP_MAX_INTERACTIVE_COMPLETIONS,
  seed: Int = this.seed
) = CppCompletionQuery(stableTokens, stablePrefixText, identifiers, activeFragment, limit, seed)

/** One clangd completion response together with the cursor that produced it. */
data class CppClangdCompletionGroup(
  val result: dynamic,
  val receiverMember: Boolean = false,
  val receiverOperator: String? = null
)

/** Raw browser-available semantic facts for one completion cursor. */
data class CppCompletionSemanticFacts(
  val completionGroups: List<CppClangdCompletionGroup> = emptyList(),
  val signatures: dynamic = null,
  val hover: dynamic = null,
  val diagnostics: dynamic = null,
  /** A compact DTO produced by the clangd worker, or a raw AST in tests/legacy callers. */
  val ast: dynamic = null
)

private data class CppPhysicalLine(
  val start: Int,
  val contentEnd: Int,
  val text: String
)

private data class CppCommentSpan(val start: Int, val end: Int, val inclusiveEnd: Boolean)

/**
 * Extracts the current physical statement prefix using 0-based LSP coordinates.
 *
 * The hot path lexes only the selected line. A token intersected by the caret retains both its
 * typed fragment and complete lexer spelling, so grammar filtering and whole-statement replacement
 * work for identifiers, literals and maximal-munch operators alike. Prefix length is not capped.
 */
fun cppEditorStatementSnapshot(source: String, line: Int, character: Int): CppEditorStatementSnapshot? {
  if (line < 0 || character < 0) return null
  val physical = cppPhysicalLine(source, line) ?: return null
  if (character > physical.text.length || cppIsPreprocessorLine(source, line)) return null

  val comments = cppCommentSpans(source, physical.start, physical.contentEnd)
  val caretOffset = physical.start + character
  if (comments.any { span ->
      caretOffset >= span.start &&
        (caretOffset < span.end || span.inclusiveEnd && caretOffset == span.end)
    }) return null

  val lineTokens = cppLines(physical.text).single().tokens
  val intersectedToken = lineTokens.singleOrNull { token ->
    token.start < character && character < token.end
  }
  val statementStartCharacter = cppStatementStartCharacter(lineTokens, character)
  val completePrefixTokens = lineTokens.filter { token ->
    token.start >= statementStartCharacter && token.end <= character
  }
  val prefixTokens = completePrefixTokens.map { token ->
    token.copy(
      start = token.start - statementStartCharacter,
      end = token.end - statementStartCharacter
    )
  }.toMutableList()
  if (intersectedToken != null && intersectedToken.start >= statementStartCharacter) {
    val fragmentText = physical.text.substring(intersectedToken.start, character)
    prefixTokens += intersectedToken.copy(
      text = fragmentText,
      start = intersectedToken.start - statementStartCharacter,
      end = character - statementStartCharacter,
      completeText = intersectedToken.text
    )
  }

  val replacementEndCharacter = cppStatementReplacementEndCharacter(
    lineText = physical.text,
    lineStartOffset = physical.start,
    lineTokens = lineTokens,
    comments = comments,
    statementStartCharacter = statementStartCharacter,
    character = character
  )
  val prefixText = physical.text.substring(statementStartCharacter, character)
  val activeFragment = prefixTokens.lastOrNull()?.takeIf { it.end == prefixText.length }
  val semanticPrefixText = when {
    activeFragment?.isCppWordFragment() == true ->
      prefixText.substring(0, activeFragment.start)
    intersectedToken != null ->
      physical.text.substring(statementStartCharacter, intersectedToken.end)
    else -> prefixText
  }
  val identity = "$line:$statementStartCharacter:$character:$replacementEndCharacter:" +
    "${prefixText.hashCode()}:${prefixTokens.hashCode()}"
  return CppEditorStatementSnapshot(
    line = line,
    character = character,
    statementStartCharacter = statementStartCharacter,
    prefixText = prefixText,
    semanticPrefixText = semanticPrefixText,
    tokens = prefixTokens,
    replacementEndCharacter = replacementEndCharacter.coerceAtLeast(character),
    cacheKey = identity,
    seed = identity.hashCode()
  )
}

private fun CppTokenKind.isCppLiteral(): Boolean = when (this) {
  CppTokenKind.INTEGER,
  CppTokenKind.FLOATING,
  CppTokenKind.CHARACTER,
  CppTokenKind.STRING,
  CppTokenKind.USER_DEFINED_INTEGER,
  CppTokenKind.USER_DEFINED_FLOATING,
  CppTokenKind.USER_DEFINED_CHARACTER,
  CppTokenKind.USER_DEFINED_STRING -> true
  else -> false
}

private fun CppToken.isCppWordFragment(): Boolean =
  !kind.isCppLiteral() && text.matches(CPP_IDENTIFIER_REGEX)

private data class CppStatementScope(
  var startCharacter: Int,
  val roundBaseline: Int,
  val squareBaseline: Int,
  val expressionBraceBaseline: Int
)

private data class CppBraceScope(
  val statementScope: Boolean,
  val advancesParentStatement: Boolean = statementScope
)

/** Finds the lexical statement boundary at the cursor's current compound-statement depth. */
private fun cppStatementStartCharacter(tokens: List<CppToken>, character: Int): Int {
  val scopes = mutableListOf(CppStatementScope(0, 0, 0, 0))
  val braces = mutableListOf<CppBraceScope>()
  var round = 0
  var square = 0
  var expressionBraces = 0
  tokens.forEachIndexed { index, token ->
    if (token.end > character) return@forEachIndexed
    when (token.text) {
      "(" -> round++
      ")" -> round = maxOf(0, round - 1)
      "[" -> square++
      "]" -> square = maxOf(0, square - 1)
      "{" -> {
        val braceScope = cppBraceScope(tokens, index)
        braces += braceScope
        if (braceScope.statementScope) {
          scopes += CppStatementScope(token.end, round, square, expressionBraces)
        } else expressionBraces++
      }
      "}" -> if (braces.isNotEmpty()) {
        val braceScope = braces.removeAt(braces.lastIndex)
        if (braceScope.statementScope) {
          if (scopes.size > 1) scopes.removeAt(scopes.lastIndex)
          if (braceScope.advancesParentStatement) scopes.last().startCharacter = token.end
        } else expressionBraces = maxOf(0, expressionBraces - 1)
      }
      ";" -> scopes.last().let { scope ->
        if (round == scope.roundBaseline && square == scope.squareBaseline &&
          expressionBraces == scope.expressionBraceBaseline
        ) scope.startCharacter = token.end
      }
    }
  }
  return scopes.last().startCharacter.coerceIn(0, character)
}

/** Distinguishes compound/lambda bodies from braced initializers for statement segmentation. */
private fun cppBraceScope(tokens: List<CppToken>, braceIndex: Int): CppBraceScope {
  val previous = tokens.getOrNull(braceIndex - 1)?.text ?: return CppBraceScope(true)
  val boundary = (braceIndex - 1 downTo 0).firstOrNull { index ->
    tokens[index].text in setOf(";", "{", "}")
  } ?: -1
  val clause = tokens.subList(boundary + 1, braceIndex).map(CppToken::text)
  val lambdaBody = previous == "]" ||
    ("[" in clause && "]" in clause && "operator" !in clause)
  return when {
    lambdaBody -> CppBraceScope(statementScope = true, advancesParentStatement = false)
    previous in setOf(")", ";", "{", "}", "else", "do", "try", "catch") ->
      CppBraceScope(true)
    ")" in clause -> CppBraceScope(true)
    clause.any { it in setOf("class", "struct", "union", "namespace", "enum") } ->
      CppBraceScope(true)
    clause.firstOrNull() in setOf("if", "for", "while", "switch") -> CppBraceScope(true)
    else -> CppBraceScope(false)
  }
}

/**
 * Replaces only the selected statement's remaining text. The next statement, a surrounding block
 * close, and trailing comments stay outside Monaco's edit range.
 */
private fun cppStatementReplacementEndCharacter(
  lineText: String,
  lineStartOffset: Int,
  lineTokens: List<CppToken>,
  comments: List<CppCommentSpan>,
  statementStartCharacter: Int,
  character: Int
): Int {
  val trailingComment = comments.asSequence().filter { span ->
    val localStart = span.start - lineStartOffset
    val localEnd = span.end - lineStartOffset
    localStart >= character && lineTokens.none { token -> token.start >= localEnd }
  }.map { span ->
    var start = span.start - lineStartOffset
    while (start > character && lineText[start - 1].isWhitespace()) start--
    start
  }.minOrNull() ?: lineText.length

  var round = 0
  var square = 0
  var brace = 0
  fun consume(token: CppToken) {
    when (token.text) {
      "(" -> round++
      ")" -> round = maxOf(0, round - 1)
      "[" -> square++
      "]" -> square = maxOf(0, square - 1)
      "{" -> brace++
      "}" -> brace = maxOf(0, brace - 1)
    }
  }
  lineTokens.asSequence()
    .filter { it.start >= statementStartCharacter && it.end <= character }
    .forEach(::consume)
  lineTokens.asSequence().filter { it.start >= character }.forEach { token ->
    if (token.text == "}" && brace == 0)
      return minOf(token.start, trailingComment).coerceAtLeast(character)
    if (token.text == ";" && round == 0 && square == 0 && brace == 0)
      return minOf(token.end, trailingComment).coerceAtLeast(character)
    consume(token)
  }
  return trailingComment.coerceIn(character, lineText.length)
}

private fun cppPhysicalLine(source: String, requestedLine: Int): CppPhysicalLine? {
  var line = 0
  var start = 0
  while (true) {
    val newline = source.indexOf('\n', start)
    val rawEnd = if (newline < 0) source.length else newline
    val contentEnd = if (rawEnd > start && source[rawEnd - 1] == '\r') rawEnd - 1 else rawEnd
    if (line == requestedLine)
      return CppPhysicalLine(start, contentEnd, source.substring(start, contentEnd))
    if (newline < 0) return null
    start = newline + 1
    line++
  }
}

/** Includes continuation lines belonging to a backslash-continued preprocessor directive. */
private fun cppIsPreprocessorLine(source: String, requestedLine: Int): Boolean {
  var line = 0
  var start = 0
  var continuing = false
  while (true) {
    val newline = source.indexOf('\n', start)
    val rawEnd = if (newline < 0) source.length else newline
    val contentEnd = if (rawEnd > start && source[rawEnd - 1] == '\r') rawEnd - 1 else rawEnd
    val text = source.substring(start, contentEnd)
    val directive = continuing || text.trimStart().startsWith('#')
    if (line == requestedLine) return directive
    continuing = directive && text.trimEnd().let { trimmed ->
      var slashes = 0
      var index = trimmed.lastIndex
      while (index >= 0 && trimmed[index--] == '\\') slashes++
      slashes % 2 == 1
    }
    if (newline < 0) return false
    start = newline + 1
    line++
  }
}

/** Finds only comment spans intersecting the requested line while respecting quoted literals. */
private fun cppCommentSpans(source: String, lineStart: Int, lineEnd: Int): List<CppCommentSpan> {
  val result = mutableListOf<CppCommentSpan>()
  var index = 0
  while (index < lineEnd) {
    when {
      source.startsWith("//", index) -> {
        val end = source.indexOf('\n', index + 2).let { if (it < 0) source.length else it }
        if (end >= lineStart && index <= lineEnd)
          result += CppCommentSpan(maxOf(index, lineStart), minOf(end, lineEnd), true)
        index = end
      }
      source.startsWith("/*", index) -> {
        val close = source.indexOf("*/", index + 2)
        val end = if (close < 0) source.length else close + 2
        if (end >= lineStart && index <= lineEnd)
          result += CppCommentSpan(
            maxOf(index, lineStart),
            minOf(end, lineEnd),
            inclusiveEnd = close < 0 || end > lineEnd
          )
        index = end
      }
      else -> {
        val raw = cppRawStringAt(source, index)
        if (raw != null) {
          index = raw
        } else if (source[index] == '"' || source[index] == '\'') {
          val quote = source[index++]
          while (index < source.length) {
            when {
              source[index] == '\\' -> index = (index + 2).coerceAtMost(source.length)
              source[index] == quote -> { index++; break }
              else -> index++
            }
          }
        } else index++
      }
    }
  }
  return result
}

/** Returns the end of a C++ raw string beginning at [index], or null when none begins there. */
private fun cppRawStringAt(source: String, index: Int): Int? {
  val prefixes = arrayOf("u8R\"", "uR\"", "UR\"", "LR\"", "R\"")
  val prefix = prefixes.firstOrNull { source.startsWith(it, index) } ?: return null
  val delimiterStart = index + prefix.length
  val open = source.indexOf('(', delimiterStart)
  if (open < 0 || open - delimiterStart > 16) return null
  val delimiter = source.substring(delimiterStart, open)
  if (delimiter.any { it.isWhitespace() || it == '\\' || it == '(' || it == ')' }) return null
  val terminator = ")$delimiter\""
  val close = source.indexOf(terminator, open + 1)
  return if (close < 0) source.length else close + terminator.length
}

/**
 * Builds a structured-clone-safe context DTO from the source and optional clangd facts.
 * Kotlin collections and data classes never cross the worker boundary.
 */
fun cppCompletionContextDto(
  source: String,
  completionGroups: List<CppClangdCompletionGroup> = emptyList(),
  signatures: dynamic = null,
  hover: dynamic = null,
  diagnostics: dynamic = null,
  ast: dynamic = null,
  snapshot: CppEditorStatementSnapshot? = null,
  lexicalFacts: CppLexicalSourceFacts? = null
): dynamic {
  val lexical = lexicalFacts ?: cppLexicalSourceFacts(source)
  val completionReferences = cppDistinctReferences(completionGroups.flatMap { group ->
    cppCompletionItems(group.result).mapNotNull { item ->
      cppReferenceFromCompletion(item, group.receiverMember)
    }
  }).filterNot { reference ->
    // A method is never a free function. clangd may report a partially typed member at an
    // ordinary completion cursor as one collapsed Method item (for example `[4 overloads]` with
    // an ellipsis signature). Without receiver/implicit-this evidence that item is not a usable
    // callable fact; the scoped AST and receiver member tables remain authoritative.
    reference.kind == "method" && !reference.receiverMember
  }
  val astContext = cppContextFromNormalizedAst(ast)
  val baseCompletions = completionReferences.filterNot(CppReference::receiverMember)
  val values = cppDistinctReferences(
    astContext.values + baseCompletions.filter { reference ->
      reference.kind in CPP_VALUE_REFERENCE_KINDS &&
        reference.name.substringAfterLast("::") in lexical.identifiers
    }
  )
  val types = cppDistinctReferences(
    lexical.types + astContext.types + baseCompletions.filter { it.kind in CPP_TYPE_REFERENCE_KINDS }
  )
  val functions = cppDistinctReferences(
    astContext.functions + baseCompletions.filter { it.kind in CPP_FUNCTION_REFERENCE_KINDS }
  )
  val hoverFacts = cppHoverFacts(hover)
  val normalizedSignatures = cppSignatures(signatures, hoverFacts)
  val expectedTypes = normalizedSignatures.mapNotNullTo(linkedSetOf()) { signature ->
    signature.activeParameter?.let(signature.parameters::getOrNull)?.type?.takeIf(String::isNotBlank)
  }
  val memberGroups = completionGroups.filter(CppClangdCompletionGroup::receiverMember)
  val receiverMembers = completionReferences.filter(CppReference::receiverMember)
  val receiverOperator = memberGroups.firstNotNullOfOrNull(CppClangdCompletionGroup::receiverOperator)
    ?: cppReceiverOperator(snapshot?.semanticPrefixText)
  val receiverExpression = cppReceiverExpression(snapshot?.semanticPrefixText, receiverOperator)
  val receiverType = hoverFacts.type
    ?: values.firstOrNull { it.name == receiverExpression }?.type
  val ownedMembers = receiverMembers.map { member ->
    if (member.ownerType != null || receiverType.isNullOrBlank()) member
    else member.copy(ownerType = receiverType)
  }
  val receiver = receiverOperator?.let { operator ->
    CppReceiver(operator, receiverExpression.orEmpty(), receiverType, ownedMembers)
  }
  val memberTables = cppDistinctTypeMembers(
    astContext.membersByType + listOfNotNull(
      receiverType?.takeIf { ownedMembers.isNotEmpty() }?.let { CppTypeMembers(it, ownedMembers) }
    )
  )
  val unresolved = cppUnresolvedIdentifiers(source, diagnostics, snapshot)
  val identifiers = linkedSetOf<String>().apply {
    addAll(lexical.identifiers)
    addAll(lexical.typeNames)
    (completionReferences + values + types + functions).forEach { reference ->
      addAll(CPP_IDENTIFIER_REGEX.findAll(reference.name).map(MatchResult::value))
    }
  }
  return cppCompletionContextToDto(CppCompletionContext(
    identifiers = identifiers,
    sourceIdentifiers = lexical.identifiers,
    headers = lexical.headers,
    typeNames = lexical.typeNames,
    values = values,
    types = types,
    functions = functions,
    completions = completionReferences,
    signatures = normalizedSignatures,
    expectedTypes = expectedTypes,
    receiver = receiver,
    membersByType = memberTables,
    conversions = astContext.conversions,
    unresolvedIdentifiers = unresolved,
    requiredIdentifier = unresolved.firstOrNull(),
    requiredTypes = astContext.requiredTypes,
    probedRequiredTypes = astContext.probedRequiredTypes,
    defaultConstructibleTypes = astContext.defaultConstructibleTypes,
    enclosingReturnType = astContext.enclosingReturnType,
    enclosingClassType = astContext.enclosingClassType,
    thisType = astContext.thisType,
    mutableFields = astContext.mutableFields
  ))
}

/** Rehydrates the worker-local immutable context from a plain structured clone. */
fun cppCompletionContextFromDto(dto: dynamic): CppCompletionContext {
  if (!cppDefined(dto)) return CppCompletionContext(emptySet())
  return CppCompletionContext(
    identifiers = cppStringSet(dto.identifiers),
    sourceIdentifiers = cppStringSet(dto.sourceIdentifiers),
    headers = cppStringSet(dto.headers),
    typeNames = cppStringSet(dto.typeNames),
    values = cppDynamicList(dto.values).mapNotNull(::cppReferenceFromDto),
    types = cppDynamicList(dto.types).mapNotNull(::cppReferenceFromDto),
    functions = cppDynamicList(dto.functions).mapNotNull(::cppReferenceFromDto),
    completions = cppDynamicList(dto.completions).mapNotNull(::cppReferenceFromDto),
    signatures = cppDynamicList(dto.signatures).mapNotNull(::cppSignatureFromDto),
    expectedTypes = cppStringSet(dto.expectedTypes),
    receiver = cppReceiverFromDto(dto.receiver),
    membersByType = cppDynamicList(dto.membersByType).mapNotNull(::cppTypeMembersFromDto),
    conversions = cppDynamicList(dto.conversions).mapNotNull(::cppConversionFromDto),
    unresolvedIdentifiers = cppStringSet(dto.unresolvedIdentifiers),
    requiredIdentifier = dto.requiredIdentifier as? String,
    requiredTypes = cppStringSet(dto.requiredTypes),
    probedRequiredTypes = cppStringSet(dto.probedRequiredTypes),
    defaultConstructibleTypes = cppStringSet(dto.defaultConstructibleTypes),
    enclosingReturnType = dto.enclosingReturnType as? String,
    enclosingClassType = dto.enclosingClassType as? String,
    thisType = dto.thisType as? String,
    mutableFields = cppStringSet(dto.mutableFields)
  )
}

data class CppLexicalSourceFacts(
  val identifiers: Set<String>,
  val headers: Set<String>,
  val typeNames: Set<String>,
  val types: List<CppReference>
)

private val CPP_IDENTIFIER_REGEX = Regex("[A-Za-z_][A-Za-z_0-9]*")
private val CPP_BUILTIN_TYPE_WORDS = setOf(
  "void", "bool", "char", "wchar_t", "char8_t", "char16_t", "char32_t", "short",
  "int", "long", "float", "double", "signed", "unsigned", "auto"
)
private val CPP_VALUE_REFERENCE_KINDS = setOf("constant", "enumMember", "field", "property", "value", "variable")
private val CPP_FUNCTION_REFERENCE_KINDS = setOf("constructor", "function", "method", "operator")
private val CPP_TYPE_REFERENCE_KINDS = setOf("class", "concept", "enum", "struct", "typeAlias", "typeParameter")

fun cppLexicalSourceFacts(source: String): CppLexicalSourceFacts {
  val directiveRanges = cppDirectiveRanges(source)
  var directiveIndex = 0
  val tokens = lexCppTokenSpans(source).filterNot { token ->
    while (directiveRanges.getOrNull(directiveIndex)?.second?.let { it <= token.startIndex } == true)
      directiveIndex++
    directiveRanges.getOrNull(directiveIndex)?.let { range ->
      token.startIndex >= range.first && token.startIndex < range.second
    } ?: false
  }
  val identifiers = tokens.filter { it.type == "Identifier" }
    .mapTo(linkedSetOf()) { it.text }
  val typeNames = linkedSetOf<String>()
  val declaredTypes = mutableListOf<CppReference>()
  tokens.forEachIndexed { index, token ->
    val candidate = when (token.text) {
      "class", "struct", "union", "namespace" -> tokens.getOrNull(index + 1)
      "enum" -> tokens.getOrNull(index + 1)?.let { next ->
        if (next.text == "class" || next.text == "struct") tokens.getOrNull(index + 2) else next
      }
      "using" -> tokens.getOrNull(index + 1)?.takeIf { next -> tokens.getOrNull(index + 2)?.text == "=" }
      else -> null
    }
    if (candidate?.type == "Identifier") {
      typeNames += candidate.text
      val kind = when (token.text) {
        "struct" -> "struct"
        "class", "union" -> "class"
        "enum" -> "enum"
        "using" -> "typeAlias"
        else -> null
      }
      if (kind != null && kind != "typeAlias")
        declaredTypes += CppReference(candidate.text, kind = kind, source = "source")
    }
  }
  var declarationStart = 0
  tokens.forEachIndexed { index, token ->
    if (token.text == ";") {
      val declaration = tokens.subList(declarationStart, index)
      if (declaration.firstOrNull()?.text == "typedef") {
        declaration.lastOrNull { it.type == "Identifier" }?.text?.let { name ->
          typeNames += name
          declaredTypes += CppReference(name, kind = "typeAlias", source = "source")
        }
      }
      declarationStart = index + 1
    }
  }
  tokens.filter { it.text in CPP_BUILTIN_TYPE_WORDS }.mapTo(typeNames) { it.text }

  val headers = Regex(
    "^\\s*#\\s*include(?:_next)?\\s*[<\"]([^>\"]+)[>\"]",
    RegexOption.MULTILINE
  )
    .findAll(source).mapTo(linkedSetOf()) { it.groupValues[1] }
  val aliases = Regex("\\busing\\s+([A-Za-z_][A-Za-z_0-9]*)\\s*=\\s*([^;{}\\r\\n]+)\\s*;")
    .findAll(source)
    .mapNotNull { match ->
      val name = match.groupValues[1]
      if (name !in typeNames) null else match.groupValues[2].trim().let { target ->
        CppReference(name, type = target, kind = "typeAlias", detail = target, source = "source")
      }
    }.toList()
  return CppLexicalSourceFacts(
    identifiers,
    headers,
    typeNames,
    cppDistinctReferences(declaredTypes.filterNot { declaration ->
      declaration.kind == "typeAlias" && aliases.any { it.name == declaration.name }
    } + aliases)
  )
}

private fun cppDirectiveRanges(source: String): List<Pair<Int, Int>> {
  val result = mutableListOf<Pair<Int, Int>>()
  var start = 0
  while (start <= source.length) {
    val newline = source.indexOf('\n', start)
    val end = if (newline < 0) source.length else newline
    if (source.substring(start, end).trimStart().startsWith('#')) result += start to end
    if (newline < 0) break
    start = newline + 1
  }
  return result
}

private fun cppCompletionItems(result: dynamic): List<dynamic> = when {
  !cppDefined(result) -> emptyList()
  cppIsArray(result) -> cppDynamicList(result)
  else -> cppDynamicList(result.items)
}

private fun cppReferenceFromCompletion(item: dynamic, receiverMember: Boolean): CppReference? {
  if (!cppDefined(item)) return null
  val label = (if (item.label is String) item.label else item.label?.label) as? String
  val name = cppCompletionSemanticName(item, label) ?: return null
  val detail = (item.detail as? String)?.trim()?.takeIf(String::isNotEmpty)
  val kind = when (cppInt(item.kind)) {
    2 -> "method"
    3 -> "function"
    4 -> "constructor"
    5 -> "field"
    6 -> "variable"
    7 -> "class"
    8 -> "typeAlias"
    9 -> "namespace"
    10 -> "property"
    12 -> "value"
    13 -> "enum"
    20 -> "enumMember"
    21 -> "constant"
    22 -> "struct"
    24 -> "operator"
    25 -> "typeParameter"
    else -> "unknown"
  }
  if (name == "main" && kind == "function") return null
  val parameters = if (kind in CPP_FUNCTION_REFERENCE_KINDS) {
    val signatureDetail = item.labelDetails?.detail as? String
    cppParameterClause(signatureDetail ?: label ?: name)
  } else emptyList()
  return CppReference(
    name = name,
    type = detail.takeIf { kind in CPP_VALUE_REFERENCE_KINDS },
    returnType = when {
      kind == "constructor" -> name
      kind in CPP_FUNCTION_REFERENCE_KINDS -> detail
      else -> null
    },
    parameters = parameters,
    kind = kind,
    detail = detail,
    receiverMember = receiverMember,
    source = "completion"
  )
}

/** Uses display/filter names as semantic facts and treats edit/snippet text as a last resort. */
private fun cppCompletionSemanticName(item: dynamic, label: String?): String? =
  listOf<dynamic>(label, item.filterText, item.textEdit?.newText, item.insertText)
    .mapNotNull { it as? String }
    .mapNotNull(::cppNormalizeCompletionName)
    .firstOrNull()

private fun cppNormalizeCompletionName(candidate: String): String? {
  if ('\n' in candidate || '\r' in candidate) return null
  val cleaned = cppStripSnippetSyntax(candidate).trim()
  if (cleaned.isEmpty()) return null
  if (cleaned.startsWith("operator")) {
    val operatorName = when {
      cleaned.startsWith("operator()") -> "operator()"
      cleaned.startsWith("operator[]") -> "operator[]"
      else -> cleaned.substringBefore('(').substringBefore(" -> ").trim()
    }
    return operatorName.takeIf(String::isNotEmpty)
  }
  return Regex("^(?:(?:[A-Za-z_][A-Za-z_0-9]*)::)*(?:~?[A-Za-z_][A-Za-z_0-9]*)")
    .find(cleaned)?.value
}

/** Minimal TextMate-snippet decoding; enough to prevent placeholders from becoming C++ names. */
private fun cppStripSnippetSyntax(snippet: String): String = buildString {
  var index = 0
  while (index < snippet.length) {
    when {
      snippet[index] == '\\' && snippet.getOrNull(index + 1) == '$' -> {
        append('$')
        index += 2
      }
      snippet[index] == '$' && snippet.getOrNull(index + 1) == '{' -> {
        val close = snippet.indexOf('}', index + 2)
        if (close < 0) {
          index++
          continue
        }
        val placeholder = snippet.substring(index + 2, close)
        val default = placeholder.substringAfter(':', "")
          .ifEmpty {
            placeholder.substringAfter('|', "").substringBefore(',').substringBefore('|')
          }
        append(default)
        index = close + 1
      }
      snippet[index] == '$' && snippet.getOrNull(index + 1)?.isDigit() == true -> {
        index += 2
        while (snippet.getOrNull(index)?.isDigit() == true) index++
      }
      else -> append(snippet[index++])
    }
  }
}

private data class CppHoverFacts(val type: String? = null, val returnType: String? = null)

private fun cppHoverFacts(hover: dynamic): CppHoverFacts {
  if (!cppDefined(hover)) return CppHoverFacts()
  val contents = hover.contents
  val text = when {
    contents is String -> contents
    cppDefined(contents?.value) -> contents.value as? String ?: ""
    cppIsArray(contents) -> cppDynamicList(contents).joinToString("\n") { item ->
      item as? String ?: item?.value as? String ?: ""
    }
    else -> ""
  }
  val type = Regex("^Type:\\s*(.+)$", RegexOption.MULTILINE)
    .find(text)?.groupValues?.get(1)?.trim()?.trim('`')
  val returnType = Regex("^[→]\\s*(.+)$", RegexOption.MULTILINE)
    .find(text)?.groupValues?.get(1)?.trim()?.trim('`')
  return CppHoverFacts(type, returnType)
}

private fun cppSignatures(result: dynamic, hover: CppHoverFacts): List<CppSignature> {
  if (!cppDefined(result)) return emptyList()
  val raw = cppDynamicList(result.signatures)
  val activeSignature = cppInt(result.activeSignature).coerceIn(0, maxOf(0, raw.lastIndex))
  val activeParameter = cppInt(result.activeParameter)
  return raw.mapIndexedNotNull { index, signature ->
    val label = signature?.label as? String ?: return@mapIndexedNotNull null
    val parameters = cppDynamicList(signature.parameters).mapNotNull { parameter ->
      val parameterLabel = parameter?.label
      val rawLabel = when {
        parameterLabel is String -> parameterLabel
        cppIsArray(parameterLabel) -> {
          val bounds = cppDynamicList(parameterLabel)
          val start = cppInt(bounds.getOrNull(0)).coerceIn(0, label.length)
          val end = cppInt(bounds.getOrNull(1)).coerceIn(start, label.length)
          label.substring(start, end)
        }
        else -> return@mapNotNull null
      }
      cppParameterFromLabel(rawLabel)
    }
    val arrow = label.lastIndexOf("->")
    CppSignature(
      label = label,
      returnType = if (index == activeSignature) hover.returnType
        ?: label.substring((arrow + 2).coerceAtMost(label.length)).trim().takeIf { arrow >= 0 }
      else label.substring((arrow + 2).coerceAtMost(label.length)).trim().takeIf { arrow >= 0 },
      parameters = parameters,
      activeParameter = if (parameters.isEmpty()) null else activeParameter.coerceIn(0, parameters.lastIndex)
    )
  }
}

private fun cppParameterClause(text: String): List<CppParameter> {
  val open = text.indexOf('(')
  if (open < 0) return emptyList()
  var depth = 0
  for (index in open until text.length) {
    if (text[index] == '(') depth++
    if (text[index] == ')' && --depth == 0)
      return cppSplitTopLevel(text.substring(open + 1, index)).map(::cppParameterFromLabel)
  }
  return emptyList()
}

private fun cppSplitTopLevel(text: String, separator: Char = ','): List<String> {
  val result = mutableListOf<String>()
  var start = 0
  var angle = 0
  var round = 0
  var square = 0
  var brace = 0
  text.forEachIndexed { index, character ->
    when (character) {
      '<' -> angle++
      '>' -> angle = maxOf(0, angle - 1)
      '(' -> round++
      ')' -> round = maxOf(0, round - 1)
      '[' -> square++
      ']' -> square = maxOf(0, square - 1)
      '{' -> brace++
      '}' -> brace = maxOf(0, brace - 1)
      separator -> if (angle + round + square + brace == 0) {
        result += text.substring(start, index).trim()
        start = index + 1
      }
    }
  }
  result += text.substring(start).trim()
  return result.filter { it.isNotEmpty() && it != "void" }
}

private fun cppParameterFromLabel(label: String): CppParameter {
  val raw = label.trim()
  val equalParts = cppSplitTopLevel(raw, '=')
  val declaration = equalParts.firstOrNull() ?: raw
  val nameMatch = Regex("(?:^|[\\s*&])([A-Za-z_][A-Za-z_0-9]*)\\s*(\\[[^\\]]*\\])?\\s*$")
    .find(declaration)
  val name = nameMatch?.groupValues?.get(1).orEmpty()
  val nameOffset = nameMatch?.let { match -> match.range.first + match.value.lastIndexOf(name) } ?: -1
  val prefix = if (nameOffset < 0) "" else declaration.substring(0, nameOffset).trim()
  val hasSeparateName = name.isNotEmpty() && prefix.isNotEmpty()
  val arraySuffix = nameMatch?.groupValues?.getOrNull(2).orEmpty()
  return CppParameter(
    label = raw,
    name = name.takeIf { hasSeparateName }.orEmpty(),
    type = (if (hasSeparateName) prefix + arraySuffix else declaration).trim(),
    defaultValue = raw.substringAfter('=', "").trim().takeIf { '=' in raw }
  )
}

private fun cppReceiverOperator(prefix: String?): String? = prefix?.trimEnd()?.let {
  when {
    it.endsWith("->") -> "->"
    it.endsWith("::") -> "::"
    it.endsWith('.') -> "."
    else -> null
  }
}

private fun cppReceiverExpression(prefix: String?, operator: String?): String? {
  if (prefix == null || operator == null) return null
  val before = prefix.trimEnd().removeSuffix(operator).trimEnd()
  return Regex("(?:[A-Za-z_][A-Za-z_0-9]*|\\([^()]*\\)|\\[[^\\]]*\\])$")
    .find(before)?.value
}

private fun cppUnresolvedIdentifiers(
  source: String,
  diagnostics: dynamic,
  snapshot: CppEditorStatementSnapshot?
): Set<String> {
  val raw = if (cppIsArray(diagnostics)) cppDynamicList(diagnostics)
  else if (cppDefined(diagnostics)) cppDynamicList(diagnostics.diagnostics) else emptyList()
  return raw.asSequence().filter { diagnostic ->
    val code = diagnostic?.code?.toString().orEmpty()
    val message = diagnostic?.message as? String ?: ""
    (code == "undeclared_var_use" || code == "undeclared_var_use_suggest" ||
      Regex("(?:use of )?undeclared identifier", RegexOption.IGNORE_CASE).containsMatchIn(message)) &&
      (snapshot == null || cppDiagnosticBelongsToStatement(diagnostic, snapshot))
  }.mapNotNull { diagnostic ->
    cppTextAtRange(source, diagnostic?.range) ?: run {
      val message = diagnostic?.message as? String ?: return@mapNotNull null
      Regex("undeclared identifier\\s+['‘]([A-Za-z_][A-Za-z_0-9]*)['’]", RegexOption.IGNORE_CASE)
        .find(message)?.groupValues?.get(1)
    }
  }.filter(CPP_IDENTIFIER_REGEX::matches).toCollection(linkedSetOf())
}

private fun cppDiagnosticBelongsToStatement(diagnostic: dynamic, snapshot: CppEditorStatementSnapshot): Boolean {
  val start = diagnostic?.range?.start
  if (!cppDefined(start) || cppInt(start.line, -1) != snapshot.line) return false
  val character = cppInt(start.character, -1)
  return character in snapshot.statementStartCharacter..snapshot.replacementEndCharacter
}

private fun cppTextAtRange(source: String, range: dynamic): String? {
  if (!cppDefined(range?.start) || !cppDefined(range?.end)) return null
  val startLine = cppInt(range.start.line, -1)
  val endLine = cppInt(range.end.line, -1)
  if (startLine < 0 || startLine != endLine) return null
  val physical = cppPhysicalLine(source, startLine) ?: return null
  val start = cppInt(range.start.character, -1)
  val end = cppInt(range.end.character, -1)
  if (start < 0 || end < start || end > physical.text.length) return null
  return physical.text.substring(start, end).trim()
}

private fun cppDistinctReferences(references: List<CppReference>): List<CppReference> {
  val seen = linkedSetOf<String>()
  return references.filter { reference ->
    seen.add(listOf(reference.name, reference.kind, reference.type, reference.returnType,
      reference.parameters.joinToString(",") { it.type }, reference.receiverMember, reference.ownerType)
      .joinToString("\u0000"))
  }
}

private fun cppDistinctTypeMembers(tables: List<CppTypeMembers>): List<CppTypeMembers> =
  tables.groupBy(CppTypeMembers::type).map { (type, grouped) ->
    CppTypeMembers(type, cppDistinctReferences(grouped.flatMap(CppTypeMembers::members)))
  }

private const val CPP_AST_NODE_LIMIT = 12_000
private const val CPP_AST_DEPTH_LIMIT = 96
private const val CPP_AST_PARAMETER_NODE_LIMIT = 256
internal const val CPP_AST_CONTEXT_REQUEST_FIELD = "__tidyparseCppCompletionAstContext"
internal const val CPP_NORMALIZED_AST_CONTEXT_FIELD = "__tidyparseCppCompletionNormalizedAst"

private val CPP_AST_SCOPE_KINDS = setOf(
  "Compound", "CompoundStmt", "CXXConstructor", "CXXForRange", "CXXMethod",
  "For", "Function", "If", "Lambda", "Switch", "While"
)

private val CPP_AST_METHOD_KINDS = setOf(
  "CXXConstructor", "CXXConversion", "CXXDestructor", "CXXMethod"
)

/**
 * Reduces clangd's raw `textDocument/ast` tree to the small, structured-clone-safe semantic DTO
 * consumed by [cppCompletionContextDto]. Only declarations visible at the requested cursor and
 * declaration metadata needed by the statement grammar are retained.
 *
 * clangd ASTs can be much larger than the edited file because recovery nodes are verbose. The
 * traversal therefore has explicit node/depth ceilings and never copies expression subtrees into
 * the worker request. An incomplete traversal is still conservative: lexical and completion facts
 * remain available to the caller.
 */
fun cppClangdAstContextDto(
  rawAst: dynamic,
  source: String,
  cursorLine: Int,
  cursorCharacter: Int
): dynamic {
  val empty = CppCompletionContext(emptySet())
  if (!cppDefined(rawAst) || cursorLine < 0 || cursorCharacter < 0)
    return cppCompletionContextToDto(empty)
  return try {
    val root = if (cppDefined(rawAst.ast)) rawAst.ast else rawAst
    cppCompletionContextToDto(
      CppClangdAstNormalizer(source, cursorLine, cursorCharacter).normalize(root)
    )
  } catch (_: Throwable) {
    // AST is an optional enrichment; malformed recovery data must not suppress lexical fallback.
    cppCompletionContextToDto(empty)
  }
}

private data class CppNormalizedAstRecord(val name: String, val bases: List<String>, val members: List<CppReference>)

private class CppClangdAstNormalizer(source: String, private val cursorLine: Int, private val cursorCharacter: Int) {
  private val sourceLines = source.replace("\r\n", "\n").replace('\r', '\n').split('\n')
  private val values = mutableListOf<CppReference>()
  private val functions = mutableListOf<CppReference>()
  private val types = mutableListOf<CppReference>()
  private val records = mutableListOf<CppNormalizedAstRecord>()
  private val conversions = mutableListOf<CppConversion>()
  private val defaultConstructibleTypes = linkedSetOf<String>()
  private val mutableFields = linkedSetOf<String>()
  private val ancestors = mutableListOf<dynamic>()
  private var visitedNodes = 0
  private var enclosingReturnType: String? = null
  private var enclosingClassType: String? = null
  private var thisType: String? = null

  fun normalize(root: dynamic): CppCompletionContext {
    visit(root, 0)
    val recordsByName = linkedMapOf<String, CppNormalizedAstRecord>()
    records.forEach { recordsByName[it.name] = it }

    fun inheritedMembers(type: String, seen: MutableSet<String>): List<CppReference> {
      if (!seen.add(type)) return emptyList()
      val record = recordsByName[type] ?: return emptyList()
      return record.members + record.bases.flatMap { base ->
        inheritedMembers(base, seen).filterNot { it.kind == "constructor" }
      }
    }

    val membersByType = recordsByName.values.map { record ->
      CppTypeMembers(record.name, cppDistinctReferences(inheritedMembers(record.name, linkedSetOf())))
    }
    val normalizedValues = cppDistinctReferences(values)
    val normalizedFunctions = cppDistinctReferences(functions)
    val normalizedTypes = cppDistinctReferences(types)
    val identifiers = linkedSetOf<String>()
    (normalizedValues + normalizedFunctions + normalizedTypes + membersByType.flatMap { it.members })
      .forEach { reference ->
        CPP_IDENTIFIER_REGEX.findAll(reference.name).mapTo(identifiers, MatchResult::value)
        reference.parameters.mapNotNullTo(identifiers) { it.name.takeIf(String::isNotBlank) }
      }
    return CppCompletionContext(
      identifiers = identifiers,
      values = normalizedValues,
      types = normalizedTypes,
      functions = normalizedFunctions,
      membersByType = membersByType,
      conversions = conversions.distinct(),
      defaultConstructibleTypes = defaultConstructibleTypes,
      enclosingReturnType = enclosingReturnType,
      enclosingClassType = enclosingClassType,
      thisType = thisType,
      mutableFields = mutableFields
    )
  }

  private fun visit(node: dynamic, depth: Int) {
    if (!cppDefined(node) || jsTypeOf(node) != "object" || depth > CPP_AST_DEPTH_LIMIT ||
      visitedNodes >= CPP_AST_NODE_LIMIT) return
    visitedNodes++

    val kind = node.kind as? String
    val role = node.role as? String
    val name = (node.detail as? String)?.trim()?.takeIf(String::isNotEmpty)
    val declaredBeforeCursor = cppAstPositionAtOrBefore(node.range?.start)

    if (kind == "Enum" && role == "declaration" && name != null && declaredBeforeCursor) {
      types += CppReference(name, type = name, kind = "enum", detail = name, source = "ast")
    }
    if (kind == "EnumConstant" && role == "declaration" && name != null && declaredBeforeCursor) {
      val owner = nearestAncestor { ancestor ->
        ancestor.kind as? String == "Enum" && ancestor.role as? String == "declaration" &&
          (ancestor.detail as? String).isNullOrBlank().not()
      }
      val ownerName = owner?.detail as? String
      if (ownerName != null) values += CppReference(
        name = name,
        type = ownerName,
        kind = "enumMember",
        detail = if (Regex("\\b(?:class|struct|scoped)\\b")
            .containsMatchIn(owner.arcana as? String ?: "")) "scoped" else "unscoped",
        ownerType = ownerName,
        source = "ast"
      )
    }

    if (role == "declaration" && cppAstRangeContainsCursor(node.range)) {
      val enclosingRecord = nearestAncestor { ancestor ->
        ancestor.kind as? String == "CXXRecord" && ancestor.role as? String == "declaration" &&
          (ancestor.detail as? String).isNullOrBlank().not()
      }
      if (kind in CPP_AST_METHOD_KINDS && cppDefined(enclosingRecord)) {
        val recordName = enclosingRecord.detail as String
        enclosingClassType = recordName
        val signature = cppQuotedAstType(node.arcana as? String).orEmpty()
        val methodConst = Regex("\\)\\s+const(?:\\s|$)").containsMatchIn(signature)
        thisType = "${if (methodConst) "const " else ""}$recordName *"
        cppDynamicList(enclosingRecord.children).filter { child ->
          child.kind as? String == "Field" && child.detail is String &&
            Regex("\\bmutable\\b").containsMatchIn(child.arcana as? String ?: "")
        }.mapNotNullTo(mutableFields) { it.detail as? String }
      }
      enclosingReturnType = when (kind) {
        "CXXConstructor", "CXXDestructor" -> "void"
        "Function", "CXXMethod", "CXXConversion" -> cppAstReturnType(node)
        else -> enclosingReturnType
      }
    }

    if ((kind == "Var" || kind == "ParmVar") && role == "declaration" &&
      name != null && declaredBeforeCursor && cppAstSpelledBeforeCursor(node, name)) {
      val scope = nearestAncestor { it.kind as? String in CPP_AST_SCOPE_KINDS }
      if (!cppDefined(scope) || cppAstRangeContainsCursor(scope.range)) {
        val type = cppQuotedAstType(node.arcana as? String).orEmpty()
        values += CppReference(name, type = type, kind = "variable", detail = type, source = "ast")
      }
    }

    if (kind == "Function" && role == "declaration" && name != null && name != "main" &&
      declaredBeforeCursor && nearestAncestor { it.kind as? String == "CXXRecord" } == null) {
      cppAstCallable(node, "function", name)?.let(functions::add)
    }

    if (kind == "CXXRecord" && role == "declaration" && name != null && declaredBeforeCursor &&
      (Regex("\\bdefinition\\b", RegexOption.IGNORE_CASE)
        .containsMatchIn(node.arcana as? String ?: "") ||
        "DefinitionData" in (node.arcana as? String ?: ""))) {
      collectRecord(node, name)
    }

    ancestors.add(node)
    cppDynamicList(node.children).forEach { child -> visit(child, depth + 1) }
    ancestors.removeAt(ancestors.lastIndex)
  }

  private fun collectRecord(record: dynamic, name: String) {
    val arcana = record.arcana as? String ?: ""
    val kind = if (Regex("\\bstruct\\s+").containsMatchIn(arcana)) "struct" else "class"
    val abstract = Regex("\\babstract\\b").containsMatchIn(arcana)
    val children = cppDynamicList(record.children)
    val bases = children.asSequence()
      .filter { it.role as? String == "base" && it.kind as? String == "public" }
      .mapNotNull(::cppFirstAstTypeName)
      .distinct()
      .toList()
    // clang does not materialize an implicit CXXConstructor node until it is used. Preserve the
    // stronger DefinitionData fact for method-only visitors such as `Describe{}`. The
    // `empty aggregate` marker by itself is insufficient: clang also uses it for records with a deleted
    // destructor and for empty derived aggregates whose base cannot be default-constructed.
    val implicitEmptyConstruction = !abstract && bases.isEmpty() &&
      "empty aggregate" in arcana &&
      Regex("\\bDefaultConstructor\\b[^\\r\\n]*\\bneeds_implicit\\b").containsMatchIn(arcana) &&
      Regex("\\bDestructor\\b[^\\r\\n]*\\bneeds_implicit\\b").containsMatchIn(arcana) &&
      children.none { child ->
        child.role as? String == "base" || child.kind as? String in setOf(
          "Field", "CXXConstructor", "ConstructorUsingShadow", "CXXDestructor"
        )
      }
    if (implicitEmptyConstruction) defaultConstructibleTypes += name
    val members = mutableListOf<CppReference>()
    var access = if (kind == "struct") "public" else "private"
    children.forEach { child ->
      val childKind = child.kind as? String
      if (childKind == "AccessSpec") {
        access = Regex("\\b(public|private|protected)\\s*$")
          .find(child.arcana as? String ?: "")?.groupValues?.get(1) ?: access
        return@forEach
      }
      if (access != "public") return@forEach
      val childName = (child.detail as? String)?.trim()?.takeIf(String::isNotEmpty)
      val childArcana = child.arcana as? String ?: ""
      val implicit = Regex("(?:^|\\s)implicit(?:\\s|$)").containsMatchIn(childArcana)
      when {
        childKind == "Field" && childName != null -> {
          val type = cppQuotedAstType(childArcana).orEmpty()
          members += CppReference(
            name = childName,
            type = type,
            kind = "field",
            detail = type,
            receiverMember = true,
            ownerType = name,
            source = "ast"
          )
        }
        childKind == "CXXMethod" && !implicit -> {
          val callableKind = if (childName?.startsWith("operator") == true) "operator" else "method"
          cppAstCallable(child, callableKind, childName)?.let { callable ->
            members += callable.copy(receiverMember = true, ownerType = name)
          }
        }
        childKind == "CXXConstructor" && !abstract -> {
          val deleted = Regex("\\b(?:deleted|delete)\\b|default_delete")
            .containsMatchIn(childArcana)
          val inherited = Regex("(?:^|\\s)implicit\\s+used(?:\\s|$)")
            .containsMatchIn(childArcana)
          val callable = cppAstCallable(child, "constructor", name)
          val implicitDefault = implicit && callable?.parameters?.isEmpty() == true
          if (!deleted && (!implicit || inherited || implicitDefault) && callable != null) {
            functions += callable.copy(ownerType = name)
            if (callable.parameters.isEmpty()) defaultConstructibleTypes += name
          }
        }
        childKind == "ConstructorUsingShadow" && !abstract -> {
          val signature = Regex("'([^'\\r\\n]+)'").findAll(childArcana)
            .map { it.groupValues[1] }
            .lastOrNull { '(' in it }
          val parameters = signature?.let(::cppParameterClause).orEmpty()
          val inheritedCopy = parameters.size == 1 && bases.any { base ->
            parameters.single().type
              .replace(Regex("\\b(?:const|volatile)\\b"), "")
              .replace(Regex("(?:&&|&)\\s*$"), "")
              .trim() == base
          }
          if (signature != null && !inheritedCopy) functions += CppReference(
            name = name,
            returnType = name,
            parameters = parameters,
            kind = "constructor",
            detail = signature,
            ownerType = name,
            source = "ast"
          )
        }
      }
    }
    types += CppReference(name, type = name, kind = kind, detail = name, source = "ast", abstract = abstract)
    records += CppNormalizedAstRecord(name, bases, cppDistinctReferences(members))
    bases.forEach { base -> conversions += CppConversion(name, base) }
  }

  private fun cppAstCallable(
    node: dynamic,
    kind: String,
    fallbackName: String?
  ): CppReference? {
    val name = (node.detail as? String)?.trim()?.takeIf(String::isNotEmpty) ?: fallbackName ?: return null
    val signature = cppQuotedAstType(node.arcana as? String).orEmpty()
    val parameters = cppAstParameterNodes(node).map(::cppAstParameter)
      .ifEmpty { cppParameterClause(signature) }
    return CppReference(
      name = name,
      returnType = if (kind == "constructor") fallbackName else cppAstReturnType(node),
      parameters = parameters,
      kind = kind,
      detail = signature,
      source = "ast"
    )
  }

  private fun cppAstParameter(node: dynamic): CppParameter {
    val type = cppQuotedAstType(node.arcana as? String).orEmpty()
    val name = node.detail as? String ?: ""
    val label = cppAstRangeText(node.range) ?: type
    val parsed = cppParameterFromLabel(label)
    return parsed.copy(
      label = label,
      name = name.ifBlank { parsed.name },
      type = type.ifBlank { parsed.type }
    )
  }

  private fun cppAstParameterNodes(root: dynamic): List<dynamic> {
    val result = mutableListOf<dynamic>()
    var visited = 0
    fun collect(node: dynamic, depth: Int) {
      if (!cppDefined(node) || depth > 32 || visited++ >= CPP_AST_PARAMETER_NODE_LIMIT) return
      if (node.kind as? String == "ParmVar" && node.role as? String == "declaration") {
        result.add(node)
        return
      }
      if (node !== root && node.role as? String in setOf("statement", "expression")) return
      cppDynamicList(node.children).forEach { collect(it, depth + 1) }
    }
    collect(root, 0)
    return result
  }

  private fun cppFirstAstTypeName(root: dynamic): String? {
    var visited = 0
    fun find(node: dynamic, depth: Int): String? {
      if (!cppDefined(node) || depth > 24 || visited++ >= CPP_AST_PARAMETER_NODE_LIMIT) return null
      if (node.role as? String == "type")
        (node.detail as? String)?.takeIf(String::isNotBlank)?.let { return it }
      cppDynamicList(node.children).forEach { child -> find(child, depth + 1)?.let { return it } }
      return null
    }
    return find(root, 0)
  }

  private fun cppAstReturnType(node: dynamic): String? {
    val signature = cppQuotedAstType(node.arcana as? String) ?: return null
    val open = signature.indexOf('(')
    return signature.substring(0, if (open < 0) signature.length else open).trim()
      .takeIf(String::isNotEmpty)
  }

  private fun cppQuotedAstType(arcana: String?): String? {
    arcana ?: return null
    Regex("(?:^|\\s)'([^'\\r\\n]+)'\\s*:\\s*'([^'\\r\\n]+)'")
      .find(arcana)?.groupValues?.get(2)?.let { return it }
    return Regex("(?:^|\\s)'([^'\\r\\n]+)'").find(arcana)?.groupValues?.get(1)
  }

  private fun cppAstRangeText(range: dynamic): String? {
    if (!cppDefined(range?.start) || !cppDefined(range?.end)) return null
    val startLine = cppInt(range.start.line, -1)
    val endLine = cppInt(range.end.line, -1)
    if (startLine < 0 || startLine != endLine) return null
    val line = sourceLines.getOrNull(startLine) ?: return null
    val start = cppInt(range.start.character, -1)
    val end = cppInt(range.end.character, -1)
    if (start < 0 || end < start || end > line.length) return null
    return line.substring(start, end).trim().takeIf(String::isNotEmpty)
  }

  private fun cppAstSpelledBeforeCursor(node: dynamic, name: String): Boolean {
    if (cppInt(node.range?.start?.line, -1) != cursorLine) return true
    val line = sourceLines.getOrNull(cursorLine).orEmpty()
    val prefix = line.substring(0, cursorCharacter.coerceIn(0, line.length))
    return Regex("(?:^|\\W)${Regex.escape(name)}(?:$|\\W)").containsMatchIn(prefix)
  }

  private fun cppAstPositionAtOrBefore(position: dynamic): Boolean {
    if (!cppDefined(position)) return false
    val line = cppInt(position.line, -1)
    val character = cppInt(position.character, -1)
    return line >= 0 && character >= 0 &&
      (line < cursorLine || line == cursorLine && character <= cursorCharacter)
  }

  private fun cppAstCursorAtOrBefore(position: dynamic): Boolean {
    if (!cppDefined(position)) return false
    val line = cppInt(position.line, -1)
    val character = cppInt(position.character, -1)
    return line > cursorLine || line == cursorLine && character >= cursorCharacter
  }

  private fun cppAstRangeContainsCursor(range: dynamic): Boolean =
    cppDefined(range) && cppAstPositionAtOrBefore(range.start) && cppAstCursorAtOrBefore(range.end)

  private inline fun nearestAncestor(predicate: (dynamic) -> Boolean): dynamic {
    for (index in ancestors.lastIndex downTo 0)
      if (predicate(ancestors[index])) return ancestors[index]
    return null
  }
}

private fun cppContextFromNormalizedAst(ast: dynamic): CppCompletionContext {
  if (!cppDefined(ast)) return CppCompletionContext(emptySet())
  return CppCompletionContext(
    identifiers = cppStringSet(ast.identifiers),
    values = cppDynamicList(ast.values).mapNotNull(::cppReferenceFromDto),
    types = cppDynamicList(ast.types).mapNotNull(::cppReferenceFromDto),
    functions = cppDynamicList(ast.functions).mapNotNull(::cppReferenceFromDto),
    membersByType = cppDynamicList(ast.membersByType).mapNotNull(::cppTypeMembersFromDto),
    conversions = cppDynamicList(ast.conversions).mapNotNull(::cppConversionFromDto),
    requiredTypes = cppStringSet(ast.requiredTypes),
    probedRequiredTypes = cppStringSet(ast.probedRequiredTypes),
    defaultConstructibleTypes = cppStringSet(ast.defaultConstructibleTypes),
    enclosingReturnType = ast.enclosingReturnType as? String,
    enclosingClassType = ast.enclosingClassType as? String,
    thisType = ast.thisType as? String,
    mutableFields = cppStringSet(ast.mutableFields)
  )
}

private fun cppCompletionContextToDto(context: CppCompletionContext): dynamic {
  val dto = js("({})")
  dto.identifiers = context.identifiers.sorted().toTypedArray()
  dto.sourceIdentifiers = context.sourceIdentifiers.sorted().toTypedArray()
  dto.headers = context.headers.sorted().toTypedArray()
  dto.typeNames = context.typeNames.sorted().toTypedArray()
  dto.values = context.values.map(::cppReferenceToDto).toTypedArray()
  dto.types = context.types.map(::cppReferenceToDto).toTypedArray()
  dto.functions = context.functions.map(::cppReferenceToDto).toTypedArray()
  dto.completions = context.completions.map(::cppReferenceToDto).toTypedArray()
  dto.signatures = context.signatures.map(::cppSignatureToDto).toTypedArray()
  dto.expectedTypes = context.expectedTypes.sorted().toTypedArray()
  dto.receiver = context.receiver?.let(::cppReceiverToDto)
  dto.membersByType = context.membersByType.map(::cppTypeMembersToDto).toTypedArray()
  dto.conversions = context.conversions.map(::cppConversionToDto).toTypedArray()
  dto.unresolvedIdentifiers = context.unresolvedIdentifiers.sorted().toTypedArray()
  dto.requiredIdentifier = context.requiredIdentifier
  dto.requiredTypes = context.requiredTypes.sorted().toTypedArray()
  dto.probedRequiredTypes = context.probedRequiredTypes.sorted().toTypedArray()
  dto.defaultConstructibleTypes = context.defaultConstructibleTypes.sorted().toTypedArray()
  dto.enclosingReturnType = context.enclosingReturnType
  dto.enclosingClassType = context.enclosingClassType
  dto.thisType = context.thisType
  dto.mutableFields = context.mutableFields.sorted().toTypedArray()
  return dto
}

private fun cppParameterToDto(parameter: CppParameter): dynamic {
  val dto = js("({})")
  dto.label = parameter.label
  dto.name = parameter.name
  dto.type = parameter.type
  dto.defaultValue = parameter.defaultValue
  return dto
}

private fun cppReferenceToDto(reference: CppReference): dynamic {
  val dto = js("({})")
  dto.name = reference.name
  dto.type = reference.type
  dto.returnType = reference.returnType
  dto.parameters = reference.parameters.map(::cppParameterToDto).toTypedArray()
  dto.kind = reference.kind
  dto.detail = reference.detail
  dto.receiverMember = reference.receiverMember
  dto.ownerType = reference.ownerType
  dto.source = reference.source
  dto.abstract = reference.abstract
  return dto
}

private fun cppSignatureToDto(signature: CppSignature): dynamic {
  val dto = js("({})")
  dto.label = signature.label
  dto.returnType = signature.returnType
  dto.parameters = signature.parameters.map(::cppParameterToDto).toTypedArray()
  dto.activeParameter = signature.activeParameter
  return dto
}

private fun cppReceiverToDto(receiver: CppReceiver): dynamic {
  val dto = js("({})")
  dto.operator = receiver.operator
  dto.expression = receiver.expression
  dto.type = receiver.type
  dto.members = receiver.members.map(::cppReferenceToDto).toTypedArray()
  return dto
}

private fun cppTypeMembersToDto(table: CppTypeMembers): dynamic {
  val dto = js("({})")
  dto.type = table.type
  dto.members = table.members.map(::cppReferenceToDto).toTypedArray()
  return dto
}

private fun cppConversionToDto(conversion: CppConversion): dynamic {
  val dto = js("({})")
  dto.from = conversion.from
  dto.to = conversion.to
  return dto
}

private fun cppParameterFromDto(value: dynamic): CppParameter? {
  if (!cppDefined(value)) return null
  val type = value.type as? String ?: ""
  return CppParameter(
    label = value.label as? String ?: type,
    name = value.name as? String ?: "",
    type = type,
    defaultValue = value.defaultValue as? String
  )
}

private fun cppReferenceFromDto(value: dynamic): CppReference? =
  if (!cppDefined(value)) null
  else CppReference(
    name = value.name as? String ?: return null,
    type = value.type as? String,
    returnType = value.returnType as? String,
    parameters = cppDynamicList(value.parameters).mapNotNull(::cppParameterFromDto),
    kind = value.kind as? String ?: "value",
    detail = value.detail as? String,
    receiverMember = value.receiverMember as? Boolean ?: false,
    ownerType = value.ownerType as? String,
    source = value.source as? String,
    abstract = value.abstract as? Boolean ?: false
  )

private fun cppSignatureFromDto(value: dynamic): CppSignature? =
  if (!cppDefined(value)) null
  else CppSignature(
    label = value.label as? String ?: return null,
    returnType = value.returnType as? String,
    parameters = cppDynamicList(value.parameters).mapNotNull(::cppParameterFromDto),
    activeParameter = (value.activeParameter as? Number)?.toInt()
  )

private fun cppReceiverFromDto(value: dynamic): CppReceiver? =
  if (!cppDefined(value)) null
  else CppReceiver(
    operator = value.operator as? String ?: return null,
    expression = value.expression as? String ?: "",
    type = value.type as? String,
    members = cppDynamicList(value.members).mapNotNull(::cppReferenceFromDto)
  )

private fun cppTypeMembersFromDto(value: dynamic): CppTypeMembers? =
  if (!cppDefined(value)) null
  else CppTypeMembers(
    type = value.type as? String ?: return null,
    members = cppDynamicList(value.members).mapNotNull(::cppReferenceFromDto)
  )

private fun cppConversionFromDto(value: dynamic): CppConversion? =
  if (!cppDefined(value)) null
  else CppConversion(
    from = value.from as? String ?: return null,
    to = value.to as? String ?: return null
  )

private fun cppDefined(value: dynamic): Boolean = value != null && jsTypeOf(value) != "undefined"

private fun cppIsArray(value: dynamic): Boolean = cppDefined(value) && js("Array.isArray(value)") as Boolean

private fun cppDynamicList(value: dynamic): List<dynamic> =
  if (!cppIsArray(value)) emptyList() else (0 until cppInt(value.length)).map { value[it] }

private fun cppStringSet(value: dynamic): Set<String> =
  cppDynamicList(value).mapNotNullTo(linkedSetOf()) { it as? String }

private fun cppInt(value: dynamic, fallback: Int = 0): Int = (value as? Number)?.toInt() ?: fallback
