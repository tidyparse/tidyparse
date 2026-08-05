import cppcompletion.CppCompletionGrammar
import cppcompletion.cppLines
import cppcompletion.shortestCompletions
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CppClangdAstContextTest {
  @Test
  fun ordinaryFunctionRetainsReturnTypeAndOnlyVisibleScopedDeclarations() {
    val source = """
      bool before(int flag) { return flag > 0; }
      std::string render(int count) {
        int visible = count;
        if (count > 0) {
          int hidden = count;
        }
        return
        int future = count;
      }
    """.trimIndent()
    val lines = source.lines()
    val cursorLine = lines.indexOfFirst { it.trim() == "return" }
    val cursorCharacter = lines[cursorLine].length
    val lastLine = lines.lastIndex

    val before = cppAstNode(
      kind = "Function",
      detail = "before",
      arcana = "FunctionDecl before 'bool (int)'",
      range = cppAstRange(0, 0, 0, lines[0].length),
      children = listOf(
        cppAstNode("ParmVar", "flag", "ParmVarDecl flag 'int'", cppAstRange(0, 12, 0, 20))
      )
    )
    val render = cppAstNode(
      kind = "Function",
      detail = "render",
      arcana = "FunctionDecl render 'std::string (int)'",
      range = cppAstRange(1, 0, lastLine, lines[lastLine].length),
      children = listOf(
        cppAstNode("ParmVar", "count", "ParmVarDecl count 'int'", cppAstRange(1, 19, 1, 28)),
        cppAstNode(
          kind = "Compound",
          arcana = "CompoundStmt",
          range = cppAstRange(1, lines[1].indexOf('{'), lastLine, lines[lastLine].length),
          role = "statement",
          children = listOf(
            cppAstNode("Var", "visible", "VarDecl visible 'int'", cppAstLineRange(lines, 2)),
            cppAstNode(
              kind = "If",
              arcana = "IfStmt",
              range = cppAstRange(3, 2, 5, lines[5].length),
              role = "statement",
              children = listOf(
                cppAstNode("Var", "hidden", "VarDecl hidden 'int'", cppAstLineRange(lines, 4))
              )
            ),
            cppAstNode("Var", "future", "VarDecl future 'int'", cppAstLineRange(lines, 7))
          )
        )
      )
    )
    val ast = cppAstNode(
      kind = "TranslationUnit",
      arcana = "TranslationUnitDecl",
      range = cppAstRange(0, 0, lastLine, lines[lastLine].length),
      role = "translation unit",
      children = listOf(before, render)
    )

    val context = cppAstContext(source, cursorLine, cursorCharacter, ast)
    assertEquals("std::string", context.enclosingReturnType)
    assertNull(context.thisType)
    assertTrue(context.values.any { it.name == "count" && it.type == "int" })
    assertTrue(context.values.any { it.name == "visible" && it.type == "int" })
    assertFalse(context.values.any { it.name == "hidden" })
    assertFalse(context.values.any { it.name == "future" })
    assertTrue(context.functions.any { it.name == "before" && it.returnType == "bool" })
  }

  @Test
  fun constMethodRetainsThisReturnMembersAndMethodScope() {
    val source = """
      struct Route {
      public:
        int weight;
        mutable int cache;
        const Route & extend(const std::string & path) const {
          int step = weight;
          return
        }
      };
    """.trimIndent()
    val lines = source.lines()
    val cursorLine = lines.indexOfFirst { it.trim() == "return" }
    val cursorCharacter = lines[cursorLine].length
    val lastLine = lines.lastIndex

    val method = cppAstNode(
      kind = "CXXMethod",
      detail = "extend",
      arcana = "CXXMethodDecl extend 'const Route &(const std::string &) const'",
      range = cppAstRange(4, 2, 7, lines[7].length),
      children = listOf(
        cppAstNode(
          "ParmVar",
          "path",
          "ParmVarDecl path 'const std::string &'",
          cppAstRange(4, lines[4].indexOf("const std::string"), 4, lines[4].indexOf(')'))
        ),
        cppAstNode(
          kind = "Compound",
          arcana = "CompoundStmt",
          range = cppAstRange(4, lines[4].indexOf('{'), 7, lines[7].length),
          role = "statement",
          children = listOf(
            cppAstNode("Var", "step", "VarDecl step 'int'", cppAstLineRange(lines, 5))
          )
        )
      )
    )
    val route = cppAstNode(
      kind = "CXXRecord",
      detail = "Route",
      arcana = "CXXRecordDecl struct Route definition",
      range = cppAstRange(0, 0, lastLine, lines[lastLine].length),
      children = listOf(
        cppAstNode("AccessSpec", arcana = "AccessSpecDecl public", range = cppAstLineRange(lines, 1)),
        cppAstNode("Field", "weight", "FieldDecl weight 'int'", cppAstLineRange(lines, 2)),
        cppAstNode("Field", "cache", "FieldDecl cache 'int' mutable", cppAstLineRange(lines, 3)),
        method
      )
    )
    val ast = cppAstNode(
      kind = "TranslationUnit",
      arcana = "TranslationUnitDecl",
      range = cppAstRange(0, 0, lastLine, lines[lastLine].length),
      role = "translation unit",
      children = listOf(route)
    )

    val context = cppAstContext(source, cursorLine, cursorCharacter, ast)
    assertEquals("const Route &", context.enclosingReturnType)
    assertEquals("Route", context.enclosingClassType)
    assertEquals("const Route *", context.thisType)
    assertEquals(setOf("cache"), context.mutableFields)
    assertTrue(context.values.any { it.name == "path" && it.type == "const std::string &" })
    assertTrue(context.values.any { it.name == "step" && it.type == "int" })
    val routeMembers = assertNotNull(context.membersByType.firstOrNull { it.type == "Route" }).members
    assertTrue(routeMembers.any { it.name == "weight" && it.kind == "field" })
    assertTrue(routeMembers.any { it.name == "cache" && it.kind == "field" })
    assertTrue(routeMembers.any { it.name == "extend" && it.kind == "method" })
    assertTrue(context.types.any { it.name == "Route" && it.source == "ast" })
  }

  @Test
  fun inheritedConstructorsFollowClangRecoverySemantics() {
    val source = """
      struct Base {
        Base(int value);
        Base(const Base & other);
      };
      struct Derived : Base {
        using Base::Base;
      };
      int main() {
        return
      }
    """.trimIndent()
    val lines = source.lines()
    val cursorLine = lines.indexOfFirst { it.trim() == "return" }
    val cursorCharacter = lines[cursorLine].length
    val base = cppAstNode(
      kind = "CXXRecord",
      detail = "Base",
      arcana = "CXXRecordDecl struct Base definition",
      range = cppAstRange(0, 0, 3, lines[3].length)
    )
    val publicBase = cppAstNode(
      kind = "public",
      arcana = "public Base",
      range = cppAstLineRange(lines, 4),
      role = "base",
      children = listOf(
        cppAstNode(
          kind = "Record",
          detail = "Base",
          arcana = "RecordType Base",
          range = cppAstLineRange(lines, 4),
          role = "type"
        )
      )
    )
    val inheritedUsed = cppAstNode(
      kind = "CXXConstructor",
      detail = "Derived",
      arcana = "CXXConstructorDecl Derived 'void (double)' implicit used",
      range = cppAstLineRange(lines, 5)
    )
    val inheritedShadow = cppAstNode(
      kind = "ConstructorUsingShadow",
      detail = "Base::Base",
      arcana = "ConstructorUsingShadowDecl Base::Base 'void (int)'",
      range = cppAstLineRange(lines, 5)
    )
    val inheritedCopyShadow = cppAstNode(
      kind = "ConstructorUsingShadow",
      detail = "Base::Base",
      arcana = "ConstructorUsingShadowDecl Base::Base 'void (const Base &)'",
      range = cppAstLineRange(lines, 5)
    )
    val derived = cppAstNode(
      kind = "CXXRecord",
      detail = "Derived",
      arcana = "CXXRecordDecl struct Derived definition",
      range = cppAstRange(4, 0, 6, lines[6].length),
      children = listOf(publicBase, inheritedUsed, inheritedShadow, inheritedCopyShadow)
    )
    val ast = cppAstNode(
      kind = "TranslationUnit",
      arcana = "TranslationUnitDecl",
      range = cppAstRange(0, 0, lines.lastIndex, lines.last().length),
      role = "translation unit",
      children = listOf(base, derived)
    )

    val context = cppAstContext(source, cursorLine, cursorCharacter, ast)
    val derivedConstructors = context.functions.filter {
      it.kind == "constructor" && it.ownerType == "Derived"
    }
    assertEquals(
      setOf(listOf("double"), listOf("int")),
      derivedConstructors.mapTo(linkedSetOf()) { constructor ->
        constructor.parameters.map { it.type }
      }
    )
    assertTrue(derivedConstructors.all { it.name == "Derived" && it.returnType == "Derived" })
    assertFalse(derivedConstructors.any { constructor ->
      constructor.parameters.singleOrNull()?.type == "const Base &"
    })
    assertTrue(context.conversions.any { it.from == "Derived" && it.to == "Base" })
  }

  @Test
  fun implicitEmptyVisitorCompletesVisitAtTheExactEditorBoundary() {
    val source = """
      #include <iostream>
      #include <optional>
      #include <string>
      #include <variant>

      struct Describe {
          std::string operator()(std::monostate) const { return "empty"; }
          std::string operator()(int value) const { return std::to_string(value); }
          std::string operator()(const std::string& value) const { return value; }
      };

      int main() {
          std::optional<std::string> nickname = std::nullopt;
          nickname.emplace("Ada");
          std::string display = nickname.value_or("anonymous");
          std::variant<std::monostate, int, std::string> payload = std::monostate{};
          payload = std::string{"ready"};
          bool textual = std::holds_alternative<std::string>(payload);
          const std::string* text = std::get_if<std::string>(&payload);
          std::string rendered = std::visit(
      }
    """.trimIndent()
    val lines = source.lines()
    val cursorLine = lines.indexOfFirst { "std::string rendered" in it }
    val cursorCharacter = lines[cursorLine].length
    val recordArcana = """
      CXXRecordDecl struct Describe definitionDefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal
      |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
      `-Destructor simple irrelevant trivial constexpr needs_implicit
    """.trimIndent()
    val describe = cppAstNode(
      kind = "CXXRecord",
      detail = "Describe",
      arcana = recordArcana,
      range = cppAstRange(5, 0, 9, lines[9].length),
      children = listOf(
        cppAstNode(
          "CXXMethod", "operator()",
          "CXXMethodDecl operator() 'std::string (std::monostate) const'",
          cppAstLineRange(lines, 6)
        ),
        cppAstNode(
          "CXXMethod", "operator()",
          "CXXMethodDecl operator() 'std::string (int) const'",
          cppAstLineRange(lines, 7)
        ),
        cppAstNode(
          "CXXMethod", "operator()",
          "CXXMethodDecl operator() 'std::string (const std::string &) const'",
          cppAstLineRange(lines, 8)
        )
      )
    )
    val main = cppAstNode(
      kind = "Function",
      detail = "main",
      arcana = "FunctionDecl main 'int ()'",
      range = cppAstRange(11, 0, lines.lastIndex, lines.last().length),
      children = listOf(
        cppAstNode(
          "Var", "payload",
          "VarDecl payload 'std::variant<std::monostate, int, std::string>'",
          cppAstLineRange(lines, 15)
        )
      )
    )
    val ast = cppAstNode(
      kind = "TranslationUnit",
      arcana = "TranslationUnitDecl",
      range = cppAstRange(0, 0, lines.lastIndex, lines.last().length),
      role = "translation unit",
      children = listOf(describe, main)
    )

    val context = cppAstContext(source, cursorLine, cursorCharacter, ast)
    assertTrue("Describe" in context.defaultConstructibleTypes)
    assertTrue(context.values.any {
      it.name == "payload" && it.type == "std::variant<std::monostate, int, std::string>"
    })
    assertEquals(
      3,
      assertNotNull(context.membersByType.firstOrNull { it.type == "Describe" })
        .members.count { it.name == "operator()" }
    )

    val snapshot = assertNotNull(cppEditorStatementSnapshot(source, cursorLine, cursorCharacter))
    val expectedTokens = cppLines("Describe{}, payload);").single().tokens.map { it.text }
    val completions = CppCompletionGrammar().generate(context, snapshot.tokens).shortestCompletions(
      prefixText = snapshot.prefixText,
      identifiersInFile = context.identifiers,
      limit = 10,
      random = Random(snapshot.seed)
    )
    assertTrue(completions.any { completion ->
      completion.tokens == expectedTokens &&
        completion.insertionText == "Describe{},payload);"
    })
  }

  @Test
  fun implicitEmptyConstructionRejectsStorageBasesAndDeclaredSpecialMembers() {
    val source = """
      struct Good {};
      struct WithField { int& value; };
      struct WithBase : Good {};
      struct DeletedConstructor { DeletedConstructor() = delete; };
      struct DeletedDestructor { ~DeletedDestructor() = delete; };
      int main() {
        return
      }
    """.trimIndent()
    val lines = source.lines()
    val cursorLine = lines.indexOfFirst { it.trim() == "return" }
    fun implicitMetadata(name: String) = """
      CXXRecordDecl struct $name definitionDefinitionData empty aggregate trivial
      |-DefaultConstructor exists trivial needs_implicit
      `-Destructor simple trivial needs_implicit
    """.trimIndent()
    fun record(name: String, line: Int, children: List<dynamic> = emptyList()) = cppAstNode(
      kind = "CXXRecord",
      detail = name,
      arcana = implicitMetadata(name),
      range = cppAstLineRange(lines, line),
      children = children
    )
    val base = cppAstNode(
      kind = "public",
      arcana = "public Good",
      range = cppAstLineRange(lines, 2),
      role = "base",
      children = listOf(
        cppAstNode(
          kind = "Record", detail = "Good", arcana = "RecordType Good",
          range = cppAstLineRange(lines, 2), role = "type"
        )
      )
    )
    val records: List<dynamic> = listOf(
      record("Good", 0),
      record(
        "WithField", 1,
        listOf(cppAstNode("Field", "value", "FieldDecl value 'int &'", cppAstLineRange(lines, 1)))
      ),
      record("WithBase", 2, listOf(base)),
      record(
        "DeletedConstructor", 3,
        listOf(cppAstNode(
          "CXXConstructor", "DeletedConstructor",
          "CXXConstructorDecl DeletedConstructor 'void ()' delete",
          cppAstLineRange(lines, 3)
        ))
      ),
      record(
        "DeletedDestructor", 4,
        listOf(cppAstNode(
          "CXXDestructor", "~DeletedDestructor",
          "CXXDestructorDecl ~DeletedDestructor 'void ()' delete",
          cppAstLineRange(lines, 4)
        ))
      )
    )
    val main = cppAstNode(
      kind = "Function", detail = "main", arcana = "FunctionDecl main 'int ()'",
      range = cppAstRange(5, 0, lines.lastIndex, lines.last().length)
    )
    val ast = cppAstNode(
      kind = "TranslationUnit",
      arcana = "TranslationUnitDecl",
      range = cppAstRange(0, 0, lines.lastIndex, lines.last().length),
      role = "translation unit",
      children = buildList<dynamic> {
        addAll(records)
        add(main)
      }
    )

    val context = cppAstContext(source, cursorLine, lines[cursorLine].length, ast)
    assertEquals(
      setOf("Good"),
      context.defaultConstructibleTypes.intersect(records.mapTo(linkedSetOf()) { it.detail as String })
    )
  }
}

private fun cppAstContext(
  source: String,
  cursorLine: Int,
  cursorCharacter: Int,
  ast: dynamic
): cppcompletion.CppCompletionContext {
  val normalized = cppClangdAstContextDto(ast, source, cursorLine, cursorCharacter)
  // The production path crosses a Worker structured-clone boundary; exercise the same plain-data
  // contract before merging lexical facts and rehydrating the grammar context.
  val clone = JSON.parse<dynamic>(JSON.stringify(normalized))
  val snapshot = assertNotNull(cppEditorStatementSnapshot(source, cursorLine, cursorCharacter))
  val dto = cppCompletionContextDto(source = source, ast = clone, snapshot = snapshot)
  return cppCompletionContextFromDto(JSON.parse(JSON.stringify(dto)))
}

private fun cppAstNode(
  kind: String,
  detail: String? = null,
  arcana: String = kind,
  range: dynamic,
  role: String = "declaration",
  children: List<dynamic> = emptyList()
): dynamic {
  val node = js("({})")
  node.kind = kind
  node.role = role
  node.arcana = arcana
  node.range = range
  if (detail != null) node.detail = detail
  node.children = children.toTypedArray()
  return node
}

private fun cppAstLineRange(lines: List<String>, line: Int): dynamic =
  cppAstRange(line, 0, line, lines[line].length)

private fun cppAstRange(
  startLine: Int,
  startCharacter: Int,
  endLine: Int,
  endCharacter: Int
): dynamic {
  val range = js("({})")
  range.start = js("({})")
  range.start.line = startLine
  range.start.character = startCharacter
  range.end = js("({})")
  range.end.line = endLine
  range.end.character = endCharacter
  return range
}
