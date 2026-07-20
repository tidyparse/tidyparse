package ai.hypergraph.tidyparse

import ai.hypergraph.tidyparse.template.BaseTest
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.application.runReadAction
import com.intellij.testFramework.PsiTestUtil
import com.intellij.util.ui.UIUtil
import java.io.File
import java.net.URLDecoder
import java.nio.charset.StandardCharsets

class KotlinArgumentCompletionTest : BaseTest() {
  fun testEnumeratorReturnsLengthOrderedTypedExpressions() {
    val string = "String"
    val int = "Int"
    val environment = TypedKotlinEnvironment(
      values = listOf(
        TypedKotlinValue("name", string),
        TypedKotlinValue("count", int)
      ),
      functions = listOf(
        TypedKotlinCallable("hello", emptyList(), string),
        TypedKotlinCallable("greet", listOf(string), string),
        TypedKotlinCallable("size", listOf(string), int)
      )
    )

    assertEquals(
      listOf("name", "hello()", "greet(name)", "greet(hello())"),
      enumerateTypedKotlinExpressions(
        expectedType = string,
        environment = environment,
        maxLength = 10,
        maxResults = 4,
        isSubtype = { actual, expected -> actual == expected }
      )
    )
  }

  fun testEnumeratorYieldsLazilyBeforeInspectingMembers() {
    var memberLookups = 0
    val completions = enumerateTypedKotlinExpressionSequence(
      expectedType = "String",
      environment = TypedKotlinEnvironment(
        values = listOf(TypedKotlinValue("name", "String"))
      ),
      maxLength = 10,
      isSubtype = { actual, expected -> actual == expected },
      membersOf = {
        memberLookups += 1
        fail("The first completion should be available before member lookup")
        TypedKotlinEnvironment()
      }
    ).take(1).toList()

    assertEquals(listOf("name"), completions)
    assertEquals(0, memberLookups)
  }

  fun testEnumeratorMemoizesMembersByReceiverType() {
    var memberLookups = 0
    val completions = enumerateTypedKotlinExpressionSequence(
      expectedType = "Int",
      environment = TypedKotlinEnvironment(
        values = listOf(
          TypedKotlinValue("left", "Box"),
          TypedKotlinValue("right", "Box")
        )
      ),
      maxLength = 3,
      isSubtype = { actual, expected -> actual == expected },
      membersOf = {
        memberLookups += 1
        TypedKotlinEnvironment(values = listOf(TypedKotlinValue("size", "Int")))
      }
    ).take(2).toList()

    assertEquals(listOf("left.size", "right.size"), completions)
    assertEquals(1, memberLookups)
  }

  fun testKotlinArgumentPopupUsesScopeIdentifiers() {
    myFixture.configureByText(
      "scratch.kt",
      """
        fun hello(): String = "hi"
        val name = "ben"
        fun greet(name: String): String = "hi ${'$'}name"
        fun println(value: Any?) {}
        fun other() {
          val hidden = "not visible"
        }

        fun main() {
          val localName = "ada"
          println(<caret>)
          val later = "too late"
        }
      """.trimIndent()
    )

    val result = ApplicationManager.getApplication().executeOnPooledThread<KotlinArgumentCompletionResult?> {
      runReadAction { buildKotlinArgumentCompletions(myFixture.editor, myFixture.file) }
    }.get()
    assertNotNull(result)

    result!!
    showTidyKotlinArgumentLookup(project, myFixture.editor, result.replacementRange, result.completions)
    UIUtil.dispatchAllInvocationEvents()

    val lookupStrings = myFixture.lookupElementStrings ?: emptyList()
    assertTrue("Missing name in $lookupStrings", lookupStrings.contains("name"))
    assertTrue("Missing localName in $lookupStrings", lookupStrings.contains("localName"))
    assertTrue("Missing hello() in $lookupStrings", lookupStrings.contains("hello()"))
    assertTrue("Missing greet(name) in $lookupStrings", lookupStrings.contains("greet(name)"))
    assertFalse("Unexpected hidden in $lookupStrings", lookupStrings.contains("hidden"))
    assertFalse("Unexpected later in $lookupStrings", lookupStrings.contains("later"))
    assertEquals("localName", lookupStrings.first())

    myFixture.type('\n')
    UIUtil.dispatchAllInvocationEvents()

    assertTrue(myFixture.editor.document.text.contains("println(localName)"))
  }

  fun testKotlinArgumentCompletionOffersImplicitThis() {
    myFixture.configureByText(
      "Greeter.kt",
      """
        class Greeter {
          fun accept(value: Greeter) {}

          fun run() {
            accept(<caret>)
          }
        }
      """.trimIndent()
    )

    val diagnostics = mutableListOf<String>()
    val result = ApplicationManager.getApplication().executeOnPooledThread<KotlinArgumentCompletionResult?> {
      runReadAction { buildKotlinArgumentCompletions(myFixture.editor, myFixture.file, diagnostics::add) }
    }.get()
    assertNotNull(result)

    val completions = result!!.completions
    assertTrue("Missing this in $completions\n${diagnostics.joinToString("\n")}", completions.contains("this"))
  }

  fun testKotlinArgumentCompletionKeepsOverloadsWithDifferentParameterTypes() {
    myFixture.configureByText(
      "scratch.kt",
      """
        val name = "ben"
        val count = 1

        fun pick(value: String): String = value
        fun pick(value: Int): String = value.toString()
        fun consume(value: String) {}

        fun main() {
          consume(<caret>)
        }
      """.trimIndent()
    )

    val diagnostics = mutableListOf<String>()
    val result = ApplicationManager.getApplication().executeOnPooledThread<KotlinArgumentCompletionResult?> {
      runReadAction { buildKotlinArgumentCompletions(myFixture.editor, myFixture.file, diagnostics::add) }
    }.get()
    assertNotNull(result)

    val completions = result!!.completions
    assertTrue("Missing pick(name) in $completions\n${diagnostics.joinToString("\n")}", completions.contains("pick(name)"))
    assertTrue("Missing pick(count) in $completions\n${diagnostics.joinToString("\n")}", completions.contains("pick(count)"))
  }

  fun testKotlinArgumentCompletionUsesDefaultImportedPrintlnExpectedType() {
    addKotlinStdlibToTestModule()

    myFixture.configureByText(
      "scratch.kt",
      """
        fun hello(): String = "hi"
        val name = "ben"
        fun greet(name: String): String = "hi ${'$'}name"

        fun main() {
          val localName = "ada"
          println(<caret>)
        }
      """.trimIndent()
    )

    val diagnostics = mutableListOf<String>()
    val result = ApplicationManager.getApplication().executeOnPooledThread<KotlinArgumentCompletionResult?> {
      runReadAction { buildKotlinArgumentCompletions(myFixture.editor, myFixture.file, diagnostics::add) }
    }.get()
    assertNotNull(result)

    val completions = result!!.completions
    assertTrue((result.message ?: "") + "\n" + diagnostics.joinToString("\n"), completions.isNotEmpty())
    assertTrue("Missing name in $completions", completions.contains("name"))
    assertTrue("Missing localName in $completions", completions.contains("localName"))
    assertTrue("Missing hello() in $completions", completions.contains("hello()"))
    assertTrue("Missing greet(name) in $completions", completions.contains("greet(name)"))
  }

  fun testKotlinArgumentCompletionUsesResolvedProjectScopeAndExternalMembers() {
    myFixture.addFileToProject(
      "demo/Model.kt",
      """
        package demo

        class User(val name: String)

        fun makeName(): String = "from another file"
      """.trimIndent()
    )

    myFixture.configureByText(
      "Main.kt",
      """
        package demo

        fun consume(value: String) {}

        fun main() {
          val user = User("ada")
          consume(<caret>)
        }
      """.trimIndent()
    )

    val result = ApplicationManager.getApplication().executeOnPooledThread<KotlinArgumentCompletionResult?> {
      runReadAction { buildKotlinArgumentCompletions(myFixture.editor, myFixture.file) }
    }.get()
    assertNotNull(result)

    val completions = result!!.completions
    assertTrue("Missing makeName() in $completions", completions.contains("makeName()"))
    assertTrue("Missing user.name in $completions", completions.contains("user.name"))
  }

  private fun addKotlinStdlibToTestModule() {
    val stdlibJar = findKotlinStdlibJar()
    assertTrue("Kotlin stdlib jar is not a file: $stdlibJar", stdlibJar.isFile)
    PsiTestUtil.addLibrary(myFixture.module, "kotlin-stdlib", stdlibJar.parent, stdlibJar.name)
  }

  private fun findKotlinStdlibJar(): File {
    val resource = Unit::class.java.getResource("/kotlin/io/ConsoleKt.class")
      ?: Unit::class.java.getResource("/kotlin/Unit.class")

    resource?.toString()
      ?.takeIf { it.startsWith("jar:file:") }
      ?.substringAfter("jar:file:")
      ?.substringBefore("!")
      ?.let { URLDecoder.decode(it, StandardCharsets.UTF_8) }
      ?.let(::File)
      ?.takeIf { it.isFile }
      ?.let { return it }

    System.getProperty("java.class.path")
      .split(File.pathSeparator)
      .asSequence()
      .map(::File)
      .firstOrNull { it.isFile && it.name.matches(Regex("kotlin-stdlib(?:-jdk[78])?-.*\\.jar")) }
      ?.let { return it }

    fail("Could not locate kotlin-stdlib.jar from test runtime")
    throw AssertionError("unreachable")
  }
}
