package cppcompletion

import kotlinx.coroutines.MainScope
import kotlinx.coroutines.promise
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Regression coverage for implementation qualifiers in Clang's semantic type spellings. */
class CppSemanticTypeSpellingTest {
  @Test
  fun oneSourceSpellingCannotAuthenticateTwoOpaqueTypeIdentities() {
    fun info(id: String) = CppTypeInfo(
      id = id, canonicalId = id, valueCanonicalId = id,
      kind = "record", isSourceSpellable = true, isComplete = true
    )
    val first = info("type:first")
    val second = info("type:second")
    val language = CppCompletionGrammar().generate(
      CppCompletionContext(
        identifiers = setOf("Clash", "source", "result"),
        sourceIdentifiers = setOf("Clash", "source", "result"),
        types = listOf(
          CppReference(
            name = "Clash", type = "PhysicalFirst", canonicalType = "PhysicalFirst",
            kind = "class", isType = true, typeInfo = first, completionVisible = true
          ),
          CppReference(
            name = "Clash", type = "PhysicalSecond", canonicalType = "PhysicalSecond",
            kind = "class", isType = true, typeInfo = second, completionVisible = true
          )
        ),
        values = listOf(CppReference(
          name = "source", type = "PhysicalSecond", canonicalType = "PhysicalSecond",
          kind = "variable", isValue = true, typeInfo = second
        )),
        requiredBinderObligation = CppRequiredBinderObligation(setOf("result")),
        completionKind = "expression"
      ),
      emptyList()
    )

    assertFalse(
      language.recognizes(cppLines("Clash result = source;").single().tokens),
      "the first opaque owner of a spelling must prevent a contradictory type from emitting it"
    )
  }

  @Test
  fun nullabilityDoesNotSplitAnImplicitPointerConversionSource() {
    val textType = CppTypeInfo(
      id = "record:Text",
      canonicalId = "record:Text",
      valueCanonicalId = "record:Text",
      kind = "record",
      isSourceSpellable = true,
      isComplete = true
    )
    val text = CppReference(
      name = "Text",
      type = "Text",
      canonicalType = "Text",
      kind = "class",
      isType = true,
      typeInfo = textType,
      completionVisible = true
    )
    val context = CppCompletionContext(
      identifiers = setOf("Text", "label"),
      sourceIdentifiers = setOf("Text", "label"),
      types = listOf(text),
      conversions = listOf(CppConversion("const char * _Nonnull", "Text")),
      requiredIdentifier = "label",
      completionKind = "expression"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes(cppLines(
      "const Text label = true ? \"publisher\" : \"reader\";"
    ).single().tokens))
  }

  @Test
  fun conversionIdentityJoinsAnAliasToItsCanonicalRecordSpelling() {
    val stringType = CppTypeInfo(
      id = "alias:std-string",
      canonicalId = "record:std-string",
      valueCanonicalId = "record:std-string",
      kind = "record",
      isSourceSpellable = true,
      isComplete = true
    )
    val cStringType = CppTypeInfo(
      canonicalId = "pointer:const-char",
      valueCanonicalId = "pointer:const-char",
      kind = "pointer",
      pointeeCanonicalId = "builtin:char",
      pointeeIsConst = true,
      isSourceSpellable = true,
      isComplete = true
    )
    val context = CppCompletionContext(
      identifiers = setOf("label"),
      sourceIdentifiers = setOf("label"),
      functions = listOf(CppReference(
        name = "consume",
        returnType = "void",
        parameters = listOf(CppParameter(
          type = "const char * __restrict",
          canonicalType = "const char * __restrict",
          typeInfo = cStringType
        )),
        kind = "function",
        isCallable = true,
        completionVisible = true
      )),
      types = listOf(CppReference(
        name = "std::string",
        type = "std::__cxx11::basic_string<char>",
        canonicalType = "std::__cxx11::basic_string<char>",
        kind = "typeAlias",
        isType = true,
        typeInfo = stringType,
        completionVisible = true
      )),
      conversions = listOf(CppConversion(
        from = "const char *",
        to = "std::basic_string<char>",
        kind = "constructor",
        canonicalFromType = "const char *",
        canonicalToType = "std::basic_string<char>",
        fromTypeInfo = cStringType,
        toTypeInfo = stringType
      )),
      requiredIdentifier = "label",
      completionKind = "expression"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(language.recognizes(cppLines(
      "const std::string label = true ? \"publisher\" : \"reader\";"
    ).single().tokens))
  }

  @Test
  fun browserSemaPointerConversionFeedsARecordInitializer(): Promise<Unit> = MainScope().promise {
    val source = """#include <string>
int main() {
  bool publish = true;
  
}
"""
    val sema = CppBrowserClangdClient().context(source, 3, 2, 4_096, 2)
    assertTrue(
      sema.conversions.any { conversion ->
        "char" in conversion.from && "string" in conversion.to
      },
      "Sema did not expose the string converting-constructor edge: ${sema.conversions}"
    )
    val context = sema.copy(
      identifiers = sema.identifiers + "label",
      sourceIdentifiers = sema.sourceIdentifiers + "label",
      requiredIdentifier = "label"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(
      language.recognizes(cppLines(
        "const std::string label = publish ? \"publisher\" : \"reader\";"
      ).single().tokens),
      "The structured endpoint conversion did not reach the conditional initializer grammar"
    )
  }

  @Test
  fun browserSemaPublicBaseEdgeFeedsAnAddressUpcast(): Promise<Unit> = MainScope().promise {
    val source = """
      struct Root { virtual ~Root() = default; };
      struct Leaf final : public Root {};
      int main() {
        Leaf leaf;
        
      }
    """.trimIndent()
    val lines = source.lines()
    val line = lines.indexOfLast(String::isBlank)
    val sema = CppBrowserClangdClient().context(source, line, lines[line].length, 4_096, 2)
    assertTrue(
      sema.conversions.any {
        it.kind == "base" && it.from.removePrefix("::") == "Leaf" &&
          it.to.removePrefix("::") == "Root"
      },
      "Sema did not expose the public base edge: ${sema.conversions}"
    )
    val context = sema.copy(
      identifiers = sema.identifiers + "base",
      sourceIdentifiers = sema.sourceIdentifiers + "base",
      requiredIdentifier = "base"
    )
    val language = CppCompletionGrammar().generate(context, emptyList())

    assertTrue(
      language.recognizes(cppLines("Root * base = &leaf;").single().tokens),
      "The structured public-base edge did not lift to its matching pointer conversion"
    )
  }
}
