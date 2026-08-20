# Generated C++ statement grammar

`CPP14Parser.g4` is an unmodified copy of grammars-v4 at commit
`e756f2a2ee5565a9300666f100ba6acd874664f7`. Its SHA-256 is
`628062e9f75710ba1d1436ced8bd7d9d8f2f08c31a6e962c175e06b28994ff27`.

The `generateCppStatementGrammar` Gradle task reads this parser together with the independently
pinned `tidyparse-cpp/antlr/cpp/CPP14Lexer.g4`. It computes the parser-rule closure rooted at
`statement`, desugars ANTLR EBNF, applies the audited grammar errata and modern-C++ overlay in
`GenerateCppStatementGrammar.kt`, and emits an epsilon-free binary CFG as packed Kotlin tables.
The browser never parses the `.g4` source or performs CNF conversion.

The language contract is the pinned C++14 statement grammar plus audited overlay revision 4. In
particular, the overlay repairs upstream's `noPointerAbstractDeclarator`, which cannot otherwise
begin with an array suffix and rejects standard type-ids such as `int[]`. Its lexical audit also
adds the standard alternative operator spellings omitted by the pinned lexer (`bitand`, `bitor`,
`compl`, `xor`, `and_eq`, `or_eq`, `xor_eq`, and `not_eq`), treats the contextual keywords `final`
and `override` as identifiers where the parser asks for an `Identifier`, and preserves four distinct
user-defined literal categories. These are build-time vocabulary additions, so the pinned lexer file
and its checksum remain unchanged. The one upstream semantic predicate (`IsPureSpecifierAllowed`)
is erased because it consumes no tokens. The result is intentionally a context-independent syntax
language, not a claim of semantic validity or
complete coverage of every later ISO C++ revision.
