# C++ completion benchmark

The browser editor and this benchmark build a finite, cursor-specific C++ statement CFG from
clangd facts. Every `.cpp`
file in `resources/cpp-completion/` is discovered and compiled intact before scoring. Semicolon-ended
statement lines are then truncated at every ANTLR token boundary and completed using only forward
continuations. The committed corpus currently contains 133 statements and 2,141 completion instances:
for each statement with `n` tokens, indexes `0..n` are all included, including the empty prefix and
the already-complete line.

## Run

```text
CPP_COMPLETION_BENCHMARK=1 ./gradlew :tidyparse-web:jsBrowserTest
```

The default run is uncapped: it scores all 2,141 current instances from all 133 statements in all 12
fixtures. Range and sample overrides are for focused diagnostics; they do not define the benchmark's
official exhaustive result.

The current controls are:

- `CPP_COMPLETION_BENCHMARK=1` enables the compiler-backed benchmark;
- `CPP_COMPLETION_START_INSTANCE` skips a zero-based prefix of the deterministic flattened corpus
  (default `0`);
- `CPP_COMPLETION_MAX_INSTANCES` limits how many remaining instances are scored (unset by default);
- `CPP_COMPLETION_SAMPLES_PER_INSTANCE` changes the per-instance precision-sample cap (default
  `100`, minimum `3` so every cursor can display three scored samples);
- `CPP_COMPLETION_TIME_LIMIT_MS` changes the internal deadline (default `60000`);
- `CPP_COMPLETION_COMPILER_JOBS` changes clang++ parallelism (by default, available processors minus
  two, clamped to `1..12`);
- `CLANGD` and `CXX` select the clangd and clang++ executables.

The full uncapped run must score exactly every discovered instance. It gates 100% recall, at least
99% aggregate precision, and at least 95% precision for every instance. Missing scores, empty CFGs,
generation failures, and deadline-truncated reports fail. Both the prepared semantic base and each
cursor residual have a 500 ms generation budget.

Benchmark mode selects only the compiler-backed sweep; the surrounding fast browser regressions
remain part of the ordinary `jsBrowserTest` suite and do not consume the sweep's one-minute budget.

The original three-fixture reference run drew 61,080 exact-length-stratified samples and recognized
all 740 ground-truth suffixes at 100% precision and recall. The expanded corpus intentionally adds
aliases and nested templates, associative and sequence containers, algorithms and lambdas, callable
pipelines, scoped-enum bitmasks, optionals and variants, raw/smart-pointer ownership, named casts,
RTTI, string/ranges transformations, structured bindings, and range-for statements. Reference
timing and scores are recorded only after an uncapped 2,141-instance run; focused slices are
diagnostic runs and are not presented as corpus-wide results.

## Corpus

Each fixture is a standalone translation unit and contributes every semicolon-ended statement at
every lexical boundary:

- `associative_records.cpp` — aliases, tuples, maps, sets, structured bindings, and iterators;
- `callable_pipeline.cpp` — abstract interfaces, stored `std::function` values, captured lambdas,
  and fluent calls;
- `container_algorithms.cpp` — sequence containers, iterators, classic algorithms, and predicates;
- `default_animals.cpp` — polymorphic ownership, constructors, virtual dispatch, and range-for;
- `enum_bitmask.cpp` — scoped enums, overloaded bitwise operators, casts, and conditionals;
- `fluent_routes.cpp` — builders, inheritance, smart pointers, generic lambdas, sorting, and streams;
- `optional_variants.cpp` — optional mutation, variant alternatives, typed queries, and visitation;
- `pointer_casts.cpp` — raw pointers, references, address-taking, abstract bases, and erased pointers;
- `polymorphic_casts.cpp` — named casts, RTTI, pointer-to-integer conversion, and virtual mutation;
- `raii_ownership.cpp` — unique/shared/weak ownership, moves, array specialization, and indexing;
- `shared_documents.cpp` — shared object graphs, arrow access, and long nested method chains;
- `string_transformations.cpp` — strings, string views, mutation, character conversion, and ranges.

The report groups cursor CFGs by their original source statement. For every evaluated lexical index
it prints the residual CFG's terminal, nonterminal, structural-production, and total-rule counts;
the suffix token bound; the exact arbitrary-precision derivation count over the explicitly reported
inspected length range; base and residual generation times; combined shortest-batch preparation
time; the exact-length sample histogram; compact clang-context counts; precision; and recall. The
count is labeled `totalDerivations` only when that range covers the full residual horizon, and
`inspectedDerivations` otherwise. It then prints up to three of that instance's scored samples as
complete source lines, choosing one representative from each shortest nonempty length slice before
filling from those slices. These are not separate showcase draws: they come from the same seeded
samples sent to clang++ for precision scoring. Empty or failed instances and the first rejected
sample diagnostic are reported explicitly. The summary also distinguishes logical compiler
candidates from alpha/token-deduplicated physical candidates and reports bundle and compiler time.

Benchmark mode suppresses raw browser-console protocol records on stdout, including Kotlin's
`--END_KOTLIN_TEST--` messages. Intentional test output still appears through Gradle's test reporter,
and the full browser console remains in `build/ci-logs/browser-console.log` for diagnosis.

## Pipeline

1. The pinned grammars-v4 `CPP14Lexer` produces token text and exact source spans through
   `lexCppTokens(String)` and `lexCppTokenSpans(String)`.
2. The test-only native bridge asks clangd for the recovery AST, scoped completions, receiver
   members, signatures, hover types, constructors, and inheritance conversions. It also extracts
   the translation unit's included header names. Independent scope and receiver probes preserve
   locals and member tables when `.`, `->`, or `::` filters the main completion request.
   On an exhaustive run it primes the first fixture's preamble while Karma launches Chrome, and
   immutable fixture resources are read once per Karma run. The first scored context and baseline
   compiler validation then overlap through the existing bounded native-worker pool.
3. Once per statement, the deletion-at-index-0 context is used to prepare a shared, depth-indexed
   semantic CFG. Identifiers are exact scoped terminals; calls are emitted only for assignable
   argument tuples; member access uses compatible receiver types; pointer, smart-pointer,
   constructor, stream, conditional, and arithmetic forms are type-specialized. Source headers are
   semantic evidence too—for example, `<algorithm>` keeps `std::sort` available even when the
   damaged line no longer spells it.
4. At every cursor index, a memoized CFG derivative computes the exact left quotient of that
   statement base by the tokens before the cursor. Prefix-introduced correlated binders, such as
   generic-lambda parameters, receive a prefix-sensitive prepared base. The resulting residual is
   therefore specific to the cursor while reusing the expensive statement facts where possible.
5. Galoisenne's `BoundedAcyclicCFG` performs recognition, arbitrary-precision derivation counting,
   and indexed uniform sampling. Kotlin/JS keeps its hot count vectors, ranks, and decoding weights
   in native exact `bigint` values; the public Galoisenne API remains `BigInteger`. Recognition-only
   CYK indexes are lazy, so count/sample-only residuals do not build them. Both the semantic base and
   every residual are acyclic and finite. Complete statements use an explicit epsilon residual;
   every other suffix is bounded so the complete statement never exceeds 48 projected tokens.
   Decoding choices are compiled once to grammar-local integer indexes, and compact CFG statistics
   come directly from the bounded grammar without populating unrelated global CFG caches.
6. The seeded sampler visits exact terminal-yield lengths in ascending order, skips empty slices,
   and draws up to 10 derivations uniformly with replacement from each slice until the default
   100-sample cap is reached. It counts only as far as needed to find those slices, reports that
   inspected range, and materializes one cached batch so repeated access cannot advance either the
   derivation RNG or fresh-name RNG. Cursor-local duplicates are compiled once and expanded back to
   their original multiplicity for scoring. Before compilation, full token sequences are deduplicated
   across cursor positions and genuinely fresh identifiers are alpha-normalized for the key; one
   real spelling is retained as the compiled representative. Candidate sites from several
   statements in the same fixture share block-isolated translation units, capped at 3,000
   candidates per shard; clang++ waves use a bounded worker pool and globally numbered `#line`
   markers keep diagnostics attributable to individual candidates. Diagnostic-only clang flags
   suppress carets, fix-its, and spell-check work while preserving compile acceptance.

The semantic grammar uses separate postfix and stable-operator tiers. This preserves valid raw
forms such as method chains and `i + 1 == times`, while parenthesizing forms whose C++ precedence
would otherwise change their inferred type. Recursive-looking chains are unrolled to depth 6, with
a 48-token complete-statement ceiling. At a prefix of length `p`, the residual horizon is at most
`48 - p` projected tokens.

In the production editor, Ctrl/⌘+Space invokes the grammar as the only user-visible C++ completion
provider; clangd's public completion items are suppressed. The grammar still issues private clangd
queries for semantic context. The provider runs only for that explicit keystroke, isolates the
current same-line statement at the caret, and requests at most ten source-distinct terminal
sequences in increasing exact token length. Ambiguous derivations and alpha-renamed fresh binders
are collapsed; when the global-minimum slice has fewer than ten useful forms, the sampler inspects
successive exact lengths until it fills the cap or reaches the finite horizon. Its edit range stops
at that statement's semicolon, an enclosing `}`, or a trailing comment, preserving neighboring
code. Semantic CFG construction and decoding
run in a dedicated worker; document-version and cursor checks discard stale replies, while active
LSP cancellation prevents obsolete context probes from queueing in WebAssembly clangd. Source
lexical facts and exact-caret AST reductions are cached independently, and a context epoch prevents
an in-flight request from repopulating the cache after diagnostics or text change. The browser path
uses source, scoped clangd completions, and a bounded reduction of clangd's recovery AST; it does
not call the benchmark's native compiler oracle. Like the benchmark grammar it completes at ANTLR
token boundaries within the 48-projected-token finite statement horizon.

`cpp_statements.tidy` documents the broader single-line coverage target: qualified/template names
with `<`, `>`, and `::`; pointers and references using `*`, `&`, and `&&`; calls and arbitrary
argument lists; `.`, `->`, indexing, and brace initialization; assignment, conditional, logical,
bitwise, comparison, shift, additive, and multiplicative operators; casts, `new`, `delete`,
`sizeof`, `alignof`, and declarations. It is a readable design skeleton, not a recursive runtime
fallback. Runtime productions are admitted only when the cursor's semantic facts make them safe.

## Scoring and terminals

Recall is the percentage of all evaluated ground-truth suffixes recognized by their cursor CFG.
Each instance contributes either 100% or 0% recall. Precision for an instance is the percentage of
its exact-length-stratified derivation draws whose completed translation units compile; aggregate
precision is the mean of those per-instance percentages. Within each length, ambiguous derivations
are counted separately, so the distribution is uniform over derivations and reproducible for a
seeded Kotlin `Random`. The default full run scores at most 100 samples for each instance: 10 from
each of the first ten nonempty exact-length slices. A CFG with fewer slices contributes fewer draws;
for example, the endpoint epsilon CFG contributes 10 length-zero draws. Duplicate source lines
remain separate draws in the precision denominator even though their compiler result is reused
within the statement bundle.

Literals are projected by type:

```text
42, 0xffU      -> @integer
3.14, 0x1.fp2 -> @floating
"abc"          -> @string
'\n'           -> @character
true, false    -> @boolean
nullptr        -> @nullptr
```

An identifier such as `cout`, `push_back`, or `Dog` is an exact terminal. `@fresh` is the sole
wildcard. For recall, each possible `@fresh` alignment is guarded by substituting a newly generated
identifier absent from the translation unit and compiling that modified ground-truth file. It only
counts when the substitution compiles.

## Layout

The production grammar and shortest-distinct sampler live under `src/jsMain/kotlin/cppcompletion/`.
The editor invokes them through the dedicated worker in `src/jsMain/kotlin/cpp/`, keeping semantic
CFG construction off Monaco's UI thread. The benchmark driver remains
`cppCompletion/kotlin/cppcompletion/CppCompletionBenchmarkTest.kt` and imports that same production
implementation.

The clangd/clang++ Karma bridge, declarative skeleton, and fixtures remain in the single
`src/jsTest/cppCompletion/` subtree. The reusable bounded-CFG implementation lives in Galoisenne;
the upstream lexer is pinned in `tidyparse-core/antlr/cpp/` as requested.
