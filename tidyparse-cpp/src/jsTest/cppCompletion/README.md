# C++ completion benchmark

The browser editor and this benchmark build a finite, cursor-specific C++ statement CFG from
clangd facts. Every `.cpp` file in `resources/cpp-completion/` is discovered and compiled intact before scoring. Semicolon-ended statement lines are then truncated at every ANTLR token boundary and completed using only forward continuations. The committed corpus currently contains 133 statements and 2,141 completion instances: for each statement with `n` tokens, indexes `0..n` are all included, including the empty prefix and the already-complete line.

## Run

```text
CPP_COMPLETION_BENCHMARK=1 ./gradlew :tidyparse-cpp:jsBrowserTest \
  --tests 'cppcompletion.CppCompletionBenchmarkTest.benchmarkCppCompletions'
```

The default run is uncapped: it scores all 2,141 current instances from all 133 statements in all 12 fixtures. Range and sample overrides are for focused diagnostics; they do not define the benchmark's official exhaustive result.

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
- `CLANGD` and `CXX` select the legacy native bridge executables in ordinary test mode. Benchmark
  mode accepts `CXX` only with a matching `CPP_COMPLETION_COMPILER_PROFILE` sidecar.

### Pinned semantic authority

The scored benchmark fails closed unless browser Sema and candidate validation have the same content-derived semantic profile: upstream Clang revision, C++ mode, wasm32-wasi target, ordered flags, libc++/WASI identity, and logical include-tree digest. It never filters target-specific names or silently falls back to the host SDK.

`refreshClangdResources` builds the browser module and a native syntax validator from the same patched LLVM checkout. The native build registers only the WebAssembly target and is retained under `.gradle/clangd/<artifact>-<host>/work/build-native/bin/clang++`; its profile sidecar is `work/native-validator-profile.json`, and its byte-identical browser include tree is `work/browser-sysroot/include`. These native files are benchmark/build-cache inputs and are not added to the deployed browser payload.

```text
./gradlew :tidyparse-cpp:refreshClangdResources
```

An old browser artifact without this sidecar, a stale manifest, or an arbitrary system `clang++` causes benchmark startup to report the profile mismatch before any completion is scored. Ordinary non-benchmark bridge tests retain their existing host-compiler behavior.

The full uncapped run must score exactly every discovered instance. It gates 100% recall, at least 99% aggregate precision, and at least 95% precision for every instance. Missing scores, empty CFGs, generation failures, and deadline-truncated reports fail. Both the prepared semantic base and each cursor residual have a 500 ms generation budget.

Benchmark mode selects only the compiler-backed sweep; the surrounding fast browser regressions remain part of the ordinary `jsBrowserTest` suite and do not consume the sweep's one-minute budget.

The original three-fixture reference run drew 61,080 exact-length-stratified samples and recognized all 740 ground-truth suffixes at 100% precision and recall. The expanded corpus intentionally adds aliases and nested templates, associative and sequence containers, algorithms and lambdas, callable pipelines, scoped-enum bitmasks, optionals and variants, raw/smart-pointer ownership, named casts, RTTI, string/ranges transformations, structured bindings, and range-for statements. Reference timing and scores are recorded only after an uncapped 2,141-instance run; focused slices are diagnostic runs and are not presented as corpus-wide results.

## Corpus

Each fixture is a standalone translation unit and contributes every semicolon-ended statement at every lexical boundary:

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

The report groups cursor CFGs by their original source statement. For every evaluated lexical index it prints the residual CFG's terminal, nonterminal, structural-production, and total-rule counts; the suffix token bound; the exact arbitrary-precision derivation count over the explicitly reported inspected length range; base and residual generation times; combined shortest-batch preparation time; the exact-length sample histogram; compact clang-context counts; precision; and recall. The count is labeled `totalDerivations` only when that range covers the full residual horizon, and `inspectedDerivations` otherwise. It then prints up to three of that instance's scored samples as complete source lines, choosing one representative from each shortest nonempty length slice before filling from those slices. These are not separate showcase draws: they come from the same seeded samples sent to clang++ for precision scoring. Empty or failed instances and the first rejected sample diagnostic are reported explicitly. The summary also distinguishes logical compiler candidates from alpha/token-deduplicated physical candidates and reports candidate-preparation and compiler time.

Benchmark mode suppresses raw browser-console protocol records on stdout, including Kotlin's `--END_KOTLIN_TEST--` messages. Intentional test output still appears through Gradle's test reporter, and the full browser console remains in `build/ci-logs/browser-console.log` for diagnosis.

## Pipeline

1. The pinned grammars-v4 `CPP14Lexer` produces token text and exact source spans through `lexCppTokens(String)` and `lexCppTokenSpans(String)`.
2. The bundled clangd extension answers one `tidyparse/semanticCompletion` request at the exact caret, with a second position identifying the statement's surrounding scope. The response keeps the declarations and active call overloads discovered by that completion Sema alive long enough to serialize their canonical type identities, parameters, ownership, and cv/ref structure. Sema declarations are the sole authority for typed CFG edges. The clangd index augments the accessible name inventory, but its display signatures and return types never become type facts.
3. Once per statement, those declaration records prepare a shared, depth-indexed semantic CFG. Identifiers are exact terminals copied from accessible declaration and insertion names. Type spellings enter the inventory only when their Sema `CppTypeInfo` is concrete and source-spellable; dependent and unspellable implementation types remain opaque. Calls are emitted only for assignable argument tuples, and member access uses compatible receiver types. There is no hardcoded standard-library or project-name catalog: library and user symbols are available only when clang reports them in the current translation unit and scope.
4. At every cursor index, a memoized CFG derivative computes the exact left quotient of that statement base by the tokens before the cursor. Prefix-introduced correlated binders, such as generic-lambda parameters, receive a prefix-sensitive prepared base. The resulting residual is therefore specific to the cursor while reusing the expensive statement facts where possible.
5. The local `BoundedAcyclicCFG` performs recognition, arbitrary-precision derivation counting, and indexed uniform sampling. Kotlin/JS keeps its hot count vectors, ranks, and decoding weights in native exact `bigint` values; the public API remains `BigInteger`. Recognition-only CYK indexes are lazy, so count/sample-only residuals do not build them. Both the semantic base and every semantic residual are acyclic and finite. Complete statements use an explicit epsilon residual; every other semantic suffix is bounded so it never exceeds 48 projected tokens. Decoding choices are compiled once to grammar-local integer indexes, and compact CFG statistics come directly from the bounded grammar without populating unrelated global CFG caches.
6. The seeded sampler visits exact terminal-yield lengths in ascending order, skips empty slices, and draws up to 10 derivations uniformly with replacement from each slice until the default 100-sample cap is reached. It counts only as far as needed to find those slices, reports that inspected range, and materializes one cached batch so repeated access cannot advance either the derivation RNG or fresh-name RNG. Cursor-local duplicates are compiled once and expanded back to their original multiplicity for scoring. Before compilation, full token sequences are deduplicated across cursor positions and genuinely fresh identifiers are alpha-normalized for the key; one real spelling is retained as the compiled representative. Candidate sites from several statements in the same fixture share block-isolated translation units, capped at 3,000 candidates per shard; clang++ waves use a bounded worker pool and globally numbered `#line` markers keep diagnostics attributable to individual candidates. Diagnostic-only clang flags suppress carets, fix-its, and spell-check work while preserving compile acceptance.

The semantic grammar uses separate postfix and stable-operator tiers. This preserves valid raw forms such as method chains and `i + 1 == times`, while parenthesizing forms whose C++ precedence would otherwise change their inferred type. Recursive-looking chains are unrolled to depth 6, with a 48-token complete-statement ceiling. At a prefix of length `p`, the residual horizon is at most `48 - p` projected tokens.

In the production editor, Ctrl/⌘+Space invokes the grammar as the only user-visible C++ completion provider; clangd's public completion items are suppressed. The provider makes the same single, cancellable `tidyparse/semanticCompletion` request used by the benchmark, only for that explicit keystroke. It isolates the current same-line statement at the caret and requests at most ten source-distinct terminal sequences in increasing exact token length. Ambiguous derivations and alpha-renamed fresh binders are collapsed; when the global-minimum slice has fewer than ten useful forms, the sampler inspects successive exact lengths until it fills the cap or reaches the finite horizon. Its edit range stops at that statement's semicolon, an enclosing `}`, or a trailing comment, preserving neighboring code. Semantic CFG construction and decoding run in a dedicated worker; document-version and cursor checks discard stale replies, while active LSP cancellation prevents obsolete semantic requests from queueing in WebAssembly clangd. The browser path consumes the structured declaration DTO directly and does not call the benchmark's native compiler oracle. The semantic lane uses the 48-token finite horizon. The generated full-statement syntax grammar remains a separate recognition/totality oracle; its untyped derivations are never published as editor suggestions. Dependent class-template declarations are instead instantiated from Sema-reported parameter roles, defaults, and accessible type spellings.

`cpp_statements.tidy` documents the broader single-line coverage target: qualified/template names with `<`, `>`, and `::`; pointers and references using `*`, `&`, and `&&`; calls and arbitrary argument lists; `.`, `->`, indexing, and brace initialization; assignment, conditional, logical, bitwise, comparison, shift, additive, and multiplicative operators; casts, `new`, `delete`, `sizeof`, `alignof`, and declarations. It is a readable design skeleton, not a recursive runtime fallback. Runtime productions are admitted only when the cursor's semantic facts make them safe.

## Scoring and terminals

Recall is the percentage of all evaluated ground-truth suffixes recognized by their cursor CFG. Each instance contributes either 100% or 0% recall. Precision for an instance is the percentage of its exact-length-stratified derivation draws whose completed translation units compile; aggregate precision is the mean of those per-instance percentages. Within each length, ambiguous derivations are counted separately, so the distribution is uniform over derivations and reproducible for a seeded Kotlin `Random`. The default full run scores at most 100 samples for each instance: 10 from each of the first ten nonempty exact-length slices. A CFG with fewer slices contributes fewer draws; for example, the endpoint epsilon CFG contributes 10 length-zero draws. Every unique candidate is compiled as an independent full-source replacement so its downstream uses and declaration order remain observable. Duplicate source lines remain separate draws in the precision denominator even when the exact full-source compiler result is reused from the content-addressed cache.

Literals are projected by type:

```text
42, 0xffU      -> @integer
3.14, 0x1.fp2 -> @floating
"abc"          -> @string
'\n'           -> @character
true, false    -> @boolean
nullptr        -> @nullptr
```

An accessible identifier such as `ambientFlux`, `scanBands`, or `Widget` is an exact terminal. These names come from the cursor's clang response rather than a grammar-side catalog. `@fresh` is the sole wildcard. For recall, each possible `@fresh` alignment is guarded by substituting a newly generated identifier absent from the translation unit and compiling that modified ground-truth file. It only counts when the substitution compiles.

## Layout

The production grammar and shortest-distinct sampler live under `src/jsMain/kotlin/cppcompletion/`. The editor invokes them through the dedicated worker in `src/jsMain/kotlin/cpp/`, keeping semantic CFG construction off Monaco's UI thread. The benchmark driver remains `cppCompletion/kotlin/cppcompletion/CppCompletionBenchmarkTest.kt` and imports that same production implementation.

The clangd/clang++ Karma bridge, declarative skeleton, and fixtures remain in the single `src/jsTest/cppCompletion/` subtree. The reusable bounded-CFG implementation lives in `src/commonMain/kotlin/ai/hypergraph/kaliningraph/parsing/acyclic/`; the upstream lexer is pinned in `tidyparse-cpp/antlr/cpp/`.
