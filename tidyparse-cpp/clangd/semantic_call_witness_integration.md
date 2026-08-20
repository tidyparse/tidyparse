# Compiler-validated correlated call witnesses

This is the integration contract for primary callable templates that cannot be
lowered from their dependent declaration pattern. The compile-checked Clang 21
API implementation lives in `semantic_call_witness_prototype.cpp`.

## Soundness boundary

Only a witness for which Clang has completed all of the following steps is
authoritative:

1. normal member/constructor lookup;
2. overload resolution and template argument deduction;
3. constraint checking;
4. selection of a concrete `FunctionDecl` specialization; and
5. recursive definition instantiation without an error diagnostic.

`requiresCompilerSubstitution` template schemas are passive candidate metadata.
They are never concrete operation nodes and cannot substitute for a witness.
In particular, overload viability alone accepts several invalid forwarding
calls in libc++ (`map::emplace`, `map::try_emplace`, and `optional::value_or`).
Their failures occur in an instantiated body.

## Response schema

Add `operations.callWitnesses`, `operations.callWitnessLimit`,
`operations.callWitnessMaxArity`, and `operations.callWitnessesIncomplete`.
Each witness is one indivisible argument vector; clients must not take a
Cartesian product of arguments from different witnesses.

```json
{
  "name": "try_emplace",
  "syntax": "memberCall",
  "validation": "recursiveDefinitionInstantiation",
  "authoritative": true,
  "primaryTemplateId": "...",
  "receiver": {
    "kind": "opaque",
    "valueCategory": "lvalue",
    "type": "std::map<int, Record>",
    "canonicalType": "...",
    "typeInfo": {}
  },
  "arguments": [
    {"kind": "integerZero", "valueCategory": "prvalue", "typeInfo": {}},
    {"kind": "integerZero", "valueCategory": "prvalue", "typeInfo": {}},
    {"kind": "emptyString", "valueCategory": "lvalue", "typeInfo": {}},
    {"kind": "floatingZero", "valueCategory": "prvalue", "typeInfo": {}}
  ],
  "callable": {},
  "result": {
    "kind": "opaque",
    "valueCategory": "prvalue",
    "typeInfo": {}
  }
}
```

Supported expression kinds initially match the projected grammar atoms:
`opaque`, `integerZero`, `floatingZero`, `characterZero`, `emptyString`,
`booleanTrue`, and `nullptr`. `emptyString` retains its exact
`const char[1]` lvalue profile; it must not be serialized merely as `char *`.

Constructor witnesses use `syntax: "parenConstruction"` or
`syntax: "listConstruction"`. They cannot be merged because initializer-list
preference and narrowing make `T(args...)` and `T{args...}` different
languages.

## Exact insertion points

### `CodeComplete.h`

After `SemanticCompletionCallableTemplate`, add:

- `SemanticCompletionExpressionProfile` (`Kind`, printed/canonical type,
  `TypeMetadata`, and `ValueCategory`);
- `SemanticCompletionCallWitness` (`Name`, `Syntax`, `Validation`,
  `PrimaryTemplateID`, receiver, whole argument vector, concrete callable,
  result, and `Authoritative`);
- `CodeCompleteOptions::SemanticCallWitnessLimit` and
  `SemanticCallWitnessMaxArity`; and
- the corresponding result vector/bounds/incomplete flag on
  `CodeCompleteResult`.

### `Protocol.h` / `Protocol.cpp`

Add optional non-negative `callWitnessLimit` and `callWitnessMaxArity` fields to
`SemanticCompletionParams` and parse them beside `operationLimit` and
`operationDepth`. Zero disables speculative instantiation.

### `ClangdLSPServer.cpp`

In `onSemanticCompletion`, capture both bounds and set them only on the scope
pass. In `semanticOperationsJSON`, serialize every correlated witness and its
nested profiles. `scopeIsIncomplete` and `operations.isIncomplete` must include
`SemanticCallWitnessesHaveMore`.

### `CodeComplete.cpp`

Add the probe helpers beside `semanticCallableTemplate`. The exact member path
is:

```cpp
LookupResult R(S, DeclarationName(&AST.Idents.get(Name)), Loc,
               Sema::LookupMemberName);
S.LookupQualifiedName(R, OwnerDefinition);
S.BuildMemberReferenceExpr(Receiver, OwnerType, Loc, false, CXXScopeSpec(),
                           SourceLocation(), nullptr, R, nullptr,
                           S.getCurScope());
S.BuildCallExpr(nullptr, Member.get(), Loc, Args, Loc, nullptr, false, false);
```

The constructor path is:

```cpp
S.BuildCXXTypeConstructExpr(AST.getTrivialTypeSourceInfo(Target, Loc), Loc,
                            Args, Loc, ListInitialization);
```

Extract the selected `FunctionDecl`, require its primary template to be one of
the emitted schemas, and then call:

```cpp
S.InstantiateFunctionDefinition(Loc, Callee,
                                /*Recursive=*/true,
                                /*DefinitionRequired=*/true,
                                /*AtEndOfTU=*/false);
```

The probe must be enclosed by all three guards below. `TentativeAnalysisScope`
alone does not suppress diagnostics from implicitly instantiated templates.

```cpp
DiagnosticErrorTrap Errors(S.getDiagnostics());
bool Old = S.getDiagnostics().getSuppressAllDiagnostics();
S.getDiagnostics().setSuppressAllDiagnostics(true);
auto Restore = llvm::make_scope_exit(
    [&] { S.getDiagnostics().setSuppressAllDiagnostics(Old); });
Sema::TentativeAnalysisScope Tentative(S);
```

`DiagnosticErrorTrap` still counts diagnostics while global emission is
suppressed. A witness is accepted only if the expression is valid, the trap saw
no error, the specialization is not invalid, and recursive definition
instantiation completed.

Collect accessible template families during the existing concrete-owner loop,
but run probes only after record traversal. Instantiation can append
specializations to declaration contexts, so probing inside `Record->decls()` is
unsafe. Group work by concrete owner canonical type plus member name and perform
lookup once per whole argument vector; associate the result with
`Callee->getPrimaryTemplate()->getCanonicalDecl()`.

The bounded seed universe consists of the exact projected literals, visible
value profiles, concrete function-result profiles, and concrete record/template
argument types already reached by the semantic operation graph. Enumerate whole
vectors breadth-first by arity, with literals first for stable behavior. For a
parameter pack, stopping at `SemanticCallWitnessMaxArity` always sets the
incomplete flag. Exhausting the attempt/output budget does the same. Never emit
an arbitrary repetition production for an unenumerated pack.

## Mutation and lifetime

`BuildCallExpr`, constructor initialization, and definition instantiation cache
specializations and can mark invalid declarations. Therefore the helper may run
only in the disposable scope-completion `Sema`; it must never run against a
shared preamble AST. Memoize owner/name/syntax/whole-profile vectors within that
one request.

