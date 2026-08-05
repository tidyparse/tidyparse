"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  batchDiagnosticClassification,
  canonicalRequiredType,
  ClangdLspSession,
  combinedTranslationUnit,
  exactIncludePreamble,
  hasActiveExpressionPrefix,
  NativeCompiler,
  requiredDeclarationTypes,
  requiredTypeCandidates
} = require("./benchmark-service");

test("preamble warmup is serialized through the clangd document queue", async () => {
  const session = new ClangdLspSession({
    clangd: "unused",
    clangxx: "unused",
    log: { debug() {} },
    workspace: "/unused"
  });
  const events = [];
  session.start = async () => { events.push("start"); };
  session.updateDocument = source => { events.push(`open:${source}`); };
  session.request = async (method, params) => {
    events.push(`${method}:${params.textDocument.uri}`);
  };

  await session.prime("int main() { return 0; }");

  assert.deepEqual(events, [
    "start",
    "open:int main() { return 0; }",
    "textDocument/documentSymbol:file:///unused/main.cpp"
  ]);
});

test("clangd operation deadlines start after the document queue wait", async () => {
  const session = new ClangdLspSession({
    clangd: "unused",
    clangxx: "unused",
    log: { debug() {} },
    workspace: "/unused"
  });
  let releaseQueue;
  session.queue = new Promise(resolve => { releaseQueue = resolve; });
  const observedDeadlines = [];
  session.start = async deadline => { observedDeadlines.push(deadline); };
  session.updateDocument = () => {};
  session.request = async () => {};

  const realNow = Date.now;
  let now = 1_000;
  Date.now = () => now;
  try {
    const primed = session.prime("int main() { return 0; }");
    now = 20_000;
    releaseQueue();
    await primed;
  } finally {
    Date.now = realNow;
  }

  assert.equal(observedDeadlines.length, 1);
  assert.ok(observedDeadlines[0] > 20_000);
});

test("blank completion lines have no active expression query", () => {
  assert.equal(hasActiveExpressionPrefix("int main() {\n"), false);
  assert.equal(hasActiveExpressionPrefix("int main() {\r\n    "), false);
  assert.equal(hasActiveExpressionPrefix("\t"), false);
});

test("partial statements retain active expression queries", () => {
  assert.equal(hasActiveExpressionPrefix("int main() {\n  animals.push_back("), true);
  assert.equal(hasActiveExpressionPrefix("  std::cout << animal."), true);
});

test("required type canonicalization permits callable templates but rejects declarators", () => {
  assert.equal(canonicalRequiredType("std::function<int(int)>"), "std::function<int(int)>");
  assert.equal(
    canonicalRequiredType("std::vector<std::function<int(int)>>"),
    "std::vector<std::function<int(int)>>"
  );
  assert.equal(canonicalRequiredType("int(int)"), undefined);
  assert.equal(canonicalRequiredType("int (*)(int)"), undefined);
});

test("required type canonicalization accepts only template-nested unsized arrays", () => {
  assert.equal(canonicalRequiredType("std::unique_ptr<int[]>"), "std::unique_ptr<int[]>");
  assert.equal(
    canonicalRequiredType("std::unique_ptr<int [ ]>"),
    "std::unique_ptr<int []>"
  );
  assert.equal(canonicalRequiredType("int[]"), undefined);
  assert.equal(canonicalRequiredType("int[] &"), undefined);
  assert.equal(canonicalRequiredType("std::unique_ptr<int[3]>"), undefined);
});

test("required type candidates ignore unrelated completion vocabulary", () => {
  const { candidates } = requiredTypeCandidates("struct Item {};", {
    identifiers: ["make_shared", "shared_ptr", "string", "vector"],
    types: [{ name: "Item", kind: "class", source: "ast" }],
    conversions: [],
    values: [],
    expectedTypes: []
  });

  assert.equal(candidates.includes("std::string"), false);
  assert.equal(candidates.some(type => type.startsWith("std::shared_ptr<")), false);
  assert.equal(candidates.some(type => type.startsWith("std::vector<")), false);
});

test("required declaration probes use compiler priority", async () => {
  const observed = [];
  const compiler = {
    compileAll: async (sources, priority) => {
      observed.push({ sources, priority });
      return sources.map(() => ({
        ok: false,
        timedOut: false,
        diagnostics: "cpp_completion_0.cpp:3: error: rejected probe"
      }));
    }
  };
  await requiredDeclarationTypes(
    "struct Item {};\nint main() {\n  \n  return item == item;\n}\n",
    2,
    2,
    {
      requiredIdentifier: "item",
      identifiers: ["Item", "item"],
      sourceIdentifiers: ["Item", "item"],
      types: [{ name: "Item", type: "Item", kind: "struct", source: "ast" }],
      conversions: [],
      values: [],
      expectedTypes: []
    },
    compiler
  );

  assert.equal(observed.length, 1);
  assert.equal(observed[0].priority, true);
  assert.ok(observed[0].sources.length > 0);
});

test("candidate diagnostics remain attributable without caret output", () => {
  const diagnostics = [
    "cpp_completion_1_candidate_7.cpp:9: error: use of undeclared identifier 'bad'",
    "cpp_completion_0_baseline.cpp:4:1: note: instantiated here"
  ].join("\n");
  const classified = batchDiagnosticClassification(diagnostics, 2, 1, null, null);

  assert.deepEqual([...classified.failed], [1]);
  assert.deepEqual(classified.globalErrors, []);
  assert.match(classified.perSource[1][0], /candidate_7/);
  assert.match(classified.perSource[0][0], /baseline/);
});

test("combined translation units preserve candidate filenames across source indexes", () => {
  const unit = combinedTranslationUnit([
    "int main() {\n#line 1 \"__TIDYPARSE_CPP_COMPLETION_BUNDLE___candidate_0.cpp\"\nreturn missing;\n}",
    "int main() { return 0; }"
  ]);

  assert.match(unit, /cpp_completion_0_candidate_0\.cpp/);
  assert.match(unit, /namespace tidyparse_cpp_completion_sample_1/);
  assert.match(unit, /#define main tidyparse_cpp_completion_main_1/);
});

test("PCH keys preserve the exact ordered header sequence", () => {
  const first = exactIncludePreamble([
    "#include <vector>\n#include <string>\nint first;",
    "#include <vector>\nint second;"
  ]);
  const same = exactIncludePreamble([
    "#include <vector>\n#include <string>\nint different_body;"
  ]);
  const reordered = exactIncludePreamble([
    "#include <string>\n#include <vector>\nint first;"
  ]);
  const superset = exactIncludePreamble([
    "#include <vector>\n#include <string>\n#include <map>\nint first;"
  ]);

  assert.deepEqual(first.directives, ["#include <vector>", "#include <string>"]);
  assert.equal(first.key, same.key);
  assert.notEqual(first.key, reordered.key);
  assert.notEqual(first.key, superset.key);
  assert.equal(first.source, "#include <vector>\n#include <string>\n");
});

test("a PCH-backed translation unit omits only its substituted includes", () => {
  const source = [
    "#include <vector>",
    "int main() {",
    "#line 1 \"__TIDYPARSE_CPP_COMPLETION_BUNDLE___candidate_0.cpp\"",
    "return missing;",
    "}"
  ].join("\n");
  const unit = combinedTranslationUnit([source], true);

  assert.doesNotMatch(unit, /#include\s*<vector>/);
  assert.match(unit, /cpp_completion_0_candidate_0\.cpp/);
  assert.match(unit, /return missing;/);
  assert.match(unit, /namespace tidyparse_cpp_completion_sample_0/);
});

test("exact-header PCH builds are coalesced but never shared with a different sequence", async () => {
  const compiler = new NativeCompiler("unused", 2, { pchDirectory: "/unused/pch" });
  const builds = [];
  compiler.buildPch = async preamble => {
    builds.push(preamble);
    return { key: preamble.key, path: `/unused/pch/${preamble.key}.pch` };
  };
  const vector = "#include <vector>\nint main() {}";
  const string = "#include <string>\nint main() {}";

  const [left, right] = await Promise.all([
    compiler.prepareExactPreamble([vector], true),
    compiler.prepareExactPreamble([vector], true)
  ]);
  const different = await compiler.prepareExactPreamble([string], true);

  assert.equal(builds.length, 2);
  assert.equal(left.path, right.path);
  assert.notEqual(left.key, different.key);
});

test("an unusable PCH retries the original include-bearing compiler path", async () => {
  const compiler = new NativeCompiler("unused", 1);
  const preamble = { key: "exact", path: "/unused/exact.pch" };
  compiler.prepareExactPreamble = async () => preamble;
  const calls = [];
  compiler.compileBatchUnbounded = async (sources, selectedPreamble) => {
    calls.push(selectedPreamble);
    return sources.map(() => selectedPreamble == null
      ? { result: { ok: true, timedOut: false, diagnostics: "" }, cacheable: true }
      : { result: { ok: false, timedOut: false, diagnostics: "invalid PCH" }, cacheable: false });
  };

  const outcomes = await compiler.compileBatch(["#include <vector>\nint main() {}"]);

  assert.deepEqual(calls, [preamble, null]);
  assert.equal(outcomes[0].result.ok, true);
  assert.equal(await compiler.pchCache.get("exact"), null);
});

test("compiler close removes its owned PCH cache directory", async () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-pch-test-"));
  const pchDirectory = path.join(temporary, "pch");
  fs.mkdirSync(pchDirectory);
  fs.writeFileSync(path.join(pchDirectory, "cached.pch"), "test");
  const compiler = new NativeCompiler("unused", 1, { pchDirectory });

  await compiler.close();

  assert.equal(fs.existsSync(pchDirectory), false);
  fs.rmSync(temporary, { recursive: true, force: true });
});

test("declaration-oracle compilation can use the permit reserved from scoring", async () => {
  const compiler = new NativeCompiler("unused", 2);
  const started = [];
  const pending = new Map();
  compiler.compileBatchUnbounded = sources => new Promise(resolve => {
    const source = sources[0];
    started.push(source);
    pending.set(source, () => resolve(sources.map(() => ({
      result: { ok: true, timedOut: false, diagnostics: "" },
      cacheable: true
    }))));
  });
  const turn = () => new Promise(resolve => setImmediate(resolve));

  const scoring = compiler.compileAll(["score-a", "score-b"]);
  await turn();
  assert.deepEqual(started, ["score-a"]);

  const oracle = compiler.compileAll(["oracle"], true);
  await turn();
  assert.deepEqual(started, ["score-a", "oracle"]);

  const secondOracle = compiler.compileAll(["oracle-b"], true);
  await turn();
  assert.deepEqual(started, ["score-a", "oracle"], "priority work is confined to its reserved lane");

  pending.get("oracle")();
  await oracle;
  await turn();
  assert.deepEqual(started, ["score-a", "oracle", "oracle-b"]);
  pending.get("oracle-b")();
  await secondOracle;
  pending.get("score-a")();
  await turn();
  assert.deepEqual(started, ["score-a", "oracle", "oracle-b", "score-b"]);
  pending.get("score-b")();
  await scoring;
  assert.equal(compiler.activeCompilers, 0);
  assert.equal(compiler.activeNormalCompilers, 0);
  assert.equal(compiler.activePriorityCompilers, 0);
});
