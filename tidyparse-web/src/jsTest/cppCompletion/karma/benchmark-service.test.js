"use strict";

const assert = require("node:assert/strict");
const childProcess = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  batchDiagnosticClassification,
  canonicalJson,
  canonicalRequiredType,
  ClangdLspSession,
  combinedTranslationUnit,
  compilerRequiredBinderEvidence,
  directoryTreeSha256,
  exactIncludePreamble,
  hasActiveExpressionPrefix,
  NativeCompiler,
  nativeCompilerEnvironment,
  nativeCompilerSemanticArguments,
  nativeSemanticFlags,
  requiredBinderObligation,
  requiredDeclarationFacts,
  requiredDeclarationTypes,
  requiredTypeCandidates,
  resolvePinnedNativeValidator,
  semanticProfileSha256,
  validateBrowserClangdAssets,
  validatePinnedNativeValidator
} = require("./benchmark-service");

function availableClangxx() {
  const candidates = [
    process.env.CXX,
    "/usr/bin/clang++",
    "/opt/homebrew/opt/llvm/bin/clang++",
    "/usr/local/bin/clang++"
  ].filter(candidate => typeof candidate === "string" && path.isAbsolute(candidate));
  if (process.platform !== "win32") {
    const located = childProcess.spawnSync("sh", ["-c", "command -v clang++"], {
      encoding: "utf8"
    });
    if (located.status === 0 && located.stdout.trim().length > 0) {
      candidates.push(located.stdout.trim());
    }
  }
  for (const candidate of [...new Set(candidates)]) {
    const version = childProcess.spawnSync(candidate, ["--version"], {
      encoding: "utf8",
      timeout: 5_000
    });
    if (version.status === 0 && /\bclang version\b/i.test(version.stdout || version.stderr || "")) {
      return candidate;
    }
  }
  return null;
}

function testSemanticProfile(headerTreeSha256 = "headers") {
  return {
    schemaVersion: 1,
    language: "c++",
    standard: "c++23",
    target: "wasm32-wasi",
    flags: [
      "-xc++",
      "-std=c++23",
      "--target=wasm32-wasi",
      "-nostdinc",
      "-nostdinc++",
      "-isystem/usr/include/wasm32-wasi/c++/v1",
      "-isystem/usr/include",
      "-isystem/usr/include/wasm32-wasi"
    ],
    frontend: { kind: "upstream-clang", version: "21.1.0", commit: "pinned" },
    headers: { logicalRoot: "/usr/include", treeSha256: headerTreeSha256 }
  };
}

test("semantic profile hashes are canonical and order independent", () => {
  const left = { target: "wasm32-wasi", flags: ["-xc++", "-std=c++23"] };
  const right = { flags: ["-xc++", "-std=c++23"], target: "wasm32-wasi" };

  assert.equal(canonicalJson(left), canonicalJson(right));
  assert.equal(semanticProfileSha256(left), semanticProfileSha256(right));
});

test("PCH and syntax validation share the exact physical wasm32-wasi semantic flags", () => {
  const includeRoot = path.resolve("/validator/include");
  const flags = nativeSemanticFlags(testSemanticProfile(), includeRoot);
  const pch = nativeCompilerSemanticArguments("-xc++-header", flags, false);
  const syntax = nativeCompilerSemanticArguments("-xc++", flags, false);

  assert.deepEqual(pch.slice(1), syntax.slice(1));
  assert.ok(flags.includes("--target=wasm32-wasi"));
  assert.ok(flags.includes("-nostdinc"));
  assert.ok(flags.includes(`-isystem${path.join(includeRoot, "wasm32-wasi", "c++", "v1")}`));
  assert.equal(flags.some(flag => flag === "-w"), false);
});

test("pinned compiler environment retains runtime essentials but removes host semantics", () => {
  const inherited = {
    PATH: "/host/bin",
    HOME: "/host/home",
    TMPDIR: "/host/tmp",
    SystemRoot: "C:\\Windows",
    CPATH: "/host/include",
    cplus_include_path: "/host/cxx",
    CCC_OVERRIDE_OPTIONS: "+-include +-host.hpp",
    CLANG_CONFIG_PATH: "/host/clang.cfg",
    CLANG_NO_DEFAULT_CONFIG: "",
    LANG: "fr_FR.UTF-8",
    LC_ALL: "de_DE.UTF-8"
  };

  const sanitized = nativeCompilerEnvironment(inherited);

  assert.deepEqual(sanitized, {
    TMPDIR: "/host/tmp",
    SystemRoot: "C:\\Windows",
    LANG: "C",
    LC_ALL: "C",
    LC_CTYPE: "C",
    CLANG_NO_DEFAULT_CONFIG: "1"
  });
  assert.equal(inherited.CPATH, "/host/include", "sanitizing must not mutate process.env");
});

test("pinned syntax validation ignores a CPATH header that shadows its profiled tree", async t => {
  const clangxx = availableClangxx();
  if (clangxx == null) {
    t.skip("clang++ is unavailable");
    return;
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-cpath-profile-"));
  const profiled = path.join(temporary, "profiled");
  const poison = path.join(temporary, "poison");
  const workspace = path.join(temporary, "workspace");
  fs.mkdirSync(profiled);
  fs.mkdirSync(poison);
  fs.mkdirSync(workspace);
  fs.writeFileSync(
    path.join(profiled, "tidyparse-semantic-profile-probe.hpp"),
    "#define TIDYPARSE_PROFILE_HEADER 1\n"
  );
  fs.writeFileSync(
    path.join(poison, "tidyparse-semantic-profile-probe.hpp"),
    "#error CPATH shadowed the profiled header tree\n"
  );
  const source = [
    "#include <tidyparse-semantic-profile-probe.hpp>",
    "static_assert(TIDYPARSE_PROFILE_HEADER == 1);"
  ].join("\n");
  const semanticFlags = [
    "-nostdinc",
    "-nostdinc++",
    `-isystem${profiled}`
  ];
  const cleanEnvironment = nativeCompilerEnvironment(process.env);
  const leaked = new NativeCompiler(clangxx, 1, {
    semanticFlags,
    suppressWarnings: false,
    environment: { ...cleanEnvironment, CPATH: poison },
    workingDirectory: workspace
  });
  const pinned = new NativeCompiler(clangxx, 1, {
    semanticFlags,
    suppressWarnings: false,
    environment: nativeCompilerEnvironment({ ...process.env, CPATH: poison }),
    workingDirectory: workspace
  });
  try {
    const leakedResult = await leaked.compile(source);
    const pinnedResult = await pinned.compile(source);
    assert.equal(leakedResult.ok, false, "the poison fixture must shadow without sanitization");
    assert.equal(pinnedResult.ok, true, pinnedResult.diagnostics);
  } finally {
    await Promise.all([leaked.close(), pinned.close()]);
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test("pinned syntax validation cannot resolve a caller-only quoted header", async t => {
  const clangxx = availableClangxx();
  if (clangxx == null) {
    t.skip("clang++ is unavailable");
    return;
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-cwd-profile-"));
  const caller = path.join(temporary, "caller");
  const workspace = path.join(temporary, "workspace");
  fs.mkdirSync(caller);
  fs.mkdirSync(workspace);
  fs.writeFileSync(
    path.join(caller, "tidyparse-cwd-probe.hpp"),
    "#define TIDYPARSE_CALLER_HEADER 1\n"
  );
  const source = [
    "#include \"tidyparse-cwd-probe.hpp\"",
    "static_assert(TIDYPARSE_CALLER_HEADER == 1);"
  ].join("\n");
  const options = {
    semanticFlags: ["-nostdinc", "-nostdinc++"],
    suppressWarnings: false,
    environment: nativeCompilerEnvironment(process.env)
  };
  const callerCompiler = new NativeCompiler(clangxx, 1, {
    ...options,
    workingDirectory: caller
  });
  const pinned = new NativeCompiler(clangxx, 1, {
    ...options,
    workingDirectory: workspace
  });
  try {
    const callerResult = await callerCompiler.compile(source);
    const pinnedResult = await pinned.compile(source);
    assert.equal(callerResult.ok, true, callerResult.diagnostics);
    assert.equal(pinnedResult.ok, false, "the isolated validator must not see caller files");
    assert.match(pinnedResult.diagnostics, /tidyparse-cwd-probe\.hpp/);
  } finally {
    await Promise.all([callerCompiler.close(), pinned.close()]);
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test("browser clangd artifacts accept exact bytes and fail closed when their manifest is stale", () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-clangd-assets-"));
  try {
    const worker = path.join(temporary, "worker.js");
    const module = path.join(temporary, "clangd.js");
    const wasm = path.join(temporary, "clangd.wasm.gz");
    const manifestPath = path.join(temporary, "clangd-manifest.json");
    fs.writeFileSync(worker, "const artifact = 'profile-test';");
    fs.writeFileSync(module, "module");
    fs.writeFileSync(wasm, "wasm");
    const profile = testSemanticProfile();
    fs.writeFileSync(manifestPath, JSON.stringify({
      artifactVersion: "profile-test",
      semanticProfile: profile,
      semanticProfileSha256: semanticProfileSha256(profile),
      artifacts: {
        "clangd.js": {
          bytes: fs.statSync(module).size,
          sha256: crypto.createHash("sha256")
            .update(fs.readFileSync(module)).digest("hex")
        },
        "clangd.wasm": {
          compressedBytes: fs.statSync(wasm).size,
          compressedSha256: crypto.createHash("sha256")
            .update(fs.readFileSync(wasm)).digest("hex")
        }
      }
    }));

    const aligned = validateBrowserClangdAssets({ worker, module, wasm, manifest: manifestPath });
    assert.equal(aligned.error, null);
    assert.equal(aligned.profileId, semanticProfileSha256(profile));

    fs.appendFileSync(wasm, "stale");
    const result = validateBrowserClangdAssets({ worker, module, wasm, manifest: manifestPath });

    assert.match(result.error, /does not match the browser clangd manifest/);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test("include-tree digest uses global logical-path order", () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-header-digest-"));
  try {
    fs.mkdirSync(path.join(temporary, "a"));
    fs.writeFileSync(path.join(temporary, "a", "z"), "nested");
    fs.writeFileSync(path.join(temporary, "a.txt"), "sibling");
    const digest = crypto.createHash("sha256");
    for (const relative of ["a.txt", "a/z"]) {
      const payload = crypto.createHash("sha256")
        .update(fs.readFileSync(path.join(temporary, ...relative.split("/"))))
        .digest();
      digest.update(relative);
      digest.update(Buffer.from([0]));
      digest.update("F");
      digest.update(Buffer.from([0]));
      digest.update(payload);
    }

    assert.equal(directoryTreeSha256(temporary), digest.digest("hex"));
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test("profile handshake accepts exact authority and rejects a target-divergent authority", () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-validator-profile-"));
  try {
    const gradleHome = path.join(temporary, "gradle-home");
    const workRoot = path.join(
      gradleHome, "tidyparse-clangd", "profile-test-host", "work"
    );
    const includeRoot = path.join(workRoot, "browser-sysroot", "include");
    const compiler = path.join(workRoot, "build-native", "bin", "clang");
    const sidecarPath = path.join(workRoot, "native-validator-profile.json");
    fs.mkdirSync(path.dirname(compiler), { recursive: true });
    fs.mkdirSync(path.join(includeRoot, "wasm32-wasi"), { recursive: true });
    fs.writeFileSync(compiler, "pinned clang 21");
    fs.writeFileSync(path.join(includeRoot, "wasm32-wasi", "stdio.h"), "typedef int wasi;");
    const headerTree = directoryTreeSha256(includeRoot);
    const browserProfile = testSemanticProfile(headerTree);
    const compilerSha256 = crypto.createHash("sha256")
      .update(fs.readFileSync(compiler)).digest("hex");
    const sidecar = profile => ({
      schemaVersion: 1,
      semanticProfile: profile,
      semanticProfileSha256: semanticProfileSha256(profile),
      compiler: { workRelativePath: "build-native/bin/clang", sha256: compilerSha256 },
      includeRoot: {
        workRelativePath: "browser-sysroot/include",
        treeSha256: headerTree
      }
    });
    fs.writeFileSync(sidecarPath, JSON.stringify(sidecar(browserProfile)));
    const browser = {
      error: null,
      profile: browserProfile,
      profileId: semanticProfileSha256(browserProfile),
      manifest: {
        artifactVersion: "profile-test",
        nativeValidator: {
          profileSha256: crypto.createHash("sha256")
            .update(fs.readFileSync(sidecarPath)).digest("hex"),
          compilerSha256,
          compilerWorkRelativePath: "build-native/bin/clang",
          includeRootWorkRelativePath: "browser-sysroot/include"
        }
      }
    };

    const aligned = validatePinnedNativeValidator(browser, sidecarPath);
    assert.equal(aligned.error, null);
    assert.equal(aligned.profileId, browser.profileId);
    assert.ok(aligned.semanticFlags.includes("--target=wasm32-wasi"));
    const discovered = resolvePinnedNativeValidator(
      path.join(temporary, "unrelated-repository"), browser, { GRADLE_USER_HOME: gradleHome }
    );
    assert.equal(discovered.error, null);
    assert.equal(discovered.compiler, compiler);

    const darwinProfile = {
      ...browserProfile,
      target: "arm64-apple-darwin",
      flags: browserProfile.flags.map(flag =>
        flag === "--target=wasm32-wasi" ? "--target=arm64-apple-darwin" : flag
      )
    };
    fs.writeFileSync(sidecarPath, JSON.stringify(sidecar(darwinProfile)));
    browser.manifest.nativeValidator.profileSha256 = crypto.createHash("sha256")
      .update(fs.readFileSync(sidecarPath)).digest("hex");
    const divergent = validatePinnedNativeValidator(browser, sidecarPath);

    assert.match(divergent.error, /semantic profiles differ/);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

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

test("required type candidates reserve source and fundamental lanes before graph noise", () => {
  const noisyTypes = Array.from({ length: 160 }, (_, index) => ({
    name: `sdk::Noise${index}`,
    type: `sdk::Noise${index}`,
    kind: "class",
    source: "graph"
  }));
  const { candidates } = requiredTypeCandidates("struct Local {};\n", {
    types: noisyTypes,
    conversions: [],
    values: noisyTypes.map(type => ({ type: type.type })),
    expectedTypes: []
  });

  assert.ok(candidates.includes("Local"));
  assert.ok(candidates.includes("Local *"));
  assert.ok(candidates.includes("const Local *"));
  assert.ok(candidates.includes("bool"));
  assert.ok(candidates.includes("short"));
  assert.ok(candidates.includes("double"));
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

test("required binder evidence preserves unknown versus proven empty", () => {
  assert.deepEqual(
    compilerRequiredBinderEvidence({ ok: true, diagnostics: "" }, 2),
    { known: true, binders: [] }
  );
  assert.deepEqual(
    compilerRequiredBinderEvidence({ ok: false, timedOut: true, diagnostics: "timed out" }, 2),
    { known: false, binders: [] }
  );
  assert.deepEqual(
    compilerRequiredBinderEvidence({ ok: false, timedOut: false, diagnostics: "" }, 2),
    { known: false, binders: [] }
  );
});

test("required binder evidence accepts only attributable downstream undeclared identifiers", () => {
  const downstream = {
    ok: false,
    timedOut: false,
    diagnostics: [
      "cpp_completion_7.cpp:4: error: use of undeclared identifier 'left'",
      "cpp_completion_7.cpp:5: warning: an irrelevant warning",
      "cpp_completion_7.cpp:6: error: use of undeclared identifier 'right'",
      "cpp_completion_7.cpp:7: note: downstream note",
      "cpp_completion_7.cpp:8: error: use of undeclared identifier 'left'"
    ].join("\n")
  };
  assert.deepEqual(
    compilerRequiredBinderEvidence(downstream, 2),
    { known: true, binders: ["left", "right"] }
  );

  const currentLine = {
    ok: false,
    timedOut: false,
    diagnostics: "cpp_completion_2.cpp:3: error: use of undeclared identifier 'item'"
  };
  assert.deepEqual(
    compilerRequiredBinderEvidence(currentLine, 2),
    { known: false, binders: [] }
  );

  const unrelatedMainError = {
    ok: false,
    timedOut: false,
    diagnostics: "cpp_completion_2.cpp:4: error: expected expression"
  };
  assert.deepEqual(
    compilerRequiredBinderEvidence(unrelatedMainError, 2),
    { known: false, binders: [] }
  );

  const externalError = {
    ok: false,
    timedOut: false,
    diagnostics: "/usr/include/example.hpp:4: error: use of undeclared identifier 'item'"
  };
  assert.deepEqual(
    compilerRequiredBinderEvidence(externalError, 2),
    { known: false, binders: [] }
  );
});

test("multiple required binders never inherit a unary declaration probe", async () => {
  const calls = [];
  const compiler = {
    compileAll: async (sources, priority) => {
      calls.push({ sources, priority });
      return [{
        ok: false,
        timedOut: false,
        diagnostics: [
          "cpp_completion_11.cpp:4: error: use of undeclared identifier 'left'",
          "cpp_completion_11.cpp:5: error: use of undeclared identifier 'right'"
        ].join("\n")
      }];
    }
  };
  const result = await requiredBinderObligation(
    "int main() {\n  int seed = 0;\n  \n  return left + right;\n}\n",
    2,
    2,
    {},
    compiler
  );

  assert.deepEqual(result, { known: true, binders: ["left", "right"] });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].priority, true);
});

test("singleton required binder keeps exact declaration profiles correlated", async () => {
  const source = "struct Item {};\nint main() {\n  \n  return item == item;\n}\n";
  const context = {
    identifiers: ["Item", "item"],
    sourceIdentifiers: ["Item", "item"],
    types: [{ name: "Item", type: "Item", kind: "struct", source: "ast" }],
    conversions: [],
    values: [],
    expectedTypes: []
  };
  const calls = [];
  const compiler = {
    compileAll: async (sources, priority) => {
      calls.push({ sources, priority });
      if (sources.length === 1 && sources[0] === source) {
        return [{
          ok: false,
          timedOut: false,
          diagnostics: "cpp_completion_19.cpp:4: error: use of undeclared identifier 'item'"
        }];
      }
      return sources.map(probe => probe.includes("extern Item item;") ? {
          ok: true,
          timedOut: false,
          diagnostics: ""
        } : {
          ok: false,
          timedOut: false,
          diagnostics: "cpp_completion_0.cpp:4: error: rejected downstream use"
        });
    }
  };

  const result = await requiredBinderObligation(source, 2, 2, context, compiler);

  assert.equal(result.known, true);
  assert.deepEqual(result.binders, ["item"]);
  assert.deepEqual(result.singletonGate?.accepted, [
    { type: "Item", declarationKind: "object" }
  ]);
  assert.ok(result.singletonGate?.probed.some(profile =>
    profile.type === "Item" && profile.declarationKind === "object"
  ));
  assert.ok(result.singletonGate?.probed.some(profile =>
    profile.type === "int" && profile.declarationKind === "object"
  ));
  assert.equal(result.singletonGate?.binder, "item");
  assert.equal(result.singletonGate?.complete, false);
  assert.equal(calls.length, 2);
  assert.ok(calls.every(call => call.priority === true));
  assert.equal(Object.hasOwn(context, "requiredIdentifier"), false);
});

test("probe-line binding failures remain unprobed while construction stays separate", async () => {
  const source = "struct Item {};\nint main() {\n  \n  return item == item;\n}\n";
  const compiler = {
    compileAll: async sources => sources.map(probe => {
      if (probe.includes("extern Item item;")) {
        return {
          ok: false,
          timedOut: false,
          diagnostics: "cpp_completion_5.cpp:3: error: Item has no external linkage"
        };
      }
      if (probe.includes("\n  Item item;\n")) {
        return { ok: true, timedOut: false, diagnostics: "" };
      }
      return {
        ok: false,
        timedOut: false,
        diagnostics: "cpp_completion_5.cpp:4: error: rejected downstream use"
      };
    })
  };
  const facts = await requiredDeclarationFacts(source, 2, 2, {
    requiredIdentifier: "item",
    types: [{ name: "Item", type: "Item", kind: "struct", source: "ast" }],
    conversions: [],
    values: [],
    expectedTypes: []
  }, compiler);

  assert.equal(facts.acceptedBindingProfiles.some(profile =>
    profile.type === "Item" && profile.declarationKind === "object"
  ), false);
  assert.equal(facts.probedBindingProfiles.some(profile =>
    profile.type === "Item" && profile.declarationKind === "object"
  ), false);
  assert.ok(facts.probedBindingProfiles.some(profile =>
    profile.type === "int" && profile.declarationKind === "object"
  ));
  assert.ok(facts.defaultConstructibleTypes.includes("Item"));
});

test("extern binding profiles preserve object and reference categories", async () => {
  const source = "struct Item {};\nint main() {\n  \n  return 0;\n}\n";
  const calls = [];
  const facts = await requiredDeclarationFacts(source, 2, 2, {
    requiredIdentifier: "item",
    types: [{ name: "Item", type: "Item", kind: "struct", source: "ast" }],
    conversions: [],
    values: [{ type: "Item &" }, { type: "Item &&" }],
    expectedTypes: []
  }, {
    compileAll: async sources => {
      calls.push(...sources);
      return sources.map(() => ({ ok: true, timedOut: false, diagnostics: "" }));
    }
  });

  for (const expected of [
    { type: "Item", declarationKind: "object" },
    { type: "Item &", declarationKind: "lvalueReference" },
    { type: "Item &&", declarationKind: "rvalueReference" }
  ]) {
    assert.ok(facts.acceptedBindingProfiles.some(profile =>
      profile.type === expected.type && profile.declarationKind === expected.declarationKind
    ));
  }
  assert.ok(calls.some(probe => probe.includes("extern Item item;")));
  assert.ok(calls.some(probe => probe.includes("extern Item & item;")));
  assert.ok(calls.some(probe => probe.includes("extern Item && item;")));
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

test("compiler translation units preserve candidate filenames without namespace rewriting", () => {
  const unit = combinedTranslationUnit([
    "int main() {\n#line 1 \"__TIDYPARSE_CPP_COMPLETION_BUNDLE___candidate_0.cpp\"\nreturn missing;\n}"
  ]);

  assert.match(unit, /^#line 1 "cpp_completion_0\.cpp"/);
  assert.match(unit, /cpp_completion_0_candidate_0\.cpp/);
  assert.doesNotMatch(unit, /namespace tidyparse_cpp_completion_sample_/);
  assert.doesNotMatch(unit, /#define main/);
  assert.throws(
    () => combinedTranslationUnit(["int first;", "int second;"]),
    /must remain independent/
  );
});

test("PCH keys require one byte-identical ordered leading include preamble", () => {
  const first = exactIncludePreamble([
    "#include <vector>\n#include <string>\nint first;",
    "#include <vector>\n#include <string>\nint second;"
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
  const distinct = exactIncludePreamble([
    "#include <vector>\n#include <string>\nint first;",
    "#include <vector>\nint second;"
  ]);
  const differentBytes = exactIncludePreamble([
    "#include <vector>\nint first;",
    "# include <vector>\nint second;"
  ]);

  assert.deepEqual(first.directives, ["#include <vector>", "#include <string>"]);
  assert.equal(first.key, same.key);
  assert.notEqual(first.key, reordered.key);
  assert.notEqual(first.key, superset.key);
  assert.equal(first.source, "#include <vector>\n#include <string>\n");
  assert.deepEqual(distinct.directives, []);
  assert.deepEqual(differentBytes.directives, []);
});

test("conditional, macro-expanded, and non-leading includes remain in an independent TU", () => {
  const sources = [
    [
      "#if FEATURE_ENABLED",
      "#include <feature.hpp>",
      "#endif",
      "int conditional;"
    ].join("\n"),
    [
      "#define SELECTED_HEADER <selected.hpp>",
      "#include SELECTED_HEADER",
      "int expanded;"
    ].join("\n"),
    [
      "int declaration_before_include;",
      "#include <late.hpp>",
      "int late;"
    ].join("\n")
  ];

  for (const source of sources) {
    const preamble = exactIncludePreamble([source]);
    const unit = combinedTranslationUnit([source], true);

    assert.deepEqual(preamble.directives, []);
    assert.ok(unit.endsWith(source), "unsafe source bytes must remain in their original order");
    assert.match(unit, /^#line 1 "cpp_completion_0\.cpp"/);
    assert.doesNotMatch(unit, /namespace tidyparse_cpp_completion_sample_/);
  }
});

test("macro state before an include is not reordered into a shared PCH", () => {
  const source = [
    "#define HEADER_MODE 7",
    "#include <configured.hpp>",
    "#undef HEADER_MODE",
    "int configured;"
  ].join("\n");
  const preamble = exactIncludePreamble([source]);
  const unit = combinedTranslationUnit([source], true);

  assert.deepEqual(preamble.directives, []);
  assert.ok(unit.endsWith(source));
  assert.ok(unit.indexOf("#define HEADER_MODE") < unit.indexOf("#include <configured.hpp>"));
  assert.ok(unit.indexOf("#include <configured.hpp>") < unit.indexOf("#undef HEADER_MODE"));
});

test("repeated leading includes are retained and never deduplicated through a PCH", () => {
  for (const source of [
    [
      "#include <repeatable.hpp>",
      "#include <repeatable.hpp>",
      "int repeated;"
    ].join("\n"),
    [
      "#include <repeatable.hpp>",
      "#define BETWEEN_INCLUDES 1",
      "#include <repeatable.hpp>",
      "int repeated_after_macro;"
    ].join("\n")
  ]) {
    const preamble = exactIncludePreamble([source]);
    const unit = combinedTranslationUnit([source], true);

    assert.deepEqual(preamble.directives, []);
    assert.equal([...unit.matchAll(/#include <repeatable\.hpp>/g)].length, 2);
    assert.ok(unit.endsWith(source));
  }
});

test("exact preamble groups retain batching while distinct preambles are separated", async () => {
  const compiler = new NativeCompiler("unused", 4);
  const batches = [];
  compiler.compileBatch = async sources => {
    batches.push([...sources]);
    return sources.map(() => ({
      result: { ok: true, timedOut: false, diagnostics: "" },
      cacheable: true
    }));
  };
  const vectorA = "#include <vector>\nint vector_a;";
  const vectorB = "#include <vector>\nint vector_b;";
  const string = "#include <string>\nint string_value;";
  const unsafeA = "#if A\n#include <a.hpp>\n#endif\nint unsafe_a;";
  const unsafeB = "#if A\n#include <a.hpp>\n#endif\nint unsafe_b;";

  const outcomes = await compiler.compileIsolated([
    vectorA, string, vectorB, unsafeA, unsafeB
  ]);

  assert.equal(outcomes.length, 5);
  assert.deepEqual(batches.map(batch => batch.length).sort((a, b) => a - b), [1, 2, 2]);
  assert.ok(batches.some(batch => batch.length === 2 &&
    batch.includes(vectorA) && batch.includes(vectorB)));
  assert.ok(batches.some(batch => batch.length === 1 && batch[0] === string));
  assert.ok(batches.some(batch => batch.length === 2 &&
    batch.includes(unsafeA) && batch.includes(unsafeB)));
});

test("batched compiler inputs retain independent builtin and global-namespace state", async t => {
  const clangxx = availableClangxx();
  if (clangxx == null) {
    t.skip("clang++ is unavailable");
    return;
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-independent-tu-"));
  const pchDirectory = path.join(temporary, "pch");
  fs.writeFileSync(
    path.join(temporary, "shared.hpp"),
    "template <typename T> struct GlobalTemplate {};\n"
  );
  const candidate = (name, type, expectedCounter) => [
    '#include "shared.hpp"',
    "static_assert(__LINE__ == 2);",
    `template <> struct GlobalTemplate<${type}> {};`,
    `constexpr int ${name} = __COUNTER__;`,
    `static_assert(${name} == ${expectedCounter});`,
    `int main() { return ${name}; }`
  ].join("\n");
  const compiler = new NativeCompiler(clangxx, 1, {
    pchDirectory,
    workingDirectory: temporary
  });
  try {
    const outcomes = await compiler.compileAll([
      candidate("counter_a", "int", 0),
      candidate("counter_b", "double", 0),
      candidate("counter_c", "long", 1)
    ]);

    assert.deepEqual(outcomes.map(outcome => outcome.ok), [true, true, false]);
    assert.match(outcomes[2].diagnostics, /cpp_completion_2\.cpp/);
    assert.equal(compiler.inputDirectories.size, 0);
  } finally {
    await compiler.close();
    fs.rmSync(temporary, { recursive: true, force: true });
  }
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
  assert.doesNotMatch(unit, /namespace tidyparse_cpp_completion_sample_/);
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
