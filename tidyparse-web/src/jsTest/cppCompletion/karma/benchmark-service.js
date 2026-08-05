"use strict";

const childProcess = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { pathToFileURL } = require("url");

const ROUTE_PREFIX = "/__cpp_completion";
const MAX_REQUEST_BYTES = 32 * 1024 * 1024;
const MAX_COMPILE_BATCH = 160;
// Twelve workers can now receive a full 160-source wave instead of twelve partial 107-source
// batches. This changes only scheduling; every candidate remains an independently diagnosed TU.
const MAX_COMPILE_REQUEST = 1920;
const REQUEST_TIMEOUT_MS = 8_000;
const CONTEXT_TIMEOUT_MS = 8_000;
const LSP_STEP_TIMEOUT_MS = 2_500;
const COMPILE_TIMEOUT_MS = 15_000;
const MAX_DIAGNOSTIC_BYTES = 8 * 1024 * 1024;
const MAX_COMPILER_CACHE_ENTRIES = 100_000;
const DEFAULT_SAMPLES_PER_INSTANCE = 100;
const CPP_COMPLETION_BUNDLE_MARKER = "__TIDYPARSE_CPP_COMPLETION_BUNDLE__";

const CPP_KEYWORDS = new Set([
  "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
  "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
  "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
  "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await", "co_return",
  "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else",
  "enum", "explicit", "export", "extern", "false", "final", "float", "for", "friend", "goto",
  "if", "import", "inline", "int", "long", "module", "mutable", "namespace", "new", "noexcept", "not",
  "not_eq", "nullptr", "operator", "or", "or_eq", "override", "private", "protected", "public",
  "reflexpr", "register", "reinterpret_cast", "requires", "return", "short", "signed",
  "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "synchronized",
  "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid",
  "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t",
  "while", "xor", "xor_eq"
]);

const COMPLETION_KINDS = new Map([
  [2, "method"], [3, "function"], [4, "constructor"], [5, "field"],
  [6, "variable"], [7, "class"], [8, "typeAlias"], [9, "namespace"],
  [10, "property"], [12, "value"], [13, "enum"], [20, "enumMember"],
  [21, "constant"], [22, "struct"], [24, "operator"], [25, "typeParameter"]
]);
const TYPE_REFERENCE_KINDS = new Set(["class", "concept", "enum", "struct", "typeAlias", "typeParameter"]);
const VALUE_REFERENCE_KINDS = new Set(["constant", "enumMember", "field", "property", "value", "variable"]);
const FUNCTION_REFERENCE_KINDS = new Set(["constructor", "function", "method", "operator"]);
const INSTANCE_MEMBER_KINDS = new Set(["enumMember", "field", "method", "operator", "property"]);

class OperationTimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = "OperationTimeoutError";
  }
}

function positiveIntegerEnvironment(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw === "") return fallback;
  if (!/^\d+$/.test(raw) || Number(raw) <= 0 || !Number.isSafeInteger(Number(raw))) {
    throw new Error(`${name} must be a positive integer, received ${JSON.stringify(raw)}`);
  }
  return Number(raw);
}

function nonnegativeIntegerEnvironment(name, fallback) {
  const raw = process.env[name];
  if (raw == null || raw === "") return fallback;
  if (!/^\d+$/.test(raw) || !Number.isSafeInteger(Number(raw))) {
    throw new Error(`${name} must be a nonnegative integer, received ${JSON.stringify(raw)}`);
  }
  return Number(raw);
}

function timeoutWithin(deadline, maximum, operation) {
  const remaining = deadline - Date.now();
  if (remaining <= 0) throw new OperationTimeoutError(`${operation} exceeded its deadline`);
  return Math.max(1, Math.min(REQUEST_TIMEOUT_MS, maximum, remaining));
}

function executableVersion(executable, args = ["--version"]) {
  try {
    const result = childProcess.spawnSync(executable, args, {
      encoding: "utf8",
      timeout: 5_000
    });
    if (result.error || result.status !== 0) return null;
    return (result.stdout || result.stderr || "").split(/\r?\n/, 1)[0].trim();
  } catch (_) {
    return null;
  }
}

function jsonResponse(response, status, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Length": Buffer.byteLength(body)
  });
  response.end(body);
}

function readJson(request, timeout = REQUEST_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let settled = false;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback(value);
    };
    const timer = setTimeout(() => {
      finish(reject, new OperationTimeoutError(`Request body timed out after ${timeout}ms`));
    }, Math.max(1, Math.min(REQUEST_TIMEOUT_MS, timeout)));
    request.on("data", chunk => {
      if (settled) return;
      size += chunk.length;
      if (size > MAX_REQUEST_BYTES) {
        finish(reject, new Error(`Request exceeds ${MAX_REQUEST_BYTES} bytes`));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      if (settled) return;
      try {
        const text = Buffer.concat(chunks).toString("utf8");
        finish(resolve, text.length === 0 ? {} : JSON.parse(text));
      } catch (error) {
        finish(reject, new Error(`Malformed JSON request: ${error.message}`));
      }
    });
    request.on("aborted", () => finish(reject, new Error("Request was aborted")));
    request.on("error", error => finish(reject, error));
  });
}

function offsetAt(source, line, character) {
  if (!Number.isInteger(line) || line < 0 || !Number.isInteger(character) || character < 0) {
    throw new Error("line and character must be non-negative integers");
  }
  const lines = source.split("\n");
  if (line >= lines.length || character > lines[line].length) {
    throw new Error(`Position ${line}:${character} is outside the document`);
  }
  let offset = 0;
  for (let index = 0; index < line; index++) offset += lines[index].length + 1;
  return offset + character;
}

function completionItems(result) {
  if (result == null) return [];
  return Array.isArray(result) ? result : Array.isArray(result.items) ? result.items : [];
}

function identifierWords(value) {
  if (typeof value !== "string") return [];
  return value.match(/[A-Za-z_][A-Za-z_0-9]*/g) || [];
}

function addIdentifierWords(target, value) {
  for (const word of identifierWords(value)) {
    if (!CPP_KEYWORDS.has(word)) target.add(word);
  }
}

/**
 * Extract C++ identifiers without admitting words from comments, literals, or include paths.
 * This is deliberately lexical rather than semantic: clangd completions and document symbols
 * supplement it with names that are merely in scope rather than already present in the file.
 */
function lexicalIdentifiers(source) {
  const identifiers = new Set();
  let index = 0;
  let lineStart = true;
  while (index < source.length) {
    const character = source[index];
    if (character === "\n" || character === "\r") {
      lineStart = true;
      index += 1;
      continue;
    }
    if (lineStart && (character === " " || character === "\t")) {
      index += 1;
      continue;
    }
    if (lineStart && character === "#") {
      const lineEnd = source.indexOf("\n", index);
      const directiveEnd = lineEnd < 0 ? source.length : lineEnd;
      const directive = source.slice(index, directiveEnd);
      if (/^#\s*include(?:_next)?\b/.test(directive)) {
        index = directiveEnd;
        continue;
      }
      lineStart = false;
    } else if (character !== " " && character !== "\t") {
      lineStart = false;
    }

    if (source.startsWith("//", index)) {
      const lineEnd = source.indexOf("\n", index + 2);
      index = lineEnd < 0 ? source.length : lineEnd;
      continue;
    }
    if (source.startsWith("/*", index)) {
      const commentEnd = source.indexOf("*/", index + 2);
      index = commentEnd < 0 ? source.length : commentEnd + 2;
      continue;
    }

    const raw = /^(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(/.exec(source.slice(index));
    if (raw != null) {
      const terminator = `)${raw[1]}"`;
      const rawEnd = source.indexOf(terminator, index + raw[0].length);
      index = rawEnd < 0 ? source.length : rawEnd + terminator.length;
      continue;
    }
    const quoted = /^(?:u8|u|U|L)?(["'])/.exec(source.slice(index));
    if (quoted != null) {
      const quote = quoted[1];
      index += quoted[0].length;
      while (index < source.length) {
        if (source[index] === "\\") {
          index += 2;
        } else if (source[index] === quote) {
          index += 1;
          break;
        } else {
          index += 1;
        }
      }
      continue;
    }
    if (/[0-9]/.test(character)) {
      index += 1;
      while (index < source.length && /[A-Za-z_0-9.']/.test(source[index])) index += 1;
      continue;
    }
    if (/[A-Za-z_]/.test(character)) {
      let end = index + 1;
      while (end < source.length && /[A-Za-z_0-9]/.test(source[end])) end += 1;
      const identifier = source.slice(index, end);
      if (!CPP_KEYWORDS.has(identifier)) identifiers.add(identifier);
      index = end;
      continue;
    }
    index += 1;
  }
  return identifiers;
}

function declaredTypeNames(source) {
  const types = new Set();
  const visibleIdentifiers = lexicalIdentifiers(source);
  const declaration = /\b(?:class|struct|union|namespace)\s+([A-Za-z_][A-Za-z_0-9]*)|\benum\s+(?:class\s+|struct\s+)?([A-Za-z_][A-Za-z_0-9]*)|\busing\s+([A-Za-z_][A-Za-z_0-9]*)\s*=/g;
  for (const match of source.matchAll(declaration)) {
    const name = match[1] || match[2] || match[3];
    if (name != null && visibleIdentifiers.has(name)) types.add(name);
  }
  const typedef = /\btypedef\b[^;{}]*\b([A-Za-z_][A-Za-z_0-9]*)\s*;/g;
  for (const match of source.matchAll(typedef)) {
    if (visibleIdentifiers.has(match[1])) types.add(match[1]);
  }
  return types;
}

/** Source aliases retain their target spelling even when clang completion only returns the name. */
function sourceTypeAliases(source) {
  return [...source.matchAll(/\busing\s+([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^;{}\r\n]+)\s*;/g)]
    .map(match => ({
      name: match[1],
      type: match[2].trim(),
      detail: match[2].trim(),
      kind: "typeAlias",
      source: "source"
    }));
}

function splitTopLevel(text, separator = ",") {
  const parts = [];
  let start = 0;
  let angle = 0;
  let round = 0;
  let square = 0;
  let brace = 0;
  for (let index = 0; index < text.length; index++) {
    switch (text[index]) {
      case "<": angle += 1; break;
      case ">": angle = Math.max(0, angle - 1); break;
      case "(": round += 1; break;
      case ")": round = Math.max(0, round - 1); break;
      case "[": square += 1; break;
      case "]": square = Math.max(0, square - 1); break;
      case "{": brace += 1; break;
      case "}": brace = Math.max(0, brace - 1); break;
      default:
        if (text[index] === separator && angle + round + square + brace === 0) {
          parts.push(text.slice(start, index).trim());
          start = index + 1;
        }
    }
  }
  parts.push(text.slice(start).trim());
  return parts.filter(part => part.length > 0 && part !== "void");
}

function parameterFromLabel(label, typeOverride) {
  const rawLabel = String(label || "").trim();
  const equals = splitTopLevel(rawLabel, "=");
  const declaration = equals[0] || rawLabel;
  const nameMatch = /(?:^|[\s*&])([A-Za-z_][A-Za-z_0-9]*)\s*(\[[^\]]*\])?\s*$/.exec(declaration);
  const nameOffset = nameMatch == null ? -1 : nameMatch.index + nameMatch[0].lastIndexOf(nameMatch[1]);
  const prefix = nameOffset < 0 ? "" : declaration.slice(0, nameOffset).trim();
  const hasSeparateName = nameMatch != null && prefix.length > 0;
  const parameter = {
    label: rawLabel,
    type: String(typeOverride || (hasSeparateName ? prefix + (nameMatch[2] || "") : declaration)).trim()
  };
  if (hasSeparateName) parameter.name = nameMatch[1];
  if (equals.length > 1) parameter.defaultValue = rawLabel.slice(rawLabel.indexOf("=") + 1).trim();
  return parameter;
}

function parameterClause(text) {
  if (typeof text !== "string") return [];
  const open = text.indexOf("(");
  if (open < 0) return [];
  let depth = 0;
  for (let index = open; index < text.length; index++) {
    if (text[index] === "(") depth += 1;
    if (text[index] === ")" && --depth === 0) {
      return splitTopLevel(text.slice(open + 1, index)).map(label => parameterFromLabel(label));
    }
  }
  return [];
}

function quotedAstType(arcana) {
  if (typeof arcana !== "string") return undefined;
  // clang's AST spells deduced/aliased types as 'sugar':'canonical'. The canonical spelling is
  // essential here: for example an `auto` initialized by make_shared is reported as
  // 'shared_ptr<Node>':'std::shared_ptr<Node>'. Keeping only the sugar loses the namespace and
  // prevents it from matching a parameter spelled std::shared_ptr<Node>.
  const canonical = /\s'([^'\r\n]+)'\s*:\s*'([^'\r\n]+)'/.exec(arcana);
  if (canonical != null) return canonical[2];
  return /\s'([^'\r\n]+)'/.exec(arcana)?.[1];
}

function astRangeText(sourceLines, range) {
  if (range?.start == null || range?.end == null || range.start.line !== range.end.line) return undefined;
  return sourceLines[range.start.line]?.slice(range.start.character, range.end.character).trim();
}

function astRangeSource(sourceLines, range) {
  if (range?.start == null || range?.end == null) return undefined;
  if (range.start.line < 0 || range.end.line >= sourceLines.length || range.start.line > range.end.line) {
    return undefined;
  }
  if (range.start.line === range.end.line) {
    return sourceLines[range.start.line]?.slice(range.start.character, range.end.character);
  }
  return [
    sourceLines[range.start.line]?.slice(range.start.character) || "",
    ...sourceLines.slice(range.start.line + 1, range.end.line),
    sourceLines[range.end.line]?.slice(0, range.end.character) || ""
  ].join("\n");
}

function sourceRangeText(source, range) {
  if (range?.start == null || range?.end == null) return undefined;
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  return astRangeSource(lines, range)?.trim();
}

/** Names clang diagnosed as undeclared in the current truncated document. */
function unresolvedDiagnosticIdentifiers(source, diagnostics) {
  const identifiers = new Map();
  for (const diagnostic of diagnostics || []) {
    const code = String(diagnostic?.code || "");
    const message = String(diagnostic?.message || "");
    if (!/^undeclared_var_use(?:_suggest)?$/.test(code) &&
      !/\b(?:use of )?undeclared identifier\b/i.test(message)) continue;
    let identifier = sourceRangeText(source, diagnostic.range);
    if (!/^[A-Za-z_][A-Za-z_0-9]*$/.test(identifier || "")) {
      identifier = /undeclared identifier\s+['‘]([A-Za-z_][A-Za-z_0-9]*)['’]/i.exec(message)?.[1];
    }
    if (identifier && !CPP_KEYWORDS.has(identifier) && !identifiers.has(identifier)) {
      identifiers.set(identifier, {
        line: Number(diagnostic.range?.start?.line) || 0,
        character: Number(diagnostic.range?.start?.character) || 0
      });
    }
  }
  return [...identifiers].sort((left, right) =>
    left[1].line - right[1].line || left[1].character - right[1].character
  ).map(([identifier]) => identifier);
}

const REQUIRED_TYPE_LIMIT = 96;

/**
 * Canonical type spellings accepted by the declaration oracle. This deliberately excludes
 * top-level function declarators, raw arrays, anonymous, dependent and deduced types. Function
 * signatures and unsized arrays inside template arguments (`function<R(A)>`, `unique_ptr<T[]>`)
 * remain spellable and are validated by the compiler probe.
 */
function canonicalRequiredType(rawType) {
  if (typeof rawType !== "string") return undefined;
  let type = rawType.trim()
    .replace(/\b(?:class|struct|enum)\s+(?=[A-Za-z_])/g, "")
    .replace(/\s+/g, " ")
    .replace(/\s*::\s*/g, "::")
    .replace(/\s*<\s*/g, "<")
    .replace(/\s*>\s*/g, ">")
    .replace(/\s*,\s*/g, ", ")
    .replace(/\s*\*\s*/g, " *")
    .replace(/\[\s*\]/g, "[]")
    .replace(/\s*&&\s*$/, " &&")
    .replace(/(?<!&)\s*&\s*$/, " &")
    .trim();
  if (type === "" || type === "void" || /\b(?:auto|decltype)\b/.test(type) ||
      /[{};'"`=]/.test(type) || /(?:anonymous|lambda|dependent|<invalid>)/i.test(type) ||
      !/^[A-Za-z_][A-Za-z_0-9:\s,<>&*()[\]]*$/.test(type)) return undefined;

  // Parentheses are ordinary type syntax inside a template argument (`std::function<int(int)>`),
  // but at top level they denote a function or pointer-to-function declarator, which this oracle
  // intentionally does not synthesize. Likewise, an unsized array is accepted only as part of a
  // template argument (`unique_ptr<T[]>`); a raw array needs its identifier inside the declarator
  // and cannot be emitted by `T identifier{}` below.
  let angle = 0;
  let round = 0;
  for (let index = 0; index < type.length; index++) {
    switch (type[index]) {
      case "<":
        angle += 1;
        break;
      case ">":
        if (angle === 0 || round > 0 && angle === 1) return undefined;
        angle -= 1;
        break;
      case "(":
        if (angle === 0) return undefined;
        round += 1;
        break;
      case ")":
        if (angle === 0 || round === 0) return undefined;
        round -= 1;
        break;
      case "[":
        if (angle === 0 || type[index + 1] !== "]") return undefined;
        index += 1;
        break;
      case "]":
        return undefined;
    }
  }
  if (angle !== 0 || round !== 0) return undefined;
  return type;
}

function unqualifiedRequiredType(rawType) {
  const type = canonicalRequiredType(rawType);
  if (!type) return undefined;
  return canonicalRequiredType(type
    .replace(/\s*(?:&&|&)\s*$/, "")
    .replace(/^\s*(?:const|volatile)\s+/, "")
    .replace(/\s+(?:const|volatile)\s*$/, ""));
}

function templateArguments(rawType, templateName) {
  const type = unqualifiedRequiredType(rawType);
  if (!type) return [];
  const expression = new RegExp(`(?:^|::)${templateName}\\s*<`, "g");
  let match;
  let open = -1;
  while ((match = expression.exec(type)) != null) open = type.indexOf("<", match.index);
  if (open < 0) return [];
  let depth = 0;
  for (let index = open; index < type.length; index++) {
    if (type[index] === "<") depth += 1;
    else if (type[index] === ">" && --depth === 0) {
      return splitTopLevel(type.slice(open + 1, index));
    }
  }
  return [];
}

function templateArgument(rawType, templateName) {
  return templateArguments(rawType, templateName)[0];
}

/** A compact, source-derived candidate universe for one undeclared local. */
function requiredTypeCandidates(source, context) {
  const candidates = new Set();
  const add = rawType => {
    const canonical = canonicalRequiredType(rawType);
    if (canonical && candidates.size < REQUIRED_TYPE_LIMIT) candidates.add(canonical);
  };
  const addObserved = rawType => {
    add(rawType);
    add(unqualifiedRequiredType(rawType));
  };
  const sourceNames = lexicalIdentifiers(source);
  // Include directives are intentionally ignored by lexicalIdentifiers, but they are strong,
  // translation-unit-local evidence that these standard vocabulary types are spellable.
  const includedHeaders = new Set([...source.matchAll(/^\s*#\s*include\s*[<"]([^>"]+)[>"]/gm)]
    .map(match => match[1]));
  if (includedHeaders.has("string")) sourceNames.add("string");
  if (includedHeaders.has("string_view")) sourceNames.add("string_view");
  if (includedHeaders.has("sstream")) sourceNames.add("ostringstream");
  if (includedHeaders.has("vector")) sourceNames.add("vector");
  if (includedHeaders.has("memory")) {
    sourceNames.add("unique_ptr");
    sourceNames.add("shared_ptr");
    sourceNames.add("weak_ptr");
  }
  if (includedHeaders.has("cstdint")) {
    sourceNames.add("uintptr_t");
    sourceNames.add("intptr_t");
  }
  if (sourceNames.has("make_unique")) sourceNames.add("unique_ptr");
  if (sourceNames.has("make_shared")) sourceNames.add("shared_ptr");
  const userTypes = new Set();
  for (const reference of context.types || []) {
    if (reference.source !== "completion" &&
        new Set(["class", "struct", "enum", "typeAlias"]).has(reference.kind)) {
      const type = unqualifiedRequiredType(reference.name || reference.type || reference.detail);
      if (type && /^[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)*$/.test(type)) {
        userTypes.add(type);
      }
    }
  }
  for (const type of declaredTypeNames(source)) {
    const canonical = unqualifiedRequiredType(type);
    if (canonical) userTypes.add(canonical);
  }
  for (const conversion of context.conversions || []) {
    for (const rawType of [conversion.from, conversion.to]) {
      const type = unqualifiedRequiredType(rawType);
      if (type) userTypes.add(type);
      addObserved(rawType);
    }
  }

  // Existing scoped values are the most reliable source of canonical standard-library and alias
  // spellings, so keep them first in the candidate order.
  for (const value of context.values || []) addObserved(value.type);
  for (const type of context.expectedTypes || []) addObserved(type);
  addObserved(context.enclosingReturnType);
  addObserved(context.thisType);
  for (const type of userTypes) add(type);

  const fundamentalTypes = [
    ["bool", "bool"], ["char", "char"], ["short", "short"], ["int", "int"],
    ["signed", "signed"], ["unsigned", "unsigned"], ["long", "long"],
    ["float", "float"], ["double", "double"],
    ["size_t", "std::size_t"]
  ];
  // A deleted declaration can remove the translation unit's only keyword spelling. These types
  // are cheap candidates and the whole-file compiler probe remains the authority on which ones
  // satisfy every downstream use.
  for (const [, type] of fundamentalTypes) add(type);
  if (sourceNames.has("string")) add("std::string");
  if (sourceNames.has("string_view")) add("std::string_view");
  if (sourceNames.has("ostringstream")) add("std::ostringstream");
  if (sourceNames.has("uintptr_t")) add("std::uintptr_t");
  if (sourceNames.has("intptr_t")) add("std::intptr_t");
  if (includedHeaders.has("memory")) {
    add("const void *");
    add("std::unique_ptr<int[]>");
  }

  const smartPointers = [];
  for (const type of userTypes) {
    add(`${type} *`);
    add(`const ${type} *`);
    for (const smart of ["unique_ptr", "shared_ptr", "weak_ptr"]) {
      if (!sourceNames.has(smart)) continue;
      const pointer = `std::${smart}<${type}>`;
      smartPointers.push(pointer);
      add(pointer);
    }
  }

  // Template query functions such as get_if preserve the selected alternative's pointee type.
  // Derive both cv forms from scoped variants so deletion of the query line does not erase them.
  for (const value of context.values || []) {
    for (const alternative of templateArguments(value.type, "variant")) {
      add(`${alternative} *`);
      add(`const ${alternative} *`);
    }
  }

  if (sourceNames.has("vector")) {
    const elements = new Set([...userTypes, ...smartPointers]);
    if (sourceNames.has("string")) elements.add("std::string");
    for (const value of context.values || []) {
      const element = templateArgument(value.type, "vector");
      if (element) elements.add(element);
    }
    for (const element of elements) add(`std::vector<${element}>`);
  }
  return { candidates: [...candidates], userTypes };
}

function safeRequiredDeclaration(type, identifier, userTypes) {
  const canonical = canonicalRequiredType(type);
  if (!canonical || !/^[A-Za-z_][A-Za-z_0-9]*$/.test(identifier)) return undefined;
  const withoutReference = canonical.replace(/\s*(?:&&|&)\s*$/, "").trim();
  const base = withoutReference.replace(/^\s*(?:const|volatile)\s+/, "")
    .replace(/\s+(?:const|volatile)\s*$/, "").trim();
  const isReference = withoutReference !== canonical;
  if (isReference || userTypes.has(base)) {
    return `${withoutReference} & ${identifier} = *static_cast<${withoutReference} *>(nullptr);`;
  }
  return `${canonical} ${identifier}{};`;
}

function replaceCursorLine(source, line, replacement) {
  const lines = source.split(/(?<=\n)/);
  if (!Number.isInteger(line) || line < 0 || line >= lines.length) return undefined;
  const terminator = /\r?\n$/.exec(lines[line])?.[0] || "";
  const content = lines[line].slice(0, lines[line].length - terminator.length);
  const indentation = /^\s*/.exec(content)?.[0] || "";
  lines[line] = indentation + replacement + terminator;
  return lines.join("");
}

/**
 * Finds types which make every use after a deleted declaration compile. The probe replaces the
 * whole cursor line, so once a required identifier is known the result is independent of how far
 * through its qualified/template type the cursor has advanced. NativeCompiler's digest cache then
 * makes every later boundary of the same declaration effectively free.
 */
async function requiredDeclarationFacts(source, line, character, context, compiler) {
  if (compiler == null || typeof context?.requiredIdentifier !== "string") {
    return { requiredTypes: [], probedRequiredTypes: [], defaultConstructibleTypes: [] };
  }
  const { candidates, userTypes } = requiredTypeCandidates(source, context);
  const probes = [];
  const descriptors = [];
  for (const type of candidates) {
    const declaration = safeRequiredDeclaration(type, context.requiredIdentifier, userTypes);
    const probe = declaration && replaceCursorLine(source, line, declaration);
    if (!probe) continue;
    descriptors.push({ type, kind: "required" });
    probes.push(probe);

    const canonical = canonicalRequiredType(type);
    const base = canonical?.replace(/\s*(?:&&|&)\s*$/, "").trim()
      .replace(/^\s*(?:const|volatile)\s+/, "")
      .replace(/\s+(?:const|volatile)\s*$/, "").trim();
    if (canonical && base && userTypes.has(base) && canonical === base) {
      const defaultProbe = replaceCursorLine(
        source, line, `${canonical} ${context.requiredIdentifier};`
      );
      if (defaultProbe) {
        descriptors.push({ type, kind: "default" });
        probes.push(defaultProbe);
      }
    }
  }
  if (probes.length === 0) {
    return { requiredTypes: [], probedRequiredTypes: [], defaultConstructibleTypes: [] };
  }
  // Every probe differs only at the deleted declaration. Packing related variants together lets
  // combinedTranslationUnit hoist their identical includes once instead of making each compiler
  // worker parse the same SDK preamble. Retain compileAll as a fallback for test/legacy bridges.
  const outcomes = await compiler.compileAll(probes, true);
  const definitiveOutcome = outcome => outcome?.ok === true ||
    outcome?.timedOut === false && /(?:^|\n)(?:.*[\\/])?cpp_completion_\d+\.cpp:\d+.*\berror:/i
      .test(outcome.diagnostics || "");
  const defaultDeclarationCompiled = outcome => {
    if (outcome?.ok === true) return true;
    if (!definitiveOutcome(outcome)) return false;
    const diagnostics = String(outcome.diagnostics || "");
    const errors = diagnostics.split(/\r?\n/).filter(diagnostic =>
      /:\s*(?:fatal\s+)?error:/i.test(diagnostic)
    );
    const errorLines = errors.map(diagnostic =>
      /(?:^|[\\/])cpp_completion_\d+\.cpp:(\d+)(?::\d+)?:\s*(?:fatal\s+)?error:/i
        .exec(diagnostic)?.[1]
    );
    // The probe is an actual `T identifier;` declaration on the damaged source line. An error on
    // that line (or in an earlier constructor definition instantiated by it) rejects T. Errors
    // strictly later belong to uses of the required identifier and do not invalidate the observed
    // default construction itself.
    return errorLines.length > 0 && errorLines.every(errorLine =>
      errorLine != null && Number(errorLine) > line + 1
    );
  };
  return {
    requiredTypes: descriptors.filter((descriptor, index) =>
      descriptor.kind === "required" && outcomes[index]?.ok === true).map(descriptor => descriptor.type),
    // This is deliberately per-type coverage, not a claim that the capped source-derived universe
    // is exhaustive. Kotlin may reject a probed-and-failed type while retaining every unprobed type.
    probedRequiredTypes: descriptors.filter((descriptor, index) =>
      descriptor.kind === "required" && definitiveOutcome(outcomes[index])
    ).map(descriptor => descriptor.type),
    defaultConstructibleTypes: descriptors.filter((descriptor, index) =>
      descriptor.kind === "default" && defaultDeclarationCompiled(outcomes[index])
    ).map(descriptor => descriptor.type)
  };
}

/** Backward-compatible narrow helper retained for direct benchmark-service consumers. */
async function requiredDeclarationTypes(source, line, character, context, compiler) {
  return (await requiredDeclarationFacts(source, line, character, context, compiler)).requiredTypes;
}

function looksLikePartialDeclaration(source, cursor, typeNames) {
  const line = source.replace(/\r\n?/g, "\n").split("\n")[cursor.line] || "";
  const prefix = line.slice(0, cursor.character).trim();
  if (prefix === "" || /[().=?!+\/%|^]/.test(prefix) || /->/.test(prefix)) return false;
  const words = [...prefix.matchAll(/[A-Za-z_][A-Za-z_0-9]*/g)].map(match => match[0]);
  if (words[0] === "auto" || words[0] === "const" || words.length === 1 && words[0] === "std") {
    return true;
  }
  const knownTypes = new Set(typeNames || []);
  return words.some(word => word !== "std" && knownTypes.has(word));
}

/**
 * clang recovery can consume the next statement's receiver as the missing declarator after a
 * complete template type, suppressing its undeclared-name diagnostic. For an unambiguously
 * type-shaped prefix, recover the first later use that is neither scoped nor declared later.
 */
function futureRequiredIdentifier(source, cursor, scopedValues, typeNames, functions) {
  if (!looksLikePartialDeclaration(source, cursor, typeNames)) return undefined;
  const known = new Set((scopedValues || []).map(value => value.name?.split("::").at(-1)).filter(Boolean));
  const knownTypes = new Set(typeNames || []);
  const knownFunctions = new Set((functions || []).map(fn => fn.name?.split("::").at(-1)).filter(Boolean));
  const lines = source.replace(/\r\n?/g, "\n").split("\n").slice(cursor.line + 1);
  for (const rawLine of lines) {
    const line = rawLine
      .replace(/\/\/.*$/, "")
      .replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, " ");
    const declaration = /^\s*(?:const\s+)?(?:auto|(?:std\s*::\s*)?[A-Za-z_][A-Za-z_0-9]*(?:\s*<[^;={}()]+>)?)(?:\s+const)?\s*[*&]*\s*([A-Za-z_][A-Za-z_0-9]*)\b/.exec(line);
    if (declaration != null) known.add(declaration[1]);
    for (const match of line.matchAll(/[A-Za-z_][A-Za-z_0-9]*/g)) {
      const name = match[0];
      const start = match.index;
      const before = line.slice(0, start).trimEnd();
      const after = line.slice(start + name.length).trimStart();
      if (CPP_KEYWORDS.has(name) || known.has(name) || knownTypes.has(name) || knownFunctions.has(name) ||
          /(?:\.|->|::)$/.test(before) || after.startsWith("::") ||
          (after.startsWith("(") && !after.startsWith("()"))) continue;
      return name;
    }
  }
  return undefined;
}

function stableDeclaredTypeKey(record, sourceLines) {
  const declaration = astRangeSource(sourceLines, record.range);
  if (!declaration) return record.detail;
  const fingerprint = crypto.createHash("sha256").update(declaration).digest("hex").slice(0, 20);
  return `${record.detail}\u0000${fingerprint}`;
}

function positionLessOrEqual(left, right) {
  return left != null && (left.line < right.line || left.line === right.line && left.character <= right.character);
}

function rangeContains(range, position) {
  return range?.start != null && range?.end != null &&
    positionLessOrEqual(range.start, position) && positionLessOrEqual(position, range.end);
}

function astParameter(node, sourceLines) {
  const label = astRangeText(sourceLines, node.range) || node.detail || quotedAstType(node.arcana) || "";
  const parameter = parameterFromLabel(label, quotedAstType(node.arcana));
  if (node.detail) parameter.name = node.detail;
  return parameter;
}

function astCallable(node, sourceLines, kind, fallbackName) {
  const signature = quotedAstType(node.arcana) || "";
  const open = signature.indexOf("(");
  const astParameters = astParameterNodes(node).map(child => astParameter(child, sourceLines));
  const reference = {
    name: node.detail || fallbackName,
    kind,
    detail: signature,
    returnType: kind === "constructor" ? fallbackName : (open < 0 ? signature : signature.slice(0, open).trim()),
    // Synthesized inherited constructors do not always expose ParmVar children in clangd's AST.
    parameters: astParameters.length > 0 ? astParameters : parameterClause(signature),
    source: "ast"
  };
  return reference.name ? reference : null;
}

function astParameterNodes(node) {
  const result = [];
  const visit = current => {
    if (current.kind === "ParmVar" && current.role === "declaration") {
      result.push(current);
      return;
    }
    if (current !== node && (current.role === "statement" || current.role === "expression")) return;
    for (const child of current.children || []) visit(child);
  };
  visit(node);
  return result;
}

function firstAstTypeName(node) {
  if (node?.role === "type" && typeof node.detail === "string") return node.detail;
  for (const child of node?.children || []) {
    const found = firstAstTypeName(child);
    if (found) return found;
  }
  return undefined;
}

function astSemanticContext(ast, source, cursor) {
  const sourceLines = source.replace(/\r\n?/g, "\n").split("\n");
  const values = [];
  const functions = [];
  const types = [];
  const membersByType = [];
  const conversions = [];
  let enclosingReturnType;
  let enclosingClassType;
  let thisType;
  let mutableFields = [];
  const scopeKinds = new Set([
    "Compound", "CompoundStmt", "CXXConstructor", "CXXForRange", "CXXMethod",
    "For", "Function", "If", "Lambda", "Switch", "While"
  ]);

  const visitScoped = (node, ancestors) => {
    if (node == null || typeof node !== "object") return;
    const declaredBeforeCursor = positionLessOrEqual(node.range?.start, cursor);
    if (node.kind === "Enum" && node.role === "declaration" && node.detail && declaredBeforeCursor) {
      types.push({
        name: node.detail,
        type: node.detail,
        kind: "enum",
        detail: node.detail,
        source: "ast"
      });
    }
    if (node.kind === "EnumConstant" && node.role === "declaration" && node.detail && declaredBeforeCursor) {
      const owner = [...ancestors].reverse().find(ancestor =>
        ancestor.kind === "Enum" && ancestor.role === "declaration" && ancestor.detail);
      if (owner != null) values.push({
        name: node.detail,
        type: owner.detail,
        kind: "enumMember",
        detail: /\b(?:class|struct|scoped)\b/.test(owner.arcana || "") ? "scoped" : "unscoped",
        source: "ast",
        ownerType: owner.detail
      });
    }
    if (node.role === "declaration" && rangeContains(node.range, cursor)) {
      const enclosingRecord = [...ancestors].reverse().find(ancestor =>
        ancestor.kind === "CXXRecord" && ancestor.role === "declaration" && ancestor.detail);
      if (enclosingRecord != null && new Set([
        "CXXConstructor", "CXXConversion", "CXXDestructor", "CXXMethod"
      ]).has(node.kind)) {
        enclosingClassType = enclosingRecord.detail;
        const signature = quotedAstType(node.arcana) || "";
        const methodConst = /\)\s+const(?:\s|$)/.test(signature);
        thisType = `${methodConst ? "const " : ""}${enclosingRecord.detail} *`;
        mutableFields = (enclosingRecord.children || [])
          .filter(child => child.kind === "Field" && child.detail && /\bmutable\b/.test(child.arcana || ""))
          .map(child => child.detail);
      }
      if (node.kind === "CXXConstructor" || node.kind === "CXXDestructor") {
        enclosingReturnType = "void";
      } else if (node.kind === "Function" || node.kind === "CXXMethod" ||
        node.kind === "CXXConversion") {
        const signature = quotedAstType(node.arcana) || "";
        const open = signature.indexOf("(");
        if (open > 0) enclosingReturnType = signature.slice(0, open).trim();
      }
    }
    if ((node.kind === "Var" || node.kind === "ParmVar") && node.role === "declaration" && declaredBeforeCursor) {
      const scope = [...ancestors].reverse().find(ancestor => scopeKinds.has(ancestor.kind));
      const startsOnCursorLine = node.range?.start?.line === cursor.line;
      const currentPrefix = sourceLines[cursor.line]?.slice(0, cursor.character) || "";
      const spelling = String(node.detail || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const spelledBeforeCursor = !startsOnCursorLine ||
        spelling !== "" && new RegExp(`(?:^|\\W)${spelling}(?:$|\\W)`).test(currentPrefix);
      // Recovery may reinterpret the next statement's callee as this unfinished declaration's
      // variable (for example `Cat` followed by `introduce(cat)`). Do not surface such a phantom
      // local until its spelling has actually appeared before the cursor.
      if (spelledBeforeCursor && (scope == null || rangeContains(scope.range, cursor))) {
        values.push({
          name: node.detail,
          type: quotedAstType(node.arcana) || "",
          kind: "variable",
          detail: quotedAstType(node.arcana) || "",
          source: "ast"
        });
      }
    }
    if (node.kind === "Function" && node.role === "declaration" && node.detail !== "main" && declaredBeforeCursor &&
        !ancestors.some(ancestor => ancestor.kind === "CXXRecord")) {
      const callable = astCallable(node, sourceLines, "function", node.detail);
      if (callable != null) functions.push(callable);
    }
    for (const child of node.children || []) visitScoped(child, [...ancestors, node]);
  };
  visitScoped(ast, []);

  const records = [];
  const visitRecords = node => {
    if (node == null || typeof node !== "object") return;
    if (node.kind === "CXXRecord" && node.role === "declaration" && node.detail &&
        positionLessOrEqual(node.range?.start, cursor) && /\bdefinition/i.test(node.arcana || "")) {
      records.push(node);
    }
    for (const child of node.children || []) visitRecords(child);
  };
  visitRecords(ast);

  for (const record of records) {
    const recordKind = /\bstruct\s+/.test(record.arcana || "") ? "struct" : "class";
    const abstract = /\babstract\b/.test(record.arcana || "");
    const bases = (record.children || [])
      .filter(child => child.role === "base" && child.kind === "public")
      .map(firstAstTypeName).filter(Boolean);
    const members = [];
    let access = recordKind === "struct" ? "public" : "private";
    for (const child of record.children || []) {
      if (child.kind === "AccessSpec") {
        access = /\b(public|private|protected)\s*$/.exec(child.arcana || "")?.[1] || access;
        continue;
      }
      const implicitDeclaration = /(?:^|\s)implicit(?:\s|$)/.test(child.arcana || "");
      const constructorDeclaration = child.kind === "CXXConstructor" || child.kind === "ConstructorUsingShadow";
      if (access !== "public" || (!constructorDeclaration && implicitDeclaration)) continue;
      if (child.kind === "Field" && child.detail) {
        members.push({
          name: child.detail,
          type: quotedAstType(child.arcana) || "",
          kind: "field",
          detail: quotedAstType(child.arcana) || "",
          source: "ast",
          receiverMember: true,
          ownerType: record.detail
        });
      } else if (child.kind === "CXXMethod") {
        const callable = astCallable(child, sourceLines, child.detail?.startsWith("operator") ? "operator" : "method", child.detail);
        if (callable != null) members.push({
          ...callable,
          receiverMember: true,
          ownerType: record.detail
        });
      } else if (child.kind === "CXXConstructor" && !abstract) {
        const implicit = /(?:^|\s)implicit(?:\s|$)/.test(child.arcana || "");
        const inherited = /(?:^|\s)implicit\s+used(?:\s|$)/.test(child.arcana || "");
        const deleted = /(?:\bdeleted\b|\bdelete\b|default_delete)/.test(child.arcana || "");
        // Implicit copy/move constructors add low-value ambiguity, but a non-deleted zero-argument
        // constructor is the only proof that `T value;` is sound. Inherited constructors likewise
        // retain patterns such as `using Node::Node`. A deleted implicit default (for example
        // `Cat()` when its base has no default constructor) remains a semantic rejection.
        const callable = astCallable(child, sourceLines, "constructor", record.detail);
        const implicitDefault = implicit && (callable?.parameters || []).length === 0;
        if ((!implicit || inherited || implicitDefault) && !deleted) {
          if (callable != null) functions.push({ ...callable, ownerType: record.detail });
        }
      } else if (child.kind === "ConstructorUsingShadow" && !abstract) {
        // `using Base::Base` is represented by ConstructorUsingShadow even when clang has not
        // instantiated a corresponding CXXConstructor node in this (possibly truncated) AST.
        const signature = [...String(child.arcana || "").matchAll(/'([^'\r\n]+)'/g)]
          .map(match => match[1]).filter(candidate => candidate.includes("(")).at(-1);
        const parameters = parameterClause(signature);
        const inheritedCopy = parameters.length === 1 && bases.some(base =>
          parameters[0].type.replace(/\b(?:const|volatile)\b/g, "")
            .replace(/(?:&&|&)\s*$/, "").trim() === base
        );
        if (signature && !inheritedCopy) functions.push({
          name: record.detail,
          kind: "constructor",
          detail: signature,
          returnType: record.detail,
          parameters,
          source: "ast",
          ownerType: record.detail
        });
      }
    }
    types.push({
      name: record.detail,
      kind: recordKind,
      detail: record.detail,
      bases,
      abstract,
      cacheKey: stableDeclaredTypeKey(record, sourceLines)
    });
    membersByType.push({ type: record.detail, members });
    for (const base of bases) conversions.push({ from: record.detail, to: base });
  }

  const tables = new Map(membersByType.map(table => [table.type, table]));
  const inherited = (type, seen = new Set()) => {
    if (seen.has(type)) return [];
    seen.add(type);
    const own = tables.get(type)?.members || [];
    const bases = types.find(entry => entry.name === type)?.bases || [];
    return [...own, ...bases.flatMap(base => inherited(base, seen).filter(member => member.kind !== "constructor"))];
  };
  for (const table of membersByType) table.members = deduplicateReferences(inherited(table.type));
  return {
    values: deduplicateReferences(values),
    functions: deduplicateReferences(functions),
    types,
    membersByType,
    conversions,
    enclosingReturnType,
    enclosingClassType,
    thisType,
    mutableFields
  };
}

/**
 * Retains declaration-stable user-type facts across reparses of the same incomplete statement.
 * Locals and free functions deliberately remain cursor-local; only record metadata, public member
 * tables, and constructors are cached. A declaration fingerprint prevents a same-named type in a
 * different fixture from borrowing facts unless its declaration is genuinely identical.
 */
function mergeCachedUserTypeFacts(astContext, source, cache) {
  if (!(cache instanceof Map)) return astContext;
  const constructors = astContext.functions.filter(reference => reference.kind === "constructor");
  const nonConstructors = astContext.functions.filter(reference => reference.kind !== "constructor");
  const currentTables = new Map(astContext.membersByType.map(table => [table.type, table]));
  const activeEntries = [];

  for (const type of astContext.types) {
    const key = type.cacheKey || type.name;
    const previous = cache.get(key);
    const currentMembers = currentTables.get(type.name)?.members || [];
    const currentConstructors = constructors.filter(reference => reference.ownerType === type.name);
    const entry = {
      key,
      name: type.name,
      type: { ...(previous?.type || {}), ...type },
      members: deduplicateReferences([...(previous?.members || []), ...currentMembers]),
      constructors: deduplicateReferences([...(previous?.constructors || []), ...currentConstructors])
    };
    cache.set(key, entry);
    activeEntries.push(entry);
  }

  // If clang's recovery AST omits an otherwise unchanged record entirely, reuse it only when the
  // current source declares that name and the cache has a single unambiguous declaration for it.
  const activeNames = new Set(activeEntries.map(entry => entry.name));
  for (const name of declaredTypeNames(source)) {
    if (activeNames.has(name)) continue;
    const candidates = [...cache.values()].filter(entry => entry.name === name);
    if (candidates.length === 1) {
      activeEntries.push(candidates[0]);
      activeNames.add(name);
    }
  }

  astContext.types = activeEntries.map(entry => entry.type);
  astContext.membersByType = activeEntries.map(entry => ({ type: entry.name, members: entry.members }));
  astContext.functions = deduplicateReferences([
    ...nonConstructors,
    ...activeEntries.flatMap(entry => entry.constructors)
  ]);
  return astContext;
}

function completionName(item) {
  for (const candidate of [item?.textEdit?.newText, item?.insertText, item?.filterText,
    typeof item?.label === "string" ? item.label.trim() : item?.label?.label?.trim()]) {
    if (typeof candidate === "string" && candidate.length > 0 && !candidate.includes("\n")) return candidate;
  }
  return undefined;
}

function normalizedCompletion(item, receiverMember) {
  const name = completionName(item);
  if (!name) return null;
  const lspKind = Number(item.kind) || 0;
  const kind = COMPLETION_KINDS.get(lspKind) || "unknown";
  if (name === "main" && kind === "function") return null;
  const label = (typeof item.label === "string" ? item.label : item.label?.label || name).trim();
  const detail = typeof item.detail === "string" ? item.detail : undefined;
  const signatureDetail = typeof item.labelDetails?.detail === "string" ? item.labelDetails.detail : undefined;
  const reference = { name, label, kind, lspKind, source: "completion", receiverMember };
  if (detail) reference.detail = detail;
  if (typeof item.filterText === "string") reference.filterText = item.filterText;
  if (typeof item.insertText === "string") reference.insertText = item.insertText;
  if (VALUE_REFERENCE_KINDS.has(kind)) reference.type = detail || "";
  if (FUNCTION_REFERENCE_KINDS.has(kind)) {
    reference.returnType = kind === "constructor" ? name : (detail || "");
    reference.parameters = parameterClause(signatureDetail || label);
  }
  return reference;
}

function deduplicateReferences(references) {
  const result = [];
  const seen = new Set();
  for (const reference of references.filter(Boolean)) {
    const key = [reference.name, reference.kind, reference.type, reference.returnType,
      (reference.parameters || []).map(parameter => parameter.type).join(","), reference.receiverMember ? 1 : 0,
      reference.ownerType].join("\u0000");
    if (!seen.has(key)) {
      seen.add(key);
      result.push(reference);
    }
  }
  return result;
}

function hoverText(hover) {
  const contents = hover?.contents;
  if (typeof contents === "string") return contents;
  if (typeof contents?.value === "string") return contents.value;
  if (Array.isArray(contents)) return contents.map(item => typeof item === "string" ? item : item?.value || "").join("\n");
  return "";
}

function hoverSemantics(hover) {
  const text = hoverText(hover);
  const type = /^Type:\s*(.+)$/m.exec(text)?.[1]?.trim();
  const returnType = /^[→]\s*(.+)$/m.exec(text)?.[1]?.trim();
  const parameters = [];
  for (const match of text.matchAll(/^-\s+(.+)$/gm)) {
    const alias = /\s+\(aka\s+(.+)\)\s*$/.exec(match[1]);
    const label = alias == null ? match[1] : match[1].slice(0, alias.index);
    parameters.push(parameterFromLabel(label, alias?.[1]));
  }
  return { text, type, returnType, parameters };
}

function normalizedSignatures(result, callableHover = { parameters: [] }) {
  if (!Array.isArray(callableHover.parameters)) callableHover.parameters = [];
  const signatures = [];
  const activeSignature = Number(result?.activeSignature) || 0;
  const activeParameter = Number(result?.activeParameter) || 0;
  for (let index = 0; index < (result?.signatures || []).length; index++) {
    const signature = result.signatures[index];
    const arrow = signature.label.lastIndexOf("->");
    const parameters = (signature.parameters || []).map(parameter => {
      const label = Array.isArray(parameter.label)
        ? signature.label.slice(parameter.label[0], parameter.label[1])
        : parameter.label;
      return parameterFromLabel(label);
    });
    if (index === activeSignature && callableHover.parameters.length === parameters.length) {
      for (let parameter = 0; parameter < parameters.length; parameter++) {
        if (callableHover.parameters[parameter].type) parameters[parameter].type = callableHover.parameters[parameter].type;
      }
    }
    signatures.push({
      label: signature.label,
      returnType: index === activeSignature && callableHover.returnType
        ? callableHover.returnType
        : (arrow < 0 ? "" : signature.label.slice(arrow + 2).trim()),
      parameters,
      activeParameter: Math.min(activeParameter, Math.max(0, parameters.length - 1))
    });
  }
  const active = signatures[activeSignature];
  if (active != null && callableHover.parameters.length === active.parameters.length) {
    const aliases = new Map();
    for (let index = 0; index < active.parameters.length; index++) {
      const original = active.parameters[index].label;
      const alias = /^([A-Za-z_][A-Za-z_0-9]*)\b/.exec(original)?.[1];
      const actual = callableHover.parameters[index]?.type;
      if (alias && actual) aliases.set(alias, actual.replace(/\s*(?:&&|&)\s*$/, "").trim());
    }
    const valueType = aliases.get("value_type");
    if (valueType) {
      aliases.set("reference", `${valueType} &`);
      aliases.set("const_reference", `const ${valueType} &`);
    }
    for (const signature of signatures) {
      for (const parameter of signature.parameters) {
        const alias = /^([A-Za-z_][A-Za-z_0-9]*)\b/.exec(parameter.type)?.[1];
        if (alias && aliases.has(alias)) {
          parameter.type = parameter.type.replace(alias, aliases.get(alias));
        }
      }
    }
  }
  return signatures;
}

const STANDARD_TYPE_VOCABULARY = [
  "array", "basic_ostream", "basic_string", "deque", "list", "map", "multimap",
  "multiset", "optional", "ostringstream", "ostream", "ptrdiff_t", "set",
  "shared_ptr", "size_t", "span", "string", "stringstream", "tuple",
  "unique_ptr", "unordered_map", "unordered_set", "variant", "vector", "weak_ptr"
];

/**
 * clang completion details are display spellings and can omit `std::` even when the AST's
 * canonical type is namespaced. Recovery ASTs are cursor-sensitive, so a scoped local may have
 * only that completion record at some boundaries. Qualify only standard names that this source
 * itself spells with `std::`; this avoids treating an unrelated global `vector` as `std::vector`.
 */
function canonicalizeSourceStandardTypes(rawType, sourceStandardTypes) {
  if (typeof rawType !== "string" || rawType === "" || sourceStandardTypes.size === 0) return rawType;
  let canonical = rawType;
  for (const name of sourceStandardTypes) {
    const pattern = new RegExp(`(^|[^A-Za-z_0-9:])${escapedRegExp(name)}\\b`, "g");
    canonical = canonical.replace(pattern, (_, prefix) => `${prefix}std::${name}`);
  }
  return canonical;
}

function canonicalizeReferenceStandardTypes(reference, sourceStandardTypes) {
  if (reference == null) return reference;
  for (const property of ["type", "returnType", "ownerType"]) {
    reference[property] = canonicalizeSourceStandardTypes(reference[property], sourceStandardTypes);
  }
  for (const parameter of reference.parameters || []) {
    parameter.type = canonicalizeSourceStandardTypes(parameter.type, sourceStandardTypes);
  }
  return reference;
}

function normalizedContext(
  source, completionGroups, ast, signatureHelp, callableHover, receiverInfo, cursor,
  userTypeCache, receiverMemberCache, diagnostics = []
) {
  const identifiers = lexicalIdentifiers(source);
  const sourceIdentifiers = new Set(identifiers);
  const headers = [...source.matchAll(/^\s*#\s*include(?:_next)?\s*[<"]([^>"]+)[>"]/gm)]
    .map(match => match[1]);
  const typeNames = declaredTypeNames(source);
  for (const match of source.matchAll(/\b(?:bool|char|signed|unsigned|short|int|long|float|double)\b/g)) {
    typeNames.add(match[0]);
  }
  const sourceStandardTypes = new Set([...source.matchAll(
    new RegExp(`\\bstd\\s*::\\s*(${STANDARD_TYPE_VOCABULARY.join("|")})\\b`, "g")
  )].map(match => match[1]));
  const completions = deduplicateReferences(completionGroups.flatMap(group =>
    group.items.map(item => canonicalizeReferenceStandardTypes(
      normalizedCompletion(item, group.receiverMember), sourceStandardTypes
    ))
      .filter(reference => reference != null &&
        (!group.receiverMember || group.receiverOperator === "::" || INSTANCE_MEMBER_KINDS.has(reference.kind)))));
  const astContext = mergeCachedUserTypeFacts(astSemanticContext(ast, source, cursor), source, userTypeCache);
  for (const reference of [...astContext.values, ...astContext.types, ...astContext.functions]) {
    canonicalizeReferenceStandardTypes(reference, sourceStandardTypes);
  }
  for (const table of astContext.membersByType) {
    table.type = canonicalizeSourceStandardTypes(table.type, sourceStandardTypes);
    for (const member of table.members) canonicalizeReferenceStandardTypes(member, sourceStandardTypes);
  }
  for (const conversion of astContext.conversions) {
    conversion.from = canonicalizeSourceStandardTypes(conversion.from, sourceStandardTypes);
    conversion.to = canonicalizeSourceStandardTypes(conversion.to, sourceStandardTypes);
  }
  const abstractTypes = new Set(astContext.types.filter(type => type.abstract).map(type => type.name));
  const baseCompletions = completions.filter(reference => !reference.receiverMember);
  const values = deduplicateReferences([
    // AST facts are truly scoped and canonically typed; preserve that provenance when clangd also
    // returns the same declaration as a broad completion item. A fallback index may also advertise
    // header constants that this translation unit's actual SDK does not expose; completion-only
    // values are therefore used solely to recover names already spelled in the source (notably a
    // loop local omitted by clang's recovery AST at a receiver/namespace cursor).
    ...astContext.values,
    ...baseCompletions.filter(reference => VALUE_REFERENCE_KINDS.has(reference.kind) &&
      sourceIdentifiers.has(reference.name.split("::").at(-1)) &&
      !astContext.values.some(scoped => scoped.name === reference.name))
  ]);
  const types = deduplicateReferences([
    ...sourceTypeAliases(source),
    ...astContext.types,
    ...baseCompletions.filter(reference => TYPE_REFERENCE_KINDS.has(reference.kind))
  ]);
  const functions = deduplicateReferences([
    ...astContext.functions,
    ...baseCompletions.filter(reference => FUNCTION_REFERENCE_KINDS.has(reference.kind) &&
      !(reference.kind === "constructor" && abstractTypes.has(reference.name)))
  ]);
  for (const reference of [...completions, ...values, ...types, ...functions]) {
    addIdentifierWords(identifiers, reference.name);
    addIdentifierWords(typeNames, reference.type);
    addIdentifierWords(typeNames, reference.returnType);
    if (TYPE_REFERENCE_KINDS.has(reference.kind)) addIdentifierWords(typeNames, reference.name);
  }
  const signatures = normalizedSignatures(signatureHelp, callableHover);
  for (const signature of signatures) {
    signature.returnType = canonicalizeSourceStandardTypes(signature.returnType, sourceStandardTypes);
    for (const parameter of signature.parameters) {
      parameter.type = canonicalizeSourceStandardTypes(parameter.type, sourceStandardTypes);
    }
  }
  const expectedTypes = [...new Set(signatures.map(signature =>
    signature.parameters[signature.activeParameter]?.type).filter(Boolean))];
  for (const typeName of typeNames) identifiers.add(typeName);

  let receiver = null;
  if (receiverInfo != null) {
    const semantics = hoverSemantics(receiverInfo.hover);
    const valueType = values.find(reference => reference.name === receiverInfo.expression)?.type;
    const receiverType = canonicalizeSourceStandardTypes(
      semantics.type || valueType || "", sourceStandardTypes
    );
    const rejectedMembers = new Set(completions.filter(reference =>
      reference.receiverMember && isCopyingMoveOnlyVectorOverload(reference, receiverType)));
    for (let index = completions.length - 1; index >= 0; index--) {
      if (rejectedMembers.has(completions[index])) completions.splice(index, 1);
    }
    const receiverMembers = completions.filter(reference => reference.receiverMember);
    const receiverAliases = standardReceiverTypeAliases(receiverType);
    for (const member of receiverMembers) specializeReferenceAliases(member, receiverAliases);
    receiver = {
      operator: receiverInfo.operator,
      expression: receiverInfo.expression,
      type: receiverType,
      members: receiverMembers
    };
    const pointeeType = receiverPointeeType(receiver.type);
    if (receiver.operator === "->" && pointeeType) receiver.pointeeType = pointeeType;
    if (receiver.type && receiver.members.length > 0) {
      const ownerType = receiver.operator === "->" && pointeeType ? pointeeType : receiver.type;
      for (const member of receiver.members) {
        if (!member.ownerType) member.ownerType = ownerType;
      }
      const existing = astContext.membersByType.find(table => table.type === ownerType);
      if (existing == null) astContext.membersByType.push({ type: ownerType, members: receiver.members });
      else existing.members = deduplicateReferences([...existing.members, ...receiver.members]);
      rememberReceiverMembers(receiverMemberCache, receiver.type, receiver.members);
    }
  }
  mergeCachedReceiverMembers(astContext, values, receiverMemberCache);
  const unresolvedIdentifiers = unresolvedDiagnosticIdentifiers(source, diagnostics);
  const requiredIdentifier = unresolvedIdentifiers[0] || futureRequiredIdentifier(
    source, cursor, values, typeNames, functions
  );
  if (requiredIdentifier && !unresolvedIdentifiers.includes(requiredIdentifier)) {
    unresolvedIdentifiers.push(requiredIdentifier);
  }
  return {
    identifiers: [...identifiers].sort(),
    sourceIdentifiers: [...sourceIdentifiers].sort(),
    headers: [...new Set(headers)].sort(),
    typeNames: [...typeNames].sort(),
    values,
    types,
    functions,
    completions,
    signatures,
    expectedTypes,
    receiver,
    membersByType: astContext.membersByType,
    conversions: astContext.conversions,
    unresolvedIdentifiers,
    requiredIdentifier,
    enclosingReturnType: astContext.enclosingReturnType,
    enclosingClassType: astContext.enclosingClassType,
    thisType: astContext.thisType,
    mutableFields: astContext.mutableFields
  };
}

function receiverPointeeType(rawType) {
  if (typeof rawType !== "string" || rawType.trim() === "") return undefined;
  const type = rawType.replace(/\b(?:const|volatile)\b/g, " ").replace(/\s+/g, " ")
    .replace(/(?:&&|&)\s*$/, "").trim();
  if (/\*\s*$/.test(type)) return type.replace(/\*\s*$/, "").trim();
  const smartPointer = /(?:^|::)(?:unique_ptr|shared_ptr|weak_ptr)\s*</g;
  let match;
  let open = -1;
  while ((match = smartPointer.exec(type)) != null) open = type.indexOf("<", match.index);
  if (open < 0) return undefined;
  let depth = 0;
  for (let index = open; index < type.length; index++) {
    if (type[index] === "<") depth += 1;
    else if (type[index] === ">" && --depth === 0) {
      return splitTopLevel(type.slice(open + 1, index))[0];
    }
  }
  return undefined;
}

function standardVectorValueType(rawType) {
  if (typeof rawType !== "string") return undefined;
  const type = rawType.replace(/\b(?:const|volatile)\b/g, " ").replace(/(?:&&|&)\s*$/, "")
    .replace(/\s+/g, " ").trim();
  const vector = /(?:^|::)vector\s*</g;
  let match;
  let open = -1;
  while ((match = vector.exec(type)) != null) open = type.indexOf("<", match.index);
  if (open < 0) return undefined;
  let depth = 0;
  for (let index = open; index < type.length; index++) {
    if (type[index] === "<") depth += 1;
    else if (type[index] === ">" && --depth === 0) {
      const valueType = splitTopLevel(type.slice(open + 1, index))[0];
      return valueType || undefined;
    }
  }
  return undefined;
}

function standardReceiverTypeAliases(rawType) {
  const valueType = standardVectorValueType(rawType);
  if (!valueType) return new Map();
  return new Map([
    ["const_reference", `const ${valueType} &`],
    ["difference_type", "std::ptrdiff_t"],
    ["value_type", valueType],
    ["reference", `${valueType} &`],
    ["size_type", "std::size_t"]
  ]);
}

function isCopyingMoveOnlyVectorOverload(reference, receiverType) {
  const valueType = standardVectorValueType(receiverType);
  if (!valueType || !/(?:^|::)unique_ptr\s*</.test(valueType)) return false;
  if (!new Set(["assign", "insert", "push_back", "resize"]).has(reference.name)) return false;
  const aliases = standardReceiverTypeAliases(receiverType);
  const compact = value => String(value || "").replace(/\s*::\s*/g, "::")
    .replace(/\s*<\s*/g, "<").replace(/\s*>\s*/g, ">")
    .replace(/\s*,\s*/g, ",").replace(/\s+/g, " ").trim();
  return (reference.parameters || []).some(parameter => {
    const original = parameter.type || parameter.label || "";
    if (/\bconst_reference\b/.test(original)) return true;
    const specialized = specializeTypeAliases(original, aliases).trim();
    if (!/\bconst\b/.test(specialized) || !specialized.endsWith("&") || specialized.endsWith("&&")) return false;
    const pointee = compact(specialized.replace(/\bconst\b/g, "").replace(/&\s*$/, ""));
    return pointee === compact(valueType);
  });
}

function canonicalReceiverCacheType(rawType) {
  if (typeof rawType !== "string" || rawType.trim() === "") return undefined;
  return rawType.trim().replace(/(?:&&|&)\s*$/, "").trim()
    .replace(/\s*::\s*/g, "::")
    .replace(/\s*<\s*/g, "<")
    .replace(/\s*>\s*/g, ">")
    .replace(/\s*,\s*/g, ",")
    .replace(/\s+/g, " ");
}

function rememberReceiverMembers(cache, receiverType, members) {
  if (!(cache instanceof Map) || !Array.isArray(members) || members.length === 0) return;
  const key = canonicalReceiverCacheType(receiverType);
  if (!key) return;
  const previous = cache.get(key);
  cache.set(key, {
    type: receiverType,
    members: deduplicateReferences([...(previous?.members || []), ...members])
  });
}

function mergeCachedReceiverMembers(astContext, values, cache) {
  if (!(cache instanceof Map)) return;
  for (const value of values) {
    const key = canonicalReceiverCacheType(value.type);
    const cached = key == null ? null : cache.get(key);
    if (cached == null) continue;
    const existing = astContext.membersByType.find(table =>
      canonicalReceiverCacheType(table.type) === key);
    if (existing == null) {
      astContext.membersByType.push({ type: cached.type, members: cached.members });
    } else {
      existing.members = deduplicateReferences([...existing.members, ...cached.members]);
    }
  }
}

function specializeTypeAliases(value, aliases) {
  if (typeof value !== "string" || aliases.size === 0) return value;
  let specialized = value;
  for (const [alias, concrete] of [...aliases].sort((left, right) => right[0].length - left[0].length)) {
    specialized = specialized.replace(new RegExp(`\\b${escapedRegExp(alias)}\\b`, "g"), concrete);
  }
  return specialized;
}

function specializeReferenceAliases(reference, aliases) {
  if (reference == null || aliases.size === 0) return reference;
  for (const property of ["type", "returnType", "detail", "label"]) {
    if (typeof reference[property] === "string") {
      reference[property] = specializeTypeAliases(reference[property], aliases);
    }
  }
  for (const parameter of reference.parameters || []) {
    for (const property of ["type", "label"]) {
      if (typeof parameter[property] === "string") {
        parameter[property] = specializeTypeAliases(parameter[property], aliases);
      }
    }
  }
  return reference;
}

function escapedRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function usefulCompletionProbes(source, beforeCursor) {
  // Never trim across a newline: on an empty replacement line that made the previous statement's
  // closing ')' look like the cursor token and caused a stray '.' probe on the new line.
  const currentLine = beforeCursor.split(/\r?\n/).at(-1) || "";
  if (/(?:\.|->|::)\s*$/.test(currentLine)) return [""];
  const trimmed = currentLine.trimEnd();
  if (trimmed.length === 0) return [""];
  const identifier = /([A-Za-z_][A-Za-z_0-9]*)$/.exec(trimmed)?.[1];
  if (identifier != null) {
    const escaped = escapedRegExp(identifier);
    const typeOrNamespace = identifier === "std" || new RegExp(
      `\\b(?:class|struct|union|namespace)\\s+${escaped}\\b|\\benum\\s+(?:class\\s+|struct\\s+)?${escaped}\\b`
    ).test(source);
    if (typeOrNamespace) return ["", "::"];
    const pointer = new RegExp(
      `(?:\\*+\\s*(?:const\\s+)?${escaped}\\b)|(?:\\b(?:unique_ptr|shared_ptr|weak_ptr)\\s*<[^;{}]*>\\s*${escaped}\\b)`
    ).test(source);
    return ["", pointer ? "->" : "."];
  }
  return /[)\]]$/.test(trimmed) ? ["", "."] : [""];
}

/** Signature help and callable hover have no active expression on an indent-only cursor line. */
function hasActiveExpressionPrefix(beforeCursor) {
  const currentLine = beforeCursor.split(/\r?\n/).at(-1) || "";
  return currentLine.trim().length > 0;
}

/**
 * Probe the most recently scoped standard-library object whose type table has not been cached yet.
 * The probe replaces the whole cursor line instead of being appended at the live cursor. It is
 * therefore valid at every token boundary, including the middle of a call/template expression,
 * and cannot mix its members with an actual `.`, `->`, or `::` receiver at that boundary.
 */
function missingScopedReceiverProbe(ast, source, cursor, receiverMemberCache) {
  const line = source.replace(/\r\n?/g, "\n").split("\n")[cursor.line] || "";
  const scoped = astSemanticContext(ast, source, cursor).values;
  for (let index = scoped.length - 1; index >= 0; index--) {
    const value = scoped[index];
    const name = value.name;
    const type = canonicalReceiverCacheType(value.type);
    if (!/^[A-Za-z_][A-Za-z_0-9]*$/.test(name || "") || type == null ||
        !/(?:^|\b)std::/.test(type) || receiverMemberCache.has(type)) continue;
    const replacement = `${name}.`;
    const probeSource = replaceCursorLine(source, cursor.line, replacement);
    if (probeSource == null) return undefined;
    const indentation = /^\s*/.exec(line)?.[0] || "";
    return {
      source: probeSource,
      position: { line: cursor.line, character: indentation.length + replacement.length },
      name,
      type: value.type,
      operator: "."
    };
  }
  return undefined;
}

function cacheScopedReceiverCompletion(receiverMemberCache, probe, result) {
  const aliases = standardReceiverTypeAliases(probe.type);
  const members = deduplicateReferences(completionItems(result)
    .map(item => normalizedCompletion(item, true))
    .filter(reference => reference != null && INSTANCE_MEMBER_KINDS.has(reference.kind) &&
      !isCopyingMoveOnlyVectorOverload(reference, probe.type)));
  for (const member of members) {
    specializeReferenceAliases(member, aliases);
    if (!member.ownerType) member.ownerType = probe.type;
  }
  rememberReceiverMembers(receiverMemberCache, probe.type, members);
}

function trailingReceiverExpression(linePrefix, operator) {
  const beforeOperator = linePrefix.slice(0, linePrefix.length - operator.length).trimEnd();
  let round = 0;
  let square = 0;
  let brace = 0;
  let angle = 0;
  for (let index = beforeOperator.length - 1; index >= 0; index--) {
    const character = beforeOperator[index];
    if (character === ")") round += 1;
    else if (character === "(" && round > 0) round -= 1;
    else if (character === "]") square += 1;
    else if (character === "[" && square > 0) square -= 1;
    else if (character === "}") brace += 1;
    else if (character === "{" && brace > 0) brace -= 1;
    else if (character === ">" && beforeOperator[index - 1] !== "-") angle += 1;
    else if (character === "<" && angle > 0) angle -= 1;
    else if (round + square + brace + angle === 0 && /[;{}=,+!&|?:]/.test(character)) {
      return beforeOperator.slice(index + 1).trim();
    } else if (round + square + brace + angle === 0 && /\s/.test(character)) {
      return beforeOperator.slice(index + 1).trim();
    }
  }
  return beforeOperator.trim();
}

function receiverProbeInfo(beforeCursor, line, character, probe) {
  const linePrefix = (beforeCursor + probe).split(/\r?\n/).at(-1) || "";
  const match = /(->|::|\.)\s*$/.exec(linePrefix);
  if (match == null) return null;
  const operator = match[1];
  return {
    operator,
    expression: trailingReceiverExpression(linePrefix.slice(0, match.index + operator.length), operator),
    hoverPosition: {
      line,
      character: Math.max(0, character + probe.length - operator.length - 1)
    }
  };
}

function callableHoverPosition(beforeCursor, line) {
  const linePrefix = beforeCursor.split(/\r?\n/).at(-1) || "";
  const opens = [];
  let quote = null;
  let escaped = false;
  for (let index = 0; index < linePrefix.length; index++) {
    const character = linePrefix[index];
    if (quote != null) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === quote) quote = null;
    } else if (character === "\"" || character === "'") quote = character;
    else if (character === "(") opens.push(index);
    else if (character === ")") opens.pop();
  }
  const open = opens.at(-1);
  if (open == null) return null;
  const name = /(?:[A-Za-z_][A-Za-z_0-9]*|operator\s*[^\s(]+)\s*$/.exec(linePrefix.slice(0, open));
  if (name == null) return null;
  return { line, character: Math.max(0, name.index + name[0].trimStart().length - 1) };
}

class ClangdLspSession {
  constructor(options) {
    this.clangd = options.clangd;
    this.clangxx = options.clangxx;
    this.log = options.log;
    this.workspace = options.workspace;
    this.sourcePath = path.join(this.workspace, "main.cpp");
    this.uri = pathToFileURL(this.sourcePath).href;
    this.process = null;
    this.buffer = Buffer.alloc(0);
    this.nextId = 1;
    this.pending = new Map();
    this.open = false;
    this.version = 0;
    this.documentKey = "main.cpp";
    this.ready = null;
    this.queue = Promise.resolve();
    this.closed = false;
    this.diagnosticsByVersion = new Map();
    this.diagnosticWaiters = new Map();
    this.userTypeCache = new Map();
    this.receiverMemberCache = new Map();
  }

  start(deadline) {
    if (this.closed) return Promise.reject(new Error("clangd session is closed"));
    if (this.ready != null) return this.ready;
    const attempt = this.startInternal(deadline);
    this.ready = attempt;
    attempt.catch(() => {
      if (this.ready === attempt) this.ready = null;
    });
    return attempt;
  }

  async startInternal(deadline) {
    fs.mkdirSync(this.workspace, { recursive: true });
    fs.writeFileSync(this.sourcePath, "", "utf8");
    fs.writeFileSync(
      path.join(this.workspace, "compile_commands.json"),
      JSON.stringify([
        {
          directory: this.workspace,
          file: this.sourcePath,
          arguments: [
            this.clangxx,
            "-xc++",
            "-std=c++23",
            "-pedantic-errors",
            "-Wall",
            "-Wextra",
            this.sourcePath
          ]
        }
      ]),
      "utf8"
    );

    const spawned = childProcess.spawn(
      this.clangd,
      [
        `--compile-commands-dir=${this.workspace}`,
        "--background-index=0",
        "--clang-tidy=0",
        "--header-insertion=never",
        "--limit-results=500",
        "--log=error"
      ],
      { cwd: this.workspace, stdio: ["pipe", "pipe", "pipe"] }
    );
    this.process = spawned;
    spawned.stdout.on("data", chunk => {
      if (this.process === spawned) this.accept(chunk);
    });
    spawned.stderr.on("data", chunk => {
      const text = chunk.toString("utf8").trim();
      if (text.length > 0) this.log.debug(`[clangd] ${text}`);
    });
    spawned.stdin.on("error", error => {
      if (this.process === spawned) this.failAll(error);
    });
    spawned.on("error", error => {
      if (this.process === spawned) this.failAll(error);
    });
    spawned.on("exit", (code, signal) => {
      if (this.process === spawned) {
        this.failAll(new Error(`clangd exited (${code ?? signal ?? "unknown"})`));
        this.process = null;
        this.ready = null;
        this.open = false;
      }
    });

    await this.request("initialize", {
      processId: process.pid,
      rootUri: pathToFileURL(this.workspace).href,
      capabilities: {
        textDocument: {
          completion: {
            completionItem: {
              snippetSupport: false,
              labelDetailsSupport: true
            }
          },
          documentSymbol: { hierarchicalDocumentSymbolSupport: true }
        },
        workspace: { symbol: {} }
      },
      initializationOptions: {
        fallbackFlags: ["-xc++", "-std=c++23", "-pedantic-errors"]
      },
      workspaceFolders: [
        { uri: pathToFileURL(this.workspace).href, name: "cpp-completion-benchmark" }
      ]
    }, timeoutWithin(deadline, REQUEST_TIMEOUT_MS, "clangd initialization"));
    this.notify("initialized", {});
  }

  enqueue(operation, shouldRecover = () => true) {
    const task = this.queue.then(operation, operation);
    // A failed operation may have changed the document or left clangd working on a cancelled
    // request. Recover the session before allowing the next queued context request to start.
    this.queue = task.then(
      () => undefined,
      error => shouldRecover(error) ? this.recover(error) : undefined
    );
    return task;
  }

  /**
   * Build clangd's first preamble while Karma is still starting the browser. The benchmark's first
   * context then changes one statement line in an already parsed document instead of paying for
   * process initialization, SDK header parsing, and its semantic query on the critical path.
   * `enqueue` is important here: an unusually fast browser request waits for this synchronization
   * rather than racing a didChange against it.
   */
  prime(source, documentKey = "main.cpp") {
    let sessionMayNeedRecovery = false;
    return this.enqueue(async () => {
      // Queueing is intentionally bounded by the browser's prefetch window. Start the native
      // operation deadline only when this document reaches clangd, rather than expiring a healthy
      // request merely because earlier document versions were still being processed.
      const deadline = Date.now() + CONTEXT_TIMEOUT_MS;
      timeoutWithin(deadline, CONTEXT_TIMEOUT_MS, "clangd preamble warmup queue");
      sessionMayNeedRecovery = true;
      await this.start(deadline);
      this.selectDocument(documentKey);
      this.updateDocument(source);
      await this.request("textDocument/documentSymbol", {
        textDocument: { uri: this.uri }
      }, timeoutWithin(deadline, LSP_STEP_TIMEOUT_MS, "clangd preamble warmup"));
    }, () => sessionMayNeedRecovery);
  }

  async context(source, line, character, documentKey = "main.cpp", includeOracle = false) {
    const baseOffset = offsetAt(source, line, character);
    const beforeCursor = source.slice(0, baseOffset);
    let sessionMayNeedRecovery = false;
    return this.enqueue(async () => {
      const deadline = Date.now() + CONTEXT_TIMEOUT_MS;
      timeoutWithin(deadline, CONTEXT_TIMEOUT_MS, "completion context queue");
      sessionMayNeedRecovery = true;
      await this.start(deadline);
      this.selectDocument(documentKey);
      const completionGroups = [];
      let ast = null;
      let signatureHelp = { signatures: [] };
      const activeExpressionPrefix = hasActiveExpressionPrefix(beforeCursor);
      const baseReceiver = receiverProbeInfo(beforeCursor, line, character, "");
      const callablePosition = activeExpressionPrefix && baseReceiver == null
        ? callableHoverPosition(beforeCursor, line)
        : null;
      // A completion request after `.`, `->`, or `::` is intentionally receiver-filtered and can
      // omit unrelated locals that are still needed later in the statement. Query the same line
      // immediately before its first token as an ordinary completion point; this preserves the
      // exact lexical scope (including an unbraced for-loop body) without speculative edits.
      const scopeCharacter = /^\s*/.exec(source.split("\n")[line] || "")?.[0].length || 0;

      this.updateDocument(source);
      const baseDocumentVersion = this.version;
      const optionalRequest = async (method, params, operation) => {
        try {
          return await this.request(method, params,
            timeoutWithin(deadline, LSP_STEP_TIMEOUT_MS, operation));
        } catch (error) {
          this.log.debug(`${operation} failed: ${error.message}`);
          return null;
        }
      };
      // Exercise the same single structured request as the browser whenever the selected clangd
      // contains tidyparse's private Sema endpoint. Stock clangd remains available only for the
      // historical native-oracle diagnostics below.
      if (!includeOracle) {
        const semantic = await optionalRequest("tidyparse/semanticCompletion", {
          textDocument: { uri: this.uri },
          position: { line, character },
          scopePosition: { line, character: scopeCharacter },
          allScopes: /[A-Za-z_][A-Za-z_0-9]*$/.test(beforeCursor),
          limit: 128,
          context: { triggerKind: 1 }
        }, "clangd semantic completion");
        if (semantic?.schemaVersion === 1) return { semantic };
      }
      // The AST request waits for the current document version. Without this barrier clangd may
      // legitimately answer completion from its lexical fallback index while rebuilding the AST,
      // which admits out-of-scope and even comment words as candidates.
      ast = await optionalRequest("textDocument/ast", {
        textDocument: { uri: this.uri }
      }, "clangd AST");
      if (ast == null) {
        await optionalRequest("textDocument/documentSymbol", {
          textDocument: { uri: this.uri }
        }, "clangd document synchronization");
      }
      // Diagnostics for this document version are normally published before clangd satisfies the
      // AST barrier. Start a short fallback wait only after that barrier so a cold first parse does
      // not exhaust the wait while clangd is still building its preamble.
      const baseDiagnostics = this.diagnosticsForVersion(
        baseDocumentVersion,
        timeoutWithin(deadline, 500, "clangd diagnostics")
      );
      const baseCompletion = this.request("textDocument/completion", {
        textDocument: { uri: this.uri },
        position: { line, character },
        context: baseReceiver == null
          ? { triggerKind: 1 }
          : { triggerKind: 2, triggerCharacter: baseReceiver.operator.slice(-1) }
      }, timeoutWithin(deadline, LSP_STEP_TIMEOUT_MS, "clangd completion"));
      const [baseResult, signatureResult, hoverResult, scopeResult] = await Promise.all([
        baseCompletion,
        activeExpressionPrefix ? optionalRequest("textDocument/signatureHelp", {
          textDocument: { uri: this.uri },
          position: { line, character },
          context: { triggerKind: 1, isRetrigger: false }
        }, "clangd signature help") : Promise.resolve(null),
        activeExpressionPrefix && (baseReceiver != null || callablePosition != null)
          ? optionalRequest("textDocument/hover", {
            textDocument: { uri: this.uri },
            position: baseReceiver?.hoverPosition || callablePosition
          }, "clangd hover")
          : Promise.resolve(null),
        character === scopeCharacter ? Promise.resolve(null) : optionalRequest("textDocument/completion", {
          textDocument: { uri: this.uri },
          position: { line, character: scopeCharacter },
          context: { triggerKind: 1 }
        }, "clangd scoped-value completion")
      ]);
      signatureHelp = signatureResult || signatureHelp;
      completionGroups.push({
        items: completionItems(baseResult),
        receiverMember: baseReceiver != null,
        receiverOperator: baseReceiver?.operator
      });
      if (scopeResult != null) completionGroups.push({
        items: completionItems(scopeResult),
        receiverMember: false,
        receiverOperator: null
      });
      timeoutWithin(deadline, CONTEXT_TIMEOUT_MS, "completion context");
      const diagnostics = await baseDiagnostics;
      const browserFacts = {
        completionGroups,
        ast,
        signatures: signatureHelp,
        hover: hoverResult,
        diagnostics
      };
      if (!includeOracle) return browserFacts;

      // Retain the historical native oracle only for focused middleware/grammar diagnostics. The
      // scored benchmark deliberately takes the browser-facts branch above.
      const probes = usefulCompletionProbes(source, beforeCursor);
      const scopedReceiverProbe = ast == null ? null : missingScopedReceiverProbe(
        ast, source, { line, character }, this.receiverMemberCache
      );
      let callableHover = { parameters: [] };
      let receiverInfo = null;
      let documentChanged = false;
      if (baseReceiver != null) receiverInfo = { ...baseReceiver, hover: hoverResult };
      else callableHover = hoverSemantics(hoverResult);
      try {
        if (scopedReceiverProbe != null) {
          this.updateDocument(scopedReceiverProbe.source);
          documentChanged = true;
          await optionalRequest("textDocument/documentSymbol", {
            textDocument: { uri: this.uri }
          }, "clangd scoped-receiver synchronization");
          const result = await this.request("textDocument/completion", {
            textDocument: { uri: this.uri },
            position: scopedReceiverProbe.position,
            context: { triggerKind: 2, triggerCharacter: scopedReceiverProbe.operator }
          }, timeoutWithin(deadline, LSP_STEP_TIMEOUT_MS, "clangd scoped-receiver completion"));
          cacheScopedReceiverCompletion(this.receiverMemberCache, scopedReceiverProbe, result);
        }
        for (const probe of probes.filter(value => value !== "")) {
          if (deadline - Date.now() < 250) break;
          const probeSource = source.slice(0, baseOffset) + probe + source.slice(baseOffset);
          const probeReceiver = receiverProbeInfo(beforeCursor, line, character, probe);
          this.updateDocument(probeSource);
          documentChanged = true;
          await optionalRequest("textDocument/documentSymbol", {
            textDocument: { uri: this.uri }
          }, "clangd receiver synchronization");
          const [result, hover] = await Promise.all([
            this.request("textDocument/completion", {
              textDocument: { uri: this.uri },
              position: { line, character: character + probe.length },
              context: { triggerKind: 2, triggerCharacter: probe.slice(-1) }
            }, timeoutWithin(deadline, LSP_STEP_TIMEOUT_MS, "clangd receiver completion")),
            probeReceiver == null ? Promise.resolve(null) : optionalRequest("textDocument/hover", {
              textDocument: { uri: this.uri },
              position: probeReceiver.hoverPosition
            }, "clangd receiver hover")
          ]);
          completionGroups.push({
            items: completionItems(result),
            receiverMember: true,
            receiverOperator: probeReceiver?.operator
          });
          if (receiverInfo == null && probeReceiver != null) receiverInfo = { ...probeReceiver, hover };
        }
      } finally {
        if (documentChanged) this.updateDocument(source);
      }
      timeoutWithin(deadline, CONTEXT_TIMEOUT_MS, "completion oracle context");
      return normalizedContext(
        source, completionGroups, ast, signatureHelp, callableHover, receiverInfo,
        { line, character }, this.userTypeCache, this.receiverMemberCache, diagnostics
      );
    }, () => sessionMayNeedRecovery);
  }

  diagnosticsForVersion(version, timeout) {
    if (this.diagnosticsByVersion.has(version)) {
      return Promise.resolve(this.diagnosticsByVersion.get(version));
    }
    return new Promise(resolve => {
      const waiter = { resolve, timer: null };
      waiter.timer = setTimeout(() => {
        const waiters = this.diagnosticWaiters.get(version) || [];
        const retained = waiters.filter(candidate => candidate !== waiter);
        if (retained.length === 0) this.diagnosticWaiters.delete(version);
        else this.diagnosticWaiters.set(version, retained);
        resolve([]);
      }, timeout);
      this.diagnosticWaiters.set(version, [
        ...(this.diagnosticWaiters.get(version) || []),
        waiter
      ]);
    });
  }

  acceptDiagnostics(params) {
    if (params?.uri !== this.uri) return;
    const version = Number.isInteger(params.version) ? params.version : this.version;
    const diagnostics = Array.isArray(params.diagnostics) ? params.diagnostics : [];
    this.diagnosticsByVersion.set(version, diagnostics);
    while (this.diagnosticsByVersion.size > 8) {
      this.diagnosticsByVersion.delete(this.diagnosticsByVersion.keys().next().value);
    }
    const waiters = this.diagnosticWaiters.get(version) || [];
    this.diagnosticWaiters.delete(version);
    for (const waiter of waiters) {
      clearTimeout(waiter.timer);
      waiter.resolve(diagnostics);
    }
  }

  clearDiagnosticState() {
    this.diagnosticsByVersion.clear();
    for (const waiters of this.diagnosticWaiters.values()) for (const waiter of waiters) {
      clearTimeout(waiter.timer);
      waiter.resolve([]);
    }
    this.diagnosticWaiters.clear();
  }

  /** A fixture boundary is a new translation unit, not an incremental edit of the previous one. */
  selectDocument(documentKey) {
    const key = typeof documentKey === "string" && documentKey !== ""
      ? documentKey
      : "main.cpp";
    if (key === this.documentKey) return;
    if (this.open) {
      this.notify("textDocument/didClose", { textDocument: { uri: this.uri } });
    }
    this.open = false;
    this.version = 0;
    this.documentKey = key;
    const basename = path.basename(key).replace(/[^A-Za-z_0-9.-]/g, "_");
    const filename = basename.endsWith(".cpp") ? basename : `${basename}.cpp`;
    this.sourcePath = path.join(this.workspace, filename);
    this.uri = pathToFileURL(this.sourcePath).href;
    this.clearDiagnosticState();
    this.userTypeCache.clear();
    this.receiverMemberCache.clear();
  }

  updateDocument(source) {
    fs.writeFileSync(this.sourcePath, source, "utf8");
    this.version += 1;
    if (!this.open) {
      this.notify("textDocument/didOpen", {
        textDocument: {
          uri: this.uri,
          languageId: "cpp",
          version: this.version,
          text: source
        }
      });
      this.open = true;
    } else {
      this.notify("textDocument/didChange", {
        textDocument: { uri: this.uri, version: this.version },
        contentChanges: [{ text: source }]
      });
    }
  }

  request(method, params, timeout = REQUEST_TIMEOUT_MS) {
    if (this.process == null) return Promise.reject(new Error("clangd is not running"));
    const boundedTimeout = Math.max(1, Math.min(REQUEST_TIMEOUT_MS, timeout));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        try {
          this.notify("$/cancelRequest", { id });
        } catch (_) {
        }
        reject(new OperationTimeoutError(`${method} timed out after ${boundedTimeout}ms`));
      }, boundedTimeout);
      this.pending.set(id, {
        resolve: value => { clearTimeout(timer); resolve(value); },
        reject: error => { clearTimeout(timer); reject(error); }
      });
      try {
        this.send({ jsonrpc: "2.0", id, method, params });
      } catch (error) {
        this.pending.delete(id);
        clearTimeout(timer);
        reject(error);
      }
    });
  }

  notify(method, params) {
    this.send({ jsonrpc: "2.0", method, params });
  }

  send(message) {
    if (this.process == null || this.process.stdin.destroyed) {
      throw new Error("clangd stdin is unavailable");
    }
    const body = Buffer.from(JSON.stringify(message), "utf8");
    this.process.stdin.write(`Content-Length: ${body.length}\r\n\r\n`);
    this.process.stdin.write(body);
  }

  accept(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (true) {
      const headerEnd = this.buffer.indexOf("\r\n\r\n");
      if (headerEnd < 0) return;
      const header = this.buffer.slice(0, headerEnd).toString("ascii");
      const match = /(?:^|\r\n)Content-Length:\s*(\d+)/i.exec(header);
      if (match == null) {
        this.failAll(new Error(`Malformed clangd header: ${header}`));
        this.buffer = Buffer.alloc(0);
        return;
      }
      const length = Number(match[1]);
      const bodyStart = headerEnd + 4;
      if (this.buffer.length < bodyStart + length) return;
      const body = this.buffer.slice(bodyStart, bodyStart + length).toString("utf8");
      this.buffer = this.buffer.slice(bodyStart + length);
      try {
        this.dispatch(JSON.parse(body));
      } catch (error) {
        this.failAll(new Error(`Malformed clangd JSON: ${error.message}`));
      }
    }
  }

  dispatch(message) {
    if (message?.method === "textDocument/publishDiagnostics") {
      this.acceptDiagnostics(message.params);
      return;
    }
    if (message == null || message.id == null) return;
    const pending = this.pending.get(Number(message.id));
    if (pending == null) return;
    this.pending.delete(Number(message.id));
    if (message.error != null) {
      pending.reject(new Error(message.error.message || JSON.stringify(message.error)));
    } else {
      pending.resolve(message.result);
    }
  }

  failAll(error) {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  async recover(error) {
    const running = this.process;
    this.process = null;
    this.ready = null;
    this.open = false;
    this.version = 0;
    this.buffer = Buffer.alloc(0);
    this.clearDiagnosticState();
    this.failAll(error);
    if (running == null) return;
    this.log.debug(`Restarting clangd after context failure: ${error.message}`);
    const exited = running.exitCode == null
      ? new Promise(resolve => running.once("exit", resolve))
      : Promise.resolve();
    try {
      running.stdin.destroy();
      running.kill("SIGKILL");
    } catch (_) {
    }
    await Promise.race([
      exited,
      new Promise(resolve => setTimeout(resolve, 500))
    ]);
  }

  async close() {
    this.closed = true;
    this.clearDiagnosticState();
    const running = this.process;
    if (running == null) {
      this.failAll(new Error("clangd session closed"));
      return;
    }

    const exited = running.exitCode == null
      ? new Promise(resolve => running.once("exit", resolve))
      : Promise.resolve();
    try {
      await this.request("shutdown", null, 1_000);
    } catch (_) {
    }
    if (this.process === running) {
      try {
        this.notify("exit", null);
        running.stdin.end();
      } catch (_) {
      }
    }
    await Promise.race([
      exited,
      new Promise(resolve => setTimeout(resolve, 1_000))
    ]);
    if (this.process === running) {
      try {
        running.kill("SIGKILL");
      } catch (_) {
      }
      await Promise.race([
        exited,
        new Promise(resolve => setTimeout(resolve, 1_000))
      ]);
    }
  }
}

function sourceDigest(source) {
  return crypto.createHash("sha256").update(source).digest("hex");
}

function separateIncludes(source) {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const includes = [];
  for (let index = 0; index < lines.length; index++) {
    if (!/^\s*#\s*include(?:_next)?\b/.test(lines[index])) continue;
    const directive = [lines[index]];
    lines[index] = "";
    while (/\\\s*$/.test(directive[directive.length - 1]) && index + 1 < lines.length) {
      index += 1;
      directive.push(lines[index]);
      lines[index] = "";
    }
    includes.push(directive.join("\n").trim());
  }
  const macros = [];
  const seenMacros = new Set();
  for (const line of lines) {
    const macro = /^\s*#\s*(?:define|undef)\s+([A-Za-z_][A-Za-z_0-9]*)\b/.exec(line)?.[1];
    if (macro != null && !seenMacros.has(macro)) {
      seenMacros.add(macro);
      macros.push(macro);
    }
  }
  return { includes, macros, body: lines.join("\n") };
}

/** The exact, ordered include sequence which the combined TU would place before every body. */
function exactIncludePreamble(sources) {
  const includes = [];
  const seenIncludes = new Set();
  for (const source of sources) {
    for (const directive of separateIncludes(source).includes) {
      if (seenIncludes.has(directive)) continue;
      seenIncludes.add(directive);
      includes.push(directive);
    }
  }
  return {
    directives: includes,
    source: includes.length === 0 ? "" : `${includes.join("\n")}\n`,
    // Inclusion order can affect declarations and macros, so it is deliberately part of the key.
    key: sourceDigest(JSON.stringify(includes))
  };
}

function combinedTranslationUnit(sources, omitIncludes = false) {
  const includes = [];
  const seenIncludes = new Set();
  const bodies = sources.map((source, index) => {
    const separated = separateIncludes(source);
    for (const directive of separated.includes) {
      if (seenIncludes.has(directive)) continue;
      seenIncludes.add(directive);
      includes.push(directive);
    }
    return {
      source: separated.body.replaceAll(
        CPP_COMPLETION_BUNDLE_MARKER,
        `cpp_completion_${index}`
      ),
      macros: separated.macros
    };
  });
  const sections = omitIncludes ? [""] : [...includes, ""];
  for (let index = 0; index < bodies.length; index++) {
    const macros = [...new Set(["main", ...bodies[index].macros])];
    sections.push(
      `namespace tidyparse_cpp_completion_sample_${index} {`,
      ...macros.map(macro => `#pragma push_macro("${macro}")`),
      `#define main tidyparse_cpp_completion_main_${index}`,
      `#line 1 "cpp_completion_${index}.cpp"`,
      bodies[index].source,
      ...macros.reverse().map(macro => `#pragma pop_macro("${macro}")`),
      `} // namespace tidyparse_cpp_completion_sample_${index}`,
      ""
    );
  }
  return sections.join("\n");
}

function batchDiagnosticClassification(diagnostics, sourceCount, exitCode, signal, extraError) {
  const perSource = Array.from({ length: sourceCount }, () => []);
  const failed = new Set();
  const globalErrors = [];
  let currentSource = null;
  for (const line of diagnostics.split(/\r?\n/)) {
    const sample = /^(?:.*[\\/])?cpp_completion_(\d+)(?:_(?:candidate_\d+|baseline))?\.cpp:\d+(?::\d+)?:\s*(fatal error|error|warning|note):/i.exec(line);
    if (sample != null) {
      const index = Number(sample[1]);
      if (index < 0 || index >= sourceCount) {
        globalErrors.push(line);
        currentSource = null;
      } else {
        currentSource = index;
        perSource[index].push(line);
        if (/^(?:fatal error|error)$/i.test(sample[2])) failed.add(index);
      }
      continue;
    }
    const globalDiagnostic = /^(?:clang(?:\+\+)?(?:-\d+)?):\s*(?:fatal )?error:/i.test(line) ||
      /^.+?:\d+(?::\d+)?:\s*(?:fatal )?error:/i.test(line) ||
      /^(?:fatal )?error:/i.test(line);
    if (globalDiagnostic) {
      globalErrors.push(line);
      currentSource = null;
    } else if (currentSource != null && line.length > 0) {
      perSource[currentSource].push(line);
    }
  }
  if (extraError != null) globalErrors.push(extraError);
  if (signal != null) {
    globalErrors.push(`clang++ was terminated by ${signal}`);
  } else if (exitCode != null && exitCode !== 0 && exitCode !== 1) {
    globalErrors.push(`clang++ exited unexpectedly with status ${exitCode}`);
  } else if (exitCode !== 0 && failed.size === 0 && globalErrors.length === 0) {
    globalErrors.push(`clang++ exited (${exitCode ?? signal ?? "unknown"}) without attributable diagnostics`);
  }
  return { failed, globalErrors, perSource };
}

class NativeCompiler {
  constructor(executable, parallelism = 1, options = {}) {
    this.executable = executable;
    this.parallelism = Math.max(1, parallelism);
    this.pchDirectory = typeof options.pchDirectory === "string"
      ? path.resolve(options.pchDirectory)
      : null;
    this.pchCache = new Map();
    // Precision bundles can be large enough to occupy every clang++ process for several seconds.
    // Keep one process slot available for the small declaration-oracle probes on `/context` so a
    // prefetched context cannot be rejected solely because scoring happened to start first.
    this.normalCompilerLimit = this.parallelism > 1 ? this.parallelism - 1 : 1;
    // A small prefetched context window may finish several clangd requests together. Let those
    // short oracle batches overlap, while reserving most compiler capacity for scoring.
    this.priorityCompilerLimit = Math.max(1, Math.min(4, this.parallelism - 1));
    this.cache = new Map();
    this.children = new Set();
    this.activeCompilers = 0;
    this.activeNormalCompilers = 0;
    this.activePriorityCompilers = 0;
    this.compilerWaiters = [];
    this.priorityCompilerWaiters = [];
  }

  async compileAll(sources, priority = false) {
    if (!Array.isArray(sources) || sources.length === 0 || sources.length > MAX_COMPILE_REQUEST) {
      throw new Error(`sources must contain 1..${MAX_COMPILE_REQUEST} translation units`);
    }
    const results = new Array(sources.length);
    const missesByDigest = new Map();
    for (let index = 0; index < sources.length; index++) {
      const source = sources[index];
      if (typeof source !== "string") throw new Error(`sources[${index}] is not a string`);
      const digest = sourceDigest(source);
      const cached = this.cache.get(digest);
      if (cached != null) {
        // Native Map iteration order is insertion order. Reinsert hits so eviction below is
        // access-ordered and commonly repeated completions survive a full exhaustive sweep.
        this.cache.delete(digest);
        this.cache.set(digest, cached);
        results[index] = cached;
        continue;
      }
      const existing = missesByDigest.get(digest);
      if (existing == null) {
        missesByDigest.set(digest, { digest, source, indexes: [index] });
      } else {
        existing.indexes.push(index);
      }
    }

    const misses = [...missesByDigest.values()];
    if (misses.length === 0) return results;
    // Size a wave to occupy every configured worker without exceeding the per-process ceiling.
    // With the usual 1,280-source request and 12 workers this yields twelve ~107-source clang
    // processes instead of eight 160-source processes and four idle workers.
    const batchSize = Math.min(
      MAX_COMPILE_BATCH,
      Math.max(1, Math.ceil(misses.length / this.parallelism))
    );
    const batches = [];
    for (let start = 0; start < misses.length; start += batchSize) {
      batches.push(misses.slice(start, start + batchSize));
    }
    const compiledBatches = new Array(batches.length);
    let nextBatch = 0;
    const workers = Array.from(
      { length: Math.min(this.parallelism, batches.length) },
      async () => {
        while (true) {
          const batchIndex = nextBatch++;
          if (batchIndex >= batches.length) return;
          compiledBatches[batchIndex] = await this.compileIsolated(
            batches[batchIndex].map(miss => miss.source),
            priority
          );
        }
      }
    );
    await Promise.all(workers);
    const compiled = compiledBatches.flat();
    for (let index = 0; index < misses.length; index++) {
      const outcome = compiled[index];
      if (outcome.cacheable) this.remember(misses[index].digest, outcome.result);
      for (const originalIndex of misses[index].indexes) results[originalIndex] = outcome.result;
    }
    return results;
  }

  /**
   * Per-source diagnostics are already attributed by the synthetic `#line` boundaries, and CFG
   * samples are lexically balanced, so a fully classified mixed-success batch is definitive. Only
   * global errors, timeouts, or otherwise uncacheable outcomes need isolation. `compileBatch`
   * releases its permit before recursion, so checking both halves concurrently still respects the
   * process-wide compiler semaphore while allowing otherwise-idle workers to help isolate them.
   */
  async compileIsolated(sources, priority = false) {
    const outcomes = await this.compileBatch(sources, priority);
    const entirelyClassified = outcomes.every(outcome => outcome.cacheable);
    if (sources.length === 1 || entirelyClassified) return outcomes;

    const middle = Math.ceil(sources.length / 2);
    const [left, right] = await Promise.all([
      this.compileIsolated(sources.slice(0, middle), priority),
      this.compileIsolated(sources.slice(middle), priority)
    ]);
    return left.concat(right);
  }

  compile(source) {
    return this.compileAll([source]).then(results => results[0]);
  }

  async compileBatch(sources, priority = false) {
    const preamble = await this.prepareExactPreamble(sources, priority);
    await this.acquireCompiler(priority);
    try {
      const outcomes = await this.compileBatchUnbounded(sources, preamble);
      if (preamble == null || outcomes.every(outcome => outcome.cacheable)) return outcomes;
      // A stale/incompatible PCH is a bridge optimization failure, never a source verdict. Disable
      // this exact sequence and retry the unchanged include-bearing TU under the same permit.
      this.pchCache.set(preamble.key, Promise.resolve(null));
      return await this.compileBatchUnbounded(sources, null);
    } finally {
      this.releaseCompiler(priority);
    }
  }

  async prepareExactPreamble(sources, priority = false) {
    if (this.pchDirectory == null) return null;
    const preamble = exactIncludePreamble(sources);
    if (preamble.directives.length === 0) return null;
    const cached = this.pchCache.get(preamble.key);
    if (cached != null) return await cached;
    const pending = (async () => {
      await this.acquireCompiler(priority);
      try {
        return await this.buildPch(preamble);
      } finally {
        this.releaseCompiler(priority);
      }
    })();
    this.pchCache.set(preamble.key, pending);
    return await pending;
  }

  buildPch(preamble) {
    fs.mkdirSync(this.pchDirectory, { recursive: true });
    const pchPath = path.join(this.pchDirectory, `${preamble.key}.pch`);
    return new Promise(resolve => {
      let compiler;
      try {
        compiler = childProcess.spawn(
          this.executable,
          [
            "-xc++-header", "-std=c++23", "-pedantic-errors", "-w",
            "-fno-color-diagnostics", "-fno-caret-diagnostics", "-fno-spell-checking",
            "-fno-show-column", "-fno-diagnostics-fixit-info", "-o", pchPath, "-"
          ],
          { stdio: ["pipe", "ignore", "ignore"] }
        );
      } catch (_) {
        resolve(null);
        return;
      }

      this.children.add(compiler);
      let settled = false;
      let timedOut = false;
      let forceKill = null;
      const finish = result => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(result);
      };
      const timer = setTimeout(() => {
        timedOut = true;
        try {
          compiler.kill("SIGTERM");
        } catch (_) {
        }
        forceKill = setTimeout(() => {
          if (compiler.exitCode == null) {
            try {
              compiler.kill("SIGKILL");
            } catch (_) {
            }
          }
        }, 250);
        forceKill.unref?.();
        finish(null);
      }, COMPILE_TIMEOUT_MS);
      compiler.stdin.on("error", () => finish(null));
      compiler.on("error", () => {
        this.children.delete(compiler);
        finish(null);
      });
      compiler.on("close", code => {
        this.children.delete(compiler);
        if (forceKill != null) clearTimeout(forceKill);
        if (timedOut || code !== 0 || !fs.existsSync(pchPath)) finish(null);
        else finish({ key: preamble.key, path: pchPath });
      });
      try {
        compiler.stdin.end(preamble.source);
      } catch (_) {
        try {
          compiler.kill("SIGKILL");
        } catch (_) {
        }
        finish(null);
      }
    });
  }

  /**
   * `compileAll` may be called concurrently by precision scoring and clangd's declaration-type
   * oracle. Keep `parallelism` as a process-wide limit for this NativeCompiler, rather than a
   * per-request limit which could silently multiply the number of clang++ children.
   */
  acquireCompiler(priority = false) {
    if (
      this.activeCompilers < this.parallelism &&
      (priority
        ? this.activePriorityCompilers < this.priorityCompilerLimit
        : this.activeNormalCompilers < this.normalCompilerLimit)
    ) {
      this.activeCompilers++;
      if (priority) this.activePriorityCompilers++;
      else this.activeNormalCompilers++;
      return Promise.resolve();
    }
    return new Promise(resolve => {
      (priority ? this.priorityCompilerWaiters : this.compilerWaiters).push(resolve);
    });
  }

  releaseCompiler(priority = false) {
    this.activeCompilers--;
    if (priority) this.activePriorityCompilers--;
    else this.activeNormalCompilers--;
    while (this.activeCompilers < this.parallelism) {
      const nextPriority = this.activePriorityCompilers < this.priorityCompilerLimit
        ? this.priorityCompilerWaiters.shift()
        : null;
      if (nextPriority != null) {
        this.activeCompilers++;
        this.activePriorityCompilers++;
        nextPriority();
        continue;
      }
      if (this.activeNormalCompilers >= this.normalCompilerLimit) return;
      const nextNormal = this.compilerWaiters.shift();
      if (nextNormal == null) return;
      this.activeCompilers++;
      this.activeNormalCompilers++;
      nextNormal();
    }
  }

  compileBatchUnbounded(sources, preamble = null) {
    const translationUnit = combinedTranslationUnit(sources, preamble != null);
    return new Promise(resolve => {
      let compiler;
      try {
        compiler = childProcess.spawn(
          this.executable,
          [
            "-xc++", "-std=c++23", "-pedantic-errors", "-w", "-fsyntax-only",
            "-ferror-limit=0", "-fno-color-diagnostics", "-fno-caret-diagnostics",
            "-fno-spell-checking", "-fno-show-column", "-fno-diagnostics-fixit-info",
            ...(preamble == null ? [] : ["-include-pch", preamble.path]),
            "-"
          ],
          { stdio: ["pipe", "ignore", "pipe"] }
        );
      } catch (error) {
        resolve(sources.map(() => ({
          result: { ok: false, timedOut: false, diagnostics: error.message },
          cacheable: false
        })));
        return;
      }

      this.children.add(compiler);
      const stderr = [];
      let diagnosticBytes = 0;
      let diagnosticsTruncated = false;
      let stdinError = null;
      let exitFallback = null;
      let forceKill = null;
      let settled = false;
      let timedOut = false;

      const finish = outcomes => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (exitFallback != null) clearTimeout(exitFallback);
        resolve(outcomes);
      };
      const failEverySource = (message, timeout) => sources.map(() => ({
        result: { ok: false, timedOut: timeout, diagnostics: message },
        cacheable: false
      }));
      const finishFromExit = (code, signal) => {
        if (settled) return;
        const diagnostics = Buffer.concat(stderr).toString("utf8");
        const extraError = diagnosticsTruncated
          ? `clang++ diagnostics exceeded ${MAX_DIAGNOSTIC_BYTES} bytes`
          : stdinError?.message;
        const classified = batchDiagnosticClassification(
          diagnostics, sources.length, code, signal, extraError
        );
        if (classified.globalErrors.length > 0) {
          const globalDiagnostics = [
            ...classified.globalErrors.filter(error => !diagnostics.includes(error)),
            diagnostics
          ].filter(text => text.length > 0).join("\n");
          finish(failEverySource(globalDiagnostics, false));
          return;
        }
        finish(sources.map((_, index) => ({
          result: {
            ok: !classified.failed.has(index),
            timedOut: false,
            diagnostics: classified.perSource[index].join("\n")
          },
          cacheable: true
        })));
      };

      const timer = setTimeout(() => {
        timedOut = true;
        try {
          compiler.kill("SIGTERM");
        } catch (_) {
        }
        forceKill = setTimeout(() => {
          if (compiler.exitCode == null) {
            try {
              compiler.kill("SIGKILL");
            } catch (_) {
            }
          }
        }, 250);
        forceKill.unref?.();
        finish(failEverySource(`clang++ batch timed out after ${COMPILE_TIMEOUT_MS}ms`, true));
      }, COMPILE_TIMEOUT_MS);

      compiler.stderr.on("data", chunk => {
        if (diagnosticBytes >= MAX_DIAGNOSTIC_BYTES) {
          diagnosticsTruncated = true;
          return;
        }
        const remaining = MAX_DIAGNOSTIC_BYTES - diagnosticBytes;
        if (chunk.length > remaining) diagnosticsTruncated = true;
        const retained = chunk.length > remaining ? chunk.subarray(0, remaining) : chunk;
        diagnosticBytes += retained.length;
        stderr.push(retained);
      });
      compiler.stdin.on("error", error => {
        stdinError = error;
      });
      compiler.on("error", error => {
        this.children.delete(compiler);
        finish(failEverySource(error.message, false));
      });
      compiler.on("exit", (code, signal) => {
        if (timedOut || settled) {
          this.children.delete(compiler);
          if (forceKill != null) clearTimeout(forceKill);
          return;
        }
        exitFallback = setTimeout(() => {
          this.children.delete(compiler);
          finishFromExit(code, signal);
        }, 100);
      });
      compiler.on("close", (code, signal) => {
        this.children.delete(compiler);
        if (forceKill != null) clearTimeout(forceKill);
        finishFromExit(code, signal);
      });
      try {
        compiler.stdin.end(translationUnit);
      } catch (error) {
        stdinError = error;
        try {
          compiler.kill("SIGKILL");
        } catch (_) {
        }
      }
    });
  }

  remember(digest, result) {
    // Keep the full benchmark's digest-only outcomes without periodic all-or-nothing cache
    // thrashing. Updating an existing key also refreshes its access order.
    this.cache.delete(digest);
    this.cache.set(digest, result);
    while (this.cache.size > MAX_COMPILER_CACHE_ENTRIES) {
      this.cache.delete(this.cache.keys().next().value);
    }
  }

  async close() {
    const running = [...this.children];
    if (running.length > 0) {
      const exited = running.map(compiler => compiler.exitCode == null
        ? new Promise(resolve => compiler.once("exit", resolve))
        : Promise.resolve());
      for (const compiler of running) {
        try {
          compiler.kill("SIGTERM");
        } catch (_) {
        }
      }
      await Promise.race([
        Promise.all(exited),
        new Promise(resolve => setTimeout(resolve, 250))
      ]);
      for (const compiler of this.children) {
        try {
          compiler.kill("SIGKILL");
        } catch (_) {
        }
      }
      await Promise.race([
        Promise.all(exited),
        new Promise(resolve => setTimeout(resolve, 500))
      ]);
    }
    if (this.pchDirectory != null) {
      try {
        fs.rmSync(this.pchDirectory, { recursive: true, force: true });
      } catch (_) {
      }
    }
  }
}

function loadFixtures(fixturesDirectory) {
  if (!fs.existsSync(fixturesDirectory)) return [];
  return fs.readdirSync(fixturesDirectory, { withFileTypes: true })
    .filter(entry => entry.isFile() && entry.name.endsWith(".cpp"))
    .map(entry => ({
      name: entry.name,
      source: fs.readFileSync(path.join(fixturesDirectory, entry.name), "utf8")
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function configureCppCompletionBenchmark(config) {
  function createMiddleware(logger, emitter) {
    const log = logger.create("cpp-completion-benchmark");
    const clangd = process.env.CLANGD || "clangd";
    const clangxx = process.env.CXX || "clang++";
    const clangdVersion = executableVersion(clangd);
    const compilerVersion = executableVersion(clangxx);
    const fixturesDirectory = path.resolve(__dirname, "../resources/cpp-completion");
    // Resources are immutable for a Karma run. Read them once instead of traversing and reading
    // the directory independently for the status and fixtures requests at browser startup.
    const fixtures = loadFixtures(fixturesDirectory);
    const maxInstances = process.env.CPP_COMPLETION_MAX_INSTANCES == null
      ? null
      : positiveIntegerEnvironment("CPP_COMPLETION_MAX_INSTANCES", 1);
    const startInstance = nonnegativeIntegerEnvironment("CPP_COMPLETION_START_INSTANCE", 0);
    const samplesPerInstance = positiveIntegerEnvironment(
      "CPP_COMPLETION_SAMPLES_PER_INSTANCE",
      DEFAULT_SAMPLES_PER_INSTANCE
    );
    const timeLimitMillis = positiveIntegerEnvironment("CPP_COMPLETION_TIME_LIMIT_MS", 60_000);
    const compilerJobs = positiveIntegerEnvironment(
      "CPP_COMPLETION_COMPILER_JOBS",
      Math.min(12, Math.max(1, os.availableParallelism() - 2))
    );
    const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "tidyparse-cpp-completion-"));
    const compiler = compilerVersion == null ? null : new NativeCompiler(
      clangxx,
      compilerJobs,
      {
        // `-include-pch` is a Clang interface. Other C++ drivers retain the original stdin path.
        pchDirectory: /\bclang version\b/i.test(compilerVersion)
          ? path.join(workspace, "compiler-pch")
          : null
      }
    );
    const lsp = clangdVersion == null ? null : new ClangdLspSession({
      clangd,
      clangxx,
      log,
      workspace
    });
    // The exhaustive run always begins in the first fixture. Start its preamble immediately so
    // clangd's cold SDK parse overlaps Karma/Chrome startup. Focused slices may begin in another
    // fixture, so do not make them parse an unrelated source first.
    if (process.env.CPP_COMPLETION_BENCHMARK === "1" && startInstance === 0 && fixtures.length > 0) {
      lsp?.prime(fixtures[0].source, fixtures[0].name).catch(error => {
        log.debug(`clangd preamble warmup failed: ${error.message}`);
      });
    }

    const cleanup = async () => {
      await Promise.allSettled([lsp?.close(), compiler?.close()]);
      try {
        fs.rmSync(workspace, { recursive: true, force: true });
      } catch (_) {
      }
    };
    // `exit` is emitted while Karma is already tearing its web server down. Registering another
    // exit listener here prevents Kotlin/JS' single-run launcher from settling on some Karma
    // versions. `run_complete` occurs after the browser has reported every result but before
    // shutdown, which is exactly when the test-only clangd process and workspace can be released.
    emitter.once("run_complete", () => {
      cleanup().catch(error => log.warn(`C++ benchmark cleanup failed: ${error.message}`));
    });

    return async function cppCompletionBenchmarkMiddleware(request, response, next) {
      const url = new URL(request.url, "http://karma.invalid");
      if (!url.pathname.startsWith(ROUTE_PREFIX)) {
        next();
        return;
      }
      try {
        if (request.method === "GET" && url.pathname === `${ROUTE_PREFIX}/status`) {
          jsonResponse(response, 200, {
            enabled: process.env.CPP_COMPLETION_BENCHMARK === "1",
            clangd: clangdVersion,
            compiler: compilerVersion,
            fixtures: fixtures.map(fixture => fixture.name),
            samplesPerInstance,
            startInstance,
            maxInstances,
            timeLimitMillis
          });
          return;
        }
        if (request.method === "GET" && url.pathname === `${ROUTE_PREFIX}/fixtures`) {
          jsonResponse(response, 200, { fixtures });
          return;
        }
        if (request.method === "POST" && url.pathname === `${ROUTE_PREFIX}/context`) {
          if (lsp == null) throw new Error("clangd is unavailable");
          const payload = await readJson(request);
          if (typeof payload.source !== "string") throw new Error("source must be a string");
          const includeOracle = payload.mode === "oracle";
          const result = await lsp.context(
            payload.source,
            payload.line,
            payload.character,
            typeof payload.fixture === "string" ? payload.fixture : "main.cpp",
            includeOracle
          );
          if (includeOracle) {
            const declarationFacts = await requiredDeclarationFacts(
              payload.source, payload.line, payload.character, result, compiler
            );
            result.requiredTypes = declarationFacts.requiredTypes;
            result.probedRequiredTypes = declarationFacts.probedRequiredTypes;
            result.defaultConstructibleTypes = declarationFacts.defaultConstructibleTypes;
          }
          // The default/scored route is the same browser-available LSP fact contract consumed by
          // production. Oracle mode is reserved for focused native bridge diagnostics.
          jsonResponse(response, 200, result);
          return;
        }
        if (request.method === "POST" && url.pathname === `${ROUTE_PREFIX}/compile`) {
          if (compiler == null) throw new Error("clang++ is unavailable");
          const payload = await readJson(request);
          const results = await compiler.compileAll(payload.sources);
          jsonResponse(response, 200, { results });
          return;
        }
        jsonResponse(response, 404, { error: "Unknown C++ completion benchmark route" });
      } catch (error) {
        log.error(error.stack || error.message || String(error));
        jsonResponse(response, 500, { error: error.message || String(error) });
      }
    };
  }
  createMiddleware.$inject = ["logger", "emitter"];

  const plugin = {
    "middleware:cppCompletionBenchmark": ["factory", createMiddleware]
  };
  config.plugins = [...(config.plugins || []), plugin];
  config.middleware = [...(config.middleware || []), "cppCompletionBenchmark"];
}

module.exports = configureCppCompletionBenchmark;
module.exports.ClangdLspSession = ClangdLspSession;
module.exports.NativeCompiler = NativeCompiler;
module.exports.requiredDeclarationTypes = requiredDeclarationTypes;
module.exports.requiredTypeCandidates = requiredTypeCandidates;
module.exports.canonicalRequiredType = canonicalRequiredType;
module.exports.exactIncludePreamble = exactIncludePreamble;
module.exports.combinedTranslationUnit = combinedTranslationUnit;
module.exports.batchDiagnosticClassification = batchDiagnosticClassification;
module.exports.hasActiveExpressionPrefix = hasActiveExpressionPrefix;
