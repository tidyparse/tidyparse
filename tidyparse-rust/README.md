# Tidyparse Rust browser demo

This subproject runs a small [Rust Glancer](https://github.com/rust-glancer/rust-glancer) analysis core inside a Web Worker and connects it directly to a vanilla Monaco editor.

## Run it

From the repository root:

```bash
./gradlew :tidyparse-rust:jsBrowserDevelopmentRun --continuous
```

The development server opens `rust.html`. The first build requires `rustup` and network access; Gradle installs the pinned Rust 1.91.0 toolchain with `wasm32-unknown-unknown`, fetches the locked Cargo and npm dependencies, and then caches the result. Kotlin, HTML, and CSS edits do not rebuild the Rust crate. Changes under `wasm/` do.

## What is in the prototype

- Glancer's Rust parser and `Analysis::document_symbols_from_syntax`, compiled from revision `7c68f1af` to a roughly 290 KB Wasm module.
- Syntax diagnostics and document symbols from Glancer.
- Lightweight same-file definitions, hovers, and lexical completions built from Glancer's syntax tree.
- A dedicated analysis worker, plus Monaco's editor worker, so parsing and editor services stay off the UI thread.
- A direct Monaco provider adapter; there is no LSP process, VS Code compatibility layer, or `monaco-languageclient` stack.
- A Run button and <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Enter</kbd> shortcut that compile and execute the current single-file program with Rust 1.91 on [Compiler Explorer](https://godbolt.org/), including standard input, output, and rustc diagnostics.

The Wasm wrapper uses a tiny JSON-over-memory ABI rather than `wasm-bindgen`. Gradle stages the compiled module in generated resources, and `--continuous` only rebuilds it when a Cargo input changes.

## Current boundary

This is deliberately a syntax-aware, single-file browser host. Its live editor analysis does not claim Cargo dependency resolution, macro expansion, rustc diagnostics, or type-aware completion. Glancer's full project analysis currently assumes a native filesystem and native project execution paths; bringing that layer to the browser needs an in-memory project host in Glancer rather than more Monaco or Gradle plumbing.

Running sends the current source and optional standard input to Compiler Explorer. The resulting rustc diagnostics are shown on demand in the Execution card; they are separate from Glancer's live syntax diagnostics. Arbitrary Cargo dependencies are not available in this single-file execution mode.
