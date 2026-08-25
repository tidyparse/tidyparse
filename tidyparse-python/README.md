# tidyparse-python

This module builds the standalone Python 3 playground served by `python3.html`. It does not
replace or modify the existing web demo.

The editor uses vanilla Monaco 0.55.1 and official `ty_wasm` for in-browser Python analysis.
Python execution runs separately in `python-runner-worker.js`. Webpack emits the worker-compatible
application bundle `tidyparse-python.js`, which is loaded by both the page and Monaco's editor
worker, plus the isolated `tidyparse-python-repair.js` syntax-repair worker from a dedicated
nested `:tidyparse-python:repair-worker` project. The runner allows up to 90 seconds for the first
lazy Pyodide boot, then enforces a separate 15-second limit on user code.

## Syntax repair completions

Ctrl+Space asks the isolated `tidyparse-python-repair.js` worker to classify the active physical
line with the shared Python statement grammar. Both Monaco modifier encodings are registered so
Control works on every platform (and Command remains accepted when the browser receives it). The
same command is available from the editor's context menu and F1 palette as
**Trigger TidyParse Syntax Repair**. Ordinary Monaco and ty completion suggestions are disabled;
valid lines return no suggestions, while malformed lines receive ranked whole-line repairs. The
worker links the shared grammar-intersection kernels and neural reranker from `tidyparse-wgpu`. It
owns the Python four-gram/WDFA path, worker protocol, asset loading, and version-locked assets under
`src/jsMain/resources/python3-repair/`. It keeps the GPU device and model warm and falls back to CPU
grammar sampling when WebGPU is unavailable.

The main thread substitutes every distinct ranked candidate into a fresh, isolated ty scratch file and
retains it only when the complete repaired document has zero ty diagnostics. All raw candidates are
classified before formatting begins. Every semantic survivor is then formatted with the Ruff
formatter already embedded in `ty_wasm`; the original line's indentation is restored and the exact
formatted insertion is checked again before Monaco displays it. Ruff output is used only when it
can be projected back to one physical line without changing semantic admissibility. Delimiter-based
multiline layouts are compacted before that check; context-dependent repairs, formatter failures,
unsafe statement-level flattening, and formatted insertions rejected by ty safely fall back to the
already-admissible original spelling. The scan and formatting pass do not stop at Monaco's display
limit. Monaco discards all partial results when the document changes while either pass is running.

The IDE asks the worker for its advertised bounded batch of highest-ranked candidates. Ty classifies
every distinct candidate in that batch; the worker-advertised
display limit is applied only after semantic filtering, formatting, revalidation, and stable
deduplication. The worker applies that same bound when a protocol caller omits `maxResults`, so a
malformed request cannot accidentally trigger an unbounded main-thread sweep.

The JSON-compatible worker protocol remains local to this project. Shared WebGPU implementation,
including the single reranker implementation, lives in `tidyparse-wgpu`; the legacy `python.html`
sources are otherwise unchanged.

## Pinned `ty_wasm` build

`ty_wasm` is built from the official
[`astral-sh/ruff`](https://github.com/astral-sh/ruff) source at commit
`423b9fbf1923b00e66f25f059b1e91dd79aacd03`, using Rust 1.98.0 and wasm-pack 0.13.1. The build
does not install the unrelated, unscoped `ty_wasm` npm package. Gradle fetches the exact Git
revision, installs/verifies Rust through rustup, and installs wasm-pack below this module's `build/`
directory. It then stages only `ty_wasm.js` and `ty_wasm_bg.wasm` as generated browser resources.

The classic external `ty-wasm-loader.js` performs the native ES import outside webpack,
initializes the generated module exactly once, and exposes its Promise as
`globalThis.tidyparseTyWasmReady` before the application bundle loads.

Run the development server with:

```shell
./gradlew :tidyparse-python:jsBrowserDevelopmentRun --continuous
```

The first build installs the pinned Rust toolchain and wasm-pack, fetches Ruff, and compiles
`crates/ty_wasm`, so it is substantially slower than later incremental builds. The Ruff revision,
Rust toolchain, and wasm-pack version are pinned in the build script. The standalone repair worker
can also be built directly with:

```shell
./gradlew :tidyparse-python:repair-worker:jsBrowserProductionWebpack
```

## Deployment

Stage the deployable files with:

```shell
./gradlew :tidyparse-python:preparePythonDeploy
```

The stage contains `python3.html`, `python3.css`, `python-runner-worker.js`,
`tidyparse-python-repair.js`, the namespaced repair assets, `ty-wasm-loader.js`, the application
JavaScript and source maps, and the generated `ty_wasm` JavaScript/Wasm pair. The Ruff and typeshed
license notices retain their original names under `licenses/ruff/` and `licenses/typeshed/`.

Deploy the independently managed `python3` site slice with:

```shell
./gradlew :tidyparse-python:deployPython --msg "update Python 3 playground"
```
