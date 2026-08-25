# tidyparse-python

This module builds the standalone Python 3 playground served by `python3.html`. It does not
replace or modify the existing web demo.

The editor uses vanilla Monaco 0.55.1 and official `ty_wasm` for in-browser Python analysis.
Python execution runs separately in `python-runner-worker.js`. Webpack emits one worker-compatible
application bundle, `tidyparse-python.js`, which is loaded by both the page and Monaco's editor
worker. The runner allows up to 90 seconds for the first lazy Pyodide boot, then enforces a separate
15-second limit on user code.

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
`crates/ty_wasm`, so it is substantially slower than later incremental builds. The build checks
the exact Ruff revision, source toolchain pin, generated JavaScript/Wasm pairing, and WebAssembly
magic bytes.

## Deployment

Stage the deployable files with:

```shell
./gradlew :tidyparse-python:preparePythonDeploy
```

The stage contains `python3.html`, `python3.css`, `python-runner-worker.js`,
`ty-wasm-loader.js`, the application JavaScript and source map, and the generated `ty_wasm`
JavaScript/Wasm pair. It also includes Ruff's root license as `ty-LICENSE` and the vendored
typeshed license as `typeshed-LICENSE`.

Deploy the independently managed `python3` site slice with:

```shell
./gradlew :tidyparse-python:deployPython --msg "update Python 3 playground"
```
