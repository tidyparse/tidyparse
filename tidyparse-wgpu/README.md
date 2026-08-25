# tidyparse-wgpu

Shared Kotlin/JS WebGPU runtime for TidyParse's browser applications.

This project is a library, not a browser executable. Its grammar-intersection kernels and result
types are linked into both `tidyparse-web` and the dedicated `tidyparse-python:repair-worker`
browser worker. It depends directly on Galoisenne for grammar and automata types; it does not depend
on `tidyparse-core`.

- `WGPUv1.kt`
- `WGPUv2.kt`
- `WGPUtils.kt`
- `WGPURuntime.kt`
- `WGPUSuffixTypes.kt`
- `IntersectionResults.kt`
- `Reranker.kt`

Consumer-specific browser UI, worker protocols, Python grammar repair, and model assets deliberately
live outside this project. The shared reranker owns its WebGPU model execution, uses the common WGPU
runtime logger, and accepts an injected weight loader from each consumer. `tidyparse-python` owns
`tidyparse-python-repair.js` and everything under its `python3-repair/` resource directory.

Compile and test the shared library with:

```shell
./gradlew :tidyparse-wgpu:jsBrowserTest
```
