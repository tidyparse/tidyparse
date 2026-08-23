internal const val RUST_GLANCER_WORKER_NAME = "tidyparse-rust-glancer"
internal const val RUST_MONACO_WORKER_NAME = "tidyparse-rust-monaco-editor"

internal fun isRustGlancerWorkerRuntime(): Boolean =
  isNamedRustWorkerRuntime(RUST_GLANCER_WORKER_NAME)

internal fun isRustMonacoWorkerRuntime(): Boolean =
  isNamedRustWorkerRuntime(RUST_MONACO_WORKER_NAME)

internal fun setupRustMonacoWorker() {
  js("require('vanilla-monaco-editor/esm/vs/editor/editor.worker.js')")
}

internal fun configureRustMonacoWorkers() {
  val environment = js("({})")
  environment.getWorker = { _: dynamic, _: dynamic ->
    createRustNamedWorker(RUST_MONACO_WORKER_NAME)
  }
  js("(environment) => { globalThis.MonacoEnvironment = environment; }")(environment)
}

internal fun createRustNamedWorker(name: String): dynamic {
  val bundleUrl = js(
    """(fileName) => new URL(fileName, document.baseURI).href"""
  )("tidyparse-rust.js")
  return js("(url, workerName) => new Worker(url, { name: workerName })")(bundleUrl, name)
}

private fun isNamedRustWorkerRuntime(name: String): Boolean =
  js(
    """(expectedName) => typeof document === "undefined" &&
       typeof globalThis.postMessage === "function" &&
       globalThis.name === expectedName"""
  )(name) as Boolean
