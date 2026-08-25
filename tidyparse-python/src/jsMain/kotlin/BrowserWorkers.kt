internal const val PYTHON_MONACO_WORKER_NAME = "tidyparse-python-monaco-editor"

internal fun isPythonMonacoWorkerRuntime(): Boolean =
  isNamedPythonWorkerRuntime(PYTHON_MONACO_WORKER_NAME)

internal fun setupPythonMonacoWorker() {
  js("require('vanilla-monaco-editor/esm/vs/editor/editor.worker.js')")
}

internal fun configurePythonMonacoWorkers() {
  val environment = js("({})")
  environment.getWorker = { _: dynamic, _: dynamic ->
    createPythonNamedWorker(PYTHON_MONACO_WORKER_NAME)
  }
  js("(environment) => { globalThis.MonacoEnvironment = environment; }")(environment)
}

internal fun createPythonNamedWorker(name: String): dynamic =
  js("(url, workerName) => new Worker(url, { name: workerName })")(
    pythonBrowserResourceUrl("tidyparse-python.js"),
    name
  )

internal fun pythonBrowserResourceUrl(fileName: String): String =
  js("(name) => new URL(name, document.baseURI).href")(fileName) as String

private fun isNamedPythonWorkerRuntime(name: String): Boolean =
  js(
    """(expectedName) => typeof document === "undefined" &&
       typeof globalThis.postMessage === "function" &&
       globalThis.name === expectedName"""
  )(name) as Boolean
