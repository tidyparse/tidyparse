internal const val CPP_MONACO_EDITOR_WORKER_NAME = "tidyparse-monaco-editor"
internal const val CPP_TEXTMATE_WORKER_NAME = "tidyparse-textmate"

fun isCppMonacoEditorWorkerRuntime(): Boolean =
  isNamedCppWorkerRuntime(CPP_MONACO_EDITOR_WORKER_NAME)

fun isCppTextMateWorkerRuntime(): Boolean =
  isNamedCppWorkerRuntime(CPP_TEXTMATE_WORKER_NAME)

/**
 * Start the upstream worker entry points inside tidyparse-web.js itself.
 *
 * Keeping these static requires in the main webpack graph avoids loader-
 * generated CommonJS shims and lets GitHub Pages serve one JavaScript bundle.
 * They are evaluated only in their matching DedicatedWorkerGlobalScope, so
 * neither worker entry point touches the browser window.
 */
fun setupCppMonacoEditorWorker() {
  js("require('@codingame/monaco-vscode-editor-api/esm/vs/editor/editor.worker.js')")
}

fun setupCppTextMateWorker() {
  js("require('@codingame/monaco-vscode-textmate-service-override/worker')")
}

private fun isNamedCppWorkerRuntime(name: String): Boolean =
  js(
    """(expectedName) => typeof document === "undefined" &&
       typeof globalThis.postMessage === "function" &&
       globalThis.name === expectedName"""
  )(name) as Boolean
