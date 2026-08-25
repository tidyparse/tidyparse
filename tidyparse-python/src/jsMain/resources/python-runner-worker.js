"use strict";

const PYODIDE_VERSION = "0.27.5";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let runtimePromise = null;

const RUNNER_BOOTSTRAP = String.raw`
import contextlib
import io
import json
import sys
import traceback

def __tidyparse_execute(source, stdin_text):
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    previous_stdin = sys.stdin
    namespace = {
        "__name__": "__main__",
        "__file__": "main.py",
        "__package__": None,
    }

    try:
        sys.stdin = io.StringIO(stdin_text)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = compile(source, "main.py", "exec")
                exec(code, namespace, namespace)
            except SystemExit as system_exit:
                value = system_exit.code
                if value is None:
                    exit_code = 0
                elif isinstance(value, int):
                    exit_code = int(value)
                else:
                    print(value, file=sys.stderr)
                    exit_code = 1
            except BaseException:
                traceback.print_exc(file=sys.stderr)
                exit_code = 1
    finally:
        sys.stdin = previous_stdin

    return json.dumps({
        "exitCode": exit_code,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "timedOut": False,
    }, ensure_ascii=False)
`;

function errorText(error) {
  if (error && typeof error.stack === "string") return error.stack;
  if (error && typeof error.message === "string") return error.message;
  return String(error);
}

function loadRuntime() {
  if (runtimePromise !== null) return runtimePromise;

  runtimePromise = (async () => {
    importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);
    if (typeof loadPyodide !== "function") {
      throw new Error("Pyodide loaded without exposing loadPyodide");
    }

    const runtime = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    runtime.runPython(RUNNER_BOOTSTRAP);
    return runtime;
  })().catch((error) => {
    runtimePromise = null;
    throw error;
  });

  return runtimePromise;
}

async function runPython(source, stdin, onRuntimeReady) {
  const runtime = await loadRuntime();
  onRuntimeReady();
  runtime.globals.set("__tidyparse_source", source);
  runtime.globals.set("__tidyparse_stdin", stdin);

  try {
    const encoded = runtime.runPython(
      "__tidyparse_execute(__tidyparse_source, __tidyparse_stdin)"
    );
    return JSON.parse(encoded);
  } finally {
    runtime.globals.delete("__tidyparse_source");
    runtime.globals.delete("__tidyparse_stdin");
  }
}

self.onmessage = async (event) => {
  const message = event.data || {};
  const id = message.id;

  if (message.type !== "run") {
    self.postMessage({
      type: "error",
      id,
      error: `Unsupported Python runner message type: ${String(message.type)}`,
    });
    return;
  }

  try {
    const source = typeof message.source === "string" ? message.source : "";
    const stdin = typeof message.stdin === "string" ? message.stdin : "";
    self.postMessage({ type: "status", id, state: "loading" });
    const result = await runPython(source, stdin, () => {
      self.postMessage({ type: "status", id, state: "running" });
    });
    self.postMessage({ type: "result", id, result });
  } catch (error) {
    self.postMessage({ type: "error", id, error: errorText(error) });
  }
};
