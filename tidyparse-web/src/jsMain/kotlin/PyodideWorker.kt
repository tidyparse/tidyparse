const val PYODIDE_BLACK_VENDOR_ARCHIVE = "/pyodide-black-25.1.0-site.zip"

private val pyCompileBootstrap = """
import sys, traceback, io, base64, textwrap, warnings

def _compile_output_b64(encoded):
    _out = io.StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_showwarning = warnings.showwarning

    def _captured_showwarning(message, category, filename, lineno, file=None, line=None):
        _out.write(warnings.formatwarning(message, category, filename, lineno, line))

    try:
        sys.stdout = _out
        sys.stderr = _out
        warnings.showwarning = _captured_showwarning

        try:
            _src = base64.b64decode(encoded).decode('utf-8')
            _src = _src.replace('NUMBER', '1').replace('STRING', '""')
            _src = textwrap.dedent(_src)
            compile(_src, 'test_compile.py', 'exec')
        except Exception:
            traceback.print_exc()

        return _out.getvalue()
    finally:
        warnings.showwarning = old_showwarning
        sys.stdout = old_stdout
        sys.stderr = old_stderr
""".trimIndent()


private val pyFormatBootstrap = """
import sys, io, base64, contextlib, traceback
from black import format_str, FileMode

def _format_output_b64(encoded):
    src = base64.b64decode(encoded).decode('utf-8')
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            pretty_code = format_str(src, mode=FileMode(string_normalization=False))
        return pretty_code.strip().replace('\n', ' ')
    except Exception:
        return '__BLACK_ERROR__ ' + traceback.format_exc().replace('\n', ' ') + ' __SRC__ ' + src
""".trimIndent()

private fun jsStringLiteral(s: String): String =
  js("(s) => JSON.stringify(s)")(s) as String

private fun pyodideWorkerSource(): String = """
let pyodide = null;
let ready = null;
let blackReady = null;
let loadedPyodideScript = false;

const PY_COMPILE_BOOTSTRAP = ${jsStringLiteral(pyCompileBootstrap)};
const PY_FORMAT_BOOTSTRAP = ${jsStringLiteral(pyFormatBootstrap)};
const BLACK_VENDOR_ARCHIVE_URL = ${jsStringLiteral(PYODIDE_BLACK_VENDOR_ARCHIVE)};

function hasPythonGlobal(name) {
    try {
        return pyodide.runPython(JSON.stringify(name) + " in globals()") === true;
    } catch (_) {
        return false;
    }
}

function markBlackReadyFromSnapshot() {
    if (hasPythonGlobal("_format_output_b64")) {
        blackReady = Promise.resolve();
        return true;
    }

    return false;
}

function hasPythonModule(name) {
    try {
        return pyodide.runPython("import importlib.util; importlib.util.find_spec(" + JSON.stringify(name) + ") is not None") === true;
    } catch (_) {
        return false;
    }
}

function appResourceURL(path) {
    if (/^https?:\/\//.test(path)) return path;

    const base = self.location.href.startsWith("blob:")
        ? self.location.href.substring("blob:".length)
        : self.location.href;

    return new URL(path, base).href;
}

async function unpackVendoredBlack() {
    if (hasPythonModule("black")) return;

    const archiveURL = appResourceURL(BLACK_VENDOR_ARCHIVE_URL);
    console.log("[py-worker] loading vendored Black:", archiveURL);

    const response = await fetch(archiveURL, {cache: "force-cache"});
    if (!response.ok) {
        throw new Error("Failed to load vendored Black archive: " + response.status + " " + response.statusText);
    }

    const buffer = await response.arrayBuffer();
    const sitePackages = pyodide.runPython("import site; site.getsitepackages()[0]");
    pyodide.unpackArchive(buffer, "zip", {extractDir: sitePackages});
}

async function ensureReady(indexURL, options = {}) {
    if (ready) return ready;

    ready = (async () => {
        const mode = options.snapshot ? "snapshot" : (options.makeSnapshot ? "snapshot-builder" : "cold");
        console.log("[py-worker] ensureReady: begin (" + mode + ")");

        if (!loadedPyodideScript) {
            importScripts(indexURL + "pyodide.js");
            loadedPyodideScript = true;
        }

        const config = {indexURL, stdout: () => {}, stderr: () => {}};
        if (options.makeSnapshot) config._makeSnapshot = true;
        if (options.snapshot) config._loadSnapshot = options.snapshot;

        pyodide = await loadPyodide(config);

        if (!hasPythonGlobal("_compile_output_b64")) {
            pyodide.runPython(PY_COMPILE_BOOTSTRAP);
        }

        markBlackReadyFromSnapshot();

        console.log("[py-worker] ensureReady: done (" + mode + ")");
    })();

    return ready;
}

async function ensureBlackReady() {
    if (blackReady) return blackReady;

    blackReady = (async () => {
        console.log("[py-worker] ensureBlackReady: begin");

        await unpackVendoredBlack();

        pyodide.runPython(PY_FORMAT_BOOTSTRAP);

        console.log("[py-worker] black ready:", runFormatNoEnsure("x=1"));
        console.log("[py-worker] ensureBlackReady: done");
    })();

    return blackReady;
}

async function makeWarmSnapshot(indexURL) {
    await ensureReady(indexURL, {makeSnapshot: true});
    await ensureBlackReady();

    if (typeof pyodide.makeMemorySnapshot !== "function") {
        throw new Error("pyodide.makeMemorySnapshot is unavailable");
    }

    return pyodide.makeMemorySnapshot();
}

async function makeBaseSnapshot(indexURL) {
    await ensureReady(indexURL, {makeSnapshot: true});

    if (typeof pyodide.makeMemorySnapshot !== "function") {
        throw new Error("pyodide.makeMemorySnapshot is unavailable");
    }

    return pyodide.makeMemorySnapshot();
}

function runFormatNoEnsure(src) {
    pyodide.globals.set("_encoded", b64(src));
    try { return pyodide.runPython("_format_output_b64(_encoded)") || (src || ""); }
    finally { try { pyodide.globals.delete("_encoded"); } catch (_) {} }
}

function b64(s) { return btoa(s || ""); }

async function runCompile(src) {
    pyodide.globals.set("_encoded", b64(src));
    try { return pyodide.runPython("_compile_output_b64(_encoded)") || ""; }
    finally { try { pyodide.globals.delete("_encoded"); } catch (_) {} }
}

async function runFormat(src) {
    await ensureBlackReady();
    return runFormatNoEnsure(src);
}

self.onmessage = async (ev) => {
    const data = ev.data || {};
    const { id, op, indexURL, src, snapshot } = data;

    try {
        if (op === "snapshot") {
            const snapshot = await makeWarmSnapshot(indexURL);

            self.postMessage({
                id,
                ok: true,
                snapshot,
                snapshotKind: "warm",
                snapshotBytes: snapshot.byteLength
            }, [snapshot.buffer]);
            return;
        }

        if (op === "snapshotBase") {
            const snapshot = await makeBaseSnapshot(indexURL);

            self.postMessage({
                id,
                ok: true,
                snapshot,
                snapshotKind: "base",
                snapshotBytes: snapshot.byteLength
            }, [snapshot.buffer]);
            return;
        }

        if (op === "initSnapshot") {
            if (!snapshot) throw new Error("Missing Pyodide memory snapshot");

            await ensureReady(indexURL, {snapshot});
            if (!markBlackReadyFromSnapshot()) await ensureBlackReady();

            self.postMessage({
                id,
                ok: true,
                output: "",
                formatted: ""
            });
            return;
        }

        await ensureReady(indexURL);

        if (op === "init") {
            await ensureBlackReady();

            self.postMessage({
                id,
                ok: true,
                output: "",
                formatted: ""
            });
            return;
        }

        if (op === "compile") {
            const output = await runCompile(src || "");
            self.postMessage({ id, ok: true, output });
            return;
        }

        if (op === "format") {
            const formatted = await runFormat(src || "");
            self.postMessage({ id, ok: true, formatted });
            return;
        }

        throw new Error("Unknown pyodide worker op: " + op);
    } catch (e) {
        self.postMessage({
            id,
            ok: true,
            output: "",
            formatted: src || "",
            infraError: String((e && e.stack) || e)
        });
    }
};
""".trimIndent()

fun createPyodideWorkerURL(): String =
  js("(source) => URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))")(pyodideWorkerSource()) as String

fun revokePyodideWorkerURL(url: String) {
  js("(url) => URL.revokeObjectURL(url)")(url)
}
