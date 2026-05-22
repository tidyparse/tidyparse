let pyodide = null;
let ready = null;
let blackReady = null;

const PY_COMPILE_BOOTSTRAP = [
    "import sys, traceback, io, base64, textwrap, warnings",
    "",
    "def _compile_output_b64(encoded):",
    "    _out = io.StringIO()",
    "",
    "    old_stdout = sys.stdout",
    "    old_stderr = sys.stderr",
    "    old_showwarning = warnings.showwarning",
    "",
    "    def _captured_showwarning(message, category, filename, lineno, file=None, line=None):",
    "        _out.write(warnings.formatwarning(message, category, filename, lineno, line))",
    "",
    "    try:",
    "        sys.stdout = _out",
    "        sys.stderr = _out",
    "        warnings.showwarning = _captured_showwarning",
    "",
    "        try:",
    "            _src = base64.b64decode(encoded).decode('utf-8')",
    "            _src = _src.replace('NUMBER', '1').replace('STRING', '\"\"')",
    "            _src = textwrap.dedent(_src)",
    "            compile(_src, 'test_compile.py', 'exec')",
    "        except Exception:",
    "            traceback.print_exc()",
    "",
    "        return _out.getvalue()",
    "    finally:",
    "        warnings.showwarning = old_showwarning",
    "        sys.stdout = old_stdout",
    "        sys.stderr = old_stderr"
].join("\n");

const PY_FORMAT_BOOTSTRAP = [
    "import sys, io, base64, contextlib, traceback",
    "from black import format_str, FileMode",
    "",
    "def _format_output_b64(encoded):",
    "    src = base64.b64decode(encoded).decode('utf-8')",
    "    sink = io.StringIO()",
    "    try:",
    "        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):",
    "            pretty_code = format_str(src, mode=FileMode(string_normalization=False))",
    "        return pretty_code.strip().replace('\\n', ' ')",
    "    except Exception:",
    "        return '__BLACK_ERROR__ ' + traceback.format_exc().replace('\\n', ' ') + ' __SRC__ ' + src"
].join("\n");

async function ensureReady(indexURL) {
    if (ready) return ready;

    ready = (async () => {
        console.log("[py-worker] ensureReady: begin");

        importScripts(indexURL + "pyodide.js");

        pyodide = await loadPyodide({indexURL, stdout: () => {}, stderr: () => {}});

        pyodide.runPython(PY_COMPILE_BOOTSTRAP);

        console.log("[py-worker] ensureReady: done");
    })();

    return ready;
}

async function ensureBlackReady() {
    if (blackReady) return blackReady;

    blackReady = (async () => {
        console.log("[py-worker] ensureBlackReady: begin");

        await pyodide.loadPackage("micropip");

        const micropip = pyodide.pyimport("micropip");
        try {
            await micropip.install("black");
        } finally {
            try { micropip.destroy(); } catch (_) {}
        }

        pyodide.runPython(PY_FORMAT_BOOTSTRAP);

        // Smoke-test the actual formatter path.
        console.log("[py-worker] black ready:", runFormatNoEnsure("x=1"));
        console.log("[py-worker] ensureBlackReady: done");
    })();

    return blackReady;
}

function runFormatNoEnsure(src) {
    pyodide.globals.set("_encoded", b64(src));
    try { return pyodide.runPython("_format_output_b64(_encoded)") || (src || ""); }
    finally { try { pyodide.globals.delete("_encoded"); } catch (_) {} }
}


// Matches the old Kotlin js("btoa")(src) path for ASCII token strings.
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
    const { id, op, indexURL, src } = data;

    try {
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

        throw new Error("Unknown pyodide_webworkers op: " + op);
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