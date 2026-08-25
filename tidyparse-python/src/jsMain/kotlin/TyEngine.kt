import kotlinx.coroutines.await
import kotlin.js.Promise

private const val TY_WASM_EXPECTED_VERSION = "423b9fbf1"
private const val TY_WORKSPACE_ROOT = "/workspace"

/**
 * Direct, single-file browser host for ty's wasm-bindgen API.
 *
 * Every value returned to the editor is copied out of its wasm-bindgen wrapper. This keeps the UI
 * independent of ty's object lifetimes and lets callers safely retain results across later updates.
 */
class TyEngine(
  private val onStatus: (state: String, message: String) -> Unit
) {
  val sourcePath: String = "$TY_WORKSPACE_ROOT/main.py"
  val version: String get() = runtimeVersion

  private var runtimeVersion = ""
  private var ty: dynamic = null
  private var workspace: dynamic = null
  private var fileHandle: dynamic = null
  private var lastSource: String? = null

  suspend fun initialize(source: String): dynamic {
    if (tyDefined(workspace) && tyDefined(fileHandle)) {
      syncSource(source)
      return snapshot(mapDiagnostics())
    }

    onStatus("loading", "Loading ty…")
    try {
      val ready = tyWasmReadyPromise()
      check(tyDefined(ready)) {
        "The ty Wasm loader did not publish globalThis.tidyparseTyWasmReady"
      }
      ty = ready.unsafeCast<Promise<dynamic>>().await()
      runtimeVersion = ty.version() as String
      check(runtimeVersion == TY_WASM_EXPECTED_VERSION) {
        "Unsupported ty Wasm version '$runtimeVersion' (expected $TY_WASM_EXPECTED_VERSION)"
      }

      workspace = newTyWorkspace(
        ty.Workspace,
        TY_WORKSPACE_ROOT,
        ty.PositionEncoding.Utf16,
        js("({})")
      )
      fileHandle = workspace.openFile(sourcePath, source)
      lastSource = source

      val result = snapshot(mapDiagnostics())
      result.capabilities = arrayOf(
        "diagnostics",
        "completion",
        "hover",
        "definition",
        "inlay hints",
        "formatting",
        "signature help",
        "semantic tokens",
        "rename"
      )
      onStatus("ready", "ty $runtimeVersion is ready")
      return result
    } catch (failure: Throwable) {
      val detail = failure.message ?: "Unable to initialize ty"
      onStatus("error", detail)
      disposeRuntime()
      throw failure
    }
  }

  /** Updates the in-memory file and returns a fully detached diagnostics snapshot. */
  fun update(source: String): dynamic {
    requireInitialized()
    return try {
      syncSource(source)
      snapshot(mapDiagnostics())
    } catch (failure: Throwable) {
      onStatus("error", failure.message ?: "ty analysis failed")
      throw failure
    }
  }

  fun completions(line: Int, column: Int): Array<dynamic> {
    requireInitialized()
    val values = workspace.completions(fileHandle, newTyPosition(line, column))
      .unsafeCast<Array<dynamic>>()
    return mapTyArray(values, ::plainCompletion)
  }

  fun hover(line: Int, column: Int): dynamic {
    requireInitialized()
    val value = workspace.hover(fileHandle, newTyPosition(line, column))
    if (!tyDefined(value)) return null

    return try {
      val result = tyObject()
      result.markdown = value.markdown
      result.range = plainRange(value.range)
      result
    } finally {
      freeTyWrapper(value)
    }
  }

  /** Returns only locations which Monaco can open in this single-file prototype. */
  fun definitions(line: Int, column: Int): Array<dynamic> {
    requireInitialized()
    val values = workspace.gotoDefinition(fileHandle, newTyPosition(line, column))
      .unsafeCast<Array<dynamic>>()
    val results: dynamic = js("[]")
    values.forEach { link ->
      val mapped = plainLocationLink(link, sameFileOnly = true)
      if (mapped != null) results.push(mapped)
    }
    return results.unsafeCast<Array<dynamic>>()
  }

  fun inlayHints(
    startLine: Int,
    startColumn: Int,
    endLine: Int,
    endColumn: Int
  ): Array<dynamic> {
    requireInitialized()
    val range = newTyRange(startLine, startColumn, endLine, endColumn)
    val values = workspace.inlayHints(fileHandle, range).unsafeCast<Array<dynamic>>()
    return mapTyArray(values, ::plainInlayHint)
  }

  fun format(): String? {
    requireInitialized()
    val formatted = workspace.format(fileHandle)
    return if (tyDefined(formatted)) formatted as String else null
  }

  fun signatureHelp(line: Int, column: Int): dynamic {
    requireInitialized()
    val value = workspace.signatureHelp(fileHandle, newTyPosition(line, column))
    if (!tyDefined(value)) return null
    return plainSignatureHelp(value)
  }

  fun semanticTokens(): Array<dynamic> {
    requireInitialized()
    val values = workspace.semanticTokens(fileHandle).unsafeCast<Array<dynamic>>()
    return mapTyArray(values, ::plainSemanticToken)
  }

  fun prepareRename(line: Int, column: Int): dynamic {
    requireInitialized()
    val value = workspace.prepareRename(fileHandle, newTyPosition(line, column))
    return if (tyDefined(value)) plainRange(value) else null
  }

  fun rename(line: Int, column: Int, newName: String): Array<dynamic> {
    requireInitialized()
    val values = workspace.rename(fileHandle, newTyPosition(line, column), newName)
      .unsafeCast<Array<dynamic>>()
    val results: dynamic = js("[]")
    values.forEach { edit ->
      try {
        if (edit.path as String == sourcePath) {
          val result = tyObject()
          result.path = sourcePath
          result.range = plainRange(edit.range)
          result.newText = edit.new_text
          results.push(result)
        }
      } finally {
        freeTyWrapper(edit)
      }
    }
    return results.unsafeCast<Array<dynamic>>()
  }

  fun dispose() {
    disposeRuntime()
    onStatus("idle", "ty stopped")
  }

  private fun snapshot(diagnostics: Array<dynamic>): dynamic {
    val result = tyObject()
    result.diagnostics = diagnostics
    result.version = runtimeVersion
    result.sourcePath = sourcePath
    return result
  }

  private fun syncSource(source: String) {
    if (source != lastSource) {
      workspace.updateFile(fileHandle, source)
      lastSource = source
    }
  }

  private fun mapDiagnostics(): Array<dynamic> {
    val values = workspace.checkFile(fileHandle).unsafeCast<Array<dynamic>>()
    return mapTyArray(values, ::plainDiagnostic)
  }

  private fun plainDiagnostic(value: dynamic): dynamic =
    try {
      val result = tyObject()
      result.id = value.id()
      result.message = value.message()
      result.severity = value.severity()
      result.tags = value.tags()
      result.range = plainRange(value.toRange(workspace))
      result.display = value.display(workspace)
      result
    } finally {
      freeTyWrapper(value)
    }

  private fun plainCompletion(value: dynamic): dynamic =
    try {
      val result = tyObject()
      val name = value.name as String
      val insertText = value.insert_text
      result.name = name
      result.kind = value.kind
      result.insertText = if (tyDefined(insertText)) insertText else name
      result.detail = value.detail
      result.documentation = value.documentation
      result.moduleName = value.module_name
      result.additionalTextEdits = plainTextEdits(value.additional_text_edits)
      result
    } finally {
      freeTyWrapper(value)
    }

  private fun plainInlayHint(value: dynamic): dynamic =
    try {
      val result = tyObject()
      val rawParts = value.label.unsafeCast<Array<dynamic>>()
      val parts = mapTyArray(rawParts) { part ->
        try {
          val mapped = tyObject()
          mapped.label = part.label
          val location = part.location
          if (tyDefined(location)) {
            val mappedLocation = plainLocationLink(location, sameFileOnly = true)
            if (mappedLocation != null) mapped.location = mappedLocation
          }
          mapped
        } finally {
          freeTyWrapper(part)
        }
      }
      result.labelParts = parts
      result.label = parts.joinToString("") { it.label as String }
      result.position = plainPosition(value.position)
      result.kind = value.kind
      result.textEdits = plainTextEdits(value.text_edits)
      result
    } finally {
      freeTyWrapper(value)
    }

  private fun plainSignatureHelp(value: dynamic): dynamic =
    try {
      val result = tyObject()
      val rawSignatures = value.signatures.unsafeCast<Array<dynamic>>()
      result.signatures = mapTyArray(rawSignatures) { signature ->
        try {
          val mapped = tyObject()
          mapped.label = signature.label
          mapped.documentation = signature.documentation
          mapped.activeParameter = signature.active_parameter
          val rawParameters = signature.parameters.unsafeCast<Array<dynamic>>()
          mapped.parameters = mapTyArray(rawParameters) { parameter ->
            try {
              val mappedParameter = tyObject()
              mappedParameter.label = parameter.label
              mappedParameter.documentation = parameter.documentation
              mappedParameter
            } finally {
              freeTyWrapper(parameter)
            }
          }
          mapped
        } finally {
          freeTyWrapper(signature)
        }
      }
      result.activeSignature = value.active_signature
      result
    } finally {
      freeTyWrapper(value)
    }

  private fun plainSemanticToken(value: dynamic): dynamic =
    try {
      val result = tyObject()
      result.kind = value.kind
      result.modifiers = value.modifiers
      result.range = plainRange(value.range)
      result
    } finally {
      freeTyWrapper(value)
    }

  private fun plainTextEdits(rawValue: dynamic): Array<dynamic> {
    if (!tyDefined(rawValue)) return emptyTyArray()
    val values = rawValue.unsafeCast<Array<dynamic>>()
    return mapTyArray(values) { edit ->
      try {
        val result = tyObject()
        result.range = plainRange(edit.range)
        result.text = edit.new_text
        result
      } finally {
        freeTyWrapper(edit)
      }
    }
  }

  private fun plainLocationLink(value: dynamic, sameFileOnly: Boolean): dynamic =
    try {
      val path = value.path as String
      if (sameFileOnly && path != sourcePath) return null

      val result = tyObject()
      result.path = path
      result.fullRange = plainRange(value.full_range)
      val selectionRange = value.selection_range
      result.selectionRange = if (tyDefined(selectionRange)) plainRange(selectionRange) else null
      val originRange = value.origin_selection_range
      result.originSelectionRange = if (tyDefined(originRange)) plainRange(originRange) else null
      result
    } finally {
      freeTyWrapper(value)
    }

  private fun plainRange(value: dynamic): dynamic {
    if (!tyDefined(value)) return null
    return try {
      val result = tyObject()
      result.start = plainPosition(value.start)
      result.end = plainPosition(value.end)
      result
    } finally {
      freeTyWrapper(value)
    }
  }

  private fun plainPosition(value: dynamic): dynamic =
    try {
      val result = tyObject()
      result.line = (value.line as Number).toInt()
      result.column = (value.column as Number).toInt()
      result
    } finally {
      freeTyWrapper(value)
    }

  private fun newTyPosition(line: Int, column: Int): dynamic =
    js("(Position, line, column) => new Position(line, column)")(ty.Position, line, column)

  private fun newTyRange(
    startLine: Int,
    startColumn: Int,
    endLine: Int,
    endColumn: Int
  ): dynamic = js("(Range, start, end) => new Range(start, end)")(
    ty.Range,
    newTyPosition(startLine, startColumn),
    newTyPosition(endLine, endColumn)
  )

  private fun requireInitialized() {
    check(tyDefined(workspace) && tyDefined(fileHandle)) { "TyEngine.initialize must complete first" }
  }

  private fun disposeRuntime() {
    val currentWorkspace = workspace
    val currentFile = fileHandle
    workspace = null
    fileHandle = null
    lastSource = null
    runtimeVersion = ""

    if (tyDefined(currentWorkspace)) {
      try {
        if (tyDefined(currentFile)) currentWorkspace.closeFile(currentFile)
      } finally {
        freeTyWrapper(currentWorkspace)
      }
    } else if (tyDefined(currentFile)) {
      freeTyWrapper(currentFile)
    }
  }
}

private fun newTyWorkspace(
  constructor: dynamic,
  root: String,
  positionEncoding: dynamic,
  options: dynamic
): dynamic =
  js("(Workspace, root, encoding, options) => new Workspace(root, encoding, options)")(
    constructor,
    root,
    positionEncoding,
    options
  )

private fun tyObject(): dynamic = js("({})")

private fun tyWasmReadyPromise(): dynamic =
  js("globalThis.tidyparseTyWasmReady")

private fun emptyTyArray(): Array<dynamic> =
  js("[]").unsafeCast<Array<dynamic>>()

private fun mapTyArray(
  values: Array<dynamic>,
  transform: (dynamic) -> dynamic
): Array<dynamic> {
  val result: dynamic = js("[]")
  values.forEach { value -> result.push(transform(value)) }
  return result.unsafeCast<Array<dynamic>>()
}

private fun tyDefined(value: dynamic): Boolean =
  js("(value) => value !== null && value !== undefined")(value) as Boolean

private fun freeTyWrapper(value: dynamic) {
  js(
    """(value) => {
      if (value && typeof value.free === "function") {
        try { value.free(); } catch (_) {}
      }
    }"""
  )(value)
}
