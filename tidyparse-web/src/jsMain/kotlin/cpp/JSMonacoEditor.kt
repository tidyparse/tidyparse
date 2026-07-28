import org.w3c.dom.HTMLElement
import kotlin.js.Promise

private const val MONACO_WORKSPACE_URI = "file:///home/web_user"
private const val MONACO_MARKER_OWNER = "clangd"
private const val MONACO_AUXILIARY_MODEL_LIMIT = 16

private var cppMonaco: dynamic = null
private var cppMonacoLanguageInstalled = false

/**
 * Monaco is required from inside the page setup instead of through a top-level
 * @JsModule import. The compiled application bundle is also used as the clangd
 * worker and service worker, where evaluating Monaco's DOM modules would fail.
 */
private fun loadCppMonaco(): dynamic {
  if (monacoValueDefined(cppMonaco)) return cppMonaco

  val global = js("globalThis")
  if (!monacoValueDefined(global.MonacoEnvironment)) {
    val workerModule =
      js("require('worker-loader?inline=no-fallback&esModule=false!monaco-editor/esm/vs/editor/editor.worker')")
    val workerConstructor =
      if (monacoValueDefined(workerModule.default)) workerModule.default
      else workerModule
    val environment = js("{}")
    environment.getWorker = { _: dynamic, _: dynamic ->
      js("new workerConstructor()")
    }
    global.MonacoEnvironment = environment
  }

  val monaco = js("require('monaco-editor/esm/vs/editor/edcore.main')")
  if (!cppMonacoLanguageInstalled) {
    val cpp = js("require('monaco-editor/esm/vs/basic-languages/cpp/cpp')")
    monaco.languages.register(
      js(
        """({
          id: 'cpp',
          extensions: ['.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx'],
          aliases: ['C', 'C++', 'cpp'],
          mimetypes: ['text/x-c', 'text/x-c++']
        })"""
      )
    )
    monaco.languages.setLanguageConfiguration("cpp", cpp.conf)
    monaco.languages.setMonarchTokensProvider("cpp", cpp.language)
    monaco.editor.defineTheme(
      "tidyparse-light",
      js(
        """({
          base: 'vs',
          inherit: true,
          rules: [
            { token: 'comment', foreground: '5f6f86', fontStyle: 'italic' },
            { token: 'keyword', foreground: '8b1a9b' },
            { token: 'number', foreground: '1750a5' },
            { token: 'string', foreground: 'a34812' },
            { token: 'type', foreground: '075985' },
            { token: 'type.identifier', foreground: '075985' }
          ],
          colors: {
            'editor.background': '#fbfcfe',
            'editor.foreground': '#172033',
            'editorLineNumber.foreground': '#8a96a8',
            'editorLineNumber.activeForeground': '#334155',
            'editorCursor.foreground': '#2563eb',
            'editor.selectionBackground': '#bfdbfe',
            'editor.inactiveSelectionBackground': '#dbeafe',
            'editorIndentGuide.background1': '#e2e8f0',
            'editorIndentGuide.activeBackground1': '#94a3b8'
          }
        })"""
      )
    )
    monaco.editor.defineTheme(
      "tidyparse-dark",
      js(
        """({
          base: 'vs-dark',
          inherit: true,
          rules: [
            { token: 'comment', foreground: '7f8da3', fontStyle: 'italic' },
            { token: 'keyword', foreground: 'c792ea' },
            { token: 'number', foreground: '82aaff' },
            { token: 'string', foreground: 'ecc48d' },
            { token: 'type', foreground: '89ddff' },
            { token: 'type.identifier', foreground: '89ddff' }
          ],
          colors: {
            'editor.background': '#111721',
            'editor.foreground': '#d8e0ed',
            'editorLineNumber.foreground': '#68758a',
            'editorLineNumber.activeForeground': '#cbd5e1',
            'editorCursor.foreground': '#60a5fa',
            'editor.selectionBackground': '#1e3a5f',
            'editor.inactiveSelectionBackground': '#172d4d',
            'editorIndentGuide.background1': '#293344',
            'editorIndentGuide.activeBackground1': '#52627a'
          }
        })"""
      )
    )
    cppMonacoLanguageInstalled = true
  }

  cppMonaco = monaco
  return monaco
}

/**
 * A thin Kotlin/JS adapter around standalone Monaco. Monaco owns all editor
 * widgets and keybindings; the existing clangd client supplies LSP results.
 */
class JSMonacoEditor(
  container: HTMLElement,
  fileName: String,
  text: String,
  darkTheme: Boolean,
  private val onChange: (String) -> Unit,
  private val onPosition: (line: Int, column: Int) -> Unit,
  private val onOpenedFile: (String) -> Unit,
  private val onRun: () -> Unit
) {
  private val monaco = loadCppMonaco()
  private val editor: dynamic
  private var mainModel: dynamic
  private var clangdClient: JSClangdClient? = null
  private var changingModel = false
  private var requestedReadOnly = false
  private var semanticTokensInstalled = false
  private val semanticTokenListeners = mutableListOf<dynamic>()
  private val disposables = mutableListOf<dynamic>()
  private val auxiliaryModels = mutableListOf<dynamic>()

  init {
    mainModel = createModel(fileName, text)

    val options = js("{}")
    options.model = mainModel
    options.automaticLayout = true
    options.theme = if (darkTheme) "tidyparse-dark" else "tidyparse-light"
    options.fontFamily = "\"SFMono-Regular\", Consolas, \"Liberation Mono\", Menlo, monospace"
    options.fontSize = 15
    options.lineHeight = 25
    options.tabSize = 2
    options.insertSpaces = true
    options.detectIndentation = false
    options.wordBasedSuggestions = "off"
    options.quickSuggestions = js("({ other: true, comments: false, strings: false })")
    options.quickSuggestionsDelay = 200
    options.suggestOnTriggerCharacters = true
    options.acceptSuggestionOnEnter = "on"
    options.parameterHints = js("({ enabled: true, cycle: true })")
    options.hover = js("({ enabled: true, delay: 300, sticky: true })")
    options.inlayHints = js("({ enabled: 'offUnlessPressed' })")
    options["semanticHighlighting.enabled"] = true
    options.bracketPairColorization = js("({ enabled: true, independentColorPoolPerBracketType: true })")
    options.guides = js("({ bracketPairs: true, bracketPairsHorizontal: 'active', highlightActiveBracketPair: true })")
    options.glyphMargin = true
    options.folding = true
    options.foldingHighlight = true
    options.showFoldingControls = "mouseover"
    options.lightbulb = js("({ enabled: 'on' })")
    options.renderValidationDecorations = "on"
    options.renderWhitespace = "selection"
    options.scrollBeyondLastLine = false
    options.smoothScrolling = true
    options.padding = js("({ top: 14, bottom: 28 })")
    options.minimap = js("({ enabled: false })")
    options.fixedOverflowWidgets = true
    options.occurrencesHighlight = "singleFile"
    options.selectionHighlight = true
    options.links = true

    editor = monaco.editor.create(container, options)
    mainModel.updateOptions(js("({ tabSize: 2, insertSpaces: true, detectIndentation: false })"))
    disposables.add(editor.onDidChangeModelContent {
      if (!changingModel && editor.getModel() === mainModel) onChange(value())
    })
    disposables.add(editor.onDidChangeModel {
      updateModelReadOnly()
      val currentUri = editor.getModel()?.uri?.path as? String
      onOpenedFile(currentUri?.substringAfterLast('/')?.takeIf { it.isNotBlank() } ?: fileName)
    })
    disposables.add(editor.onDidChangeCursorPosition { event: dynamic ->
      val position = event.position
      onPosition(monacoNumber(position.lineNumber), monacoNumber(position.column))
    })

    val runKey = monacoNumber(monaco.KeyMod.CtrlCmd) or monacoNumber(monaco.KeyCode.Enter)
    editor.addCommand(runKey, { onRun() })
    installLanguageProviders()
    installEditorOpener()
    editor.setPosition(js("({ lineNumber: 1, column: 1 })"))
    onOpenedFile(fileName)
  }

  fun bindClangd(client: JSClangdClient) {
    clangdClient = client
  }

  fun enableClangdSemanticTokens() {
    if (semanticTokensInstalled) return
    val client = clangdClient ?: return
    val legend = client.semanticTokensLegend() ?: return

    val mappedLegend = js("{}")
    mappedLegend.tokenTypes = legend.tokenTypes.toTypedArray()
    mappedLegend.tokenModifiers = legend.tokenModifiers.toTypedArray()

    val provider = js("{}")
    provider.getLegend = { mappedLegend }
    provider.onDidChange = { listener: dynamic, _: dynamic, _: dynamic ->
      semanticTokenListeners.add(listener)
      val disposable = js("{}")
      disposable.dispose = {
        semanticTokenListeners.remove(listener)
        Unit
      }
      disposable
    }
    provider.provideDocumentSemanticTokens =
      { model: dynamic, _: dynamic, token: dynamic ->
        if (!isMainModel(model)) {
          Promise.resolve(null)
        } else {
          Promise<dynamic> { resolve, reject ->
            val handle = client.requestSemanticTokens { tokens, failure ->
              if (failure != null) {
                reject(monacoRequestError(failure))
              } else if (tokens == null) {
                resolve(null)
              } else {
                val result = js("{}")
                tokens.resultId?.let { result.resultId = it }
                result.data = monacoUint32Array(tokens.data)
                resolve(result)
              }
            }
            bindCancellation(token, handle)
          }
        }
      }
    provider.releaseDocumentSemanticTokens = { _: dynamic -> }
    disposables.add(monaco.languages.registerDocumentSemanticTokensProvider("cpp", provider))
    semanticTokensInstalled = true
  }

  fun refreshSemanticTokens() {
    if (!semanticTokensInstalled) return
    semanticTokenListeners.toList().forEach { listener ->
      try {
        listener()
      } catch (_: Throwable) {
      }
    }
  }

  fun value(): String = mainModel.getValue() as? String ?: ""

  fun focus() {
    editor.focus()
  }

  fun setReadOnly(readOnly: Boolean) {
    requestedReadOnly = readOnly
    updateModelReadOnly()
  }

  fun setTheme(dark: Boolean) {
    monaco.editor.setTheme(if (dark) "tidyparse-dark" else "tidyparse-light")
  }

  fun setDocument(fileName: String, text: String) {
    val previous = mainModel
    changingModel = true
    try {
      mainModel = createModel(fileName, text)
      mainModel.updateOptions(js("({ tabSize: 2, insertSpaces: true, detectIndentation: false })"))
      editor.setModel(mainModel)
      editor.setPosition(js("({ lineNumber: 1, column: 1 })"))
      editor.revealPosition(js("({ lineNumber: 1, column: 1 })"))
    } finally {
      changingModel = false
      previous.dispose()
    }
    clearDiagnostics()
    onOpenedFile(fileName)
    onPosition(1, 1)
  }

  fun setValue(text: String) {
    if (editor.getModel() !== mainModel) editor.setModel(mainModel)
    if (value() == text) {
      editor.setPosition(js("({ lineNumber: 1, column: 1 })"))
      editor.revealPosition(js("({ lineNumber: 1, column: 1 })"))
      onPosition(1, 1)
      return
    }
    changingModel = true
    try {
      mainModel.setValue(text)
      editor.setPosition(js("({ lineNumber: 1, column: 1 })"))
      editor.revealPosition(js("({ lineNumber: 1, column: 1 })"))
    } finally {
      changingModel = false
    }
    onChange(text)
    onPosition(1, 1)
  }

  fun setDiagnostics(diagnostics: List<ClangdDiagnostic>) {
    val markers = diagnostics.map { diagnostic ->
      val marker = js("{}")
      marker.startLineNumber = diagnostic.range.start.line + 1
      marker.startColumn = diagnostic.range.start.character + 1
      marker.endLineNumber = diagnostic.range.end.line + 1
      marker.endColumn = diagnostic.range.end.character + 1
      marker.severity = when (diagnostic.severity) {
        1 -> monaco.MarkerSeverity.Error
        2 -> monaco.MarkerSeverity.Warning
        3 -> monaco.MarkerSeverity.Info
        else -> monaco.MarkerSeverity.Hint
      }
      marker.message = diagnostic.message
      if (!diagnostic.source.isNullOrBlank()) marker.source = diagnostic.source
      if (!diagnostic.code.isNullOrBlank()) marker.code = diagnostic.code
      marker
    }.toTypedArray()
    monaco.editor.setModelMarkers(mainModel, MONACO_MARKER_OWNER, markers)
  }

  fun clearDiagnostics() {
    monaco.editor.setModelMarkers(mainModel, MONACO_MARKER_OWNER, emptyArray<dynamic>())
  }

  fun dispose() {
    disposables.forEach { disposable ->
      try {
        disposable.dispose()
      } catch (_: Throwable) {
      }
    }
    disposables.clear()
    editor.dispose()
    mainModel.dispose()
    auxiliaryModels.forEach { model ->
      try {
        model.dispose()
      } catch (_: Throwable) {
      }
    }
    auxiliaryModels.clear()
    semanticTokenListeners.clear()
  }

  private fun createModel(fileName: String, text: String): dynamic {
    val uri = monaco.Uri.parse("$MONACO_WORKSPACE_URI/${encodeMonacoPathSegment(fileName)}")
    monaco.editor.getModel(uri)?.let { existing: dynamic ->
      existing.dispose()
    }
    return monaco.editor.createModel(text, "cpp", uri)
  }

  private fun installLanguageProviders() {
    installCompletionProvider()
    installHoverProvider()
    installNavigationProviders()
    installReferenceProvider()
    installHighlightProvider()
    installDocumentSymbolProvider()
    installSignatureHelpProvider()
  }

  private fun installEditorOpener() {
    val opener = js("{}")
    opener.openCodeEditor =
      { _: dynamic, resource: dynamic, selectionOrPosition: dynamic ->
        val target = monaco.editor.getModel(resource)
        if (!monacoValueDefined(target)) {
          false
        } else {
          editor.setModel(target)
          if (monacoValueDefined(selectionOrPosition)) {
            if (monacoValueDefined(selectionOrPosition.startLineNumber)) {
              editor.setSelection(selectionOrPosition)
              editor.revealRangeInCenter(selectionOrPosition)
            } else {
              editor.setPosition(selectionOrPosition)
              editor.revealPositionInCenter(selectionOrPosition)
            }
          }
          editor.focus()
          true
        }
      }
    disposables.add(monaco.editor.registerEditorOpener(opener))
  }

  private fun installCompletionProvider() {
    val provider = js("{}")
    provider.triggerCharacters = arrayOf(".", ">", ":", "\"", "<")
    provider.provideCompletionItems =
      { model: dynamic, position: dynamic, _: dynamic, token: dynamic ->
        if (!isMainModel(model)) {
          Promise.resolve(js("({ suggestions: [] })"))
        } else {
          val line = monacoNumber(position.lineNumber) - 1
          val column = monacoNumber(position.column) - 1
          Promise<dynamic> { resolve, _ ->
            val client = clangdClient
            if (client == null) {
              resolve(js("({ suggestions: [] })"))
            } else {
              val handle = client.requestCompletion(line, column) { completions ->
                val result = js("{}")
                result.suggestions = completions.map(::completionItem).toTypedArray()
                resolve(result)
              }
              bindCancellation(token, handle)
            }
          }
        }
      }
    disposables.add(monaco.languages.registerCompletionItemProvider("cpp", provider))
  }

  private fun completionItem(item: ClangdCompletion): dynamic {
    val value = js("{}")
    value.label = item.label
    value.kind = completionKind(item.kind)
    value.insertText = item.textEdit?.newText ?: item.insertText ?: item.label
    value.detail = item.detail
    value.sortText = item.sortText
    value.filterText = item.filterText
    value.preselect = item.preselect
    if (!item.documentation.isNullOrBlank()) {
      value.documentation = js("({ value: item.documentation })")
    }
    item.textEdit?.let { value.range = monacoRange(it.range) }
    if (item.insertTextFormat == 2) {
      value.insertTextRules = monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
    }
    if (item.additionalTextEdits.isNotEmpty()) {
      value.additionalTextEdits = item.additionalTextEdits.map { edit ->
        val mapped = js("{}")
        mapped.range = monacoRange(edit.range)
        mapped.text = edit.newText
        mapped
      }.toTypedArray()
    }
    if (item.commitCharacters.isNotEmpty()) {
      value.commitCharacters = item.commitCharacters.toTypedArray()
    }
    if (item.deprecated) {
      value.tags = arrayOf(monaco.languages.CompletionItemTag.Deprecated)
    }
    return value
  }

  private fun installHoverProvider() {
    val provider = js("{}")
    provider.provideHover = { model: dynamic, position: dynamic, token: dynamic ->
      if (!isMainModel(model)) {
        Promise.resolve(null)
      } else {
        Promise<dynamic> { resolve, _ ->
          val client = clangdClient
          if (client == null) {
            resolve(null)
          } else {
            val handle = client.requestHover(
              monacoNumber(position.lineNumber) - 1,
              monacoNumber(position.column) - 1
            ) { hover ->
              if (hover == null) {
                resolve(null)
              } else {
                val result = js("{}")
                result.contents = hover.contents.map { content ->
                  val markdown = js("{}")
                  markdown.value = when {
                    content.language != null ->
                      "```${content.language}\n${content.value}\n```"
                    else -> content.value
                  }
                  markdown.isTrusted = false
                  markdown.supportHtml = false
                  markdown
                }.toTypedArray()
                hover.range?.let { result.range = monacoRange(it) }
                resolve(result)
              }
            }
            bindCancellation(token, handle)
          }
        }
      }
    }
    disposables.add(monaco.languages.registerHoverProvider("cpp", provider))
  }

  private fun installNavigationProviders() {
    fun provider(request: (
      JSClangdClient,
      Int,
      Int,
      (List<ClangdNavigationTarget>) -> Unit
    ) -> ClangdRequestHandle?): dynamic {
      val result = js("{}")
      result.provideDefinition = callback@{ model: dynamic, position: dynamic, token: dynamic ->
        if (!isMainModel(model)) return@callback Promise.resolve(emptyArray<dynamic>())
        Promise<dynamic> { resolve, _ ->
          val client = clangdClient
          if (client == null) {
            resolve(emptyArray<dynamic>())
          } else {
            val handle = request(
              client,
              monacoNumber(position.lineNumber) - 1,
              monacoNumber(position.column) - 1
            ) { targets ->
              prepareNavigationTargets(targets, token) { prepared ->
                resolve(prepared)
              }
            }
            bindCancellation(token, handle)
          }
        }
      }
      return result
    }

    disposables.add(monaco.languages.registerDefinitionProvider(
      "cpp",
      provider { client, line, column, callback ->
        client.requestDefinition(line, column, callback)
      }
    ))

    val declaration = provider { client, line, column, callback ->
      client.requestDeclaration(line, column, callback)
    }
    declaration.provideDeclaration = declaration.provideDefinition
    declaration.provideDefinition = js("undefined")
    disposables.add(monaco.languages.registerDeclarationProvider("cpp", declaration))

    val implementation = provider { client, line, column, callback ->
      client.requestImplementation(line, column, callback)
    }
    implementation.provideImplementation = implementation.provideDefinition
    implementation.provideDefinition = js("undefined")
    disposables.add(monaco.languages.registerImplementationProvider("cpp", implementation))
  }

  private fun installReferenceProvider() {
    val provider = js("{}")
    provider.provideReferences =
      callback@{ model: dynamic, position: dynamic, context: dynamic, token: dynamic ->
        if (!isMainModel(model)) return@callback Promise.resolve(emptyArray<dynamic>())
        Promise<dynamic> { resolve, _ ->
          val client = clangdClient
          if (client == null) {
            resolve(emptyArray<dynamic>())
          } else {
            val includeDeclaration = context?.includeDeclaration as? Boolean ?: true
            val handle = client.requestReferences(
              monacoNumber(position.lineNumber) - 1,
              monacoNumber(position.column) - 1,
              includeDeclaration
            ) { targets ->
              prepareNavigationTargets(targets, token) { prepared ->
                resolve(prepared)
              }
            }
            bindCancellation(token, handle)
          }
        }
      }
    disposables.add(monaco.languages.registerReferenceProvider("cpp", provider))
  }

  private fun installHighlightProvider() {
    val provider = js("{}")
    provider.provideDocumentHighlights =
      callback@{ model: dynamic, position: dynamic, token: dynamic ->
        if (!isMainModel(model)) return@callback Promise.resolve(emptyArray<dynamic>())
        Promise<dynamic> { resolve, _ ->
          val client = clangdClient
          if (client == null) {
            resolve(emptyArray<dynamic>())
          } else {
            val handle = client.requestDocumentHighlights(
              monacoNumber(position.lineNumber) - 1,
              monacoNumber(position.column) - 1
            ) { highlights ->
              resolve(highlights.map { highlight ->
                val mapped = js("{}")
                mapped.range = monacoRange(highlight.range)
                mapped.kind = when (highlight.kind) {
                  2 -> monaco.languages.DocumentHighlightKind.Read
                  3 -> monaco.languages.DocumentHighlightKind.Write
                  else -> monaco.languages.DocumentHighlightKind.Text
                }
                mapped
              }.toTypedArray())
            }
            bindCancellation(token, handle)
          }
        }
      }
    disposables.add(monaco.languages.registerDocumentHighlightProvider("cpp", provider))
  }

  private fun installDocumentSymbolProvider() {
    val provider = js("{}")
    provider.provideDocumentSymbols = { model: dynamic, token: dynamic ->
      if (!isMainModel(model)) {
        Promise.resolve(emptyArray<dynamic>())
      } else {
        Promise<dynamic> { resolve, _ ->
          val client = clangdClient
          if (client == null) {
            resolve(emptyArray<dynamic>())
          } else {
            val handle = client.requestDocumentSymbols { symbols ->
              resolve(symbols.map(::documentSymbol).toTypedArray())
            }
            bindCancellation(token, handle)
          }
        }
      }
    }
    disposables.add(monaco.languages.registerDocumentSymbolProvider("cpp", provider))
  }

  private fun documentSymbol(symbol: ClangdDocumentSymbol): dynamic {
    val mapped = js("{}")
    mapped.name = symbol.name
    mapped.detail = symbol.detail ?: symbol.containerName ?: ""
    mapped.kind = symbolKind(symbol.kind)
    mapped.range = monacoRange(symbol.range)
    mapped.selectionRange = monacoRange(symbol.selectionRange)
    if (symbol.tags.isNotEmpty()) {
      mapped.tags = symbol.tags.map {
        monaco.languages.SymbolTag.Deprecated
      }.toTypedArray()
    }
    if (symbol.children.isNotEmpty()) {
      mapped.children = symbol.children.map(::documentSymbol).toTypedArray()
    }
    return mapped
  }

  private fun installSignatureHelpProvider() {
    val provider = js("{}")
    provider.signatureHelpTriggerCharacters = arrayOf("(", ",")
    provider.signatureHelpRetriggerCharacters = arrayOf(",")
    provider.provideSignatureHelp =
      callback@{ model: dynamic, position: dynamic, context: dynamic, token: dynamic ->
        if (!isMainModel(model)) return@callback Promise.resolve(null)
        Promise<dynamic> { resolve, _ ->
          val client = clangdClient
          if (client == null) {
            resolve(null)
          } else {
            val triggerKind = monacoNumber(context?.triggerKind).coerceAtLeast(1)
            val triggerCharacter = context?.triggerCharacter as? String
            val isRetrigger = context?.isRetrigger as? Boolean ?: false
            val handle = client.requestSignatureHelp(
              monacoNumber(position.lineNumber) - 1,
              monacoNumber(position.column) - 1,
              triggerKind,
              triggerCharacter,
              isRetrigger
            ) { help ->
              if (help == null) {
                resolve(null)
              } else {
                val value = js("{}")
                value.activeSignature = help.activeSignature
                value.activeParameter = help.activeParameter
                value.signatures = help.signatures.map { signature ->
                  val mapped = js("{}")
                  mapped.label = signature.label
                  signature.documentation?.let { mapped.documentation = it.value }
                  mapped.parameters = signature.parameters.map { parameter ->
                    val mappedParameter = js("{}")
                    mappedParameter.label =
                      parameter.label
                        ?: if (parameter.labelStart != null && parameter.labelEnd != null) {
                          arrayOf(parameter.labelStart, parameter.labelEnd)
                        } else {
                          ""
                        }
                    parameter.documentation?.let {
                      mappedParameter.documentation = it.value
                    }
                    mappedParameter
                  }.toTypedArray()
                  mapped
                }.toTypedArray()
                val result = js("{}")
                result.value = value
                result.dispose = {}
                resolve(result)
              }
            }
            bindCancellation(token, handle)
          }
        }
      }
    disposables.add(monaco.languages.registerSignatureHelpProvider("cpp", provider))
  }

  private fun prepareNavigationTargets(
    targets: List<ClangdNavigationTarget>,
    token: dynamic,
    callback: (Array<dynamic>) -> Unit
  ) {
    val missingUris = targets
      .map { it.uri }
      .distinct()
      .filter { uri -> !monacoValueDefined(monaco.editor.getModel(monaco.Uri.parse(uri))) }
    val client = clangdClient
    if (missingUris.isEmpty() || client == null) {
      callback(navigationTargetsWithModels(targets))
      return
    }

    var remaining = missingUris.size
    missingUris.forEach { uri ->
      val handle = client.readVirtualFile(uri) { file, _ ->
        if (file != null && !monacoValueDefined(monaco.editor.getModel(monaco.Uri.parse(uri)))) {
          try {
            val model = monaco.editor.createModel(file.text, "cpp", monaco.Uri.parse(uri))
            model.updateOptions(js("({ tabSize: 2, insertSpaces: true, detectIndentation: false })"))
            rememberAuxiliaryModel(model)
          } catch (_: Throwable) {
          }
        }
        remaining--
        if (remaining == 0) callback(navigationTargetsWithModels(targets))
      }
      bindCancellation(token, handle)
    }
  }

  private fun navigationTargetsWithModels(targets: List<ClangdNavigationTarget>): Array<dynamic> =
    targets
      .filter { monacoValueDefined(monaco.editor.getModel(monaco.Uri.parse(it.uri))) }
      .map { target ->
        val mapped = js("{}")
        mapped.uri = monaco.Uri.parse(target.uri)
        mapped.range = monacoRange(target.selectionRange)
        mapped
      }
      .toTypedArray()

  private fun rememberAuxiliaryModel(model: dynamic) {
    auxiliaryModels.add(model)
    while (auxiliaryModels.size > MONACO_AUXILIARY_MODEL_LIMIT) {
      val current = editor.getModel()
      val evictionIndex = auxiliaryModels.indexOfFirst { candidate -> candidate !== current }
      if (evictionIndex < 0) return
      val evicted = auxiliaryModels.removeAt(evictionIndex)
      try {
        evicted.dispose()
      } catch (_: Throwable) {
      }
    }
  }

  private fun updateModelReadOnly() {
    val current = editor.getModel()
    val readOnly = requestedReadOnly || (monacoValueDefined(current) && current !== mainModel)
    editor.updateOptions(js("({ readOnly: readOnly })"))
  }

  private fun monacoRange(range: ClangdRange): dynamic {
    val mapped = js("{}")
    mapped.startLineNumber = range.start.line + 1
    mapped.startColumn = range.start.character + 1
    mapped.endLineNumber = range.end.line + 1
    mapped.endColumn = range.end.character + 1
    return mapped
  }

  private fun isMainModel(model: dynamic): Boolean =
    model === mainModel ||
      model?.uri?.toString() == mainModel.uri.toString()

  private fun bindCancellation(token: dynamic, handle: ClangdRequestHandle?) {
    if (handle == null || !monacoValueDefined(token?.onCancellationRequested)) return
    token.onCancellationRequested { handle.cancel() }
  }

  private fun completionKind(kind: Int?): dynamic = when (kind) {
    1 -> monaco.languages.CompletionItemKind.Text
    2 -> monaco.languages.CompletionItemKind.Method
    3 -> monaco.languages.CompletionItemKind.Function
    4 -> monaco.languages.CompletionItemKind.Constructor
    5 -> monaco.languages.CompletionItemKind.Field
    6 -> monaco.languages.CompletionItemKind.Variable
    7 -> monaco.languages.CompletionItemKind.Class
    8 -> monaco.languages.CompletionItemKind.Interface
    9 -> monaco.languages.CompletionItemKind.Module
    10 -> monaco.languages.CompletionItemKind.Property
    11 -> monaco.languages.CompletionItemKind.Unit
    12 -> monaco.languages.CompletionItemKind.Value
    13 -> monaco.languages.CompletionItemKind.Enum
    14 -> monaco.languages.CompletionItemKind.Keyword
    15 -> monaco.languages.CompletionItemKind.Snippet
    16 -> monaco.languages.CompletionItemKind.Color
    17 -> monaco.languages.CompletionItemKind.File
    18 -> monaco.languages.CompletionItemKind.Reference
    19 -> monaco.languages.CompletionItemKind.Folder
    20 -> monaco.languages.CompletionItemKind.EnumMember
    21 -> monaco.languages.CompletionItemKind.Constant
    22 -> monaco.languages.CompletionItemKind.Struct
    23 -> monaco.languages.CompletionItemKind.Event
    24 -> monaco.languages.CompletionItemKind.Operator
    25 -> monaco.languages.CompletionItemKind.TypeParameter
    else -> monaco.languages.CompletionItemKind.Text
  }

  private fun symbolKind(kind: Int): dynamic {
    val mapped = (kind - 1).coerceIn(0, 25)
    return mapped
  }
}

private fun monacoNumber(value: dynamic): Int = when (value) {
  is Int -> value
  is Number -> value.toInt()
  else -> 0
}

private fun monacoValueDefined(value: dynamic): Boolean =
  value != null && jsTypeOf(value) != "undefined"

private fun monacoUint32Array(values: List<Int>): dynamic {
  val size = values.size
  val result = js("new Uint32Array(size)")
  values.forEachIndexed { index, value -> result[index] = value }
  return result
}

private fun monacoRequestError(failure: ClangdRequestFailure): dynamic {
  val message = failure.message
  val error = js("new Error(message)")
  // ContentModified is an expected transient result while the user is typing.
  // Monaco retains the prior tokens and refetches only when both the Error name
  // and message use its recognized cancellation value.
  if (failure.code == -32800 || failure.code == -32801) {
    error.name = "Canceled"
    error.message = "Canceled"
  }
  return error
}

private fun encodeMonacoPathSegment(value: String): String =
  js("encodeURIComponent(value)") as String
