import kotlinx.browser.document
import kotlinx.browser.window
import org.w3c.dom.HTMLTextAreaElement
import org.w3c.dom.events.Event

fun initTidyCodeMirror(options: dynamic = null): dynamic {
  val w = window.asDynamic()
  if (w.cmEditor != null && w.cmEditor != js("undefined")) return w.cmEditor

  val textarea = document.getElementById("tidyparse-input") as? HTMLTextAreaElement ?: return null
  val codeMirror = w.CodeMirror
  if (codeMirror == null || codeMirror == js("undefined")) return null

  val defaults = js("{}")
  defaults.mode = "text/plain"
  defaults.theme = "material-darker"
  defaults.lineNumbers = true
  defaults.lineWrapping = false
  defaults.indentUnit = 2
  defaults.tabSize = 2
  defaults.viewportMargin = js("Infinity")
  defaults.extraKeys = js("({ Tab: false })")

  val editor = codeMirror.fromTextArea(textarea, js("Object").assign(defaults, options ?: js("{}")))

  editor.on("change") { _: dynamic, change: dynamic ->
    clearFreshCodeMirrorInsertion(editor)
    val insertedText = change.text.join("\n") as String
    if (
      editor.tidyparseAutomaticInsertionActive != true &&
      change.origin in arrayOf("+input", "*compose") &&
      insertedText.isNotEmpty()
    ) recordFreshCodeMirrorInsertion(editor)
    editor.tidyparseLastChangeOrigin = change.origin
    syncCodeMirrorTextareaAndEvents(editor, textarea, dispatchInput = true)
  }

  editor.on("cursorActivity") { _: dynamic ->
    invalidateFreshCodeMirrorInsertionIfStateChanged(editor)
    syncCodeMirrorTextareaAndEvents(editor, textarea, dispatchInput = false)
  }

  editor.getInputField().addEventListener("compositionstart", {
    clearFreshCodeMirrorInsertion(editor)
  })
  editor.getInputField().addEventListener("compositionend", {
    // Contenteditable input reads the committed DOM text on a delayed poll.
    // Let that change run first so the snapshot describes the committed value.
    val commitDelay = if (editor.getOption("inputStyle") == "contenteditable") 100 else 0
    window.setTimeout({
      invalidateFreshCodeMirrorInsertionIfStateChanged(editor)
      syncCodeMirrorTextareaAndEvents(editor, textarea, dispatchInput = true)
    }, commitDelay)
  })

  syncCodeMirrorTextareaAndEvents(editor, textarea, dispatchInput = false)
  w.cmEditor = editor
  return editor
}

fun initPythonCodeMirror(): dynamic {
  installFixedHtmlHint()

  val options = js("{}")
  options.mode = "python"
  options.gutters = js("['CodeMirror-linenumbers', 'cm-warn-gutter']")
  options.indentUnit = 4
  options.tabSize = 4
  options.indentWithTabs = false

  return initTidyCodeMirror(options)
}

private fun clearFreshCodeMirrorInsertion(editor: dynamic) {
  editor.tidyparseFreshInsertionText = null
  editor.tidyparseFreshInsertionStart = null
  editor.tidyparseFreshInsertionEnd = null
}

private fun recordFreshCodeMirrorInsertion(editor: dynamic) {
  editor.tidyparseFreshInsertionText = editor.getValue()
  editor.tidyparseFreshInsertionStart =
    editor.indexFromPos(editor.getCursor("from"))
  editor.tidyparseFreshInsertionEnd =
    editor.indexFromPos(editor.getCursor("to"))
}

private fun invalidateFreshCodeMirrorInsertionIfStateChanged(editor: dynamic) {
  val text = editor.tidyparseFreshInsertionText
  if (text == null || text == js("undefined")) return

  val start = editor.indexFromPos(editor.getCursor("from"))
  val end = editor.indexFromPos(editor.getCursor("to"))
  if (
    text != editor.getValue() ||
    editor.tidyparseFreshInsertionStart != start ||
    editor.tidyparseFreshInsertionEnd != end
  ) clearFreshCodeMirrorInsertion(editor)
}

private fun installFixedHtmlHint() {
  val w = window.asDynamic()
  if (w.COMPLETIONS == null || w.COMPLETIONS == js("undefined")) w.COMPLETIONS = emptyArray<String>()
  if (w.fixedHtmlHint != null && w.fixedHtmlHint != js("undefined")) return

  w.fixedHtmlHint = js("""(function(cm) {
    function htmlToPlaintext(html) {
      const tmp = document.createElement("div");
      tmp.innerHTML = html;
      return (tmp.textContent || tmp.innerText || "").trim();
    }

    const cur = cm.getCursor();
    const lineStr = cm.getLine(cur.line);
    const from = CodeMirror.Pos(cur.line, (lineStr.match(/^\s*/) || [""])[0].length);
    const to = CodeMirror.Pos(cur.line, lineStr.length);
    const list = (window.COMPLETIONS || []).map(html => {
      const plain = htmlToPlaintext(html);
      return {
        text: plain,
        displayText: plain,
        _html: html,
        render: function(elt, data, completion) { elt.innerHTML = completion._html; }
      };
    });

    return { list, from, to };
  })""")
}

private fun syncCodeMirrorTextareaAndEvents(editor: dynamic, textarea: HTMLTextAreaElement, dispatchInput: Boolean) {
  editor.save()

  val from = editor.indexFromPos(editor.getCursor("from")) as Int
  val to = editor.indexFromPos(editor.getCursor("to")) as Int
  try {
    textarea.asDynamic().selectionStart = from
    textarea.asDynamic().selectionEnd = to
  } catch (_: dynamic) {}

  try { textarea.dispatchEvent(js("new Event('selectionchange', { bubbles: true })") as Event) } catch (_: dynamic) {}
  if (dispatchInput) {
    try { textarea.dispatchEvent(js("new Event('input', { bubbles: true })") as Event) } catch (_: dynamic) {}
  }
}
