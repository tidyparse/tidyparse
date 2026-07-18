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

  editor.on("change") { _: dynamic -> syncCodeMirrorTextareaAndEvents(editor, textarea, dispatchInput = true) }

  editor.on("cursorActivity") { _: dynamic -> syncCodeMirrorTextareaAndEvents(editor, textarea, dispatchInput = false) }

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
