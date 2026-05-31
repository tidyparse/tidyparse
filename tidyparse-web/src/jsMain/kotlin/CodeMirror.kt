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
