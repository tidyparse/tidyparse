(function () {
  function syncTextareaAndEvents(editor, textarea, dispatchInput) {
    editor.save();

    const from = editor.indexFromPos(editor.getCursor("from"));
    const to = editor.indexFromPos(editor.getCursor("to"));
    try {
      textarea.selectionStart = from;
      textarea.selectionEnd = to;
    } catch (e) {}

    try { textarea.dispatchEvent(new Event("selectionchange", { bubbles: true })); } catch (e) {}
    if (dispatchInput) {
      try { textarea.dispatchEvent(new Event("input", { bubbles: true })); } catch (e) {}
    }
  }

  window.initTidyCodeMirror = function initTidyCodeMirror(options) {
    if (window.cmEditor) return window.cmEditor;

    const textarea = document.getElementById("tidyparse-input");
    if (!textarea || !window.CodeMirror) return null;

    const editor = CodeMirror.fromTextArea(textarea, Object.assign({
      mode: "text/plain",
      theme: "material-darker",
      lineNumbers: true,
      lineWrapping: false,
      indentUnit: 2,
      tabSize: 2,
      viewportMargin: Infinity,
      extraKeys: {
        Tab: false
      }
    }, options || {}));

    editor.on("change", function () {
      syncTextareaAndEvents(editor, textarea, true);
    });

    editor.on("cursorActivity", function () {
      syncTextareaAndEvents(editor, textarea, false);
    });

    syncTextareaAndEvents(editor, textarea, false);
    window.cmEditor = editor;
    return editor;
  };
})();
