package org.jetbrains.plugins.template

import com.intellij.openapi.editor.Editor
import com.intellij.openapi.util.TextRange

fun Editor.currentLine(): String =
    caretModel.let { it.visualLineStart to it.visualLineEnd }.let { (lineStart, lineEnd) ->
        document.getText(TextRange.create(lineStart, lineEnd))
    }