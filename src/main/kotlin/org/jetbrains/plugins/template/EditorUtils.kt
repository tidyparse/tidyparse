package org.jetbrains.plugins.template

import com.intellij.openapi.editor.Editor
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile

fun Editor.currentLine(): String =
    caretModel.let { it.visualLineStart to it.visualLineEnd }
        .let { (lineStart, lineEnd) -> document.getText(TextRange.create(lineStart, lineEnd)) }

fun PsiFile.getGrammarFile() = containingDirectory.files.firstOrNull { it.name.endsWith(".cfg") }