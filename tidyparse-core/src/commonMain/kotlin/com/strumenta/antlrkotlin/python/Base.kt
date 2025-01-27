package com.strumenta.antlrkotlin.python

import org.antlr.v4.kotlinruntime.*

abstract class Python3ParserBase(input: TokenStream) : Parser(input) {
  fun CannotBePlusMinus() = false
  fun CannotBeDotLpEq() = false
}

abstract class Python3LexerBase(input: CharStream) : Lexer(input) {
  fun onNewLine() { }
  fun openBrace() { }
  fun closeBrace() { }
  fun atStartOfInput() = true
}