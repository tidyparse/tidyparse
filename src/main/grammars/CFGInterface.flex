package ai.hypergraph.tidyparse.cfg;

import com.intellij.psi.tree.IElementType;
import ai.hypergraph.tidyparse.psi.CfgInterfaceTypes;
import com.intellij.psi.TokenType;

%%

%class CfgInterfaceLexer
%implements FlexLexer
%unicode
%function advance
%type IElementType

CRLF=\R
WHITE_SPACE=[\ \t]
VALUE_CHARACTER=[^\r\n\f#|]
SEPARATOR=[->]
OR=[|]
KEY_CHARACTER=[^\ \r\n\t\f\\|]
TRIPLE_DASH=---

%state KEY_STATE
%state WAITING_STATE
%state VALUE_STATE

%%

<YYINITIAL> {KEY_CHARACTER}+                    { yybegin(KEY_STATE); return CfgInterfaceTypes.KEY; }

<KEY_STATE> {WHITE_SPACE}+                      { yybegin(WAITING_STATE); return TokenType.WHITE_SPACE; }

<WAITING_STATE> {CRLF}({CRLF}|{WHITE_SPACE})+   { yybegin(YYINITIAL); return TokenType.WHITE_SPACE; }

<WAITING_STATE> {SEPARATOR}                     { yybegin(VALUE_STATE); return CfgInterfaceTypes.SEPARATOR; }

<VALUE_STATE> {OR}                              { yybegin(VALUE_STATE); return CfgInterfaceTypes.OR; }

<VALUE_STATE> {WHITE_SPACE}                     { yybegin(VALUE_STATE); return TokenType.WHITE_SPACE; }

<VALUE_STATE> {VALUE_CHARACTER}+                { yybegin(VALUE_STATE); return CfgInterfaceTypes.VALUE; }

<VALUE_STATE> {CRLF}({CRLF}|{WHITE_SPACE})+     { yybegin(YYINITIAL); return TokenType.WHITE_SPACE; }

({CRLF}|{WHITE_SPACE}|{TRIPLE_DASH})+           { yybegin(YYINITIAL); return TokenType.WHITE_SPACE; }

[^]                                             { return TokenType.BAD_CHARACTER; }