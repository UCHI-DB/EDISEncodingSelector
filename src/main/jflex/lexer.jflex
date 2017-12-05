/**
 * Lexer for Pattern Extractor
 */
package edu.uchicago.cs.encsel.ptnmining.parser;

import java_cup.runtime.*;
%%
%class Lexer
%unicode
%cupsym Sym
%function scan
%type Token

LineTerminator = \r|\n|\r\n
Whitespace     = ({LineTerminator} | [ \t\f])+

IntLiteral=[0-9]+
DoubleLiteral=[0-9]+\.[0-9]+
WordLiteral=[a-zA-Z][a-zA-Z\.']*

%%

<YYINITIAL> {
{IntLiteral}        {return new TInt(yytext());}
{DoubleLiteral}     {return new TDouble(yytext());}
{WordLiteral}       {return new TWord(yytext());}
{Whitespace}        {return new TSpace();}
"-"                 {return new TSymbol(yytext());}
"_"                 {return new TSymbol(yytext());}
"("                 {return new TPara(Sym.PARA, true);}
")"                 {return new TPara(Sym.PARA, false);}
"["                 {return new TPara(Sym.BRAK, true);}
"]"                 {return new TPara(Sym.BRAK, false);}
"{"                 {return new TPara(Sym.BRAC, true);}
"}"                 {return new TPara(Sym.BRAC, false);}
","                 {return new TSymbol(yytext());}
"."                 {return new TSymbol(yytext());}
":"                 {return new TSymbol(yytext());}
";"                 {return new TSymbol(yytext());}
"/"                 {return new TSymbol(yytext());}
"\\"                 {return new TSymbol(yytext());}
}
.                   {return new TSymbol(yytext());}

