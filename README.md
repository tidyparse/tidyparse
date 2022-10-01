<p align="center"><a href="https://plugins.jetbrains.com/plugin/19570-tidyparse"><img src="src/main/resources/META-INF/pluginIcon.svg" alt="Tidyparse Logo" height="160"></a></p>

<h1 align="center">Tidyparse</h1>
<p align="center">
 	<a href="https://plugins.jetbrains.com/plugin/19570-tidyparse" title="Tidyparse"><img src="https://img.shields.io/jetbrains/plugin/v/19570-tidyparse.svg"></a>
 	<a href="LICENSE" title="License"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<!-- Plugin description -->
The main goal of this project is to speed up the process of learning a new language by suggesting ways to fix source code.

Tidyparse expects a file ending in `*.tidy` which contains a context free grammar, followed by three consecutive dashes (`---`), followed by the string to parse (with optional holes). If the string is valid according to the CFG, it will print out the parse tree, otherwise if the line contains errors, it will print out suggestions how the string in question can be fixed, alongside the fragments which can be parsed.

<!-- Plugin description end -->

## Getting Started

To use this plugin, first clone the parent repository and initialize the submodule like so:

```bash
git clone https://github.com/breandan/tidyparse && \
cd tidyparse && \
git submodule update --init --recursive && \
./gradlew runIde
```

To launch IntelliJ IDEA with the plugin installed, run: `./gradlew runIde` from the parent directory.

## Usage

Create a new file, `if_lang.tidy`, containing the following context free grammar:

```
 S -> X
 X -> B | I | F | P
 P -> I O I
 F -> IF | BF
IF -> if B then I else I
BF -> if B then B else B
 O -> + | - | * | /
 I -> 1 | 2 | 3 | 4 | IF
 B -> true | false | B BO B | ( B ) | BF
BO -> and | or
---
```

The same file may also contain a test case, for example:

```
if true then if true then 1 else 2 else 3
```

If the string is valid, as shown above, Tidyparse should display the following output:

```
✅ Current line parses! Tree:

START [0..10]
├── if
├── true
├── then
├── I [3..8]
│   ├── if
│   ├── true
│   ├── then
│   ├── 1
│   ├── else
│   └── 2
├── else
└── 1
```

If Tidyparse is unable to parse the string, for example, as shown below: 

`if ( true or false ) then true else 1` 

It will instead display possible fixes sorted by edit distance and partial AST branches for all syntactically valid substrings:

```
❌ Current line invalid, possible fixes:

if ( true or false ) then <I> else 1
if ( true or false ) then true else <B>
if ( true or false ) then true else ! <B>
if ( true or false ) then true else <N> <B>
if ( true or false ) then true else ! ! <B>
if ( true or false ) then true else ! <N> <B>
if ( true or false ) then true else <N> <N> <B>
if ( true or false ) then true else <B> <BO> <B>
if ( true or false ) then true else ( ! <B> )
if ( true or false ) then true else <N> ! ! <B>
if ( true or false ) then true else <N> ! <N> <B>
if ( true or false ) then true else ! <B> <BO> <B>
if ( true or false ) then true else <N> <N> <N> <B>
if ( true or false ) then true else <N> <B> <BO> <B>

──────────────────────────────────────────────────
Parseable subtrees (5 leaves / 1 branch)

🌿            🌿            🌿
└── if [0]    └── then [6]  └── true [7]

🌿            🌿
└── else [8]  └── 1 [9]

🌿
└── B [1..5]
    ├── ( [1]
    ├── B [2..4]
    │   ├── true [2]
    │   ├── or [3]
    │   └── false [4]
    └── ) [5]
```

Tidyparse also accepts holes  in the test case. Holes can be `_`, or a nonterminal enclosed in angle brackets, such as:

```
if _ _ _ _ _ _ <BO> _ _ _ _ _
```

Providing such a test case will suggest candidates that are consistent with the provided CFG, ranked by length:

```
if <B> then <B> else <B> <BO> <B>
if <B> then <B> else ! <B> <BO> <B>
if <B> then <B> else <N> <B> <BO> <B>
if <B> then <B> else ( <B> <BO> <B> )
if <B> then <B> else <N> <B> <BO> ! <B>
if <B> then <B> else <N> <B> <BO> <N> <B>
if <B> then <B> else <B> <BO> <B> <BO> <B>
if <B> then <B> else ( <B> <BO> <N> <B> )
if <B> then <B> else <N> <B> <BO> ! <N> <B>
if <B> then <B> else ! <B> <BO> <B> <BO> <B>
if <B> then <B> else <N> <B> <BO> <N> <N> <B>
if <B> then <B> else <N> <B> <BO> <B> <BO> <B>
if ( <B> ) then <N> <B> <BO> <N> <B> else <B>
if <B> then <B> else ( <B> <BO> <B> ) <BO> <B>
if <N> <N> ! <B> then <B> <BO> <N> <B> else <B>
if <B> then <B> else ! <B> <BO> <B> <BO> <N> <B>
if <N> <B> then <B> else <B> <BO> <N> <N> <N> <B>
if <B> then <B> else <N> <B> <BO> <B> <BO> <N> <B>
if <B> then <B> else ( <B> <BO> <B> ) <BO> ! <B>
if <B> then <B> else ( <B> <BO> <B> ) <BO> <N> <B>
if <B> then <B> else <N> <B> <BO> <N> <B> <BO> <N> <B>
```

For diagnostic purposes, Tidyparse will also display the rewritten and normalized CFG, e.g.:

```
─────────────────────────────────────────────────────────────────────────────
Original form (12 nonterminals / 18 terminals / 38 productions)

O → +       I → IF   START → N                 
O → -       F → IF   START → IF                
O → *       F → BF   START → BF                
O → /      BO → or   START → BO                
X → I      BO → and      B → BF                
X → F       P → I O I    B → N B               
X → P   START → S        B → true              
N → !   START → X        B → false             
S → X   START → P        B → ( B )             
I → 1   START → F        B → B BO B            
I → 2   START → O       BF → if B then B else B
I → 3   START → I       IF → if B then I else I
I → 4   START → B                              
─────────────────────────────────────────────────────────────────────────────
Normal form (24 nonterminals / 24 terminals / 72 productions)

   F.( → (      else.B → F.else B               START → 2                    
   F.) → )      else.I → F.else I               START → 3                    
   F.ε → ε    B.else.B → B else.B               START → 4                    
     O → *    I.else.I → I else.I               START → !                    
     O → +           B → N B                    START → or                   
     O → -           B → <B>                    START → N B                  
     O → /           B → true                   START → and                  
     O → <O>         B → B ε+                   START → O ε+                 
     O → O ε+        B → false                  START → N ε+                 
     N → !           B → B BO.B                 START → B ε+                 
     N → <N>         B → F.( B.)                START → true                 
     N → N ε+        B → F.if B.then.B.else.B   START → I ε+                 
  F.if → if          I → 1                      START → BO ε+                
   O.I → O I         I → 2                      START → I O.I                
    BO → or          I → 3                      START → false                
    BO → and         I → 4                      START → B BO.B               
    BO → <BO>        I → <I>                    START → F.( B.)              
    BO → BO ε+       I → I ε+                   START → START F.ε            
    ε+ → ε           I → F.if B.then.I.else.I   START → F.if B.then.I.else.I 
    ε+ → ε+ ε+   START → *                      START → F.if B.then.B.else.B 
   B.) → B F.)   START → +              then.B.else.B → F.then B.else.B      
  BO.B → BO B    START → -              then.I.else.I → F.then I.else.I      
F.else → else    START → /            B.then.B.else.B → B then.B.else.B      
F.then → then    START → 1            B.then.I.else.I → B then.I.else.I   
```

For further examples, please refer to the [`examples`](/examples) subdirectory.

### Notes

* Nonterminals are surrounded by angle brackets, e.g., `<F>`. If the autocompletion dialog is invoked while the editor caret is above a nonterminal, Tidyparse will display a list of possible expansions.
* Currently, rendering is done on-the-fly but may not reflect the current state of the editor. To refresh the display, type an extra whitespace character.
* The grammar is sensitive to whitespace characters. Each nonterminal must be separated by at least one whitespace character.
* There is currently no lexical analysis. Each terminal in the grammar corresponds to a single token in text, separated by a whitespace. All names must be specified in the grammar.
* Tidyparse adds ε-productions and terminal literals for each nonterminal in the CFG. For further details about these transformations and the repair procedure, please refer to our [whitepaper](https://github.com/breandan/galoisenne/blob/master/latex/live/acmart.pdf).