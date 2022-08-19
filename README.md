<p align="center"><a href="https://plugins.jetbrains.com/plugin/19570-tidyparse"><img src="src/main/resources/META-INF/pluginIcon.svg" alt="Tidyparse Logo" height="200"></a></p>

<p align="center">  <font size="+10">Tidyparse</font>
</p>
<p align="center">
 	<a href="https://plugins.jetbrains.com/plugin/19570-tidyparse" title="Tidyparse"><img src="https://img.shields.io/jetbrains/plugin/v/19570-tidyparse.svg"></a>
 	<a href="LICENSE" title="License"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<!-- Plugin description -->
The main goal of this project is to speed up the process of learning a new language by suggesting ways to fix source code.

Tidyparse expects a file ending in `*.tidy` which contains, first, the grammar, followed by three consecutive dashes (`---`), followed by the string to parse (with optional holes). If you provide a string containing holes, it will provide some suggestions inside a tool window on the right hand side. If the string contains no holes, it will print out the parse tree.
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

The file `ocaml.tidy` can contain this grammar:

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
```
The file `ocaml.tidy` can also contain this test case:

```
if true then if true then 1 else 2 else 3
```

This should produce the following output:

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

If the string does not parse, for example, as shown below: 

`if ( true or false ) then true else 1` 

Tidyparse will display possible fixes and branches for all syntactically valid substrings:

```
❌ Current line invalid, possible fixes:

if ( true or false ) then <I> else 1
if ( true or false ) then true else <B>
if ( <B> or false ) then <I> else 1
if ( <B> or false ) then true else <B>
if ( true <BO> false ) then <I> else 1
if ( true <BO> false ) then true else <B>
if ( true or <B> ) then <I> else 1
if ( true or <B> ) then true else <B>
if ( true or false ) then <I> else <I>
if ( true or false ) then <B> else <B>

──────────────────────────────────────────────────
Partial AST branches:

else  if  then  true  1
B [1..5]
├── (
├── B [2..4]
│   ├── true
│   ├── or
│   └── false
└── )
```

Tidyparse also accepts holes  in the test case. Holes can be `_`, or a nonterminal enclosed in angle brackets, such as:

```
if _ _ _ _ _ _ <BO> _ _ _ _ _
```

Providing such a test case will suggest candidates that are consistent with the provided CFG:

```
if <B> then if <B> then <B> <BO> <B> else <B> else <B>
if <B> then <I> else if <B> <BO> <B> then <I> else <I>
if <B> then ( <B> <BO> <B> <BO> <B> ) else <B>
if <B> then ( ( <B> ) <BO> <B> ) else <B>
if ( <B> ) then ( <B> <BO> <B> ) else <B>
if if <B> then <B> else <B> <BO> <B> then <I> else <I>
if if <B> then <B> else <B> <BO> <B> then <B> else <B>
if <B> then <B> else if <B> <BO> <B> then <B> else <B>
if <B> then <B> else ( <B> <BO> <B> <BO> <B> )
if <B> then <B> else ( <B> <BO> <B> )
if <B> <BO> <B> then ( <B> <BO> <B> ) else <B>
if <B> then <B> <BO> ( <B> <BO> <B> ) else <B>
if <B> then <B> else ( <B> <BO> ( <B> ) )
if <B> then <B> else ( <B> <BO> <B> ) <BO> <B>
if <B> then <B> <BO> <B> else <B>
if <B> then ( <B> <BO> <B> ) else <B>
if ( ( <B> ) ) <BO> <B> then <B> else <B>
if ( ( <B> ) ) <BO> <B> then <I> else <I>
if <B> then <B> else <B> <BO> ( <B> <BO> <B> )
```

### Notes

* Currently, rendering is done on-the-fly but may not reflect the current state of the editor. To refresh the display, type an extra whitespace character.
* The grammar is sensitive to whitespace characters. Each nonterminal must be separated by at least one whitespace character.
* There is currently no lexical analysis. Each terminal in the grammar corresponds to a single token in text. All names must be specified in the grammar.