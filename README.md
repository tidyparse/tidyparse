# Tidyparse

<!-- Plugin description -->
The main goal of this project is to speed up the process of learning a new language by suggesting ways to fix source code.

Tidyparse expects two files in the same directory -- one ending in `*.tidy` which contains the string to parse (with optional holes) and one ending in `*.cfg` which contains the grammar. If you provide a string containing holes, it will provide some suggestions inside a tool window on the right hand side (can be opened by pressing `Shift` twice in rapid succession and searching for `Tidyparse`. If the string contains no holes, it will print out the parse tree in Chomsky normal form.
<!-- Plugin description end -->

## Getting Started

To use this plugin, first clone the parent repository and initialize the submodule like so:

```bash
git clone https://github.com/breandan/tidyparse && cd tidyparse && git submodule update --init --recursive && ./gradlew runIde
```

To launch IntelliJ IDEA with the plugin installed, run: `./gradlew runIde` from the parent directory.

<img width="1395" alt="Screen Shot 2022-05-10 at 8 16 33 PM" src="https://user-images.githubusercontent.com/175716/167747603-e2bed035-0232-4da7-95fd-f8909fc0eb9a.png">

<img width="1398" alt="Screen Shot 2022-05-10 at 8 34 20 PM" src="https://user-images.githubusercontent.com/175716/167747605-9226f7de-5d92-43b7-bb3b-5300b5320b56.png">
