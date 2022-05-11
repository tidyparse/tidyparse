# Tidyparse

<!-- Plugin description -->
**IntelliJ Platform Plugin Template** is a repository that provides a pure template to make it easier to create a new plugin project (check the [Creating a repository from a template][gh:template] article).

The main goal of this template is to speed up the setup phase of plugin development for both new and experienced developers by preconfiguring the project scaffold and CI, linking to the proper documentation pages, and keeping everything organized.

[gh:template]: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template
<!-- Plugin description end -->

Tidyparse expects two files in the same directory -- one ending in `*.tidy` which contains the string to parse (with optional holes) and one ending in `*.cfg` which contains the grammar. If you provide a string containing holes, it will provide some suggestions inside a tool window on the right hand side (can be opened by pressing `Shift` twice in rapid succession and searching for `Tidyparse`. If the string contains no holes, it will print out the parse tree in Chomsky normal form.

To launch IntelliJ IDEA with the plugin installed, run: `./gradlew runIde` from the parent directory.

<img width="1395" alt="Screen Shot 2022-05-10 at 8 16 33 PM" src="https://user-images.githubusercontent.com/175716/167747603-e2bed035-0232-4da7-95fd-f8909fc0eb9a.png">

<img width="1398" alt="Screen Shot 2022-05-10 at 8 34 20 PM" src="https://user-images.githubusercontent.com/175716/167747605-9226f7de-5d92-43b7-bb3b-5300b5320b56.png">
