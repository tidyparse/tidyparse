rootProject.name = "Tidyparse"

pluginManagement {
  val kotlinVersion: String by settings
  plugins {
    kotlin("js") version kotlinVersion
    kotlin("jvm") version kotlinVersion
    kotlin("multiplatform") version kotlinVersion
  }
}

// Watch out for https://youtrack.jetbrains.com/issue/KT-56536/Multiplatform-Composite-build-fails-on-included-build-with-rootProject.name-buildIdentifier.name
includeBuild("galoisenne") {
  dependencySubstitution {
    substitute(module("ai.hypergraph:kaliningraph")).using(project(":"))
  }
}
//includeBuild("galoisenne")
include("tidyparse-core")
include("tidyparse-intellij")
include("tidyparse-web")