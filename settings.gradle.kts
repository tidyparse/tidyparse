rootProject.name = "Tidyparse"

// Watch out for https://youtrack.jetbrains.com/issue/KT-56536/Multiplatform-Composite-build-fails-on-included-build-with-rootProject.name-buildIdentifier.name
includeBuild("galoisenne") {
  dependencySubstitution {
    substitute(module("ai.hypergraph:kaliningraph")).using(project(":"))
  }
}

include("tidyparse-core")
include("tidyparse-intellij")
include("tidyparse-web")