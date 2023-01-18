rootProject.name = "Tidyparse"

pluginManagement {
  val kotlinVersion: String by settings
  plugins {
    kotlin("js") version kotlinVersion
    kotlin("jvm") version kotlinVersion
    kotlin("multiplatform") version kotlinVersion
  }
}

includeBuild("galoisenne")
include("tidyparse-core")
include("tidyparse-intellij")
include("tidyparse-web")