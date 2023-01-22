rootProject.name = "Tidyparse"

pluginManagement {
  repositories {
    maven(url = "https://maven.pkg.jetbrains.space/kotlin/p/kotlin/dev")
    gradlePluginPortal()
    mavenCentral()
  }
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