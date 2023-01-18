// https://github.com/JVAAS/kotlin-multiplatform-multi-module-setup

plugins {
  idea
  kotlin("multiplatform") apply false
  id("com.github.ben-manes.versions") version "0.44.0"
}

allprojects {
  repositories {
    mavenCentral()
    maven("https://maven.pkg.jetbrains.space/public/p/kotlinx-html/maven")
  }
}

idea {
  module.isDownloadJavadoc = true
  module.isDownloadSources = true
}