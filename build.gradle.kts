// https://github.com/JVAAS/kotlin-multiplatform-multi-module-setup

plugins {
  idea
  // https://youtrack.jetbrains.com/issue/KT-52172/Multiplatform-Support-composite-builds
  kotlin("multiplatform") version "1.8.20-Beta" apply false
  id("com.github.ben-manes.versions") version "0.45.0"
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