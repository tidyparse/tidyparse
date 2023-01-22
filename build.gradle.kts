// https://github.com/JVAAS/kotlin-multiplatform-multi-module-setup

plugins {
  idea
  // https://youtrack.jetbrains.com/issue/KT-52172/Multiplatform-Support-composite-builds
  kotlin("multiplatform") version "1.8.20-dev-6044" apply false
//  id("com.github.ben-manes.versions") version "0.44.0"
}

allprojects {
  repositories {
    maven(url = "https://maven.pkg.jetbrains.space/kotlin/p/kotlin/dev")
    mavenCentral()
    maven("https://maven.pkg.jetbrains.space/public/p/kotlinx-html/maven")
  }
}

idea {
  module.isDownloadJavadoc = true
  module.isDownloadSources = true
}