plugins {
  idea
  kotlin("multiplatform") version "1.9.20" apply false
  id("com.github.ben-manes.versions") version "0.49.0"
  kotlin("plugin.serialization") version "1.9.20" apply false
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