plugins {
  val kotVer = "2.0.0-Beta2"
  idea
  kotlin("multiplatform") version kotVer apply false
  id("com.github.ben-manes.versions") version "0.51.0"
  kotlin("plugin.serialization") version kotVer apply false
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