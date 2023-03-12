plugins {
  kotlin("js")
}

fun properties(key: String) = project.findProperty(key).toString()
group = properties("pluginGroup")
version = properties("pluginVersion")

dependencies {
  testImplementation(kotlin("test"))
  api(project(":tidyparse-core"))
  implementation("org.jetbrains.kotlinx:kotlinx-html:0.8.1")
//  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core")
  val coroutinesVersion = "1.7.0-Beta"
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:$coroutinesVersion")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:$coroutinesVersion")
}

kotlin {
  js(IR) {
    browser()
    binaries.executable()
  }
//  sourceSets {
//    all {
//      languageSettings.apply {
//        languageVersion = "2.0"
//      }
//    }
//  }
}