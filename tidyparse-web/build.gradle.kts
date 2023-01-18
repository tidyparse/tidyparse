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
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:1.6.4")
}

kotlin {
  js(IR) {
    browser()
    binaries.executable()
  }
}