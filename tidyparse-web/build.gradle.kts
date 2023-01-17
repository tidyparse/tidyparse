plugins {
  kotlin("js") version "1.8.0"
  id("com.github.ben-manes.versions") version "0.44.0"
}

group = "me.breandan"
version = "0.1-SNAPSHOT"

dependencies {
  testImplementation(kotlin("test"))
  implementation(project(":tidyparse-core"))
  implementation("ai.hypergraph:kaliningraph") {
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib")
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib-common")
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-reflect")
    exclude(group = "guru.nidi", module = "graphviz-kotlin")
    exclude(group = "org.graalvm.js", module = "js")
    exclude(group = "org.jetbrains.kotlinx", module = "kotlinx-coroutines-core")
    exclude(group = "org.jetbrains.kotlinx", module = "kotlinx-html-jvm")
//          exclude(group = "org.jetbrains.kotlinx", module = "multik-core")
//          exclude(group = "org.jetbrains.kotlinx", module = "multik-default")
    exclude(group = "org.jetbrains.lets-plot", module = "lets-plot-kotlin-jvm")
    exclude(group = "org.apache.datasketches", module = "datasketches")
    exclude(group = "org.apache.datasketches", module = "datasketches-java")
    exclude(group = "ca.umontreal.iro.simul", module = "ssj")
    exclude(group = "org.sosy-lab", module = "common")
    exclude(group = "org.sosy-lab", module = "java-smt")
    exclude(group = "org.sosy-lab", module = "javasmt-solver-mathsat5")
  }
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