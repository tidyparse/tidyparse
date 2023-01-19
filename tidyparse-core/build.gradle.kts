fun property(key: String) = project.findProperty(key).toString()

plugins {
  kotlin("multiplatform")
}

group = property("pluginGroup")
version = property("pluginVersion")

kotlin {
  jvm {
    compilations.all { kotlinOptions.jvmTarget = property("javaVersion") }
    withJava()
    testRuns["test"].executionTask.configure {
      useJUnitPlatform()
    }
  }

  js(IR) {
    browser()
    binaries.executable()
  }

  sourceSets {
    commonMain {
      dependencies {
        api("ai.hypergraph:kaliningraph:0.2.1") {
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
      }
    }
    commonTest {
      dependencies {
        implementation(kotlin("test"))
      }
    }
    val jvmMain by getting {
      dependencies {
        api("io.github.java-diff-utils:java-diff-utils:4.12")
      }
    }
  }
}