import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
import org.gradle.api.tasks.testing.logging.TestLogEvent.*

fun properties(key: String) = project.findProperty(key).toString()

plugins {
  kotlin("multiplatform") version "1.8.0"
  id("com.github.ben-manes.versions") version "0.44.0"
}

group = properties("pluginGroup")
version = properties("pluginVersion")

kotlin {
  jvm {
    compilations.all {
      kotlinOptions.jvmTarget = "17"
    }
    withJava()
    testRuns["test"].executionTask.configure {
      useJUnitPlatform()
    }
  }

  js(IR) {
    browser()
    binaries.executable()
  }

  jvmToolchain {
    run {
      languageVersion.set(JavaLanguageVersion.of(17))
    }
  }

  sourceSets {
    val commonMain by getting {
      dependencies {
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
      }
    }
    val commonTest by getting {
      dependencies {
        implementation(kotlin("test"))
      }
    }
    val jvmMain by getting {
      dependencies {
//        implementation("io.github.java-diff-utils:java-diff-utils:4.12")
      }
    }
//    val jvmTest by getting
//    val jsMain by getting
//    val jsTest by getting
  }
}