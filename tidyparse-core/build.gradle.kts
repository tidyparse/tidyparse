fun property(key: String) = project.findProperty(key).toString()

plugins {
  kotlin("plugin.serialization")
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
//    all {
//      languageSettings.apply {
//        languageVersion = "2.0"
//      }
//    }
    commonMain {
      dependencies {
        api(kotlin("stdlib:2.1.0"))
        api("ai.hypergraph:kaliningraph") {
          exclude(group = "org.jetbrains.kotlin")
          exclude(group = "guru.nidi")
          exclude(group = "org.graalvm.js")
          exclude(group = "org.jetbrains.kotlinx")
          exclude(group = "org.jetbrains.lets-plot")
          exclude(group = "org.apache.datasketches")
          exclude(group = "ca.umontreal.iro.simul")
          exclude(group = "org.sosy-lab")
          exclude(group = "org.logicng")
        }

        api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
      }
    }
    commonTest {
      dependencies {
        implementation(kotlin("test"))
      }
    }
  }
}