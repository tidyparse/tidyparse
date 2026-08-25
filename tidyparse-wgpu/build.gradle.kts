plugins {
  kotlin("multiplatform")
}

group = "ai.hypergraph"
version = "0.23.0"

kotlin {
  js {
    browser {
      testTask { useKarma { useChromeHeadless() } }
    }
  }

  sourceSets {
    getByName("jsMain") {
      dependencies {
        implementation(kotlin("stdlib"))
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
        api("org.jetbrains.kotlin-wrappers:kotlin-web:2026.8.0")
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
      }
    }

    getByName("jsTest") {
      dependencies {
        implementation(kotlin("test-js"))
      }
    }
  }
}
