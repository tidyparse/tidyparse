import com.strumenta.antlrkotlin.gradle.AntlrKotlinTask
import org.jetbrains.kotlin.gradle.dsl.KotlinCompile

fun property(key: String) = project.findProperty(key).toString()

plugins {
  kotlin("plugin.serialization")
  kotlin("multiplatform")
  id("com.strumenta.antlr-kotlin") version "1.0.2"
}

group = property("pluginGroup")!!
version = property("pluginVersion")!!

kotlin {
  jvm {
    compilations.all { kotlinOptions.jvmTarget = property("javaVersion") as String }
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

        api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
        api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")

        api("com.strumenta:antlr-kotlin-runtime:1.0.2")
      }
      kotlin { srcDir(layout.buildDirectory.dir("generatedAntlr")) }
    }
    commonTest {
      dependencies {
        implementation(kotlin("test"))
      }
    }
  }
}

// https://github.com/Strumenta/antlr-kotlin/tree/master?tab=readme-ov-file#gradle-setup
val generateKotlinGrammarSource = tasks.register<AntlrKotlinTask>("generateKotlinGrammarSource") {
  dependsOn("cleanGenerateKotlinGrammarSource")

  // ANTLR .g4 files are under {example-project}/antlr
  // Only include *.g4 files. This allows tools (e.g., IDE plugins)
  // to generate temporary files inside the base path
  source = fileTree(layout.projectDirectory.dir("antlr")) {
    include("**/*.g4")
  }

  // We want the generated source files to have this package name
  val pkgName = "com.strumenta.antlrkotlin.parsers.generated"
  packageName = pkgName

  // We want visitors alongside listeners.
  // The Kotlin target language is implicit, as is the file encoding (UTF-8)
  arguments = listOf("-visitor")

  // Generated files are outputted inside build/generatedAntlr/{package-name}
  val outDir = "generatedAntlr/${pkgName.replace(".", "/")}"
  outputDirectory = layout.buildDirectory.dir(outDir).get().asFile
}

tasks.withType<KotlinCompile<*>> { dependsOn(generateKotlinGrammarSource) }