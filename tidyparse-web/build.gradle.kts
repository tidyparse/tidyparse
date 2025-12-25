@file:OptIn(ExperimentalEncodingApi::class)

import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT
import org.jetbrains.kotlin.gradle.targets.js.webpack.WebpackDevtool
import org.jetbrains.letsPlot.*
import org.jetbrains.letsPlot.export.ggsave
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.label.ggtitle
import java.awt.Desktop
import java.io.ByteArrayOutputStream
import kotlin.io.encoding.*

buildscript {
  repositories { mavenCentral() }
  dependencies {
    classpath("org.jetbrains.lets-plot:platf-awt-jvm:4.4.1")
    classpath("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:4.11.0")
  }
}

plugins {
  kotlin("multiplatform")
}

group = "ai.hypergraph"
version = "0.23.0"

kotlin {
  js(IR) {
    binaries.executable()

    browser {
      runTask {
        mainOutputFileName = "tidyparse-web.js"
        sourceMaps = true
        devtool = WebpackDevtool.SOURCE_MAP
      }

      webpackTask {
        // We need this to work on Chrome when deployed due to the PLATFORM_CALLER_STACKTRACE_DEPTH hack
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-web.js"
        devtool = "source-map" // For debugging; remove for production
      }

      testTask { useKarma { } }
    }
  }

  sourceSets {
    val jsMain by getting {
      dependencies {
        implementation(project(":tidyparse-core"))
        // Do not update until 2025.8.0 is stable
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2025.6.4")
      }
    }

    val jsTest by getting {
      dependencies {
        implementation(kotlin("test-js"))
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
      }
    }
  }
}

fun saveStats(stat: String, name: String) =
  if (stat.isEmpty()) throw IllegalStateException("Total time not found in output")
  else file("$name.txt").also { if (!it.exists()) it.createNewFile() }
    .apply { println("$name:\t$stat"); appendText("${stat.trim().toLong()}\n") }

fun plotGrid(name: String, x_lbl: String = "x", y_lbl: String = "y"): Plot {
  val timings = file("$name.txt").readLines().mapNotNull { it.toDoubleOrNull() }
  val data = mapOf(x_lbl to (1..timings.size).toList(), y_lbl to timings)
  return letsPlot(data) { x = x_lbl; y = y_lbl } + geomLine() + ggtitle(name)
}

fun plotGrids(vararg names: String) {
  ggsave(gggrid(names.map { plotGrid(it) }, ncol = 3), "grid.svg", path = projectDir.absolutePath)
  Desktop.getDesktop().browse(file("grid.svg").toURI())
}

tasks {
  register<Exec>("replotMetrics") {
    commandLine("../gradlew", "clean", "jsBrowserTest", "--info") // --info flag is crucial to read output

    standardOutput = ByteArrayOutputStream()

    doLast {
      val output = standardOutput.toString()
      if ("Total CPU latency" !in output) throw Exception("No output found: $output")
      val totalGPUTime = output.substringAfter("Total GPU latency:").substringBefore("\n")
      val totalGPURepairs = output.substringAfter("Total GPU repairs:").substringBefore("\n")
      val totalGPUMatches = output.substringAfter("Total GPU matches:").substringBefore("\n")
      saveStats(totalGPUTime, "repair_gpu_timings")
      saveStats(totalGPURepairs, "total_gpu_repairs")
      saveStats(totalGPUMatches, "total_gpu_matches")

      val totalCPUTime = output.substringAfter("Total CPU latency:").substringBefore("\n")
      val totalCPURepairs = output.substringAfter("Total CPU repairs:").substringBefore("\n")
      val totalCPUMatches = output.substringAfter("Total CPU matches:").substringBefore("\n")
      saveStats(totalCPUTime, "repair_cpu_timings")
      saveStats(totalCPURepairs, "total_cpu_repairs")
      saveStats(totalCPUMatches, "total_cpu_matches")

      plotGrids("repair_gpu_timings", "total_gpu_repairs", "total_gpu_matches", "repair_cpu_timings", "total_cpu_repairs", "total_cpu_matches")
    }
  }

  register("bundleHeadless") {
    dependsOn(project(":tidyparse-web").tasks.named("jsBrowserProductionWebpack"))

    val bundleDir = project(":tidyparse-web").layout.buildDirectory
      .dir("kotlin-webpack/js/productionExecutable/")

    val jsFile = bundleDir.map { it.file("tidyparse-web.js").asFile }
    val mapFile = bundleDir.map { it.file("tidyparse-web.js.map").asFile }

    inputs.files(jsFile, mapFile)
    outputs.file(File(System.getProperty("user.home"), "gpu.html"))

    doLast {
      val jsCode = jsFile.get().readText()
      val mapJson = mapFile.get().readText()

      val mapB64 = Base64.encode(mapJson.toByteArray())
      val inlinedJs = jsCode.replace(Regex("""(?m)^//# sourceMappingURL=.*$"""), "")
        .trimEnd('\n', '\r') + "\n//# sourceMappingURL=data:application/json;base64,$mapB64"
      val ngramPath = "src/jsMain/resources/python_4grams.txt"
      val rawNgrams = project(":tidyparse-web").layout.projectDirectory.file(ngramPath).asFile.readText()

      val html = """
        <!doctype html>
        <meta charset="utf-8">
        <title>TidyParse Headless</title>
        <script type="text/javascript">var REPAIR_MODE = "headless"</script>
        <script type="text/javascript">var raw_ngrams = `$rawNgrams`;</script>
        <script type="module">
        $inlinedJs
        </script>
        <script src="https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js"></script>
    """.trimIndent()

      val outHtml = File(System.getProperty("user.home"), "tidyparse.html")
      outHtml.writeText(html)

      println("✓ Self-contained headless bundle written to ${outHtml.absolutePath}")
    }
  }

  register("deployWeb") {
    group = "deployment"
    description = "Builds app, opens Finder to the build and resources directories, and launches the browser for upload"
    dependsOn(":tidyparse-web:jsBrowserProductionWebpack")
    doLast {
      val webProject = project(":tidyparse-web")
      val buildDir = webProject.layout.buildDirectory.asFile.get().resolve("kotlin-webpack/js/productionExecutable/")
      val resourcesDir = webProject.file("src/jsMain/resources")

      exec { commandLine("open", buildDir.absolutePath) }
      exec { commandLine("open", resourcesDir.absolutePath) }
      exec { commandLine("open", "https://github.com/tidyparse/tidyparse.github.io/upload/main") }
    }
  }
}

// To deploy the browser application, run:
// ./gradlew deployWeb
// Then copy the contents of tidyparse-web/build/kotlin-webpack/js/productionExecutable/tidyparse-web
// (and optionally, if static resources have been modified, tidyparse-web/src/jsMain/resources) to:
//  https://github.com/tidyparse/tidyparse.github.io/upload/main
// Wait a few minutes for CI to finish, then check the website:
//  https://tidyparse.github.io

// To run on localhost and open a browser, run:
//  ./gradlew :tidyparse-web:jsBrowserDevelopmentRun --continuous

// To test on localhost, run:
// ./gradlew jsTest