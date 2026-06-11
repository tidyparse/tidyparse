@file:OptIn(ExperimentalEncodingApi::class)

import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT
import org.jetbrains.kotlin.gradle.targets.js.webpack.WebpackDevtool
import org.jetbrains.kotlin.gradle.targets.js.testing.KotlinJsTest
import org.gradle.api.tasks.testing.logging.TestExceptionFormat
import org.gradle.api.services.BuildService
import org.gradle.api.services.BuildServiceParameters
import org.jetbrains.letsPlot.*
import org.jetbrains.letsPlot.export.ggsave
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.label.ggtitle
import java.awt.Desktop
import java.io.ByteArrayOutputStream
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference
import kotlin.io.encoding.*

abstract class BrowserConsoleTailService : BuildService<BuildServiceParameters.None>, AutoCloseable {
  private val processes = CopyOnWriteArrayList<Process>()

  fun register(process: Process) {
    processes.add(process)
  }

  fun stop(process: Process?) {
    if (process == null) return
    process.destroy()
    processes.remove(process)
  }

  override fun close() {
    processes.forEach { it.destroy() }
    processes.clear()
  }
}

buildscript {
  repositories { mavenCentral() }
  dependencies {
    classpath("org.jetbrains.lets-plot:platf-awt-jvm:4.4.1")
    classpath("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:4.14.0")
  }
}

plugins {
  kotlin("multiplatform")
}

group = "ai.hypergraph"
version = "0.23.0"

kotlin {
  js {
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

      testTask { useKarma { useChromeHeadless() } }
    }
  }

  sourceSets {
    getByName("jsMain") {
      dependencies {
        implementation(project(":tidyparse-core"))
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2026.6.3")
      }
    }

    getByName("jsTest") {
      dependencies {
        implementation(kotlin("test-js"))
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.11.0")
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

fun jsString(s: String): String =
  buildString {
    append('"')
    for (c in s) when (c) {
      '\\' -> append("\\\\")
      '"' -> append("\\\"")
      '\n' -> append("\\n")
      '\r' -> append("\\r")
      '\t' -> append("\\t")
      '\b' -> append("\\b")
      '\u000C' -> append("\\f")
      else -> if (c.code < 0x20) append("\\u%04x".format(c.code)) else append(c)
    }
    append('"')
  }

tasks {
  val browserConsoleTailService = gradle.sharedServices.registerIfAbsent(
    "browserConsoleTailService",
    BrowserConsoleTailService::class
  ) {}

  withType<KotlinJsTest>().configureEach {
    val testTaskPath = path
    val browserConsoleTailProcess = AtomicReference<Process?>()
    usesService(browserConsoleTailService)

    testLogging {
      showStandardStreams = true
      showExceptions = true
      showCauses = true
      showStackTraces = true
      exceptionFormat = TestExceptionFormat.FULL
      events("passed", "skipped", "failed", "standardOut", "standardError")
    }

    doFirst {
      val browserConsoleLog = rootProject.layout.buildDirectory.file("ci-logs/browser-console.log").get().asFile
      browserConsoleLog.parentFile.mkdirs()
      browserConsoleLog.writeText("")

      val tailProcess = ProcessBuilder("tail", "-n", "+1", "-f", browserConsoleLog.absolutePath)
        .redirectErrorStream(true)
        .start()

      browserConsoleTailService.get().stop(browserConsoleTailProcess.getAndSet(tailProcess))
      browserConsoleTailService.get().register(tailProcess)
      Thread {
        tailProcess.inputStream.bufferedReader().useLines { lines ->
          lines.forEach { println(it) }
        }
      }.apply {
        name = "browser-console-tail-$testTaskPath"
        isDaemon = true
        start()
      }
    }

    fun stopBrowserConsoleTail() {
      browserConsoleTailService.get().stop(browserConsoleTailProcess.getAndSet(null))
    }

    doLast { stopBrowserConsoleTail() }
  }

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
    """.trimIndent()

      val outHtml = File(System.getProperty("user.home"), "tidyparse.html")
      outHtml.writeText(html)

      println("✓ Self-contained headless bundle written to ${outHtml.absolutePath}")
    }
  }

  register("bundleJCEF") {
    group = "build"
    description = "Bundles a single self-contained JCEF HTML template for the IntelliJ plugin"

    dependsOn(project(":tidyparse-web").tasks.named("jsBrowserProductionWebpack"))

    val webProject = project(":tidyparse-web")

    val bundleDir = webProject.layout.buildDirectory
      .dir("kotlin-webpack/js/productionExecutable/")

    val jsFile = bundleDir.map { it.file("tidyparse-web.js").asFile }
    val mapFile = bundleDir.map { it.file("tidyparse-web.js.map").asFile }

    val ngramFile = webProject.layout.projectDirectory
      .file("src/jsMain/resources/python_4grams.txt")
      .asFile
    val wdfaFile = webProject.layout.projectDirectory
      .file("src/jsMain/resources/wdfa.bin")
      .asFile

    val outHtml = rootProject.layout.projectDirectory
      .file("tidyparse-intellij/src/main/resources/jcef/tidyparse-jcef.html")

    inputs.files(jsFile, mapFile, ngramFile, wdfaFile)
    outputs.file(outHtml)

    doLast {
      val jsCode = jsFile.get().readText()
      val mapJson = mapFile.get().takeIf { it.exists() }?.readText()

      val inlinedJs = if (mapJson != null) {
        val mapB64 = Base64.encode(mapJson.toByteArray())
        jsCode.replace(Regex("""(?m)^//# sourceMappingURL=.*$"""), "")
          .trimEnd('\n', '\r') + "\n//# sourceMappingURL=data:application/json;base64,$mapB64"
      } else { jsCode.replace(Regex("""(?m)^//# sourceMappingURL=.*$"""), "") }

      val rawNgrams = ngramFile.readText()
      val rawWdfaB64 = Base64.encode(wdfaFile.readBytes())

      val html = """
  <!doctype html>
  <html>
  <head>
    <meta charset="utf-8">
    <title>TidyParse JCEF Runtime</title>
  </head>
  <body>
    <pre id="tidyparse-jcef-log" style="display:none"></pre>

<script>
window.REPAIR_MODE = "jcef";
window.raw_ngrams = ${jsString(rawNgrams)};
window.raw_wdfa_b64 = ${jsString(rawWdfaB64)};

function __tidyparseJcefSend(payload) {
  __JCEF_EVENT_CALLBACK__;
}

window.__tidyparseJcefSend = __tidyparseJcefSend;
</script>

    <script type="module">
    $inlinedJs
    </script>
  </body>
  </html>
""".trimIndent()

      outHtml.asFile.apply { parentFile.mkdirs(); writeText(html) }
      println("✓ JCEF bundle written to ${outHtml.asFile.absolutePath}")
      println("  Placeholder to replace at runtime: __JCEF_EVENT_CALLBACK__")
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

      ProcessBuilder("open", buildDir.absolutePath).start()
      ProcessBuilder("open", resourcesDir.absolutePath).start()
      ProcessBuilder("open", "https://github.com/tidyparse/tidyparse.github.io/upload/main").start()
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
