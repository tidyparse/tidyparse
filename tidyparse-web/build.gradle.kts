@file:OptIn(ExperimentalEncodingApi::class)

import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT
import org.jetbrains.kotlin.gradle.targets.js.webpack.WebpackDevtool
import kotlin.io.encoding.*

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

      testTask {
        useKarma {
          useChromeHeadless()
        }
      }
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
      }
    }
  }
}

tasks.register("bundleHeadless") {
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

// To deploy the browser application, run:
//  ./gradlew :tidyparse-web:jsBrowserProductionWebpack
// Then copy the contents of tidyparse-web/build/kotlin-webpack/js/productionExecutable/tidyparse-web
// (and optionally, if static resources have been modified, tidyparse-web/src/jsMain/resources) to:
//  https://github.com/tidyparse/tidyparse.github.io/upload/main
// Wait a few minutes for CI to finish, then check the website:
//  https://tidyparse.github.io

// To run on localhost and open a browser, run:
//  ./gradlew :tidyparse-web:jsBrowserDevelopmentRun --continuous