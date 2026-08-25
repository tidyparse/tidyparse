import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpack
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.PRODUCTION

plugins {
  kotlin("multiplatform")
}

group = "ai.hypergraph"
version = "0.23.0"

val repairWorkerBundleName = "tidyparse-python-repair.js"
val workerWebpackConfigDir = layout.buildDirectory.dir("generated/worker-webpack-config")
val workerWebpackConfig = """
  const webpack = require("webpack");
  config.target = "webworker";
  config.plugins.push(new webpack.optimize.LimitChunkCountPlugin({ maxChunks: 1 }));
""".trimIndent() + "\n"

val prepareWorkerWebpackConfig = tasks.register("prepareWorkerWebpackConfig") {
  val configFile = workerWebpackConfigDir.map { it.file("single-worker.js") }
  inputs.property("contents", workerWebpackConfig)
  outputs.file(configFile)

  doLast {
    configFile.get().asFile.apply {
      parentFile.mkdirs()
      writeText(workerWebpackConfig)
    }
  }
}

kotlin {
  js {
    binaries.executable()

    browser {
      commonWebpackConfig {
        configDirectory = workerWebpackConfigDir.get().asFile
      }

      runTask {
        mainOutputFileName = repairWorkerBundleName
      }

      webpackTask {
        mode = PRODUCTION
        mainOutputFileName = repairWorkerBundleName
        devtool = "source-map"
      }
    }
  }

  sourceSets {
    getByName("jsMain") {
      dependencies {
        implementation(kotlin("stdlib"))
        implementation(project(":tidyparse-core"))
        implementation(project(":tidyparse-wgpu"))
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
      }
    }
  }
}

tasks.withType<KotlinWebpack>().configureEach {
  dependsOn(prepareWorkerWebpackConfig)
}

tasks.named("jsBrowserProductionWebpack") {
  outputs.files(
    layout.buildDirectory.file("kotlin-webpack/js/productionExecutable/$repairWorkerBundleName"),
    layout.buildDirectory.file("kotlin-webpack/js/productionExecutable/$repairWorkerBundleName.map")
  )
}
