import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT
import org.jetbrains.kotlin.gradle.targets.js.webpack.WebpackDevtool

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
    }
  }

  sourceSets {
    val jsMain by getting {
      dependencies {
        implementation(project(":tidyparse-core"))
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2025.4.16")
      }
    }
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