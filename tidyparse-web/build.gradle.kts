import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.*

plugins {
  kotlin("multiplatform")
}

fun properties(key: String) = project.findProperty(key).toString()

group = properties("pluginGroup")
version = properties("pluginVersion")

kotlin {
  js(IR) {
    binaries.executable()

    browser {
      runTask {
        mainOutputFileName = "tidyparse-web.js"
      }

      webpackTask {
        mode = PRODUCTION
        mainOutputFileName = "tidyparse-web.js"
        devtool = "source-map" // For debugging; remove for production
      }
    }
  }

  sourceSets {
    val jsMain by getting {
      dependencies {
        implementation(project(":tidyparse-core"))
      }
    }
  }
}

// To deploy the browser application, run:
//  ./gradlew :tidyparse-web:jsBrowserProductionWebpack
// Then copy the contents of tidyparse-web/build/kotlin-webpack/js/tidyparse-web to the server:
//  https://github.com/tidyparse/tidyparse.github.io/upload/main
// Wait a few minutes for CI to finish, then check the website:
//  https://tidyparse.github.io

// To run on localhost and open a browser, run:
//  ./gradlew :tidyparse-web:jsBrowserDevelopmentRun --continuous