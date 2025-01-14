import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT

plugins {
  kotlin("js")
}

fun properties(key: String) = project.findProperty(key).toString()
group = properties("pluginGroup")
version = properties("pluginVersion")

kotlin {
  js(IR) {
    dependencies {
      implementation(project(":tidyparse-core"))
    }

    binaries.executable()
    browser {
      runTask {
        mainOutputFileName = "tidyparse-web.js"
      }

      webpackTask {
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-web.js"
        devtool = "source-map" // Remove later for production
      }

      distribution {
        distributionName = "tidyparse-web"
      }
    }
  }
}

// To deploy the browser application, run:
//  ./gradlew browserProductionWebpack
// Then copy the contents of tidyparse-web/build/kotlin-webpack/js/tidyparse-web to the server:
//  https://github.com/tidyparse/tidyparse.github.io/upload/main
// Wait a few minutes for CI to finish, then check the website:
//  https://tidyparse.github.io

// To run on localhost and open a browser, run:
//  ./gradlew :tidyparse-web:browserDevelopmentRun --continuous