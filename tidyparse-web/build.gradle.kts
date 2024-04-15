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

// To deploy, run:
//  ./gradlew browserProductionWebpack

// To run in the browser, run:
//  ./gradlew :tidyparse-web:browserDevelopmentRun --continuous