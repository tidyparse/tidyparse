import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT


plugins {
  kotlin("js")
}

fun properties(key: String) = project.findProperty(key).toString()
group = properties("pluginGroup")
version = properties("pluginVersion")

dependencies {
  api(project(":tidyparse-core"))
  implementation("org.jetbrains.kotlinx:kotlinx-html:0.10.1")
//  implementation("dev.andrewbailey.difference:difference:1.0.0")
//  implementation("dev.andrewbailey.difference:difference-js:1.0.0")
}

kotlin {
  js(IR) {
    binaries.executable()
    browser {
      runTask {
        mainOutputFileName = "tidyparse-web-frontend.js"
      }

      webpackTask {
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-web-frontend.js"
        devtool = "source-map" // Remove later for production
      }

      distribution {
        distributionName = "tidyparse-web-frontend"
      }
    }
  }

  sourceSets["main"].dependencies {
//    implementation("dev.andrewbailey.difference:difference:1.0.0")
//    implementation("dev.andrewbailey.difference:difference-js:1.0.0")
  }

//  sourceSets {
//    all {
//      languageSettings.apply {
//        languageVersion = "2.0"
//      }
//    }
//  }
}

// To deploy, run:
//  ./gradlew browserProductionWebpack

// To run in the browser, run:
//  ./gradlew :tidyparse-web-frontend:browserDevelopmentRun --continuous