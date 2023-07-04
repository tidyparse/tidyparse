import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT


plugins {
  kotlin("js")
}

fun properties(key: String) = project.findProperty(key).toString()
group = properties("pluginGroup")
version = properties("pluginVersion")

dependencies {
  api(project(":tidyparse-core"))
  implementation("org.jetbrains.kotlinx:kotlinx-html:0.9.0")
//  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core")
  val coroutinesVersion = "1.7.1"
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:$coroutinesVersion")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:$coroutinesVersion")
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

//  sourceSets {
//    all {
//      languageSettings.apply {
//        languageVersion = "2.0"
//      }
//    }
//  }
}

//tasks.register<Copy>("copyJsTask") {
////  dependsOn("browserDistribution")
//  val worker = "tidyparse-web-worker"
//  from("$rootDir/$worker/build/$worker/$worker.js")
//  into("$rootDir/tidyparse-web-frontend/build/processedResources/js/main/")
//}
//
//tasks["processResources"].dependsOn.add(":tidyparse-web-worker:browserDistribution")
//tasks["browserDevelopmentRun"].dependsOn.add("copyJsTask")
/*
No clue how Gradle actually works.

First run:

./gradlew :tidyparse-web-backend:browserDevelopmentRun --continuous

Ensure tidyparse-web-worker/build/tidyparse-web-worker/tidyparse-web-worker.js exists

Then run:

./gradlew :tidyparse-web-frontend:browserDevelopmentRun --continuous
 */
tasks["processResources"].dependsOn.add(":tidyparse-web-worker:copyJsTask")
