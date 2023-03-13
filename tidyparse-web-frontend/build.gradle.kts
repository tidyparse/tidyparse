plugins {
  kotlin("js")
}

fun properties(key: String) = project.findProperty(key).toString()
group = properties("pluginGroup")
version = properties("pluginVersion")

dependencies {
  testImplementation(kotlin("test"))
  api(project(":tidyparse-core"))
  implementation("org.jetbrains.kotlinx:kotlinx-html:0.8.1")
//  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core")
  val coroutinesVersion = "1.7.0-Beta"
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:$coroutinesVersion")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:$coroutinesVersion")
}

kotlin {
  js(IR) {
    binaries.executable()
    browser {
      runTask {
        outputFileName = "tidyparse-web-frontend.js"
      }

      webpackTask {
        outputFileName = "tidyparse-web-frontend.js"
      }

      distribution {
        name = "tidyparse-web-frontend"
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
