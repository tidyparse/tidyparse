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
      webpackTask {
        outputFileName = "tidyparse-web-worker.js"
        devtool = "source-map"// Remove later for production
      }
      distribution {
        name = "tidyparse-web-worker"
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

tasks.register<Copy>("copyJsTask") {
//  dependsOn("browserDistribution")
  mustRunAfter("browserDistribution")
  val worker = "tidyparse-web-worker"
  from("$rootDir/$worker/build/$worker/$worker.js".also { assert(File(it).exists())})
  into("$rootDir/tidyparse-web-frontend/build/processedResources/js/main/")
}

tasks["copyJsTask"].dependsOn.add("browserDistribution")
tasks["developmentExecutableCompileSync"].dependsOn.add("browserProductionWebpack")
//tasks["browserDevelopmentRun"].mustRunAfter(":tidyparse-web-frontend:developmentExecutableCompileSync")