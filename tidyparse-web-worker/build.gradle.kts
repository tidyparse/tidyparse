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
  val coroutinesVersion = "1.7.3"
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:$coroutinesVersion")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:$coroutinesVersion")
}

kotlin {
  js(IR) {
    binaries.executable()
    browser {
      // Disable minification
      webpackTask {
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-web-worker.js"
        devtool = "source-map"// Remove later for production
      }
      distribution {
        distributionName = "tidyparse-web-worker"
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
  val dir = "$rootDir/$worker/build/$worker/".also { assert(File(it).exists())}
  // Copy all files in directory
  from("$dir/$worker.js", "$dir/$worker.js.map")
  into("$rootDir/tidyparse-web-frontend/build/processedResources/js/main/")
}

tasks["copyJsTask"].dependsOn.add("browserDistribution")
tasks["developmentExecutableCompileSync"].dependsOn.add("browserProductionWebpack")
//tasks["browserDevelopmentRun"].mustRunAfter(":tidyparse-web-frontend:developmentExecutableCompileSync")