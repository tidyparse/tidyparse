import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
import org.gradle.api.tasks.testing.logging.TestLogEvent.*
import org.jetbrains.changelog.Changelog.OutputType.HTML
import org.jetbrains.changelog.markdownToHTML

fun properties(key: String) = project.findProperty(key).toString()

plugins {
  kotlin("multiplatform") version "1.8.0-RC2"
  id("org.jetbrains.intellij") version "1.11.0"
  id("org.jetbrains.changelog") version "2.0.0"
  id("com.github.ben-manes.versions") version "0.44.0"
}

group = properties("pluginGroup")
version = properties("pluginVersion")

// Configure project's dependencies
repositories {
  mavenCentral()
  maven("https://maven.pkg.jetbrains.space/public/p/kotlinx-html/maven")
}

// Configure Gradle IntelliJ Plugin - read more: https://github.com/JetBrains/gradle-intellij-plugin
intellij {
  pluginName.set(properties("pluginName"))
  version.set(properties("platformVersion"))
  type.set(properties("platformType"))

  // Plugin Dependencies. Uses `platformPlugins` property from the gradle.properties file.
  plugins.set(properties("platformPlugins").split(',').map(String::trim).filter(String::isNotEmpty))
}

// Configure Gradle Changelog Plugin - read more: https://github.com/JetBrains/gradle-changelog-plugin
changelog {
  version.set(properties("pluginVersion"))
  groups.set(emptyList())
}

kotlin {
  jvm {
    compilations.all {
      kotlinOptions.jvmTarget = "17"
    }
    withJava()
    testRuns["test"].executionTask.configure {
      useJUnitPlatform()
    }
  }

  js(IR) {
    binaries.executable()
    browser()
  }

  jvmToolchain {
    run {
      languageVersion.set(JavaLanguageVersion.of(17))
    }
  }

  sourceSets {
    val commonMain by getting {
      dependencies {
        implementation("ai.hypergraph:kaliningraph") {
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib")
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib-common")
//  exclude(group = "org.jetbrains.kotlin", module = "kotlin-reflect")
          exclude(group = "guru.nidi", module = "graphviz-kotlin")
          exclude(group = "org.graalvm.js", module = "js")
          exclude(group = "org.jetbrains.kotlinx", module = "kotlinx-coroutines-core")
          exclude(group = "org.jetbrains.kotlinx", module = "kotlinx-html-jvm")
//          exclude(group = "org.jetbrains.kotlinx", module = "multik-core")
//          exclude(group = "org.jetbrains.kotlinx", module = "multik-default")
          exclude(group = "org.jetbrains.lets-plot", module = "lets-plot-kotlin-jvm")
          exclude(group = "org.apache.datasketches", module = "datasketches")
          exclude(group = "org.apache.datasketches", module = "datasketches-java")
          exclude(group = "ca.umontreal.iro.simul", module = "ssj")
          exclude(group = "org.sosy-lab", module = "common")
          exclude(group = "org.sosy-lab", module = "java-smt")
          exclude(group = "org.sosy-lab", module = "javasmt-solver-mathsat5")
        }
      }
    }
    val commonTest by getting {
      dependencies {
        implementation(kotlin("test"))
      }
    }
    val jvmMain by getting {
      dependencies {
        implementation("io.github.java-diff-utils:java-diff-utils:4.12")
      }
    }
    val jvmTest by getting
    val jsMain by getting {
      dependencies {
        implementation("org.jetbrains.kotlinx:kotlinx-html:0.8.1")
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:1.6.4")
      }
    }
    val jsTest by getting
  }
}

tasks {
  withType<Test> {
    minHeapSize = "1g"
    maxHeapSize = "3g"

    testLogging {
      events = setOf(
        FAILED,
        PASSED,
        SKIPPED,
        STANDARD_OUT
      )
      exceptionFormat = FULL
      showExceptions = true
      showCauses = true
      showStackTraces = true
      showStandardStreams = true
    }
  }

  // Set the JVM compatibility versions
  properties("javaVersion").let {
    withType<JavaCompile> {
      sourceCompatibility = "17"
      targetCompatibility = it
    }
    withType<KotlinCompile> {
      kotlinOptions {
        jvmTarget = it
        apiVersion = languageVersion
      }
    }
  }

  wrapper {
    gradleVersion = properties("gradleVersion")
  }

  patchPluginXml {
    version.set(properties("pluginVersion"))
    sinceBuild.set(properties("pluginSinceBuild"))
    untilBuild.set(properties("pluginUntilBuild"))

    // Extract the <!-- Plugin description --> section from README.md and provide for the plugin's manifest
    pluginDescription.set(
      projectDir.resolve("README.md").readText().lines().run {
        val start = "<!-- Plugin description -->"
        val end = "<!-- Plugin description end -->"

        if (!containsAll(listOf(start, end)))
          throw GradleException("Plugin description section not found in README.md:\n$start ... $end")
        subList(indexOf(start) + 1, indexOf(end))
      }.joinToString("\n").run { markdownToHTML(this) }
    )

    // Get the latest available change notes from the changelog file
    changeNotes.set(provider {
      changelog.renderItem(changelog.getAll().values.first(), HTML)
    })
  }

  runPluginVerifier {
    ideVersions.set(listOf("2022.3"))
  }

  runIde {
    maxHeapSize = "4g"
    args = listOf(projectDir.absolutePath + "/examples")
  }
//
//  // Configure UI tests plugin
//  // Read more: https://github.com/JetBrains/intellij-ui-test-robot
//  runIdeForUiTests {
//    systemProperty("robot-server.port", "8082")
//    systemProperty("ide.mac.message.dialogs.as.sheets", "false")
//    systemProperty("jb.privacy.policy.text", "<!--999.999-->")
//    systemProperty("jb.consents.confirmation.enabled", "false")
//  }
//
//  signPlugin {
//    certificateChain.set(System.getenv("CERTIFICATE_CHAIN"))
//    privateKey.set(System.getenv("PRIVATE_KEY"))
//    password.set(System.getenv("PRIVATE_KEY_PASSWORD"))
//  }
//
//  publishPlugin {
//    dependsOn("patchChangelog")
//    token.set(System.getenv("PUBLISH_TOKEN"))
//    // pluginVersion is based on the SemVer (https://semver.org) and supports pre-release labels, like 2.1.7-alpha.3
//    // Specify pre-release label to publish the plugin in a custom Release Channel automatically. Read more:
//    // https://plugins.jetbrains.com/docs/intellij/deployment.html#specifying-a-release-channel
//    channels.set(listOf(properties("pluginVersion").split('-').getOrElse(1) { "default" }.split('.').first()))
//  }
}