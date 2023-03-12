import org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
import org.gradle.api.tasks.testing.logging.TestLogEvent.*
import org.jetbrains.changelog.Changelog.OutputType.HTML
import org.jetbrains.changelog.markdownToHTML

plugins {
  kotlin("jvm")
  id("org.jetbrains.intellij") version "1.13.2"
  id("org.jetbrains.changelog") version "2.0.0"
}

fun properties(key: String) = project.findProperty(key).toString()

group = properties("pluginGroup")
version = properties("pluginVersion")

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

dependencies {
  api(project(":tidyparse-core"))
  testImplementation(kotlin("test"))
}

//kotlin {
//  sourceSets {
//    all {
//      languageSettings.apply {
//        languageVersion = "2.0"
//      }
//    }
//  }
//}

tasks {
  test {
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
  val javaVersion = properties("javaVersion")
  compileJava {
    sourceCompatibility = javaVersion
    targetCompatibility = javaVersion
  }

  compileKotlin {
    kotlinOptions {
      jvmTarget = javaVersion
      apiVersion = languageVersion
    }
  }

  patchPluginXml {
    version.set(properties("pluginVersion"))
    sinceBuild.set(properties("pluginSinceBuild"))
    untilBuild.set(properties("pluginUntilBuild"))

// Extract the <!-- Plugin description --> section from README.md and provide for the plugin's manifest
    pluginDescription.set(
      projectDir.parentFile.resolve("README.md").readText().lines().run {
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
    args = listOf(projectDir.parent + "/examples")
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