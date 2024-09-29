import org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
import org.gradle.api.tasks.testing.logging.TestLogEvent.*
import org.jetbrains.changelog.Changelog.OutputType.HTML
import org.jetbrains.changelog.markdownToHTML
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.intellij.platform.gradle.TestFrameworkType

plugins {
  kotlin("jvm")
  id("org.jetbrains.intellij.platform") version "2.1.0"
  id("org.jetbrains.changelog") version "2.2.1"
}

repositories {
  mavenCentral()
  intellijPlatform.defaultRepositories()
}

fun properties(key: String) = project.findProperty(key).toString()

group = properties("pluginGroup")
version = properties("pluginVersion")

// Configure Gradle IntelliJ Plugin - read more: https://github.com/JetBrains/gradle-intellij-plugin
intellijPlatform {
  pluginConfiguration {
    version = properties("pluginVersion")
    name = "TidyParse"
  }

//  pluginVerification.ides.recommended()
}

// Configure Gradle Changelog Plugin - read more: https://github.com/JetBrains/gradle-changelog-plugin
changelog {
  version = properties("pluginVersion")
  groups = emptyList()
}

dependencies {
  implementation(project(":tidyparse-core")) {
    exclude(group = "org.jetbrains.kotlin")
    exclude(group = "org.jetbrains.kotlinx")
  }
//  testImplementation(kotlin("test"))
  testImplementation("org.opentest4j:opentest4j:1.3.0")
  intellijPlatform {
    testImplementation("junit:junit:4.13.2")

//    bundledPlugins("com.intellij.java")
    create("IC", "2024.2.2")
    pluginVerifier()
    instrumentationTools()
    testFramework(TestFrameworkType.Platform)
  }
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

val javaVersion = properties("javaVersion")
kotlin {
  compilerOptions {
    jvmTarget = JvmTarget.fromTarget(javaVersion)
    apiVersion = languageVersion
  }
}

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
  compileJava {
    sourceCompatibility = javaVersion
    targetCompatibility = javaVersion
  }

  patchPluginXml {
    version = properties("pluginVersion")
    sinceBuild = properties("pluginSinceBuild")
    untilBuild = properties("pluginUntilBuild")

  pluginDescription =
    projectDir.parentFile.resolve("README.md").readText().lines().run {
      val start = "<!-- Plugin description -->"
      val end = "<!-- Plugin description end -->"

      if (!containsAll(listOf(start, end)))
        throw GradleException("Plugin description section not found in README.md:\n$start ... $end")
      subList(indexOf(start) + 1, indexOf(end))
    }.joinToString("\n").run { markdownToHTML(this) }

// Get the latest available change notes from the changelog file
    changeNotes = provider { changelog.renderItem(changelog.getAll().values.first(), HTML) }
  }

  runIde {
    maxHeapSize = "4g"
    args = listOf(projectDir.parent + "/examples")
  }
}