import org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
import org.gradle.api.tasks.testing.logging.TestLogEvent.*
import org.jetbrains.changelog.Changelog.OutputType.HTML
import org.jetbrains.changelog.markdownToHTML
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.intellij.platform.gradle.TestFrameworkType

plugins {
  kotlin("jvm")
  id("org.jetbrains.intellij.platform") version "2.9.0"
  id("org.jetbrains.changelog") version "2.4.0"
}

repositories {
  mavenCentral()
  intellijPlatform.defaultRepositories()
}

group = providers.gradleProperty("pluginGroup")
version = providers.gradleProperty("pluginVersion")

// Configure Gradle IntelliJ Plugin - read more: https://github.com/JetBrains/gradle-intellij-plugin
intellijPlatform {
  pluginConfiguration {
    version = providers.gradleProperty("pluginVersion")
    name = "TidyParse"
  }

//  pluginVerification.ides.recommended()
}

// Configure Gradle Changelog Plugin - read more: https://github.com/JetBrains/gradle-changelog-plugin
changelog {
  version = providers.gradleProperty("pluginVersion")
  groups = emptyList()
}

val swiftGenerator by configurations.creating
dependencies {
  implementation("net.java.dev.jna:jna:5.18.1")
  implementation(project(":tidyparse-core")) {
    exclude(group = "org.jetbrains.kotlin")
    exclude(group = "org.jetbrains.kotlinx")
  }
//  testImplementation(kotlin("test"))
  swiftGenerator(kotlin("stdlib"))
  testImplementation("org.opentest4j:opentest4j:1.3.0")
  intellijPlatform {
    testImplementation("junit:junit:4.13.2")

//    bundledPlugins("com.intellij.java")
    create("IC", "2025.1")
    pluginVerifier()
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

val javaVersion = providers.gradleProperty("javaVersion").get()

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
    version = providers.gradleProperty("pluginVersion")
    sinceBuild = providers.gradleProperty("pluginSinceBuild")
//    untilBuild = providers.gradleProperty("pluginUntilBuild")

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

  val generateDylib by registering(JavaExec::class) {
    mainClass.set("ai.hypergraph.tidyparse.MetalShadersKt")
    classpath = sourceSets["main"].runtimeClasspath + swiftGenerator
  }
}