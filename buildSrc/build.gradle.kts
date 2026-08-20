plugins {
  `kotlin-dsl`
}

repositories {
  gradlePluginPortal()
  mavenCentral()
}

dependencies {
  testImplementation(kotlin("test"))
}

sourceSets {
  main {
    kotlin.srcDir("../tidyparse-cpp/build-logic/src/main/kotlin")
  }
  test {
    kotlin.srcDir("../tidyparse-cpp/build-logic/src/test/kotlin")
  }
}

tasks.test {
  useJUnitPlatform()
}
