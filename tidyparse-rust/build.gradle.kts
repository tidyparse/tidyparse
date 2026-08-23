import buildlogic.ManagedSiteDeployTask
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpack
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT

plugins {
  kotlin("multiplatform")
}

group = "ai.hypergraph"
version = "0.23.0"

val rustCrateDir = layout.projectDirectory.dir("wasm")
val rustTargetDir = layout.buildDirectory.dir("rust-target")
val rustToolchain = "1.91.0"
val rustupProxyDir = providers.environmentVariable("CARGO_HOME")
  .orElse(File(System.getProperty("user.home"), ".cargo").absolutePath)
  .map { File(it, "bin") }
val rustToolchainMarker = layout.buildDirectory.file("rust-toolchain/$rustToolchain-installed")
val rustWasmArtifact = rustTargetDir.map {
  it.file("wasm32-unknown-unknown/release/tidyparse_rust_glancer.wasm")
}
val generatedRustResources = layout.buildDirectory.dir("generated/rust-glancer-resources")

val installRustWasmToolchain = tasks.register<Exec>("installRustWasmToolchain") {
  group = "build setup"
  description = "Installs the pinned Rust toolchain and wasm32 target used by Rust Glancer"
  commandLine(
    "rustup",
    "toolchain",
    "install",
    rustToolchain,
    "--profile",
    "minimal",
    "--target",
    "wasm32-unknown-unknown"
  )
  inputs.property("rustToolchain", rustToolchain)
  outputs.file(rustToolchainMarker)
  doLast {
    rustToolchainMarker.get().asFile.apply {
      parentFile.mkdirs()
      writeText("$rustToolchain\n")
    }
  }
}

val buildRustGlancerWasm = tasks.register<Exec>("buildRustGlancerWasm") {
  group = "build"
  description = "Builds the Rust Glancer syntax engine for wasm32-unknown-unknown"

  dependsOn(installRustWasmToolchain)
  workingDir(rustCrateDir)
  commandLine(
    "rustup",
    "run",
    rustToolchain,
    "cargo",
    "build",
    "--locked",
    "--release",
    "--target",
    "wasm32-unknown-unknown"
  )
  environment("CARGO_TARGET_DIR", rustTargetDir.get().asFile.absolutePath)
  // A Homebrew rustc may precede rustup in PATH. Cargo's child compiler must use the same pinned
  // toolchain as Cargo itself, so put rustup's proxy directory first for the whole build.
  environment(
    "PATH",
    rustupProxyDir.get().absolutePath + File.pathSeparator + (System.getenv("PATH") ?: "")
  )

  inputs.files(fileTree(rustCrateDir) {
    include("Cargo.toml")
    include("Cargo.lock")
    include("rust-toolchain.toml")
    include("src/**/*.rs")
  }).withPathSensitivity(PathSensitivity.RELATIVE)
  outputs.file(rustWasmArtifact)
}

val stageRustGlancerWasm = tasks.register<Sync>("stageRustGlancerWasm") {
  group = "build"
  description = "Stages the Rust Glancer WebAssembly module as a browser resource"
  dependsOn(buildRustGlancerWasm)

  from(rustWasmArtifact) {
    rename { "tidyparse-rust-glancer.wasm" }
  }
  into(generatedRustResources)
}

val monacoWebpackConfigDir = layout.buildDirectory.dir("generated/monaco-webpack-config")
val monacoWebpackConfig = """
  const webpack = require("webpack");
  // The same Kotlin/JS bundle is evaluated in the page and in named workers.
  config.target = "webworker";
  config.plugins.push(new webpack.optimize.LimitChunkCountPlugin({ maxChunks: 1 }));
  config.module.rules.push({
    test: /\.(ttf|woff2?|eot)$/i,
    type: "asset/inline"
  });
""".trimIndent() + "\n"

val prepareMonacoWebpackConfig = tasks.register("prepareMonacoWebpackConfig") {
  val configFile = monacoWebpackConfigDir.map { it.file("single-bundle.js") }
  inputs.property("contents", monacoWebpackConfig)
  outputs.file(configFile)

  doLast {
    configFile.get().asFile.apply {
      parentFile.mkdirs()
      writeText(monacoWebpackConfig)
    }
  }
}

kotlin {
  js {
    binaries.executable()

    browser {
      commonWebpackConfig {
        configDirectory = monacoWebpackConfigDir.get().asFile
        cssSupport { enabled.set(true) }
      }

      runTask {
        mainOutputFileName = "tidyparse-rust.js"
        webpackConfigApplier {
          devServer = (devServer ?: KotlinWebpackConfig.DevServer()).apply {
            open = "rust.html"
          }
        }
      }

      webpackTask {
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-rust.js"
        devtool = "source-map"
      }
    }
  }

  sourceSets {
    getByName("jsMain") {
      resources.srcDir(generatedRustResources)
      dependencies {
        implementation(kotlin("stdlib"))
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2026.8.0")
        // Keep vanilla Monaco separate from tidyparse-cpp's Codingame package alias.
        // Monaco 0.56 changed its package export layout; 0.55 keeps the stable ESM worker and
        // basic-language entrypoints that webpack can bundle without a loader plugin.
        implementation(npm("vanilla-monaco-editor", "npm:monaco-editor@0.55.1"))
      }
    }
  }
}

tasks.named("jsProcessResources") {
  dependsOn(stageRustGlancerWasm)
}

tasks.withType<KotlinWebpack>().configureEach {
  dependsOn(prepareMonacoWebpackConfig)
}

val rustProductionBundleDir = layout.buildDirectory.dir("kotlin-webpack/js/productionExecutable")
val rustProductionJsFile = rustProductionBundleDir.map { it.file("tidyparse-rust.js").asFile }
val rustProductionJsMapFile = rustProductionBundleDir.map { it.file("tidyparse-rust.js.map").asFile }
val rustGlancerWasmFile = generatedRustResources.map { it.file("tidyparse-rust-glancer.wasm").asFile }
val rustDeployStagingDir = layout.buildDirectory.dir("rust-deploy")

val prepareRustDeploy = tasks.register<Sync>("prepareRustDeploy") {
  group = "deployment"
  description = "Stages the Rust playground files for deployment to tidyparse.github.io"

  dependsOn("jsBrowserProductionWebpack", stageRustGlancerWasm)

  into(rustDeployStagingDir)
  from("src/jsMain/resources") {
    exclude(".DS_Store")
    exclude("**/.DS_Store")
  }
  from(generatedRustResources) {
    include("tidyparse-rust-glancer.wasm")
  }
  from(rustProductionBundleDir) {
    include("tidyparse-rust.js")
    include("tidyparse-rust.js.map")
  }

  inputs.files(rustProductionJsFile, rustProductionJsMapFile, rustGlancerWasmFile)
  outputs.files(
    rustDeployStagingDir.map { it.file("rust.html") },
    rustDeployStagingDir.map { it.file("rust.css") },
    rustDeployStagingDir.map { it.file("tidyparse-rust.js") },
    rustDeployStagingDir.map { it.file("tidyparse-rust.js.map") },
    rustDeployStagingDir.map { it.file("tidyparse-rust-glancer.wasm") }
  )

  doLast {
    val stagingDir = rustDeployStagingDir.get().asFile
    val requiredFiles = listOf(
      "rust.html",
      "rust.css",
      "tidyparse-rust.js",
      "tidyparse-rust.js.map",
      "tidyparse-rust-glancer.wasm"
    )
    requiredFiles.forEach { relativePath ->
      val stagedFile = stagingDir.resolve(relativePath)
      check(stagedFile.isFile && stagedFile.length() > 0) {
        "Rust deployment is missing required file: $relativePath"
      }
    }

    val html = stagingDir.resolve("rust.html").readText()
    check("src=\"tidyparse-rust.js\"" in html) {
      "Staged rust.html must load tidyparse-rust.js"
    }
    check(stagingDir.resolve("tidyparse-rust.js").readText().contains("tidyparse-rust-glancer.wasm")) {
      "Staged Rust bundle must reference tidyparse-rust-glancer.wasm"
    }
    val wasmMagic = stagingDir.resolve("tidyparse-rust-glancer.wasm").inputStream().use {
      it.readNBytes(4)
    }
    check(wasmMagic.contentEquals(byteArrayOf(0x00, 0x61, 0x73, 0x6d))) {
      "Staged Rust Glancer resource is not a WebAssembly binary"
    }
  }
}

tasks.register<ManagedSiteDeployTask>("deployRust") {
  group = "deployment"
  description = "Builds, commits, and pushes the Rust playground to tidyparse.github.io. Requires --msg \"commit message\"."

  dependsOn(prepareRustDeploy)

  sourceDirectory.set(rustDeployStagingDir)
  deploymentId.set("rust")
  requiredSiteEntrypoints.put("rust.html", "tidyparse-rust.js")
  commitMessage.convention(providers.gradleProperty("deployRustMessage"))
  repositoryUrl.convention(
    providers.gradleProperty("deployRustRepoUrl")
      .orElse("https://github.com/tidyparse/tidyparse.github.io.git")
  )
  pushUrl.convention(
    providers.gradleProperty("deployRustPushUrl")
      .orElse("git@github.com:tidyparse/tidyparse.github.io.git")
  )
  branch.convention(providers.gradleProperty("deployRustBranch").orElse("main"))
  checkoutPath.convention(
    providers.gradleProperty("deployRustRepoDir")
      .orElse(layout.buildDirectory.dir("deploy/tidyparse.github.io").map { it.asFile.absolutePath })
  )
}

// ./gradlew :tidyparse-rust:jsBrowserDevelopmentRun --continuous
// ./gradlew :tidyparse-rust:deployRust --msg "update Rust playground"
