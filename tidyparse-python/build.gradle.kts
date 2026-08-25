import buildlogic.ManagedSiteDeployTask
import org.gradle.api.GradleException
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpack
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT

plugins { kotlin("multiplatform") }

group = "ai.hypergraph"
version = "0.23.0"

val tyWasmRepository = "https://github.com/astral-sh/ruff.git"
val tyWasmCommit = "423b9fbf1923b00e66f25f059b1e91dd79aacd03"
val tyWasmShortCommit = tyWasmCommit.take(9)
val rustToolchain = "1.98.0"
val wasmPackVersion = "0.13.1"

val executableSuffix = if (System.getProperty("os.name").startsWith("Windows", ignoreCase = true)) ".exe" else ""
val rustupProxyDir = providers.environmentVariable("CARGO_HOME")
  .orElse(File(System.getProperty("user.home"), ".cargo").absolutePath)
  .map { File(it, "bin") }
val rustToolchainMarker = layout.buildDirectory.file("rust-toolchain/$rustToolchain-installed")
val wasmPackInstallDir = layout.buildDirectory.dir("wasm-pack-$wasmPackVersion")
val wasmPackExecutable = wasmPackInstallDir.map { it.file("bin/wasm-pack$executableSuffix") }
val tyWasmSourceDir = layout.buildDirectory.dir("ty-wasm-source/ruff")
val tyWasmSourceMarker = layout.buildDirectory.file("ty-wasm-source/$tyWasmCommit.ready")
val tyWasmTargetDir = layout.buildDirectory.dir("ty-wasm-target")
val tyWasmPackageDir = layout.buildDirectory.dir("ty-wasm-package")
val generatedTyWasmResources = layout.buildDirectory.dir("generated/ty-wasm-resources")
val generatedRepairWorkerResources = layout.buildDirectory.dir("generated/repair-worker-resources")
val repairWorkerBundleName = "tidyparse-python-repair.js"
val repairWorkerSourceMapName = "$repairWorkerBundleName.map"
val repairWorkerProject = project(":tidyparse-python:repair-worker")
val repairWorkerBundleDir = repairWorkerProject.layout.buildDirectory.dir("kotlin-webpack/js/productionExecutable")

fun runChecked(workingDirectory: File, vararg command: String): String {
  val process = ProcessBuilder(command.toList()).directory(workingDirectory).redirectErrorStream(true).start()
  val output = process.inputStream.bufferedReader().readText()
  val exitCode = process.waitFor()
  if (exitCode != 0) {
    val rendered = command.joinToString(" ") { argument ->
      if (argument.any(Char::isWhitespace)) "\"${argument.replace("\"", "\\\"")}\"" else argument
    }
    throw GradleException("Command failed ($rendered) with exit code $exitCode:\n${output.trim()}")
  }
  return output.trim()
}

val installTyWasmToolchain = tasks.register<Exec>("installTyWasmToolchain") {
  group = "build setup"
  description = "Installs the pinned Rust toolchain and wasm32 target used by ty_wasm"

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

val installPinnedWasmPack = tasks.register<Exec>("installPinnedWasmPack") {
  group = "build setup"
  description = "Installs wasm-pack $wasmPackVersion into this module's generated build tools"

  dependsOn(installTyWasmToolchain)
  commandLine(
    "rustup",
    "run",
    rustToolchain,
    "cargo",
    "install",
    "wasm-pack",
    "--version",
    wasmPackVersion,
    "--locked",
    "--root",
    wasmPackInstallDir.get().asFile.absolutePath
  )
  environment(
    "PATH",
    rustupProxyDir.get().absolutePath + File.pathSeparator + (System.getenv("PATH") ?: "")
  )
  inputs.property("wasmPackVersion", wasmPackVersion)
  inputs.property("rustToolchain", rustToolchain)
  outputs.file(wasmPackExecutable)
}

val checkoutPinnedTyWasmSource = tasks.register("checkoutPinnedTyWasmSource") {
  group = "build setup"
  description = "Fetches the exact Astral Ruff revision containing the browser ty_wasm source"

  inputs.property("repository", tyWasmRepository)
  inputs.property("commit", tyWasmCommit)
  outputs.files(tyWasmSourceMarker)

  doLast {
    val sourceDir = tyWasmSourceDir.get().asFile
    project.delete(sourceDir)
    sourceDir.mkdirs()

    runChecked(sourceDir, "git", "init")
    runChecked(sourceDir, "git", "remote", "add", "origin", tyWasmRepository)
    runChecked(sourceDir, "git", "fetch", "--depth=1", "origin", tyWasmCommit)
    runChecked(sourceDir, "git", "checkout", "--detach", "FETCH_HEAD")

    tyWasmSourceMarker.get().asFile.apply {
      parentFile.mkdirs()
      writeText("$tyWasmRepository\n$tyWasmCommit\n")
    }
  }
}

val buildTyWasm = tasks.register<Exec>("buildTyWasm") {
  group = "build"
  description = "Builds official ty_wasm $tyWasmShortCommit for the browser"

  dependsOn(installPinnedWasmPack, checkoutPinnedTyWasmSource)
  workingDir(tyWasmSourceDir)
  commandLine(
    wasmPackExecutable.get().asFile.absolutePath,
    "build",
    "--target",
    "web",
    "--release",
    "--out-dir",
    tyWasmPackageDir.get().asFile.absolutePath,
    tyWasmSourceDir.get().dir("crates/ty_wasm").asFile.absolutePath
  )
  environment("CARGO_INCREMENTAL", "0")
  environment("CARGO_NET_RETRY", "10")
  environment("CARGO_TARGET_DIR", tyWasmTargetDir.get().asFile.absolutePath)
  environment("RUSTUP_TOOLCHAIN", rustToolchain)
  environment("TY_WASM_COMMIT_SHORT_HASH", tyWasmShortCommit)
  environment(
    "PATH",
    rustupProxyDir.get().absolutePath + File.pathSeparator + (System.getenv("PATH") ?: "")
  )

  inputs.file(tyWasmSourceMarker)
  inputs.property("commit", tyWasmCommit)
  inputs.property("rustToolchain", rustToolchain)
  inputs.property("wasmPackVersion", wasmPackVersion)
  outputs.files(
    tyWasmPackageDir.map { it.file("ty_wasm.js") },
    tyWasmPackageDir.map { it.file("ty_wasm_bg.wasm") }
  )
}

val stageTyWasmResources = tasks.register<Sync>("stageTyWasmResources") {
  group = "build"
  description = "Stages ty_wasm's browser ES module and WebAssembly binary as generated resources"

  dependsOn(buildTyWasm)
  from(tyWasmPackageDir) {
    include("ty_wasm.js")
    include("ty_wasm_bg.wasm")
  }
  into(generatedTyWasmResources)
}

val stageRepairWorkerResources = tasks.register<Sync>("stageRepairWorkerResources") {
  group = "build"
  description = "Stages the standalone Python syntax-repair worker as a main browser resource"

  dependsOn(repairWorkerProject.tasks.named("jsBrowserProductionWebpack"))
  from(repairWorkerBundleDir) { include(repairWorkerBundleName) }
  into(generatedRepairWorkerResources)
}

val monacoWebpackConfigDir = layout.buildDirectory.dir("generated/monaco-webpack-config")
val monacoWebpackConfig = """
  const webpack = require("webpack");
  // The same Kotlin/JS bundle is evaluated in the page and Monaco's editor worker.
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
        mainOutputFileName = "tidyparse-python.js"
        webpackConfigApplier {
          devServer = (devServer ?: KotlinWebpackConfig.DevServer()).apply { open = "python3.html" }
        }
      }

      webpackTask {
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-python.js"
        devtool = "source-map"
      }
    }
  }

  sourceSets {
    getByName("jsMain") {
      resources.srcDir(generatedTyWasmResources)
      resources.srcDir(generatedRepairWorkerResources)
      dependencies {
        implementation(kotlin("stdlib"))
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2026.8.0")
        implementation(npm("vanilla-monaco-editor", "npm:monaco-editor@0.55.1"))
      }
    }

    getByName("jsTest") { dependencies { implementation(kotlin("test-js")) } }
  }
}

val browserRuntimeResources = listOf(stageTyWasmResources, stageRepairWorkerResources)

tasks.named("jsProcessResources") { mustRunAfter(browserRuntimeResources) }
tasks.withType<KotlinWebpack>().configureEach {
  dependsOn(prepareMonacoWebpackConfig)
  dependsOn(browserRuntimeResources)
}

val pythonProductionBundleDir = layout.buildDirectory.dir("kotlin-webpack/js/productionExecutable")
val pythonProcessedResourcesDir = layout.buildDirectory.dir("processedResources/js/main")
val pythonDeployStagingDir = layout.buildDirectory.dir("python-deploy")

val preparePythonDeploy = tasks.register<Sync>("preparePythonDeploy") {
  group = "deployment"
  description = "Stages the standalone Python 3 playground for deployment to tidyparse.github.io"

  dependsOn("jsBrowserProductionWebpack")

  into(pythonDeployStagingDir)
  from(pythonProcessedResourcesDir)
  from(pythonProductionBundleDir) {
    include("tidyparse-python.js")
    include("tidyparse-python.js.map")
  }
  from(repairWorkerBundleDir) { include(repairWorkerSourceMapName) }
}

tasks.register<ManagedSiteDeployTask>("deployPython") {
  group = "deployment"
  description = "Builds, commits, and pushes the Python 3 playground to tidyparse.github.io. Requires --msg \"commit message\"."

  dependsOn(preparePythonDeploy)

  sourceDirectory.set(pythonDeployStagingDir)
  deploymentId.set("python3")
  requiredSiteEntrypoints.put("python3.html", "tidyparse-python.js")
  commitMessage.convention(providers.gradleProperty("deployPythonMessage"))
  repositoryUrl.convention(
    providers.gradleProperty("deployPythonRepoUrl")
      .orElse("https://github.com/tidyparse/tidyparse.github.io.git")
  )
  pushUrl.convention(
    providers.gradleProperty("deployPythonPushUrl")
      .orElse("git@github.com:tidyparse/tidyparse.github.io.git")
  )
  branch.convention(providers.gradleProperty("deployPythonBranch").orElse("main"))
  checkoutPath.convention(
    providers.gradleProperty("deployPythonRepoDir")
      .orElse(layout.buildDirectory.dir("deploy/tidyparse.github.io").map { it.asFile.absolutePath })
  )
}

// ./gradlew :tidyparse-python:jsBrowserDevelopmentRun --continuous
// ./gradlew :tidyparse-python:deployPython --msg "update Python 3 playground"
