import buildlogic.ManagedSiteDeployTask
import org.gradle.api.GradleException
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpack
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT

plugins {
  kotlin("multiplatform")
}

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
val tyLicenseFile = tyWasmSourceDir.map { it.file("LICENSE") }
val typeshedLicenseFile = tyWasmSourceDir.map {
  it.file("crates/ty_vendored/vendor/typeshed/LICENSE")
}

fun runChecked(workingDirectory: File, vararg command: String): String {
  val process = ProcessBuilder(command.toList())
    .directory(workingDirectory)
    .redirectErrorStream(true)
    .start()
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

  doLast {
    val executable = wasmPackExecutable.get().asFile
    check(executable.isFile) { "Pinned wasm-pack executable was not installed at ${executable.absolutePath}" }
    val installedVersion = runChecked(projectDir, executable.absolutePath, "--version")
    check(installedVersion == "wasm-pack $wasmPackVersion") {
      "Expected wasm-pack $wasmPackVersion, got '$installedVersion'"
    }
  }
}

val checkoutPinnedTyWasmSource = tasks.register("checkoutPinnedTyWasmSource") {
  group = "build setup"
  description = "Fetches the exact Astral Ruff revision containing the browser ty_wasm source"

  inputs.property("repository", tyWasmRepository)
  inputs.property("commit", tyWasmCommit)
  outputs.files(tyWasmSourceMarker, tyLicenseFile, typeshedLicenseFile)

  doLast {
    val sourceDir = tyWasmSourceDir.get().asFile
    project.delete(sourceDir)
    sourceDir.mkdirs()

    runChecked(sourceDir, "git", "init")
    runChecked(sourceDir, "git", "remote", "add", "origin", tyWasmRepository)
    runChecked(sourceDir, "git", "fetch", "--depth=1", "origin", tyWasmCommit)
    runChecked(sourceDir, "git", "checkout", "--detach", "FETCH_HEAD")

    val actualCommit = runChecked(sourceDir, "git", "rev-parse", "HEAD")
    check(actualCommit == tyWasmCommit) {
      "Expected Ruff revision $tyWasmCommit, checked out $actualCommit"
    }
    val pinnedToolchain = sourceDir.resolve("rust-toolchain.toml").readText()
    check("channel = \"$rustToolchain\"" in pinnedToolchain) {
      "Ruff revision $tyWasmCommit no longer matches Rust $rustToolchain"
    }
    check(sourceDir.resolve("crates/ty_wasm/Cargo.toml").isFile) {
      "Ruff revision $tyWasmCommit does not contain crates/ty_wasm"
    }
    check(tyLicenseFile.get().asFile.isFile) {
      "Ruff revision $tyWasmCommit does not contain its root LICENSE"
    }
    check(typeshedLicenseFile.get().asFile.isFile) {
      "Ruff revision $tyWasmCommit does not contain the vendored typeshed LICENSE"
    }

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
    tyWasmPackageDir.map { it.file("ty_wasm_bg.wasm") },
    tyWasmPackageDir.map { it.file("ty_wasm.d.ts") },
    tyWasmPackageDir.map { it.file("package.json") }
  )

  doFirst {
    val sourceDir = tyWasmSourceDir.get().asFile
    check(sourceDir.resolve(".git").isDirectory) {
      "Pinned ty_wasm source is missing; rerun checkoutPinnedTyWasmSource"
    }
    val actualCommit = runChecked(sourceDir, "git", "rev-parse", "HEAD")
    check(actualCommit == tyWasmCommit) {
      "Refusing to build ty_wasm from $actualCommit; expected $tyWasmCommit"
    }
    project.delete(tyWasmPackageDir)
  }

  doLast {
    val packageDir = tyWasmPackageDir.get().asFile
    val module = packageDir.resolve("ty_wasm.js")
    val wasm = packageDir.resolve("ty_wasm_bg.wasm")
    check(module.isFile && module.readText().contains("ty_wasm_bg.wasm")) {
      "Generated ty_wasm JavaScript does not reference its sibling WebAssembly module"
    }
    val magic = wasm.inputStream().use { it.readNBytes(4) }
    check(magic.contentEquals(byteArrayOf(0x00, 0x61, 0x73, 0x6d))) {
      "Generated ty_wasm_bg.wasm is not a WebAssembly binary"
    }
  }
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

  doLast {
    val resourceDir = generatedTyWasmResources.get().asFile
    val module = resourceDir.resolve("ty_wasm.js")
    val wasm = resourceDir.resolve("ty_wasm_bg.wasm")
    check(module.isFile && wasm.isFile) { "Generated ty_wasm browser resources are incomplete" }
    val magic = wasm.inputStream().use { it.readNBytes(4) }
    check(magic.contentEquals(byteArrayOf(0x00, 0x61, 0x73, 0x6d))) {
      "Staged ty_wasm_bg.wasm is not a WebAssembly binary"
    }
  }
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
          devServer = (devServer ?: KotlinWebpackConfig.DevServer()).apply {
            open = "python3.html"
          }
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
      dependencies {
        implementation(kotlin("stdlib"))
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2026.8.0")
        implementation(npm("vanilla-monaco-editor", "npm:monaco-editor@0.55.1"))
      }
    }

    getByName("jsTest") {
      dependencies {
        implementation(kotlin("test-js"))
      }
    }
  }
}

tasks.named("jsProcessResources") {
  dependsOn(stageTyWasmResources)
}

tasks.withType<KotlinWebpack>().configureEach {
  dependsOn(prepareMonacoWebpackConfig)
}

val pythonProductionBundleDir = layout.buildDirectory.dir("kotlin-webpack/js/productionExecutable")
val pythonProductionJsFile = pythonProductionBundleDir.map { it.file("tidyparse-python.js").asFile }
val pythonProductionJsMapFile = pythonProductionBundleDir.map { it.file("tidyparse-python.js.map").asFile }
val pythonHtmlFile = layout.projectDirectory.file("src/jsMain/resources/python3.html")
val pythonCssFile = layout.projectDirectory.file("src/jsMain/resources/python3.css")
val pythonRunnerWorkerFile = layout.projectDirectory.file("src/jsMain/resources/python-runner-worker.js")
val tyWasmLoaderFile = layout.projectDirectory.file("src/jsMain/resources/ty-wasm-loader.js")
val stagedTyWasmJsFile = generatedTyWasmResources.map { it.file("ty_wasm.js").asFile }
val stagedTyWasmBinaryFile = generatedTyWasmResources.map { it.file("ty_wasm_bg.wasm").asFile }
val pythonDeployStagingDir = layout.buildDirectory.dir("python-deploy")

val preparePythonDeploy = tasks.register<Sync>("preparePythonDeploy") {
  group = "deployment"
  description = "Stages the standalone Python 3 playground for deployment to tidyparse.github.io"

  dependsOn("jsBrowserProductionWebpack", stageTyWasmResources)

  into(pythonDeployStagingDir)
  from("src/jsMain/resources") {
    include("python3.html")
    include("python3.css")
    include("python-runner-worker.js")
    include("ty-wasm-loader.js")
  }
  from(generatedTyWasmResources) {
    include("ty_wasm.js")
    include("ty_wasm_bg.wasm")
  }
  from(pythonProductionBundleDir) {
    include("tidyparse-python.js")
    include("tidyparse-python.js.map")
  }
  from(tyLicenseFile) {
    rename { "ty-LICENSE" }
  }
  from(typeshedLicenseFile) {
    rename { "typeshed-LICENSE" }
  }

  inputs.files(
    pythonHtmlFile,
    pythonCssFile,
    pythonRunnerWorkerFile,
    tyWasmLoaderFile,
    pythonProductionJsFile,
    pythonProductionJsMapFile,
    stagedTyWasmJsFile,
    stagedTyWasmBinaryFile,
    tyLicenseFile,
    typeshedLicenseFile
  )
  outputs.files(
    pythonDeployStagingDir.map { it.file("python3.html") },
    pythonDeployStagingDir.map { it.file("python3.css") },
    pythonDeployStagingDir.map { it.file("python-runner-worker.js") },
    pythonDeployStagingDir.map { it.file("ty-wasm-loader.js") },
    pythonDeployStagingDir.map { it.file("ty_wasm.js") },
    pythonDeployStagingDir.map { it.file("ty_wasm_bg.wasm") },
    pythonDeployStagingDir.map { it.file("tidyparse-python.js") },
    pythonDeployStagingDir.map { it.file("tidyparse-python.js.map") },
    pythonDeployStagingDir.map { it.file("ty-LICENSE") },
    pythonDeployStagingDir.map { it.file("typeshed-LICENSE") }
  )

  doLast {
    val stagingDir = pythonDeployStagingDir.get().asFile
    val requiredFiles = listOf(
      "python3.html",
      "python3.css",
      "python-runner-worker.js",
      "ty-wasm-loader.js",
      "ty_wasm.js",
      "ty_wasm_bg.wasm",
      "tidyparse-python.js",
      "tidyparse-python.js.map",
      "ty-LICENSE",
      "typeshed-LICENSE"
    )
    requiredFiles.forEach { relativePath ->
      val stagedFile = stagingDir.resolve(relativePath)
      check(stagedFile.isFile && stagedFile.length() > 0) {
        "Python deployment is missing required file: $relativePath"
      }
    }

    val html = stagingDir.resolve("python3.html").readText()
    val loaderScriptIndex = html.indexOf("src=\"ty-wasm-loader.js\"")
    val applicationScriptIndex = html.indexOf("src=\"tidyparse-python.js\"")
    check(
      "href=\"python3.css\"" in html &&
        loaderScriptIndex >= 0 &&
        applicationScriptIndex > loaderScriptIndex
    ) {
      "Staged python3.html must load python3.css, then ty-wasm-loader.js before tidyparse-python.js"
    }
    val tyWasmLoader = stagingDir.resolve("ty-wasm-loader.js").readText()
    check(
      "tidyparseTyWasmReady" in tyWasmLoader &&
        "import(\"./ty_wasm.js\")" in tyWasmLoader &&
        ".default()" in tyWasmLoader
    ) {
      "Staged ty-wasm-loader.js must expose and initialize the native ty_wasm browser import"
    }
    val applicationBundle = stagingDir.resolve("tidyparse-python.js").readText()
    check("python-runner-worker.js" in applicationBundle) {
      "Staged Python bundle must reference python-runner-worker.js"
    }
    check("tidyparseTyWasmReady" in applicationBundle) {
      "Staged Python bundle must consume the external ty_wasm readiness promise"
    }
    check("./kotlin lazy recursive" !in applicationBundle) {
      "Staged Python bundle must preserve ty_wasm.js as a native browser import, not a webpack context"
    }
    check(stagingDir.resolve("ty_wasm.js").readText().contains("ty_wasm_bg.wasm")) {
      "Staged ty_wasm.js must load the sibling ty_wasm_bg.wasm binary"
    }
    check(stagingDir.resolve("ty-LICENSE").readBytes().contentEquals(tyLicenseFile.get().asFile.readBytes())) {
      "Staged ty-LICENSE does not match the pinned Ruff source license"
    }
    check(
      stagingDir.resolve("typeshed-LICENSE").readBytes()
        .contentEquals(typeshedLicenseFile.get().asFile.readBytes())
    ) {
      "Staged typeshed-LICENSE does not match the pinned vendored typeshed license"
    }
    val wasmMagic = stagingDir.resolve("ty_wasm_bg.wasm").inputStream().use {
      it.readNBytes(4)
    }
    check(wasmMagic.contentEquals(byteArrayOf(0x00, 0x61, 0x73, 0x6d))) {
      "Staged ty_wasm_bg.wasm is not a WebAssembly binary"
    }
  }
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
