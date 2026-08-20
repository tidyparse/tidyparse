import buildlogic.GenerateCppStatementGrammar
import buildlogic.ManagedSiteDeployTask
import com.strumenta.antlrkotlin.gradle.AntlrKotlinTask
import groovy.json.JsonSlurper
import org.gradle.api.tasks.testing.logging.TestExceptionFormat
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.provider.ValueSource
import org.gradle.api.provider.ValueSourceParameters
import org.gradle.api.tasks.CacheableTask
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.TaskAction
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.targets.js.testing.KotlinJsTest
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpack
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT
import org.jetbrains.kotlin.gradle.tasks.BaseKotlinCompile
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.util.*
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference

plugins {
  kotlin("multiplatform")
  alias(libs.plugins.antlrKotlin)
}

group = "ai.hypergraph"
version = "0.23.0"

val monacoWebpackConfigDir = layout.buildDirectory.dir("generated/monaco-webpack-config")
val monacoWebpackConfig = """
  const webpack = require("webpack");
  // This output is evaluated in both Window and Worker globals. With all lazy
  // chunks folded in, the worker target's self.location base URI is safe in both.
  config.target = "webworker";
  config.plugins.push(new webpack.optimize.LimitChunkCountPlugin({ maxChunks: 1 }));
  config.module.rules.push({
    test: /\.(ttf|woff2?|eot)$/i,
    type: "asset/inline"
  });
  // VS Code extensions describe grammars, snippets, and language
  // configuration with new URL(..., import.meta.url). Keep those dependency
  // resources in the single GitHub Pages bundle as data URLs.
  config.module.rules.push({
    dependency: "url",
    type: "asset/inline",
    generator: {
      dataUrl: {
        encoding: "base64",
        mimetype: "application/octet-stream"
      }
    }
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

val clangdArtifactBaseVersion = "llvm-21.1.0-emsdk-4.0.22-wasi-29.0-r5"
val clangdHostId = listOf(
  System.getProperty("os.name"),
  System.getProperty("os.arch")
).joinToString("-")
  .lowercase(Locale.ROOT)
  .replace(Regex("[^a-z0-9._-]+"), "-")
val clangdRecipeDir = layout.projectDirectory.dir("clangd")
val cppSemanticProfileFile = clangdRecipeDir.file("semantic-profile.json")
val clangdRecipeFiles = fileTree(clangdRecipeDir) {
  exclude("**/.DS_Store")
}
val clangdRecipeSha256 = providers.of(ClangdRecipeSha256Source::class) {
  parameters.directory.set(clangdRecipeDir)
}
val clangdRecipeSha256Value = clangdRecipeSha256.get()
val clangdArtifactVersion = "$clangdArtifactBaseVersion-$clangdRecipeSha256Value"
// Project `.gradle` is owned by Gradle's cache cleanup and can be removed by another daemon while
// this external CMake build is running. Keep the expensive recipe-keyed toolchain in a dedicated
// Gradle-user-home sibling instead; it survives project `clean` without being project-cache state.
val clangdCacheDir = gradle.gradleUserHomeDir.resolve("tidyparse-clangd")
val clangdStateDir = layout.dir(providers.provider {
  clangdCacheDir.resolve("$clangdArtifactVersion-$clangdHostId")
}).get()
val clangdWorkDir = clangdStateDir.dir("work")
val clangdArtifactDir = clangdStateDir.dir("artifacts")
val clangdResourceDir = layout.projectDirectory.dir("src/jsMain/resources")
val generatedClangdVersionDir = layout.buildDirectory.dir("generated/clangd-version")
val generatedCppSemanticProfileDir = layout.buildDirectory.dir("generated/cpp-semantic-profile")
val generateClangdArtifactVersion = tasks.register<GenerateClangdArtifactVersion>(
  "generateClangdArtifactVersion"
) {
  artifactVersion.set(clangdArtifactVersion)
  outputFile.set(generatedClangdVersionDir.map { it.file("JSClangdArtifactVersion.kt") })
}
val generateCppSemanticProfile = tasks.register("generateCppSemanticProfile") {
  val outputFile = generatedCppSemanticProfileDir.map { it.file("CppSemanticProfile.generated.kt") }
  inputs.file(cppSemanticProfileFile)
  outputs.file(outputFile)

  doLast {
    @Suppress("UNCHECKED_CAST")
    val profile = JsonSlurper().parse(cppSemanticProfileFile.asFile) as Map<String, Any?>
    require((profile["schemaVersion"] as? Number)?.toInt() == 1) {
      "clangd/semantic-profile.json must use schemaVersion 1"
    }
    require(profile["language"] == "c++" && profile["standard"] == "c++23") {
      "clangd/semantic-profile.json must describe the browser C++23 translation unit"
    }
    require(profile["target"] == "wasm32-wasi") {
      "clangd/semantic-profile.json must target wasm32-wasi"
    }
    val flags = (profile["flags"] as? List<*>)?.map {
      require(it is String) { "Every semantic profile flag must be a string" }
      it
    } ?: error("clangd/semantic-profile.json is missing flags")
    require(flags.firstOrNull() == "-xc++") {
      "The C++ semantic profile must begin with -xc++"
    }
    val source = flags.joinToString(
      prefix = "// Generated from clangd/semantic-profile.json.\n" +
              "internal val CPP_CLANGD_CPP_SEMANTIC_FLAGS = arrayOf(\n",
      separator = ",\n",
      postfix = "\n)\n"
    ) { flag -> "  \"${flag.replace("\\", "\\\\").replace("\"", "\\\"")}\"" }
    outputFile.get().asFile.apply {
      parentFile.mkdirs()
      if (!exists() || readText() != source) writeText(source)
    }
  }
}

val generatedCppStatementGrammarDir = layout.buildDirectory.dir("generated/cpp-statement-grammar")
val generateCppStatementGrammar = tasks.register<GenerateCppStatementGrammar>(
  "generateCppStatementGrammar"
) {
  parserGrammar.set(layout.projectDirectory.file("grammar/cpp/CPP14Parser.g4"))
  lexerGrammar.set(layout.projectDirectory.file("antlr/cpp/CPP14Lexer.g4"))
  expectedParserSha256.set("628062e9f75710ba1d1436ced8bd7d9d8f2f08c31a6e962c175e06b28994ff27")
  expectedLexerSha256.set("739a8782e05279318dccab76bf05af1ff5e3ff9e43f1b5b0d04e14d91d4fff47")
  outputFile.set(generatedCppStatementGrammarDir.map { it.file("cppcompletion/Cpp14StatementGrammar.generated.kt") })
}

val buildClangdWasm = tasks.register<Exec>("buildClangdWasm") {
  group = "build"
  description = "Builds the pinned, self-hosted clangd WebAssembly artifact"

  workingDir(clangdRecipeDir)
  val buildScript = clangdRecipeDir.file("build.sh").asFile.absolutePath
  val buildLock = clangdCacheDir.resolve("browser-clangd-build.lock").absolutePath
  // Separate Gradle daemons do not coordinate task output locks. Hold one global OS lock while a
  // configured recipe reads the live patch/build inputs and mutates its persistent CMake tree;
  // fcntl releases it automatically if the wrapper exits or is killed.
  commandLine(
    "python3",
    "-c",
    """
      import fcntl, hashlib, pathlib, subprocess, sys
      def recipe_digest(root):
          root = pathlib.Path(root)
          digest = hashlib.sha256(b"tidyparse-clangd-recipe-v1\0")
          files = sorted(
              (path for path in root.rglob("*") if path.is_file() and path.name != ".DS_Store"),
              key=lambda path: path.relative_to(root).as_posix(),
          )
          for path in files:
              digest.update(path.relative_to(root).as_posix().encode())
              digest.update(b"\0")
              digest.update(hashlib.sha256(path.read_bytes()).digest())
          return digest.hexdigest()
      with open(sys.argv[1], "w") as lock:
          print("Waiting for browser clangd build lock:", sys.argv[1], flush=True)
          fcntl.flock(lock, fcntl.LOCK_EX)
          print("Acquired browser clangd build lock", flush=True)
          if recipe_digest(sys.argv[3]) != sys.argv[4]:
              raise SystemExit("clangd recipe changed before the locked build; rerun Gradle")
          result = subprocess.run(["bash", sys.argv[2]])
          if recipe_digest(sys.argv[3]) != sys.argv[4]:
              raise SystemExit("clangd recipe changed during the locked build; refusing its output")
          raise SystemExit(result.returncode)
    """.trimIndent(),
    buildLock,
    buildScript,
    clangdRecipeDir.asFile.absolutePath,
    clangdRecipeSha256Value
  )
  environment("ROOT_DIR", clangdWorkDir.asFile.absolutePath)
  environment("OUTPUT_DIR", clangdArtifactDir.asFile.absolutePath)
  environment("CLANGD_RECIPE_SHA256", clangdRecipeSha256Value)

  inputs.files(clangdRecipeFiles).withPathSensitivity(PathSensitivity.RELATIVE)
  inputs.property("clangdRecipeSha256", clangdRecipeSha256)
  inputs.property("clangdArtifactVersion", clangdArtifactVersion)
  inputs.property("clangdHost", clangdHostId)
  outputs.files(
    clangdArtifactDir.file("clangd.js"),
    clangdArtifactDir.file("clangd.wasm.gz"),
    clangdArtifactDir.file("clangd-manifest.json")
  )
  outputs.file(clangdWorkDir.file("build-native/bin/clang++"))
  outputs.file(clangdWorkDir.file("native-validator-profile.json"))
  outputs.dir(clangdWorkDir.dir("browser-sysroot/include"))

  doFirst {
    check(clangdRecipeDigest(clangdRecipeDir.asFile) == clangdRecipeSha256Value) {
      "The clangd recipe changed after this Gradle invocation was configured; rerun the task"
    }
    clangdWorkDir.asFile.mkdirs()
    clangdArtifactDir.asFile.mkdirs()
  }
  doLast {
    check(clangdRecipeDigest(clangdRecipeDir.asFile) == clangdRecipeSha256Value) {
      "The clangd recipe changed while its artifact was building; refusing to publish it"
    }
    listOf("clangd.js", "clangd.wasm.gz", "clangd-manifest.json").forEach { name ->
      val artifact = clangdArtifactDir.file(name).asFile
      check(artifact.isFile && artifact.length() > 0) {
        "The clangd build did not produce ${artifact.absolutePath}"
      }
    }
    listOf(
      clangdWorkDir.file("build-native/bin/clang++").asFile,
      clangdWorkDir.file("native-validator-profile.json").asFile
    ).forEach { artifact ->
      check(artifact.isFile && artifact.length() > 0) { "The clangd build did not produce ${artifact.absolutePath}" }
    }
    check(clangdWorkDir.dir("browser-sysroot/include").asFile.isDirectory) {
      "The clangd build did not retain its semantic include tree"
    }
  }
}

val refreshClangdResources = tasks.register("refreshClangdResources") {
  group = "build"
  description = "Rebuilds and refreshes the ignored clangd browser resources"
  dependsOn(buildClangdWasm)

  val artifactModule = clangdArtifactDir.file("clangd.js")
  val artifactWasm = clangdArtifactDir.file("clangd.wasm.gz")
  val artifactManifest = clangdArtifactDir.file("clangd-manifest.json")
  val resourceModule = clangdResourceDir.file("clangd.js")
  val resourceWasm = clangdResourceDir.file("clangd.wasm.gz")
  val resourceManifest = clangdResourceDir.file("clangd-manifest.json")
  val publicationLock = clangdCacheDir.resolve("browser-clangd-publish.lock")
  inputs.files(artifactModule, artifactWasm, artifactManifest)
  inputs.property("clangdRecipeSha256", clangdRecipeSha256Value)
  outputs.files(resourceModule, resourceWasm, resourceManifest)

  doLast {
    check(clangdRecipeDigest(clangdRecipeDir.asFile) == clangdRecipeSha256Value) {
      "The clangd recipe changed before resource publication; rerun the task"
    }
    val sourceManifest = JsonSlurper().parse(artifactManifest.asFile) as Map<*, *>
    check(sourceManifest["artifactVersion"] == clangdArtifactVersion) {
      "Built browser clangd manifest does not match $clangdArtifactVersion"
    }

    clangdResourceDir.asFile.mkdirs()
    publicationLock.parentFile.mkdirs()
    FileChannel.open(
      publicationLock.toPath(),
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    ).use { channel ->
      channel.lock().use {
        check(clangdRecipeDigest(clangdRecipeDir.asFile) == clangdRecipeSha256Value) {
          "The clangd recipe changed while waiting to publish resources; rerun the task"
        }

        val sources = listOf(artifactModule, artifactWasm, artifactManifest)
        val targets = listOf(resourceModule, resourceWasm, resourceManifest)
        val temporaries = targets.map { target ->
          target.asFile.toPath().resolveSibling(
            ".${target.asFile.name}.${clangdRecipeSha256Value.take(12)}.tmp"
          )
        }
        fun atomicMove(source: java.nio.file.Path, target: java.nio.file.Path) {
          try {
            Files.move(
              source,
              target,
              StandardCopyOption.ATOMIC_MOVE,
              StandardCopyOption.REPLACE_EXISTING
            )
          } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING)
          }
        }

        try {
          sources.zip(temporaries).forEach { (source, temporary) ->
            Files.copy(source.asFile.toPath(), temporary, StandardCopyOption.REPLACE_EXISTING)
          }
          // The manifest is the commit marker. Remove it before changing either payload and
          // publish the replacement only after both payload moves and a final live-recipe check.
          Files.deleteIfExists(resourceManifest.asFile.toPath())
          atomicMove(temporaries[0], resourceModule.asFile.toPath())
          atomicMove(temporaries[1], resourceWasm.asFile.toPath())
          check(clangdRecipeDigest(clangdRecipeDir.asFile) == clangdRecipeSha256Value) {
            "The clangd recipe changed during resource publication; bundle left uncommitted"
          }
          atomicMove(temporaries[2], resourceManifest.asFile.toPath())
        } finally {
          temporaries.forEach(Files::deleteIfExists)
        }
      }
    }
  }
}

val verifyClangdResources = tasks.register("verifyClangdResources") {
  group = "verification"
  description = "Verifies that the staged browser clangd bundle matches the current recipe"
  dependsOn(refreshClangdResources)

  val module = clangdResourceDir.file("clangd.js")
  val wasm = clangdResourceDir.file("clangd.wasm.gz")
  val manifestFile = clangdResourceDir.file("clangd-manifest.json")
  inputs.files(module, wasm, manifestFile)
  inputs.property("clangdArtifactVersion", clangdArtifactVersion)

  doLast {
    @Suppress("UNCHECKED_CAST")
    val manifest = JsonSlurper().parse(manifestFile.asFile) as Map<String, Any?>
    check(manifest["artifactVersion"] == clangdArtifactVersion) {
      "Staged browser clangd has artifact version ${manifest["artifactVersion"]}; " +
              "expected $clangdArtifactVersion"
    }
    val artifacts = manifest["artifacts"] as? Map<*, *>
      ?: error("Staged browser clangd manifest has no artifacts")
    val javascript = artifacts["clangd.js"] as? Map<*, *>
      ?: error("Staged browser clangd manifest has no clangd.js entry")
    val webAssembly = artifacts["clangd.wasm"] as? Map<*, *>
      ?: error("Staged browser clangd manifest has no clangd.wasm entry")

    fun sha256(file: File): String {
      val digest = MessageDigest.getInstance("SHA-256")
      file.inputStream().buffered().use { input ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
          val read = input.read(buffer)
          if (read < 0) break
          digest.update(buffer, 0, read)
        }
      }
      return HexFormat.of().formatHex(digest.digest())
    }

    fun verify(file: File, bytes: Any?, digest: Any?, label: String) {
      check((bytes as? Number)?.toLong() == file.length()) { "$label size does not match the staged browser clangd manifest" }
      check(digest == sha256(file)) { "$label digest does not match the staged browser clangd manifest" }
    }

    verify(module.asFile, javascript["bytes"], javascript["sha256"], "clangd.js")
    verify(
      wasm.asFile,
      webAssembly["compressedBytes"],
      webAssembly["compressedSha256"],
      "clangd.wasm.gz"
    )
  }
}

val antlrPackageName = "com.strumenta.antlrkotlin.parsers.generated"
val generatedAntlrDir = layout.buildDirectory.dir("generatedAntlr")
val generateKotlinGrammarSource = tasks.register<AntlrKotlinTask>("generateKotlinGrammarSource") {
  source = fileTree(layout.projectDirectory.dir("antlr")) { include("**/*.g4") }
  packageName = antlrPackageName
  arguments = listOf("-visitor")
  outputDirectory = generatedAntlrDir
    .map { it.dir(antlrPackageName.replace(".", "/")) }
    .get().asFile
}

kotlin {
  jvm {
    compilerOptions.jvmTarget = JvmTarget.JVM_21
  }

  js {
    binaries.executable()

    browser {
      commonWebpackConfig {
        configDirectory = monacoWebpackConfigDir.get().asFile
        cssSupport { enabled.set(true) }
      }

      runTask {
        mainOutputFileName = "tidyparse-cpp.js"
        webpackConfigApplier {
          devServer = (devServer ?: KotlinWebpackConfig.DevServer()).apply {
            open = "cpp.html"
          }
        }
      }

      webpackTask {
        // We need this to work on Chrome when deployed due to the PLATFORM_CALLER_STACKTRACE_DEPTH hack
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-cpp.js"
        devtool = "source-map" // For debugging; remove for production
      }

      testTask {
        useKarma { useChromeHeadless() }
        if (System.getenv("CPP_COMPLETION_BENCHMARK") == "1") {
          // The compiler-backed sweep is intentionally isolated from the fast grammar/service
          // regressions. Benchmark mode should measure the full discovered completion corpus, not rerun the
          // surrounding unit suite before starting its one-minute clock.
          filter.includeTestsMatching("cppcompletion.CppCompletionBenchmarkTest.benchmarkCppCompletions")
        }
      }
    }
  }

  sourceSets {
    getByName("commonMain") {
      dependencies {
        implementation(project(":tidyparse-core"))
        implementation("com.ionspin.kotlin:bignum:0.3.10")
      }
    }

    getByName("commonTest") {
      dependencies {
        implementation(kotlin("test"))
      }
    }

    getByName("jsMain") {
      kotlin.srcDir(generatedClangdVersionDir)
      kotlin.srcDir(generatedCppSemanticProfileDir)
      kotlin.srcDir(generatedCppStatementGrammarDir)
      kotlin.srcDir(generatedAntlrDir)
      dependencies {
        implementation(libs.antlrKotlinRuntime)
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2026.6.3")
        // Keep this family pinned together. monaco-languageclient 10 is the
        // maintained successor to the now-discontinued monaco-editor-wrapper
        implementation(npm("monaco-editor", "npm:@codingame/monaco-vscode-editor-api@25.1.2"))
        implementation(npm("vscode", "npm:@codingame/monaco-vscode-extension-api@25.1.2"))
        implementation(npm("monaco-languageclient", "10.7.0"))
        implementation(npm("vscode-languageclient", "9.0.1"))
        implementation(npm("@codingame/monaco-vscode-configuration-service-override", "25.1.2"))
        implementation(npm("@codingame/monaco-vscode-textmate-service-override", "25.1.2"))
        implementation(npm("@codingame/monaco-vscode-theme-defaults-default-extension", "25.1.2"))
        implementation(npm("@codingame/monaco-vscode-cpp-default-extension", "25.1.2"))
      }
    }

    getByName("jsTest") {
      kotlin.srcDir("src/jsTest/cppCompletion/kotlin")
      resources.srcDir("src/jsTest/cppCompletion/resources")
      dependencies {
        implementation(kotlin("test-js"))
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.11.0")
        implementation("com.ionspin.kotlin:bignum:0.3.10")
      }
    }
  }
}

tasks.withType<BaseKotlinCompile>().configureEach {
  dependsOn(generateKotlinGrammarSource, generateCppStatementGrammar, generateCppSemanticProfile)
}
tasks.named("compileKotlinJs") { dependsOn(generateClangdArtifactVersion) }
tasks.named("jsProcessResources") { mustRunAfter(refreshClangdResources) }

if (System.getenv("CPP_COMPLETION_BENCHMARK") == "1") {
  // The benchmark and cpp.html both consume the same recipe-keyed worker bundle. Make the
  // benchmark command self-contained: regenerate stale artifacts, publish them before resource
  // processing, compile the worker served by the benchmark middleware, and verify their manifest
  // rather than relying on an earlier manual refresh or development build.
  tasks.named("jsBrowserTest") {
    dependsOn(verifyClangdResources, "jsDevelopmentExecutableCompileSync")
    doFirst {
      val worker = rootProject.layout.buildDirectory.file(
        "js/packages/${rootProject.name}-${project.name}/kotlin/" +
                "${rootProject.name}-${project.name}.js"
      ).get().asFile
      check(worker.isFile && worker.readText().contains(clangdArtifactVersion)) {
        "Browser clangd worker and semantic manifest use different artifact versions"
      }
    }
  }
}

tasks.withType<KotlinWebpack>().configureEach {
  dependsOn(prepareMonacoWebpackConfig)
  if (!name.contains("test", ignoreCase = true)) {
    // Development/production browser bundles back cpp.html and must never pair a newly compiled
    // recipe-keyed worker with stale ignored clangd payloads. Test webpack stays fast; the
    // compiler-backed benchmark opts into the same verification explicitly above.
    dependsOn(verifyClangdResources)
  }
}

val cppProductionBundleDir = layout.buildDirectory.dir("kotlin-webpack/js/productionExecutable")
val cppProductionJsFile = cppProductionBundleDir.map { it.file("tidyparse-cpp.js").asFile }
val cppProductionJsMapFile = cppProductionBundleDir.map { it.file("tidyparse-cpp.js.map").asFile }
val cppDeployStagingDir = layout.buildDirectory.dir("cpp-deploy")

tasks {
  val browserConsoleTailService = gradle.sharedServices.registerIfAbsent(
    "cppBrowserConsoleTailService",
    CppBrowserConsoleTailService::class
  ) {}

  withType<KotlinJsTest>().configureEach {
    val testTaskPath = path
    val browserConsoleTailProcess = AtomicReference<Process?>()
    val streamBrowserConsole = System.getenv("CPP_COMPLETION_BENCHMARK") != "1"
    // Gradle otherwise considers two diagnostic benchmark ranges the same up-to-date invocation.
    // These values affect selection/scoring but not the compiled test executable.
    inputs.property("cppCompletionBenchmark", System.getenv("CPP_COMPLETION_BENCHMARK") ?: "")
    inputs.property("cppCompletionStart", System.getenv("CPP_COMPLETION_START_INSTANCE") ?: "")
    inputs.property("cppCompletionMax", System.getenv("CPP_COMPLETION_MAX_INSTANCES") ?: "")
    inputs.property("cppCompletionSamples", System.getenv("CPP_COMPLETION_SAMPLES_PER_INSTANCE") ?: "")
    inputs.property("cppCompletionDeadline", System.getenv("CPP_COMPLETION_TIME_LIMIT_MS") ?: "")
    usesService(browserConsoleTailService)

    testLogging {
      showStandardStreams = true
      showExceptions = true
      showCauses = true
      showStackTraces = true
      exceptionFormat = TestExceptionFormat.FULL
      events("passed", "skipped", "failed", "standardOut", "standardError")
    }

    doFirst {
      if (streamBrowserConsole) {
        val browserConsoleLog = rootProject.layout.buildDirectory.file("ci-logs/browser-console.log").get().asFile
        browserConsoleLog.parentFile.mkdirs()
        browserConsoleLog.writeText("")

        val tailProcess = ProcessBuilder("tail", "-n", "+1", "-f", browserConsoleLog.absolutePath)
          .redirectErrorStream(true)
          .start()

        browserConsoleTailService.get().stop(browserConsoleTailProcess.getAndSet(tailProcess))
        browserConsoleTailService.get().register(tailProcess)
        Thread {
          tailProcess.inputStream.bufferedReader().useLines { lines ->
            lines.forEach { println(it) }
          }
        }.apply {
          name = "browser-console-tail-$testTaskPath"
          isDaemon = true
          start()
        }
      }
    }

    fun stopBrowserConsoleTail() {
      browserConsoleTailService.get().stop(browserConsoleTailProcess.getAndSet(null))
    }

    doLast { stopBrowserConsoleTail() }
  }

  val prepareCppDeploy = register<Sync>("prepareCppDeploy") {
    group = "deployment"
    description = "Stages the C++ playground files for deployment to tidyparse.github.io"

    dependsOn("jsBrowserProductionWebpack", refreshClangdResources)

    into(cppDeployStagingDir)
    from("src/jsMain/resources") {
      exclude(".DS_Store")
      exclude("**/.DS_Store")
    }
    from(cppProductionBundleDir) {
      include("tidyparse-cpp.js")
      include("tidyparse-cpp.js.map")
    }

    inputs.files(cppProductionJsFile, cppProductionJsMapFile)
    outputs.files(
      cppDeployStagingDir.map { it.file("cpp.html") },
      cppDeployStagingDir.map { it.file("tidyparse-cpp.js") },
      cppDeployStagingDir.map { it.file("tidyparse-cpp.js.map") },
      cppDeployStagingDir.map { it.file("clangd.js") },
      cppDeployStagingDir.map { it.file("clangd.wasm.gz") },
      cppDeployStagingDir.map { it.file("clangd-manifest.json") },
      cppDeployStagingDir.map { it.file("examples/c/cpp_statements.tidy") }
    )
  }

  register<ManagedSiteDeployTask>("deployCpp") {
    group = "deployment"
    description = "Builds, commits, and pushes the C++ playground to tidyparse.github.io. Requires --msg \"commit message\"."

    dependsOn(prepareCppDeploy)

    sourceDirectory.set(cppDeployStagingDir)
    deploymentId.set("cpp")
    commitMessage.convention(providers.gradleProperty("deployCppMessage"))
    repositoryUrl.convention(
      providers.gradleProperty("deployCppRepoUrl")
        .orElse("https://github.com/tidyparse/tidyparse.github.io.git")
    )
    pushUrl.convention(
      providers.gradleProperty("deployCppPushUrl")
        .orElse("git@github.com:tidyparse/tidyparse.github.io.git")
    )
    branch.convention(providers.gradleProperty("deployCppBranch").orElse("main"))
    checkoutPath.convention(
      providers.gradleProperty("deployCppRepoDir")
        .orElse(layout.buildDirectory.dir("deploy/tidyparse.github.io").map { it.asFile.absolutePath })
    )
  }
}

// To run the C++ playground locally:
// ./gradlew :tidyparse-cpp:jsBrowserDevelopmentRun --continuous
//
// To deploy only the C++-owned website resources:
// ./gradlew :tidyparse-cpp:deployCpp --msg "update C++ playground"

abstract class CppBrowserConsoleTailService : BuildService<BuildServiceParameters.None>, AutoCloseable {
  private val processes = CopyOnWriteArrayList<Process>()

  fun register(process: Process) = processes.add(process)

  fun stop(process: Process?) {
    if (process == null) return
    process.destroy()
    processes.remove(process)
  }

  override fun close() {
    processes.forEach { it.destroy() }
    processes.clear()
  }
}

abstract class ClangdRecipeSha256Source :
  ValueSource<String, ClangdRecipeSha256Source.Parameters> {
  interface Parameters : ValueSourceParameters {
    val directory: DirectoryProperty
  }

  override fun obtain(): String {
    val root = parameters.directory.get().asFile
    val files = root.walkTopDown()
      .filter { it.isFile && it.name != ".DS_Store" }
      .map { file -> file.relativeTo(root).invariantSeparatorsPath to file }
      .sortedBy { it.first }
      .toList()
    val recipeDigest = MessageDigest.getInstance("SHA-256")
    recipeDigest.update("tidyparse-clangd-recipe-v1\u0000".toByteArray())

    files.forEach { (relativePath, file) ->
      val fileDigest = MessageDigest.getInstance("SHA-256")
      file.inputStream().buffered().use { input ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
          val read = input.read(buffer)
          if (read < 0) break
          fileDigest.update(buffer, 0, read)
        }
      }
      recipeDigest.update(relativePath.toByteArray())
      recipeDigest.update(0.toByte())
      recipeDigest.update(fileDigest.digest())
    }

    return HexFormat.of().formatHex(recipeDigest.digest())
  }
}

private fun clangdRecipeDigest(root: File): String {
  val files = root.walkTopDown()
    .filter { it.isFile && it.name != ".DS_Store" }
    .map { file -> file.relativeTo(root).invariantSeparatorsPath to file }
    .sortedBy { it.first }
    .toList()
  val recipeDigest = MessageDigest.getInstance("SHA-256")
  recipeDigest.update("tidyparse-clangd-recipe-v1\u0000".toByteArray())

  files.forEach { (relativePath, file) ->
    val fileDigest = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered().use { input ->
      val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
      while (true) {
        val read = input.read(buffer)
        if (read < 0) break
        fileDigest.update(buffer, 0, read)
      }
    }
    recipeDigest.update(relativePath.toByteArray())
    recipeDigest.update(0.toByte())
    recipeDigest.update(fileDigest.digest())
  }

  return HexFormat.of().formatHex(recipeDigest.digest())
}

@CacheableTask
abstract class GenerateClangdArtifactVersion : DefaultTask() {
  @get:Input
  abstract val artifactVersion: Property<String>

  @get:OutputFile
  abstract val outputFile: RegularFileProperty

  @TaskAction
  fun generate() {
    val version = artifactVersion.get()
    require(version.matches(Regex("[a-zA-Z0-9._-]+"))) {
      "Invalid clangd artifact version: $version"
    }
    val source = """
      // Generated from the complete tidyparse-cpp/clangd recipe.
      // Keep this non-const: Kotlin/JS incremental compilation does not reliably
      // invalidate every consumer when a generated const value changes.
      internal val CPP_CLANGD_ARTIFACT_VERSION = "$version"
    """.trimIndent() + "\n"
    outputFile.get().asFile.apply {
      parentFile.mkdirs()
      if (!exists() || readText() != source) writeText(source)
    }
  }
}
