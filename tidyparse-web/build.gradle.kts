@file:OptIn(ExperimentalEncodingApi::class)

import buildlogic.GenerateCppStatementGrammar
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
import org.jetbrains.kotlin.gradle.targets.js.testing.KotlinJsTest
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpack
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT
import org.jetbrains.kotlin.gradle.tasks.BaseKotlinCompile
import org.jetbrains.letsPlot.*
import org.jetbrains.letsPlot.export.ggsave
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.label.ggtitle
import java.awt.Desktop
import java.io.ByteArrayOutputStream
import java.nio.channels.FileChannel
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.util.*
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference
import java.util.zip.GZIPOutputStream
import kotlin.io.encoding.*
import kotlin.io.encoding.Base64

buildscript {
  repositories { mavenCentral() }
  dependencies {
    classpath("org.jetbrains.lets-plot:platf-awt-jvm:4.4.1")
    classpath("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:4.14.0")
  }
}

plugins {
  kotlin("multiplatform")
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
  lexerGrammar.set(rootProject.layout.projectDirectory.file("tidyparse-core/antlr/cpp/CPP14Lexer.g4"))
  expectedParserSha256.set("628062e9f75710ba1d1436ced8bd7d9d8f2f08c31a6e962c175e06b28994ff27")
  expectedLexerSha256.set("739a8782e05279318dccab76bf05af1ff5e3ff9e43f1b5b0d04e14d91d4fff47")
  outputFile.set(generatedCppStatementGrammarDir.map {
    it.file("cppcompletion/Cpp14StatementGrammar.generated.kt")
  })
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
      check(artifact.isFile && artifact.length() > 0) {
        "The clangd build did not produce ${artifact.absolutePath}"
      }
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
      check((bytes as? Number)?.toLong() == file.length()) {
        "$label size does not match the staged browser clangd manifest"
      }
      check(digest == sha256(file)) {
        "$label digest does not match the staged browser clangd manifest"
      }
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

kotlin {
  js {
    binaries.executable()

    browser {
      commonWebpackConfig {
        configDirectory = monacoWebpackConfigDir.get().asFile
        cssSupport {
          enabled.set(true)
        }
      }

      runTask { mainOutputFileName = "tidyparse-web.js" }

      webpackTask {
        // We need this to work on Chrome when deployed due to the PLATFORM_CALLER_STACKTRACE_DEPTH hack
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-web.js"
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
    getByName("jsMain") {
      kotlin.srcDir(generatedClangdVersionDir)
      kotlin.srcDir(generatedCppSemanticProfileDir)
      kotlin.srcDir(generatedCppStatementGrammarDir)
      dependencies {
        implementation(project(":tidyparse-core"))
        implementation("com.ionspin.kotlin:bignum:0.3.10")
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
  dependsOn(generateCppStatementGrammar, generateCppSemanticProfile)
}

tasks.named("compileKotlinJs") {
  dependsOn(generateClangdArtifactVersion)
}

tasks.named("jsProcessResources") {
  mustRunAfter(refreshClangdResources)
}

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

fun saveStats(stat: String, name: String) =
  if (stat.isEmpty()) throw IllegalStateException("Total time not found in output")
  else file("$name.txt").also { if (!it.exists()) it.createNewFile() }
    .apply { println("$name:\t$stat"); appendText("${stat.trim().toLong()}\n") }

fun plotGrid(name: String, x_lbl: String = "x", y_lbl: String = "y"): Plot {
  val timings = file("$name.txt").readLines().mapNotNull { it.toDoubleOrNull() }
  val data = mapOf(x_lbl to (1..timings.size).toList(), y_lbl to timings)
  return letsPlot(data) { x = x_lbl; y = y_lbl } + geomLine() + ggtitle(name)
}

fun plotGrids(vararg names: String) {
  ggsave(gggrid(names.map { plotGrid(it) }, ncol = 3), "grid.svg", path = projectDir.absolutePath)
  Desktop.getDesktop().browse(file("grid.svg").toURI())
}

fun jsString(s: String): String =
  buildString {
    append('"')
    for (c in s) when (c) {
      '\\' -> append("\\\\")
      '"' -> append("\\\"")
      '\n' -> append("\\n")
      '\r' -> append("\\r")
      '\t' -> append("\\t")
      '\b' -> append("\\b")
      '\u000C' -> append("\\f")
      else -> if (c.code < 0x20) append("\\u%04x".format(c.code)) else append(c)
    }
    append('"')
  }

fun gzip(bytes: ByteArray): ByteArray {
  val out = ByteArrayOutputStream()
  GZIPOutputStream(out).use { it.write(bytes) }
  return out.toByteArray()
}

fun gzipBase64(bytes: ByteArray): String = Base64.encode(gzip(bytes))
fun gzipBase64(text: String): String = gzipBase64(text.toByteArray())

fun jsObjectLiteral(entries: Map<String, String>): String =
  "{${entries.entries.joinToString(",") { (key, value) -> "${jsString(key)}:${jsString(value)}" }}}"

val EMBEDDED_WEB_RESOURCES_START = "/* __TIDYPARSE_EMBEDDED_WEB_RESOURCES_START__ */"
val EMBEDDED_WEB_RESOURCES_END = "/* __TIDYPARSE_EMBEDDED_WEB_RESOURCES_END__ */"

fun embeddedWebResourcesScript(
  rawNgramsGzipB64: String,
  rawWdfaGzipB64: String,
  rawRerankerWeightsGzipB64: String,
  rawExamplesGzipB64: String
): String = """
$EMBEDDED_WEB_RESOURCES_START
globalThis.raw_ngrams_gzip_b64 = ${jsString(rawNgramsGzipB64)};
globalThis.raw_wdfa_gzip_b64 = ${jsString(rawWdfaGzipB64)};
globalThis.raw_reranker_weights_gzip_b64 = ${jsString(rawRerankerWeightsGzipB64)};
globalThis.raw_examples_gzip_b64 = ${jsString(rawExamplesGzipB64)};
$EMBEDDED_WEB_RESOURCES_END
""".trimIndent() + "\n"

fun String.withoutEmbeddedWebResources(): String =
  replace(
    Regex(
      "^${Regex.escape(EMBEDDED_WEB_RESOURCES_START)}.*?${Regex.escape(EMBEDDED_WEB_RESOURCES_END)}\\s*",
      RegexOption.DOT_MATCHES_ALL
    ),
    ""
  )

val productionBundleDir = layout.buildDirectory.dir("kotlin-webpack/js/productionExecutable")
val productionJsFile = productionBundleDir.map { it.file("tidyparse-web.js").asFile }
val productionJsMapFile = productionBundleDir.map { it.file("tidyparse-web.js.map").asFile }
val productionExecutableJsFile = layout.buildDirectory.file(
  "compileSync/js/main/productionExecutable/kotlin/${rootProject.name}-${project.name}.js"
)
val hostedCoreBundleDir = layout.buildDirectory.dir("kotlin-webpack/js/hostedCore")
val hostedCoreJsFile = hostedCoreBundleDir.map { it.file("tidyparse-core.js").asFile }
val hostedCoreJsMapFile = hostedCoreBundleDir.map { it.file("tidyparse-core.js.map").asFile }
val webDeployStagingDir = layout.buildDirectory.dir("web-deploy")

val ngramFile = layout.projectDirectory.file("src/jsMain/resources/python_4grams.txt").asFile
val wdfaFile = layout.projectDirectory.file("src/jsMain/resources/wdfa.bin").asFile
val rerankerWeightsFile = layout.projectDirectory.file("src/jsMain/resources/reranker_2000.q8.safetensors").asFile
val exampleFiles = rootProject.fileTree("examples") {
  include("**/*.tidy")
  include("**/*.txt")
}
val deployExampleFiles = rootProject.fileTree("examples") {
  exclude(".idea/**")
  exclude("**/.DS_Store")
}

fun exampleResourceMap(): Map<String, String> =
  exampleFiles.files
    .sortedBy { rootProject.projectDir.toPath().relativize(it.toPath()).toString() }
    .associate {
      rootProject.projectDir.toPath()
        .relativize(it.toPath())
        .toString()
        .replace(File.separatorChar, '/') to it.readText()
    }

fun embeddedRuntimeResources(includeExamples: Boolean = true): String =
  embeddedWebResourcesScript(
    rawNgramsGzipB64 = gzipBase64(ngramFile.readBytes()),
    rawWdfaGzipB64 = gzipBase64(wdfaFile.readBytes()),
    rawRerankerWeightsGzipB64 = gzipBase64(rerankerWeightsFile.readBytes()),
    rawExamplesGzipB64 = if (includeExamples) gzipBase64(jsObjectLiteral(exampleResourceMap())) else ""
  )

// Measure the ASCII payload actually added to JavaScript, after gzip/base64 encoding.
val maxHostedInlinePayloadBytes = 1L * 1024L * 1024L
val maxHostedInitialJavaScriptGzipBytes = 2L * 1024L * 1024L

fun hostedGzipBase64(bytes: ByteArray): String =
  gzipBase64(bytes).takeIf { it.length.toLong() <= maxHostedInlinePayloadBytes } ?: ""

fun hostedGzipBase64(text: String): String = hostedGzipBase64(text.toByteArray())

fun embeddedHostedIndexResources(): String =
  embeddedWebResourcesScript(
    rawNgramsGzipB64 = "",
    rawWdfaGzipB64 = "",
    rawRerankerWeightsGzipB64 = "",
    rawExamplesGzipB64 = hostedGzipBase64(jsObjectLiteral(exampleResourceMap()))
  )

fun embeddedHostedPythonResources(): String =
  embeddedWebResourcesScript(
    rawNgramsGzipB64 = hostedGzipBase64(ngramFile.readBytes()),
    rawWdfaGzipB64 = hostedGzipBase64(wdfaFile.readBytes()),
    rawRerankerWeightsGzipB64 = hostedGzipBase64(rerankerWeightsFile.readBytes()),
    rawExamplesGzipB64 = ""
  )

val generatedHostedWebpackDir = layout.buildDirectory.dir("generated/hosted-webpack")
val hostedWebpackConfigFile = generatedHostedWebpackDir.map { it.file("webpack.config.js") }
val generatedWebpackPackageDir = rootProject.layout.buildDirectory.dir(
  "js/packages/${rootProject.name}-${project.name}"
)
val generatedWebpackConfigFile = generatedWebpackPackageDir.map { it.file("webpack.config.js") }
val webpackExecutable = rootProject.layout.buildDirectory.file("js/node_modules/.bin/webpack")

val prepareHostedWebpackConfig = tasks.register("prepareHostedWebpackConfig") {
  val configContents = """
    const config = require(${jsString(generatedWebpackConfigFile.get().asFile.absolutePath)});
    // Development and production webpack tasks share this generated package directory. A
    // concurrently running development server can therefore replace its entry with the larger
    // development compiler output. Pin the hosted build to the stable production compiler output.
    config.entry = {
      main: [${jsString(productionExecutableJsFile.get().asFile.absolutePath)}]
    };
    const cppOnlyRequest =
      /^(?:@codingame\/monaco-vscode|monaco-languageclient(?:\/|${'$'})|monaco-editor(?:\/|${'$'})|vscode${'$'})/;
    const existingExternals =
      config.externals == null
        ? []
        : Array.isArray(config.externals)
          ? config.externals
          : [config.externals];

    config.externals = [
      ...existingExternals,
      ({ request }, callback) =>
        cppOnlyRequest.test(request || "")
          ? callback(null, "commonjs " + request)
          : callback()
    ];
    config.output.path = ${jsString(hostedCoreBundleDir.get().asFile.absolutePath)};
    config.output.filename = "tidyparse-core.js";
    config.output.clean = true;

    module.exports = config;
  """.trimIndent() + "\n"

  inputs.property("contents", configContents)
  outputs.file(hostedWebpackConfigFile)

  doLast {
    hostedWebpackConfigFile.get().asFile.apply {
      parentFile.mkdirs()
      writeText(configContents)
    }
  }
}

val jsBrowserHostedCoreWebpack = tasks.register<Exec>("jsBrowserHostedCoreWebpack") {
  group = "build"
  description = "Builds the hosted non-C++ bundle without Monaco/VS Code dependencies"

  dependsOn("jsBrowserProductionWebpack", prepareHostedWebpackConfig)

  inputs.files(
    productionJsFile,
    productionExecutableJsFile,
    generatedWebpackConfigFile,
    hostedWebpackConfigFile
  )
  outputs.files(hostedCoreJsFile, hostedCoreJsMapFile)

  doFirst {
    workingDir(generatedWebpackPackageDir.get().asFile)
    commandLine(
      webpackExecutable.get().asFile.absolutePath,
      "--config",
      hostedWebpackConfigFile.get().asFile.absolutePath
    )
  }
}

fun File.withInlineSourceMap(mapFile: File): String {
  val jsCode = readText()
    .withoutEmbeddedWebResources()
    .replace(Regex("""(?m)^//# sourceMappingURL=.*$"""), "")
  val mapJson = mapFile.takeIf { it.exists() }?.readText() ?: return jsCode
  val mapB64 = Base64.encode(mapJson.toByteArray())
  return jsCode.trimEnd('\n', '\r') + "\n//# sourceMappingURL=data:application/json;base64,$mapB64"
}

fun ByteArray.replaceAllBytes(target: ByteArray, replacement: ByteArray): Pair<ByteArray, Int> {
  require(target.isNotEmpty()) { "target must not be empty" }

  val out = ByteArrayOutputStream(size)
  var replacements = 0
  var i = 0
  while (i < size) {
    val matches = i + target.size <= size && target.indices.all { this[i + it] == target[it] }
    if (matches) {
      out.write(replacement)
      i += target.size
      replacements++
    } else {
      out.write(this[i].toInt())
      i++
    }
  }

  return out.toByteArray() to replacements
}

fun anonymizeArtifact(root: File, original: String = "breandan"): Int {
  val replacement = UUID.randomUUID().toString()
    .replace("-", "")
    .take(original.length)
  val targetBytes = original.toByteArray()
  val replacementBytes = replacement.toByteArray()
  var replacements = 0

  root.walkTopDown()
    .filter { it.isFile }
    .forEach { file ->
      val (anonymized, count) = file.readBytes().replaceAllBytes(targetBytes, replacementBytes)
      if (count > 0) {
        file.writeBytes(anonymized)
        replacements += count
      }
    }

  root.walkBottomUp()
    .filter { it != root && original in it.name }
    .forEach { file ->
      val dest = file.resolveSibling(file.name.replace(original, replacement))
      check(file.renameTo(dest)) { "Failed to rename ${file.absolutePath} to ${dest.absolutePath}" }
      replacements++
    }

  println("Anonymized $replacements artifact occurrence(s) of '$original'.")
  return replacements
}

tasks {
  val browserConsoleTailService = gradle.sharedServices.registerIfAbsent(
    "browserConsoleTailService",
    BrowserConsoleTailService::class
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

  register<Exec>("replotMetrics") {
    commandLine("../gradlew", "clean", "jsBrowserTest", "--info") // --info flag is crucial to read output

    standardOutput = ByteArrayOutputStream()

    doLast {
      val output = standardOutput.toString()
      if ("Total CPU latency" !in output) throw Exception("No output found: $output")
      val totalGPUTime = output.substringAfter("Total GPU latency:").substringBefore("\n")
      val totalGPURepairs = output.substringAfter("Total GPU repairs:").substringBefore("\n")
      val totalGPUMatches = output.substringAfter("Total GPU matches:").substringBefore("\n")
      saveStats(totalGPUTime, "repair_gpu_timings")
      saveStats(totalGPURepairs, "total_gpu_repairs")
      saveStats(totalGPUMatches, "total_gpu_matches")

      val totalCPUTime = output.substringAfter("Total CPU latency:").substringBefore("\n")
      val totalCPURepairs = output.substringAfter("Total CPU repairs:").substringBefore("\n")
      val totalCPUMatches = output.substringAfter("Total CPU matches:").substringBefore("\n")
      saveStats(totalCPUTime, "repair_cpu_timings")
      saveStats(totalCPURepairs, "total_cpu_repairs")
      saveStats(totalCPUMatches, "total_cpu_matches")

      plotGrids("repair_gpu_timings", "total_gpu_repairs", "total_gpu_matches", "repair_cpu_timings", "total_cpu_repairs", "total_cpu_matches")
    }
  }

  register("bundleHeadless") {
    dependsOn(project(":tidyparse-web").tasks.named("jsBrowserProductionWebpack"))

    val outHtml = File(System.getProperty("user.home"), "tidyparse.html")

    inputs.files(productionJsFile, productionJsMapFile, ngramFile, wdfaFile, rerankerWeightsFile)
    outputs.file(outHtml)

    doLast {
      val inlinedJs = productionJsFile.get().withInlineSourceMap(productionJsMapFile.get())
      val embeddedResources = embeddedRuntimeResources(includeExamples = false)

      val html = """
        <!doctype html>
        <meta charset="utf-8">
        <title>TidyParse Headless</title>
        <script type="text/javascript">
        window.REPAIR_MODE = "headless";
        $embeddedResources
        </script>
        <script type="module">
        $inlinedJs
        </script>
    """.trimIndent()

      outHtml.writeText(html)

      println("✓ Self-contained headless bundle written to ${outHtml.absolutePath}")
    }
  }

  register("bundleJCEF") {
    group = "build"
    description = "Bundles a single self-contained JCEF HTML template for the IntelliJ plugin"

    dependsOn(project(":tidyparse-web").tasks.named("jsBrowserProductionWebpack"))

    val outHtml = rootProject.layout.projectDirectory
      .file("tidyparse-intellij/src/main/resources/jcef/tidyparse-jcef.html")

    inputs.files(productionJsFile, productionJsMapFile, ngramFile, wdfaFile, rerankerWeightsFile)
    outputs.file(outHtml)

    doLast {
      val inlinedJs = productionJsFile.get().withInlineSourceMap(productionJsMapFile.get())
      val embeddedResources = embeddedRuntimeResources(includeExamples = false)

      val html = """
  <!doctype html>
  <html>
  <head>
    <meta charset="utf-8">
    <title>TidyParse JCEF Runtime</title>
  </head>
  <body>
    <pre id="tidyparse-jcef-log" style="display:none"></pre>

<script>
window.REPAIR_MODE = "jcef";
$embeddedResources

function __tidyparseJcefSend(payload) {
  __JCEF_EVENT_CALLBACK__;
}

window.__tidyparseJcefSend = __tidyparseJcefSend;
</script>

    <script type="module">
    $inlinedJs
    </script>
  </body>
  </html>
""".trimIndent()

      outHtml.asFile.apply { parentFile.mkdirs(); writeText(html) }
      println("✓ JCEF bundle written to ${outHtml.asFile.absolutePath}")
      println("  Placeholder to replace at runtime: __JCEF_EVENT_CALLBACK__")
    }
  }

  val prepareWebDeploy = register<Sync>("prepareWebDeploy") {
    group = "deployment"
    description = "Stages tidyparse-web files for deployment to tidyparse.github.io"

    dependsOn(jsBrowserHostedCoreWebpack, refreshClangdResources)

    into(webDeployStagingDir)
    from("src/jsMain/resources") {
      exclude(".DS_Store")
      exclude("**/.DS_Store")
      exclude("**/.idea/**")
      exclude(".idea/**")
    }
    from(productionBundleDir) {
      include("tidyparse-web.js")
      include("tidyparse-web.js.map")
    }
    from(hostedCoreBundleDir) {
      include("tidyparse-core.js")
      include("tidyparse-core.js.map")
    }

    inputs.files(
      productionJsFile,
      hostedCoreJsFile,
      ngramFile,
      wdfaFile,
      rerankerWeightsFile,
      exampleFiles,
      deployExampleFiles
    )
    outputs.file(webDeployStagingDir.map { it.file("tidyparse-web.js") })
    outputs.file(webDeployStagingDir.map { it.file("tidyparse-core.js") })
    outputs.file(webDeployStagingDir.map { it.file("tidyparse-index-resources.js") })
    outputs.file(webDeployStagingDir.map { it.file("tidyparse-python-resources.js") })

    doLast {
      val outDir = webDeployStagingDir.get().asFile
      val fullBundle = outDir.resolve("tidyparse-web.js")
      val coreBundle = outDir.resolve("tidyparse-core.js")
      val indexResources = outDir.resolve("tidyparse-index-resources.js")
      val pythonResources = outDir.resolve("tidyparse-python-resources.js")
      val originalBundleScript = """<script src="tidyparse-web.js"></script>"""

      indexResources.writeText(embeddedHostedIndexResources())
      pythonResources.writeText(embeddedHostedPythonResources())

      fun rewriteHostedPage(page: String, resourceScript: String? = null) {
        val htmlFile = outDir.resolve(page)
        val html = htmlFile.readText()
        check(originalBundleScript in html) { "Could not find the web bundle script in ${htmlFile.absolutePath}" }
        val replacement = buildString {
          if (resourceScript != null) appendLine("""<script src="$resourceScript"></script>""")
          append("""<script src="tidyparse-core.js"></script>""")
        }
        htmlFile.writeText(html.replace(originalBundleScript, replacement))
      }

      rewriteHostedPage("index.html", "tidyparse-index-resources.js")
      rewriteHostedPage("cnf.html")
      rewriteHostedPage("python.html", "tidyparse-python-resources.js")

      val indexInitialJavaScriptGzipBytes =
        gzip(coreBundle.readBytes()).size.toLong() + gzip(indexResources.readBytes()).size.toLong()
      check(indexInitialJavaScriptGzipBytes <= maxHostedInitialJavaScriptGzipBytes) {
        "Hosted index JavaScript exceeds the ${maxHostedInitialJavaScriptGzipBytes}-byte gzip budget: " +
                "$indexInitialJavaScriptGzipBytes bytes"
      }

      println("✓ Staged tidyparse-web deployment at ${webDeployStagingDir.get().asFile.absolutePath}")
      println("  Hosted core: ${coreBundle.length()} bytes")
      println("  C++/worker bundle: ${fullBundle.length()} bytes")
      println("  Index resources: ${indexResources.length()} bytes (${exampleFiles.files.size} examples)")
      println("  Python resources: ${pythonResources.length()} bytes (small resources only)")
      println("  Index initial JavaScript: $indexInitialJavaScriptGzipBytes gzip bytes")
      println("  Compressed/base64 resource payloads larger than $maxHostedInlinePayloadBytes bytes stay as individual files")
    }
  }

  register<DeployWebTask>("deployWeb") {
    group = "deployment"
    description = "Builds, commits, and pushes tidyparse-web to tidyparse.github.io. Requires --msg \"commit message\"."

    dependsOn(prepareWebDeploy)

    sourceDirectory.set(webDeployStagingDir)
    commitMessage.convention(providers.gradleProperty("deployWebMessage"))
    repositoryUrl.convention(providers.gradleProperty("deployWebRepoUrl").orElse("https://github.com/tidyparse/tidyparse.github.io.git"))
    pushUrl.convention(providers.gradleProperty("deployWebPushUrl").orElse("git@github.com:tidyparse/tidyparse.github.io.git"))
    branch.convention(providers.gradleProperty("deployWebBranch").orElse("main"))
    checkoutPath.convention(
      providers.gradleProperty("deployWebRepoDir")
        .orElse(layout.buildDirectory.dir("deploy/tidyparse.github.io").map { it.asFile.absolutePath })
    )
    preservedRootEntries.convention(listOf(".git", ".github", ".gitignore", "CNAME", ".nojekyll", "README", "README.md", "LICENSE"))
  }

  val prepareZipArtifact = register<Sync>("prepareZipArtifact") {
    group = "distribution"
    description = "Stages tidyparse-web as a Chrome-openable local browser artifact"

    dependsOn("jsBrowserProductionWebpack", "jsProcessResources")

    val stagingDir = layout.buildDirectory.dir("browser-artifact")
    val resourcesDir = layout.buildDirectory.dir("processedResources/js/main")

    into(stagingDir)
    from(resourcesDir) {
      exclude(".DS_Store")
      exclude("**/.DS_Store")
      exclude("examples/.idea/**")
    }
    from(productionBundleDir) {
      include("tidyparse-web.js")
      include("tidyparse-web.js.map")
    }

    inputs.files(ngramFile, wdfaFile, rerankerWeightsFile, exampleFiles)
    outputs.file(stagingDir.map { it.file("tidyparse-local-resources.js") })
    outputs.file(stagingDir.map { it.file("README.txt") })
    outputs.upToDateWhen { false }

    doLast {
      val outDir = stagingDir.get().asFile

      outDir.resolve("tidyparse-local-resources.js").writeText(embeddedRuntimeResources())

      outDir.walkTopDown()
        .filter { it.isFile && it.extension.equals("html", ignoreCase = true) }
        .forEach { htmlFile ->
          val html = htmlFile.readText()
            .replace("""href="/pluginIcon.svg"""", """href="pluginIcon.svg"""")
            .replace("""src="/pluginIcon.svg"""", """src="pluginIcon.svg"""")
            .let {
              val resourceScript = """<script src="tidyparse-local-resources.js"></script>"""
              if ("tidyparse-web.js" !in it || resourceScript in it) it
              else it.replace(
                Regex("""(?m)(\s*<script\s+src=["']tidyparse-web\.js["']\s*></script>)"""),
                "\n$resourceScript$1"
              )
            }
          htmlFile.writeText(html)
        }

      outDir.resolve("README.txt").writeText(
        """
        Tidyparse browser artifact

        Open index.html or python.html in Chrome after unzipping this archive.
        """.trimIndent() + "\n"
      )

      anonymizeArtifact(outDir)
    }
  }

  register<Zip>("browserArtifact") {
    group = "distribution"
    description = "Creates browser-artifact.zip with the tidyparse-web local browser app"

    dependsOn(prepareZipArtifact)
    archiveFileName = "browser-artifact.zip"
    destinationDirectory = layout.buildDirectory.dir("distributions")
    from(prepareZipArtifact.map { it.destinationDir })
  }
}

// To deploy the browser application, run:
// ./gradlew deployWeb --msg "add visual status indicators"
// By default this clones/fetches:
//  https://github.com/tidyparse/tidyparse.github.io.git
// and pushes via SSH:
//  git@github.com:tidyparse/tidyparse.github.io.git
// into:
//  tidyparse-web/build/deploy/tidyparse.github.io
// Override the checkout with:
// ./gradlew deployWeb --msg "..." --repo-dir /path/to/tidyparse.github.io
// Wait a few minutes for CI to finish, then check the website:
//  https://tidyparse.github.io

// To run on localhost and open a browser, run:
//  ./gradlew :tidyparse-web:jsBrowserDevelopmentRun --continuous

// To test on localhost, run:
// ./gradlew jsTest

abstract class BrowserConsoleTailService : BuildService<BuildServiceParameters.None>, AutoCloseable {
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

abstract class DeployWebTask : DefaultTask() {
  @get:InputDirectory
  @get:PathSensitive(PathSensitivity.RELATIVE)
  abstract val sourceDirectory: DirectoryProperty

  @get:Input
  @get:Optional
  abstract val commitMessage: Property<String>

  @get:Input
  abstract val repositoryUrl: Property<String>

  @get:Input
  abstract val pushUrl: Property<String>

  @get:Input
  abstract val branch: Property<String>

  @get:Input
  abstract val checkoutPath: Property<String>

  @get:Input
  abstract val preservedRootEntries: ListProperty<String>

  @Option(option = "msg", description = "Commit message for the GitHub Pages deployment.")
  fun setCommitMessageOption(message: String) = commitMessage.set(message)

  @Option(option = "message", description = "Commit message for the GitHub Pages deployment.")
  fun setCommitMessageLongOption(message: String) = commitMessage.set(message)

  @Option(option = "repo-dir", description = "Local tidyparse.github.io checkout directory.")
  fun setCheckoutPathOption(path: String) = checkoutPath.set(path)

  @Option(option = "repo-url", description = "GitHub Pages repository URL.")
  fun setRepositoryUrlOption(url: String) = repositoryUrl.set(url)

  @Option(option = "push-url", description = "GitHub Pages repository push URL.")
  fun setPushUrlOption(url: String) = pushUrl.set(url)

  @Option(option = "branch", description = "GitHub Pages branch to deploy.")
  fun setBranchOption(branchName: String) = branch.set(branchName)

  @TaskAction
  fun deploy() {
    val message = commitMessage.orNull?.trim()
      ?: throw GradleException("Pass a deployment commit message, e.g. ./gradlew deployWeb --msg \"add visual status indicators\"")

    if (message.isEmpty()) throw GradleException("Deployment commit message cannot be empty.")

    val sourceDir = sourceDirectory.get().asFile
    require(sourceDir.isDirectory) { "Deploy source directory does not exist: ${sourceDir.absolutePath}" }

    val repoDir = File(checkoutPath.get()).absoluteFile
    val repoUrl = repositoryUrl.get()
    val repoPushUrl = pushUrl.get()
    val deployBranch = branch.get()

    ensureCheckout(repoDir, repoUrl, repoPushUrl, deployBranch)
    syncToCheckout(sourceDir, repoDir)

    val status = git(repoDir, "status", "--porcelain")
    if (status.isBlank()) {
      pushIfAhead(repoDir, deployBranch)
      return
    }

    git(repoDir, "add", "--all")
    val staged = git(repoDir, "diff", "--cached", "--name-status")
    if (staged.isBlank()) {
      pushIfAhead(repoDir, deployBranch)
      return
    }

    println("Deployment changes:")
    staged.lineSequence().take(40).forEach { println("  $it") }
    if (staged.lineSequence().count() > 40) println("  ...")

    git(repoDir, "commit", "-m", message)
    pushBranch(repoDir, deployBranch)

    println("✓ Deployed tidyparse-web to $repoUrl ($deployBranch)")
  }

  private fun ensureCheckout(repoDir: File, repoUrl: String, repoPushUrl: String, deployBranch: String) {
    if (repoPushUrl.normalizedGitHubRepo() != repoUrl.normalizedGitHubRepo()) {
      throw GradleException("Refusing to deploy: push URL '$repoPushUrl' does not match repository URL '$repoUrl'.")
    }

    if (!repoDir.exists()) {
      repoDir.parentFile.mkdirs()
      runCommand(listOf("git", "clone", "--branch", deployBranch, "--single-branch", repoUrl, repoDir.absolutePath))
      configurePushUrl(repoDir, repoPushUrl)
      return
    }

    if (!repoDir.isDirectory) throw GradleException("Deploy checkout path exists but is not a directory: ${repoDir.absolutePath}")

    if (!repoDir.resolve(".git").exists()) throw GradleException("Deploy checkout path is not a Git repository: ${repoDir.absolutePath}")

    val remote = git(repoDir, "remote", "get-url", "origin").trim()
    if (remote.normalizedGitHubRepo() != repoUrl.normalizedGitHubRepo())
      throw GradleException("Refusing to deploy from ${repoDir.absolutePath}: origin is '$remote', expected '$repoUrl'.")

    configurePushUrl(repoDir, repoPushUrl)
    failIfDirty(repoDir, "before updating")
    git(repoDir, "fetch", "origin", deployBranch)

    val currentBranch = git(repoDir, "rev-parse", "--abbrev-ref", "HEAD").trim()
    if (currentBranch != deployBranch) {
      val localBranch = git(repoDir, "branch", "--list", deployBranch).trim()
      if (localBranch.isBlank()) {
        git(repoDir, "checkout", "-b", deployBranch, "origin/$deployBranch")
      } else {
        git(repoDir, "checkout", deployBranch)
      }
    }

    git(repoDir, "pull", "--ff-only", "origin", deployBranch)
    failIfDirty(repoDir, "after updating")
  }

  private fun configurePushUrl(repoDir: File, repoPushUrl: String) {
    val currentPushUrl = git(repoDir, "remote", "get-url", "--push", "origin").trim()
    if (currentPushUrl != repoPushUrl) git(repoDir, "remote", "set-url", "--push", "origin", repoPushUrl)
  }

  private fun pushIfAhead(repoDir: File, deployBranch: String) {
    val commitsAhead = git(repoDir, "rev-list", "--count", "origin/$deployBranch..HEAD").trim().toInt()
    if (commitsAhead == 0) {
      println("No deployment changes detected in ${repoDir.absolutePath}; nothing to commit or push.")
      return
    }

    println("No working tree changes detected, but $commitsAhead unpushed deployment commit(s) exist; pushing.")
    pushBranch(repoDir, deployBranch)
    println("✓ Pushed pending tidyparse-web deployment commit(s) to $deployBranch")
  }

  private fun pushBranch(repoDir: File, deployBranch: String) = git(repoDir, "push", "origin", "HEAD:$deployBranch")

  private fun syncToCheckout(sourceDir: File, repoDir: File) {
    val preservedNames = preservedRootEntries.get().toSet()
    repoDir.listFiles()
      ?.filter { it.name !in preservedNames }
      ?.forEach { entry ->
        if (!entry.deleteRecursively() && entry.exists()) {
          throw GradleException("Failed to remove stale deployment entry: ${entry.absolutePath}")
        }
      }

    sourceDir.copyRecursively(repoDir, overwrite = true)
  }

  private fun failIfDirty(repoDir: File, phase: String) {
    val status = git(repoDir, "status", "--porcelain")
    if (status.isNotBlank()) {
      throw GradleException(
        "Refusing to deploy because ${repoDir.absolutePath} has uncommitted changes $phase:\n$status"
      )
    }
  }

  private fun git(workingDir: File, vararg args: String): String = runCommand(listOf("git") + args, workingDir)

  private fun runCommand(command: List<String>, workingDir: File? = null): String {
    val process = ProcessBuilder(command)
      .apply { if (workingDir != null) directory(workingDir) }
      .redirectErrorStream(true)
      .start()

    val output = process.inputStream.bufferedReader().readText()
    val exitCode = process.waitFor()
    if (exitCode != 0) {
      throw GradleException(
        "Command failed (${command.displayCommand()}) with exit code $exitCode:\n${output.trim()}"
      )
    }
    return output.trimEnd()
  }

  private fun String.normalizedGitHubRepo(): String {
    val withoutGitSuffix = trim().removeSuffix("/").removeSuffix(".git")
    return when {
      withoutGitSuffix.startsWith("git@github.com:") -> withoutGitSuffix.removePrefix("git@github.com:")
      withoutGitSuffix.startsWith("ssh://git@github.com/") -> withoutGitSuffix.removePrefix("ssh://git@github.com/")
      withoutGitSuffix.startsWith("https://github.com/") -> withoutGitSuffix.removePrefix("https://github.com/")
      withoutGitSuffix.startsWith("http://github.com/") -> withoutGitSuffix.removePrefix("http://github.com/")
      else -> withoutGitSuffix
    }.lowercase()
  }

  private fun List<String>.displayCommand(): String = joinToString(" ") { arg ->
    if (arg.any { it.isWhitespace() }) "\"${arg.replace("\"", "\\\"")}\"" else arg
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
      // Generated from the complete tidyparse-web/clangd recipe.
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
