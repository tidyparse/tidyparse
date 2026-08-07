@file:OptIn(ExperimentalEncodingApi::class)

import org.gradle.api.tasks.testing.logging.TestExceptionFormat
import org.jetbrains.kotlin.gradle.targets.js.testing.KotlinJsTest
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig.Mode.DEVELOPMENT
import org.jetbrains.letsPlot.*
import org.jetbrains.letsPlot.export.ggsave
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.label.ggtitle
import java.awt.Desktop
import java.io.ByteArrayOutputStream
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
  id("org.ajoberstar.git-publish") version "6.0.0"
}

group = "ai.hypergraph"
version = "0.23.0"

kotlin {
  js {
    binaries.executable()

    browser {
      runTask { mainOutputFileName = "tidyparse-web.js" }

      webpackTask {
        // We need this to work on Chrome when deployed due to the PLATFORM_CALLER_STACKTRACE_DEPTH hack
        mode = DEVELOPMENT
        mainOutputFileName = "tidyparse-web.js"
        devtool = "source-map" // For debugging; remove for production
      }

      testTask { useKarma { useChromeHeadless() } }
    }
  }

  sourceSets {
    getByName("jsMain") {
      dependencies {
        implementation(project(":tidyparse-core"))
        implementation("org.jetbrains.kotlin-wrappers:kotlin-web:2026.6.3")
      }
    }

    getByName("jsTest") {
      dependencies {
        implementation(kotlin("test-js"))
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.11.0")
      }
    }
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

fun gzip(bytes: ByteArray): ByteArray =
  ByteArrayOutputStream().apply { GZIPOutputStream(this).use { it.write(bytes) } }.toByteArray()

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
window.raw_ngrams_gzip_b64 = ${jsString(rawNgramsGzipB64)};
window.raw_wdfa_gzip_b64 = ${jsString(rawWdfaGzipB64)};
window.raw_reranker_weights_gzip_b64 = ${jsString(rawRerankerWeightsGzipB64)};
window.raw_examples_gzip_b64 = ${jsString(rawExamplesGzipB64)};
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
      rootProject.projectDir.toPath().relativize(it.toPath()).toString()
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
val generatedWebpackPackageDir =
  rootProject.layout.buildDirectory.dir("js/packages/${rootProject.name}-${project.name}")
val generatedWebpackConfigFile = generatedWebpackPackageDir.map { it.file("webpack.config.js") }
val webpackExecutable = rootProject.layout.buildDirectory.file("js/node_modules/.bin/webpack")

val prepareHostedWebpackConfig = tasks.register("prepareHostedWebpackConfig") {
  val configContents = """
    const config = require(${jsString(generatedWebpackConfigFile.get().asFile.absolutePath)});
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

  doLast { hostedWebpackConfigFile.get().asFile.apply { parentFile.mkdirs(); writeText(configContents) } }
}

val jsBrowserHostedCoreWebpack = tasks.register<Exec>("jsBrowserHostedCoreWebpack") {
  group = "build"
  description = "Builds the hosted non-C++ bundle without Monaco/VS Code dependencies"

  dependsOn("jsBrowserProductionWebpack", prepareHostedWebpackConfig)

  inputs.files(productionJsFile, generatedWebpackConfigFile, hostedWebpackConfigFile)
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
  val jsCode = readText().withoutEmbeddedWebResources()
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

val deployWebMessage = providers.gradleProperty("deployWebMessage").map { it.trim() }

val deployWebRepoUri =
  providers.gradleProperty("deployWebPushUrl")
    .orElse(providers.gradleProperty("deployWebRepoUrl"))
    .orElse("git@github.com:tidyparse/tidyparse.github.io.git")

val deployWebBranch = providers.gradleProperty("deployWebBranch").orElse("main")

val deployWebRepoDirFile =
  providers.gradleProperty("deployWebRepoDir").map { file(it) }
    .orElse(layout.buildDirectory.dir("deploy/tidyparse.github.io").map { it.asFile })

gitPublish {
  repoUri.set(deployWebRepoUri)
  branch.set(deployWebBranch)
  repoDir.set(layout.dir(deployWebRepoDirFile))
  commitMessage.set(deployWebMessage.orElse(""))

  contents { from(webDeployStagingDir) }

  preserve {
    include(
      ".github/**",
      ".gitignore",
      "CNAME",
      ".nojekyll",
      "README",
      "README.md",
      "LICENSE"
    )
  }
}

tasks {
  named("gitPublishCopy") { dependsOn("prepareWebDeploy") }

  named("gitPublishReset") {
    doFirst {
      if (deployWebMessage.orNull.isNullOrBlank())
        throw GradleException("""Pass a deployment message with -PdeployWebMessage="add visual status indicators".""")
    }
  }

  register("deployWeb") {
    group = "deployment"
    description = "Builds, commits, and pushes tidyparse-web to tidyparse.github.io."
    dependsOn("gitPublishPush")
  }

  val browserConsoleTailService =
    gradle.sharedServices.registerIfAbsent(
      "browserConsoleTailService",
      BrowserConsoleTailService::class
    ) {}

  withType<KotlinJsTest>().configureEach {
    val testTaskPath = path
    val browserConsoleTailProcess = AtomicReference<Process?>()
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
      val browserConsoleLog = rootProject.layout.buildDirectory.file("ci-logs/browser-console.log").get().asFile
      browserConsoleLog.parentFile.mkdirs()
      browserConsoleLog.writeText("")

      val tailProcess = ProcessBuilder("tail", "-n", "+1", "-f", browserConsoleLog.absolutePath)
        .redirectErrorStream(true).start()

      browserConsoleTailService.get().stop(browserConsoleTailProcess.getAndSet(tailProcess))
      browserConsoleTailService.get().register(tailProcess)
      Thread {
        tailProcess.inputStream.bufferedReader().useLines { lines -> lines.forEach { println(it) } }
      }.apply {
        name = "browser-console-tail-$testTaskPath"
        isDaemon = true
        start()
      }
    }

    fun stopBrowserConsoleTail() =
      browserConsoleTailService.get().stop(browserConsoleTailProcess.getAndSet(null))

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

function __tidyparseJcefSend(payload) { __JCEF_EVENT_CALLBACK__; }

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

  register<Sync>("prepareWebDeploy") {
    group = "deployment"
    description = "Stages tidyparse-web files for deployment to tidyparse.github.io"

    dependsOn(jsBrowserHostedCoreWebpack)

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
}

// To deploy the browser application, run:
// ./gradlew deployWeb -PdeployWebMessage="commit message"
// Wait a few minutes for CI to finish, then check the website:
//  https://tidyparse.github.io

// To run on localhost and open a browser, run:
//  ./gradlew :tidyparse-web:jsBrowserDevelopmentRun --continuous

// To test on localhost, run:
// ./gradlew jsTest

abstract class BrowserConsoleTailService : BuildService<BuildServiceParameters.None>, AutoCloseable {
  private val processes = CopyOnWriteArrayList<Process>()
  fun register(process: Process) = processes.add(process)
  fun stop(process: Process?) = process?.let { processes.remove (it.apply { destroy() }) }
  override fun close() = processes.onEach { it.destroy() }.clear()
}