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

    dependsOn("jsBrowserProductionWebpack")

    into(webDeployStagingDir)
    from("src/jsMain/resources") {
      exclude(".DS_Store")
      exclude("**/.DS_Store")
      exclude("**/.idea/**")
      exclude(".idea/**")
    }
    from(productionBundleDir) {
      include("tidyparse-web.js.map")
    }

    inputs.files(productionJsFile, ngramFile, wdfaFile, rerankerWeightsFile, exampleFiles, deployExampleFiles)
    outputs.file(webDeployStagingDir.map { it.file("tidyparse-web.js") })

    doLast {
      val stagedJsBundle = webDeployStagingDir.get().asFile.resolve("tidyparse-web.js")
      val embeddedResources = embeddedRuntimeResources()

      stagedJsBundle.writeText(embeddedResources + productionJsFile.get().readText().withoutEmbeddedWebResources())
      println("✓ Staged tidyparse-web deployment at ${webDeployStagingDir.get().asFile.absolutePath}")
      println("  Embedded gzip-compressed python_4grams.txt, wdfa.bin, reranker_2000.q8.safetensors, and ${exampleFiles.files.size} examples into ${stagedJsBundle.absolutePath}")
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
