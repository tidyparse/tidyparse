package buildlogic

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.api.tasks.options.Option
import java.io.File

/** Deploys one independently owned slice of the shared website repository. */
abstract class ManagedSiteDeployTask : DefaultTask() {
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
  abstract val deploymentId: Property<String>

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
      ?: throw GradleException("Pass a deployment commit message with --msg \"commit message\".")
    if (message.isEmpty()) throw GradleException("Deployment commit message cannot be empty.")

    val sourceDir = sourceDirectory.get().asFile
    require(sourceDir.isDirectory) { "Deploy source directory does not exist: ${sourceDir.absolutePath}" }

    val repoDir = File(checkoutPath.get()).absoluteFile
    val repoUrl = repositoryUrl.get()
    val repoPushUrl = pushUrl.get()
    val deployBranch = branch.get()

    ensureCheckout(repoDir, repoUrl, repoPushUrl, deployBranch)
    syncOwnedFiles(sourceDir, repoDir)

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

    println("✓ Deployed ${deploymentId.get()} to $repoUrl ($deployBranch)")
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

    if (!repoDir.isDirectory) {
      throw GradleException("Deploy checkout path exists but is not a directory: ${repoDir.absolutePath}")
    }
    if (!repoDir.resolve(".git").exists()) {
      throw GradleException("Deploy checkout path is not a Git repository: ${repoDir.absolutePath}")
    }

    val remote = git(repoDir, "remote", "get-url", "origin").trim()
    if (remote.normalizedGitHubRepo() != repoUrl.normalizedGitHubRepo()) {
      throw GradleException(
        "Refusing to deploy from ${repoDir.absolutePath}: origin is '$remote', expected '$repoUrl'."
      )
    }

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
    println("✓ Pushed pending ${deploymentId.get()} deployment commit(s) to $deployBranch")
  }

  private fun pushBranch(repoDir: File, deployBranch: String) =
    git(repoDir, "push", "origin", "HEAD:$deployBranch")

  private fun syncOwnedFiles(sourceDir: File, repoDir: File) {
    val id = deploymentId.get().also {
      require(it.matches(Regex("[a-zA-Z0-9._-]+"))) { "Invalid deployment ID: $it" }
    }
    val manifest = repoDir.resolve(".tidyparse-deploy-$id.manifest")
    val ownedFiles = sourceDir.walkTopDown()
      .filter(File::isFile)
      .map { it.relativeTo(sourceDir).invariantSeparatorsPath }
      .sorted()
      .toList()
    val ownedFileSet = ownedFiles.toSet()
    val ownedTargets = ownedFiles.associateWith { managedFile(repoDir, it) }

    val previousFiles = manifest.takeIf(File::isFile)
      ?.readLines()
      .orEmpty()
      .filter(String::isNotBlank)
    val previousTargets = previousFiles.associateWith { managedFile(repoDir, it) }

    val otherOwners = repoDir.listFiles()
      .orEmpty()
      .filter {
        it.isFile &&
          it.name.startsWith(".tidyparse-deploy-") &&
          it.name.endsWith(".manifest") &&
          it != manifest
      }
      .associateWith { otherManifest ->
        otherManifest.readLines()
          .filter(String::isNotBlank)
          .onEach { managedFile(repoDir, it) }
          .toSet()
      }
    val conflicts = otherOwners.mapValues { (_, paths) -> paths.intersect(ownedFileSet) }
      .filterValues(Set<String>::isNotEmpty)
    if (conflicts.isNotEmpty()) {
      val detail = conflicts.entries.joinToString("\n") { (otherManifest, paths) ->
        "${otherManifest.name}: ${paths.sorted().joinToString()}"
      }
      throw GradleException("Deployment '$id' overlaps files owned by another deployment:\n$detail")
    }

    previousTargets.values.forEach { owned ->
      if (owned.isDirectory) owned.deleteRecursively() else owned.delete()
    }
    previousFiles
      .map { previousTargets.getValue(it).parentFile }
      .distinct()
      .sortedByDescending { it.toPath().nameCount }
      .forEach { directory ->
        var candidate = directory
        while (candidate != repoDir && candidate.isDirectory && candidate.list().isNullOrEmpty()) {
          candidate.delete()
          candidate = candidate.parentFile
        }
      }

    sourceDir.copyRecursively(repoDir, overwrite = true)

    check(ownedTargets.values.all(File::isFile)) {
      "Deployment '$id' did not copy every managed source file into the checkout"
    }
    manifest.writeText(ownedFiles.joinToString(separator = "\n", postfix = "\n"))
  }

  private fun managedFile(repoDir: File, relativePath: String): File {
    val candidate = File(relativePath)
    require(!candidate.isAbsolute && relativePath.isNotBlank()) { "Invalid managed deployment path: $relativePath" }

    val repoPath = repoDir.toPath().toAbsolutePath().normalize()
    val target = repoPath.resolve(relativePath).normalize()
    require(target.startsWith(repoPath) && target != repoPath) { "Managed deployment path escapes checkout: $relativePath" }
    require(repoPath.relativize(target).firstOrNull()?.toString() != ".git") {
      "Managed deployment path may not modify .git: $relativePath"
    }
    return target.toFile()
  }

  private fun failIfDirty(repoDir: File, phase: String) {
    val status = git(repoDir, "status", "--porcelain")
    if (status.isNotBlank()) {
      throw GradleException(
        "Refusing to deploy because ${repoDir.absolutePath} has uncommitted changes $phase:\n$status"
      )
    }
  }

  private fun git(workingDir: File, vararg args: String): String =
    runCommand(listOf("git") + args, workingDir)

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
    if (arg.any(Char::isWhitespace)) "\"${arg.replace("\"", "\\\"")}\"" else arg
  }
}
