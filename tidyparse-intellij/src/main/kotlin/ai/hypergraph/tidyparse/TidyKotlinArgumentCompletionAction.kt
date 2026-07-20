@file:OptIn(org.jetbrains.kotlin.analysis.api.KaExperimentalApi::class)

package ai.hypergraph.tidyparse

import com.intellij.codeInsight.lookup.LookupElement
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.codeInsight.lookup.LookupEvent
import com.intellij.codeInsight.lookup.LookupListener
import com.intellij.codeInsight.lookup.LookupManager
import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.actionSystem.ActionManager
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.actionSystem.CommonDataKeys
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.application.runReadAction
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.keymap.Keymap
import com.intellij.openapi.keymap.KeymapManager
import com.intellij.openapi.keymap.KeymapUtil
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.project.DumbAwareAction
import com.intellij.openapi.project.Project
import com.intellij.openapi.startup.StartupActivity
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiDocumentManager
import com.intellij.psi.PsiFile
import com.intellij.psi.util.PsiTreeUtil
import org.jetbrains.kotlin.analysis.api.KaSession
import org.jetbrains.kotlin.analysis.api.analyze
import org.jetbrains.kotlin.analysis.api.components.KaScopeKind
import org.jetbrains.kotlin.analysis.api.components.KaSubtypingErrorTypePolicy
import org.jetbrains.kotlin.analysis.api.components.expectedType
import org.jetbrains.kotlin.analysis.api.components.isSubtypeOf
import org.jetbrains.kotlin.analysis.api.components.returnType
import org.jetbrains.kotlin.analysis.api.components.resolveCall
import org.jetbrains.kotlin.analysis.api.components.resolveToCallCandidates
import org.jetbrains.kotlin.analysis.api.components.scope
import org.jetbrains.kotlin.analysis.api.components.scopeContext
import org.jetbrains.kotlin.analysis.api.resolution.KaFunctionCall
import org.jetbrains.kotlin.analysis.api.scopes.KaTypeScope
import org.jetbrains.kotlin.analysis.api.signatures.KaFunctionSignature
import org.jetbrains.kotlin.analysis.api.signatures.KaVariableSignature
import org.jetbrains.kotlin.analysis.api.symbols.KaCallableSymbol
import org.jetbrains.kotlin.analysis.api.symbols.KaNamedFunctionSymbol
import org.jetbrains.kotlin.analysis.api.symbols.KaVariableSymbol
import org.jetbrains.kotlin.analysis.api.symbols.markers.KaNamedSymbol
import org.jetbrains.kotlin.analysis.api.types.KaType
import org.jetbrains.kotlin.psi.KtCallElement
import org.jetbrains.kotlin.psi.KtElement
import org.jetbrains.kotlin.psi.KtExpression
import org.jetbrains.kotlin.psi.KtFile
import org.jetbrains.kotlin.psi.KtNamedFunction
import org.jetbrains.kotlin.psi.KtParameter
import org.jetbrains.kotlin.psi.KtProperty
import org.jetbrains.kotlin.psi.KtPsiFactory
import org.jetbrains.kotlin.psi.KtValueArgument
import org.jetbrains.kotlin.psi.KtValueArgumentList

private const val KOTLIN_ARGUMENT_COMPLETION_ACTION_ID = "TidyParse.KotlinArgumentCompletion"
private const val KOTLIN_PROBE_NAME = "__tidyparse_probe"
private const val KOTLIN_MAX_COMPLETION_TOKENS = 10
private const val KOTLIN_MAX_COMPLETION_RESULTS = 10
private const val KOTLIN_COMPLETION_SEARCH_MILLIS = 1_500L
private const val KOTLIN_MAX_COMPLETION_PARAMETERS = 4

internal data class TypedKotlinValue<T>(val name: String, val type: T)

internal data class TypedKotlinCallable<T>(
  val name: String,
  val parameters: List<T>,
  val returnType: T
)

internal data class TypedKotlinEnvironment<T>(
  val values: List<TypedKotlinValue<T>> = emptyList(),
  val functions: List<TypedKotlinCallable<T>> = emptyList()
)

internal data class TypedKotlinExpression<T>(
  val text: String,
  val type: T,
  val length: Int
)

internal data class KotlinArgumentCompletionResult(
  val replacementRange: TextRange,
  val completions: List<String>,
  val message: String? = null
)

private class KotlinArgumentSearchBudget(
  private val maxMillis: Long = KOTLIN_COMPLETION_SEARCH_MILLIS
) {
  private val startedNanos = System.nanoTime()
  private val deadlineNanos = startedNanos + maxMillis * 1_000_000L

  @Volatile
  var expired: Boolean = false
    private set

  fun keepGoing(): Boolean {
    ProgressManager.checkCanceled()
    val stillWithinBudget = System.nanoTime() <= deadlineNanos
    if (!stillWithinBudget) expired = true
    return stillWithinBudget
  }

  fun elapsedMillis(): Long = (System.nanoTime() - startedNanos) / 1_000_000L
}

private data class KotlinArgumentSlot(
  val replacementRange: TextRange,
  val calleeName: String?,
  val argumentIndex: Int,
  val scopeAnchor: KtElement?
)

class TidyKotlinArgumentCompletionAction : DumbAwareAction() {
  init {
    TidyKotlinArgumentCompletionDiagnostics.log("action constructed: ${javaClass.name}")
  }

  override fun actionPerformed(e: AnActionEvent) {
    val editor = e.getData(CommonDataKeys.EDITOR)
    val psiFile = e.psiFileFromEditor(editor)
    TidyKotlinArgumentCompletionDiagnostics.log(
      "actionPerformed: place=${e.place} project=${e.project?.name ?: "<none>"} " +
        "editor=${editor != null} " +
        "psiFile=${psiFile?.javaClass?.name ?: "<none>"}:${psiFile?.name ?: "<none>"}"
    )

    val project = e.project ?: run {
      TidyKotlinArgumentCompletionDiagnostics.log("actionPerformed aborted: missing project")
      return
    }
    val ktEditor = editor ?: run {
      TidyKotlinArgumentCompletionDiagnostics.log("actionPerformed aborted: missing editor")
      return
    }
    val ktFile = psiFile as? KtFile ?: run {
      TidyKotlinArgumentCompletionDiagnostics.log("actionPerformed aborted: PSI file is not a KtFile")
      project.notifyKotlinArgumentCompletion("TidyParse did not receive a Kotlin PSI file.")
      return
    }

    TidyKotlinArgumentCompletionDiagnostics.log(
      "actionPerformed accepted: file=${ktFile.virtualFile?.path ?: ktFile.name} " +
        "offset=${ktEditor.caretModel.offset} documentLength=${ktEditor.document.textLength}"
    )

    ApplicationManager.getApplication().executeOnPooledThread {
      val completionAttempt = runCatching {
        runReadAction {
          buildKotlinArgumentCompletions(ktEditor, ktFile, TidyKotlinArgumentCompletionDiagnostics::log)
        }
      }

      ApplicationManager.getApplication().invokeLater {
        val result = completionAttempt.getOrElse { throwable ->
          TidyKotlinArgumentCompletionDiagnostics.warn("completion search failed", throwable)
          project.notifyKotlinArgumentCompletion("Kotlin argument completion failed. See runIde console or idea.log.")
          return@invokeLater
        }

        TidyKotlinArgumentCompletionDiagnostics.log(
          "completion result: " +
            if (result == null) {
              "null"
            } else {
              "range=${result.replacementRange} completions=${result.completions.size} " +
                "message=${result.message ?: "<none>"} preview=${result.completions.take(10)}"
            }
        )

        when {
          result == null -> project.notifyKotlinArgumentCompletion("Place the caret inside a Kotlin call argument.")
          result.completions.isEmpty() -> project.notifyKotlinArgumentCompletion(
            result.message ?: "No well-typed Kotlin argument expressions found."
          )
          else -> showTidyKotlinArgumentLookup(project, ktEditor, result.replacementRange, result.completions)
        }
      }
    }
  }

  override fun update(e: AnActionEvent) {
    val editor = e.getData(CommonDataKeys.EDITOR)
    val psiFile = e.psiFileFromEditor(editor)
    val ktFile = psiFile as? KtFile
    val enabled = e.project != null && editor != null && ktFile != null
    e.presentation.isEnabledAndVisible = enabled
    TidyKotlinArgumentCompletionDiagnostics.logUpdate(e, editor, psiFile, ktFile, enabled)
  }
}

private fun AnActionEvent.psiFileFromEditor(editor: Editor?): PsiFile? {
  getData(CommonDataKeys.PSI_FILE)?.let { return it }
  val project = project ?: return null
  val fallbackEditor = editor ?: return null
  return PsiDocumentManager.getInstance(project).getPsiFile(fallbackEditor.document)
}

class TidyKotlinArgumentCompletionDiagnosticsStartupActivity : StartupActivity.DumbAware {
  override fun runActivity(project: Project) {
    TidyKotlinArgumentCompletionDiagnostics.logStartup(project)
  }
}

internal object TidyKotlinArgumentCompletionDiagnostics {
  private val logger = Logger.getInstance("ai.hypergraph.tidyparse.kotlinArgumentCompletion")

  @Volatile
  private var lastUpdateLine: String? = null

  @Volatile
  private var lastUpdateNanos: Long = 0

  fun logStartup(project: Project) {
    val actionManager = ActionManager.getInstance()
    val activeKeymap = KeymapManager.getInstance().activeKeymap
    val shortcuts = activeKeymap.getShortcuts(KOTLIN_ARGUMENT_COMPLETION_ACTION_ID).toList()
    val shortcutTexts = shortcuts.map { KeymapUtil.getShortcutText(it) }.toSet()
    val conflictsByShortcut = shortcutTexts.associateWith {
      findShortcutConflicts(actionManager, activeKeymap, setOf(it))
    }

    log(
      "startup: project=${project.name} actionRegistered=" +
        "${actionManager.getAction(KOTLIN_ARGUMENT_COMPLETION_ACTION_ID) != null} " +
        "activeKeymap=${activeKeymap.name} shortcuts=${shortcutTexts.displayList()} " +
        "conflicts=${conflictsByShortcut.displayShortcutConflicts()}"
    )
  }

  fun logUpdate(e: AnActionEvent, editor: Editor?, psiFile: PsiFile?, ktFile: KtFile?, enabled: Boolean) {
    val line = "update: place=${e.place} project=${e.project?.name ?: "<none>"} editor=${editor != null} " +
      "psiFile=${psiFile?.javaClass?.name ?: "<none>"}:${psiFile?.name ?: "<none>"} " +
      "ktFile=${ktFile != null} file=${ktFile?.virtualFile?.path ?: psiFile?.name ?: "<none>"} " +
      "offset=${editor?.caretModel?.offset ?: -1} enabled=$enabled"

    val now = System.nanoTime()
    if (line != lastUpdateLine || now - lastUpdateNanos > 5_000_000_000L) {
      lastUpdateLine = line
      lastUpdateNanos = now
      log(line)
    }
  }

  fun log(message: String) {
    val line = "[TidyParse KotlinArg] $message"
    logger.info(line)
    System.err.println(line)
  }

  fun warn(message: String, throwable: Throwable) {
    val line = "[TidyParse KotlinArg] $message: ${throwable.javaClass.name}: ${throwable.message}"
    logger.warn(line, throwable)
    System.err.println(line)
    throwable.printStackTrace(System.err)
  }

  private fun findShortcutConflicts(
    actionManager: ActionManager,
    keymap: Keymap,
    shortcutTexts: Set<String>
  ): List<String> {
    if (shortcutTexts.isEmpty()) return emptyList()

    return actionManager.getActionIdList("")
      .asSequence()
      .filter { it != KOTLIN_ARGUMENT_COMPLETION_ACTION_ID }
      .mapNotNull { actionId ->
        val hasMatchingShortcut = keymap.getShortcuts(actionId)
          .map { KeymapUtil.getShortcutText(it) }
          .any { it in shortcutTexts }

        if (!hasMatchingShortcut) return@mapNotNull null

        val text = actionManager.getAction(actionId)?.templatePresentation?.text
        if (text.isNullOrBlank()) actionId else "$actionId ($text)"
      }
      .distinct()
      .take(20)
      .toList()
  }
}

internal fun buildKotlinArgumentCompletions(
  editor: Editor,
  psiFile: PsiFile,
  diagnostic: ((String) -> Unit)? = null
): KotlinArgumentCompletionResult? {
  diagnostic?.invoke(
    "build: psiFile=${psiFile.javaClass.name}:${psiFile.name} offset=${editor.caretModel.offset} " +
      "documentLength=${editor.document.textLength}"
  )

  val ktFile = psiFile as? KtFile ?: run {
    diagnostic?.invoke("build aborted: PSI file is not KtFile")
    return null
  }
  val nearbyElement = ktFile.findElementNear(editor.caretModel.offset)
  diagnostic?.invoke("build: nearbyElement=${nearbyElement?.diagnosticSummary() ?: "<none>"}")

  val slot = findKotlinArgumentSlot(ktFile, editor.caretModel.offset) ?: run {
    diagnostic?.invoke("build aborted: no Kotlin argument slot at offset=${editor.caretModel.offset}")
    return null
  }
  diagnostic?.invoke("build: slot replacementRange=${slot.replacementRange}")

  val probeFile = ktFile.withProbeAt(slot.replacementRange)
  val probeExpression = probeFile.findProbeExpression(slot.replacementRange.startOffset) ?: run {
    diagnostic?.invoke("build aborted: could not find probe expression in generated PSI")
    return null
  }
  diagnostic?.invoke("build: probeExpression=${probeExpression.diagnosticSummary()}")

  val completions = analyze(probeExpression) {
    val session = this
    val budget = KotlinArgumentSearchBudget()
    val expectedTypes = probeExpression.expectedType
      ?.let(::listOf)
      ?: probeExpression.expectedArgumentTypesFromContainingCall(session, slot, diagnostic)

    if (expectedTypes.isEmpty()) {
      diagnostic?.invoke("analyze: no expected type for probe expression")
      return@analyze null
    }
    diagnostic?.invoke("analyze: expectedTypes=$expectedTypes")

    val environment = probeFile.collectKotlinEnvironment(session, probeExpression, budget::keepGoing)
    diagnostic?.invoke("analyze: environment ${environment.diagnosticSummary()}")

    val candidates = enumerateTypedKotlinExpressionSequence(
      expectedType = expectedTypes.first(),
      environment = environment,
      maxLength = KOTLIN_MAX_COMPLETION_TOKENS,
      isSubtype = { actual, expected ->
        actual.hasSameRenderedKotlinType(expected) ||
          with(session) { actual.isSubtypeOf(expected, KaSubtypingErrorTypePolicy.LENIENT) }
      },
      resultMatches = { actual ->
        expectedTypes.any { expected ->
          actual.hasSameRenderedKotlinType(expected) ||
            with(session) { actual.isSubtypeOf(expected, KaSubtypingErrorTypePolicy.LENIENT) }
        }
      },
      membersOf = { receiver -> receiver.kotlinMembers(session, budget::keepGoing) },
      keepGoing = budget::keepGoing,
      diagnostic = diagnostic
    )
      .take(KOTLIN_MAX_COMPLETION_RESULTS)
      .toList()
    diagnostic?.invoke(
      "analyze: candidates=${candidates.size} preview=${candidates.take(10)} " +
        "elapsedMillis=${budget.elapsedMillis()} budgetExpired=${budget.expired}"
    )
    candidates
  }

  return when (completions) {
    null -> KotlinArgumentCompletionResult(slot.replacementRange, emptyList(), "No expected Kotlin type at the caret.")
    else -> KotlinArgumentCompletionResult(
      slot.replacementRange,
      completions,
      if (completions.isEmpty()) "No well-typed Kotlin argument expressions found in scope." else null
    )
  }
}

internal fun showTidyKotlinArgumentLookup(
  project: Project,
  editor: Editor,
  replacementRange: TextRange,
  completions: List<String>
) {
  val lookupStrings = completions.distinct().take(10)
  TidyKotlinArgumentCompletionDiagnostics.log(
    "showLookup requested: range=$replacementRange completions=${lookupStrings.size} preview=$lookupStrings"
  )
  if (lookupStrings.isEmpty()) {
    TidyKotlinArgumentCompletionDiagnostics.log("showLookup aborted: no lookup strings")
    return
  }

  val lookupElements = lookupStrings
    .map { LookupElementBuilder.create(it) }
    .toTypedArray<LookupElement>()

  val lookup = LookupManager.getInstance(project).showLookup(editor, lookupElements, "") ?: run {
    TidyKotlinArgumentCompletionDiagnostics.log("showLookup aborted: LookupManager returned null")
    return
  }
  TidyKotlinArgumentCompletionDiagnostics.log("showLookup shown")
  lookup.addLookupListener(object : LookupListener {
    override fun beforeItemSelected(event: LookupEvent): Boolean {
      val completion = event.item?.lookupString ?: return false
      TidyKotlinArgumentCompletionDiagnostics.log("lookup item selected: $completion")
      WriteCommandAction.runWriteCommandAction(project) {
        editor.document.replaceString(replacementRange.startOffset, replacementRange.endOffset, completion)
        editor.caretModel.moveToOffset(replacementRange.startOffset + completion.length)
      }
      lookup.hideLookup(true)
      return false
    }
  })
}

private fun Project.notifyKotlinArgumentCompletion(message: String) {
  TidyKotlinArgumentCompletionDiagnostics.log("notify: project=$name message=$message")
  NotificationGroupManager.getInstance()
    .getNotificationGroup("TidyParse")
    .createNotification("TidyParse Kotlin Argument Completion", message, NotificationType.INFORMATION)
    .notify(this)
}

internal fun <T> enumerateTypedKotlinExpressions(
  expectedType: T,
  environment: TypedKotlinEnvironment<T>,
  maxLength: Int = 10,
  maxResults: Int = 10,
  isSubtype: (actual: T, expected: T) -> Boolean,
  membersOf: (TypedKotlinExpression<T>) -> TypedKotlinEnvironment<T> = { TypedKotlinEnvironment() }
): List<String> {
  if (maxResults <= 0) return emptyList()
  return enumerateTypedKotlinExpressionSequence(
    expectedType = expectedType,
    environment = environment,
    maxLength = maxLength,
    isSubtype = isSubtype,
    membersOf = membersOf
  ).take(maxResults).toList()
}

internal fun <T> enumerateTypedKotlinExpressionSequence(
  expectedType: T,
  environment: TypedKotlinEnvironment<T>,
  maxLength: Int = 10,
  isSubtype: (actual: T, expected: T) -> Boolean,
  resultMatches: (actual: T) -> Boolean = { actual -> isSubtype(actual, expectedType) },
  membersOf: (TypedKotlinExpression<T>) -> TypedKotlinEnvironment<T> = { TypedKotlinEnvironment() },
  keepGoing: () -> Boolean = { true },
  typeKey: (T) -> String = { it.toString() },
  diagnostic: ((String) -> Unit)? = null
): Sequence<String> = sequence {
  if (maxLength <= 0) return@sequence

  val byLength = List(maxLength + 1) { mutableListOf<TypedKotlinExpression<T>>() }
  val seenExpressions = linkedSetOf<Pair<String, String>>()
  val yieldedTexts = linkedSetOf<String>()
  val memberCache = linkedMapOf<String, TypedKotlinEnvironment<T>>()
  var budgetStopLogged = false
  var memberLookupLogs = 0

  fun canContinue(): Boolean {
    val keepSearching = keepGoing()
    if (!keepSearching && !budgetStopLogged) {
      budgetStopLogged = true
      diagnostic?.invoke("enumerate: stopped by cancellation or search budget")
    }
    return keepSearching
  }

  fun addExpression(expression: TypedKotlinExpression<T>): Boolean {
    if (expression.length !in 1..maxLength) return false
    if (!seenExpressions.add(expression.text to typeKey(expression.type))) return false
    byLength[expression.length] += expression
    return true
  }

  fun argumentCombinations(
    parameters: List<T>,
    parameterIndex: Int,
    remainingLength: Int,
    current: MutableList<TypedKotlinExpression<T>>
  ): Sequence<List<TypedKotlinExpression<T>>> = sequence {
    if (!canContinue()) return@sequence
    if (parameterIndex == parameters.size) {
      if (remainingLength == 0) yield(current.toList())
      return@sequence
    }

    val remainingParameters = parameters.size - parameterIndex - 1
    val expectedParameterType = parameters[parameterIndex]
    val maxArgumentLength = remainingLength - remainingParameters
    if (maxArgumentLength < 1) return@sequence

    for (length in 1..maxArgumentLength) {
      if (!canContinue()) return@sequence
      for (expression in byLength[length].toList()) {
        if (!canContinue()) return@sequence
        if (!isSubtype(expression.type, expectedParameterType)) continue
        current += expression
        yieldAll(argumentCombinations(parameters, parameterIndex + 1, remainingLength - length, current))
        current.removeAt(current.lastIndex)
      }
    }
  }

  fun callExpressions(
    targetLength: Int,
    receiver: TypedKotlinExpression<T>?,
    functions: List<TypedKotlinCallable<T>>
  ): Sequence<TypedKotlinExpression<T>> = sequence {
    for (function in functions) {
      if (!canContinue()) return@sequence
      val baseLength = (receiver?.length ?: 0) + if (receiver == null) 3 else 4
      val commaLength = (function.parameters.size - 1).coerceAtLeast(0)
      val argumentsLength = targetLength - baseLength - commaLength
      if (argumentsLength < 0) continue

      if (function.parameters.isEmpty()) {
        if (argumentsLength == 0) {
          val prefix = receiver?.let { "${it.text}." } ?: ""
          yield(TypedKotlinExpression("$prefix${function.name}()", function.returnType, targetLength))
        }
        continue
      }

      for (arguments in argumentCombinations(function.parameters, 0, argumentsLength, mutableListOf())) {
        if (!canContinue()) return@sequence
        val prefix = receiver?.let { "${it.text}." } ?: ""
        val text = "$prefix${function.name}(${arguments.joinToString(", ") { it.text }})"
        yield(TypedKotlinExpression(text, function.returnType, targetLength))
      }
    }
  }

  fun membersFor(receiver: TypedKotlinExpression<T>): TypedKotlinEnvironment<T> {
    val key = typeKey(receiver.type)
    memberCache[key]?.let { return it }
    if (!canContinue()) return TypedKotlinEnvironment()

    val members = membersOf(receiver)
    memberCache[key] = members
    if (memberLookupLogs < 20) {
      diagnostic?.invoke("enumerate: receiverType=$key members ${members.diagnosticSummary()}")
      memberLookupLogs += 1
    }
    return members
  }

  diagnostic?.invoke(
    "enumerate: breadthFirst maxTokens=$maxLength values=${environment.values.size} " +
      "functions=${environment.functions.size}"
  )

  for (length in 1..maxLength) {
    if (!canContinue()) return@sequence

    if (length == 1) {
      for (value in environment.values) {
        if (!canContinue()) return@sequence
        val expression = TypedKotlinExpression(value.name, value.type, 1)
        if (addExpression(expression) && resultMatches(expression.type) && yieldedTexts.add(expression.text)) {
          yield(expression.text)
        }
      }
    }

    for (expression in callExpressions(length, receiver = null, environment.functions)) {
      if (addExpression(expression) && resultMatches(expression.type) && yieldedTexts.add(expression.text)) {
        yield(expression.text)
      }
    }

    for (receiverLength in 1..(length - 2)) {
      for (receiver in byLength[receiverLength].toList()) {
        if (!canContinue()) return@sequence
        val members = membersFor(receiver)
        if (receiver.length + 2 == length) {
          for (value in members.values) {
            if (!canContinue()) return@sequence
            val expression = TypedKotlinExpression("${receiver.text}.${value.name}", value.type, length)
            if (addExpression(expression) && resultMatches(expression.type) && yieldedTexts.add(expression.text)) {
              yield(expression.text)
            }
          }
        }

        for (expression in callExpressions(length, receiver, members.functions)) {
          if (addExpression(expression) && resultMatches(expression.type) && yieldedTexts.add(expression.text)) {
            yield(expression.text)
          }
        }
      }
    }
  }
}

private fun KtFile.withProbeAt(range: TextRange): KtFile {
  val probedText = text.replaceRange(range.startOffset, range.endOffset, KOTLIN_PROBE_NAME)
  return KtPsiFactory(this).createAnalyzableFile(name, probedText, this)
}

private fun KtFile.findProbeExpression(offset: Int): KtExpression? {
  val probeOffset = (offset + KOTLIN_PROBE_NAME.lastIndex).coerceIn(0, textLength - 1)
  return generateSequence(findElementAt(probeOffset)) { it.parent }
    .filterIsInstance<KtExpression>()
    .firstOrNull { it.text == KOTLIN_PROBE_NAME }
}

private fun KtExpression.expectedArgumentTypesFromContainingCall(
  session: KaSession,
  slot: KotlinArgumentSlot,
  diagnostic: ((String) -> Unit)?
): List<KaType> {
  val argument = parentsWithSelf().filterIsInstance<KtValueArgument>().firstOrNull() ?: run {
    diagnostic?.invoke("analyze fallback: probe is not inside a KtValueArgument")
    return emptyList()
  }
  val argumentExpression = argument.getArgumentExpression() ?: run {
    diagnostic?.invoke("analyze fallback: value argument has no expression")
    return emptyList()
  }
  val callElement = argument.parentsWithSelf().filterIsInstance<KtCallElement>().firstOrNull() ?: run {
    diagnostic?.invoke("analyze fallback: value argument is not inside a KtCallElement")
    return emptyList()
  }

  val resolvedFunctionCall = runCatching { with(session) { callElement.resolveCall() } }
    .getOrElse {
      diagnostic?.invoke("analyze fallback: call resolution failed for ${callElement.text.take(80)}: ${it.message}")
      null
    }

  val functionCalls = when (resolvedFunctionCall) {
    null -> {
      diagnostic?.invoke("analyze fallback: call resolution returned null for ${callElement.text.take(80)}")
      runCatching {
        with(session) {
          callElement.resolveToCallCandidates()
            .asSequence()
            .filter { it.isInBestCandidates }
            .mapNotNull { it.candidate as? KaFunctionCall<*> }
            .toList()
        }
      }.getOrElse {
        diagnostic?.invoke("analyze fallback: call candidate resolution failed: ${it.message}")
        emptyList()
      }
    }
    else -> listOf(resolvedFunctionCall)
  }

  val parameterTypes = functionCalls
    .flatMap { it.argumentTypes(argument, argumentExpression) }
    .ifEmpty { callElement.argumentTypesFromVisibleCallables(session, argument, diagnostic) }
    .ifEmpty { slot.argumentTypesFromVisibleCallables(session, diagnostic) }
    .distinctBy { it.toString() }
    .preferAnyType()

  diagnostic?.invoke(
    "analyze fallback: resolved call ${callElement.calleeExpression?.text ?: "<unknown>"} " +
      "candidates=${functionCalls.size} argumentTypes=${parameterTypes.ifEmpty { listOf("<none>") }}"
  )
  return parameterTypes
}

private fun KaFunctionCall<*>.argumentTypes(
  argument: KtValueArgument,
  argumentExpression: KtExpression
): List<KaType> =
  listOfNotNull(
    valueArgumentMapping[argumentExpression]?.returnType,
    combinedArgumentMapping[argumentExpression]?.returnType,
    valueArgumentMapping.entries.firstOrNull { (expression, _) ->
      expression.textRange == argumentExpression.textRange
    }?.value?.returnType,
    combinedArgumentMapping.entries.firstOrNull { (expression, _) ->
      expression.textRange == argumentExpression.textRange
    }?.value?.returnType,
    positionalArgumentType(argument)
  )

private fun KaFunctionCall<*>.positionalArgumentType(argument: KtValueArgument): KaType? {
  val argumentList = argument.parent as? KtValueArgumentList ?: return null
  val argumentIndex = argumentList.arguments.indexOf(argument).takeIf { it >= 0 } ?: return null
  return partiallyAppliedSymbol.signature.valueParameters.getOrNull(argumentIndex)?.returnType
    ?: signature.valueParameters.getOrNull(argumentIndex)?.returnType
}

private fun KtCallElement.argumentTypesFromVisibleCallables(
  session: KaSession,
  argument: KtValueArgument,
  diagnostic: ((String) -> Unit)?
): List<KaType> {
  val calleeName = calleeExpression?.text?.takeIf { it.isKotlinIdentifier() } ?: run {
    diagnostic?.invoke("analyze fallback: no simple callee name for scope lookup")
    return emptyList()
  }
  val argumentList = argument.parent as? KtValueArgumentList ?: return emptyList()
  val argumentIndex = argumentList.arguments.indexOf(argument).takeIf { it >= 0 } ?: return emptyList()
  val ktFile = containingFile as? KtFile ?: return emptyList()

  return with(session) {
    ktFile.scopeContext(this@argumentTypesFromVisibleCallables)
      .scopes
      .asSequence()
      .flatMap { scopeWithKind -> scopeWithKind.scope.callables { it.asString() == calleeName } }
      .filterIsInstance<KaNamedFunctionSymbol>()
      .filterNot { it.isExtension }
      .mapNotNull { it.valueParameters.getOrNull(argumentIndex)?.returnType }
      .toList()
  }
}

private fun KotlinArgumentSlot.argumentTypesFromVisibleCallables(
  session: KaSession,
  diagnostic: ((String) -> Unit)?
): List<KaType> {
  val calleeName = calleeName?.takeIf { it.isKotlinIdentifier() } ?: run {
    diagnostic?.invoke("analyze fallback: original slot has no simple callee name for scope lookup")
    return emptyList()
  }
  val anchor = scopeAnchor ?: run {
    diagnostic?.invoke("analyze fallback: original slot has no scope anchor")
    return emptyList()
  }
  val ktFile = anchor.containingFile as? KtFile ?: run {
    diagnostic?.invoke("analyze fallback: original scope anchor is not in a KtFile")
    return emptyList()
  }

  return with(session) {
    val scopes = ktFile.scopeContext(anchor).scopes
    val matchingCallables = scopes
      .asSequence()
      .flatMap { scopeWithKind -> scopeWithKind.scope.callables { it.asString() == calleeName } }
      .toList()

    diagnostic?.invoke(
      "analyze fallback: original scope lookup callee=$calleeName argumentIndex=$argumentIndex " +
        "scopes=${scopes.map { it.kind }} matchingCallables=${matchingCallables.size}"
    )

    matchingCallables
      .asSequence()
      .filterIsInstance<KaNamedFunctionSymbol>()
      .filterNot { it.isExtension }
      .mapNotNull { it.valueParameters.getOrNull(argumentIndex)?.returnType }
      .toList()
  }
}

private fun List<KaType>.preferAnyType(): List<KaType> =
  sortedWith(compareByDescending<KaType> { it.toString() == "kotlin/Any?" }.thenBy { it.toString() })

private fun KaType.hasSameRenderedKotlinType(other: KaType): Boolean =
  toString() == other.toString()

private fun findKotlinArgumentSlot(ktFile: KtFile, offset: Int): KotlinArgumentSlot? {
  val element = ktFile.findElementNear(offset) ?: return null
  val valueArgument = element.parentsWithSelf().filterIsInstance<KtValueArgument>().firstOrNull()
  val argumentList = valueArgument?.parent as? KtValueArgumentList
    ?: element.parentsWithSelf().filterIsInstance<KtValueArgumentList>().firstOrNull()
    ?: return null
  val callElement = argumentList.parentsWithSelf().filterIsInstance<KtCallElement>().firstOrNull()

  if (!argumentList.containsOffsetInArgumentArea(offset)) return null

  val argumentAtCaret = valueArgument ?: argumentList.arguments.firstOrNull {
    it.textRange.containsOffset(offset) || it.textRange.endOffset == offset
  }
  val expression = argumentAtCaret?.getArgumentExpression()
  val replacementRange = expression
    ?.textRange
    ?.takeIf { it.containsOffset(offset) || it.endOffset == offset }
    ?: TextRange(offset, offset)

  return KotlinArgumentSlot(
    replacementRange = replacementRange,
    calleeName = callElement?.calleeExpression?.text,
    argumentIndex = argumentList.argumentIndexAt(offset, argumentAtCaret),
    scopeAnchor = callElement ?: argumentList
  )
}

private fun KtValueArgumentList.containsOffsetInArgumentArea(offset: Int): Boolean {
  val left = leftParenthesis?.textRange?.endOffset ?: textRange.startOffset
  val right = rightParenthesis?.textRange?.startOffset ?: textRange.endOffset
  return offset in left..right
}

private fun KtValueArgumentList.argumentIndexAt(offset: Int, argumentAtCaret: KtValueArgument?): Int {
  val argumentIndex = argumentAtCaret?.let { arguments.indexOf(it) } ?: -1
  if (argumentIndex >= 0) return argumentIndex

  val left = leftParenthesis?.textRange?.endOffset ?: textRange.startOffset
  val boundedOffset = offset.coerceIn(left, textRange.endOffset)
  return containingFile.text
    .substring(left, boundedOffset)
    .count { it == ',' }
}

private fun PsiFile.findElementNear(offset: Int): PsiElement? =
  sequenceOf(offset, offset - 1, offset + 1)
    .filter { it in 0 until textLength }
    .mapNotNull { findElementAt(it) }
    .firstOrNull()

private fun PsiElement.parentsWithSelf(): Sequence<PsiElement> =
  generateSequence(this) { it.parent }

private fun PsiElement.diagnosticSummary(): String =
  "${javaClass.simpleName} range=$textRange text='${text.replace("\n", "\\n").take(80)}'"

private fun <T> TypedKotlinEnvironment<T>.diagnosticSummary(): String {
  val valueNames = values.take(20).joinToString { it.name }.ifBlank { "<none>" }
  val functionNames = functions.take(20).joinToString { "${it.name}/${it.parameters.size}" }.ifBlank { "<none>" }
  val valueSuffix = if (values.size > 20) ", ..." else ""
  val functionSuffix = if (functions.size > 20) ", ..." else ""
  return "values=${values.size}[$valueNames$valueSuffix] functions=${functions.size}[$functionNames$functionSuffix]"
}

private fun Collection<String>.displayList(): String =
  if (isEmpty()) "<none>" else joinToString(prefix = "[", postfix = "]")

private fun Map<String, Collection<String>>.displayShortcutConflicts(): String =
  if (isEmpty()) {
    "<none>"
  } else {
    entries.joinToString(prefix = "[", postfix = "]") { (shortcut, conflicts) ->
      "$shortcut -> ${conflicts.displayList()}"
    }
  }

private fun KtFile.collectKotlinEnvironment(
  session: KaSession,
  anchor: KtExpression,
  keepGoing: () -> Boolean = { true }
): TypedKotlinEnvironment<KaType> {
  val values = linkedMapOf<String, TypedKotlinValue<KaType>>()
  val functions = linkedMapOf<String, TypedKotlinCallable<KaType>>()

  fun add(symbol: CompletionSymbol) {
    when (symbol) {
      is CompletionSymbol.Value -> values.putIfAbsent(symbol.key, TypedKotlinValue(symbol.name, symbol.type))
      is CompletionSymbol.Function -> functions.putIfAbsent(
        symbol.key,
        TypedKotlinCallable(symbol.name, symbol.parameters, symbol.returnType)
      )
    }
  }

  with(session) {
    val scopeContext = this@collectKotlinEnvironment.scopeContext(anchor)

    for ((index, receiver) in scopeContext.implicitReceivers.withIndex()) {
      if (!keepGoing()) return@with
      val type = runCatching { receiver.type }.getOrNull() ?: continue
      values.putIfAbsent("scope:value:this:$index:${type.hashCode()}", TypedKotlinValue("this", type))
    }

    scopeLoop@ for (scopeWithKind in scopeContext.scopes) {
      if (!keepGoing()) break
      if (scopeWithKind.kind.isDefaultImportScope()) continue
      for (callable in scopeWithKind.scope.callables { true }) {
        if (!keepGoing()) break@scopeLoop
        callable.asCompletionSymbol(anchor, KOTLIN_MAX_COMPLETION_PARAMETERS)?.let(::add)
      }
    }

    for (property in PsiTreeUtil.collectElementsOfType(this@collectKotlinEnvironment, KtProperty::class.java)) {
      if (!keepGoing()) return@with
      if (!property.isVisibleAt(anchor)) continue
      val name = property.name?.takeIf { it.isKotlinIdentifier() } ?: continue
      val type = runCatching { property.returnType }.getOrNull() ?: continue
      values.putIfAbsent("psi:value:$name", TypedKotlinValue(name, type))
    }

    for (parameter in PsiTreeUtil.collectElementsOfType(this@collectKotlinEnvironment, KtParameter::class.java)) {
      if (!keepGoing()) return@with
      if (!parameter.isVisibleAt(anchor)) continue
      val name = parameter.name?.takeIf { it.isKotlinIdentifier() } ?: continue
      val type = runCatching { parameter.returnType }.getOrNull() ?: continue
      values.putIfAbsent("psi:value:$name", TypedKotlinValue(name, type))
    }

    for (function in PsiTreeUtil.collectElementsOfType(this@collectKotlinEnvironment, KtNamedFunction::class.java)) {
      if (!keepGoing()) return@with
      if (!function.isVisibleAt(anchor)) continue
      if (function.valueParameters.size > KOTLIN_MAX_COMPLETION_PARAMETERS) continue
      val name = function.name?.takeIf { it.isKotlinIdentifier() } ?: continue
      val parameters = mutableListOf<KaType>()
      var hasUnsupportedParameter = false
      for (parameter in function.valueParameters) {
        val type = runCatching { parameter.returnType }.getOrNull()
        if (type == null) {
          hasUnsupportedParameter = true
          break
        }
        parameters += type
      }
      if (hasUnsupportedParameter) continue
      val returnType = runCatching { function.returnType }.getOrNull() ?: continue
      val callable = TypedKotlinCallable(name, parameters, returnType)
      functions.putIfAbsent("psi:function:${callable.signatureKey()}", callable)
    }
  }

  return TypedKotlinEnvironment(values.values.toList(), functions.values.toList())
}

private fun KtParameter.isVisibleAt(anchor: KtExpression): Boolean {
  val ownerFunction = ownerFunction
  if (ownerFunction != null) return ownerFunction.textRange.containsOffset(anchor.textOffset)

  val ownerDeclaration = ownerDeclaration
  return ownerDeclaration?.textRange?.containsOffset(anchor.textOffset) == true
}

private fun KtProperty.isVisibleAt(anchor: KtExpression): Boolean {
  val anchorOffset = anchor.textOffset
  if (textRange.containsOffset(anchorOffset)) return false

  val enclosingFunction = strictParents().filterIsInstance<KtNamedFunction>().firstOrNull()
  val anchorFunction = anchor.strictParents().filterIsInstance<KtNamedFunction>().firstOrNull()
  return enclosingFunction == null || (enclosingFunction == anchorFunction && textRange.endOffset <= anchorOffset)
}

private fun KtNamedFunction.isVisibleAt(anchor: KtExpression): Boolean {
  val anchorOffset = anchor.textOffset
  if (textRange.containsOffset(anchorOffset)) return false

  val enclosingFunction = strictParents().filterIsInstance<KtNamedFunction>().firstOrNull()
  val anchorFunction = anchor.strictParents().filterIsInstance<KtNamedFunction>().firstOrNull()
  return enclosingFunction == null || (enclosingFunction == anchorFunction && textRange.endOffset <= anchorOffset)
}

private fun PsiElement.strictParents(): Sequence<PsiElement> =
  generateSequence(parent) { it.parent }

private fun KaScopeKind.isDefaultImportScope(): Boolean =
  this is KaScopeKind.DefaultSimpleImportingScope || this is KaScopeKind.DefaultStarImportingScope

private fun TypedKotlinExpression<KaType>.kotlinMembers(
  session: KaSession,
  keepGoing: () -> Boolean = { true }
): TypedKotlinEnvironment<KaType> {
  if (!keepGoing()) return TypedKotlinEnvironment()
  val memberScope = with(session) { type.scope } ?: return TypedKotlinEnvironment()
  return memberScope.collectMemberEnvironment(session, keepGoing)
}

private fun KaTypeScope.collectMemberEnvironment(
  session: KaSession,
  keepGoing: () -> Boolean = { true }
): TypedKotlinEnvironment<KaType> {
  val values = mutableListOf<TypedKotlinValue<KaType>>()
  val functions = mutableListOf<TypedKotlinCallable<KaType>>()

  with(session) {
    val seen = mutableSetOf<String>()
    for (signature in getCallableSignatures { true }) {
      if (!keepGoing()) break
      val symbol = signature.asMemberCompletionSymbol(KOTLIN_MAX_COMPLETION_PARAMETERS) ?: continue
      if (!seen.add(symbol.key)) continue
      when (symbol) {
        is CompletionSymbol.Value -> values += TypedKotlinValue(symbol.name, symbol.type)
        is CompletionSymbol.Function -> functions += TypedKotlinCallable(symbol.name, symbol.parameters, symbol.returnType)
      }
    }
  }

  return TypedKotlinEnvironment(values, functions)
}

private sealed class CompletionSymbol {
  abstract val key: String
  abstract val name: String

  data class Value(
    override val key: String,
    override val name: String,
    val type: KaType
  ) : CompletionSymbol()

  data class Function(
    override val key: String,
    override val name: String,
    val parameters: List<KaType>,
    val returnType: KaType
  ) : CompletionSymbol()
}

private fun <T> TypedKotlinCallable<T>.signatureKey(): String =
  "$name/${parameters.joinToString(",") { it.toString() }}:$returnType"

private fun KaCallableSymbol.asCompletionSymbol(anchor: KtExpression, maxParameters: Int): CompletionSymbol? {
  if (isExtension) return null

  val name = (this as? KaNamedSymbol)?.name?.asString()?.takeIf { it.isKotlinIdentifier() } ?: return null
  val psi = runCatching { psi }.getOrNull()
  if (psi?.containingFile == anchor.containingFile && psi.textRange.containsOffset(anchor.textOffset)) return null

  return when (this) {
    is KaVariableSymbol -> CompletionSymbol.Value("scope:value:$name:${returnType.hashCode()}", name, returnType)
    is KaNamedFunctionSymbol -> {
      if (valueParameters.size > maxParameters) return null
      val parameterTypes = valueParameters.map { it.returnType }
      CompletionSymbol.Function(
        key = "scope:function:$name:${parameterTypes.joinToString(",")}:$returnType",
        name = name,
        parameters = parameterTypes,
        returnType = returnType
      )
    }
    else -> null
  }
}

private fun org.jetbrains.kotlin.analysis.api.signatures.KaCallableSignature<*>.asMemberCompletionSymbol(
  maxParameters: Int
): CompletionSymbol? {
  val symbol = symbol
  val name = (symbol as? KaNamedSymbol)?.name?.asString()?.takeIf { it.isKotlinIdentifier() } ?: return null

  return when (this) {
    is KaVariableSignature<*> -> CompletionSymbol.Value("member:value:$name:${returnType.hashCode()}", name, returnType)
    is KaFunctionSignature<*> -> {
      if (valueParameters.size > maxParameters) return null
      val parameterTypes = valueParameters.map { it.returnType }
      CompletionSymbol.Function(
        key = "member:function:$name:${parameterTypes.joinToString(",")}:$returnType",
        name = name,
        parameters = parameterTypes,
        returnType = returnType
      )
    }
  }
}

private fun String.isKotlinIdentifier(): Boolean =
  matches(Regex("[A-Za-z_][A-Za-z0-9_]*")) && this != KOTLIN_PROBE_NAME
