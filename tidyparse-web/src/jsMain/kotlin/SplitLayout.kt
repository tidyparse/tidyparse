import kotlinx.browser.document
import kotlinx.browser.window
import org.w3c.dom.Element
import org.w3c.dom.HTMLDivElement
import org.w3c.dom.HTMLElement
import org.w3c.dom.events.KeyboardEvent
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private const val DEFAULT_INPUT_SHARE = 48.0
private const val MIN_INPUT_SHARE = 30.0
private const val MAX_INPUT_SHARE = 75.0

fun initSplitLayout() {
  val content = document.getElementById("content") as? HTMLElement ?: return
  val output = document.getElementById("tidyparse-output") as? HTMLElement ?: return
  if (output.parentNode != content) return

  content.classList.add("workspace-split")
  val divider = getOrCreateWorkspaceDivider(content, output)

  setInputShare(content, content.inputShareFromDataset() ?: DEFAULT_INPUT_SHARE)

  fun updateFromClientX(clientX: Double) {
    val rect = content.getBoundingClientRect()
    if (rect.width <= 0.0) return
    setInputShare(content, ((clientX - rect.left) / rect.width) * 100)
  }

  divider.addEventListener("pointerdown", { event ->
    event.preventDefault()
    val pointerEvent = event.asDynamic()
    divider.asDynamic().setPointerCapture(pointerEvent.pointerId)
    updateFromClientX((pointerEvent.clientX as Number).toDouble())
  })

  divider.addEventListener("pointermove", { event ->
    val pointerEvent = event.asDynamic()
    if (divider.asDynamic().hasPointerCapture(pointerEvent.pointerId) as Boolean) {
      updateFromClientX((pointerEvent.clientX as Number).toDouble())
    }
  })

  divider.addEventListener("keydown", { event ->
    val keyEvent = event as KeyboardEvent
    if (keyEvent.key != "ArrowLeft" && keyEvent.key != "ArrowRight") return@addEventListener

    keyEvent.preventDefault()
    val current = content.inputShareFromDataset() ?: DEFAULT_INPUT_SHARE
    val delta = if (keyEvent.shiftKey) 5.0 else 1.0
    setInputShare(content, current + if (keyEvent.key == "ArrowRight") delta else -delta)
  })

  window.addEventListener("resize", { refreshCodeMirror() })
}

private fun getOrCreateWorkspaceDivider(content: HTMLElement, output: Element): HTMLDivElement {
  val existing = document.getElementById("workspace-divider") as? HTMLDivElement
  val divider = existing ?: (document.createElement("div") as HTMLDivElement).apply {
    id = "workspace-divider"
    className = "workspace-divider"
    setAttribute("role", "separator")
    setAttribute("aria-label", "Resize input and output panels")
    setAttribute("aria-orientation", "vertical")
    tabIndex = 0
  }

  if (divider.parentNode != content || divider.nextSibling != output) content.insertBefore(divider, output)
  return divider
}

private fun setInputShare(content: HTMLElement, percent: Double) {
  val clamped = max(MIN_INPUT_SHARE, min(MAX_INPUT_SHARE, percent))
  val rounded = (clamped * 100).roundToInt() / 100.0
  document.body?.style?.setProperty("--workspace-input-share", "$rounded%")
  content.asDynamic().dataset.inputShare = rounded.toString()
  refreshCodeMirror()
}

private fun HTMLElement.inputShareFromDataset(): Double? =
  (asDynamic().dataset?.inputShare as? String)?.toDoubleOrNull()

private fun refreshCodeMirror() {
  window.requestAnimationFrame {
    val editor = window.asDynamic().cmEditor
    val refresh = editor?.refresh
    if (editor != null && editor != js("undefined") && refresh != null && refresh != js("undefined")) {
      editor.refresh()
    }
  }
}
