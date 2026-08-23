import kotlinx.browser.window
import org.w3c.dom.events.KeyboardEvent

internal object BackspaceDebouncer {
  private var pressed = false
  private val pending = mutableMapOf<Any, () -> Unit>()

  init {
    window.addEventListener("keydown", { event ->
      if ((event as KeyboardEvent).key == "Backspace") pressed = true
    }, true)
    window.addEventListener("keyup", { event ->
      if ((event as KeyboardEvent).key == "Backspace") release()
    }, true)
    window.addEventListener("blur", { release() })
  }

  /** Runs immediately, or once Backspace is released if it is currently held. */
  fun submit(owner: Any, block: () -> Unit): Boolean = if (pressed) {
    pending[owner] = block
    true
  } else {
    block()
    false
  }

  private fun release() {
    pressed = false
    pending.values.toList().also { pending.clear() }.forEach { it() }
  }
}