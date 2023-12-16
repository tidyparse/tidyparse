import kotlinx.browser.*
import org.w3c.dom.*

class Parser(val ruleMap: Map<String, Regex>) {
  constructor(vararg rules: Pair<String, String>): this(
    rules.associate { (color, regex) -> color to Regex(regex) }
  )
  private val parseRE: Regex =
    ruleMap.values.joinToString("|") { it.pattern }.toRegex()

  fun tokenize(input: String): List<String> =
    parseRE.findAll(input).map { it.value }.toList()

  fun identify(token: String): String? =
    ruleMap.entries.firstOrNull { it.value.matches(token) }?.key
}

class TextareaDecorator(private val textarea: HTMLTextAreaElement, private val parser: Parser) {
  private val output: HTMLPreElement

  init {
    // Construct editor DOM
    val parent = document.createElement("div") as HTMLDivElement
    output = document.createElement("pre") as HTMLPreElement
    parent.appendChild(output)
    val label = document.createElement("label") as HTMLLabelElement
    parent.appendChild(label)

    // Replace the textarea with RTA DOM and reattach on label
    textarea.parentNode?.replaceChild(parent, textarea)
    label.appendChild(textarea)

    // Transfer the CSS styles to our editor
    parent.className = "ldt ${textarea.className}"
    textarea.className = ""
    textarea.spellcheck = false
    textarea.wrap = "off"

    // Detect all changes to the textarea
    textarea.addEventListener("input", { update() })

    // Initial highlighting
    update()
  }

  private fun update() {
    val input = textarea.value
    if (input.isNotEmpty()) {
      color(input)
      // Determine the best size for the textarea
      val lines = input.split('\n')
      val maxlen = lines.maxOfOrNull { line ->
        line.length + line.count { it == '\t' } * 7 // Approximation for tab length
      } ?: 0
      textarea.cols = maxlen + 1
      textarea.rows = lines.size + 2
    } else {
      // Clear the display
      output.innerHTML = ""
      textarea.cols = 1
      textarea.rows = 1
    }
  }

  private fun color(input: String) {
    val oldTokens = output.childNodes.asList()
    val newTokens = parser.tokenize(input)
    var firstDiff = newTokens.zip(oldTokens)
      .indexOfFirst { (new, old) -> new != (old as? HTMLElement)?.textContent }
    if (firstDiff == -1) firstDiff = minOf(newTokens.size, oldTokens.size)

    // Trim the length of output nodes to the size of the input
    while (newTokens.size < oldTokens.size) {
      output.removeChild(oldTokens[firstDiff])
    }

    // Update modified spans
    for (index in firstDiff until oldTokens.size) {
      val oldNode = oldTokens[index] as HTMLElement
      oldNode.className = parser.identify(newTokens[index]) ?: ""
      oldNode.textContent = newTokens[index]
    }

    // Add in new spans
    for (index in oldTokens.size until newTokens.size) {
      val span = document.createElement("span") as HTMLElement
      span.className = parser.identify(newTokens[index]) ?: ""
      span.textContent = newTokens[index]
      output.appendChild(span)
    }
  }
}