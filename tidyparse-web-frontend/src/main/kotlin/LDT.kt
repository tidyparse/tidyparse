import kotlinx.browser.*
import org.w3c.dom.*

class Parser(private val ruleMap: Map<String, Regex>) {
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

class TextareaDecorator(inputField: HTMLTextAreaElement, private val parser: Parser) {
  private val output: HTMLPreElement = document.createElement("pre") as HTMLPreElement

  init {
    // Construct editor DOM
    val parent = document.createElement("div") as HTMLDivElement
    parent.apply { className = "ldt $className" }.appendChild(output)
    val label = document.createElement("label") as HTMLLabelElement
    parent.appendChild(label)

    inputField.apply {
      // Replace the textarea with RTA DOM and reattach on label
      parentNode?.replaceChild(parent, this)
      label.appendChild(this)

      // Transfer the CSS styles to our editor
      className = ""
      spellcheck = false
      wrap = "off"

      // Detect all changes to the textarea
      addEventListener("input", { update() })

      // Initial highlighting
      update()
    }
  }

  // Surrounds the given line with <u>...</u> in output like color() does
  fun underline(lineNumber: Int) {
    output.innerHTML.also { println("HTML:\n\n$it") }.split('\n')
      .mapIndexed { index, s -> if (index == lineNumber) "<u>$s</u>" else s }
      .joinToString("\n").also { output.innerHTML = it }
  }

  private fun HTMLTextAreaElement.update() =
    if (value.isNotEmpty()) {
      color(value)
      // Determine the best size for the textarea
      val lines = value.split('\n')
      val maxlen = lines.maxOfOrNull { line ->
        line.length + line.count { it == '\t' } * 7 // Approximation for tab length
      } ?: 0
      cols = maxlen + 1
      rows = lines.size + 2
    } else {
      // Clear the display
      output.innerHTML = ""
      cols = 1
      rows = 1
    }

  private fun color(input: String) {
    val oldTokens = output.childNodes.asList()
    val newTokens = parser.tokenize(input)
    val firstDiff = newTokens.zip(oldTokens)
      .indexOfFirst { (new, old) -> new != (old as? HTMLElement)?.textContent }
      .let { if (it == -1) minOf(newTokens.size, oldTokens.size) else it }

    // Trim the length of output nodes to the size of the input
    while (newTokens.size < oldTokens.size) output.removeChild(oldTokens[firstDiff])

    // Update modified spans
    for (index in firstDiff until oldTokens.size)
      (oldTokens[index] as HTMLElement).apply {
        className = parser.identify(newTokens[index]) ?: ""
        textContent = newTokens[index]
      }

    // Add in new spans
    for (index in oldTokens.size until newTokens.size)
      output.appendChild(
        (document.createElement("span") as HTMLElement).apply {
          className = parser.identify(newTokens[index]) ?: ""
          textContent = newTokens[index]
        }
      )
  }
}