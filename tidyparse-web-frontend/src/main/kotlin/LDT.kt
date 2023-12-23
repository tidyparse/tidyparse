import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
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

class TextareaDecorator(val inputField: HTMLTextAreaElement, private val parser: Parser) {
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
    }
  }

  fun quickDecorate() {
    val sb = StringBuilder()
    var lines: Int
    var maxLen = 0
    inputField.value.lines().also { lines = it.size }.forEach { line ->
      if (line.length > maxLen) maxLen = line.length
      sb.appendLine(line.toColorfulHTML())
    }

    output.innerHTML = sb.toString()
    inputField.cols = maxLen + 1
    inputField.rows = lines + 2
  }

  fun fullDecorate(cfg: CFG = emptySet()) {
    val sb = StringBuilder()
    var lines: Int
    var maxLen = 0
    inputField.value.lines().also { lines = it.size }.forEach { line ->
      if (line.length > maxLen) maxLen = line.length
      sb.appendLine(segmentationCacheHTML.getOrElse(cfg.hashCode() + line.hashCode()) { line.toColorfulHTML() })
    }

    output.innerHTML = sb.toString()
    inputField.cols = maxLen + 1
    inputField.rows = lines + 2
  }

  private fun String.toColorfulHTML() =
    tokenizeByWhitespaceAndKeepDelimiters().joinToString("") { token ->
      val escapedToken = token.escapeHTML()
      parser.identify(token)?.let { "<span class=\"$it\">$escapedToken</span>" } ?: escapedToken
    }
}