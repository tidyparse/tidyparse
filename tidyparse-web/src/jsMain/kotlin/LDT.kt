import ai.hypergraph.kaliningraph.image.escapeHTML
import ai.hypergraph.kaliningraph.parsing.*
import ai.hypergraph.kaliningraph.tokenizeByWhitespaceAndKeepDelimiters
import kotlinx.browser.*
import org.w3c.dom.*

class Parser(private val ruleMap: Map<String, Regex>) {
  constructor(vararg rules: Pair<String, String>) : this(
    rules.associate { (color, regex) -> color to Regex(regex) }
  )
  private val parseRE: Regex =
    ruleMap.values.joinToString("|") { it.pattern }.toRegex()

  fun tokenize(input: String): List<String> =
    parseRE.findAll(input).map { it.value }.toList()

  fun findAll(input: String): Sequence<MatchResult> = parseRE.findAll(input)

  fun identify(token: String): String? =
    ruleMap.entries.firstOrNull { it.value.matches(token) }?.key
}

open class TextareaDecorator(val inputField: HTMLTextAreaElement, private val parser: Parser) {
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

  open fun quickDecorate() {
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

  open fun fullDecorate(cfg: CFG = emptySet()) {
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

class PyTextareaDecorator(private val cm: dynamic) {
  private val GUTTER_ID = "cm-warn-gutter"

  fun tidyLintObj(): dynamic {
    val w = window.asDynamic()
    if (w.__tidyLint == null) w.__tidyLint = js("({ version: 0, lines: new Map() })")
    return w.__tidyLint
  }

  fun setInvalidLines(lines: Map<Int, String>) {
    val o = tidyLintObj()
    o.version = (o.version as Int) + 1
    o.lines.clear()
    lines.forEach { (ln, msg) -> o.lines.set(ln, msg) }
  }

  private fun lintCache(): dynamic = tidyLintObj()

  private fun clearGutterMarkers() {
    if (cm == null) return
    val doc = cm.getDoc()
    val n = doc.lineCount() as Int
    cm.operation { for (ln in 0 until n) { cm.setGutterMarker(ln, GUTTER_ID, null) } }
  }

  private fun setWarnMarker(lineNo: Int, message: String) {
    val marker = document.createElement("div") as HTMLElement
    marker.className = "CodeMirror-lint-marker CodeMirror-lint-marker-warning"
    marker.setAttribute("title", message) // tooltip
    cm.setGutterMarker(lineNo, GUTTER_ID, marker)
  }

  fun fullDecorate() {
    if (cm == null) return

    try {
      val gutters = (cm.getOption("gutters") as Array<dynamic>).toMutableList()
      if (!gutters.contains(GUTTER_ID)) {
        gutters.add(GUTTER_ID)
        cm.setOption("gutters", gutters.toTypedArray())
      }
    } catch (_: dynamic) {}

    clearGutterMarkers()

    val cache = lintCache() ?: return
    val lines = cache.lines ?: return

    cm.operation {
      val it = lines.entries()
      while (true) {
        val step = it.next()
        if (step.done as Boolean) break
        val entry = step.value
        val ln = (entry[0] as Int)
        val msg = (entry[1] as String)
        setWarnMarker(ln, msg)
      }
    }
  }
}