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

/**
 * CodeMirror squiggle-only decorator.
 *
 * - Does NOT do token coloring (CodeMirror handles highlighting).
 * - Only applies squiggly underline marks.
 * - Subclasses TextareaDecorator without mutating the real DOM (dummy textarea trick).
 *
 * Drive it by setting [isLineInvalid] (or [invalidLines]) before calling quick/fullDecorate.
 */

class PyTextareaDecorator(
  private val realInput: HTMLTextAreaElement,
  private val cm: dynamic = window.asDynamic().tidyCm,
  parser: Parser
) : TextareaDecorator(
  inputField = (document.createElement("textarea") as HTMLTextAreaElement),
  parser = parser
) {
  private val marks = mutableListOf<dynamic>()

  /** Preferred: set this from your editor logic (fast, no re-parsing here). */
  var isLineInvalid: (Int, String) -> Boolean = { _, _ -> false }

  /** Optional convenience: if you already have a set of invalid line numbers. */
  var invalidLines: Set<Int>? = null

  private fun clearMarks() {
    if (cm == null) { marks.clear(); return }
    try {
      cm.operation {
        for (m in marks) try { m.clear() } catch (_: dynamic) {}
        marks.clear()
      }
    } catch (_: dynamic) {
      marks.clear()
    }
  }

  private fun squiggleWholeLine(lineNo: Int, line: String) {
    if (line.isEmpty()) return
    val doc = cm.getDoc()
    val from = js("({line: lineNo, ch: 0})")
    val to = js("({line: lineNo, ch: line.length})")
    val opts = js("({className: 'cm-squiggle'})")
    val m = doc.markText(from, to, opts)
    marks.add(m)
  }

  private fun shouldSquiggle(lineNo: Int, line: String): Boolean {
    invalidLines?.let { return lineNo in it }
    return isLineInvalid(lineNo, line)
  }

  /** Decorate only viewport lines (fast). */
  override fun quickDecorate() {
    if (cm == null) return
    clearMarks()

    val doc = cm.getDoc()
    val vp = cm.getViewport() // {from: Int, to: Int}
    val fromLine = (vp.from as Int)
    val toLineExcl = (vp.to as Int)

    cm.operation {
      for (ln in fromLine until toLineExcl) {
        val line = doc.getLine(ln) as String
        if (shouldSquiggle(ln, line)) squiggleWholeLine(ln, line)
      }
    }
  }

  /** Decorate all lines (slower). */
  override fun fullDecorate(cfg: CFG) {
    if (cm == null) return
    clearMarks()

    val doc = cm.getDoc()
    val n = doc.lineCount() as Int

    cm.operation {
      for (ln in 0 until n) {
        val line = doc.getLine(ln) as String
        if (shouldSquiggle(ln, line)) squiggleWholeLine(ln, line)
      }
    }
  }

  /**
   * Optional: if you later want squiggles on *ranges* (e.g., token spans, columns),
   * call this directly and skip full/quickDecorate.
   */
  fun setSquiggleRanges(ranges: List<Triple<Int, Int, Int>>) {
    // Triple(lineNo, fromCh, toChExclusive)
    if (cm == null) return
    clearMarks()
    val doc = cm.getDoc()
    cm.operation {
      for ((ln, a, b) in ranges) {
        if (b <= a) continue
        val from = js("({line: ln, ch: a})")
        val to = js("({line: ln, ch: b})")
        val opts = js("({className: 'cm-squiggle'})")
        marks.add(doc.markText(from, to, opts))
      }
    }
  }
}