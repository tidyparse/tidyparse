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
  private val cmMarks = mutableListOf<dynamic>()

  private val codeMirror: dynamic
    get() = window.asDynamic().cmEditor

  private fun hasCodeMirror(): Boolean = codeMirror != null && codeMirror != js("undefined")

  init {
    if (hasCodeMirror()) {
      inputField.className = ""
      inputField.spellcheck = false
      inputField.wrap = "off"
    } else {
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
  }

  open fun quickDecorate() {
    if (hasCodeMirror()) {
      decorateCodeMirrorLexically()
      return
    }

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
    if (hasCodeMirror()) {
      decorateCodeMirror(cfg)
      return
    }

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

  private fun decorateCodeMirrorLexically() {
    val cm = codeMirror
    if (cm == null || cm == js("undefined")) return

    cm.operation {
      clearCodeMirrorMarks()

      readCodeMirrorLines().forEachIndexed { lineNo, line ->
        markCodeMirrorTokens(lineNo, line)
      }
    }
  }

  private fun decorateCodeMirror(cfg: CFG) {
    val cm = codeMirror
    if (cm == null || cm == js("undefined")) return

    val lines = readCodeMirrorLines()
    val separatorLine = lines.indexOfFirst { it.trim() == "---" }

    cm.operation {
      clearCodeMirrorMarks()

      lines.forEachIndexed { lineNo, line ->
        if (shouldUseSegmentation(cfg, lineNo, line, separatorLine)) markSegmentedLine(cfg, lineNo, line)
        else markCodeMirrorTokens(lineNo, line)
      }
    }
  }

  private fun clearCodeMirrorMarks() {
    cmMarks.forEach {
      try { it.clear() } catch (_: dynamic) {}
    }
    cmMarks.clear()
  }

  private fun shouldUseSegmentation(cfg: CFG, lineNo: Int, line: String, separatorLine: Int): Boolean =
    cfg.isNotEmpty() && line.isNotBlank() && !line.containsHole() && (separatorLine == -1 || separatorLine < lineNo)

  private fun markSegmentedLine(cfg: CFG, lineNo: Int, line: String) {
    val leading = line.length - line.trimStart().length
    val trimmed = line.trim()
    val segmentation = segmentationCache.getOrPut(cfg.hashCode() + trimmed.hashCode()) {
      Segmentation.build(cfg, trimmed)
    }
    val invalidRegions = segmentation.unparseableRegions + segmentation.illegalRegions

    if (invalidRegions.isEmpty()) return

    val markedRegions = segmentation.parseableRegions + invalidRegions
    val underlineStart = markedRegions.minOf { it.first } + leading
    val underlineEnd = markedRegions.maxOf { it.last } + leading + 1
    markCodeMirrorRange(lineNo, underlineStart, underlineEnd, "cm-invalid-snippet")

    segmentation.unparseableRegions.forEach { markCodeMirrorRange(lineNo, it.first + leading, it.last + leading + 1, "cm-orange") }
    segmentation.illegalRegions.forEach { markCodeMirrorRange(lineNo, it.first + leading, it.last + leading + 1, "cm-red") }
  }

  private fun markCodeMirrorTokens(lineNo: Int, line: String) {
    var ch = 0
    line.tokenizeByWhitespaceAndKeepDelimiters().forEach { token ->
      val fromCh = ch
      ch += token.length
      val cls = parser.identify(token)?.takeIf { it in setOf("red", "orange", "yellow", "green", "blue", "gray") }
        ?: return@forEach
      if (token.isBlank()) return@forEach

      markCodeMirrorRange(lineNo, fromCh, ch, "cm-$cls")
    }
  }

  private fun markCodeMirrorRange(lineNo: Int, fromCh: Int, toCh: Int, className: String) {
    if (fromCh >= toCh) return

    val opts = js("{}")
    opts.className = className
    cmMarks.add(codeMirror.markText(codeMirrorPos(lineNo, fromCh), codeMirrorPos(lineNo, toCh), opts))
  }

  private fun readCodeMirrorLines(): List<String> =
    (codeMirror.getValue() as String).lines()

  private fun codeMirrorPos(line: Int, ch: Int): dynamic =
    window.asDynamic().CodeMirror.Pos(line, ch)
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
