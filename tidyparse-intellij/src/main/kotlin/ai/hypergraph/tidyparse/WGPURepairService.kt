package ai.hypergraph.tidyparse

import com.intellij.openapi.Disposable
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.util.Disposer
import com.intellij.ui.jcef.JBCefApp
import com.intellij.ui.jcef.JBCefBrowser
import com.intellij.ui.jcef.JBCefBrowserBase
import com.intellij.ui.jcef.JBCefJSQuery
import com.intellij.util.concurrency.AppExecutorUtil
import com.sun.net.httpserver.HttpServer
import java.awt.BorderLayout
import java.awt.Window
import java.net.InetAddress
import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import javax.swing.JWindow

class WGPURepairService : Disposable {
  private val initialized = AtomicBoolean(false)
  private val ready = CompletableFuture<Unit>()
  private var pending: CompletableFuture<String>? = null
  private var browser: JBCefBrowserBase? = null
  private var query: JBCefJSQuery? = null
  private var server: HttpServer? = null
  private var window: JWindow? = null
  private var url: String = ""

  fun repairPythonLine(input: String): CompletableFuture<String> = runString(input)

  fun runString(input: String): CompletableFuture<String> {
    initRuntime()

    return ready.thenCompose {
      val result = CompletableFuture<String>()
      ApplicationManager.getApplication().invokeLater {
        if (pending != null) {
          result.completeExceptionally(IllegalStateException("TidyParse WebGPU is already running"))
          return@invokeLater
        }

        pending = result
        browser?.cefBrowser?.executeJavaScript("window.__tidyparseRunString(${jsString(input)});", url, 0)
      }
      result
    }
  }

  private fun initRuntime() {
    if (!initialized.compareAndSet(false, true)) return

    ApplicationManager.getApplication().invokeLater {
      if (!JBCefApp.isSupported()) {
        ready.completeExceptionally(IllegalStateException("JCEF not supported"))
        return@invokeLater
      }

      val cefBrowser: JBCefBrowserBase = JBCefBrowser()
      val cefQuery = JBCefJSQuery.create(cefBrowser)
      val httpServer = HttpServer.create(InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0)

      browser = cefBrowser
      query = cefQuery
      server = httpServer
      url = "http://127.0.0.1:${httpServer.address.port}/"

      cefQuery.addHandler { raw ->
        ApplicationManager.getApplication().invokeLater {
          if (raw == "READY" && pending == null) ready.complete(Unit)
          else if (raw.startsWith(TIDYPARSE_JCEF_ERROR)) {
            val error = IllegalStateException(raw.removePrefix(TIDYPARSE_JCEF_ERROR))
            pending?.completeExceptionally(error)?.also { pending = null } ?: ready.completeExceptionally(error)
          }
          else pending?.complete(raw).also { pending = null }
        }
        null
      }

      val template = javaClass
        .getResourceAsStream("/jcef/tidyparse-jcef.html")!!
        .reader(StandardCharsets.UTF_8)
        .readText()

      val html = template.replace("__JCEF_EVENT_CALLBACK__", cefQuery.inject("payload"))

      httpServer.createContext("/") { ex ->
        val bytes = html.toByteArray(StandardCharsets.UTF_8)
        ex.responseHeaders.add("Content-Type", "text/html; charset=utf-8")
        ex.responseHeaders.add("Cache-Control", "no-store")
        ex.sendResponseHeaders(200, bytes.size.toLong())
        ex.responseBody.use { it.write(bytes) }
      }

      httpServer.executor = AppExecutorUtil.getAppExecutorService()
      httpServer.start()
      window = JWindow().apply {
        setType(Window.Type.UTILITY)
        setFocusableWindowState(false)
        layout = BorderLayout()
        add(cefBrowser.component, BorderLayout.CENTER)
        setSize(1, 1)
        setLocation(-10_000, -10_000)
        isVisible = true
      }
      cefBrowser.loadURL(url)
      AppExecutorUtil.getAppScheduledExecutorService().schedule({
        if (!ready.isDone) ready.completeExceptionally(IllegalStateException("Timed out waiting for TidyParse JCEF runtime"))
      }, 30, TimeUnit.SECONDS)
    }
  }

  override fun dispose() {
    pending?.cancel(true)
    pending = null
    server?.stop(0)
    query?.dispose()
    window?.dispose()
    browser?.let { Disposer.dispose(it) }
  }
}

fun Project.webGpuStringService(): WGPURepairService = getService(WGPURepairService::class.java)

fun Throwable.showTidyParseError(project: Project) =
  ApplicationManager.getApplication().invokeLater {
    val root = cause ?: this
    Messages.showErrorDialog(project, root.message ?: root.toString(), "TidyParse WebGPU")
  }

private fun jsString(s: String): String =
  buildString {
    append('"')
    for (c in s) when (c) {
      '\\' -> append("\\\\")
      '"' -> append("\\\"")
      '\n' -> append("\\n")
      '\r' -> append("\\r")
      '\t' -> append("\\t")
      '\b' -> append("\\b")
      '\u000C' -> append("\\f")
      else -> if (c.code < 0x20) append("\\u%04x".format(c.code)) else append(c)
    }
    append('"')
  }

private const val TIDYPARSE_JCEF_ERROR = "__TIDYPARSE_ERROR__"
