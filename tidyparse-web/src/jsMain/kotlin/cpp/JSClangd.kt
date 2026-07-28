import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.await
import kotlin.js.Promise

private const val CPP_COI_SERVICE_WORKER_VERSION = "3-" + CPP_CLANGD_ARTIFACT_VERSION
private const val CPP_COI_RELOAD_KEY = "tidyparse-cpp-coi-reload"

fun isCppCoiServiceWorkerRuntime(): Boolean =
  js("typeof ServiceWorkerGlobalScope !== 'undefined' && globalThis instanceof ServiceWorkerGlobalScope") as Boolean

fun setupCppCoiServiceWorker() {
  val worker = js("globalThis")

  worker.addEventListener("install", { event: dynamic ->
    event.waitUntil(worker.skipWaiting())
  })

  worker.addEventListener("activate", { event: dynamic ->
    event.waitUntil(worker.clients.claim())
  })

  worker.addEventListener("fetch", { event: dynamic ->
    val request = event.request
    val invalidCacheMode =
      request.cache == "only-if-cached" && request.mode != "same-origin"

    if (!invalidCacheMode) {
      val response = js("""(request) => {
        const url = new URL(request.url);
        const clangdWorkerRequest =
          request.destination === "worker" &&
          url.searchParams.get("cpp-worker") === "clangd";
        const target = clangdWorkerRequest
          ? new URL("tidyparse-web.js?cpp-worker-bundle=clangd", url)
          : request;
        return fetch(target).then(response => {
        const sameOrigin = new URL(request.url).origin === globalThis.location.origin;
        if (!sameOrigin || response.type === "opaque" || response.status === 0) return response;
            const headers = new Headers(response.headers);
            headers.set("Cross-Origin-Opener-Policy", "same-origin");
            headers.set("Cross-Origin-Embedder-Policy", "require-corp");
            headers.set("Cross-Origin-Resource-Policy", "cross-origin");
            return new Response(response.body, {
              status: response.status,
              statusText: response.statusText,
              headers
            });
        });
      }""")(request)
      event.respondWith(response)
    }
  })
}

fun cppCrossOriginIsolated(): Boolean =
  window.asDynamic().crossOriginIsolated as? Boolean ?: false

/**
 * Returns an error message when isolation cannot be enabled. A null result means
 * the page is reloading under the newly installed exact-page service worker.
 */
suspend fun ensureCppCrossOriginIsolation(): String? {
  if (window.location.protocol !in setOf("https:", "http:")) {
    return "clangd requires HTTPS or localhost"
  }

  val serviceWorkers = window.navigator.asDynamic().serviceWorker
  if (serviceWorkers == null || serviceWorkers == js("undefined")) {
    return "Service workers are unavailable; clangd cannot enable shared memory"
  }

  return try {
    val scriptUrl = js("(base, version) => new URL('tidyparse-web.js?cpp-coi=' + version, base).href")(
      document.baseURI,
      CPP_COI_SERVICE_WORKER_VERSION
    ) as String
    val currentController = serviceWorkers.controller?.scriptURL as? String
    if (cppCrossOriginIsolated() && currentController == scriptUrl) {
      clearCppCoiReloadGuard()
      return null
    }

    val options = js("{}")
    options.scope = window.location.pathname
    options.updateViaCache = "none"

    val registration =
      (serviceWorkers.register(scriptUrl, options) as Promise<dynamic>).await()
    val activated = waitForCppCoiController(registration, serviceWorkers, scriptUrl)
    if (!activated) {
      return "Unable to activate the clangd isolation worker"
    }

    if (cppCrossOriginIsolated()) {
      clearCppCoiReloadGuard()
      return null
    }

    val alreadyReloaded = try {
      window.sessionStorage.getItem(CPP_COI_RELOAD_KEY) == CPP_COI_SERVICE_WORKER_VERSION
    } catch (_: Throwable) {
      false
    }
    if (alreadyReloaded) {
      return "Cross-origin isolation could not be enabled by this host"
    }

    try {
      window.sessionStorage.setItem(CPP_COI_RELOAD_KEY, CPP_COI_SERVICE_WORKER_VERSION)
    } catch (_: Throwable) {
    }

    window.location.reload()
    null
  } catch (failure: Throwable) {
    "Unable to enable clangd isolation: ${failure.message ?: failure}"
  }
}

private suspend fun waitForCppCoiController(
  registration: dynamic,
  serviceWorkers: dynamic,
  scriptUrl: String
): Boolean =
  (js("""(registration, container, expected) => new Promise(resolve => {
    let settled = false;
    let interval = 0;
    let timeout = 0;
    const matches = () =>
      registration.active && registration.active.scriptURL === expected &&
      container.controller && container.controller.scriptURL === expected;
    const finish = value => {
      if (settled) return;
      settled = true;
      clearInterval(interval);
      clearTimeout(timeout);
      container.removeEventListener("controllerchange", check);
      registration.removeEventListener("updatefound", watchInstalling);
      resolve(value);
    };
    const check = () => {
      if (matches()) finish(true);
    };
    const watchInstalling = () => {
      if (registration.installing) {
        registration.installing.addEventListener("statechange", check);
      }
      check();
    };
    container.addEventListener("controllerchange", check);
    registration.addEventListener("updatefound", watchInstalling);
    if (registration.installing) registration.installing.addEventListener("statechange", check);
    if (registration.waiting) registration.waiting.addEventListener("statechange", check);
    if (registration.active) registration.active.addEventListener("statechange", check);
    interval = setInterval(check, 50);
    timeout = setTimeout(() => finish(matches()), 20000);
    check();
  })""")(registration, serviceWorkers, scriptUrl) as Promise<Boolean>).await()

private fun clearCppCoiReloadGuard() {
  try {
    window.sessionStorage.removeItem(CPP_COI_RELOAD_KEY)
  } catch (_: Throwable) {
  }
}
