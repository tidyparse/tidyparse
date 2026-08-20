import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch

// ./gradlew :tidyparse-cpp:jsBrowserDevelopmentRun --continuous
fun main() {
  if (isCppCoiServiceWorkerRuntime()) {
    setupCppCoiServiceWorker()
    return
  }
  if (isCppCompletionWorkerRuntime()) {
    setupCppCompletionWorker()
    return
  }
  if (isCppClangdWorkerRuntime()) {
    setupCppClangdWorker()
    return
  }
  if (isCppMonacoEditorWorkerRuntime()) {
    setupCppMonacoEditorWorker()
    return
  }
  if (isCppTextMateWorkerRuntime()) {
    setupCppTextMateWorker()
    return
  }

  MainScope().launch {
    try {
      cppSetup()
    } catch (t: Throwable) {
      throw t
    }
  }
}
