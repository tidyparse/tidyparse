fun main() {
  when {
    isRustGlancerWorkerRuntime() -> setupRustGlancerWorker()
    isRustMonacoWorkerRuntime() -> setupRustMonacoWorker()
    else -> rustSetup()
  }
}
