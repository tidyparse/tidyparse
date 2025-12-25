module.exports = function(config) {
    config.set({
        browserDisconnectTimeout: 540000,  // 9min for reconnect attempts
        browserNoActivityTimeout: 900000,  // 15min for inactivity
        captureTimeout: 180000,            // 3min to capture browser
        browserDisconnectTolerance: 5,     // Allow up to 5 disconnects
        client: {
            mocha: { timeout: 540000 }       // Per-test timeout
        },
        customLaunchers: {
            ChromeHeadlessWebGPU: {
                base: 'ChromeHeadless',
                flags: [
                    '--headless=new',                   // New headless mode for better GPU/WebGL support (Chrome 109+)
                    '--enable-unsafe-webgpu',           // Explicitly enable WebGPU (required as it's blocklisted)
                    '--use-angle=swiftshader',          // Software GPU fallback to prevent init failures; try '--use-gl=swiftshader' as alt
                    '--disable-gpu-driver-bug-workarounds',  // Avoid unnecessary GPU blacklisting
                    '--enable-logging=stderr',          // Route Chrome logs to stderr (visible in Gradle/CI output)
                    '--v=1',                            // Verbose logging level 1 (adjust to --v=2 for more)
                    '--no-sandbox',                     // Required for CI security restrictions
                    '--disable-software-rasterizer',    // Prefer hardware if available (fallback to software otherwise)
                    '--disable-dev-shm-usage',          // Avoid shared memory issues in VMs
                    '--remote-debugging-port=9222'      // Optional: For remote inspection if needed
                ]
            }
            // ChromeSmall: { ... }  // Commented: Non-headless won't launch in headless-only CI env
        },
        browsers: ['ChromeHeadlessWebGPU']
    });
};