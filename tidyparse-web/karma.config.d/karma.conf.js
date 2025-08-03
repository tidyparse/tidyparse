config.set({
    client: {
        mocha: { timeout: 60000 },
        captureConsole: true
    },
    // logLevel: config.LOG_DEBUG,  // Verbose Karma logs
    // reporters: ['karma-kotlin-reporter'],
    // browserConsoleLogOptions: {
    //     level: '',  // Include all levels (error, warn, info, log, debug)
    //     format: '%b %T: %m',  // Browser, time, message
    //     terminal: true  // Output to terminal
    // },
    customLaunchers: {
        ChromeSmall: {
            base: 'Chrome',
            flags: ['--window-size=1,1']
        }
    },
    browsers: ['ChromeSmall'],
    // customLaunchers: {
    //     ChromeHeadlessWebGPU: {
    //         base: 'ChromeHeadless',
    //         flags: [
    //             '--enable-unsafe-webgpu',
    //             // '--use-angle=swiftshader',  // CPU fallback
    //             '--disable-gpu-driver-bug-workarounds',
    //             '--enable-logging=stderr',  // Route Chrome logs to stderr (visible in Gradle)
    //             '--v=1',  // Verbose Chrome logging
    //             '--no-sandbox',  // Required for some CI/headless envs
    //             '--headless=new'  // New headless mode (better stability in Chrome 109+)
    //         ]
    //     }
    // },
    // browsers: ['ChromeHeadlessWebGPU']
});