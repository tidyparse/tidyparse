config.set({
    browserDisconnectTimeout: 540000,
    client: { mocha: { timeout: 540000 } },
    customLaunchers: {
        ChromeSmall: {
            base: 'Chrome',
            // flags: ['--window-size=1,1', '--enable-profiling', '--profiling-at-start=gpu-process', '--no-sandbox', '--profiling-flush']
            flags: [
                '--window-size=1,1',
                // '--enable-profiling',
                // '--profiling-at-start=renderer',  // Or gpu-process if that's your focus
                // '--no-sandbox',
                // `--profiling-file=/Users/breandan/IdeaProjects/tidyparse/profile.log',
                // '--profiling-flush=5'  // Adjust interval as needed
                // '--trace-startup',  // Auto-starts tracing
                // '--trace-startup-file=/Users/breandan/IdeaProjects/tidyparse/trace.json',  // e.g., '/Users/yourusername/project/trace.json'
                // '--trace-startup-duration=60',  // Adjust to cover test runtime
                // '--trace-startup-categories=*,disabled-by-default-v8.cpu_profiler,disabled-by-default-v8.runtime_stats,devtools.timeline,blink.user_timing'  // Captures JS CPU data
            ]
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