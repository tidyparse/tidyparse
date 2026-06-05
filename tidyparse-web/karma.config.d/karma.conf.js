const isCi = process.env.GITHUB_ACTIONS === 'true';
const chromeFlags = ['--window-size=1,1'];
const localKarmaTimeoutMs = 540000;
const ciKarmaTimeoutMs = 30 * 60 * 1000;
const karmaTimeoutMs = isCi ? ciKarmaTimeoutMs : localKarmaTimeoutMs;

if (isCi) {
    chromeFlags.push(
        '--enable-logging=stderr'
    );
}

config.set({
    logLevel: config.LOG_INFO,
    browserDisconnectTimeout: karmaTimeoutMs,
    browserDisconnectTolerance: 0,
    browserNoActivityTimeout: karmaTimeoutMs,
    captureTimeout: karmaTimeoutMs,
    processKillTimeout: isCi ? 30000 : 2000,
    client: { captureConsole: true, mocha: { timeout: karmaTimeoutMs } },
    browserConsoleLogOptions: { level: 'debug', terminal: true },
    customLaunchers: {
        ChromeSmall: {
            base: 'Chrome',
            // flags: ['--window-size=1,1', '--enable-profiling', '--profiling-at-start=gpu-process', '--no-sandbox', '--profiling-flush']
            flags: [
                ...chromeFlags,
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
    // browsers: ['ChromeSmall'],
    // customLaunchers: {
    //     ChromeHeadlessWebGPU: {
    //         base: 'ChromeHeadless',
    //         flags: [
    //             '--headless=new',
    //             '--enable-unsafe-webgpu',
    //             '--enable-gpu',
    //
    //             '--use-angle=metal',
    //
    //             '--ignore-gpu-blocklist',
    //             '--disable-gpu-driver-bug-workarounds',
    //             '--no-sandbox',
    //             '--disable-software-rasterizer',
    //             '--enable-logging=stderr',  // Route Chrome logs to stderr (visible in Gradle)
    //             // '--v=1',  // Verbose Chrome logging
    //         ]
    //     }
    // },
    // browsers: ['ChromeHeadlessWebGPU']
});
