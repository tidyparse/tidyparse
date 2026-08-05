const fs = require('fs');
const path = require('path');

const isCi = process.env.GITHUB_ACTIONS === 'true';
const isCppCompletionBenchmark = process.env.CPP_COMPLETION_BENCHMARK === '1';
const chromeFlags = ['--window-size=1,1'];
const chromeHeapMb = process.env.CHROME_V8_HEAP_MB || '8192';
const benchmarkHarnessMs = Number(process.env.CPP_COMPLETION_TIME_LIMIT_MS || 60 * 1000);
const benchmarkTimeoutMs = benchmarkHarnessMs + 60 * 1000;
const localKarmaTimeoutMs = isCppCompletionBenchmark ? benchmarkTimeoutMs : 540000;
const ciKarmaTimeoutMs = isCppCompletionBenchmark ? benchmarkTimeoutMs : 30 * 60 * 1000;
const karmaTimeoutMs = isCi ? ciKarmaTimeoutMs : localKarmaTimeoutMs;
const pingTimeoutMs = karmaTimeoutMs;
const browserDisconnectTimeoutMs = karmaTimeoutMs;
const ciLogsDir = path.resolve(__dirname, '../../../ci-logs');
const browserConsoleLog = path.join(ciLogsDir, 'browser-console.log');
const blackArchiveName = 'pyodide-black-25.1.0-site.zip';

fs.mkdirSync(ciLogsDir, { recursive: true });

// Kotlin/JS copies resources into kotlin/, but Karma only serves files declared
// in its config. Keep the production root URL working in browser tests.
config.files.push({
    pattern: path.resolve(__dirname, 'kotlin', blackArchiveName),
    included: false,
    served: true,
    watched: false
});
config.proxies[`/${blackArchiveName}`] = `/base/kotlin/${blackArchiveName}`;

if (isCi) {
    chromeFlags.push(
        '--enable-logging=stderr',
        '--v=1',
        `--user-data-dir=${path.join(ciLogsDir, 'chrome-user-data')}`,
        `--crash-dumps-dir=${path.join(ciLogsDir, 'chrome-crash-dumps')}`
    );
}

config.set({
    logLevel: config.LOG_INFO,
    browserDisconnectTimeout: browserDisconnectTimeoutMs,
    browserDisconnectTolerance: 0,
    browserNoActivityTimeout: karmaTimeoutMs,
    captureTimeout: karmaTimeoutMs,
    pingTimeout: pingTimeoutMs,
    retryLimit: 0,
    processKillTimeout: isCi ? 30000 : 2000,
    client: { captureConsole: true, mocha: { timeout: karmaTimeoutMs } },
    browserConsoleLogOptions: {
        level: 'debug',
        // Kotlin's test adapter consumes the browser protocol internally. Printing the raw
        // console in benchmark mode duplicates every result as noisy --END_KOTLIN_TEST-- JSON.
        terminal: !isCppCompletionBenchmark,
        path: browserConsoleLog
    },
    customLaunchers: {
        ChromeHeadlessWebGPU: {
            // Use Chrome directly instead of Karma's ChromeHeadless base; that base appends --disable-gpu.
            base: 'Chrome',
            flags: [
                ...chromeFlags,
                '--headless=new',
                '--enable-gpu',
                '--enable-unsafe-webgpu',
                '--ignore-gpu-blocklist',
                '--use-angle=metal',
                `--js-flags=--max-old-space-size=${chromeHeapMb}`
            ]
        }
    },
    browsers: ['ChromeHeadlessWebGPU']
});

// The implementation stays with the experiment under src/jsTest; this small
// hook only makes its local clangd/compiler service visible to Karma.
let benchmarkSearchDir = __dirname;
let benchmarkService = null;
for (let depth = 0; depth < 10 && benchmarkService == null; depth++) {
    for (const relative of [
        'src/jsTest/cppCompletion/karma/benchmark-service.js',
        'tidyparse-web/src/jsTest/cppCompletion/karma/benchmark-service.js'
    ]) {
        const candidate = path.join(benchmarkSearchDir, relative);
        if (fs.existsSync(candidate)) {
            benchmarkService = candidate;
            break;
        }
    }
    benchmarkSearchDir = path.dirname(benchmarkSearchDir);
}
if (benchmarkService == null) {
    throw new Error('Unable to locate the C++ completion benchmark Karma service');
}
require(benchmarkService)(config);
