const fs = require('fs');
const path = require('path');

const isCi = process.env.GITHUB_ACTIONS === 'true';
const chromeFlags = ['--window-size=1,1'];
const chromeHeapMb = process.env.CHROME_V8_HEAP_MB || '3072';
const localKarmaTimeoutMs = 540000;
const ciKarmaTimeoutMs = 30 * 60 * 1000;
const karmaTimeoutMs = isCi ? ciKarmaTimeoutMs : localKarmaTimeoutMs;
const pingTimeoutMs = isCi ? 5 * 60 * 1000 : 5000;
const browserDisconnectTimeoutMs = isCi ? 5000 : localKarmaTimeoutMs;
const ciLogsDir = path.resolve(__dirname, '../../../ci-logs');
const browserConsoleLog = path.join(ciLogsDir, 'browser-console.log');

fs.mkdirSync(ciLogsDir, { recursive: true });

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
        terminal: true,
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
