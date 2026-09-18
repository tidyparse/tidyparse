config.set({
    client: { mocha: { timeout: 60000 } },
    customLaunchers: {
        ChromeHeadlessWebGPU: {
            // Karma's ChromeHeadless base appends --disable-gpu.
            base: 'Chrome',
            flags: [
                '--headless=new',
                '--enable-gpu',
                '--enable-unsafe-webgpu',
                '--ignore-gpu-blocklist',
                ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])
            ]
        }
    },
    browsers: ['ChromeHeadlessWebGPU']
});
