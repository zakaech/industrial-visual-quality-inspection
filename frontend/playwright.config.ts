import { defineConfig } from '@playwright/test';

/**
 * Real end-to-end check: Chrome → Angular dev server → FastAPI (must already run on
 * http://127.0.0.1:8000, e.g. `python scripts/serve_api.py`). Uses the locally installed Chrome
 * (`channel: 'chrome'`), so no browser download is needed. Run with `npm run e2e`.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:4200',
    channel: 'chrome',
    headless: true,
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'npm start',
    url: 'http://localhost:4200',
    reuseExistingServer: true,
    timeout: 180_000,
  },
});
