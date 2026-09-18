/**
 * Real browser → Angular → FastAPI → Predictor → frozen model.
 * Only VALIDATION images of the dataset are used (never the official test split); the three
 * ids are the ones already used in Phases 5 and 6. For each one the page's displayed raw values
 * (data-raw attributes) must equal the API's own JSON answer for the same bytes.
 */
import fs from 'node:fs';
import path from 'node:path';

import { expect, test, type Page } from '@playwright/test';

const API = process.env['VQI_API_URL'] ?? 'http://127.0.0.1:8000';
const IMAGES = path.resolve(__dirname, '../../data/raw/kolektor_sdd2/train');

const CASES = [
  { id: '10004', expected: 'ACCEPT', note: 'healthy part' },
  {
    id: '10045',
    expected: 'ACCEPT',
    note: 'known false negative (23 px defect), reproduced as-is',
  },
  { id: '10952', expected: 'REJECT', note: 'large defect' },
] as const;

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  return errors;
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('shows the model information of the running service (no CORS error)', async ({
  page,
  request,
}) => {
  const errors = collectConsoleErrors(page);
  const info = await (await request.get(`${API}/model-info`)).json();
  await expect(page.locator('.api-status')).toContainText('Service online', { timeout: 15_000 });
  await expect(page.locator('[data-testid=model-version]')).toHaveText(info.model_version);
  await expect(page.locator('[data-testid=model-threshold]')).toContainText(
    Number(info.threshold.value).toFixed(6),
  );
  expect(errors, `browser console errors: ${errors.join(' | ')}`).toEqual([]);
});

for (const c of CASES) {
  test(`${c.id}.png → ${c.expected} (${c.note})`, async ({ page, request }) => {
    const file = path.join(IMAGES, `${c.id}.png`);
    test.skip(!fs.existsSync(file), `validation image ${file} not available`);

    await page.setInputFiles('#file-input', file);
    await expect(page.locator('.file-name')).toHaveText(`${c.id}.png`);
    await expect(page.locator('.preview img')).toBeVisible();
    await expect(page.locator('.result-body')).toContainText('Ready to inspect');

    await page.getByRole('button', { name: 'Inspect image' }).click();
    await expect(page.locator('[data-testid=decision]')).toBeVisible({ timeout: 30_000 });

    const shown = {
      decision: (await page.locator('[data-testid=decision]').textContent())?.trim(),
      probability: await page.locator('[data-testid=probability]').getAttribute('data-raw'),
      threshold: await page.locator('[data-testid=threshold]').getAttribute('data-raw'),
      probabilityText: (await page.locator('[data-testid=probability]').textContent())?.trim(),
      thresholdText: (await page.locator('[data-testid=threshold]').textContent())?.trim(),
      latencyText: (await page.locator('[data-testid=latency]').textContent())?.trim(),
      model: (await page.locator('.meta').textContent())?.replace(/\s+/g, ' ').trim(),
    };

    // the same bytes, sent straight to the API
    const direct = await (
      await request.post(`${API}/predict`, {
        multipart: {
          file: { name: `${c.id}.png`, mimeType: 'image/png', buffer: fs.readFileSync(file) },
        },
      })
    ).json();

    expect(shown.decision).toBe(c.expected);
    expect(direct.decision).toBe(c.expected);
    expect(shown.probability).toBe(String(direct.probability));
    expect(shown.threshold).toBe(String(direct.threshold));
    expect(shown.latencyText).toMatch(/^\d+ ms$/);
    expect(shown.model).toContain(direct.model_version);

    console.log(
      `[e2e] ${c.id}.png: UI ${shown.decision} ${shown.probabilityText} (raw ${shown.probability}) ` +
        `thr ${shown.thresholdText} latency ${shown.latencyText} | API ${direct.decision} ${direct.probability} ` +
        `thr ${direct.threshold} latency ${direct.latency_ms} ms`,
    );
  });
}

test('client-side rejection of a text file; API rejection (422) of a fake PNG is displayed', async ({
  page,
}) => {
  await page.setInputFiles('#file-input', {
    name: 'notes.txt',
    mimeType: 'text/plain',
    buffer: Buffer.from('hello'),
  });
  await expect(page.locator('[role=alert]')).toContainText('Unsupported file type');
  await expect(page.getByRole('button', { name: 'Inspect image' })).toBeDisabled();

  await page.setInputFiles('#file-input', {
    name: 'fake.png',
    mimeType: 'image/png',
    buffer: Buffer.from('definitely not a png'),
  });
  await page.getByRole('button', { name: 'Inspect image' }).click();
  await expect(page.locator('[role=alert]')).toContainText('could not be decoded', {
    timeout: 15_000,
  });
  await expect(page.locator('[data-testid=decision]')).toHaveCount(0);
});

test('no horizontal scroll at 1440 / 1024 / 768 / 390 px, with a result displayed', async ({
  page,
}) => {
  const file = path.join(IMAGES, '10952.png');
  test.skip(!fs.existsSync(file), 'validation image not available');
  await page.setInputFiles('#file-input', file);
  await page.getByRole('button', { name: 'Inspect image' }).click();
  await expect(page.locator('[data-testid=decision]')).toBeVisible({ timeout: 30_000 });

  fs.mkdirSync(path.join(__dirname, 'screenshots'), { recursive: true });
  for (const width of [1440, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, `horizontal overflow at ${width}px`).toBeLessThanOrEqual(0);
    await page.screenshot({
      path: path.join(__dirname, 'screenshots', `inspection-${width}.png`),
      fullPage: true,
    });
  }
});
