import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { environment } from '../../../environments/environment';
import { OFFLINE_MESSAGE } from '../../core/api-error';
import { ModelInfoResponse, PredictionResponse } from '../../models/inspection.models';
import { InspectionPage } from './inspection-page';

const BASE = environment.apiBaseUrl;

const MODEL_INFO = {
  model_version: 'SYN-0123456789ab',
  model: 'synthetic',
  experiment_id: 'SYN',
  architecture: { architecture: 'resnet18', pooling: 'avg', n_params: 11177025 },
  model_sha256: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
  preprocessor: { input_size: [640, 224], variant: 'none' },
  threshold: {
    value: 0.25,
    policy: 'min_recall',
    target_recall: 0.95,
    margin: 'midpoint_logit',
    selected_on: 'synthetic',
    calibration: 'raw',
    frozen_at: '2026-01-01T00:00:00+00:00',
  },
  decision_labels: { positive: 'REJECT', negative: 'ACCEPT' },
  project_version: '0.1.0',
  bundle_frozen_at: '2026-01-01T00:00:00+00:00',
  validation_summary: {},
  final_test_summary: null,
} satisfies ModelInfoResponse;

/** Values deliberately different from the real frozen threshold: the page must display what it receives. */
const ACCEPTED: PredictionResponse = {
  image_id: '10004.png',
  probability: 0.016112767159938812,
  decision: 'ACCEPT',
  threshold: 0.018752005540198832,
  model_version: 'SYN-0123456789ab',
  latency_ms: 133.4,
  request_id: 'req-accept',
};

const REJECTED: PredictionResponse = {
  ...ACCEPTED,
  image_id: '10952.png',
  probability: 1.0,
  decision: 'REJECT',
  latency_ms: 118.6,
  request_id: 'req-reject',
};

function pngFile(name = 'part.png', size = 1234): File {
  const file = new File([new Uint8Array(16)], name, { type: 'image/png' });
  Object.defineProperty(file, 'size', { value: size });
  return file;
}

describe('InspectionPage', () => {
  let fixture: ComponentFixture<InspectionPage>;
  let page: InspectionPage;
  let http: HttpTestingController;
  let el: HTMLElement;

  const text = (selector: string) =>
    el.querySelector(selector)?.textContent?.replace(/\s+/g, ' ').trim();
  const inspectButton = () => el.querySelector<HTMLButtonElement>('button.inspect')!;

  const flushModelInfo = async (body: ModelInfoResponse = MODEL_INFO) => {
    http.expectOne(`${BASE}/model-info`).flush(body);
    await fixture.whenStable();
  };

  const choose = async (file: File) => {
    page.selectFile(file);
    await fixture.whenStable();
  };

  beforeEach(async () => {
    Object.defineProperty(URL, 'createObjectURL', {
      value: vi.fn(() => 'blob:preview'),
      configurable: true,
    });
    Object.defineProperty(URL, 'revokeObjectURL', { value: vi.fn(), configurable: true });
    await TestBed.configureTestingModule({
      imports: [InspectionPage],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(InspectionPage);
    page = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    await fixture.whenStable();
  });

  afterEach(() => http.verify());

  // ------------------------------------------------------------------ creation / initial state
  it('creates with the idle state and a disabled Inspect button', async () => {
    await flushModelInfo();
    expect(page).toBeTruthy();
    expect(text('h1')).toBe('Industrial Visual Quality Inspection');
    expect(text('.result-body')).toBe('No image selected.');
    expect(inspectButton().disabled).toBe(true);
    expect(el.querySelector('input[type=file]')).not.toBeNull();
  });

  it('shows the model information received from /model-info (nothing hardcoded)', async () => {
    await flushModelInfo();
    expect(text('[data-testid=model-version]')).toBe('SYN-0123456789ab');
    expect(text('[data-testid=model-threshold]')).toContain('0.250000');
    expect(text('.api-status')).toContain('Service online');
    expect(text('.model-strip')).toContain('resnet18 · 11.18 M params');
    expect(text('.model-strip')).toContain('640 × 224 px');
  });

  // ------------------------------------------------------------------ file selection
  it('selecting a valid PNG shows the preview, name and size and enables Inspect', async () => {
    await flushModelInfo();
    await choose(pngFile('10004.png', 252277));
    const img = el.querySelector<HTMLImageElement>('.preview img')!;
    expect(img.getAttribute('src')).toBe('blob:preview');
    expect(img.alt).toBe('Preview of 10004.png');
    expect(text('.file-name')).toBe('10004.png');
    expect(text('.file-meta')).toContain('246.4 KB');
    expect(text('.result-body')).toContain('Ready to inspect 10004.png');
    expect(inspectButton().disabled).toBe(false);
  });

  it('rejects an unsupported type client-side without calling the API', async () => {
    await flushModelInfo();
    await choose(new File(['hello'], 'notes.txt', { type: 'text/plain' }));
    expect(page.selectedFile()).toBeNull();
    expect(text('[role=alert]')).toContain('Unsupported file type (text/plain)');
    expect(inspectButton().disabled).toBe(true);
    http.expectNone(`${BASE}/predict`);
  });

  it('rejects a file above the configured size limit', async () => {
    await flushModelInfo();
    await choose(pngFile('huge.png', (environment.maxUploadMb + 1) * 1024 * 1024));
    expect(page.selectedFile()).toBeNull();
    expect(text('[role=alert]')).toContain(`Maximum size: ${environment.maxUploadMb} MB`);
  });

  it('rejects an empty file', async () => {
    await flushModelInfo();
    await choose(pngFile('empty.png', 0));
    expect(text('[role=alert]')).toContain('empty');
  });

  it('accepts a dropped file', async () => {
    await flushModelInfo();
    const file = pngFile('dropped.png');
    const event = {
      preventDefault: vi.fn(),
      dataTransfer: { files: [file] },
    } as unknown as DragEvent;
    page.onDrop(event);
    await fixture.whenStable();
    expect(event.preventDefault).toHaveBeenCalled();
    expect(page.selectedFile()?.name).toBe('dropped.png');
    expect(text('.file-name')).toBe('dropped.png');
  });

  it('Remove clears the selection and returns to the idle state', async () => {
    await flushModelInfo();
    await choose(pngFile());
    page.clearFile();
    await fixture.whenStable();
    expect(page.selectedFile()).toBeNull();
    expect(text('.result-body')).toBe('No image selected.');
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:preview');
  });

  // ------------------------------------------------------------------ inspection
  it('shows the loading state, posts the file once, and ignores concurrent clicks', async () => {
    await flushModelInfo();
    const file = pngFile('10004.png');
    await choose(file);

    inspectButton().click();
    await fixture.whenStable();
    expect(text('button.inspect')).toBe('Inspecting…');
    expect(inspectButton().disabled).toBe(true);
    expect(inspectButton().getAttribute('aria-busy')).toBe('true');
    expect(text('.result-body')).toBe('Inspecting image…');

    page.inspect(); // second call while loading: no second request
    const req = http.expectOne(`${BASE}/predict`);
    expect(req.request.method).toBe('POST');
    const sent = (req.request.body as FormData).get('file') as File; // jsdom returns a copy
    expect(sent.name).toBe(file.name);
    expect(sent.type).toBe(file.type);
    req.flush(ACCEPTED);
    await fixture.whenStable();
    expect(text('button.inspect')).toBe('Inspect image');
  });

  it('displays an ACCEPT decision with score, threshold and latency from the response', async () => {
    await flushModelInfo();
    await choose(pngFile('10004.png'));
    inspectButton().click();
    http.expectOne(`${BASE}/predict`).flush(ACCEPTED);
    await fixture.whenStable();

    expect(text('[data-testid=decision]')).toBe('ACCEPT');
    expect(el.querySelector('.decision')?.classList.contains('accept')).toBe(true);
    expect(text('.decision-hint')).toBe('No defect detected — part accepted');
    expect(text('[data-testid=probability]')).toBe('1.61 %');
    expect(el.querySelector('[data-testid=probability]')?.getAttribute('data-raw')).toBe(
      '0.016112767159938812',
    );
    expect(text('[data-testid=threshold]')).toBe('1.8752 %');
    expect(el.querySelector('[data-testid=threshold]')?.getAttribute('data-raw')).toBe(
      '0.018752005540198832',
    );
    expect(text('[data-testid=latency]')).toBe('133 ms');
    expect(text('.meta')).toContain('req-accept');
  });

  it('displays a REJECT decision (not only by colour: text + icon)', async () => {
    await flushModelInfo();
    await choose(pngFile('10952.png'));
    inspectButton().click();
    http.expectOne(`${BASE}/predict`).flush(REJECTED);
    await fixture.whenStable();

    expect(text('[data-testid=decision]')).toBe('REJECT');
    expect(el.querySelector('.decision')?.classList.contains('reject')).toBe(true);
    expect(el.querySelector('.decision-icon svg')).not.toBeNull();
    expect(text('.decision-hint')).toBe('Defect detected — part rejected');
    expect(text('[data-testid=probability]')).toBe('100.00 %');
    expect(text('[data-testid=latency]')).toBe('119 ms');
  });

  it('never decides itself: the decision and threshold shown are exactly those of the response', async () => {
    await flushModelInfo();
    await choose(pngFile());
    inspectButton().click();
    // inconsistent on purpose: probability above threshold but decision ACCEPT → shown as received
    http
      .expectOne(`${BASE}/predict`)
      .flush({ ...ACCEPTED, probability: 0.9, threshold: 0.25, decision: 'ACCEPT' });
    await fixture.whenStable();
    expect(text('[data-testid=decision]')).toBe('ACCEPT');
    expect(text('[data-testid=threshold]')).toBe('25.0000 %');
    expect(text('[data-testid=probability]')).toBe('90.00 %');
  });

  // ------------------------------------------------------------------ errors
  it('shows the API detail when the service rejects the image (HTTP 422)', async () => {
    await flushModelInfo();
    await choose(pngFile('bad.png'));
    inspectButton().click();
    http
      .expectOne(`${BASE}/predict`)
      .flush(
        { detail: 'the file could not be decoded as an image', request_id: 'r-422' },
        { status: 422, statusText: 'Unprocessable' },
      );
    await fixture.whenStable();
    expect(page.status()).toBe('error');
    expect(text('[role=alert]')).toContain('the file could not be decoded as an image');
    expect(text('[role=alert]')).toContain('r-422');
    expect(el.querySelector('[data-testid=decision]')).toBeNull();
    expect(inspectButton().disabled).toBe(false); // the user can retry
  });

  it('shows a generic message on HTTP 500 without any traceback', async () => {
    await flushModelInfo();
    await choose(pngFile());
    inspectButton().click();
    http
      .expectOne(`${BASE}/predict`)
      .flush({ detail: 'internal error' }, { status: 500, statusText: 'Internal' });
    await fixture.whenStable();
    expect(text('[role=alert]')).toContain('HTTP 500');
    expect(el.innerHTML).not.toContain('Traceback');
  });

  it('reports the service as unavailable when the request never reaches it', async () => {
    await flushModelInfo();
    await choose(pngFile());
    inspectButton().click();
    http.expectOne(`${BASE}/predict`).error(new ProgressEvent('error'), { status: 0 });
    await fixture.whenStable();
    expect(text('[role=alert]')).toContain(OFFLINE_MESSAGE);
    expect(text('[role=alert]')).toContain(BASE);
    expect(text('.api-status')).toContain('Service unavailable');
  });

  it('marks the service offline when /model-info fails, and Retry asks again', async () => {
    http.expectOne(`${BASE}/model-info`).error(new ProgressEvent('error'), { status: 0 });
    await fixture.whenStable();
    expect(text('.api-status')).toContain('Service unavailable');
    expect(text('.model-strip')).toContain('Model information unavailable');

    el.querySelector<HTMLButtonElement>('.api-status button')!.click();
    await fixture.whenStable();
    http.expectOne(`${BASE}/model-info`).flush(MODEL_INFO);
    await fixture.whenStable();
    expect(text('.api-status')).toContain('Service online');
  });
});
