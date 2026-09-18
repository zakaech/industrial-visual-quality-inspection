import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { environment } from '../../../environments/environment';
import { ModelInfoResponse, PredictionResponse } from '../../models/inspection.models';
import { OFFLINE_MESSAGE, describeHttpError } from '../api-error';
import { InspectionService } from './inspection.service';

const BASE = environment.apiBaseUrl;

describe('InspectionService', () => {
  let service: InspectionService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(InspectionService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('GET /health', () => {
    let body: unknown;
    service.health().subscribe((r) => (body = r));
    const req = http.expectOne(`${BASE}/health`);
    expect(req.request.method).toBe('GET');
    req.flush({ status: 'ok' });
    expect(body).toEqual({ status: 'ok' });
  });

  it('GET /model-info returns the typed payload untouched', () => {
    const payload = {
      model_version: 'SYN-abc',
      threshold: { value: 0.25, policy: 'min_recall', calibration: 'raw' },
      decision_labels: { positive: 'REJECT', negative: 'ACCEPT' },
    } as unknown as ModelInfoResponse;
    let body: ModelInfoResponse | undefined;
    service.modelInfo().subscribe((r) => (body = r));
    http.expectOne(`${BASE}/model-info`).flush(payload);
    expect(body).toEqual(payload);
  });

  it('POST /predict sends multipart/form-data with the field `file` and no manual Content-Type', () => {
    const file = new File([new Uint8Array([1, 2, 3])], 'part.png', { type: 'image/png' });
    let body: PredictionResponse | undefined;
    service.predict(file).subscribe((r) => (body = r));

    const req = http.expectOne(`${BASE}/predict`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toBeInstanceOf(FormData);
    const sent = (req.request.body as FormData).get('file');
    expect(sent).toBeInstanceOf(File);
    expect((sent as File).name).toBe('part.png');
    // the browser must generate the multipart boundary itself
    expect(req.request.headers.has('Content-Type')).toBe(false);

    const response: PredictionResponse = {
      image_id: 'part.png',
      probability: 0.7,
      decision: 'REJECT',
      threshold: 0.25,
      model_version: 'SYN-abc',
      latency_ms: 12.3,
      request_id: 'rid',
    };
    req.flush(response);
    expect(body).toEqual(response);
  });
});

describe('describeHttpError', () => {
  const err = (status: number, body: unknown = null) =>
    new HttpErrorResponse({ status, error: body, url: `${BASE}/predict` });

  it('status 0 → service unreachable', () => {
    const e = describeHttpError(err(0));
    expect(e.kind).toBe('offline');
    expect(e.message).toBe(OFFLINE_MESSAGE);
  });

  it('4xx with a detail → rejected, showing the API detail', () => {
    const e = describeHttpError(err(415, { detail: "unsupported content type 'text/plain'" }));
    expect(e.kind).toBe('rejected');
    expect(e.status).toBe(415);
    expect(e.message).toContain("unsupported content type 'text/plain'");
  });

  it('422 without a usable body still gives a readable message', () => {
    const e = describeHttpError(err(422, 'garbage'));
    expect(e.kind).toBe('rejected');
    expect(e.message).toBe('The image was rejected by the service.');
  });

  it('503 → model not loaded', () => {
    expect(describeHttpError(err(503, { detail: 'model not loaded' })).kind).toBe('unavailable');
  });

  it('500 → generic message, never the body, with the request id', () => {
    const e = describeHttpError(err(500, { detail: 'internal error', request_id: 'r-1' }));
    expect(e.kind).toBe('server');
    expect(e.message).toContain('HTTP 500');
    expect(e.message).not.toContain('Traceback');
    expect(e.requestId).toBe('r-1');
  });

  it('non-HTTP failures → unknown', () => {
    expect(describeHttpError(new Error('boom')).kind).toBe('unknown');
  });
});
