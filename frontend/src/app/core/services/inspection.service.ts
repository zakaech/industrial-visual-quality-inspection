import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import {
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
} from '../../models/inspection.models';

/**
 * The only place that talks to the FastAPI service.
 * GET /health, GET /model-info, POST /predict (multipart/form-data, field `file`).
 */
@Injectable({ providedIn: 'root' })
export class InspectionService {
  private readonly http = inject(HttpClient);
  readonly baseUrl = environment.apiBaseUrl.replace(/\/+$/, '');

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  modelInfo(): Observable<ModelInfoResponse> {
    return this.http.get<ModelInfoResponse>(`${this.baseUrl}/model-info`);
  }

  /** Sends the file untouched. No Content-Type header is set: the browser adds the multipart boundary. */
  predict(file: File): Observable<PredictionResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<PredictionResponse>(`${this.baseUrl}/predict`, form);
  }
}
