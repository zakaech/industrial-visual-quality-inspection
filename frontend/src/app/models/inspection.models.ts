/**
 * TypeScript mirrors of the FastAPI response models (src/vqi/api/schemas.py).
 * Field names and types follow the API contract exactly; nothing is added or renamed.
 */

/** GET /health */
export interface HealthResponse {
  status: string;
}

/** Nested in ModelInfoResponse.threshold */
export interface ThresholdInfo {
  /** Frozen decision threshold on the raw model score (0..1). */
  value: number;
  policy: string;
  target_recall: number;
  margin: string;
  selected_on: string;
  /** 'raw' = no post-hoc calibration of the scores. */
  calibration: string;
  frozen_at: string;
}

/** GET /model-info */
export interface ModelInfoResponse {
  model_version: string;
  model: string;
  experiment_id: string;
  architecture: Record<string, unknown>;
  model_sha256: string;
  preprocessor: Record<string, unknown>;
  threshold: ThresholdInfo;
  /** Labels used by the API for the positive (defect) and negative classes. */
  decision_labels: Record<string, string>;
  project_version: string;
  bundle_frozen_at: string;
  validation_summary: Record<string, unknown>;
  final_test_summary: Record<string, unknown> | null;
}

/** POST /predict */
export interface PredictionResponse {
  image_id: string;
  /** Raw model score P(defective); not a calibrated probability. */
  probability: number;
  /** Decided by the API (REJECT iff probability >= threshold); never recomputed here. */
  decision: string;
  threshold: number;
  model_version: string;
  /** Server-side preprocessing + inference time. */
  latency_ms: number;
  request_id: string;
}

/** Error body of the API (4xx / 5xx). */
export interface ErrorResponse {
  detail: string;
  request_id?: string | null;
}

/** What the page shows when a request fails (never a traceback). */
export type InspectionErrorKind =
  | 'offline' // no HTTP response at all (service down, CORS, network)
  | 'rejected' // 400 / 413 / 415 / 422: the API refused the file
  | 'unavailable' // 503: model not loaded
  | 'server' // 500 or any other status
  | 'unknown';

export interface InspectionError {
  kind: InspectionErrorKind;
  message: string;
  status?: number;
  requestId?: string | null;
}
