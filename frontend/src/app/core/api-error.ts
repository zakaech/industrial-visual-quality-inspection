import { HttpErrorResponse } from '@angular/common/http';

import { ErrorResponse, InspectionError } from '../models/inspection.models';

export const OFFLINE_MESSAGE =
  'Unable to connect to the inspection service. Please verify that the API is running.';

function detailOf(err: HttpErrorResponse): string | null {
  const body: unknown = err.error;
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as ErrorResponse).detail;
    return typeof detail === 'string' && detail.trim() ? detail : null;
  }
  return null;
}

function requestIdOf(err: HttpErrorResponse): string | null {
  const body: unknown = err.error;
  if (body && typeof body === 'object' && 'request_id' in body) {
    const id = (body as ErrorResponse).request_id;
    return typeof id === 'string' ? id : null;
  }
  return err.headers?.get('X-Request-ID') ?? null;
}

/** Map any HTTP failure to a user-facing message; the API's `detail` is short and safe to show. */
export function describeHttpError(err: unknown): InspectionError {
  if (!(err instanceof HttpErrorResponse)) {
    return {
      kind: 'unknown',
      message: 'Unexpected error while contacting the inspection service.',
    };
  }
  const requestId = requestIdOf(err);
  if (err.status === 0) {
    return { kind: 'offline', message: OFFLINE_MESSAGE, status: 0, requestId };
  }
  const detail = detailOf(err);
  switch (err.status) {
    case 400:
    case 413:
    case 415:
    case 422:
      return {
        kind: 'rejected',
        message: detail
          ? `The image was rejected by the service: ${detail}.`
          : 'The image was rejected by the service.',
        status: err.status,
        requestId,
      };
    case 503:
      return {
        kind: 'unavailable',
        message:
          'The inspection service is running but its model is not loaded. Please retry later.',
        status: 503,
        requestId,
      };
    default:
      return {
        kind: 'server',
        message: `The inspection service reported an internal error (HTTP ${err.status}). Please retry.`,
        status: err.status,
        requestId,
      };
  }
}
