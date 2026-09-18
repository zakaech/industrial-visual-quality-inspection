import { AppEnvironment } from '../app/models/app-environment';

/** `ng serve` / `ng test`: the API started with `python scripts/serve_api.py` or Docker. */
export const environment: AppEnvironment = {
  production: false,
  apiBaseUrl: 'http://127.0.0.1:8000',
  maxUploadMb: 10,
  allowedTypes: ['image/png', 'image/jpeg', 'image/bmp'],
};
