import { AppEnvironment } from '../app/models/app-environment';

/** Production build. Change `apiBaseUrl` to where the FastAPI service is reachable. */
export const environment: AppEnvironment = {
  production: true,
  apiBaseUrl: 'http://127.0.0.1:8000',
  maxUploadMb: 10,
  allowedTypes: ['image/png', 'image/jpeg', 'image/bmp'],
};
