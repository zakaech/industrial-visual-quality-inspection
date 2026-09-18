/** Shape of `src/environments/environment*.ts`. */
export interface AppEnvironment {
  readonly production: boolean;
  /** Base URL of the FastAPI inference service, without trailing slash. */
  readonly apiBaseUrl: string;
  /**
   * Client-side pre-checks that mirror `api.max_upload_mb` and `api.allowed_content_types` of
   * `configs/default.yaml`. They only give early feedback: the API remains the authority
   * (it answers 413 / 415 / 422 on its own).
   */
  readonly maxUploadMb: number;
  readonly allowedTypes: readonly string[];
}
