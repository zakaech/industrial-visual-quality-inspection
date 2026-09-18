import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  inject,
  signal,
  viewChild,
} from '@angular/core';

import { environment } from '../../../environments/environment';
import { describeHttpError } from '../../core/api-error';
import { InspectionService } from '../../core/services/inspection.service';
import {
  InspectionError,
  ModelInfoResponse,
  PredictionResponse,
} from '../../models/inspection.models';

export type InspectionStatus = 'idle' | 'ready' | 'loading' | 'success' | 'error';
export type ApiState = 'checking' | 'online' | 'offline';

/** Image → Inspect → Decision. Everything shown comes from the API; nothing ML is computed here. */
@Component({
  selector: 'app-inspection-page',
  imports: [],
  templateUrl: './inspection-page.html',
  styleUrl: './inspection-page.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class InspectionPage {
  private readonly api = inject(InspectionService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly fileInput = viewChild.required<ElementRef<HTMLInputElement>>('fileInput');

  readonly apiBaseUrl = this.api.baseUrl;
  readonly maxUploadMb = environment.maxUploadMb;
  readonly allowedTypes = environment.allowedTypes;
  readonly acceptAttribute = environment.allowedTypes.join(',');

  // ---------------------------------------------------------------- state
  readonly selectedFile = signal<File | null>(null);
  readonly previewUrl = signal<string | null>(null);
  readonly fileError = signal<string | null>(null);
  readonly dragActive = signal(false);
  readonly status = signal<InspectionStatus>('idle');
  readonly result = signal<PredictionResponse | null>(null);
  readonly error = signal<InspectionError | null>(null);
  readonly modelInfo = signal<ModelInfoResponse | null>(null);
  readonly apiState = signal<ApiState>('checking');

  readonly canInspect = computed(() => this.selectedFile() !== null && this.status() !== 'loading');
  /** The label the API uses for the defect class (from /model-info), 'REJECT' until it is known. */
  readonly positiveLabel = computed(
    () => this.modelInfo()?.decision_labels['positive'] ?? 'REJECT',
  );
  readonly isReject = computed(() => this.result()?.decision === this.positiveLabel());

  constructor() {
    this.loadModelInfo();
    this.destroyRef.onDestroy(() => this.revokePreview());
  }

  // ---------------------------------------------------------------- model info / connectivity
  loadModelInfo(): void {
    this.apiState.set('checking');
    this.api.modelInfo().subscribe({
      next: (info) => {
        this.modelInfo.set(info);
        this.apiState.set('online');
      },
      error: () => {
        this.modelInfo.set(null);
        this.apiState.set('offline');
      },
    });
  }

  // ---------------------------------------------------------------- file selection
  openFilePicker(): void {
    this.fileInput().nativeElement.click();
  }

  onFileInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    if (file) {
      this.selectFile(file);
    }
    input.value = ''; // allow re-selecting the same file
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragActive.set(true);
  }

  onDragLeave(): void {
    this.dragActive.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragActive.set(false);
    const file = event.dataTransfer?.files?.[0] ?? null;
    if (file) {
      this.selectFile(file);
    }
  }

  /** Client-side pre-check only; the API performs the final validation (400/413/415/422). */
  selectFile(file: File): void {
    const problem = this.validate(file);
    if (problem) {
      this.clearFile();
      this.fileError.set(problem);
      return;
    }
    this.revokePreview();
    this.fileError.set(null);
    this.selectedFile.set(file);
    this.previewUrl.set(URL.createObjectURL(file));
    this.result.set(null);
    this.error.set(null);
    this.status.set('ready');
  }

  clearFile(): void {
    this.revokePreview();
    this.selectedFile.set(null);
    this.fileError.set(null);
    this.result.set(null);
    this.error.set(null);
    this.status.set('idle');
  }

  private validate(file: File): string | null {
    const type = (file.type || '').toLowerCase();
    if (!this.allowedTypes.includes(type)) {
      return `Unsupported file type${type ? ` (${type})` : ''}. Please choose a PNG, JPEG or BMP image.`;
    }
    if (file.size === 0) {
      return 'The selected file is empty.';
    }
    if (file.size > this.maxUploadMb * 1024 * 1024) {
      return `The file is too large (${this.formatBytes(file.size)}). Maximum size: ${this.maxUploadMb} MB.`;
    }
    return null;
  }

  private revokePreview(): void {
    const url = this.previewUrl();
    if (url) {
      URL.revokeObjectURL(url);
      this.previewUrl.set(null);
    }
  }

  // ---------------------------------------------------------------- inspection
  inspect(): void {
    const file = this.selectedFile();
    if (!file || this.status() === 'loading') {
      return; // no concurrent requests
    }
    this.status.set('loading');
    this.result.set(null);
    this.error.set(null);
    this.api.predict(file).subscribe({
      next: (prediction) => {
        this.result.set(prediction);
        this.status.set('success');
        this.apiState.set('online');
      },
      error: (err: unknown) => {
        const described = describeHttpError(err);
        this.error.set(described);
        this.status.set('error');
        if (described.kind === 'offline') {
          this.apiState.set('offline');
        }
      },
    });
  }

  // ---------------------------------------------------------------- formatting (display only)
  formatPercent(fraction: number, digits = 2): string {
    return `${(fraction * 100).toFixed(digits)} %`;
  }

  formatScore(fraction: number): string {
    return fraction.toFixed(6);
  }

  formatLatency(ms: number): string {
    return `${Math.round(ms)} ms`;
  }

  formatBytes(bytes: number): string {
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  architectureLabel(info: ModelInfoResponse): string {
    const arch = info.architecture['architecture'];
    const params = info.architecture['n_params'];
    const name = typeof arch === 'string' ? arch : 'unknown';
    return typeof params === 'number' ? `${name} · ${(params / 1e6).toFixed(2)} M params` : name;
  }

  inputSizeLabel(info: ModelInfoResponse): string {
    const size = info.preprocessor['input_size'];
    return Array.isArray(size) && size.length === 2 ? `${size[0]} × ${size[1]} px` : '—';
  }
}
