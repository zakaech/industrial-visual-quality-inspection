"""Typed configuration loaded from YAML.

Every hyper-parameter of the pipeline lives in ``configs/*.yaml``. This module turns the YAML
tree into frozen dataclasses so that unknown keys, missing keys and wrong types are rejected at
load time, and so that no module ever hard-codes a value.

Usage::

    cfg = load_config("configs/default.yaml", overrides=["configs/experiments/x.yaml"])
    cfg.training.batch_size
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the YAML tree does not match the typed schema."""


# --------------------------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    task: str
    positive_class_name: str
    negative_class_name: str


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int
    deterministic: bool
    torch_num_threads: int


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    splits_dir: Path
    runs_dir: Path
    logs_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class ExpectedCounts:
    train_total: int
    train_defective: int
    test_total: int
    test_defective: int


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source_url: str
    archive_name: str
    archive_sha256: str
    raw_subdir: str
    train_subdir: str
    test_subdir: str
    image_ext: str
    mask_suffix: str
    image_id_pattern: str
    label_rule: str
    use_official_split: bool
    expected_counts: ExpectedCounts


@dataclass(frozen=True)
class DuplicatesConfig:
    exact_hash: str
    near_duplicate_method: str
    phash_size: int
    phash_highfreq_factor: int
    max_hamming_distance: int | None
    candidate_thresholds: tuple[int, ...]
    policy: str


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    val_fraction: float
    stratify_on: str
    group_column: str | None
    duplicates: DuplicatesConfig
    test_policy: str
    manifest_version: str


@dataclass(frozen=True)
class InputSize:
    height: int
    width: int


@dataclass(frozen=True)
class NormalizationConfig:
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class EnhancementConfig:
    active: str
    # variant name -> ordered list of {"op": name, **params}; validated when a pipeline is built
    variants: dict[str, Any]


@dataclass(frozen=True)
class PreprocessingConfig:
    color_mode: str
    input_size: InputSize
    resize_mode: str
    pad_value: str
    interpolation: str
    normalization: NormalizationConfig
    enhancement: EnhancementConfig


@dataclass(frozen=True)
class BrightnessContrastConfig:
    enabled: bool
    brightness_limit: float
    contrast_limit: float
    p: float


@dataclass(frozen=True)
class RandomCropConfig:
    enabled: bool


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_p: float
    vertical_flip_p: float
    rotate_90_p: float
    brightness_contrast: BrightnessContrastConfig
    random_crop: RandomCropConfig


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    pretrained: bool
    num_outputs: int
    dropout: float
    freeze_backbone: bool
    freeze_bn_stats: bool


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    warmup_epochs: int


@dataclass(frozen=True)
class LossConfig:
    name: str
    pos_weight: float
    focal_gamma: float


@dataclass(frozen=True)
class EarlyStoppingConfig:
    monitor: str
    mode: str
    patience: int


@dataclass(frozen=True)
class CheckpointConfig:
    save_best_only: bool
    monitor: str


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    num_workers: int
    epochs: int
    batch_size: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    sampler: str
    early_stopping: EarlyStoppingConfig
    checkpoint: CheckpointConfig
    grad_clip_norm: float
    log_every_n_steps: int
    dry_run_epochs: int


@dataclass(frozen=True)
class ThresholdPolicyConfig:
    name: str
    target_recall: float
    cost_fn: float
    cost_fp: float
    f_beta: float


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int
    confidence_level: float


@dataclass(frozen=True)
class CalibrationConfig:
    n_bins: int
    temperature_scaling: bool


@dataclass(frozen=True)
class GradCamConfig:
    enabled: bool
    target_layer: str
    n_examples: int


@dataclass(frozen=True)
class EvaluationConfig:
    positive_label: int
    threshold_policy: ThresholdPolicyConfig
    report_recall_targets: tuple[float, ...]
    bootstrap: BootstrapConfig
    calibration: CalibrationConfig
    defect_size_analysis: bool
    gradcam: GradCamConfig


@dataclass(frozen=True)
class ContactSheetConfig:
    n_per_class: int
    thumb_height: int


@dataclass(frozen=True)
class EdaConfig:
    report_subdir: str
    contact_sheet: ContactSheetConfig
    defect_area_bins: tuple[int, ...]
    n_closest_pairs: int


@dataclass(frozen=True)
class LbpScale:
    points: int
    radius: int


@dataclass(frozen=True)
class LocalStatsConfig:
    patch_size: int


@dataclass(frozen=True)
class SvmConfig:
    kernel: str
    c_grid: tuple[float, ...]
    gamma_grid: tuple[Any, ...]
    cv_folds: int
    cv_scoring: str


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_features: str
    class_weight: str


@dataclass(frozen=True)
class RepresentativeImagesConfig:
    n_healthy: int
    small_defect_max_area: int


@dataclass(frozen=True)
class CnrConfig:
    ring_width: int
    area_bins: tuple[int, ...]
    variants: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    features: str
    preprocessing: str
    classifier: str
    class_weight: str
    pad_value: str | None = None
    # optional per-experiment SVM grid (grid-edge checks); defaults to baseline.svm
    c_grid: tuple[float, ...] | None = None
    gamma_grid: tuple[Any, ...] | None = None
    kernel: str | None = None  # defaults to baseline.svm.kernel; "linear" ignores gamma


@dataclass(frozen=True)
class BaselineConfig:
    report_subdir: str
    features_cache_dir: Path
    allowed_splits: tuple[str, ...]
    histogram_bins: int
    lbp: tuple[LbpScale, ...]
    local_stats: LocalStatsConfig
    feature_groups: dict[str, Any]
    svm: SvmConfig
    class_weights: dict[str, Any]
    random_forest: RandomForestConfig
    representative_images: RepresentativeImagesConfig
    cnr: CnrConfig
    experiments: tuple[ExperimentSpec, ...]


@dataclass(frozen=True)
class InferenceConfig:
    export_format: str
    bundle_dir: Path
    batch_size: int


@dataclass(frozen=True)
class DecisionLabels:
    positive: str
    negative: str


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    model_version: str
    max_upload_mb: int
    allowed_content_types: tuple[str, ...]
    decision_labels: DecisionLabels


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    file: Path | None


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    reproducibility: ReproducibilityConfig
    paths: PathsConfig
    dataset: DatasetConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    eda: EdaConfig
    baseline: BaselineConfig
    inference: InferenceConfig
    api: ApiConfig
    logging: LoggingConfig
    # Merged raw tree (kept for hashing and for freezing a copy next to every run).
    raw: dict[str, Any] = field(repr=False, compare=False)
    source_files: tuple[Path, ...] = field(default=(), repr=False, compare=False)

    @property
    def hash(self) -> str:
        """Short SHA-256 of the canonical JSON of the merged tree — identifies a configuration."""
        return config_hash(self.raw)


# --------------------------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------------------------- #


def _coerce(hint: Any, value: Any, path: str) -> Any:
    """Recursively validate/convert ``value`` against the type ``hint``."""
    origin = typing.get_origin(hint)

    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(hint)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed")
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value, path)
        raise ConfigError(f"{path}: unsupported union type {hint}")

    if is_dataclass(hint):
        return _build(hint, value, path)

    if origin in (tuple, list):
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        args = typing.get_args(hint)
        elem = args[0] if args else Any
        return tuple(_coerce(elem, v, f"{path}[{i}]") for i, v in enumerate(value))

    if hint is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a path string, got {type(value).__name__}")
        return Path(value)

    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
        return float(value)

    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {type(value).__name__}")
        return value

    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {type(value).__name__}")
        return value

    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {type(value).__name__}")
        return value

    return value


def _build(cls: type, data: Any, path: str) -> Any:
    """Instantiate dataclass ``cls`` from a mapping, rejecting unknown and missing keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls) if f.init}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}")
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            has_default = (
                f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            )
            if not has_default:
                raise ConfigError(f"{path}: missing required key '{name}'")
            continue
        kwargs[name] = _coerce(hints[name], data[name], f"{path}.{name}")
    return cls(**kwargs)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``override`` is merged recursively on top of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def config_hash(tree: dict[str, Any]) -> str:
    canonical = json.dumps(tree, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_config(
    path: str | Path,
    overrides: typing.Sequence[str | Path] = (),
) -> Config:
    """Load a YAML config, merge optional override files on top, validate and type it."""
    sources = [Path(path), *map(Path, overrides)]
    merged: dict[str, Any] = {}
    for src in sources:
        merged = deep_merge(merged, read_yaml(src))

    section_fields = [f for f in fields(Config) if f.name not in ("raw", "source_files")]
    unknown = sorted(set(merged) - {f.name for f in section_fields})
    if unknown:
        raise ConfigError(f"config: unknown top-level sections {unknown}")
    hints = typing.get_type_hints(Config)
    sections = {}
    for f in section_fields:
        if f.name not in merged:
            raise ConfigError(f"config: missing section '{f.name}'")
        sections[f.name] = _build(hints[f.name], merged[f.name], f.name)
    return Config(**sections, raw=merged, source_files=tuple(sources))


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write the merged tree next to a run so that the run is fully reproducible."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg.raw, fh, sort_keys=False, allow_unicode=True)
