"""Request/response models of the inference API (Pydantic v2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class ThresholdInfo(BaseModel):
    value: float = Field(description="Frozen decision threshold on the raw model score")
    policy: str = Field(description="Selection rule applied on the validation split")
    target_recall: float
    margin: str
    selected_on: str
    calibration: str = Field(description="'raw' = no post-hoc calibration of the scores")
    frozen_at: str


class ModelInfoResponse(BaseModel):
    model_version: str = Field(examples=["X6_C_winner-4233a9cd8053"])
    model: str
    experiment_id: str
    architecture: dict[str, Any]
    model_sha256: str
    preprocessor: dict[str, Any]
    threshold: ThresholdInfo
    decision_labels: dict[str, str]
    project_version: str
    bundle_frozen_at: str
    validation_summary: dict[str, Any] = Field(
        description="Development metrics (validation split) — the model was selected on them"
    )
    final_test_summary: dict[str, Any] | None = Field(
        default=None, description="Single official test evaluation (Phase 4), if available"
    )


class PredictionResponse(BaseModel):
    image_id: str = Field(description="Uploaded file name (client-side identifier)")
    probability: float = Field(
        ge=0.0, le=1.0, description="Raw model score P(defective); not a calibrated probability"
    )
    decision: str = Field(examples=["REJECT", "ACCEPT"])
    threshold: float = Field(
        description="Frozen threshold: decision = REJECT iff probability >= threshold"
    )
    model_version: str
    latency_ms: float = Field(description="Server-side preprocessing + inference time")
    request_id: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
