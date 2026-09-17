"""Endpoints: GET /health, GET /model-info, POST /predict.

Input validation happens here; scoring and the decision are delegated to the shared
:class:`vqi.inference.Predictor` held in ``app.state`` (loaded once at startup).
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from vqi.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ThresholdInfo,
)
from vqi.logging_utils import get_logger, log_event
from vqi.preprocessing.io import ImageDecodeError

logger = get_logger("vqi.api")
router = APIRouter()

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Empty file"},
    413: {"model": ErrorResponse, "description": "File larger than api.max_upload_mb"},
    415: {"model": ErrorResponse, "description": "Unsupported content type"},
    422: {"model": ErrorResponse, "description": "Missing file or undecodable image"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def _predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
    return predictor


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: ERROR_RESPONSES[503]},
    summary="Liveness probe (used by the Docker HEALTHCHECK)",
    description="Cheap check that the process answers AND the model is loaded; performs no "
    "inference. uvicorn only serves once the lifespan has loaded the bundle, so 'ok' means the "
    "service can score; 503 is returned if the predictor is missing.",
)
def health(request: Request) -> HealthResponse:
    _predictor(request)  # 503 if the bundle is not loaded
    return HealthResponse(status="ok")


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={503: ERROR_RESPONSES[503]},
    summary="Frozen model metadata",
    description="Everything the decision depends on, read from the frozen bundle "
    "(models/final): architecture, checkpoint hash, preprocessing, threshold and its "
    "selection rule, development and final-test summaries.",
)
def model_info(request: Request) -> ModelInfoResponse:
    predictor = _predictor(request)
    info = request.app.state.model_info
    thr = predictor.metadata["threshold"]
    return ModelInfoResponse(
        model_version=info["model_version"],
        model=info["card"]["model"],
        experiment_id=info["card"]["experiment_id"],
        architecture={
            k: info["card"]["architecture"][k]
            for k in ("architecture", "pooling", "n_params")
            if k in info["card"]["architecture"]
        },
        model_sha256=predictor.metadata["model_sha256"],
        preprocessor=predictor.pre.describe(),
        threshold=ThresholdInfo(
            value=float(thr["threshold"]),
            policy=thr["policy"],
            target_recall=float(thr["target_recall"]),
            margin=thr["margin"],
            selected_on=thr["selected_on"],
            calibration=thr["calibration_decision"],
            frozen_at=thr["frozen_at"],
        ),
        decision_labels={
            "positive": predictor.positive_label,
            "negative": predictor.negative_label,
        },
        project_version=info["project_version"],
        bundle_frozen_at=info["card"]["frozen_at"],
        validation_summary=info["validation_summary"],
        final_test_summary=info["final_test_summary"],
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=ERROR_RESPONSES,
    summary="Score one image and decide ACCEPT / REJECT",
    description="Upload one image (multipart/form-data, field `file`; PNG, JPEG or BMP). "
    "The image goes through the frozen preprocessing and model; the decision is "
    "REJECT iff the raw score is >= the frozen threshold. The score is not a calibrated "
    "probability.",
)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:  # noqa: B008
    predictor = _predictor(request)
    request_id = _request_id(request)
    settings = request.app.state.settings

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in settings["allowed_content_types"]:
        log_event(
            logger,
            "predict.rejected",
            level=30,
            request_id=request_id,
            reason="unsupported_content_type",
            content_type=content_type,
        )
        allowed = ", ".join(settings["allowed_content_types"])
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported content type '{content_type}'; allowed: {allowed}",
        )
    limit = settings["max_upload_bytes"]
    data = await file.read(limit + 1)
    if len(data) == 0:
        log_event(logger, "predict.rejected", level=30, request_id=request_id, reason="empty_file")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    if len(data) > limit:
        log_event(
            logger,
            "predict.rejected",
            level=30,
            request_id=request_id,
            reason="too_large",
            limit_bytes=limit,
        )
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"file larger than {limit} bytes"
        )

    t0 = time.perf_counter()
    try:
        pred = predictor.predict(data, image_id=file.filename or "")
    except ImageDecodeError as exc:
        log_event(
            logger,
            "predict.rejected",
            level=30,
            request_id=request_id,
            reason="undecodable_image",
            error=str(exc),
        )
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "the file could not be decoded as an image"
        ) from None
    latency_ms = (time.perf_counter() - t0) * 1000.0
    log_event(
        logger,
        "predict.done",
        request_id=request_id,
        image_id=pred.image_id,
        bytes=len(data),
        content_type=content_type,
        probability=round(pred.probability, 6),
        decision=pred.decision,
        latency_ms=round(latency_ms, 1),
    )
    return PredictionResponse(
        image_id=pred.image_id,
        probability=pred.probability,
        decision=pred.decision,
        threshold=pred.threshold,
        model_version=request.app.state.model_info["model_version"],
        latency_ms=round(latency_ms, 2),
        request_id=request_id,
    )
