"""Application factory.

``create_app(cfg, bundle_dir)`` builds a FastAPI app whose lifespan loads the frozen bundle
**once** (``Predictor.from_bundle``) into ``app.state.predictor``; requests only read it.
The module-level ``app`` (for ``uvicorn vqi.api.app:app``) reads its configuration from the
environment: ``VQI_CONFIG`` (default ``configs/default.yaml``) and ``VQI_BUNDLE_DIR``
(default ``inference.bundle_dir`` of the config).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vqi import __version__
from vqi.api.routes import router
from vqi.config import Config, load_config
from vqi.inference.predictor import BundleError, Predictor
from vqi.logging_utils import get_logger, log_event, setup_logging

logger = get_logger("vqi.api")


def _model_info(cfg: Config, predictor: Predictor, bundle_dir: Path) -> dict[str, Any]:
    card = predictor.metadata["model_card"]
    version = cfg.api.model_version
    if version == "from_bundle":
        version = f"{card['experiment_id']}-{predictor.metadata['model_sha256'][:12]}"
    vm = card.get("validation_metrics", {})
    at95 = next(
        (t for t in vm.get("at_recall_targets", []) if abs(t["recall_target"] - 0.95) < 1e-9), None
    )
    validation_summary = {
        "split": "val (466 images, 49 defective)",
        "average_precision": vm.get("average_precision"),
        "roc_auc": vm.get("roc_auc"),
        "precision_at_recall_0.95": at95["precision"] if at95 else None,
        "fpr_at_recall_0.95": at95["fpr"] if at95 else None,
    }
    final_test = None
    test_path = bundle_dir / "final_test_metrics.json"
    if test_path.exists():
        t = json.loads(test_path.read_text(encoding="utf-8"))
        final_test = {
            "evaluated_at": t["evaluated_at"],
            "n": t["n"],
            "n_pos": t["n_pos"],
            "average_precision": t["average_precision"],
            "roc_auc": t["roc_auc"],
            "at_threshold": {
                k: t["at_threshold"][k]
                for k in ("recall", "precision", "fpr", "tp", "fp", "fn", "tn")
            },
        }
    return {
        "model_version": version,
        "card": card,
        "project_version": __version__,
        "validation_summary": validation_summary,
        "final_test_summary": final_test,
    }


def create_app(cfg: Config | None = None, bundle_dir: str | Path | None = None) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("VQI_CONFIG", "configs/default.yaml"))
    bundle = Path(bundle_dir or os.environ.get("VQI_BUNDLE_DIR") or cfg.inference.bundle_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(cfg.logging.level, cfg.logging.format, cfg.logging.file)
        t0 = time.perf_counter()
        try:
            predictor = Predictor.from_bundle(cfg, bundle)
        except (BundleError, FileNotFoundError) as exc:
            log_event(logger, "startup.bundle_error", level=40, bundle=str(bundle), error=str(exc))
            raise
        app.state.predictor = predictor
        app.state.model_info = _model_info(cfg, predictor, bundle)
        app.state.settings = {
            "allowed_content_types": [c.lower() for c in cfg.api.allowed_content_types],
            "max_upload_bytes": int(cfg.api.max_upload_mb * 1024 * 1024),
        }
        app.state.load_count = getattr(app.state, "load_count", 0) + 1
        log_event(
            logger,
            "startup.model_loaded",
            bundle=str(bundle),
            model_version=app.state.model_info["model_version"],
            model_sha256=predictor.metadata["model_sha256"][:16],
            threshold=predictor.threshold,
            seconds=round(time.perf_counter() - t0, 2),
        )
        yield
        log_event(logger, "shutdown")

    app = FastAPI(
        title="Industrial Visual Quality Inspection — inference API",
        version=__version__,
        description=(
            "Binary surface-defect classifier (ResNet-18, KolektorSDD2). "
            "Every request goes through the same frozen preprocessing, model and threshold "
            "that produced the Phase 4 final evaluation. Scores are raw model outputs, not "
            "calibrated probabilities."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    if cfg.api.cors_origins:
        # browsers only (the Angular interface): explicit origins, no wildcard, no credentials
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.api.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", None)
        log_event(logger, "request.invalid", level=30, request_id=rid, path=request.url.path)
        return JSONResponse(
            status_code=422,
            content={
                "detail": "invalid request: expected a multipart field 'file'",
                "request_id": rid,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        log_event(
            logger,
            "request.failed",
            level=40,
            request_id=rid,
            path=request.url.path,
            error=type(exc).__name__,
        )
        # this handler runs outside request_id_middleware and CORSMiddleware, so both headers
        # are set here (otherwise a browser client would see a network error instead of a 500)
        headers = {"X-Request-ID": rid} if rid else {}
        origin = request.headers.get("origin")
        if origin and origin in cfg.api.cors_origins:
            headers.update({"Access-Control-Allow-Origin": origin, "Vary": "Origin"})
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "request_id": rid},
            headers=headers,
        )

    return app


# ASGI entry point for `uvicorn vqi.api.app:app` (config read now, model loaded at startup)
app = create_app()
