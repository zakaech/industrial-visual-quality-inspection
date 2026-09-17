"""Inference API: endpoints, input validation, single model load, Predictor/API contract.

Most tests run on a *synthetic* bundle (untrained ResNet-18 weights, threshold 0.5) so they need
no real data. The contract test on the real frozen bundle uses a **validation** image (never a
test image) and is skipped when the bundle or the data are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from vqi.api import create_app
from vqi.config import Config, load_config
from vqi.inference.predictor import Predictor
from vqi.models.factory import build_model
from vqi.preprocessing import Preprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_BUNDLE = REPO_ROOT / "models" / "final"


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def synthetic_bundle(cfg: Config, tmp_path_factory) -> Path:
    """A bundle with untrained weights: exercises the whole API path without real data."""
    d = tmp_path_factory.mktemp("bundle")
    torch.manual_seed(0)
    model = build_model(cfg, 197, 1668)
    torch.nn.init.normal_(model.head[1].weight, std=0.05)
    pre = Preprocessor.from_config(cfg)
    torch.save(
        {"model_state": model.state_dict(), "pooling": model.pooling, "stage": "test"},
        d / "model.pt",
    )
    (d / "preprocessor.json").write_text(json.dumps(pre.describe()), encoding="utf-8")
    (d / "threshold.json").write_text(
        json.dumps(
            {
                "frozen": True,
                "threshold": 0.5,
                "policy": "min_recall",
                "target_recall": 0.95,
                "margin": "midpoint_logit",
                "selected_on": "synthetic",
                "calibration_decision": "raw",
                "frozen_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (d / "model_card.json").write_text(
        json.dumps(
            {
                "model": "synthetic",
                "experiment_id": "SYN",
                "frozen_at": "2026-01-01T00:00:00+00:00",
                "architecture": {
                    "architecture": "resnet18",
                    "pooling": "avg",
                    "n_params": 11177025,
                },
                "validation_metrics": {
                    "average_precision": 0.5,
                    "roc_auc": 0.5,
                    "at_recall_targets": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return d


@pytest.fixture(scope="module")
def client(cfg: Config, synthetic_bundle: Path):
    app = create_app(cfg, synthetic_bundle)
    with TestClient(app) as c:
        yield c


def _png_bytes(h: int = 637, w: int = 229, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 60, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------------------ endpoints


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert "x-request-id" in {k.lower() for k in r.headers}


def test_model_info(client: TestClient):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "model_version",
        "model_sha256",
        "preprocessor",
        "threshold",
        "decision_labels",
        "project_version",
        "bundle_frozen_at",
        "validation_summary",
    ):
        assert key in body, key
    assert body["threshold"]["value"] == 0.5 and body["threshold"]["calibration"] == "raw"
    assert body["preprocessor"]["variant"] == "none" and body["preprocessor"]["input_size"] == [
        640,
        224,
    ]
    assert body["model_version"].startswith("SYN-")
    assert body["decision_labels"] == {"positive": "REJECT", "negative": "ACCEPT"}


def test_predict_valid_image(client: TestClient):
    r = client.post("/predict", files={"file": ("part.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {
        "image_id",
        "probability",
        "decision",
        "threshold",
        "model_version",
        "latency_ms",
        "request_id",
    }
    assert body["image_id"] == "part.png" and 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == 0.5 and body["latency_ms"] > 0
    expected = "REJECT" if body["probability"] >= body["threshold"] else "ACCEPT"
    assert body["decision"] == expected


def test_predict_accepts_jpeg_and_bmp(client: TestClient):
    img = np.random.default_rng(1).integers(30, 60, size=(100, 40, 3), dtype=np.uint8)
    for ext, ctype in ((".jpg", "image/jpeg"), (".bmp", "image/bmp")):
        ok, buf = cv2.imencode(ext, img)
        r = client.post("/predict", files={"file": (f"p{ext}", buf.tobytes(), ctype)})
        assert r.status_code == 200, (ext, r.text)


# ------------------------------------------------------------------------------ invalid inputs


def test_predict_missing_file(client: TestClient):
    r = client.post("/predict")
    assert r.status_code == 422 and "file" in r.json()["detail"]


def test_predict_empty_file(client: TestClient):
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_predict_text_file_is_rejected(client: TestClient):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_corrupted_image(client: TestClient):
    r = client.post("/predict", files={"file": ("bad.png", b"\x89PNG not really", "image/png")})
    assert r.status_code == 422 and "decoded" in r.json()["detail"]


def test_predict_too_large(cfg: Config, synthetic_bundle: Path):
    import yaml

    override = synthetic_bundle / "small_limit.yaml"
    override.write_text(yaml.safe_dump({"api": {"max_upload_mb": 1}}), encoding="utf-8")
    small = load_config(REPO_ROOT / "configs" / "default.yaml", overrides=[override])
    # 1024x1024 BMP (uncompressed) = 3 MB > 1 MB limit; rejected before any decoding
    ok, buf = cv2.imencode(".bmp", np.zeros((1024, 1024, 3), dtype=np.uint8))
    assert ok and len(buf) > 1024 * 1024
    with TestClient(create_app(small, synthetic_bundle)) as c:
        r = c.post("/predict", files={"file": ("big.bmp", buf.tobytes(), "image/bmp")})
        assert r.status_code == 413
        assert c.get("/health").status_code == 200  # the process survives the rejection


def test_no_traceback_leaks_on_internal_error(cfg: Config, synthetic_bundle: Path):
    class Broken:
        threshold = 0.5

        def predict(self, *_a, **_k):
            raise RuntimeError("boom")

    # raise_server_exceptions=False: observe the HTTP response the client would really receive
    with TestClient(create_app(cfg, synthetic_bundle), raise_server_exceptions=False) as c:
        c.app.state.predictor = Broken()
        r = c.post("/predict", files={"file": ("p.png", _png_bytes(), "image/png")})
    assert r.status_code == 500 and r.json()["detail"] == "internal error"
    assert r.json()["request_id"] == r.headers["x-request-id"]
    assert "boom" not in r.text and "Traceback" not in r.text


# ------------------------------------------------------------------------------ single load


def test_model_is_loaded_once(client: TestClient):
    app = client.app
    before = id(app.state.predictor)
    assert app.state.load_count == 1
    for i in range(3):
        client.post("/predict", files={"file": (f"p{i}.png", _png_bytes(seed=i), "image/png")})
    assert id(app.state.predictor) == before and app.state.load_count == 1


# ------------------------------------------------------------------------------ contract


def test_api_matches_predictor_on_synthetic_bundle(
    client: TestClient, cfg: Config, synthetic_bundle: Path
):
    data = _png_bytes(seed=7)
    direct = Predictor.from_bundle(cfg, synthetic_bundle).predict(data, image_id="p.png")
    r = client.post("/predict", files={"file": ("p.png", data, "image/png")})
    assert r.status_code == 200
    # same bytes → same imdecode → same tensor → same forward pass: float32 arithmetic on identical
    # inputs is bit-reproducible on this CPU (dry-run), so the tolerance is JSON float rounding
    assert abs(r.json()["probability"] - direct.probability) < 1e-9
    assert r.json()["decision"] == direct.decision


@pytest.mark.skipif(
    not (REAL_BUNDLE / "model.pt").exists()
    or not (REPO_ROOT / "data" / "splits" / "v1.csv").exists(),
    reason="real frozen bundle or data not available",
)
def test_api_matches_predictor_on_real_bundle_with_a_validation_image(cfg: Config):
    """Uses one VALIDATION image (never the official test split)."""
    import pandas as pd

    splits = pd.read_csv(REPO_ROOT / "data" / "splits" / "v1.csv", dtype={"image_id": str})
    row = splits[splits["split"] == "val"].iloc[0]
    path = REPO_ROOT / row["image_path"]
    if not path.exists():
        pytest.skip("raw data not available")
    predictor = Predictor.from_bundle(cfg, REAL_BUNDLE)
    direct = predictor.predict(path, image_id=row["image_id"])
    with TestClient(create_app(cfg, REAL_BUNDLE)) as c:
        info = c.get("/model-info").json()
        assert info["threshold"]["value"] == pytest.approx(0.018752, abs=1e-6)
        assert info["model_sha256"].startswith("4233a9cd80534b8e")
        r = c.post(
            "/predict", files={"file": (f"{row['image_id']}.png", path.read_bytes(), "image/png")}
        )
    assert r.status_code == 200
    assert abs(r.json()["probability"] - direct.probability) < 1e-9
    assert r.json()["decision"] == direct.decision
    assert r.json()["threshold"] == pytest.approx(0.018752, abs=1e-6)


def test_openapi_documents_the_endpoints(client: TestClient):
    spec = client.get("/openapi.json").json()
    assert {"/health", "/model-info", "/predict"} <= set(spec["paths"])
    predict = spec["paths"]["/predict"]["post"]
    assert {"200", "400", "413", "415", "422", "503"} <= set(predict["responses"])
    assert client.get("/docs").status_code == 200
