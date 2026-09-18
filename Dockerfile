# syntax=docker/dockerfile:1
# Inference API image — CPU only, Python 3.11, non-root.
#
# Bundle strategy (option A): the frozen bundle models/final/ is copied INTO the image, so the
# image is self-contained and serves exactly the Phase 4 model (SHA-256 verifiable at runtime via
# GET /model-info). model.pt is git-ignored: a build from a bare clone fails on purpose unless
# --build-arg REQUIRE_MODEL=0 is given (CI builds an app-only image and mounts a synthetic bundle
# through VQI_BUNDLE_DIR). Nothing is ever downloaded at run time.
#
#   docker build -t vqi-api .
#   docker run --rm -p 8000:8000 vqi-api

FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VQI_CONFIG=configs/default.yaml

WORKDIR /app

# 1. pinned runtime dependencies (own layer: rebuilt only when the pins change)
COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

# 2. the package itself (README.md is referenced by pyproject's `readme`)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-deps . && rm -rf build

# 3. configuration + frozen bundle (model.pt, preprocessor.json, threshold.json, model_card.json,
#    final_test_metrics.json). No ML value is redefined here: everything is read from these files.
COPY configs/default.yaml configs/default.yaml
COPY models/final/ models/final/
ARG REQUIRE_MODEL=1
RUN if [ "$REQUIRE_MODEL" = "1" ] && [ ! -f models/final/model.pt ]; then \
        echo "ERROR: models/final/model.pt is missing (git-ignored). Copy the frozen bundle," \
             "or build an app-only image with --build-arg REQUIRE_MODEL=0." >&2; \
        exit 1; \
    fi

# 4. non-root user; logs/ is the only directory the service writes to (logging.file)
RUN useradd --system --uid 10001 --user-group --no-create-home app \
    && mkdir -p /app/logs && chown app:app /app/logs
USER app

EXPOSE 8000

# Liveness: GET /health, no inference. uvicorn only serves once the lifespan has loaded the
# bundle (a load failure aborts the process), so a 200 here means "process up AND model loaded".
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "vqi.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
