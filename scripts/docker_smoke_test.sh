#!/usr/bin/env bash
# Smoke test of a built image, used by CI and locally (Git Bash on Windows works too).
#
#   scripts/docker_smoke_test.sh IMAGE [synthetic|real]
#
# synthetic (default, what CI runs): a synthetic bundle is generated INSIDE the image
#   (python -m vqi.inference.synthetic_bundle — untrained weights, no download), mounted read-only
#   and selected with VQI_BUNDLE_DIR; /health, /model-info (version must start with SYN-) and
#   /predict on the bundle's sample.png are checked, then the Docker HEALTHCHECK must turn healthy.
# real: the bundle baked into the image is used; /model-info must expose the frozen SHA-256 and
#   threshold read from models/final/ on the host. No image of the official test split is used.
set -euo pipefail

IMAGE="${1:?usage: $0 IMAGE [synthetic|real]}"
MODE="${2:-synthetic}"
PORT="${SMOKE_PORT:-8000}"
NAME="vqi-smoke-$$"
export MSYS_NO_PATHCONV=1  # Git Bash: do not rewrite /bundle into a Windows path

log() { printf '[smoke] %s\n' "$*"; }
# a working Python on the host, for JSON parsing (PYTHON=... to force one; Windows aliases fail)
PY=""
for cand in "${PYTHON:-}" python3 python; do
    if [ -n "$cand" ] && "$cand" -c "pass" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { log "no usable python on the host"; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

host_pwd="$(pwd -W 2>/dev/null || pwd)"
run_args=(-d --name "$NAME" -p "${PORT}:8000")
if [ "$MODE" = "synthetic" ]; then
    mkdir -p ci_bundle && chmod 777 ci_bundle
    log "generating a synthetic bundle inside the image"
    docker run --rm -v "${host_pwd}/ci_bundle:/bundle" "$IMAGE" \
        python -m vqi.inference.synthetic_bundle --out /bundle
    run_args+=(-e VQI_BUNDLE_DIR=/bundle -v "${host_pwd}/ci_bundle:/bundle:ro")
    sample="ci_bundle/sample.png"
fi

log "starting $IMAGE ($MODE bundle) as $NAME"
docker run "${run_args[@]}" "$IMAGE" >/dev/null

log "waiting for GET /health"
ok=0
for _ in $(seq 1 45); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 2
done
if [ "$ok" != 1 ]; then log "FAIL: /health never answered"; docker logs "$NAME"; exit 1; fi
health="$(curl -fsS "http://127.0.0.1:${PORT}/health")"
[ "$health" = '{"status":"ok"}' ] || { log "FAIL: unexpected /health body: $health"; exit 1; }
log "/health -> $health"

info="$(curl -fsS "http://127.0.0.1:${PORT}/model-info")"
version="$(printf '%s' "$info" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model_version"])')"
sha="$(printf '%s' "$info" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model_sha256"])')"
thr="$(printf '%s' "$info" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["threshold"]["value"])')"
log "/model-info -> model_version=$version sha256=${sha:0:16}... threshold=$thr"
if [ "$MODE" = "synthetic" ]; then
    case "$version" in SYN-*) ;; *) log "FAIL: expected a SYN- version, got $version"; exit 1;; esac
else
    expected_sha="$(sha256sum models/final/model.pt | cut -d' ' -f1)"
    expected_thr="$("$PY" -c 'import json; print(json.load(open("models/final/threshold.json"))["threshold"])')"
    [ "$sha" = "$expected_sha" ] || { log "FAIL: sha256 $sha != host $expected_sha"; exit 1; }
    [ "$thr" = "$expected_thr" ] || { log "FAIL: threshold $thr != host $expected_thr"; exit 1; }
    log "sha256 and threshold match models/final/ on the host"
fi

if [ "$MODE" = "synthetic" ]; then
    pred="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/predict" -F "file=@${sample};type=image/png")"
    decision="$(printf '%s' "$pred" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["decision"])')"
    case "$decision" in ACCEPT|REJECT) ;; *) log "FAIL: /predict -> $pred"; exit 1;; esac
    log "/predict (synthetic sample) -> $pred"
fi

code="$(curl -s -w '%{http_code}' -X POST "http://127.0.0.1:${PORT}/predict" -F "file=@requirements-api.txt;type=text/plain" | tail -c 3)"
[ "$code" = "415" ] || { log "FAIL: text upload expected 415, got $code"; exit 1; }
log "/predict text/plain -> 415"

log "waiting for the Docker HEALTHCHECK to report healthy"
status=starting
for _ in $(seq 1 60); do
    status="$(docker inspect --format '{{.State.Health.Status}}' "$NAME")"
    [ "$status" = "healthy" ] && break
    sleep 2
done
[ "$status" = "healthy" ] || { log "FAIL: health status is $status"; docker logs "$NAME"; exit 1; }
log "container health: $status"
log "running as: $(docker exec "$NAME" id)"
log "OK"
