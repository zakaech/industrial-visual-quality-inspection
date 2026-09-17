"""Start the inference API locally with uvicorn, using host/port from the config.

Equivalent to: uvicorn vqi.api.app:app --host <api.host> --port <api.port>
Environment: VQI_CONFIG (config path), VQI_BUNDLE_DIR (frozen bundle directory).

Usage (from the repository root):
    python scripts/serve_api.py [--config configs/default.yaml] [--port 8000]
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from vqi.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--bundle-dir", default=None, help="override inference.bundle_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ["VQI_CONFIG"] = args.config
    if args.bundle_dir:
        os.environ["VQI_BUNDLE_DIR"] = args.bundle_dir
    uvicorn.run(
        "vqi.api.app:app",
        host=args.host or cfg.api.host,
        port=args.port or cfg.api.port,
        log_level=cfg.logging.level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
