"""Shared command-line plumbing for the scripts in ``scripts/``.

Every script takes ``--config`` (default ``configs/default.yaml``) and optional ``--override``
files, sets up structured logging and seeds every random source before doing anything else.
Relative paths in the config are resolved against the current working directory, which must be
the repository root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vqi.config import Config, load_config
from vqi.logging_utils import get_logger, log_event, setup_logging
from vqi.seed import seed_everything

logger = get_logger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/default.yaml", help="base YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="YAML",
        help="override YAML merged on top of --config (repeatable)",
    )
    parser.add_argument(
        "--log-format",
        choices=["json", "text"],
        default=None,
        help="override logging.format from the config",
    )
    return parser


def bootstrap(args: argparse.Namespace) -> tuple[Config, Path]:
    """Load config, configure logging, seed. Returns (config, project_root)."""
    cfg = load_config(args.config, overrides=args.override)
    setup_logging(
        level=cfg.logging.level,
        fmt=args.log_format or cfg.logging.format,
        file=cfg.logging.file,
    )
    seed_everything(
        cfg.reproducibility.seed,
        deterministic=cfg.reproducibility.deterministic,
        num_threads=cfg.reproducibility.torch_num_threads,
    )
    project_root = Path.cwd()
    log_event(
        logger,
        "bootstrap",
        config=str(args.config),
        overrides=list(args.override),
        config_hash=cfg.hash,
        seed=cfg.reproducibility.seed,
        cwd=str(project_root),
    )
    return cfg, project_root
