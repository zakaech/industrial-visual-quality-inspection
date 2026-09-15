"""Provenance helpers: every artefact records when, from which code and which config it was made."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vqi import __version__
from vqi.config import Config


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit(short: bool = False) -> str | None:
    """Current git commit hash, or None outside a repository / with git missing."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def git_is_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return bool(out.stdout.strip())


def provenance(cfg: Config, **extra: Any) -> dict[str, Any]:
    """Standard provenance block written into every ``*.meta.json``."""
    return {
        "created_at": now_iso(),
        "vqi_version": __version__,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "config_hash": cfg.hash,
        "config_files": [str(Path(p)) for p in cfg.source_files],
        "seed": cfg.reproducibility.seed,
        **extra,
    }
