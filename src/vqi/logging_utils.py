"""Structured logging.

One JSON object per line (``format: json``) so that logs can be grepped, parsed and shipped;
a human-readable text format is available for interactive use. Extra structured fields are
attached with :func:`log_event`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EXTRA_KEY = "vqi_fields"


class JsonFormatter(logging.Formatter):
    """Serialise a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, _EXTRA_KEY, None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Compact human-readable line with the structured fields appended as key=value."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{datetime.now().strftime('%H:%M:%S')} {record.levelname:<7} {record.name}: "
        base += record.getMessage()
        extra = getattr(record, _EXTRA_KEY, None)
        if extra:
            base += " | " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: str = "INFO", fmt: str = "json", file: Path | None = None) -> None:
    """Configure the root logger once; safe to call repeatedly (handlers are replaced)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter: logging.Formatter = JsonFormatter() if fmt == "json" else TextFormatter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if file is not None:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())  # the file is always machine-readable
        root.addHandler(file_handler)

    # Third-party chatter is not part of the pipeline's structured log.
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log ``msg`` with structured ``fields`` attached to the record."""
    logger.log(level, msg, extra={_EXTRA_KEY: fields})
