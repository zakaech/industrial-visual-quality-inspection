"""Download and extract the dataset archive under an integrity check.

The archive's SHA-256 is pinned in the config (``dataset.archive_sha256``). A mismatch aborts
before extraction: the whole pipeline is only reproducible if it starts from the same bytes.
The extracted tree under ``data/raw/<raw_subdir>/`` is considered immutable — nothing in the
pipeline ever writes to or deletes from it; exclusions are expressed in the manifest.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from vqi.config import Config
from vqi.data.hashing import sha256_file
from vqi.logging_utils import get_logger, log_event

logger = get_logger(__name__)


class IntegrityError(RuntimeError):
    """The archive on disk does not match the pinned checksum."""


@dataclass(frozen=True)
class RawLayout:
    """Where the raw data lives, derived from the config."""

    archive: Path
    root: Path
    train_dir: Path
    test_dir: Path


def raw_layout(cfg: Config) -> RawLayout:
    root = cfg.paths.raw_dir / cfg.dataset.raw_subdir
    return RawLayout(
        archive=cfg.paths.raw_dir / cfg.dataset.archive_name,
        root=root,
        train_dir=root / cfg.dataset.train_subdir,
        test_dir=root / cfg.dataset.test_subdir,
    )


def download_archive(url: str, target: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream ``url`` to ``target`` (via a temporary file so a partial download is never mistaken
    for a complete one)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    log_event(logger, "download.start", url=url, target=str(target))
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0.1
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                log_event(logger, "download.progress", fraction=round(done / total, 2))
                next_report += 0.1
    tmp.replace(target)
    log_event(logger, "download.done", bytes=target.stat().st_size)
    return target


def verify_archive(archive: Path, expected_sha256: str) -> str:
    """Return the archive's SHA-256, raising :class:`IntegrityError` if it differs from the pin.

    ``expected_sha256 == "TBD"`` is the bootstrap case: the digest is computed and returned so
    the caller can pin it, and a warning is logged because nothing was verified.
    """
    actual = sha256_file(archive)
    if expected_sha256.upper() == "TBD":
        log_event(logger, "integrity.unpinned", level=30, sha256=actual, archive=str(archive))
        return actual
    if actual.lower() != expected_sha256.lower():
        raise IntegrityError(
            f"SHA-256 mismatch for {archive}: expected {expected_sha256}, got {actual}"
        )
    log_event(logger, "integrity.ok", sha256=actual, archive=str(archive))
    return actual


def extract_archive(archive: Path, root: Path, overwrite: bool = False) -> Path:
    """Extract the zip into ``root``. Refuses to touch an existing extraction unless asked."""
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            log_event(logger, "extract.skip_existing", root=str(root))
            return root
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        # Guard against path traversal in archive member names.
        for member in zf.namelist():
            dest = (root / member).resolve()
            if root.resolve() not in dest.parents and dest != root.resolve():
                raise IntegrityError(f"Unsafe path in archive: {member}")
        zf.extractall(root)
    log_event(logger, "extract.done", root=str(root))
    return root


def ensure_dataset(cfg: Config, force_download: bool = False, overwrite: bool = False) -> RawLayout:
    """Download (if needed), verify and extract. Returns the raw layout."""
    layout = raw_layout(cfg)
    if force_download or not layout.archive.exists():
        download_archive(cfg.dataset.source_url, layout.archive)
    verify_archive(layout.archive, cfg.dataset.archive_sha256)
    extract_archive(layout.archive, layout.root, overwrite=overwrite)
    for sub in (layout.train_dir, layout.test_dir):
        if not sub.is_dir():
            raise IntegrityError(f"Expected directory missing after extraction: {sub}")
    return layout
