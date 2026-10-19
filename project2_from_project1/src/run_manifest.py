"""Shared, text-free provenance helpers for experimental run manifests."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

_TRACKED_PACKAGES = ("numpy", "scipy", "scikit-learn", "torch")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def git_provenance(directory: str | Path | None = None) -> dict[str, Any]:
    """Return the checked-out commit and whether tracked files differ from it."""
    cwd = Path(directory) if directory is not None else Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def dependency_versions(packages: Iterable[str] = _TRACKED_PACKAGES) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": dict(sorted(versions.items())),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


__all__ = ["dependency_versions", "git_provenance", "sha256_file", "sha256_text"]
