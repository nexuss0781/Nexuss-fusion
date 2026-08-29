"""Immutable dataset/artifact manifest with content addressing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "nexuss-fusion.manifest.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    manifest = {**manifest, "schema_version": MANIFEST_SCHEMA}
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    Path(path).with_suffix(Path(path).suffix + ".sha256").write_text(sha256_bytes(json.dumps(manifest, sort_keys=True).encode()) + "\n")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def verify_manifest(path: str | Path) -> str | None:
    data = Path(path).read_text()
    digest_file = Path(path).with_suffix(Path(path).suffix + ".sha256")
    if not digest_file.exists():
        return "missing sidecar sha256"
    digest = digest_file.read_text().strip()
    computed = sha256_bytes(data.encode())
    return None if computed == digest else "sha256 mismatch"