"""Model-asset manifest + fetch with checksum verification (NFR-5 ``argus fetch-models``).

First-run downloads are permitted, then cached/pinned for offline operation. The real
downloader uses urllib; tests inject a fake downloader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    url: str
    sha256: str
    license: str


MODEL_MANIFEST: list[ModelAsset] = [
    ModelAsset("face_landmarker", "face_landmarker.task",
               "https://example/face_landmarker.task", "", "Apache-2.0"),
    ModelAsset("pose_landmarker", "pose_landmarker.task",
               "https://example/pose_landmarker.task", "", "Apache-2.0"),
    ModelAsset("hsemotion", "enet_b0_8_va_mtl.onnx",
               "https://example/hsemotion.onnx", "", "Apache-2.0"),
    ModelAsset("l2cs", "l2cs.onnx", "https://example/l2cs.onnx", "", "MIT"),
    ModelAsset("libreface", "libreface_au.onnx",
               "https://example/libreface.onnx", "", "research"),
]


def _urllib_download(url: str) -> bytes:  # pragma: no cover - network
    import urllib.request

    with urllib.request.urlopen(url) as r:
        return r.read()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_models(dest: str, manifest: list[ModelAsset] | None = None,
                 downloader: Callable[[str], bytes] = _urllib_download) -> list[Path]:
    """Download each asset, verify its checksum (when pinned), cache under ``dest``."""
    manifest = manifest if manifest is not None else MODEL_MANIFEST
    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for asset in manifest:
        data = downloader(asset.url)
        if asset.sha256 and sha256_hex(data) != asset.sha256:
            raise ValueError(f"checksum mismatch for {asset.name}")
        path = out_dir / asset.filename
        path.write_bytes(data)
        paths.append(path)
    return paths
