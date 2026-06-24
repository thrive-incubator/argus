"""Runtime policy & dependency manifest (J1.AC3, NFR-2/3/4).

The docs require CPython 3.11, numpy<2 (MediaPipe constraint), CPU-only torch, macOS/Linux
(Windows unsupported). ``check_runtime`` reports compliance of the *current* interpreter;
``PINNED_DEPS`` is the pinned-version manifest backing reproducibility (NFR-4).
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

REQUIRED_PYTHON = (3, 11)
SUPPORTED_PLATFORMS = ("Darwin", "Linux")  # macOS, Linux; Windows unsupported (NFR-3)

# Pinned dependency manifest (the versions the full hardware build targets).
PINNED_DEPS: dict[str, str] = {
    "python": "3.11.*",
    "numpy": "<2",
    "scipy": ">=1.11",
    "opencv-python-headless": ">=4.8",
    "mediapipe": ">=0.10",
    "neurokit2": ">=0.2",
    "pylsl": ">=1.16",
    "pyxdf": ">=1.16",
    "bleak": ">=0.21",
    "onnxruntime": ">=1.16",
    "torch": "cpu",
}


@dataclass(frozen=True)
class RuntimeReport:
    python_ok: bool
    platform_ok: bool
    numpy_lt2: bool
    platform_name: str
    python_version: tuple[int, int]

    @property
    def compliant(self) -> bool:
        return self.python_ok and self.platform_ok and self.numpy_lt2


def check_runtime() -> RuntimeReport:
    py = sys.version_info[:2]
    plat = platform.system()
    try:
        import numpy

        numpy_lt2 = int(numpy.__version__.split(".")[0]) < 2
    except Exception:  # pragma: no cover
        numpy_lt2 = False
    return RuntimeReport(
        python_ok=py == REQUIRED_PYTHON,
        platform_ok=plat in SUPPORTED_PLATFORMS,
        numpy_lt2=numpy_lt2,
        platform_name=plat,
        python_version=py,
    )


def windows_supported() -> bool:
    return False  # NFR-3
