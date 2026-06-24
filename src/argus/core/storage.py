"""Privacy-by-construction storage (NFR-5, J4).

Default storage = derived signals + XDF. Raw face video is written ONLY when an explicit,
off-by-default opt-in is set, stored locally and labelled.
"""

from __future__ import annotations

import json
from pathlib import Path


class StorageManager:
    def __init__(self, root: str, allow_raw_video: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_raw_video = allow_raw_video  # off by default (J4.AC1)

    def write_derived(self, name: str, payload: dict) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload))
        return path

    def xdf_path(self, session: str) -> Path:
        return self.root / f"{session}.xdf"

    def write_raw_video(self, session: str, frames_writer) -> Path:
        """Write raw face video only if explicitly opted in; labelled (J4.AC2)."""
        if not self.allow_raw_video:
            raise PermissionError(
                "raw video storage is off by default (NFR-5); pass allow_raw_video=True to opt in"
            )
        path = self.root / f"{session}.RAWFACE.local-only.mp4"
        frames_writer(str(path))  # the actual encode is the caller's (device) concern
        # label sidecar so the artifact is never mistaken for shareable
        (self.root / f"{session}.RAWFACE.LABEL.txt").write_text(
            "RAW FACE VIDEO — local only, never commit, contains PII"
        )
        return path
