"""Debug camera overlay drawing + JPEG/base64 encoding."""

import base64

import numpy as np

from argus.backbone.face import SyntheticFaceBackbone
from argus.contracts import FrameContext
from argus.viz.overlay import draw_debug, encode_jpeg_b64


def test_draw_debug_and_encode():
    frame = np.full((120, 160, 3), 60, np.uint8)
    face = SyntheticFaceBackbone().process(frame, ts=0.1)
    ctx = FrameContext(frame=frame, ts=0.1, frame_id=0, face=face, pose=None)
    img = draw_debug(frame, ctx, hud={"hr": 72.0, "resp": 15.0, "emotion": "happiness", "gaze": "center"})
    assert img.shape == frame.shape and img.dtype == np.uint8
    # overlay actually drew something (pixels differ from the flat input)
    assert not np.array_equal(img, frame)
    b64 = encode_jpeg_b64(img, max_width=120)
    assert isinstance(b64, str) and len(b64) > 100
    assert base64.b64decode(b64)[:2] == b"\xff\xd8"  # valid JPEG SOI marker


def test_draw_debug_no_face():
    frame = np.zeros((80, 80, 3), np.uint8)
    ctx = FrameContext(frame=frame, ts=0.1, frame_id=0, face=None, pose=None)
    img = draw_debug(frame, ctx, hud={})  # must not crash with no face
    assert img.shape == frame.shape
