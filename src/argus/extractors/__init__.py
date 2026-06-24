"""Extractor plugins: each consumes a FrameContext and emits list[SignalRecord]."""

from .blink_extractor import BlinkExtractor
from .hrv_extractor import HrvExtractor
from .motion_extractor import MotionExtractor
from .respiration_extractor import RespirationExtractor
from .rppg_extractor import RppgExtractor

__all__ = [
    "RppgExtractor",
    "HrvExtractor",
    "RespirationExtractor",
    "BlinkExtractor",
    "MotionExtractor",
]
