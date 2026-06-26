"""Respiration rate from chest/shoulder motion (ADR-07 rev 2, TECH §6.3).

Indicative signal only (no respiratory belt → no accuracy/MAE claim). The band-pass
low edge is widened to 0.08 Hz so a 6-brpm (0.1 Hz) paced block is not attenuated at
the filter edge (review item sM1).

Algorithm-review §3 adds a **chest-ROI dense optical-flow (Farnebäck)** estimator: in the
literature this lands ~0.5–1 brpm MAE seated vs ~1.6 brpm for shoulder-landmark+FFT, at the
same MediaPipe front-end cost. It is used as an additional primary that wins by SQI when the
camera actually sees chest motion; the shoulder-landmark signal and the rPPG-derived RR stay
as cross-checks/fallbacks.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

RESP_BAND_HZ = (0.08, 0.5)  # 4.8–30 brpm


def resp_band_fraction(x: np.ndarray, fps: float) -> float:
    """Fraction of spectral power inside the respiration band — a 0..1 motion-quality SQI."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if len(x) < 8:
        return 0.0
    f, p = signal.periodogram(x, fs=fps)
    band = (f >= RESP_BAND_HZ[0]) & (f <= RESP_BAND_HZ[1])
    total = float(p.sum())
    return float(p[band].sum() / total) if total > 0 else 0.0


def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    f = np.asarray(frame)
    if f.ndim == 3:
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f.astype(np.uint8) if f.dtype != np.uint8 else f


def farneback_vertical_motion(prev_gray: np.ndarray, gray: np.ndarray, roi=None,
                              max_width: int = 160) -> float:
    """Mean vertical optical flow (dy, pixels/frame) over an ROI via Farnebäck dense flow.

    Farnebäck is the most consistent classical flow variant for respiration (Srestha & Kim
    2026; Maxwell 2023). ``roi`` is ``(x0, y0, x1, y1)`` in pixels, or None for the full frame.

    The ROI is downscaled to ``max_width`` before the (expensive) dense flow — we only need the
    breathing *frequency*, so absolute pixel scale is irrelevant, and this keeps the per-frame
    cost ~sub-millisecond so the live pipeline stays at full frame rate (perf fix).
    """
    import cv2

    p, g = prev_gray, gray
    if roi is not None:
        x0, y0, x1, y1 = roi
        p, g = p[y0:y1, x0:x1], g[y0:y1, x0:x1]
    if p.size == 0 or g.size == 0:
        return 0.0
    w = p.shape[1]
    if max_width and w > max_width:
        scale = max_width / float(w)
        new_size = (max_width, max(int(round(p.shape[0] * scale)), 1))
        p = cv2.resize(p, new_size, interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, new_size, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(p, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(flow[..., 1].mean())  # mean vertical component (downscaled px/frame)


def chest_roi_box(width: int, height: int) -> tuple[int, int, int, int]:
    """A default frontal chest/upper-torso ROI box (central-lower third of the frame)."""
    x0 = int(width * 0.30); x1 = int(width * 0.70)
    y0 = int(height * 0.55); y1 = int(height * 0.95)
    return x0, y0, x1, y1


class ChestFlowRespiration:
    """Streaming chest-ROI optical-flow respiration estimator (review §3).

    Accumulates per-frame vertical flow into a displacement series; ``estimate`` band-passes
    it and returns ``(rr_brpm, sqi)`` once enough samples are buffered, or ``None``.
    """

    def __init__(self, fps: float, window_s: float = 20.0, min_window_s: float = 15.0):
        self.fps = fps
        self.window_s = window_s
        self.min_window_s = min_window_s
        self._prev: np.ndarray | None = None
        self._ts: list[float] = []
        self._disp: list[float] = []
        self._cum = 0.0

    def update(self, ts: float, frame: np.ndarray, roi=None) -> None:
        gray = _to_gray(frame)
        if roi is None:
            h, w = gray.shape[:2]
            roi = chest_roi_box(w, h)
        if self._prev is not None:
            self._cum += farneback_vertical_motion(self._prev, gray, roi)  # integrate velocity
            self._ts.append(ts)
            self._disp.append(self._cum)
        self._prev = gray
        # trim to window
        if self._ts:
            cutoff = self._ts[-1] - self.window_s
            keep = sum(1 for t in self._ts if t < cutoff)
            if keep:
                self._ts = self._ts[keep:]
                self._disp = self._disp[keep:]

    def estimate(self):
        n = len(self._disp)
        # time-span gate (robust to live throughput below nominal fps)
        if n < 8 or (self._ts[-1] - self._ts[0]) < self.min_window_s:
            return None
        disp = np.asarray(self._disp, dtype=float)
        rr = respiration_rate(disp, self.fps)
        sqi = resp_band_fraction(disp, self.fps)
        return float(rr), float(sqi)


def rppg_derived_rr(bvp: np.ndarray, fps: float) -> float:
    """Secondary respiration estimate from the rPPG amplitude envelope (E1.AC2).

    Respiration amplitude-modulates the pulse; the Hilbert envelope's dominant frequency in
    the respiration band gives a (weaker, secondary) RR cross-check.
    """
    from scipy.signal import hilbert

    x = np.asarray(bvp, dtype=float)
    env = np.abs(hilbert(x - x.mean()))
    return respiration_rate(env, fps)


def respiration_rate(displacement: np.ndarray, fps: float, nfft: int = 4096) -> float:
    """Estimate respiration rate (breaths/min) from a chest-displacement series.

    band-pass [0.08, 0.5] Hz -> periodogram peak -> brpm.
    """
    x = np.asarray(displacement, dtype=float)
    x = x - x.mean()
    if len(x) < 8:
        raise ValueError("need at least 8 samples")
    nyq = fps / 2.0
    lo = max(RESP_BAND_HZ[0] / nyq, 1e-4)
    hi = min(RESP_BAND_HZ[1] / nyq, 0.999)
    b, a = signal.butter(2, [lo, hi], btype="band")
    xf = signal.filtfilt(b, a, x)
    nfft = max(nfft, len(xf))
    freqs, psd = signal.periodogram(xf, fs=fps, nfft=nfft, detrend="constant")
    mask = (freqs >= RESP_BAND_HZ[0]) & (freqs <= RESP_BAND_HZ[1])
    peak = freqs[mask][int(np.argmax(psd[mask]))]
    return float(peak * 60.0)
