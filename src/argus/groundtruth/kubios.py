"""RR-interval artifact correction (C1.AC4) via NeuroKit2 Kubios.

Wraps ``neurokit2.signal_fixpeaks(method="kubios")`` so ectopic/missed/extra beats are
corrected before any HRV statistic.
"""

from __future__ import annotations

import numpy as np


def kubios_correct(rr_ms, sampling_rate: float = 1000.0):
    """Correct an RR-interval series (ms) with the Kubios method.

    Returns the corrected RR series (ms). Falls back to the input if NeuroKit2 cannot
    process a very short series.
    """
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 4:
        return rr
    import neurokit2 as nk

    # Reconstruct R-peak sample indices from cumulative RR.
    peak_times = np.cumsum(np.r_[0.0, rr]) / 1000.0
    peaks = np.round(peak_times * sampling_rate).astype(int)

    out = nk.signal_fixpeaks(peaks, sampling_rate=sampling_rate, method="kubios", iterative=True)
    peaks_clean = out[1] if isinstance(out, tuple) else out
    peaks_clean = np.asarray(peaks_clean, dtype=float)
    if peaks_clean.size < 2:
        return rr
    return np.diff(peaks_clean) / sampling_rate * 1000.0
