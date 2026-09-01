"""Stage 3 -- Detect (CLAUDE.md §4.3).

Spectrogram (§4.1) -> per-bin noise floor (§4.2) -> threshold -> binary closing then
opening -> connected components -> size rejection -> bounding boxes -> time-gap merge.
Every constant lives in :class:`DetectorConfig`.

**The threshold is derived, not fixed.** §4.3 suggests ``threshold_db = 8`` over the §4.2
per-bin floor. That floor is the 25th percentile of the magnitude over time, and for
complex Gaussian noise the STFT power is exponentially distributed, so the 25th percentile
sits 5.41 dB *below* the mean. An 8 dB margin over it therefore passes every cell above
2.59 dB over the mean -- which is ``exp(-10**0.259) = 16%`` of all cells. On a 2048 x 2000
spectrogram that is 667,000 noise cells, and enough of them cluster past ``min_pixels`` to
produce thousands of phantom detections: measured, 9,245 of them on one second of pure
noise, against the zero §9 C requires.

:func:`cfar_threshold_db` instead solves for the margin that gives a target per-cell
false-alarm probability, the standard CFAR construction. ``false_alarm_rate=1e-5`` puts the
threshold at 16.0 dB over the same floor and leaves ~43 isolated cells on that spectrogram,
which the morphology then clears. Set ``threshold_db`` explicitly to override the
derivation and get §4.3's literal behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from sigscope.dsp.condition import condition
from sigscope.dsp.noise import NoiseEstimate, estimate_noise
from sigscope.dsp.spectrogram import Spectrogram, compute_spectrogram
from sigscope.types import Burst


@dataclass(frozen=True)
class DetectorConfig:
    """Every detector constant, with the CLAUDE.md §4.1-§4.3 defaults."""

    # §4.1 spectrogram
    target_time_cols: int = 2000
    nfft_min: int = 256
    nfft_max: int = 8192
    overlap_frac: float = 0.75
    window: str = "hann"
    # §4.2 noise floor
    noise_percentile: float = 25.0  # over time, within each bin
    noise_bin_percentile: float = 25.0  # across bins, for the scalar floor
    # a bin whose floor sits far above the global median holds a persistent signal, not
    # extra noise; without this cap a continuous carrier raises its own threshold above
    # itself and becomes invisible. 6 dB still allows genuine filter roll-off through.
    max_bin_excess_db: float = 6.0
    # §4.3 detection
    # target per-cell false-alarm probability; the threshold is derived from it unless
    # threshold_db is set explicitly (see the module docstring and cfar_threshold_db)
    false_alarm_rate: float = 1e-2
    threshold_db: float | None = None
    closing_size: tuple[int, int] = (3, 3)  # (freq bins, time cols) -- bridge a fade
    opening_size: tuple[int, int] = (2, 2)  # kill single-pixel specks
    min_pixels: int = 40
    min_freq_bins: int = 2
    min_time_cols: int = 2  # one column is a click, not a transmission
    merge_gap_hops: int = 3  # stitch a bursty transmission back into one detection
    merge_gap_bins: int = 3  # ...and the same along frequency (see _should_merge)
    # spectral-skirt suppression (see _suppress_leakage)
    leakage_margin_db: float = 20.0
    leakage_span: float = 2.0
    # §4.3 special case: continuous / wideband
    wideband_time_frac: float = 0.90
    wideband_freq_frac: float = 0.60
    # Stage 2 pre-conditioning (DC removal); off if the caller already conditioned
    precondition: bool = True


def cfar_threshold_db(false_alarm_rate: float, noise_percentile: float = 25.0) -> float:
    """Detection margin over the §4.2 floor for a target per-cell false-alarm rate.

    For complex Gaussian noise the STFT power in one cell is exponentially distributed.
    Writing ``q`` for the reference quantile in units of the mean
    (``q = -ln(1 - P/100)``), a cell exceeds ``m`` dB over that quantile with probability
    ``exp(-10**(m/10) * q)``. Solving for ``m`` at the requested probability gives::

        m = 10 * log10(-ln(p_fa) / q)

    which is 16.0 dB at ``p_fa = 1e-5`` against §4.3's suggested 8 dB. This is the standard
    cell-averaging CFAR argument, and it is what makes "pure noise gives zero detections"
    (§9 C) a property of the design rather than a hope.

    The STFT overlaps by 75%, so neighbouring columns are correlated and the true number of
    independent cells is lower than the nominal count -- the real false-alarm rate comes out
    below the target, which is the safe direction.
    """
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must be in (0, 1)")
    if not 0.0 < noise_percentile < 100.0:
        raise ValueError("noise_percentile must be in (0, 100)")
    quantile = -math.log(1.0 - noise_percentile / 100.0)
    return 10.0 * math.log10(-math.log(false_alarm_rate) / quantile)


@dataclass
class DetectionResult:
    bursts: list[Burst]
    spectrogram: Spectrogram
    noise: NoiseEstimate
    method: str = "stft-cfar-morphology"
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Box:
    f0: int
    f1: int  # inclusive bin indices
    t0: int
    t1: int  # inclusive column indices
    npix: int
    peak_db: float = -np.inf


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _freq_overlap(a: _Box, b: _Box) -> bool:
    return a.f0 <= b.f1 and b.f0 <= a.f1


def _time_overlap(a: _Box, b: _Box) -> bool:
    return a.t0 <= b.t1 and b.t0 <= a.t1


def _time_gap(a: _Box, b: _Box) -> int:
    return max(0, a.t0 - b.t1, b.t0 - a.t1)


def _freq_gap(a: _Box, b: _Box) -> int:
    return max(0, a.f0 - b.f1, b.f0 - a.f1)


def _should_merge(a: _Box, b: _Box, gap_hops: int, gap_bins: int) -> bool:
    """Two boxes are the same transmission if they touch along either axis.

    §4.3 step 7 only merges along time -- "boxes overlapping in frequency and separated in
    time by fewer than 3 STFT hops" -- which stitches a bursty transmission back together
    but leaves the frequency-axis twin unhandled. A signal whose spectrum dips in the middle
    (an FSK pair, or any modulation with a spectral null at its centre) splits into two
    side-by-side boxes at the same instant and gets reported as two signals. Measured on a
    known-truth 2-FSK burst, that is exactly what happened. The frequency clause below is
    the same rule with the axes swapped.
    """
    if _freq_overlap(a, b) and _time_gap(a, b) < gap_hops:
        return True
    return bool(_time_overlap(a, b) and _freq_gap(a, b) < gap_bins)


def _merge_boxes(boxes: list[_Box], gap_hops: int, gap_bins: int) -> list[_Box]:
    """§4.3 step 7, applied along both axes -- see :func:`_should_merge`."""
    if not boxes:
        return []
    uf = _UnionFind(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _should_merge(boxes[i], boxes[j], gap_hops, gap_bins):
                uf.union(i, j)
    groups: dict[int, _Box] = {}
    for i, box in enumerate(boxes):
        r = uf.find(i)
        g = groups.get(r)
        if g is None:
            groups[r] = _Box(box.f0, box.f1, box.t0, box.t1, box.npix, box.peak_db)
        else:
            groups[r] = _Box(
                min(g.f0, box.f0), max(g.f1, box.f1),
                min(g.t0, box.t0), max(g.t1, box.t1),
                g.npix + box.npix,
                max(g.peak_db, box.peak_db),
            )
    return list(groups.values())


def _suppress_leakage(
    boxes: list[_Box], margin_db: float, span: float
) -> tuple[list[_Box], int]:
    """Drop boxes that are the spectral skirt of a much stronger neighbour.

    A strong emitter does not stop at the edge of its own bandwidth. Window sidelobes and
    real transmitter splatter put energy either side of it that clears the detection
    threshold on its own, and each patch is then reported as a separate signal -- on a
    known-truth scene with four signals at 25 dB SNR, ten of the fifteen boxes were skirts
    of one strong FSK burst.

    A box is discarded when another box overlapping it in time is more than ``margin_db``
    stronger at its peak and sits within ``span`` of its own width in frequency. Both
    conditions are needed: the level difference is what distinguishes a skirt from a
    genuine weak neighbour, and the proximity is what stops a strong signal suppressing an
    unrelated weak one at the far end of the band.
    """
    keep: list[_Box] = []
    dropped = 0
    for b in boxes:
        b_width = max(1, b.f1 - b.f0 + 1)
        shadowed = False
        for a in boxes:
            if a is b:
                continue
            if a.peak_db - b.peak_db <= margin_db:
                continue
            if not _time_overlap(a, b):
                continue
            reach = span * max(1, a.f1 - a.f0 + 1)
            if _freq_gap(a, b) <= reach and b_width <= a.f1 - a.f0 + 1:
                shadowed = True
                break
        if shadowed:
            dropped += 1
        else:
            keep.append(b)
    return keep, dropped


def detect_bursts(
    x: np.ndarray, fs: float, cfg: DetectorConfig | None = None
) -> DetectionResult:
    """Find every separate signal as a time-frequency rectangle (CLAUDE.md §4.3)."""
    cfg = cfg or DetectorConfig()
    x = np.ascontiguousarray(x, dtype=np.complex64)
    if cfg.precondition:
        x, _ = condition(x)

    spec = compute_spectrogram(
        x, fs,
        overlap_frac=cfg.overlap_frac, window=cfg.window,
        target_cols=cfg.target_time_cols, nfft_lo=cfg.nfft_min, nfft_hi=cfg.nfft_max,
    )
    noise = estimate_noise(
        spec.S_db,
        percentile=cfg.noise_percentile,
        bin_percentile=cfg.noise_bin_percentile,
    )
    n_freq, n_time = spec.S_db.shape

    threshold_db = (
        cfg.threshold_db
        if cfg.threshold_db is not None
        else cfar_threshold_db(cfg.false_alarm_rate, cfg.noise_percentile)
    )
    # cap each bin's floor so a persistent carrier cannot hide behind its own threshold
    floor_db = np.minimum(noise.per_bin_db, noise.floor_db + cfg.max_bin_excess_db)

    mask = spec.S_db > (floor_db[:, None] + threshold_db)
    mask = ndimage.binary_closing(mask, structure=np.ones(cfg.closing_size, dtype=bool))
    mask = ndimage.binary_opening(mask, structure=np.ones(cfg.opening_size, dtype=bool))

    labelled, n_labels = ndimage.label(mask)
    boxes: list[_Box] = []
    for i, sl in enumerate(ndimage.find_objects(labelled), start=1):
        if sl is None:
            continue
        fsl, tsl = sl
        npix = int(np.count_nonzero(labelled[fsl, tsl] == i))
        if npix < cfg.min_pixels:
            continue
        if (fsl.stop - fsl.start) < cfg.min_freq_bins:
            continue
        if (tsl.stop - tsl.start) < cfg.min_time_cols:
            continue
        region = spec.S_db[fsl, tsl]
        peak_db = float(region.max()) if region.size else -np.inf
        boxes.append(_Box(fsl.start, fsl.stop - 1, tsl.start, tsl.stop - 1, npix, peak_db))

    merged = _merge_boxes(boxes, cfg.merge_gap_hops, cfg.merge_gap_bins)
    merged, n_leak = _suppress_leakage(merged, cfg.leakage_margin_db, cfg.leakage_span)

    bursts: list[Burst] = []
    for b in merged:
        wideband = (
            (b.t1 - b.t0 + 1) / n_time >= cfg.wideband_time_frac
            and (b.f1 - b.f0 + 1) / n_freq >= cfg.wideband_freq_frac
        )
        bursts.append(
            Burst(
                t0=float(spec.t[b.t0]),
                t1=float(spec.t[b.t1]),
                f_lo=float(spec.f[b.f0]),
                f_hi=float(spec.f[b.f1]),
                wideband=wideband,
            )
        )
    bursts.sort(key=lambda bu: (bu.t0, bu.f_lo))

    warnings: list[str] = []
    if n_leak:
        warnings.append(
            f"{n_leak} weak detection(s) adjacent to a much stronger signal were "
            "suppressed as spectral skirt"
        )
    if not bursts:
        warnings.append(
            f"no signals found above {threshold_db:.1f} dB over the noise floor "
            f"({noise.floor_db:.1f} dBFS). Try lowering the detection threshold."
        )
    return DetectionResult(bursts=bursts, spectrogram=spec, noise=noise, warnings=warnings)
