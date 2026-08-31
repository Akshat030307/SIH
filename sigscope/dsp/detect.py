"""Stage 3 -- Detect (CLAUDE.md §4.3).

Spectrogram (§4.1) -> per-bin noise floor (§4.2) -> threshold -> binary closing then
opening -> connected components -> size rejection -> bounding boxes -> time-gap merge.
Every constant lives in :class:`DetectorConfig`.
"""

from __future__ import annotations

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
    noise_percentile: float = 25.0
    # §4.3 detection
    threshold_db: float = 8.0
    closing_size: tuple[int, int] = (3, 3)  # (freq bins, time cols) -- bridge a fade
    opening_size: tuple[int, int] = (2, 2)  # kill single-pixel specks
    min_pixels: int = 20
    min_freq_bins: int = 2
    merge_gap_hops: int = 3  # stitch a bursty transmission back into one detection
    # §4.3 special case: continuous / wideband
    wideband_time_frac: float = 0.90
    wideband_freq_frac: float = 0.60
    # Stage 2 pre-conditioning (DC removal); off if the caller already conditioned
    precondition: bool = True


@dataclass
class DetectionResult:
    bursts: list[Burst]
    spectrogram: Spectrogram
    noise: NoiseEstimate
    method: str = "stft-percentile-morphology"
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Box:
    f0: int
    f1: int  # inclusive bin indices
    t0: int
    t1: int  # inclusive column indices
    npix: int


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


def _time_gap(a: _Box, b: _Box) -> int:
    return max(0, a.t0 - b.t1, b.t0 - a.t1)


def _merge_boxes(boxes: list[_Box], gap_hops: int) -> list[_Box]:
    """§4.3 step 7: merge boxes overlapping in frequency and < ``gap_hops`` apart in time."""
    if not boxes:
        return []
    uf = _UnionFind(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _freq_overlap(boxes[i], boxes[j]) and _time_gap(boxes[i], boxes[j]) < gap_hops:
                uf.union(i, j)
    groups: dict[int, _Box] = {}
    for i, box in enumerate(boxes):
        r = uf.find(i)
        g = groups.get(r)
        if g is None:
            groups[r] = _Box(box.f0, box.f1, box.t0, box.t1, box.npix)
        else:
            groups[r] = _Box(
                min(g.f0, box.f0), max(g.f1, box.f1),
                min(g.t0, box.t0), max(g.t1, box.t1),
                g.npix + box.npix,
            )
    return list(groups.values())


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
    noise = estimate_noise(spec.S_db, percentile=cfg.noise_percentile)
    n_freq, n_time = spec.S_db.shape

    mask = spec.S_db > (noise.per_bin_db[:, None] + cfg.threshold_db)
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
        boxes.append(_Box(fsl.start, fsl.stop - 1, tsl.start, tsl.stop - 1, npix))

    merged = _merge_boxes(boxes, cfg.merge_gap_hops)

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
    if not bursts:
        warnings.append(
            f"no signals found above {cfg.threshold_db:.0f} dB over the noise floor "
            f"({noise.floor_db:.1f} dBFS)"
        )
    return DetectionResult(bursts=bursts, spectrogram=spec, noise=noise, warnings=warnings)
