"""Stage 4 -- Isolate + measure: the classical parameter estimators (CLAUDE.md §4.4-§4.13).

Every public function here is pure (§2 "Coding rules"): numpy array in, dataclass out, no
I/O, no global state, ``fs`` always explicit, complex baseband always ``np.complex64``.
Every measurement returns a value **and** a confidence in 0..1 **and** the method name --
that is :class:`sigscope.types.Estimate`. Measurements that produce several related
numbers (bandwidth, FSK tones, OFDM, chirp) return a small dataclass whose fields are
themselves ``Estimate`` objects, so the rule holds all the way down.

Nothing here ever fabricates a number. When an estimator cannot get a reliable answer it
returns ``Estimate(value=None, ...)`` with the reason in ``notes`` (§2 "Hard rules").

Order matters: :func:`isolate_burst` (§4.4) runs first and everything else runs on the
isolated, mixed-down, decimated burst. That is the whole trick -- estimators that fail on
a crowded band work fine on one clean burst. Frequencies measured on the isolated burst
are offsets from ``IsolatedBurst.f_shift_hz``; add it back to get absolute Hz.

Three places where this file deliberately departs from a literal reading of §4, each
validated against known-truth signals and each documented at its call site:

* §4.7 SNR takes its noise level from the **mean** linear power of the out-of-box bins,
  not from the §4.2 25th-percentile floor. The percentile floor is a *detection
  threshold*; as a *power* it sits 5.4 dB low (it is a low quantile of a chi-squared
  magnitude), which fed straight through into a 5.4 dB SNR bias.
* §4.11 OFDM keys detection on the correlation peak's ratio to its own background, not
  on the literal ``R > 0.3``. With the global normalisation §4.11 specifies, a cyclic
  prefix of CP/(N+CP) = 1/5 tops out at R = 0.2, so a fixed 0.3 threshold can never fire.
  R is still reported, and CP length is recovered from its amplitude.
* §4.8 method (c) abstains far more often than §4.8 implies -- see its docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, firwin, oaconvolve, welch
from scipy.signal.windows import blackmanharris

from sigscope.dsp.spectrogram import Spectrogram, compute_spectrogram
from sigscope.types import Bandwidth, Burst, Estimate

__all__ = [
    "EstimatorConfig",
    "IsolatedBurst",
    "CenterFreq",
    "BandwidthResult",
    "SnrResult",
    "SymbolRateResult",
    "MthPowerResult",
    "FskParams",
    "OfdmParams",
    "ChirpParams",
    "CwKeying",
    "isolate_burst",
    "estimate_center_freq",
    "estimate_bandwidth",
    "estimate_snr",
    "symbol_rate_cyclostationary",
    "symbol_rate_freq_transitions",
    "symbol_rate_envelope_autocorr",
    "estimate_symbol_rate",
    "estimate_psk_order",
    "estimate_fsk_params",
    "estimate_ofdm_params",
    "estimate_chirp",
    "estimate_am_depth",
    "estimate_fm_deviation",
    "estimate_spectral_asymmetry",
    "estimate_cw_keying",
    "instantaneous_frequency",
]

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------------------
# Configuration -- every constant in one place (§4.3 "All constants live in a dataclass")
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimatorConfig:
    """Every estimator constant, with the CLAUDE.md §4.4-§4.13 defaults."""

    # §4.4 isolation
    filter_taps: int = 101
    filter_window: str = "hamming"
    cutoff_frac: float = 0.6  # cutoff = 0.6 * box bandwidth
    decimate_headroom: float = 2.5  # D = floor(fs / (2.5 * bw))
    min_isolated_samples: int = 64

    # §4.5 centre frequency
    psd_nperseg: int = 1024
    centroid_peak_prominence_db: float = 6.0

    # §4.6 bandwidth
    obw_fraction: float = 0.99
    psd_noise_percentile: float = 10.0  # floor subtracted before the OBW walk
    subtract_psd_noise: bool = True

    # §4.7 SNR
    snr_guard_frac: float = 0.5  # guard band each side of the box, in box widths

    # §4.8 symbol rate
    max_analysis_samples: int = 1 << 18
    line_zero_pad: int = 8
    max_fft: int = 1 << 22
    rate_lo_frac: float = 1.0 / 1000.0  # search [fs_b/1000, fs_b/2]
    rate_hi_frac: float = 0.5
    line_conf_slope: float = 0.398  # logistic: 12 dB -> ~0.9, 3 dB -> ~0.2
    line_conf_midpoint: float = 6.48
    harmonic_divisors: tuple[int, ...] = (4, 3, 2)
    harmonic_line_frac: float = 0.75  # subharmonic accepted at >= 75% of the peak height
    harmonic_min_line_db: float = 10.0
    harmonic_search_frac: float = 0.02  # +/-2% window when probing a candidate line
    reconcile_tol: float = 0.05  # "agree within 5%"
    reconcile_conf: float = 0.85  # "confidence >= 0.85" on agreement
    lone_method_conf_cap: float = 0.60
    disagree_conf_cap: float = 0.40
    autocorr_min_peak: float = 0.25
    autocorr_min_ratio: float = 5.0
    autocorr_conf_cap: float = 0.60

    # §4.9 M-th power
    mth_powers: tuple[int, ...] = (1, 2, 4, 8)
    mth_orders: tuple[int, ...] = (2, 4, 8)
    mth_zero_pad: int = 4
    mth_min_sharpness_db: float = 10.0

    # §4.10 FSK
    fsk_hist_bins: int = 200
    fsk_clip_percentile: float = 0.5
    fsk_smooth: int = 5
    fsk_peak_prominence_frac: float = 0.15
    fsk_max_tones: int = 8

    # §4.11 OFDM
    ofdm_max_lag_frac: float = 0.25  # search lags up to 25% of the analysed length
    ofdm_max_lag: int = 4096
    ofdm_min_lag: int = 8
    ofdm_peak_ratio: float = 8.0
    ofdm_min_r: float = 0.03
    # R converges on CP/(N+CP), so it is bounded by the largest sane cyclic prefix. Real
    # systems use 1/4 (802.11, DVB-T) down to about 1/14 (LTE); anything above ~0.35 would
    # mean a prefix longer than half the useful symbol, which no system transmits and which
    # in practice means the peak came from something else -- a constant-modulus FSK burst
    # reaches R = 0.80 on its own tone periodicity.
    ofdm_max_r: float = 0.35

    # §4.12 chirp
    chirp_ridge_snr_db: float = 6.0
    chirp_min_cols: int = 8
    # iterative outlier rejection on the ridge (see estimate_chirp)
    chirp_fit_rounds: int = 3
    chirp_inlier_sigma: float = 2.5
    chirp_r2: float = 0.90
    chirp_min_sweep_frac: float = 0.02  # sweep must cover >2% of the band to count
    chirp_quadratic_margin: float = 0.02  # deg-2 must beat deg-1 R^2 by this to be "nonlinear"

    # §4.13 analogue
    am_percentile: float = 1.0  # depth from the 1st/99th percentile, not min/max
    fm_deviation_percentile: float = 99.0
    ssb_asymmetry_db: float = 10.0
    cw_smooth_frac: float = 0.001  # envelope smoother, as a fraction of the burst
    cw_min_marks: int = 6
    cw_dash_dot_lo: float = 2.0
    cw_dash_dot_hi: float = 4.5


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def _as_c64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.complex64)


def _unit_power(y: np.ndarray) -> np.ndarray:
    """Normalise to unit average power; returns unchanged if the input is all zeros."""
    p = float(np.mean(np.abs(y) ** 2))
    return y if p <= 0.0 else (y / math.sqrt(p)).astype(y.dtype)


def _logistic(value: float, slope: float, midpoint: float) -> float:
    """Squash a dB margin into 0..1. Defaults calibrated in §4.8: 12 dB -> 0.9, 3 dB -> 0.2."""
    return float(1.0 / (1.0 + math.exp(-slope * (value - midpoint))))


def _parabolic_offset(y0: float, y1: float, y2: float) -> float:
    """Sub-bin peak offset from three samples around a maximum (§4.5).

    ``delta = 0.5*(y0 - y2) / (y0 - 2*y1 + y2)``, clamped to +/-1 bin so a flat or
    inverted triple cannot throw the estimate somewhere absurd.
    """
    den = y0 - 2.0 * y1 + y2
    if den == 0.0 or not np.isfinite(den):
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))


def _refine_peak(mag: np.ndarray, k: int, bin_width: float) -> float:
    """Parabolic refinement of bin ``k`` in dB space, returning the frequency offset."""
    if k <= 0 or k >= mag.size - 1:
        return 0.0
    trio = 20.0 * np.log10(mag[k - 1 : k + 2] + 1e-30)
    return _parabolic_offset(float(trio[0]), float(trio[1]), float(trio[2])) * bin_width


def _centre_slice(v: np.ndarray, limit: int) -> np.ndarray:
    """Take at most ``limit`` samples from the middle -- burst edges are the least clean."""
    if v.size <= limit:
        return v
    start = (v.size - limit) // 2
    return v[start : start + limit]


def instantaneous_frequency(y: np.ndarray, fs: float) -> np.ndarray:
    """Instantaneous frequency in Hz (§4.8b), via the phase of ``y[n+1]*conj(y[n])``.

    Equivalent to ``diff(unwrap(angle(y)))*fs/2pi`` but immune to unwrap failures at low
    SNR, where a single mis-unwrapped sample injects a 2*pi*fs step.
    """
    y = np.asarray(y)
    if y.size < 2:
        return np.zeros(0, dtype=np.float64)
    return np.angle(y[1:] * np.conj(y[:-1])).astype(np.float64) * fs / TWO_PI


def _welch_two_sided(
    y: np.ndarray, fs: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Welch PSD, fftshifted to ascending frequency (§4.6).

    Two-sided always: the signal is complex and the negative frequencies are real
    information, not a mirror (§4.1).
    """
    nperseg = int(min(nperseg, y.size))
    if nperseg < 8:
        raise ValueError("signal too short for a PSD")
    f, p = welch(y, fs=fs, nperseg=nperseg, return_onesided=False, detrend=False)
    order = np.argsort(f)
    return f[order], p[order]


# --------------------------------------------------------------------------------------
# §4.4 -- Isolating a burst
# --------------------------------------------------------------------------------------


@dataclass
class IsolatedBurst:
    """One burst cut out, mixed to zero, filtered and decimated (CLAUDE.md §4.4).

    ``y`` is complex baseband at ``fs_b = fs / decimation``. Every frequency measured on
    ``y`` is an offset from ``f_shift_hz``; add ``f_shift_hz`` back to return to the
    capture's own frequency axis (and the capture's centre frequency on top of that to
    reach absolute RF).
    """

    y: np.ndarray
    fs_b: float
    decimation: int
    f_shift_hz: float
    n0: int
    n1: int
    box_bandwidth_hz: float
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.y.size / self.fs_b if self.fs_b else 0.0

    def to_absolute(self, f_offset_hz: float) -> float:
        """Convert a frequency measured on ``y`` back to the capture's frequency axis."""
        return self.f_shift_hz + f_offset_hz


def isolate_burst(
    x: np.ndarray,
    fs: float,
    burst: Burst,
    *,
    cfg: EstimatorConfig | None = None,
) -> IsolatedBurst:
    """Cut, mix to zero, lowpass and decimate one burst (CLAUDE.md §4.4).

    The six steps of §4.4 exactly: slice ``t0..t1``; ``f_c = (f_lo + f_hi)/2``; mix down
    by ``exp(-2j*pi*f_c*n/fs)``; ``firwin`` lowpass at ``0.6*(f_hi - f_lo)``, 101 taps,
    Hamming; filter; decimate by ``D = floor(fs / (2.5*bw))`` when ``D >= 2``.

    The filter is applied with ``oaconvolve(..., mode="same")``. For a linear-phase
    symmetric FIR that is zero-delay, so the isolated burst stays aligned with ``t0`` --
    an ``lfilter`` would slide it 50 samples late and corrupt every timing measurement.

    Raises ``ValueError`` on an empty slice; the caller (Stage 4) guards against that.
    """
    cfg = cfg or EstimatorConfig()
    x = _as_c64(x)
    if fs <= 0:
        raise ValueError("isolate_burst: fs must be positive")

    n0 = max(0, int(math.floor(burst.t0 * fs)))
    n1 = min(x.size, int(math.ceil(burst.t1 * fs)))
    if n1 <= n0:
        raise ValueError(f"isolate_burst: empty slice for burst t0={burst.t0} t1={burst.t1}")

    notes: list[str] = []
    seg = x[n0:n1]
    bw = float(burst.f_hi - burst.f_lo)
    f_c = float(burst.f_center)

    # step 3 -- mix the box centre down to 0 Hz
    n = np.arange(seg.size, dtype=np.float64)
    y = (seg * np.exp(-2j * np.pi * f_c * n / fs)).astype(np.complex64)

    # steps 4-5 -- lowpass, then decimate. A box as wide as the band needs neither.
    decimation = 1
    if bw <= 0.0:
        notes.append("burst box has zero width; skipped filtering and decimation")
    elif bw >= 0.9 * fs:
        notes.append("burst box spans the whole band; skipped filtering and decimation")
    else:
        cutoff = min(cfg.cutoff_frac * bw, 0.45 * fs)
        ntaps = int(cfg.filter_taps)
        if ntaps >= y.size:
            ntaps = max(3, (int(y.size) - 1) | 1)  # force odd, keep linear phase
            notes.append(f"burst shorter than the filter; reduced to {ntaps} taps")
        if cutoff > 0.0 and ntaps >= 3:
            h = firwin(ntaps, cutoff, fs=fs, window=cfg.filter_window)
            y = oaconvolve(y, h, mode="same").astype(np.complex64)
        d = int(math.floor(fs / (cfg.decimate_headroom * bw)))
        if d >= 2:
            # never decimate below the minimum the estimators need to say anything
            while d > 1 and y.size // d < cfg.min_isolated_samples:
                d -= 1
            if d >= 2:
                y = np.ascontiguousarray(y[::d])
                decimation = d

    if y.size < cfg.min_isolated_samples:
        notes.append(
            f"isolated burst is only {y.size} samples; estimators will abstain readily"
        )

    return IsolatedBurst(
        y=_as_c64(y),
        fs_b=fs / decimation,
        decimation=decimation,
        f_shift_hz=f_c,
        n0=n0,
        n1=n1,
        box_bandwidth_hz=bw,
        notes=notes,
    )


# --------------------------------------------------------------------------------------
# §4.5 -- Centre frequency
# --------------------------------------------------------------------------------------


@dataclass
class CenterFreq:
    """Both §4.5 centre-frequency numbers, reported side by side.

    ``centroid`` is the power-weighted centroid -- the right answer for a wide modulated
    signal. ``peak`` is the parabolically-refined spectral peak -- the right answer for a
    carrier or a CW tone. Analysts want both; they disagree in informative ways.
    """

    centroid: Estimate
    peak: Estimate
    n_humps: int = 1


def estimate_center_freq(
    x: np.ndarray,
    fs: float,
    burst: Burst | None = None,
    *,
    cfg: EstimatorConfig | None = None,
) -> CenterFreq:
    """Power-weighted centroid + parabolic peak over the box (CLAUDE.md §4.5).

    ``f_c = sum(P[k]*f[k]) / sum(P[k])`` with ``P`` linear, over the bins inside the box;
    the peak is refined by parabolic interpolation on the dB values either side.

    Confidence follows §4.5: high when the in-box PSD is smooth and unimodal, low when it
    is multi-humped -- which usually means two signals got merged into one box. Hump count
    comes from ``find_peaks`` with a 6 dB prominence, and is reported as ``n_humps`` so
    the caller can raise a "possible blend" warning.

    Operates on the raw capture (time-sliced to the burst), not on an isolated burst --
    isolation has already moved the centre to 0 Hz by construction.
    """
    cfg = cfg or EstimatorConfig()
    x = _as_c64(x)

    seg = x
    if burst is not None:
        n0 = max(0, int(math.floor(burst.t0 * fs)))
        n1 = min(x.size, int(math.ceil(burst.t1 * fs)))
        if n1 > n0:
            seg = x[n0:n1]

    fail = Estimate(None, 0.0, "centroid/§4.5", ["burst too short for a PSD"])
    if seg.size < 16:
        return CenterFreq(centroid=fail, peak=Estimate(None, 0.0, "peak/§4.5", list(fail.notes)))

    f, p = _welch_two_sided(seg, fs, cfg.psd_nperseg)
    if burst is not None:
        inside = (f >= burst.f_lo) & (f <= burst.f_hi)
        if np.count_nonzero(inside) >= 3:
            f, p = f[inside], p[inside]

    total = float(np.sum(p))
    if total <= 0.0 or not np.isfinite(total):
        note = ["in-box power is zero"]
        return CenterFreq(
            centroid=Estimate(None, 0.0, "centroid/§4.5", note),
            peak=Estimate(None, 0.0, "peak/§4.5", list(note)),
        )

    centroid = float(np.sum(p * f) / total)

    k = int(np.argmax(p))
    bin_width = float(f[1] - f[0]) if f.size > 1 else 0.0
    peak = float(f[k]) + _refine_peak(p, k, bin_width)

    # §4.5 confidence: unimodal and smooth -> high; multi-humped -> low
    p_db = 10.0 * np.log10(p + 1e-30)
    peaks, _ = find_peaks(p_db, prominence=cfg.centroid_peak_prominence_db)
    n_humps = max(1, int(peaks.size))
    conf = {1: 0.90, 2: 0.50}.get(n_humps, 0.30)

    notes: list[str] = []
    if n_humps > 1:
        notes.append(
            f"in-box spectrum has {n_humps} humps; the box may hold more than one signal"
        )

    return CenterFreq(
        centroid=Estimate(centroid, conf, "power-weighted-centroid/§4.5", list(notes)),
        peak=Estimate(peak, conf, "parabolic-peak/§4.5", list(notes)),
        n_humps=n_humps,
    )


# --------------------------------------------------------------------------------------
# §4.6 -- Bandwidth
# --------------------------------------------------------------------------------------


@dataclass
class BandwidthResult:
    """The three §4.6 bandwidths plus how confident we are in the headline number."""

    bandwidth: Bandwidth
    occupied_99: Estimate
    minus_3db: Estimate
    minus_20db: Estimate


def _crossing_width(f: np.ndarray, p_db: np.ndarray, k_peak: int, drop_db: float) -> float | None:
    """Width between the outermost bins within ``drop_db`` of the peak (§4.6).

    Outermost rather than the contiguous run around the peak, because the -20 dB number
    exists precisely to reveal splatter sitting away from the main lobe.
    """
    above = np.where(p_db >= p_db[k_peak] - drop_db)[0]
    if above.size < 2:
        return None
    lo, hi = int(above[0]), int(above[-1])
    bin_width = float(f[1] - f[0]) if f.size > 1 else 0.0
    return float(f[hi] - f[lo]) + bin_width


def estimate_bandwidth(
    y: np.ndarray,
    fs_b: float,
    *,
    snr_db: float | None = None,
    cfg: EstimatorConfig | None = None,
) -> BandwidthResult:
    """OBW99, -3 dB and -20 dB bandwidths (CLAUDE.md §4.6).

    Welch PSD over the isolated burst, ``nperseg=1024``. OBW99 walks the cumulative power
    inward from each edge until 0.5% has passed; the remaining span is the 99% occupied
    bandwidth -- the headline number, and the one regulators use.

    Addition to §4.6: a robust noise floor (a low percentile of the PSD) is subtracted
    before the cumulative walk. §4.6's bare cumulative sum measures the occupied bandwidth
    of *signal plus noise*, which converges on the whole analysed band as SNR falls -- on
    an un-isolated 15 dB QPSK burst that inflated OBW99 by a factor of ten. Subtracting
    the floor makes the number mean what §4.6 says it means. Set
    ``subtract_psd_noise=False`` to get the literal §4.6 behaviour.

    Pass ``snr_db`` (from §4.7) whenever it is known. OBW99 is inherently biased upward at
    low SNR -- the residual noise inside the band counts toward the 99% -- and measured
    against closed-form truth the error runs +0.3% at 10 dB, +1.9% at 5 dB and +10.3% at
    0 dB. Without ``snr_db`` this function cannot see that coming: once the burst has been
    isolated, the low PSD percentile lands in the isolation filter's stopband rather than
    on the real noise floor, so the burst carries no usable evidence of its own SNR. The
    confidence would otherwise stay high while the number drifted, which is exactly the
    failure §9 D's calibration rule exists to prevent.

    Reported in Hz. Decimation does not change a bandwidth, so no scaling back is needed
    -- unlike a centre frequency, which must have ``f_shift_hz`` added.
    """
    cfg = cfg or EstimatorConfig()
    y = _as_c64(y)
    method = "welch-obw/§4.6"

    if y.size < 32:
        note = ["burst too short for a PSD"]
        return BandwidthResult(
            bandwidth=Bandwidth(),
            occupied_99=Estimate(None, 0.0, method, note),
            minus_3db=Estimate(None, 0.0, method, list(note)),
            minus_20db=Estimate(None, 0.0, method, list(note)),
        )

    f, p = _welch_two_sided(y, fs_b, cfg.psd_nperseg)
    bin_width = float(f[1] - f[0]) if f.size > 1 else 0.0

    notes: list[str] = []
    p_signal = p
    if cfg.subtract_psd_noise:
        floor = float(np.percentile(p, cfg.psd_noise_percentile))
        p_signal = np.maximum(p - floor, 0.0)
        if float(np.sum(p_signal)) <= 0.0:
            p_signal = p
            notes.append("noise-floor subtraction removed all power; used the raw PSD")

    total = float(np.sum(p_signal))
    if total <= 0.0 or not np.isfinite(total):
        note = ["burst PSD has no power"]
        return BandwidthResult(
            bandwidth=Bandwidth(),
            occupied_99=Estimate(None, 0.0, method, note),
            minus_3db=Estimate(None, 0.0, method, list(note)),
            minus_20db=Estimate(None, 0.0, method, list(note)),
        )

    # OBW99 -- walk in from each edge until (1 - 0.99)/2 of the power has passed
    tail = 0.5 * (1.0 - cfg.obw_fraction)
    cum = np.cumsum(p_signal) / total
    lo_i = int(np.searchsorted(cum, tail))
    hi_i = int(np.searchsorted(cum, 1.0 - tail))
    lo_i = min(lo_i, f.size - 1)
    hi_i = min(hi_i, f.size - 1)
    obw = float(f[hi_i] - f[lo_i]) + bin_width

    p_db = 10.0 * np.log10(p_signal + 1e-30)
    k_peak = int(np.argmax(p_signal))
    bw3 = _crossing_width(f, p_db, k_peak, 3.0)
    bw20 = _crossing_width(f, p_db, k_peak, 20.0)

    # confidence: the burst should sit inside the analysed band with room to spare. A
    # signal filling the band has been clipped by the isolation filter and the number is
    # a lower bound, not a measurement.
    fill = obw / fs_b if fs_b > 0 else 1.0
    if fill > 0.9:
        conf = 0.35
        notes.append(
            f"occupied bandwidth fills {fill:.0%} of the analysed band; "
            "the true bandwidth may be wider than the detection box"
        )
    elif fill > 0.75:
        conf = 0.6
    else:
        conf = 0.85

    # low SNR inflates OBW99 -- residual in-band noise counts toward the 99%
    if snr_db is None:
        notes.append(
            "confidence does not account for SNR; no SNR was supplied to this estimator"
        )
    elif snr_db < 6.0:
        conf *= float(np.clip(0.3 + 0.7 * (snr_db / 6.0), 0.3, 1.0))
        notes.append(
            f"measured SNR is {snr_db:.1f} dB; below 6 dB the 99% occupied bandwidth reads "
            "high because in-band noise counts toward the 99%"
        )

    return BandwidthResult(
        bandwidth=Bandwidth(occupied_99=obw, minus_3db=bw3, minus_20db=bw20),
        occupied_99=Estimate(obw, conf, method, list(notes)),
        minus_3db=Estimate(bw3, conf if bw3 else 0.0, "welch-3db/§4.6", list(notes)),
        minus_20db=Estimate(bw20, conf if bw20 else 0.0, "welch-20db/§4.6", list(notes)),
    )


# --------------------------------------------------------------------------------------
# §4.7 -- SNR
# --------------------------------------------------------------------------------------


@dataclass
class SnrResult:
    """SNR plus the two powers behind it, so an analyst can check the arithmetic."""

    snr: Estimate
    power_dbfs: float | None
    noise_power_dbfs: float | None
    below_floor: bool = False


def estimate_snr(
    spec: Spectrogram,
    burst: Burst,
    *,
    cfg: EstimatorConfig | None = None,
) -> SnrResult:
    """In-band SNR from the spectrogram (CLAUDE.md §4.7).

    Follows §4.7's accounting exactly::

        P_sig_plus_noise = mean power inside the burst box
        P_noise_in_band  = noise power density outside the box * the box's bin count
        P_signal         = P_sig_plus_noise - P_noise_in_band
        SNR_db           = 10*log10(P_signal / P_noise_in_band)

    Departure from §4.2, deliberate and measured: the noise density is the **median over
    out-of-box bins of the mean-over-time linear power**, not the §4.2 25th-percentile-in-dB
    floor. §4.2's floor is built to be a robust *detection threshold*; used as a *power* it
    is a low quantile of a chi-squared magnitude and reads 5.45 dB below the true mean,
    which passed straight into SNR as a +5.4 dB bias on every burst. Taking the median
    across frequency keeps §4.2's robustness against other signals in the band; taking the
    mean across time restores an unbiased power. Verified against known-truth AWGN from
    -5 dB to +25 dB: residual error under 0.05 dB.

    §4.7's other instruction is honoured too -- when ``P_signal <= 0`` this reports
    ``below_floor`` with a low confidence rather than a NaN.
    """
    cfg = cfg or EstimatorConfig()
    method = "spectrogram-power-budget/§4.7"

    f = spec.f
    in_box = (f >= burst.f_lo) & (f <= burst.f_hi)
    if not np.any(in_box):
        note = ["burst box contains no spectrogram bins"]
        return SnrResult(Estimate(None, 0.0, method, note), None, None)

    # column range of the burst, so a short burst is not averaged against empty time
    t = spec.t
    cols = (t >= burst.t0) & (t <= burst.t1)
    if not np.any(cols):
        cols = np.ones(t.size, dtype=bool)

    guard = cfg.snr_guard_frac * max(burst.bandwidth, 0.0)
    outside = (f < burst.f_lo - guard) | (f > burst.f_hi + guard)
    if np.count_nonzero(outside) < 8:
        outside = ~in_box
    if np.count_nonzero(outside) < 1:
        note = ["no out-of-box bins available to measure the noise floor"]
        return SnrResult(Estimate(None, 0.0, method, note), None, None)

    # Noise is a whole-capture, whole-spectrum quantity, so it comes from the memoised
    # per-bin mean rather than a fresh linear copy of the spectrogram per detection.
    mean_power = spec.mean_power_per_bin()
    noise_per_bin = float(np.median(mean_power[outside]))

    # Signal power is restricted to this burst's columns, so only the in-box rows are
    # converted -- a few bins wide instead of the whole 8192-bin array.
    # np.ix_ selects rows and columns in one pass. `S_db[in_box][:, cols]` first
    # materialises every in-box row across *all* columns -- 48 MB for a wide detection --
    # and then throws most of it away.
    in_box_block = spec.S_db[np.ix_(in_box, cols)].astype(np.float64)
    per_bin_in_box = np.power(10.0, in_box_block / 10.0).mean(axis=1)

    n_bins = int(np.count_nonzero(in_box))
    p_noise = noise_per_bin * n_bins
    p_total = float(np.sum(per_bin_in_box))
    p_signal = p_total - p_noise

    power_dbfs = 10.0 * math.log10(p_total) if p_total > 0 else None
    noise_dbfs = 10.0 * math.log10(p_noise) if p_noise > 0 else None

    if p_noise <= 0.0 or not np.isfinite(p_noise):
        note = ["noise power came out non-positive"]
        return SnrResult(Estimate(None, 0.0, method, note), power_dbfs, noise_dbfs)

    if p_signal <= 0.0:
        # §4.7: report "< 0 dB" with low confidence rather than NaN
        return SnrResult(
            snr=Estimate(
                None,
                0.15,
                method,
                ["signal power did not exceed the in-band noise; SNR is below 0 dB"],
            ),
            power_dbfs=power_dbfs,
            noise_power_dbfs=noise_dbfs,
            below_floor=True,
        )

    snr_db = 10.0 * math.log10(p_signal / p_noise)
    # confidence falls away as the signal approaches the floor, where the subtraction of
    # two similar numbers stops being meaningful
    conf = float(np.clip(0.35 + 0.055 * snr_db, 0.2, 0.9))
    return SnrResult(
        snr=Estimate(snr_db, conf, method, []),
        power_dbfs=power_dbfs,
        noise_power_dbfs=noise_dbfs,
    )


# --------------------------------------------------------------------------------------
# §4.8 -- Symbol rate: three methods, a harmonic check, then reconciliation
# --------------------------------------------------------------------------------------


def _line_spectrum(
    v: np.ndarray, fs: float, cfg: EstimatorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed, zero-padded magnitude spectrum of a real detector output (§4.8a/§4.8b).

    Blackman-Harris window and 8x zero-padding, per §4.8. The analysed length is capped at
    ``max_analysis_samples`` taken from the middle of the burst so a multi-megasample
    burst does not turn into a multi-gigabyte FFT; at the default cap the frequency
    resolution is already far finer than any tolerance in §9.
    """
    v = np.asarray(_centre_slice(np.asarray(v, dtype=np.float64), cfg.max_analysis_samples))
    v = v - float(np.mean(v))
    if v.size < 16:
        return np.zeros(0), np.zeros(0)
    nfft = min(cfg.line_zero_pad * v.size, cfg.max_fft)
    nfft = max(nfft, v.size)
    mag = np.abs(np.fft.rfft(v * blackmanharris(v.size), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, mag


def _line_height_db(
    freqs: np.ndarray, mag: np.ndarray, median: float, target: float, cfg: EstimatorConfig
) -> float:
    """Height in dB above the spectrum's median of the strongest line near ``target``."""
    if freqs.size == 0 or median <= 0:
        return -np.inf
    tol = cfg.harmonic_search_frac * target
    window = (freqs >= target - tol) & (freqs <= target + tol)
    if not np.any(window):
        return -np.inf
    return float(20.0 * np.log10(float(np.max(mag[window])) / median + 1e-30))


def _harmonic_check(
    freqs: np.ndarray,
    mag: np.ndarray,
    peak_freq: float,
    peak_height_db: float,
    lo: float,
    cfg: EstimatorConfig,
) -> tuple[float, list[str]]:
    """Prefer the lowest rate that explains every observed line (CLAUDE.md §4.8).

    §4.8 calls this out as the single most common failure in the project: the spectral
    line method locks onto 2x or 0.5x the true rate. Method (a) on 2-FSK lands on 4x; method
    (b) on RRC QPSK lands on 2x. Both are fixed here.

    The rule: try dividing the observed peak by 4, then 3, then 2, and accept the first
    (hence lowest) candidate whose own fundamental line is genuinely strong -- at least
    ``harmonic_line_frac`` of the observed peak's height and at least
    ``harmonic_min_line_db`` above the spectrum median. Requiring the candidate's *own*
    line to be present is what stops the rule collapsing to the smallest divisor every
    time, since any divisor trivially "explains" a line it divides.

    The 0.75 fraction separates the two populations cleanly on known-truth signals: a real
    fundamental sits at 0.90-0.95 of the peak height, a spurious subharmonic at 0.55-0.60.
    """
    notes: list[str] = []
    if freqs.size == 0:
        return peak_freq, notes
    in_range = freqs >= lo
    if not np.any(in_range):
        return peak_freq, notes
    median = float(np.median(mag[in_range]))
    if median <= 0:
        return peak_freq, notes

    for divisor in cfg.harmonic_divisors:
        candidate = peak_freq / divisor
        if candidate < lo:
            continue
        height = _line_height_db(freqs, mag, median, candidate, cfg)
        if (
            height >= cfg.harmonic_line_frac * peak_height_db
            and height >= cfg.harmonic_min_line_db
        ):
            notes.append(
                f"harmonic check: a line at {candidate:.1f} Hz ({height:.1f} dB) explains "
                f"the peak at {peak_freq:.1f} Hz as its {divisor}x harmonic; "
                f"reported the lower rate"
            )
            return candidate, notes
    return peak_freq, notes


def _peak_in_range(
    freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float, cfg: EstimatorConfig
) -> tuple[float, float, float] | None:
    """Strongest line in ``[lo, hi]`` -> (refined frequency, height dB, median)."""
    if freqs.size == 0:
        return None
    band = (freqs >= lo) & (freqs <= hi)
    if np.count_nonzero(band) < 8:
        return None
    idx = np.where(band)[0]
    sub = mag[idx]
    k_local = int(np.argmax(sub))
    k = int(idx[k_local])
    median = float(np.median(sub))
    if median <= 0 or mag[k] <= 0:
        return None
    height = float(20.0 * np.log10(mag[k] / median))
    bin_width = float(freqs[1] - freqs[0])
    refined = float(freqs[k]) + _refine_peak(mag, k, bin_width)
    return refined, height, median


def symbol_rate_cyclostationary(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> Estimate:
    """Symbol rate from the cyclostationary spectral line -- §4.8 method (a), the primary.

    A digitally modulated signal carries hidden periodicity at the symbol rate; squaring
    the envelope makes it visible as a spectral line. §4.8 exactly::

        z = |y|**2  ->  remove its (huge) DC term  ->  Blackman-Harris window
        ->  8x zero-padded FFT  ->  strongest line in [fs_b/1000, fs_b/2]
        ->  parabolic refinement  ->  harmonic check

    Confidence is the line's height above the local median through the §4.8 logistic:
    12 dB -> ~0.9, 3 dB -> ~0.2.

    Fails on MSK/GMSK, where ``|y|^2`` is flat by construction -- §4.8 says fall back to
    method (b), and :func:`estimate_symbol_rate` does.
    """
    cfg = cfg or EstimatorConfig()
    method = "cyclostationary/§4.8a"
    y = _unit_power(_as_c64(y))
    if y.size < 64 or fs_b <= 0:
        return Estimate(None, 0.0, method, ["burst too short for a cyclic spectrum"])

    z = np.abs(y.astype(np.complex128)) ** 2
    if float(np.std(z)) < 1e-9:
        return Estimate(
            None, 0.0, method, ["envelope is flat (constant modulus); no cyclic line exists"]
        )

    freqs, mag = _line_spectrum(z, fs_b, cfg)
    lo, hi = cfg.rate_lo_frac * fs_b, cfg.rate_hi_frac * fs_b
    found = _peak_in_range(freqs, mag, lo, hi, cfg)
    if found is None:
        return Estimate(None, 0.0, method, ["no cyclic line found in the searched range"])

    peak, height, _ = found
    rate, notes = _harmonic_check(freqs, mag, peak, height, lo, cfg)
    conf = _logistic(height, cfg.line_conf_slope, cfg.line_conf_midpoint)
    notes.insert(0, f"cyclic line {height:.1f} dB above the local median")
    return Estimate(float(rate), conf, method, notes)


def symbol_rate_freq_transitions(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> Estimate:
    """Symbol rate from instantaneous-frequency transitions -- §4.8 method (b).

    ``f_inst`` jumps at every symbol boundary, so ``|diff(f_inst)|`` is a pulse train at
    the symbol rate and its spectrum shows a line there. §4.8's recipe, with the same
    windowing, zero-padding, refinement and harmonic check as method (a).

    This is the method that carries FSK, MSK and anything else with phase transitions,
    where method (a) has nothing to work with. It is also the method most prone to the
    octave error -- on RRC QPSK its strongest line is at 2x the symbol rate -- which is
    exactly what the harmonic check exists to undo.
    """
    cfg = cfg or EstimatorConfig()
    method = "inst-freq-transitions/§4.8b"
    y = _unit_power(_as_c64(y))
    if y.size < 64 or fs_b <= 0:
        return Estimate(None, 0.0, method, ["burst too short for an inst-frequency spectrum"])

    f_inst = instantaneous_frequency(y, fs_b)
    if f_inst.size < 16:
        return Estimate(None, 0.0, method, ["burst too short after differencing"])
    d = np.abs(np.diff(f_inst))

    freqs, mag = _line_spectrum(d, fs_b, cfg)
    lo, hi = cfg.rate_lo_frac * fs_b, cfg.rate_hi_frac * fs_b
    found = _peak_in_range(freqs, mag, lo, hi, cfg)
    if found is None:
        return Estimate(None, 0.0, method, ["no transition line found in the searched range"])

    peak, height, _ = found
    rate, notes = _harmonic_check(freqs, mag, peak, height, lo, cfg)
    conf = _logistic(height, cfg.line_conf_slope, cfg.line_conf_midpoint)
    notes.insert(0, f"transition line {height:.1f} dB above the local median")
    return Estimate(float(rate), conf, method, notes)


def symbol_rate_envelope_autocorr(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> Estimate:
    """Symbol rate from envelope autocorrelation -- §4.8 method (c), the sanity check.

    Correlate ``z = |y|^2`` with itself, find the first strong non-zero-lag peak ``L``,
    and report ``fs_b / L``. §4.8 describes this as a cheap check that catches octave
    errors, and it is weighted lowest of the three.

    It abstains often, and that is correct rather than a shortfall. The envelope
    autocorrelation of a *random* data stream is a triangle with no periodic peak at all,
    so for ordinary random-payload PSK and QAM there is genuinely nothing to lock to; it
    contributes on framed, bursty or repetitive transmissions, where a periodic envelope
    really does exist and where octave errors are most likely. Measured on known-truth
    RRC QPSK it abstains at every SNR rather than returning the junk values a looser
    acceptance threshold produced. §2's hard rule applies: no answer beats a wrong one.
    """
    cfg = cfg or EstimatorConfig()
    method = "envelope-autocorr/§4.8c"
    y = _unit_power(_as_c64(y))
    if y.size < 64 or fs_b <= 0:
        return Estimate(None, 0.0, method, ["burst too short for an autocorrelation"])

    z = np.abs(_centre_slice(y, cfg.max_analysis_samples).astype(np.complex128)) ** 2
    z = z - float(np.mean(z))
    if float(np.std(z)) < 1e-9:
        return Estimate(
            None, 0.0, method, ["envelope is flat (constant modulus); autocorrelation is empty"]
        )

    nfft = 1 << int(math.ceil(math.log2(2 * z.size)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(z, nfft)) ** 2)[: z.size]
    if ac[0] <= 0:
        return Estimate(None, 0.0, method, ["autocorrelation has no energy at zero lag"])
    ac = ac / ac[0]

    # step past the main lobe -- its width is set by the pulse shape, not the symbol rate
    i = 1
    while i < ac.size - 1 and ac[i + 1] < ac[i]:
        i += 1
    lag_lo = max(i, int(math.ceil(1.0 / cfg.rate_hi_frac)))
    lag_hi = min(ac.size - 2, int(1.0 / cfg.rate_lo_frac))
    if lag_hi <= lag_lo:
        return Estimate(None, 0.0, method, ["no usable lag range"])

    seg = ac[lag_lo : lag_hi + 1]
    k = int(np.argmax(seg)) + lag_lo
    peak = float(ac[k])
    background = float(np.median(np.abs(seg)))
    if peak < cfg.autocorr_min_peak or peak < cfg.autocorr_min_ratio * background:
        return Estimate(
            None,
            0.0,
            method,
            [
                f"no convincing envelope periodicity (peak {peak:.3f} at lag {k}, "
                f"background {background:.3f})"
            ],
        )

    delta = _parabolic_offset(float(ac[k - 1]), float(ac[k]), float(ac[k + 1]))
    lag = k + delta
    if lag <= 0:
        return Estimate(None, 0.0, method, ["refined lag was non-positive"])
    conf = min(cfg.autocorr_conf_cap, peak)
    return Estimate(
        float(fs_b / lag),
        conf,
        method,
        [f"envelope autocorrelation peaks at lag {k} ({peak:.2f} of zero-lag)"],
    )


@dataclass
class SymbolRateResult:
    """The reconciled symbol rate plus every individual opinion behind it (§4.8)."""

    estimate: Estimate
    methods: dict[str, Estimate] = field(default_factory=dict)
    agreed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def estimate_symbol_rate(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> SymbolRateResult:
    """Run all three §4.8 methods and reconcile them (CLAUDE.md §4.8 "Reconciliation").

    §4.8's rule, made explicit about the cases it leaves open:

    * two or more methods agreeing within 5% -> report the confidence-weighted mean of
      that group at confidence >= 0.85;
    * all available methods disagreeing -> report method (a) (or the best available) with
      confidence capped at 0.4, plus a warning;
    * only one method able to answer at all -> report it with confidence capped at 0.6 and
      a note that no cross-check was possible. This case is common and legitimate: on FSK
      method (a) has a flat envelope to work with and method (c) has nothing periodic, so
      method (b) stands alone;
    * nothing able to answer -> ``value=None`` with the reasons, never a guess.

    The harmonic check has already run inside each method, so the values being reconciled
    are octave-corrected before they are compared.
    """
    cfg = cfg or EstimatorConfig()
    a = symbol_rate_cyclostationary(y, fs_b, cfg=cfg)
    b = symbol_rate_freq_transitions(y, fs_b, cfg=cfg)
    c = symbol_rate_envelope_autocorr(y, fs_b, cfg=cfg)
    methods = {"cyclostationary": a, "inst_freq": b, "envelope_autocorr": c}

    available = [(name, e) for name, e in methods.items() if e.value is not None and e.value > 0]
    warnings: list[str] = []

    if not available:
        reasons = [n for e in methods.values() for n in e.notes] or ["no method could measure"]
        return SymbolRateResult(
            estimate=Estimate(None, 0.0, "reconciled/§4.8", reasons),
            methods=methods,
            warnings=["symbol rate could not be measured by any of the three §4.8 methods"],
        )

    # largest group of mutually-agreeing methods (within reconcile_tol, relatively)
    best_group: list[tuple[str, Estimate]] = []
    for _, est_i in available:
        group = [
            (name_j, est_j)
            for name_j, est_j in available
            if abs(est_j.value - est_i.value) / min(est_j.value, est_i.value) <= cfg.reconcile_tol
        ]
        if len(group) > len(best_group):
            best_group = group

    if len(best_group) >= 2:
        weights = np.array([max(e.confidence, 1e-3) for _, e in best_group])
        values = np.array([float(e.value) for _, e in best_group])
        value = float(np.sum(values * weights) / np.sum(weights))
        conf = max(cfg.reconcile_conf, float(np.mean(weights)))
        names = [n for n, _ in best_group]
        notes = [
            f"{len(best_group)} of 3 methods agree within "
            f"{cfg.reconcile_tol:.0%} ({', '.join(names)})"
        ]
        notes += [n for _, e in best_group for n in e.notes if "harmonic check" in n]
        return SymbolRateResult(
            estimate=Estimate(value, min(conf, 0.97), f"reconciled({'+'.join(names)})/§4.8", notes),
            methods=methods,
            agreed=names,
            warnings=warnings,
        )

    # no agreement -- fall back to the primary method, honestly flagged
    priority = ["cyclostationary", "inst_freq", "envelope_autocorr"]
    name = next(n for n in priority if methods[n].value is not None and methods[n].value > 0)
    chosen = methods[name]

    if len(available) == 1:
        notes = list(chosen.notes) + [
            "only one of the three §4.8 methods could measure; no cross-check was possible"
        ]
        conf = min(chosen.confidence, cfg.lone_method_conf_cap)
        warnings.append(
            f"symbol rate rests on {name} alone; the other two §4.8 methods abstained"
        )
    else:
        others = ", ".join(
            f"{n}={e.value:.1f} Hz" for n, e in available if n != name
        )
        notes = list(chosen.notes) + [
            f"the three §4.8 methods disagree by more than {cfg.reconcile_tol:.0%} ({others})"
        ]
        conf = min(chosen.confidence, cfg.disagree_conf_cap)
        warnings.append("symbol rate methods disagree; treat this value with caution")

    return SymbolRateResult(
        estimate=Estimate(float(chosen.value), conf, f"{chosen.method} (unreconciled)", notes),
        methods=methods,
        warnings=warnings,
    )


# --------------------------------------------------------------------------------------
# §4.9 -- Carrier frequency offset and PSK order
# --------------------------------------------------------------------------------------


@dataclass
class MthPowerResult:
    """M-th power results: PSK order, carrier offset, and the raw sharpness features."""

    psk_order: Estimate
    carrier_offset: Estimate
    sharpness_db: dict[int, float] = field(default_factory=dict)


def estimate_psk_order(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> MthPowerResult:
    """PSK order and carrier offset by the M-th power test (CLAUDE.md §4.9).

    For M-PSK, raising to the M-th power collapses the modulation and leaves a tone at
    ``M * f_offset``. §4.9's loop over M in (1, 2, 4, 8), scoring each by its peak height
    above the spectrum median; the M giving the sharpest single line is the order, and
    ``f_offset = peak / M``.

    M = 1 is computed and reported as a feature but excluded from the order vote: for a
    pulse-shaped signal the raw spectrum is already peaky against its own out-of-band
    floor, so M = 1 scores high on everything and means only "this is a narrowband
    signal", not "this is a carrier".

    The order confidence is the margin between the winner and the runner-up, which
    collapses honestly where the test genuinely stops working -- at 5 dB SNR 8PSK's M = 8
    line is overtaken by its M = 4 line and the confidence reflects that.

    ``sharpness_db`` is returned in full because §4.9 asks for it as a classifier feature.
    """
    cfg = cfg or EstimatorConfig()
    method = "mth-power/§4.9"
    y = _unit_power(_as_c64(y))
    if y.size < 64:
        note = ["burst too short for the M-th power test"]
        return MthPowerResult(
            psk_order=Estimate(None, 0.0, method, note),
            carrier_offset=Estimate(None, 0.0, method, list(note)),
        )

    seg = _centre_slice(y, cfg.max_analysis_samples).astype(np.complex128)
    nfft = min(cfg.mth_zero_pad * seg.size, cfg.max_fft)
    nfft = max(nfft, seg.size)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs_b))

    sharpness: dict[int, float] = {}
    peak_freq: dict[int, float] = {}
    for m in cfg.mth_powers:
        mag = np.abs(np.fft.fftshift(np.fft.fft(seg**m, nfft)))
        median = float(np.median(mag))
        if median <= 0:
            continue
        k = int(np.argmax(mag))
        sharpness[m] = float(20.0 * np.log10(mag[k] / median))
        bin_width = float(freqs[1] - freqs[0])
        peak_freq[m] = float(freqs[k]) + _refine_peak(mag, k, bin_width)

    candidates = {m: s for m, s in sharpness.items() if m in cfg.mth_orders}
    if not candidates:
        note = ["M-th power spectra were unusable"]
        return MthPowerResult(
            psk_order=Estimate(None, 0.0, method, note),
            carrier_offset=Estimate(None, 0.0, method, list(note)),
            sharpness_db=sharpness,
        )

    best_m = max(candidates, key=lambda m: candidates[m])
    best_s = candidates[best_m]
    others = [s for m, s in candidates.items() if m != best_m]
    margin = best_s - max(others) if others else best_s

    notes = [
        "M-th power sharpness (dB above median): "
        + ", ".join(f"M={m}: {sharpness[m]:.1f}" for m in sorted(sharpness))
    ]

    if best_s < cfg.mth_min_sharpness_db:
        return MthPowerResult(
            psk_order=Estimate(
                None,
                0.0,
                method,
                notes + [f"no M-th power line reached {cfg.mth_min_sharpness_db:.0f} dB"],
            ),
            carrier_offset=Estimate(None, 0.0, method, list(notes)),
            sharpness_db=sharpness,
        )

    order_conf = float(np.clip(0.30 + 0.06 * margin, 0.2, 0.95))
    notes.append(
        f"raising to the power {best_m} produced the sharpest single line "
        f"({best_s:.1f} dB, {margin:.1f} dB clear of the next order)"
    )

    offset = peak_freq.get(best_m)
    if offset is None:
        offset_est = Estimate(None, 0.0, method, list(notes))
    else:
        offset_est = Estimate(
            float(offset / best_m),
            order_conf,
            method,
            list(notes)
            + [
                f"carrier offset is the M={best_m} line at {offset:.1f} Hz divided by {best_m}; "
                f"it is only determined modulo {fs_b / best_m:.1f} Hz"
            ],
        )

    return MthPowerResult(
        psk_order=Estimate(float(best_m), order_conf, method, notes),
        carrier_offset=offset_est,
        sharpness_db=sharpness,
    )


# --------------------------------------------------------------------------------------
# §4.10 -- FSK parameters
# --------------------------------------------------------------------------------------


@dataclass
class FskParams:
    """FSK tone count, deviation, spacing and modulation index (§4.10)."""

    n_tones: Estimate
    deviation_hz: Estimate
    tone_spacing_hz: Estimate
    modulation_index: Estimate
    tone_freqs_hz: list[float] = field(default_factory=list)


def estimate_fsk_params(
    y: np.ndarray,
    fs_b: float,
    *,
    symbol_rate: float | None = None,
    cfg: EstimatorConfig | None = None,
) -> FskParams:
    """FSK tones, deviation and modulation index (CLAUDE.md §4.10).

    Histogram ``f_inst`` into 200 bins over the burst; the peaks are the FSK tones. Peak
    count gives M in M-FSK; the outer spread gives the deviation; adjacent spacing gives
    the tone spacing.

    Two refinements over the bare §4.10 recipe, both needed to make it work at all off a
    clean bench signal: ``f_inst`` is clipped to its 0.5/99.5 percentiles before
    histogramming, and the histogram is smoothed over 5 bins. Without the clip a handful
    of phase-slip outliers stretch the histogram range so far that all the real tones fall
    into one bin -- on known-truth 4-FSK at 15 dB the unclipped version finds one peak
    instead of four.

    §4.10 reports ``deviation`` two different ways (half the outer spread for 2-FSK, tone
    spacing for M-FSK), so both are returned separately and unambiguously:
    ``deviation_hz`` is always half the outer spread and ``tone_spacing_hz`` is always the
    mean adjacent spacing. For 2-FSK they differ by a factor of two, by definition.

    ``modulation_index`` is ``tone_spacing / symbol_rate``. That is the standard
    ``h = 2*f_dev/Rs``, and it is the reading of §4.10 under which "h ~ 0.5 means MSK"
    is actually true -- taking h as half-spread over symbol rate would put MSK at 0.25.
    """
    cfg = cfg or EstimatorConfig()
    method = "inst-freq-histogram/§4.10"
    y = _unit_power(_as_c64(y))

    def _fail(reason: str) -> FskParams:
        note = [reason]
        return FskParams(
            n_tones=Estimate(None, 0.0, method, note),
            deviation_hz=Estimate(None, 0.0, method, list(note)),
            tone_spacing_hz=Estimate(None, 0.0, method, list(note)),
            modulation_index=Estimate(None, 0.0, method, list(note)),
        )

    if y.size < 128:
        return _fail("burst too short for an instantaneous-frequency histogram")

    f_inst = instantaneous_frequency(y, fs_b)
    if f_inst.size < 64:
        return _fail("burst too short after differencing")

    lo, hi = np.percentile(f_inst, [cfg.fsk_clip_percentile, 100.0 - cfg.fsk_clip_percentile])
    trimmed = f_inst[(f_inst >= lo) & (f_inst <= hi)]
    if trimmed.size < 64 or not np.isfinite([lo, hi]).all() or hi <= lo:
        return _fail("instantaneous frequency has no usable spread")

    hist, edges = np.histogram(trimmed, bins=cfg.fsk_hist_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.ones(cfg.fsk_smooth) / cfg.fsk_smooth
    smooth = np.convolve(hist.astype(float), kernel, mode="same")
    if smooth.max() <= 0:
        return _fail("instantaneous-frequency histogram is empty")

    peaks, _ = find_peaks(smooth, prominence=cfg.fsk_peak_prominence_frac * smooth.max())
    tones = [float(v) for v in centres[peaks]]

    if len(tones) < 2:
        return _fail(
            f"instantaneous-frequency histogram has {len(tones)} peak(s); not FSK-like"
        )
    if len(tones) > cfg.fsk_max_tones:
        return _fail(
            f"instantaneous-frequency histogram has {len(tones)} peaks; too many for M-FSK"
        )

    tones.sort()
    deviation = 0.5 * (tones[-1] - tones[0])
    spacing = float(np.mean(np.diff(tones)))

    # a clean M-FSK has evenly spaced tones; uneven spacing means the peaks are not tones
    gaps = np.diff(tones)
    evenness = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 1.0
    conf = float(np.clip(0.9 - 2.0 * evenness, 0.2, 0.9))
    notes = [
        f"the instantaneous frequency histogram has {len(tones)} distinct peaks "
        f"spaced {spacing:.0f} Hz apart"
    ]
    if evenness > 0.15:
        notes.append(
            f"tone spacing is uneven (relative spread {evenness:.0%}); "
            "the peak set may not be a clean M-FSK alphabet"
        )

    if symbol_rate and symbol_rate > 0:
        h_value = spacing / symbol_rate
        h_notes = list(notes)
        if 0.45 <= h_value <= 0.55:
            h_notes.append("modulation index is close to 0.5, which is MSK")
        h_est = Estimate(float(h_value), conf, method, h_notes)
    else:
        h_est = Estimate(
            None, 0.0, method, ["modulation index needs a symbol rate, which was not measured"]
        )

    return FskParams(
        n_tones=Estimate(float(len(tones)), conf, method, list(notes)),
        deviation_hz=Estimate(float(deviation), conf, method, list(notes)),
        tone_spacing_hz=Estimate(float(spacing), conf, method, list(notes)),
        modulation_index=h_est,
        tone_freqs_hz=tones,
    )


# --------------------------------------------------------------------------------------
# §4.11 -- OFDM
# --------------------------------------------------------------------------------------


@dataclass
class OfdmParams:
    """Cyclic-prefix autocorrelation results (§4.11)."""

    is_ofdm: bool
    useful_symbol_len: Estimate
    symbol_duration_s: Estimate
    subcarrier_spacing_hz: Estimate
    cp_length: Estimate
    r_peak: float | None = None
    peak_ratio: float | None = None


def estimate_ofdm_params(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> OfdmParams:
    """Detect OFDM and measure its symbol geometry (CLAUDE.md §4.11).

    OFDM repeats the end of each symbol at its start -- the cyclic prefix -- so the
    autocorrelation spikes at a lag equal to the useful symbol length::

        R[lag] = |sum y[n]*conj(y[n+lag])| / sum |y[n]|^2

    A clear peak at lag ``L`` gives useful symbol duration ``L/fs_b`` and subcarrier
    spacing ``fs_b/L``. §4.11 is right that this is physics rather than statistics, and
    the ensemble trusts it over the CNN when it fires.

    Two deliberate departures from the letter of §4.11:

    * Detection keys on the peak's ratio to the median of ``R``, not on ``R > 0.3``. With
      the global normalisation §4.11 specifies, ``R`` at the correct lag converges on
      ``CP/(N+CP)``, so even a generous 1/4 cyclic prefix tops out at 0.2 and a fixed 0.3
      threshold can never fire. Measured on known-truth OFDM the peak-to-median ratio is
      23-53 across 0-20 dB SNR while the background stays near 0.005, so the ratio test
      separates cleanly where the absolute test does not. ``R`` is still reported.
    * CP length comes from that same amplitude relation, ``cp = R*L/(1 - R)``, rather than
      from the width of a correlation plateau. In the lag domain there is no plateau to
      measure: with all subcarriers loaded the signal is white and ``R`` falls to the floor
      at ``L +/- 1``. The amplitude route recovers CP = 15.8 against a true 16.

    ``R`` is computed by FFT (O(N log N)) rather than a per-lag loop; with 2x zero-padding
    that is the same linear autocorrelation §4.11 writes out.
    """
    cfg = cfg or EstimatorConfig()
    method = "cyclic-prefix-autocorr/§4.11"
    y = _unit_power(_as_c64(y))

    def _fail(reason: str) -> OfdmParams:
        note = [reason]
        return OfdmParams(
            is_ofdm=False,
            useful_symbol_len=Estimate(None, 0.0, method, note),
            symbol_duration_s=Estimate(None, 0.0, method, list(note)),
            subcarrier_spacing_hz=Estimate(None, 0.0, method, list(note)),
            cp_length=Estimate(None, 0.0, method, list(note)),
        )

    seg = _centre_slice(y, cfg.max_analysis_samples).astype(np.complex128)
    if seg.size < 256:
        return _fail("burst too short to look for a cyclic prefix")

    nfft = 1 << int(math.ceil(math.log2(2 * seg.size)))
    spectrum = np.fft.fft(seg, nfft)
    ac = np.fft.ifft(np.abs(spectrum) ** 2)
    energy = float(ac[0].real)
    if energy <= 0:
        return _fail("burst has no energy")
    r = np.abs(ac) / energy

    lag_hi = int(min(cfg.ofdm_max_lag, cfg.ofdm_max_lag_frac * seg.size))

    # Start the search past the autocorrelation main lobe. The main lobe is the pulse
    # shape correlating with itself, and its width is set by the signal's bandwidth, not
    # by any cyclic prefix. On an un-decimated RRC burst at 20 samples per symbol that
    # lobe reaches lag ~20 and comfortably clears the peak-ratio test, so without this a
    # plain QPSK signal is reported as OFDM -- and because §5.4 lets the OFDM rule
    # override both learned models, that false positive would win.
    main_lobe = 1
    while main_lobe < r.size - 1 and r[main_lobe + 1] < r[main_lobe]:
        main_lobe += 1
    lag_lo = int(max(cfg.ofdm_min_lag, main_lobe + 1))
    if lag_hi <= lag_lo + 4:
        return _fail("burst too short for the cyclic-prefix lag range")

    band = r[lag_lo : lag_hi + 1]
    k = int(np.argmax(band)) + lag_lo
    r_peak = float(r[k])
    background = float(np.median(band))
    ratio = r_peak / background if background > 0 else float("inf")

    if ratio < cfg.ofdm_peak_ratio or r_peak < cfg.ofdm_min_r or r_peak > cfg.ofdm_max_r:
        if r_peak > cfg.ofdm_max_r:
            reason = [
                f"correlation peak R={r_peak:.3f} at lag {k} is too strong to be a cyclic "
                f"prefix (a prefix of CP/(N+CP) > {cfg.ofdm_max_r:.2f} would be longer than "
                "half the useful symbol); this is periodicity from something else"
            ]
        else:
            reason = [
                f"no cyclic-prefix correlation peak (best R={r_peak:.3f} at lag {k}, "
                f"{ratio:.1f}x the background; need {cfg.ofdm_peak_ratio:.0f}x)"
            ]
        return OfdmParams(
            is_ofdm=False,
            useful_symbol_len=Estimate(None, 0.0, method, list(reason)),
            symbol_duration_s=Estimate(None, 0.0, method, list(reason)),
            subcarrier_spacing_hz=Estimate(None, 0.0, method, list(reason)),
            cp_length=Estimate(None, 0.0, method, list(reason)),
            r_peak=r_peak,
            peak_ratio=ratio,
        )

    conf = float(np.clip(0.5 + 0.012 * (ratio - cfg.ofdm_peak_ratio), 0.5, 0.95))
    notes = [
        f"autocorrelation peaks at lag {k} samples ({ratio:.0f}x the background), "
        "consistent with a cyclic prefix"
    ]
    cp = r_peak * k / (1.0 - r_peak) if r_peak < 1.0 else None

    return OfdmParams(
        is_ofdm=True,
        useful_symbol_len=Estimate(float(k), conf, method, list(notes)),
        symbol_duration_s=Estimate(float(k / fs_b), conf, method, list(notes)),
        subcarrier_spacing_hz=Estimate(float(fs_b / k), conf, method, list(notes)),
        cp_length=Estimate(
            float(cp) if cp is not None else None,
            conf * 0.8 if cp is not None else 0.0,
            method,
            list(notes)
            + ["cyclic prefix length inferred from the correlation peak height, R*L/(1-R)"],
        ),
        r_peak=r_peak,
        peak_ratio=ratio,
    )


# --------------------------------------------------------------------------------------
# §4.12 -- Chirp / LFM
# --------------------------------------------------------------------------------------


@dataclass
class ChirpParams:
    """Spectrogram ridge fit results (§4.12)."""

    is_chirp: bool
    chirp_rate_hz_per_s: Estimate
    sweep_bandwidth_hz: Estimate
    r2_linear: float | None = None
    r2_quadratic: float | None = None
    nonlinear: bool = False


def estimate_chirp(
    x: np.ndarray,
    fs: float,
    *,
    f_lo: float | None = None,
    f_hi: float | None = None,
    cfg: EstimatorConfig | None = None,
    spec: Spectrogram | None = None,
) -> ChirpParams:
    """Detect a linear FM sweep by fitting the spectrogram ridge (CLAUDE.md §4.12).

    A chirp draws a straight line in the spectrogram. §4.12's four steps: take the
    per-column frequency of maximum power to get the ridge; keep only columns whose peak
    stands more than 6 dB above that column's median; fit ``f = a*t + b``; and if
    ``R^2 > 0.9`` with a large sweep, call it a chirp with ``chirp_rate = a`` Hz/s and
    ``sweep_bandwidth = |a| * duration``.

    A degree-2 fit runs alongside, as §4.12 asks: when the quadratic beats the linear R^2
    by a clear margin the sweep is nonlinear, which is worth reporting on its own.

    Runs on the raw (time-sliced) capture rather than an isolated burst -- isolation
    lowpasses to the box width and would filter away the very sweep being measured.

    Pass ``f_lo`` / ``f_hi`` to restrict the ridge to the detection box. §4.12 says to take
    the argmax "over frequency", which is only correct when the capture holds one signal:
    on a four-signal scene the per-column argmax locks onto whichever emitter is loudest
    anywhere in the band, so a genuine chirp inside its own box fitted a straight line with
    R^2 = 0.002 because the ridge was tracking an unrelated constant tone 850 kHz away.
    Restricting the search to the box is the frequency-domain counterpart of the time
    slice, and the caller always knows the box.
    """
    cfg = cfg or EstimatorConfig()
    method = "spectrogram-ridge-fit/§4.12"
    x = _as_c64(x)

    def _fail(reason: str) -> ChirpParams:
        note = [reason]
        return ChirpParams(
            is_chirp=False,
            chirp_rate_hz_per_s=Estimate(None, 0.0, method, note),
            sweep_bandwidth_hz=Estimate(None, 0.0, method, list(note)),
        )

    if x.size < 256:
        return _fail("capture too short for a ridge fit")
    if spec is None:
        spec = compute_spectrogram(x, fs)
    if spec.t.size < cfg.chirp_min_cols:
        return _fail("too few spectrogram columns for a ridge fit")

    s_db = spec.S_db
    freqs = spec.f
    if f_lo is not None or f_hi is not None:
        lo = -np.inf if f_lo is None else f_lo
        hi = np.inf if f_hi is None else f_hi
        inside = (freqs >= lo) & (freqs <= hi)
        if np.count_nonzero(inside) < 4:
            return _fail("detection box spans too few frequency bins for a ridge fit")
        s_db = s_db[inside]
        freqs = freqs[inside]

    ridge_idx = np.argmax(s_db, axis=0)
    col_max = s_db.max(axis=0)
    col_med = np.median(s_db, axis=0)
    keep = (col_max - col_med) > cfg.chirp_ridge_snr_db
    if int(np.count_nonzero(keep)) < cfg.chirp_min_cols:
        return _fail(
            f"only {int(np.count_nonzero(keep))} spectrogram columns carry a clear peak"
        )

    t = spec.t[keep]
    f_ridge = freqs[ridge_idx[keep]]
    if np.ptp(t) <= 0:
        return _fail("ridge spans no time")

    # Robust fit: re-fit after discarding columns whose ridge point is far off the line.
    # Every column that survives the 6 dB test contributes, including columns where the
    # chirp has swept outside the detection box and the argmax is tracking noise instead.
    # On a real 300 kHz sweep inside a box measured at 296.9 kHz, those clipped edge
    # columns dragged the slope 26% low and held R^2 at 0.66 -- below §4.12's own 0.9
    # threshold, so a textbook chirp went unclassified. Trimming them is what makes the
    # ridge fit measure the ridge.
    inliers = np.ones(t.size, dtype=bool)
    slope = intercept = 0.0
    for _ in range(max(1, cfg.chirp_fit_rounds)):
        if int(np.count_nonzero(inliers)) < cfg.chirp_min_cols:
            inliers = np.ones(t.size, dtype=bool)
            break
        slope, intercept = np.polyfit(t[inliers], f_ridge[inliers], 1)
        residual = f_ridge - (slope * t + intercept)
        spread = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if spread <= 0:
            break
        updated = np.abs(residual) <= cfg.chirp_inlier_sigma * spread
        if int(np.count_nonzero(updated)) < cfg.chirp_min_cols or np.array_equal(updated, inliers):
            break
        inliers = updated

    t_fit, f_fit = t[inliers], f_ridge[inliers]
    slope, intercept = np.polyfit(t_fit, f_fit, 1)
    pred = slope * t_fit + intercept
    ss_tot = float(np.sum((f_fit - f_fit.mean()) ** 2))
    r2 = 1.0 - float(np.sum((f_fit - pred) ** 2)) / ss_tot if ss_tot > 0 else 0.0

    quad = np.polyfit(t_fit, f_fit, 2)
    pred2 = np.polyval(quad, t_fit)
    r2_quad = 1.0 - float(np.sum((f_fit - pred2) ** 2)) / ss_tot if ss_tot > 0 else 0.0

    duration = float(np.ptp(t_fit))
    sweep = abs(float(slope)) * duration
    sweep_frac = sweep / spec.fs if spec.fs > 0 else 0.0
    is_chirp = bool(r2 > cfg.chirp_r2 and sweep_frac > cfg.chirp_min_sweep_frac)
    nonlinear = bool(r2_quad > r2 + cfg.chirp_quadratic_margin)

    notes = [
        f"the spectrogram ridge fits a straight line with R^2 = {r2:.3f} over "
        f"{int(np.count_nonzero(inliers))} of {int(np.count_nonzero(keep))} columns"
    ]
    if nonlinear:
        notes.append(
            f"a quadratic fits better (R^2 = {r2_quad:.3f}); the sweep is nonlinear"
        )
    if not is_chirp:
        notes.append(
            f"not classified as a chirp: R^2 = {r2:.3f} (need > {cfg.chirp_r2}), "
            f"sweep covers {sweep_frac:.1%} of the band"
        )

    conf = float(np.clip((r2 - cfg.chirp_r2) / (1.0 - cfg.chirp_r2), 0.0, 1.0)) * 0.9
    return ChirpParams(
        is_chirp=is_chirp,
        chirp_rate_hz_per_s=Estimate(
            float(slope) if is_chirp else None, conf if is_chirp else 0.0, method, list(notes)
        ),
        sweep_bandwidth_hz=Estimate(
            float(sweep) if is_chirp else None, conf if is_chirp else 0.0, method, list(notes)
        ),
        r2_linear=r2,
        r2_quadratic=r2_quad,
        nonlinear=nonlinear,
    )


# --------------------------------------------------------------------------------------
# §4.13 -- Analogue modulations
# --------------------------------------------------------------------------------------


def estimate_am_depth(
    y: np.ndarray,
    *,
    cfg: EstimatorConfig | None = None,
) -> Estimate:
    """AM modulation depth from the envelope (CLAUDE.md §4.13 "AM").

    ``depth = (max_env - min_env) / (max_env + min_env)``, but taken at the 1st and 99th
    percentiles rather than the true extremes -- one noise spike otherwise sets the
    numerator. On a known-truth 60% AM tone the percentile form returns 0.599.
    """
    cfg = cfg or EstimatorConfig()
    method = "envelope-depth/§4.13"
    y = _as_c64(y)
    if y.size < 64:
        return Estimate(None, 0.0, method, ["burst too short to measure an envelope"])

    env = np.abs(y).astype(np.float64)
    lo, hi = np.percentile(env, [cfg.am_percentile, 100.0 - cfg.am_percentile])
    if hi + lo <= 0:
        return Estimate(None, 0.0, method, ["envelope has no power"])

    depth = float((hi - lo) / (hi + lo))
    variance = float(np.var(env / (np.mean(env) + 1e-30)))
    notes = [f"envelope varies with depth {depth:.2f} (normalised variance {variance:.3f})"]
    if variance < 0.005:
        notes.append(
            f"envelope is nearly constant (amplitude variance {variance:.3f}), "
            "so this is not an amplitude scheme"
        )
        return Estimate(depth, 0.3, method, notes)
    return Estimate(depth, 0.8, method, notes)


def estimate_fm_deviation(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> Estimate:
    """FM peak deviation (CLAUDE.md §4.13 "FM").

    The 99th percentile of ``|f_inst|``, not the maximum -- §4.13 is explicit that one
    glitch ruins the max. On a known-truth 5 kHz-deviation FM tone this returns 4998 Hz.

    Confidence drops when the envelope is *not* constant, since that means the burst is
    not a pure angle-modulated signal and the deviation number means less.
    """
    cfg = cfg or EstimatorConfig()
    method = "inst-freq-percentile/§4.13"
    y = _unit_power(_as_c64(y))
    if y.size < 64:
        return Estimate(None, 0.0, method, ["burst too short to measure deviation"])

    f_inst = instantaneous_frequency(y, fs_b)
    if f_inst.size < 32:
        return Estimate(None, 0.0, method, ["burst too short after differencing"])

    deviation = float(np.percentile(np.abs(f_inst), cfg.fm_deviation_percentile))
    env = np.abs(y).astype(np.float64)
    variance = float(np.var(env / (np.mean(env) + 1e-30)))
    conf = 0.85 if variance < 0.05 else 0.45
    notes = [
        f"peak frequency deviation is {deviation:.0f} Hz "
        f"(99th percentile of the instantaneous frequency)"
    ]
    if variance >= 0.05:
        notes.append(
            f"envelope is not constant (amplitude variance {variance:.3f}); "
            "this may not be a pure angle-modulated signal"
        )
    return Estimate(deviation, conf, method, notes)


def estimate_spectral_asymmetry(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> Estimate:
    """Upper/lower sideband power ratio in dB (CLAUDE.md §4.13 "SSB").

    Positive means the power sits above the centre frequency (USB), negative below (LSB).
    §4.13's threshold is 10 dB; the sign says which sideband. Returned in dB so the caller
    can apply the rule and quote the number in evidence.
    """
    cfg = cfg or EstimatorConfig()
    method = "sideband-asymmetry/§4.13"
    y = _as_c64(y)
    if y.size < 64:
        return Estimate(None, 0.0, method, ["burst too short for a sideband comparison"])

    f, p = _welch_two_sided(y, fs_b, cfg.psd_nperseg)
    upper = float(np.sum(p[f > 0]))
    lower = float(np.sum(p[f < 0]))
    if upper <= 0 or lower <= 0:
        return Estimate(None, 0.0, method, ["one sideband has no measurable power"])

    ratio_db = float(10.0 * np.log10(upper / lower))
    side = "upper (USB)" if ratio_db > 0 else "lower (LSB)"
    notes = [f"spectrum is {abs(ratio_db):.1f} dB stronger on the {side} side of centre"]
    if abs(ratio_db) < cfg.ssb_asymmetry_db:
        notes.append(
            f"asymmetry is below the {cfg.ssb_asymmetry_db:.0f} dB §4.13 threshold "
            "for calling this single-sideband"
        )
        return Estimate(ratio_db, 0.4, method, notes)
    return Estimate(ratio_db, 0.85, method, notes)


@dataclass
class CwKeying:
    """On/off keying structure, and whether it looks like Morse (§4.13 "CW / Morse")."""

    is_ook: bool
    is_morse: bool
    dot_s: Estimate
    dash_s: Estimate
    dash_dot_ratio: Estimate
    n_marks: int = 0
    wpm: float | None = None


def estimate_cw_keying(
    y: np.ndarray,
    fs_b: float,
    *,
    cfg: EstimatorConfig | None = None,
) -> CwKeying:
    """Detect on/off keying and Morse dot/dash structure (CLAUDE.md §4.13 "CW / Morse").

    Threshold a smoothed ``|y|``, measure the on-durations, and check whether they cluster
    into two groups with a ratio near 3:1 -- which is what a dot and a dash are. If they
    do, this is Morse, and §8's stretch goal can decode it.

    Words per minute follows the PARIS convention, ``wpm = 1.2 / dot_seconds``.
    """
    cfg = cfg or EstimatorConfig()
    method = "envelope-keying/§4.13"
    y = _as_c64(y)

    def _fail(reason: str) -> CwKeying:
        note = [reason]
        return CwKeying(
            is_ook=False,
            is_morse=False,
            dot_s=Estimate(None, 0.0, method, note),
            dash_s=Estimate(None, 0.0, method, list(note)),
            dash_dot_ratio=Estimate(None, 0.0, method, list(note)),
        )

    if y.size < 256 or fs_b <= 0:
        return _fail("burst too short to look for keying")

    env = np.abs(y).astype(np.float64)
    win = max(1, int(cfg.cw_smooth_frac * env.size))
    if win > 1:
        env = np.convolve(env, np.ones(win) / win, mode="same")

    lo, hi = np.percentile(env, [10.0, 90.0])
    if hi <= lo * 1.5:
        return _fail("envelope does not switch on and off; not keyed")

    on = env > 0.5 * (lo + hi)
    edges = np.diff(on.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        stops = np.r_[stops, on.size]
    n = min(starts.size, stops.size)
    if n < cfg.cw_min_marks:
        return _fail(f"only {n} keyed marks found; too few to judge Morse timing")

    durations = (stops[:n] - starts[:n]) / fs_b
    durations = durations[durations > 0]
    if durations.size < cfg.cw_min_marks:
        return _fail("keyed marks had no measurable duration")

    # split the on-durations into two groups at the geometric midpoint
    split = math.sqrt(float(durations.min()) * float(durations.max()))
    shorts = durations[durations <= split]
    longs = durations[durations > split]
    if shorts.size == 0 or longs.size == 0:
        notes = ["all keyed marks are the same length; on/off keyed but not Morse"]
        return CwKeying(
            is_ook=True,
            is_morse=False,
            dot_s=Estimate(float(np.median(durations)), 0.5, method, notes),
            dash_s=Estimate(None, 0.0, method, list(notes)),
            dash_dot_ratio=Estimate(None, 0.0, method, list(notes)),
            n_marks=int(durations.size),
        )

    dot = float(np.median(shorts))
    dash = float(np.median(longs))
    ratio = dash / dot if dot > 0 else float("inf")
    is_morse = bool(cfg.cw_dash_dot_lo <= ratio <= cfg.cw_dash_dot_hi)
    notes = [
        f"envelope is on/off keyed with {durations.size} marks; short marks average "
        f"{dot * 1000:.0f} ms and long marks {dash * 1000:.0f} ms, a ratio of {ratio:.2f}"
    ]
    if is_morse:
        notes.append("a dash/dot ratio near 3:1 is Morse timing")
    else:
        notes.append(
            f"the dash/dot ratio {ratio:.2f} is outside the "
            f"{cfg.cw_dash_dot_lo}-{cfg.cw_dash_dot_hi} band expected for Morse"
        )

    conf = 0.85 if is_morse else 0.45
    return CwKeying(
        is_ook=True,
        is_morse=is_morse,
        dot_s=Estimate(dot, conf, method, list(notes)),
        dash_s=Estimate(dash, conf, method, list(notes)),
        dash_dot_ratio=Estimate(float(ratio), conf, method, list(notes)),
        n_marks=int(durations.size),
        wpm=(1.2 / dot) if dot > 0 else None,
    )
