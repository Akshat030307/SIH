"""Acceptance test A -- estimators against known truth (CLAUDE.md §9 A).

Every row of the §9 A table is here, run against closed-form signals from
:mod:`sigscope.testsignals` at the 15 dB SNR the table specifies, plus the §9 A SNR sweep
(20, 10, 5, 0 dB) that records where each estimator breaks. The sweep tests assert the
property that actually matters at low SNR and that §9 D calls calibration: an estimator
may lose accuracy, but it must not stay confident while doing so.

The numbers the sweeps produce are tabulated by ``scripts/evaluate.py`` into
``ACCURACY.md``; these tests are the pass/fail gate, that script is the report.

Two rows are asserted against a tighter, more honest target than the §9 A wording, both
explained at the test itself: the OBW99 row (§9 A quotes a target that a *correct*
estimator cannot hit) and the cumulant row (the §5.2 table is a property of the
constellation, so it needs symbol-rate samples).
"""

from __future__ import annotations

import numpy as np
import pytest

from sigscope import testsignals as ts
from sigscope.dsp import estimators as est
from sigscope.dsp.spectrogram import compute_spectrogram
from sigscope.features.cumulants import THEORETICAL_RATIOS, cumulants
from sigscope.types import Burst

SNR_SWEEP = [20.0, 15.0, 10.0, 5.0, 0.0]
TABLE_SNR = 15.0

# Estimators whose §9 A tolerance survives all the way down to 0 dB use SNR_SWEEP. Two do
# not, and their break points are measured rather than assumed: OBW99 reads +10.3% high at
# 0 dB (against +1.9% at 5 dB) because residual in-band noise counts toward the 99%, and
# the 4-FSK tone histogram collapses from four peaks to one somewhere between 3 dB and
# 0 dB. Both are asserted accurate down to 5 dB and separately asserted to degrade
# *honestly* at 0 dB, which is the §9 D property that matters.
SNR_SWEEP_TO_5DB = [20.0, 15.0, 10.0, 5.0]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _maybe_noise(x: np.ndarray, snr_db: float | None, seed: int) -> np.ndarray:
    """Add AWGN at ``snr_db``; ``None`` means leave the signal clean."""
    return x if snr_db is None else ts.add_awgn(x, snr_db, rng=seed)


def _box(fs: float, n: int, f_lo: float, f_hi: float) -> Burst:
    """A detection box spanning the whole capture in time and ``[f_lo, f_hi]`` in frequency."""
    return Burst(t0=0.0, t1=n / fs, f_lo=f_lo, f_hi=f_hi)


def _isolate(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> est.IsolatedBurst:
    """§4.4 isolation over the whole capture, for a signal known to occupy ``[f_lo, f_hi]``."""
    return est.isolate_burst(x, fs, _box(fs, x.size, f_lo, f_hi))


def raised_cosine_obw(symbol_rate: float, beta: float, fraction: float = 0.99) -> float:
    """Closed-form occupied bandwidth of a raised-cosine spectrum.

    An RRC transmit filter gives a raised-cosine *power* spectrum: flat out to
    ``(1-beta)*Rs/2``, then a cosine roll-off to ``(1+beta)*Rs/2``, then nothing. The
    99% occupied bandwidth of that shape is strictly narrower than the absolute
    ``(1+beta)*Rs`` because the roll-off skirts carry very little power -- for beta=0.35 it
    is 11.7 kHz against an absolute 13.5 kHz, 13% narrower. This computes the true value so
    the OBW99 test can check the estimator rather than the discrepancy.
    """
    edge = (1.0 + beta) * symbol_rate / 2.0
    flat = (1.0 - beta) * symbol_rate / 2.0
    f = np.linspace(-edge, edge, 400001)
    a = np.abs(f)
    psd = np.where(
        a <= flat,
        1.0,
        0.5 * (1.0 + np.cos(np.pi * (a - flat) / (beta * symbol_rate))),
    )
    cum = np.cumsum(psd)
    cum /= cum[-1]
    tail = 0.5 * (1.0 - fraction)
    lo = f[int(np.searchsorted(cum, tail))]
    hi = f[int(np.searchsorted(cum, 1.0 - tail))]
    return float(hi - lo)


# --------------------------------------------------------------------------------------
# §9 A row 1 -- centre frequency, single tone at +137 kHz, within 1 FFT bin
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP)
def test_center_freq_tone_within_one_fft_bin(snr_db):
    fs, n, f0 = 1_000_000.0, 65536, 137_000.0
    x = _maybe_noise(ts.tone(f0, fs, n), snr_db, seed=1)

    result = est.estimate_center_freq(x, fs, _box(fs, n, f0 - 30_000.0, f0 + 30_000.0))

    bin_width = fs / est.EstimatorConfig().psd_nperseg
    assert result.peak.value is not None
    assert abs(result.peak.value - f0) <= bin_width, f"peak off by more than one bin at {snr_db} dB"
    assert result.peak.confidence > 0.5


# --------------------------------------------------------------------------------------
# §9 A row 2 -- centre frequency, QPSK at +250 kHz, within 2% of bandwidth
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP)
def test_center_freq_qpsk_within_two_percent_of_bandwidth(snr_db):
    fs, rs, beta = 2_000_000.0, 100_000.0, 0.35
    offset, bw = 250_000.0, (1.0 + beta) * rs

    baseband = ts.psk(4, rs, fs, 4000, beta=beta, rng=2)
    x = (baseband * ts.tone(offset, fs, baseband.size)).astype(np.complex64)
    x = _maybe_noise(x, snr_db, seed=3)

    result = est.estimate_center_freq(x, fs, _box(fs, x.size, offset - bw / 2, offset + bw / 2))

    assert result.centroid.value is not None
    assert abs(result.centroid.value - offset) <= 0.02 * bw


# --------------------------------------------------------------------------------------
# §9 A row 3 -- OBW99 of RRC QPSK, roll-off 0.35
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP_TO_5DB)
def test_obw99_of_rrc_qpsk(snr_db):
    """§9 A asks for OBW99 within 10% of ``(1+beta)*Rs``. That target is unreachable by a
    correct estimator: ``(1+beta)*Rs`` is the *absolute* bandwidth of the raised-cosine
    spectrum, and its 99% occupied bandwidth is 13% narrower by construction. We assert
    against the closed-form OBW99 instead, which is the quantity §4.6 actually defines,
    and separately record that the measurement lands within 15% of the §9 A figure.
    """
    fs, rs, beta = 200_000.0, 10_000.0, 0.35
    absolute_bw = (1.0 + beta) * rs
    truth = raised_cosine_obw(rs, beta)

    x = _maybe_noise(ts.psk(4, rs, fs, 8000, beta=beta, rng=4), snr_db, seed=5)
    iso = _isolate(x, fs, -absolute_bw / 2, absolute_bw / 2)
    result = est.estimate_bandwidth(iso.y, iso.fs_b, snr_db=snr_db)

    assert result.occupied_99.value is not None
    assert abs(result.occupied_99.value - truth) / truth <= 0.10, (
        f"OBW99 {result.occupied_99.value:.0f} Hz vs closed-form {truth:.0f} Hz"
    )
    assert abs(result.occupied_99.value - absolute_bw) / absolute_bw <= 0.15


def test_obw99_degrades_honestly_at_0db():
    """The measured break point. At 0 dB OBW99 reads about 10% high because the residual
    in-band noise counts toward the 99%. That is tolerable; staying confident about it is
    not, so the SNR-aware confidence has to come down with it (§9 D calibration)."""
    fs, rs, beta = 200_000.0, 10_000.0, 0.35
    absolute_bw = (1.0 + beta) * rs
    truth = raised_cosine_obw(rs, beta)

    x = ts.add_awgn(ts.psk(4, rs, fs, 8000, beta=beta, rng=4), 0.0, rng=5)
    iso = _isolate(x, fs, -absolute_bw / 2, absolute_bw / 2)

    uninformed = est.estimate_bandwidth(iso.y, iso.fs_b)
    informed = est.estimate_bandwidth(iso.y, iso.fs_b, snr_db=0.0)

    assert informed.occupied_99.value == pytest.approx(uninformed.occupied_99.value)
    assert informed.occupied_99.value > truth  # biased high, as expected
    assert informed.occupied_99.confidence < 0.5
    assert informed.occupied_99.confidence < uninformed.occupied_99.confidence
    assert any("SNR" in note for note in informed.occupied_99.notes)


def test_minus_20db_bandwidth_tracks_the_absolute_bandwidth():
    """The -20 dB width is the number that *does* approach ``(1+beta)*Rs`` -- which is why
    §4.6 insists on reporting all three bandwidths rather than just the headline one."""
    fs, rs, beta = 200_000.0, 10_000.0, 0.35
    absolute_bw = (1.0 + beta) * rs

    x = _maybe_noise(ts.psk(4, rs, fs, 8000, beta=beta, rng=6), TABLE_SNR, seed=7)
    iso = _isolate(x, fs, -absolute_bw / 2, absolute_bw / 2)
    result = est.estimate_bandwidth(iso.y, iso.fs_b)

    assert result.minus_20db.value is not None
    assert abs(result.minus_20db.value - absolute_bw) / absolute_bw <= 0.15


def test_all_three_bandwidths_are_ordered():
    """-3 dB <= OBW99 <= -20 dB, always. A violation means one of them is mis-measured."""
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=8), TABLE_SNR, seed=9)
    iso = _isolate(x, fs, -6750.0, 6750.0)
    bw = est.estimate_bandwidth(iso.y, iso.fs_b).bandwidth

    assert bw.minus_3db is not None and bw.occupied_99 is not None and bw.minus_20db is not None
    assert bw.minus_3db <= bw.occupied_99 <= bw.minus_20db


# --------------------------------------------------------------------------------------
# §9 A row 4 -- SNR estimate at a set SNR, within 2 dB
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [25.0, 20.0, 15.0, 10.0, 5.0, 0.0])
def test_snr_estimate_within_two_db(snr_db):
    """``add_awgn`` sets SNR over the whole sampled band, so the truth *inside* the burst
    box is higher by ``10*log10(fs / box_bandwidth)`` -- the box only admits that fraction
    of the noise. Both terms are known in closed form, so this is still a known-truth test.
    """
    fs, rs, beta = 200_000.0, 10_000.0, 0.35
    box_bw = (1.0 + beta) * rs

    x = ts.add_awgn(ts.psk(4, rs, fs, 20000, beta=beta, rng=10), snr_db, rng=11)
    spec = compute_spectrogram(x, fs)
    result = est.estimate_snr(spec, _box(fs, x.size, -box_bw / 2, box_bw / 2))

    in_band_truth = snr_db + 10.0 * np.log10(fs / box_bw)
    assert result.snr.value is not None
    assert abs(result.snr.value - in_band_truth) <= 2.0, (
        f"estimated {result.snr.value:.2f} dB against an in-band truth of {in_band_truth:.2f} dB"
    )


def test_snr_reports_below_floor_rather_than_nan():
    """§4.7: when the signal does not clear the in-band noise, say so with low confidence."""
    fs = 200_000.0
    rng = np.random.default_rng(12)
    noise = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)).astype(np.complex64)
    spec = compute_spectrogram(noise, fs)

    result = est.estimate_snr(spec, _box(fs, noise.size, -13_500.0 / 2, 13_500.0 / 2))

    assert result.snr.value is None or result.snr.value < 3.0
    assert result.snr.confidence < 0.5
    if result.snr.value is None:
        assert result.snr.notes


# --------------------------------------------------------------------------------------
# §9 A row 5 -- symbol rate, QPSK at 10 kBd, within 2%
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP)
def test_symbol_rate_qpsk_within_two_percent(snr_db):
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=13), snr_db, seed=14)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    result = est.estimate_symbol_rate(iso.y, iso.fs_b)

    assert result.estimate.value is not None
    assert abs(result.estimate.value - rs) / rs <= 0.02
    assert result.estimate.confidence >= 0.4


def test_symbol_rate_harmonic_check_undoes_the_octave_error():
    """§4.8's headline failure mode, reproduced and then corrected.

    On RRC QPSK the raw peak of method (b) sits at 2x the symbol rate -- its 2x line is
    genuinely stronger than its fundamental. The harmonic check has to notice that a strong
    line also exists at half that frequency and report the lower rate. Without it this
    estimator returns 20 kBd for a 10 kBd signal.
    """
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=15), TABLE_SNR, seed=16)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    corrected = est.symbol_rate_freq_transitions(iso.y, iso.fs_b)

    assert corrected.value is not None
    assert abs(corrected.value - rs) / rs <= 0.03
    assert any("harmonic check" in note for note in corrected.notes), (
        "expected the harmonic check to fire and say so in the notes"
    )


def test_symbol_rate_reconciliation_reports_every_method():
    """The reconciled answer must carry the individual opinions, so a low-confidence
    result can be explained rather than merely flagged (§4.8 reconciliation, §5.6)."""
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=17), TABLE_SNR, seed=18)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    result = est.estimate_symbol_rate(iso.y, iso.fs_b)

    assert set(result.methods) == {"cyclostationary", "inst_freq", "envelope_autocorr"}
    for estimate in result.methods.values():
        assert estimate.method
        assert 0.0 <= estimate.confidence <= 1.0
        assert estimate.value is not None or estimate.notes


def test_symbol_rate_returns_none_on_pure_noise():
    """§2 hard rule: no answer beats a wrong one."""
    fs = 40_000.0
    rng = np.random.default_rng(19)
    noise = (rng.standard_normal(40_000) + 1j * rng.standard_normal(40_000)).astype(np.complex64)

    result = est.estimate_symbol_rate(noise, fs)

    assert result.estimate.value is None or result.estimate.confidence <= 0.5


# --------------------------------------------------------------------------------------
# §9 A row 6 -- symbol rate, 2FSK at 4.8 kBd, within 3%
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP_TO_5DB)
def test_symbol_rate_2fsk_within_three_percent(snr_db):
    """2-FSK is the case that makes the three-method design necessary: the envelope is
    constant, so method (a) has no cyclic line and method (c) has nothing periodic, and
    method (b) carries the measurement alone.

    Swept to 5 dB rather than 0 dB: across eight noise seeds the 0 dB point is right on
    six, so ``scripts/evaluate.py`` records the break point at 5 dB and this gate matches
    it. Asserting 0 dB here on one lucky seed would make the suite claim more than the
    accuracy table does.
    """
    fs, rs, dev = 240_000.0, 4_800.0, 6_000.0
    x = _maybe_noise(ts.fsk(2, dev, rs, fs, 4000, rng=20), snr_db, seed=21)
    iso = _isolate(x, fs, -10_800.0, 10_800.0)

    result = est.estimate_symbol_rate(iso.y, iso.fs_b)

    assert result.estimate.value is not None
    assert abs(result.estimate.value - rs) / rs <= 0.03


@pytest.mark.parametrize("seed", [21, 22, 23, 24])
def test_symbol_rate_2fsk_is_not_confidently_wrong_at_0db(seed):
    """Below the break point the answer may be wrong, but the confidence must come down
    with it (§9 D calibration)."""
    fs, rs, dev = 240_000.0, 4_800.0, 6_000.0
    x = ts.add_awgn(ts.fsk(2, dev, rs, fs, 4000, rng=20), 0.0, rng=seed)
    iso = _isolate(x, fs, -10_800.0, 10_800.0)

    result = est.estimate_symbol_rate(iso.y, iso.fs_b)

    if result.estimate.value is None:
        assert result.estimate.notes
    elif abs(result.estimate.value - rs) / rs > 0.03:
        assert result.estimate.confidence <= 0.6, "a wrong symbol rate must not be confident"


def test_cyclostationary_abstains_on_constant_envelope():
    """§4.8: method (a) "fails on MSK/GMSK because |y|^2 is flat". It must say so, not guess."""
    fs, rs, dev = 240_000.0, 4_800.0, 6_000.0
    x = ts.fsk(2, dev, rs, fs, 4000, rng=22)  # clean: envelope is exactly constant
    iso = _isolate(x, fs, -10_800.0, 10_800.0)

    result = est.symbol_rate_cyclostationary(iso.y, iso.fs_b)

    if result.value is not None:
        assert abs(result.value - rs) / rs <= 0.03
    else:
        assert result.notes


# --------------------------------------------------------------------------------------
# §9 A row 7 -- FSK tone count and deviation, 4FSK
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP_TO_5DB)
def test_fsk_4fsk_tone_count_and_deviation(snr_db):
    """``testsignals.fsk`` places 4-FSK tones at ``[-dev, -dev/3, +dev/3, +dev]``, so the
    outer half-spread is ``dev`` and the adjacent spacing is ``2*dev/3``. §4.10 names both
    numbers "deviation" in different places, so the estimator returns them separately and
    this test checks each against its own truth.
    """
    fs, rs, dev = 240_000.0, 4_800.0, 9_600.0
    spacing = 2.0 * dev / 3.0

    x = _maybe_noise(ts.fsk(4, dev, rs, fs, 4000, rng=23), snr_db, seed=24)
    iso = _isolate(x, fs, -14_400.0, 14_400.0)
    result = est.estimate_fsk_params(iso.y, iso.fs_b, symbol_rate=rs)

    assert result.n_tones.value == 4
    assert result.deviation_hz.value is not None
    assert abs(result.deviation_hz.value - dev) / dev <= 0.10
    assert abs(result.tone_spacing_hz.value - spacing) / spacing <= 0.10


def test_fsk_4fsk_abstains_rather_than_guessing_at_0db():
    """The measured break point: between 3 dB and 0 dB the instantaneous-frequency
    histogram collapses from four peaks to one. The estimator must then say it found no
    tone alphabet, not report a wrong tone count (§2 hard rule)."""
    fs, rs, dev = 240_000.0, 4_800.0, 9_600.0
    x = ts.add_awgn(ts.fsk(4, dev, rs, fs, 4000, rng=23), 0.0, rng=24)
    iso = _isolate(x, fs, -14_400.0, 14_400.0)

    result = est.estimate_fsk_params(iso.y, iso.fs_b, symbol_rate=rs)

    if result.n_tones.value is None:
        assert result.n_tones.notes
    else:
        assert result.n_tones.value == 4 or result.n_tones.confidence < 0.8


def test_fsk_2fsk_deviation_and_modulation_index():
    fs, rs, dev = 240_000.0, 4_800.0, 6_000.0
    x = _maybe_noise(ts.fsk(2, dev, rs, fs, 4000, rng=25), TABLE_SNR, seed=26)
    iso = _isolate(x, fs, -10_800.0, 10_800.0)

    result = est.estimate_fsk_params(iso.y, iso.fs_b, symbol_rate=rs)

    assert result.n_tones.value == 2
    assert abs(result.deviation_hz.value - dev) / dev <= 0.10
    # h = tone spacing / symbol rate = 2*dev/Rs
    assert abs(result.modulation_index.value - 2.0 * dev / rs) <= 0.25


def test_fsk_abstains_on_psk():
    """A PSK burst has no FSK tone alphabet; the estimator must not invent one."""
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 4000, rng=27), TABLE_SNR, seed=28)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    result = est.estimate_fsk_params(iso.y, iso.fs_b, symbol_rate=rs)

    if result.n_tones.value is not None:
        assert result.n_tones.confidence < 0.9


# --------------------------------------------------------------------------------------
# §9 A row 8 -- OFDM symbol duration and subcarrier spacing, within 2%
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP)
def test_ofdm_symbol_duration_and_subcarrier_spacing(snr_db):
    n_sc, cp, fs = 64, 16, 64_000.0
    truth_duration = n_sc / fs
    truth_spacing = fs / n_sc

    x = _maybe_noise(ts.ofdm(n_sc, cp, 400, fs, rng=29), snr_db, seed=30)
    result = est.estimate_ofdm_params(x, fs)

    assert result.is_ofdm, f"cyclic prefix not detected at {snr_db} dB"
    assert abs(result.symbol_duration_s.value - truth_duration) / truth_duration <= 0.02
    assert abs(result.subcarrier_spacing_hz.value - truth_spacing) / truth_spacing <= 0.02


def test_ofdm_cp_length_from_peak_amplitude():
    """CP length comes from ``R*L/(1-R)`` (see :func:`estimate_ofdm_params`); at high SNR
    that should land close to the true prefix."""
    n_sc, cp, fs = 64, 16, 64_000.0
    x = _maybe_noise(ts.ofdm(n_sc, cp, 400, fs, rng=31), 20.0, seed=32)

    result = est.estimate_ofdm_params(x, fs)

    assert result.is_ofdm
    assert abs(result.cp_length.value - cp) / cp <= 0.15


def test_ofdm_does_not_fire_on_psk():
    """§5.4 lets the OFDM rule override the learned models, so a false positive here is
    expensive. A PSK burst has no cyclic prefix and must not produce one."""
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=33), TABLE_SNR, seed=34)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    assert not est.estimate_ofdm_params(iso.y, iso.fs_b).is_ofdm


def test_ofdm_does_not_fire_on_noise():
    fs = 64_000.0
    rng = np.random.default_rng(35)
    noise = (rng.standard_normal(64_000) + 1j * rng.standard_normal(64_000)).astype(np.complex64)

    assert not est.estimate_ofdm_params(noise, fs).is_ofdm


# --------------------------------------------------------------------------------------
# §9 A row 9 -- chirp rate, 1 MHz in 10 ms, within 5%
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_SWEEP)
def test_chirp_rate_within_five_percent(snr_db):
    fs, duration = 2_000_000.0, 0.01
    f0, f1 = -500_000.0, 500_000.0
    truth_rate = (f1 - f0) / duration  # 1 MHz in 10 ms = 1e8 Hz/s

    x = _maybe_noise(ts.lfm_chirp(f0, f1, duration, fs), snr_db, seed=36)
    result = est.estimate_chirp(x, fs)

    assert result.is_chirp, f"chirp not detected at {snr_db} dB"
    assert abs(result.chirp_rate_hz_per_s.value - truth_rate) / truth_rate <= 0.05
    assert abs(result.sweep_bandwidth_hz.value - (f1 - f0)) / (f1 - f0) <= 0.10


def test_chirp_does_not_fire_on_psk():
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=37), TABLE_SNR, seed=38)

    assert not est.estimate_chirp(x, fs).is_chirp


def test_nonlinear_chirp_is_flagged():
    """§4.12 asks for a degree-2 fit alongside the linear one; a quadratic sweep must be
    reported as nonlinear rather than silently fitted with a straight line."""
    fs, duration = 2_000_000.0, 0.01
    t = np.arange(int(duration * fs)) / fs
    # quadratic frequency sweep: f(t) = 1e10 * t^2 over the same span
    phase = 2 * np.pi * np.cumsum(-500_000.0 + 3.0e10 * t**2) / fs
    x = np.exp(1j * phase).astype(np.complex64)

    result = est.estimate_chirp(x, fs)

    assert result.nonlinear
    assert result.r2_quadratic > result.r2_linear


# --------------------------------------------------------------------------------------
# §9 A row 10 -- PSK order via the M-th power test, orders 2/4/8
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 4, 8])
def test_psk_order_via_mth_power(order):
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(order, rs, fs, 8000, rng=39), TABLE_SNR, seed=40)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    result = est.estimate_psk_order(iso.y, iso.fs_b)

    assert result.psk_order.value == order
    assert set(result.sharpness_db) >= {2, 4, 8}


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("snr_db", SNR_SWEEP)
def test_psk_order_is_never_confidently_wrong(order, snr_db):
    """§9 D's calibration rule, applied to the M-th power test: below about 10 dB the
    8PSK line is overtaken by its own 4th-power line and the order estimate fails. That is
    allowed. Failing *confidently* is not.
    """
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(order, rs, fs, 8000, rng=41), snr_db, seed=42)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    result = est.estimate_psk_order(iso.y, iso.fs_b)

    if result.psk_order.value is not None and result.psk_order.value != order:
        assert result.psk_order.confidence < 0.8


def test_psk_order_abstains_on_noise():
    fs = 40_000.0
    rng = np.random.default_rng(43)
    noise = (rng.standard_normal(40_000) + 1j * rng.standard_normal(40_000)).astype(np.complex64)

    result = est.estimate_psk_order(noise, fs)

    assert result.psk_order.value is None or result.psk_order.confidence < 0.6


# --------------------------------------------------------------------------------------
# §9 A row 11 -- |C40|/C21^2 for BPSK/QPSK/8PSK, within 0.15 of the §5.2 table
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "order"), [("BPSK", 2), ("QPSK", 4), ("8PSK", 8)])
def test_cumulant_ratios_match_the_table(label, order):
    """The §5.2 table is a property of the constellation, so the cumulants have to be
    measured on symbol-rate samples. Measured on the pulse-shaped waveform instead, BPSK
    reads 1.62 against a table value of 2.00 -- which would make the §5.6 evidence
    sentence quoting it plainly wrong.
    """
    fs, rs, sps = 40_000.0, 10_000.0, 4
    ratio_c40, ratio_c42 = THEORETICAL_RATIOS[label]

    x = _maybe_noise(ts.psk(order, rs, fs, 20000, rng=44), TABLE_SNR, seed=45)
    result = cumulants(x, sps=sps)

    assert abs(result.ratio_c40 - ratio_c40) <= 0.15
    assert abs(result.ratio_c42 - ratio_c42) <= 0.15
    assert result.symbol_sampled


@pytest.mark.parametrize(("label", "order"), [("BPSK", 2), ("QPSK", 4), ("8PSK", 8)])
def test_cumulant_ratios_order_the_schemes_correctly(label, order):
    """Even where the absolute value drifts with SNR, the ordering BPSK > QPSK > 8PSK on
    ``|C40|/C21^2`` is what the classifier keys on."""
    fs, rs, sps = 40_000.0, 10_000.0, 4
    ratios = {}
    for name, o in (("BPSK", 2), ("QPSK", 4), ("8PSK", 8)):
        x = _maybe_noise(ts.psk(o, rs, fs, 20000, rng=46), TABLE_SNR, seed=47)
        ratios[name] = cumulants(x, sps=sps).ratio_c40

    assert ratios["BPSK"] > ratios["QPSK"] > ratios["8PSK"]
    assert ratios["8PSK"] < 0.15


def test_cumulants_of_noise_are_near_zero():
    """§5.2's table gives Gaussian noise 0.00 / 0.00 -- the rule that lets the classifier
    recognise an empty channel."""
    rng = np.random.default_rng(48)
    noise = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)).astype(np.complex64)

    result = cumulants(noise)

    assert abs(result.ratio_c40) <= 0.15
    assert abs(result.ratio_c42) <= 0.15


# --------------------------------------------------------------------------------------
# §4.4 -- isolation, the step everything else depends on
# --------------------------------------------------------------------------------------


def test_isolation_mixes_the_burst_to_zero_and_decimates():
    fs, rs = 200_000.0, 10_000.0
    offset, bw = 40_000.0, 13_500.0

    baseband = ts.psk(4, rs, fs, 8000, rng=49)
    x = (baseband * ts.tone(offset, fs, baseband.size)).astype(np.complex64)
    iso = est.isolate_burst(x, fs, _box(fs, x.size, offset - bw / 2, offset + bw / 2))

    assert iso.f_shift_hz == pytest.approx(offset)
    assert iso.decimation == int(np.floor(fs / (2.5 * bw)))
    assert iso.fs_b == pytest.approx(fs / iso.decimation)
    assert iso.y.dtype == np.complex64

    # the isolated burst is centred: its own centre frequency is ~0, and adding the shift
    # back recovers the original offset
    centre = est.estimate_center_freq(iso.y, iso.fs_b)
    assert abs(centre.centroid.value) < 0.02 * bw
    assert iso.to_absolute(centre.centroid.value) == pytest.approx(offset, abs=0.02 * bw)


def test_isolation_preserves_burst_timing():
    """``oaconvolve(mode="same")`` keeps the burst aligned with ``t0``; an ``lfilter``
    would slide it by half the filter length and corrupt every timing measurement."""
    fs, rs = 200_000.0, 10_000.0
    quiet = np.zeros(20_000, dtype=np.complex64)
    burst = ts.psk(4, rs, fs, 2000, rng=50)
    x = np.concatenate([quiet, burst, quiet]).astype(np.complex64)

    iso = est.isolate_burst(x, fs, Burst(t0=0.0, t1=x.size / fs, f_lo=-6750.0, f_hi=6750.0))

    envelope = np.abs(iso.y)
    threshold = 0.3 * envelope.max()
    first_on = int(np.argmax(envelope > threshold))
    expected = int(quiet.size / iso.decimation)
    taps = est.EstimatorConfig().filter_taps / iso.decimation
    assert abs(first_on - expected) <= taps


def test_isolation_rejects_an_empty_slice():
    fs = 200_000.0
    x = ts.psk(4, 10_000.0, fs, 1000, rng=51)
    with pytest.raises(ValueError, match="empty slice"):
        est.isolate_burst(x, fs, Burst(t0=0.5, t1=0.5, f_lo=-1000.0, f_hi=1000.0))


def test_isolation_handles_a_full_band_box():
    """A wideband detection needs neither filtering nor decimation (§4.3 special case)."""
    fs = 200_000.0
    x = ts.psk(4, 50_000.0, fs, 4000, rng=52)
    iso = est.isolate_burst(x, fs, _box(fs, x.size, -fs / 2, fs / 2))

    assert iso.decimation == 1
    assert iso.fs_b == pytest.approx(fs)
    assert iso.notes


# --------------------------------------------------------------------------------------
# §4.13 -- analogue modulations
# --------------------------------------------------------------------------------------


def test_am_depth():
    fs, depth = 100_000.0, 0.6
    t = np.arange(int(0.2 * fs)) / fs
    x = (1.0 + depth * np.cos(2 * np.pi * 1000.0 * t)).astype(np.complex64)

    result = est.estimate_am_depth(x)

    assert result.value == pytest.approx(depth, abs=0.05)
    assert result.confidence > 0.5


def test_fm_deviation():
    fs, deviation = 100_000.0, 5_000.0
    t = np.arange(int(0.2 * fs)) / fs
    x = np.exp(2j * np.pi * np.cumsum(deviation * np.cos(2 * np.pi * 500.0 * t)) / fs)

    result = est.estimate_fm_deviation(x.astype(np.complex64), fs)

    assert result.value == pytest.approx(deviation, rel=0.05)


def test_constant_envelope_is_reported_as_such():
    """The §5.6 evidence sentence "envelope is nearly constant ... so this is not an
    amplitude scheme" has to be backed by an actual measurement."""
    fs = 96_000.0  # an integer multiple of the symbol rate, as testsignals.fsk requires
    x = ts.fsk(2, 6_000.0, 4_800.0, fs, 2000, rng=53)

    result = est.estimate_am_depth(x)

    assert result.confidence < 0.5
    assert any("nearly constant" in note for note in result.notes)


def test_ssb_asymmetry_detects_the_sideband():
    from scipy.signal import hilbert

    fs = 100_000.0
    t = np.arange(int(0.2 * fs)) / fs
    audio = np.cos(2 * np.pi * 1200.0 * t) + 0.5 * np.cos(2 * np.pi * 2400.0 * t)
    usb = hilbert(audio).astype(np.complex64)

    upper = est.estimate_spectral_asymmetry(usb, fs)
    lower = est.estimate_spectral_asymmetry(np.conj(usb), fs)

    assert upper.value > 10.0
    assert lower.value < -10.0
    assert upper.confidence > 0.5


def test_symmetric_spectrum_is_not_called_ssb():
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 4000, rng=54), TABLE_SNR, seed=55)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    result = est.estimate_spectral_asymmetry(iso.y, iso.fs_b)

    assert abs(result.value) < 10.0
    assert result.confidence < 0.5


def test_cw_morse_dot_dash_ratio():
    """§4.13: on-durations clustering into two groups with a ratio near 3:1 is Morse."""
    fs = 100_000.0
    dot = int(0.06 * fs)
    lengths = [1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 3, 1]  # alternating mark / space, in dots
    envelope = np.concatenate(
        [np.ones(n * dot) if i % 2 == 0 else np.zeros(n * dot) for i, n in enumerate(lengths)]
    )
    x = envelope.astype(np.complex64)

    result = est.estimate_cw_keying(x, fs)

    assert result.is_ook
    assert result.is_morse
    assert result.dash_dot_ratio.value == pytest.approx(3.0, abs=0.5)
    assert result.dot_s.value == pytest.approx(0.06, rel=0.15)
    assert result.wpm == pytest.approx(20.0, rel=0.2)


def test_cw_does_not_fire_on_continuous_psk():
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=56), TABLE_SNR, seed=57)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    assert not est.estimate_cw_keying(iso.y, iso.fs_b).is_morse


# --------------------------------------------------------------------------------------
# Cross-cutting contracts -- §2 "Coding rules", every estimator, every time
# --------------------------------------------------------------------------------------


def _all_estimates(iso: est.IsolatedBurst, x: np.ndarray, fs: float) -> list:
    """One of every Estimate the §4.4-§4.13 estimators can produce."""
    rate = est.estimate_symbol_rate(iso.y, iso.fs_b)
    fsk = est.estimate_fsk_params(iso.y, iso.fs_b, symbol_rate=rate.estimate.value)
    ofdm = est.estimate_ofdm_params(iso.y, iso.fs_b)
    chirp = est.estimate_chirp(x, fs)
    centre = est.estimate_center_freq(x, fs)
    bandwidth = est.estimate_bandwidth(iso.y, iso.fs_b)
    mth = est.estimate_psk_order(iso.y, iso.fs_b)
    cw = est.estimate_cw_keying(iso.y, iso.fs_b)
    return [
        rate.estimate,
        *rate.methods.values(),
        fsk.n_tones,
        fsk.deviation_hz,
        fsk.tone_spacing_hz,
        fsk.modulation_index,
        ofdm.useful_symbol_len,
        ofdm.symbol_duration_s,
        ofdm.subcarrier_spacing_hz,
        ofdm.cp_length,
        chirp.chirp_rate_hz_per_s,
        chirp.sweep_bandwidth_hz,
        centre.centroid,
        centre.peak,
        bandwidth.occupied_99,
        bandwidth.minus_3db,
        bandwidth.minus_20db,
        mth.psk_order,
        mth.carrier_offset,
        cw.dot_s,
        cw.dash_s,
        cw.dash_dot_ratio,
        est.estimate_am_depth(iso.y),
        est.estimate_fm_deviation(iso.y, iso.fs_b),
        est.estimate_spectral_asymmetry(iso.y, iso.fs_b),
    ]


@pytest.mark.parametrize("snr_db", SNR_SWEEP)
def test_every_estimate_carries_value_confidence_and_method(snr_db):
    """§2: "A number with no confidence is useless to an analyst." Also §2's hard rule --
    a ``None`` value must always be accompanied by a reason."""
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 8000, rng=58), snr_db, seed=59)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    for estimate in _all_estimates(iso, x, fs):
        assert estimate.method, "every estimate must name its method"
        assert 0.0 <= estimate.confidence <= 1.0
        if estimate.value is None:
            assert estimate.notes, f"{estimate.method} returned None without a reason"
        else:
            assert np.isfinite(estimate.value)


def test_estimators_survive_pure_noise_without_raising():
    """§9 B's spirit, applied to Stage 4: a hostile input gives a clean answer, never a
    traceback."""
    fs = 200_000.0
    rng = np.random.default_rng(60)
    noise = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)).astype(np.complex64)
    iso = _isolate(noise, fs, -6750.0, 6750.0)

    for estimate in _all_estimates(iso, noise, fs):
        assert 0.0 <= estimate.confidence <= 1.0
        if estimate.value is None:
            assert estimate.notes


def test_estimators_survive_a_very_short_burst():
    fs = 200_000.0
    x = ts.psk(4, 10_000.0, fs, 12, rng=61)
    iso = est.isolate_burst(x, fs, _box(fs, x.size, -6750.0, 6750.0))

    for estimate in _all_estimates(iso, x, fs):
        assert 0.0 <= estimate.confidence <= 1.0
        if estimate.value is None:
            assert estimate.notes


def test_estimators_are_deterministic():
    """§2: pure functions. Same input, same output, every time."""
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 4000, rng=62), TABLE_SNR, seed=63)
    iso = _isolate(x, fs, -6750.0, 6750.0)

    first = est.estimate_symbol_rate(iso.y, iso.fs_b).estimate
    second = est.estimate_symbol_rate(iso.y, iso.fs_b).estimate

    assert first.value == second.value
    assert first.confidence == second.confidence


def test_estimators_do_not_mutate_their_input():
    fs, rs = 200_000.0, 10_000.0
    x = _maybe_noise(ts.psk(4, rs, fs, 4000, rng=64), TABLE_SNR, seed=65)
    iso = _isolate(x, fs, -6750.0, 6750.0)
    before = iso.y.copy()

    _all_estimates(iso, x, fs)

    assert np.array_equal(iso.y, before)
