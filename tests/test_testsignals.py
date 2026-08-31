"""Known-truth checks for sigscope.testsignals (CLAUDE.md §6.3).

These are the closed-form signals the DSP estimators (Phase 4) will be tested against,
so the generators themselves must be right.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigscope import testsignals as ts


def _dominant_freq(x: np.ndarray, fs: float) -> float:
    X = np.fft.fftshift(np.abs(np.fft.fft(x)))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))
    return f[np.argmax(X)]


def test_tone_frequency_within_one_bin():
    fs, n, f0 = 1_000_000.0, 4096, 137_000.0
    x = ts.tone(f0, fs, n)
    assert x.dtype == np.complex64
    assert abs(_dominant_freq(x, fs) - f0) <= fs / n


def test_psk_phase_states_via_mth_power():
    """Raising M-PSK to the M-th power collapses it to one phase -> a line at DC (§4.9)."""
    fs, rs, n = 200_000.0, 10_000.0, 8000
    for order in (2, 4, 8):
        y = ts.psk(order, rs, fs, n, rng=0)
        assert y.dtype == np.complex64
        spectrum = np.abs(np.fft.fft(y**order))
        assert np.argmax(spectrum) == 0  # no carrier offset -> line at bin 0
        assert spectrum.max() / np.median(spectrum) > 1_000  # one sharp line


def test_psk_cumulant_ratio_orders_by_scheme():
    """|C40|/C21^2 falls BPSK(2.0) > QPSK(1.0) > 8PSK(0.0) -- the §5.2 table order."""
    fs, rs, n = 200_000.0, 10_000.0, 16000

    def c40_ratio(order: int) -> float:
        y = ts.psk(order, rs, fs, n, rng=1)
        y = y / np.sqrt(np.mean(np.abs(y) ** 2))
        m20, m40 = np.mean(y**2), np.mean(y**4)
        c40 = m40 - 3 * m20**2
        return float(abs(c40) / np.mean(np.abs(y) ** 2) ** 2)

    r2, r4, r8 = c40_ratio(2), c40_ratio(4), c40_ratio(8)
    assert r2 > r4 > r8
    assert r8 < 0.15  # 8PSK theoretical value is 0.00
    assert r2 > 1.3  # BPSK clearly above QPSK


def test_fsk_2fsk_tones_at_plus_minus_deviation():
    fs, rs, dev = 240_000.0, 4_800.0, 6_000.0
    x = ts.fsk(2, dev, rs, fs, 3000, rng=1)
    f_inst = np.diff(np.unwrap(np.angle(x))) * fs / (2 * np.pi)
    assert abs(np.mean(f_inst[f_inst > 0]) - dev) < 0.1 * dev
    assert abs(np.mean(f_inst[f_inst < 0]) + dev) < 0.1 * dev


def test_fsk_4fsk_has_four_levels():
    x = ts.fsk(4, 9_600.0, 4_800.0, 240_000.0, 4000, rng=2)
    f_inst = np.diff(np.unwrap(np.angle(x))) * 240_000.0 / (2 * np.pi)
    hist, edges = np.histogram(f_inst, bins=200)
    centres = 0.5 * (edges[:-1] + edges[1:])
    peaks = centres[hist > 0.2 * hist.max()]
    # cluster the peak centres; expect 4 well-separated groups
    groups = np.split(peaks, np.where(np.diff(peaks) > 1000)[0] + 1)
    assert len(groups) == 4


def test_lfm_chirp_rate_within_one_percent():
    f0, f1, dur, fs = -400_000.0, 400_000.0, 0.01, 2_000_000.0
    x = ts.lfm_chirp(f0, f1, dur, fs)
    phi = np.unwrap(np.angle(x))
    f_inst = np.diff(phi) * fs / (2 * np.pi)
    t = np.arange(len(f_inst)) / fs
    slope = np.polyfit(t, f_inst, 1)[0]
    assert abs(slope - (f1 - f0) / dur) / ((f1 - f0) / dur) < 0.01


def test_ofdm_autocorrelation_spikes_at_useful_symbol_length():
    n_sc, cp = 64, 16
    x = ts.ofdm(n_sc, cp, 300, 1.0, rng=3)
    lags = np.arange(1, 2 * n_sc + 1)
    denom = np.vdot(x, x).real
    r = np.array([abs(np.vdot(x[:-lag], x[lag:])) / denom for lag in lags])
    assert lags[np.argmax(r)] == n_sc  # cyclic prefix repeats at the useful symbol length
    assert r.max() > 10 * np.median(r)  # a clear peak
    # §4.11 global normalisation puts the peak near cp/(n_sc+cp)
    assert r.max() == pytest.approx(cp / (n_sc + cp), abs=0.06)


def test_add_awgn_hits_target_snr():
    x = ts.tone(50_000.0, 1_000_000.0, 200_000)
    y = ts.add_awgn(x, 10.0, rng=4)
    noise = y - x
    snr = 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(noise) ** 2))
    assert abs(snr - 10.0) < 0.5


def test_generators_are_deterministic_with_seed():
    a = ts.psk(2, 10_000.0, 200_000.0, 500, rng=7)
    b = ts.psk(2, 10_000.0, 200_000.0, 500, rng=7)
    assert np.array_equal(a, b)
