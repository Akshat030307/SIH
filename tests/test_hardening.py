"""Hardening regressions (CLAUDE.md §9 B, §9 E, §9 F, §8 Phase 8).

``scripts/hardening.py`` runs the full acceptance sweep, including the cases that cost
minutes and gigabytes. This file keeps the fast, load-bearing ones in the suite so they
cannot regress silently:

* analysis completes with **every socket blocked**, which is how §9 E's "works identically
  with wifi disabled" is proved rather than asserted;
* the streaming path finds what the whole-file path finds, and stitches detections split
  by a segment seam;
* the project installs and runs without a downloadable build backend.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from sigscope import testsignals as ts
from sigscope.pipeline import (
    SEGMENT_SAMPLES,
    STREAM_THRESHOLD_SAMPLES,
    AnalysisConfig,
    _estimate_sample_count,
    _merge_across_seams,
    analyse_file,
    analyse_streaming,
)
from sigscope.types import Bandwidth, Detection, Modulation

REPO = Path(__file__).resolve().parent.parent
FS = 1_000_000.0


def _scene(path: Path, seconds: float, fs: float = FS) -> Path:
    """A headerless int16 capture with a steady tone and a recurring QPSK burst."""
    rng = np.random.default_rng(4)
    n = int(fs * seconds)
    canvas = (
        np.sqrt(5e-4) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ).astype(np.complex64)

    def place(signal, f_offset, t0, amplitude):
        n0 = int(t0 * fs)
        n1 = min(n, n0 + len(signal))
        if n1 <= n0:
            return
        t = np.arange(n0, n1) / fs
        canvas[n0:n1] += (
            amplitude * signal[: n1 - n0] * np.exp(2j * np.pi * f_offset * t)
        ).astype(np.complex64)

    for k in range(max(1, int(seconds))):
        place(ts.psk(4, 25_000.0, fs, 4000, rng=k), +200_000.0, 0.1 + k, 0.9)
    place(ts.tone(0.0, fs, int(0.9 * seconds * fs)), -300_000.0, 0.02, 0.4)
    canvas /= np.max(np.abs(canvas)) * 1.05

    interleaved = np.empty(2 * n, dtype=np.int16)
    interleaved[0::2] = np.clip(canvas.real * 32767, -32768, 32767).astype(np.int16)
    interleaved[1::2] = np.clip(canvas.imag * 32767, -32768, 32767).astype(np.int16)
    interleaved.tofile(path)
    return path


# --------------------------------------------------------------------------------------
# §9 E -- offline
# --------------------------------------------------------------------------------------


def test_analysis_completes_with_every_socket_blocked(tmp_path, monkeypatch):
    """§9 E: "Everything works identically with wifi disabled."

    Proved by making every outbound socket raise for the duration of the analysis. Any
    code path that reached for a network -- a model download, a font, a telemetry ping --
    would fail loudly right here instead of on the demo laptop.
    """
    path = _scene(tmp_path / "offline_1000000sps.iq", 1.0)

    def refuse(*args, **kwargs):
        raise OSError("network access is disabled for this test")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    analysis = analyse_file(path, fs=FS)

    assert analysis.report.detections, "the capture holds two signals"
    assert all(d.modulation is not None for d in analysis.report.detections)


def test_report_writers_work_offline(tmp_path, monkeypatch):
    """The three §3 outputs must also not need a network (the SigMF writer imports sigmf)."""
    from sigscope.report import write_all

    analysis = analyse_file(_scene(tmp_path / "cap_1000000sps.iq", 1.0), fs=FS)

    def refuse(*args, **kwargs):
        raise OSError("network access is disabled for this test")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    written = write_all(
        analysis.report, tmp_path / "out",
        spectrogram=analysis.spectrogram, bursts=analysis.bursts,
    )
    for kind in ("json", "sigmf", "html"):
        assert written[kind].is_file() and written[kind].stat().st_size > 0


# --------------------------------------------------------------------------------------
# §9 B -- large captures stream
# --------------------------------------------------------------------------------------


def test_sample_count_is_estimated_without_reading_the_samples(tmp_path):
    """The whole-file / streaming decision has to be made *before* allocating anything."""
    path = _scene(tmp_path / "sized_1000000sps.iq", 0.5)
    estimated = _estimate_sample_count(path)
    assert estimated == pytest.approx(int(0.5 * FS), rel=0.01)


def test_streaming_finds_what_the_whole_file_path_finds(tmp_path):
    """Segmenting must not change the answer, only the memory it takes to get there."""
    path = _scene(tmp_path / "compare_1000000sps.iq", 3.0)

    whole = analyse_file(path, fs=FS).report
    streamed = analyse_streaming(path, fs=FS, segment_samples=1 << 19).report

    assert whole.detections and streamed.detections
    # the steady tone spans the whole capture and must survive being cut into segments
    def has_tone(report):
        return any(
            abs((d.center_freq_offset_hz or 0.0) + 300_000.0) < 30_000.0
            for d in report.detections
        )

    assert has_tone(whole), "the whole-file path should find the tone"
    assert has_tone(streamed), "the streaming path lost the tone at a seam"


def test_streaming_says_how_it_segmented(tmp_path):
    """§2 Stage 2: an analyst has to know the capture was cut up, because a merged
    detection's parameters come from one segment rather than the whole burst."""
    report = analyse_streaming(
        _scene(tmp_path / "seg_1000000sps.iq", 2.0), fs=FS, segment_samples=1 << 19
    ).report
    assert any("segments" in w for w in report.warnings)


def test_streaming_threshold_is_below_a_gigabyte_of_samples():
    """A capture that would need more RAM than §9 B allows must route to streaming."""
    # complex64 is 8 bytes, and the pipeline peaks at roughly 15x the sample array
    assert STREAM_THRESHOLD_SAMPLES * 8 * 15 < 2 * 1024**4
    assert SEGMENT_SAMPLES <= STREAM_THRESHOLD_SAMPLES


def test_streamed_overview_spectrogram_has_a_monotonic_time_axis(tmp_path):
    """Segments overlap, so their pooled slices must be trimmed before concatenation.

    The PNG writer maps columns linearly from ``t[0]`` to ``t[-1]``; a non-monotonic axis
    would draw each overlap twice and put every detection box after the first seam at the
    wrong x.
    """
    analysis = analyse_streaming(
        _scene(tmp_path / "overview_1000000sps.iq", 3.0), fs=FS, segment_samples=1 << 19
    )
    assert analysis.spectrogram is not None
    times = analysis.spectrogram.t
    assert np.all(np.diff(times) > 0), "the overview time axis must increase"
    assert times[0] >= 0.0


def test_reports_write_from_a_streamed_analysis(tmp_path):
    """The streaming path returns a pooled overview and ``noise=None``; all three §3
    writers have to cope with both."""
    from sigscope.report import write_all

    analysis = analyse_streaming(
        _scene(tmp_path / "written_1000000sps.iq", 2.0), fs=FS, segment_samples=1 << 19
    )
    written = write_all(
        analysis.report, tmp_path / "out",
        spectrogram=analysis.spectrogram, bursts=analysis.bursts,
    )
    for kind in ("json", "sigmf", "html"):
        assert written[kind].is_file() and written[kind].stat().st_size > 0


def test_warning_list_is_folded_rather_than_unbounded():
    """A long streamed capture raises per-detection warnings in every segment; the report
    keeps every *kind* visible with a count instead of thousands of lines."""
    from sigscope.pipeline import _fold_warnings

    warnings = [f"detection {i}: symbol rate methods disagree" for i in range(500)]
    folded = _fold_warnings(warnings, limit=10)
    assert len(folded) == 11
    assert "490 further warning(s)" in folded[-1]
    assert "symbol rate methods disagree" in folded[-1]


def _detection(index, t0, t1, centre, bandwidth, snr):
    return Detection(
        id=index,
        time_start_s=t0,
        time_stop_s=t1,
        duration_s=t1 - t0,
        center_freq_hz=None,
        center_freq_offset_hz=centre,
        bandwidth_hz=Bandwidth(occupied_99=bandwidth),
        snr_db=snr,
        power_dbfs=-20.0,
        modulation=Modulation(label="unclassified", confidence=0.0, evidence=["a", "b"]),
    )


def test_seam_merge_joins_a_split_transmission():
    warnings: list[str] = []
    merged = _merge_across_seams(
        [
            _detection(1, 0.0, 1.0, 200_000.0, 30_000.0, 20.0),
            _detection(2, 1.001, 2.0, 200_000.0, 30_000.0, 25.0),
        ],
        gap_s=0.01,
        warnings=warnings,
    )
    assert len(merged) == 1
    assert merged[0].time_start_s == 0.0
    assert merged[0].time_stop_s == 2.0
    assert merged[0].snr_db == 25.0, "the stronger segment supplies the parameters"
    assert merged[0].extra["crossed_segment_boundary"] is True
    assert warnings


def test_seam_merge_says_so_in_the_evidence():
    """A parameter measured on half a burst must not be presented as a whole-burst one."""
    merged = _merge_across_seams(
        [
            _detection(1, 0.0, 1.0, 200_000.0, 30_000.0, 20.0),
            _detection(2, 1.001, 2.0, 200_000.0, 30_000.0, 25.0),
        ],
        gap_s=0.01,
        warnings=[],
    )
    assert any("segment boundary" in s for s in merged[0].modulation.evidence)


def test_seam_merge_keeps_separate_signals_apart():
    """Two transmissions at different frequencies are not one signal, however adjacent."""
    merged = _merge_across_seams(
        [
            _detection(1, 0.0, 1.0, 200_000.0, 20_000.0, 20.0),
            _detection(2, 1.001, 2.0, -400_000.0, 20_000.0, 20.0),
        ],
        gap_s=0.01,
        warnings=[],
    )
    assert len(merged) == 2


def test_seam_merge_does_not_join_a_long_silence():
    merged = _merge_across_seams(
        [
            _detection(1, 0.0, 1.0, 200_000.0, 30_000.0, 20.0),
            _detection(2, 5.0, 6.0, 200_000.0, 30_000.0, 20.0),
        ],
        gap_s=0.01,
        warnings=[],
    )
    assert len(merged) == 2


# --------------------------------------------------------------------------------------
# §9 E -- a fresh clone installs and runs
# --------------------------------------------------------------------------------------


def test_build_backend_is_installable_offline():
    """§9 E wants a fresh clone working on a clean machine; §10 runs it with wifi off.

    hatchling is not part of any standard Python install, so ``pip install -e .`` on an
    air-gapped laptop fails at the build-backend step before it reads a line of our code.
    setuptools ships with pip.
    """
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    requires = " ".join(config["build-system"]["requires"]).lower()
    assert "setuptools" in requires
    assert "hatchling" not in requires


def test_module_entry_point_runs_without_installation():
    """``python -m sigscope`` is the zero-install path, and the fallback if the install
    step is ever the thing that breaks on stage."""
    result = subprocess.run(
        [sys.executable, "-m", "sigscope", "--help"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "analyse" in result.stdout and "serve" in result.stdout


def test_runtime_dependencies_are_all_importable():
    """A dependency that is declared but missing turns into a traceback mid-demo."""
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    modules = {
        "numpy": "numpy", "scipy": "scipy", "soundfile": "soundfile", "sigmf": "sigmf",
        "scikit-learn": "sklearn", "torch": "torch", "fastapi": "fastapi",
        "uvicorn": "uvicorn", "pandas": "pandas", "pyarrow": "pyarrow",
    }
    declared = {d.split(">")[0].split("=")[0].strip() for d in config["project"]["dependencies"]}
    import importlib

    for name, module in modules.items():
        if name in declared:
            assert importlib.import_module(module) is not None


# --------------------------------------------------------------------------------------
# §9 F -- the judge test
# --------------------------------------------------------------------------------------


def test_judge_test_unseen_headerless_capture(tmp_path):
    """§9 F: an unseen capture, no metadata, five minutes.

    The name carries no rate and no centre frequency, so the reader has to guess the
    sample format from the histogram and then report every frequency as a fraction of the
    sample rate. The one thing it must never do is print a fabricated Hz value.
    """
    fs = 1_234_567.0  # not a round number, and recorded nowhere
    rng = np.random.default_rng(99)
    n = int(fs * 2)
    canvas = (
        np.sqrt(6e-4) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ).astype(np.complex64)

    def place(signal, f_offset, t0, amplitude):
        n0 = int(t0 * fs)
        n1 = min(n, n0 + len(signal))
        t = np.arange(n0, n1) / fs
        canvas[n0:n1] += (
            amplitude * signal[: n1 - n0] * np.exp(2j * np.pi * f_offset * t)
        ).astype(np.complex64)

    place(ts.fsk(4, 12_000.0, 6_000.0, 1_200_000.0, 6000, rng=1), -300_000.0, 0.15, 0.85)
    # not OFDM here: testsignals.ofdm has one sample per subcarrier, so it occupies
    # the entire band whatever fs is, and would swallow the FSK rather than sit
    # beside it. The OFDM rule is exercised on its own in tests/test_models.py.
    place(ts.psk(4, 40_000.0, 1_200_000.0, 8000, rng=2), +250_000.0, 0.4, 0.75)
    canvas /= np.max(np.abs(canvas)) * 1.05

    path = tmp_path / "UNKNOWN_HANDOVER.bin"
    interleaved = np.empty(2 * n, dtype=np.int16)
    interleaved[0::2] = np.clip(canvas.real * 32767, -32768, 32767).astype(np.int16)
    interleaved[1::2] = np.clip(canvas.imag * 32767, -32768, 32767).astype(np.int16)
    interleaved.tofile(path)

    report = analyse_file(path, cfg=AnalysisConfig()).report

    assert len(report.detections) >= 2, "two signals were placed"
    # §3: never a fabricated Hz value
    assert report.capture.frequencies_are_normalised or report.capture.center_freq_hz is None
    assert any("normalised" in w or "unknown" in w for w in report.warnings)
    # §3: every detection carries at least two evidence sentences
    for detection in report.detections:
        assert detection.modulation is not None
        assert len(detection.modulation.evidence) >= 2
