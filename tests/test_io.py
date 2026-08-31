"""Stage 1 ingest tests -- CLAUDE.md §9 acceptance test B and the §8 Phase 2 checkpoint."""

from __future__ import annotations

import json
import struct
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sigscope import testsignals as ts
from sigscope.io import CaptureError, iter_blocks, read_capture

# --------------------------------------------------------------------------- helpers


def _interleave(x: np.ndarray) -> np.ndarray:
    out = np.empty(2 * x.size, dtype=np.float64)
    out[0::2], out[1::2] = x.real, x.imag
    return out


def _write_raw(path: Path, x: np.ndarray, dtype: str, fill: float = 0.6) -> Path:
    inter = _interleave(x / np.sqrt(np.mean(np.abs(x) ** 2)))
    if dtype == "int8":
        buf = np.clip(inter * fill * 128, -128, 127).astype("<i1")
    elif dtype == "int16":
        buf = np.clip(inter * fill * 32768, -32768, 32767).astype("<i2")
    else:
        buf = (inter * fill).astype("<f4")
    path.write_bytes(buf.tobytes())
    return path


def _write_wav_with_auxi(path: Path, x: np.ndarray, fs: int, center_hz: int) -> Path:
    pcm = np.clip(_interleave(x / np.max(np.abs(x))) * 32767, -32768, 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 2, fs, fs * 2 * 2, 2 * 2, 16)
    systime = struct.pack("<8H", 2021, 2, 0, 28, 14, 40, 30, 0)
    auxi = systime + systime + struct.pack("<II", center_hz, fs)
    body = b"WAVE"
    body += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"auxi" + struct.pack("<I", len(auxi)) + auxi
    body += b"data" + struct.pack("<I", len(pcm)) + pcm
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def _write_sigmf(base: Path, x: np.ndarray, fs: int, fc: int) -> Path:
    data = np.clip(_interleave(x / np.max(np.abs(x))) * 32767, -32768, 32767).astype("<i2")
    (base.with_suffix(".sigmf-data")).write_bytes(data.tobytes())
    meta = {
        "global": {"core:datatype": "ci16_le", "core:sample_rate": fs, "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0, "core:frequency": fc,
                      "core:datetime": "2021-02-28T14:40:30Z"}],
        "annotations": [],
    }
    (base.with_suffix(".sigmf-meta")).write_text(json.dumps(meta))
    return base.with_suffix(".sigmf-data")


# --------------------------------------------------------------------------- WAV


def test_stereo_wav_roundtrips_to_1e6(tmp_path):
    fs = 48_000
    x = ts.tone(1234.0, fs, 10_000).astype(np.complex64)
    sf.write(tmp_path / "iq.wav", np.stack([x.real, x.imag], axis=1), fs, subtype="FLOAT")

    iq, meta = read_capture(tmp_path / "iq.wav")
    assert iq.dtype == np.complex64
    assert meta.iq_from_stereo is True
    assert meta.sample_rate == fs
    assert meta.source_format == "wav-iq"
    assert np.max(np.abs(iq[: x.size] - x)) < 1e-6


def test_mono_real_wav_converts_without_crashing(tmp_path):
    fs = 8_000
    audio = np.sin(2 * np.pi * 440 * np.arange(fs) / fs).astype(np.float32)
    sf.write(tmp_path / "audio.wav", audio, fs, subtype="FLOAT")

    iq, meta = read_capture(tmp_path / "audio.wav")
    assert iq.dtype == np.complex64
    assert iq.size == fs
    assert meta.iq_from_stereo is False
    assert any("Hilbert" in n for n in meta.notes)
    # analytic signal of a pure tone has near-constant envelope
    assert np.std(np.abs(iq[200:-200])) < 0.05


def test_auxi_chunk_gives_centre_freq_and_time(tmp_path):
    fs = 96_000
    x = ts.tone(2000.0, fs, 4_000)
    _write_wav_with_auxi(tmp_path / "HDSDR_cap.wav", x, fs, center_hz=7_100_000)

    _iq, meta = read_capture(tmp_path / "HDSDR_cap.wav")
    assert meta.center_freq == 7_100_000
    assert meta.center_freq_source == "auxi"
    assert meta.start_time == "2021-02-28T14:40:30Z"


# --------------------------------------------------------------------------- headerless raw


@pytest.mark.parametrize("dtype", ["int8", "int16", "float32"])
def test_headerless_dtype_guessed_over_90_percent(dtype, tmp_path):
    rng = np.random.default_rng(20260831)
    hits = 0
    trials = 30
    for k in range(trials):
        order = int(rng.choice([2, 4, 8]))
        rs = float(rng.choice([20e3, 50e3, 100e3]))
        x = ts.psk(order, rs, 1e6, 4000, rng=rng)
        x = ts.add_awgn(x, float(rng.uniform(8, 22)), rng=rng)
        p = _write_raw(tmp_path / f"{dtype}_{k}.bin", x, dtype, fill=float(rng.uniform(0.2, 0.9)))
        _iq, meta = read_capture(p, fs=1e6)
        hits += meta.dtype_guessed == dtype
    assert hits / trials >= 0.9
    assert set(meta.dtype_candidates) == {"int8", "int16", "float32"}


def test_headerless_int16_roundtrip_within_quantisation(tmp_path):
    """§8 Phase 2 checkpoint: write int16, read back with dtype guessed, samples match."""
    fs = 1e6
    x = ts.psk(4, 50e3, fs, 4000, rng=0)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    p = tmp_path / "capture_1000000sps.iq"
    _write_raw(p, x, "int16", fill=0.6)

    iq, meta = read_capture(p)
    assert meta.dtype_guessed == "int16"
    assert meta.sample_rate == 1e6  # recovered from the _1000000sps token
    recovered = iq[: x.size] / 0.6  # undo the fill applied by the test writer
    assert np.max(np.abs(recovered - x)) < 4.0 / 32768.0  # a few LSB: within quantisation


def test_cs16_extension_pins_dtype(tmp_path):
    x = ts.tone(1000.0, 1e6, 5000)
    p = _write_raw(tmp_path / "grab.cs16", x, "int16")
    _iq, meta = read_capture(p)
    assert meta.dtype_guessed == "int16"
    assert meta.dtype_confidence >= 0.9
    assert any(".cs16" in n or "cs16" in n for n in meta.notes)


def test_dtype_override_beats_guess(tmp_path):
    x = ts.tone(1000.0, 1e6, 5000)
    p = _write_raw(tmp_path / "grab.iq", x, "int16")
    _iq, meta = read_capture(p, dtype="int16", fs=1e6)
    assert meta.dtype_guessed == "int16"
    assert meta.dtype_confidence == 1.0


# --------------------------------------------------------------------------- SigMF


def test_sigmf_pair_read_exactly_no_guessing(tmp_path):
    fs, fc = 2_000_000, 100_000_000
    x = ts.psk(4, 200e3, fs, 2000, rng=1)
    data_path = _write_sigmf(tmp_path / "rec", x, fs, fc)

    for target in (data_path, tmp_path / "rec.sigmf-meta"):
        iq, meta = read_capture(target)
        assert meta.source_format == "sigmf"
        assert meta.sample_rate == fs
        assert meta.center_freq == fc
        assert meta.center_freq_source == "sigmf"
        assert meta.metadata_confidence == 1.0
        assert meta.dtype_confidence == 1.0
        assert meta.frequencies_are_normalised is False
        ref = (x / np.max(np.abs(x))).astype(np.complex64)
        assert np.max(np.abs(iq[: x.size] - ref)) < 4.0 / 32768.0


# --------------------------------------------------------------------------- filename parsing


def test_sdrsharp_filename_parsed(tmp_path):
    x = ts.tone(1000.0, 1e6, 4000)
    name = "SDRSharp_20210228_144030Z_100000000Hz_IQ.wav"
    sf.write(tmp_path / name, np.stack([x.real, x.imag], axis=1), 1_000_000, subtype="FLOAT")

    _iq, meta = read_capture(tmp_path / name)
    assert meta.center_freq == 100_000_000
    assert meta.center_freq_source == "filename"
    assert meta.start_time == "2021-02-28T14:40:30Z"


def test_hdsdr_filename_khz_parsed(tmp_path):
    x = ts.tone(10.0, 8000, 4000)
    p = _write_raw(tmp_path / "HDSDR_20180101_000000Z_7100kHz_RF.iq", x, "int16")
    _iq, meta = read_capture(p, fs=8000)
    assert meta.center_freq == 7_100_000


def test_fc_and_sps_tokens_parsed(tmp_path):
    x = ts.tone(1000.0, 1e6, 5000)
    p = _write_raw(tmp_path / "grab_2000000sps_fc433920000Hz.iq", x, "int16")
    _iq, meta = read_capture(p)
    assert meta.sample_rate == 2_000_000
    assert meta.center_freq == 433_920_000
    assert meta.frequencies_are_normalised is False


# --------------------------------------------------------------------------- unknown sample rate


def test_unknown_sample_rate_is_normalised_and_propagates(tmp_path):
    x = ts.psk(4, 50e3, 1e6, 4000, rng=2)
    p = _write_raw(tmp_path / "mystery.iq", x, "int16")

    _iq, meta = read_capture(p)
    assert meta.sample_rate == 1.0
    assert meta.frequencies_are_normalised is True
    assert meta.center_freq is None
    assert meta.to_capture_dict()["frequencies_are_normalised"] is True
    assert any("normalised" in n for n in meta.notes)


# --------------------------------------------------------------------------- malformed inputs


def test_empty_file_gives_clean_message(tmp_path):
    p = tmp_path / "empty.iq"
    p.write_bytes(b"")
    with pytest.raises(CaptureError, match="empty"):
        read_capture(p)


def test_text_file_renamed_iq_gives_clean_message(tmp_path):
    p = tmp_path / "notes.iq"
    p.write_text("this is prose, not a radio recording, not IQ samples at all.\n" * 200)
    with pytest.raises(CaptureError, match="does not look like raw"):
        read_capture(p)


def test_missing_file_gives_clean_message(tmp_path):
    with pytest.raises(CaptureError, match="no such file"):
        read_capture(tmp_path / "nope.iq")


# --------------------------------------------------------------------------- block streaming


def test_large_file_streamed_in_bounded_memory(tmp_path):
    """§9 B: a big file is processed in blocks, peak memory well under the file size."""
    fs = 4_000_000
    x = np.tile(ts.psk(4, 500e3, fs, 40_000, rng=3), 12)  # ~3 M samples -> ~12 MB as cs16
    p = _write_raw(tmp_path / f"scene_{fs}sps.cs16", x, "int16")
    file_bytes = p.stat().st_size
    block_samples = 1 << 16

    full, _ = read_capture(p)  # reference (eager path)

    # de-overlapped reconstruction, one block at a time (nothing retained)
    step = int(block_samples * 0.75)
    meta, blocks = iter_blocks(p, block_samples=block_samples, overlap=0.25)
    out = np.empty(meta.n_samples, dtype=np.complex64)
    pos = 0
    dtypes_ok = True
    for blk in blocks:
        dtypes_ok &= blk.dtype == np.complex64
        take = min(step, meta.n_samples - pos) if pos + len(blk) < meta.n_samples else len(blk)
        out[pos : pos + take] = blk[:take]
        pos += take
    assert meta.n_samples == full.size
    assert np.array_equal(out, full)
    assert dtypes_ok

    # peak heap during a streamed pass stays near one block, not the whole capture
    tracemalloc.start()
    meta, blocks = iter_blocks(p, block_samples=block_samples, overlap=0.25)
    acc = np.complex128(0)
    for blk in blocks:
        acc += blk.sum()
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < file_bytes // 2  # memory does not track the file size
    assert peak < 64 * block_samples  # a handful of block-sized buffers, not the whole capture
