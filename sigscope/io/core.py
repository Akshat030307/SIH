"""Stage 1 orchestration: sniff the format, dispatch, build an honest ``CaptureMeta``.

Sniffing order (CLAUDE.md §3 Stage 1):
  1. a ``.sigmf-meta`` sidecar  -> trust it completely
  2. ``.wav`` / ``.wave``       -> stereo = IQ, mono = real audio (Hilbert)
  3. anything else              -> headerless raw IQ (extension or histogram heuristic)

Every guess is recorded in ``meta.notes`` with a confidence. When no sample rate can be
found anywhere, ``sample_rate`` is set to ``1.0`` and ``frequencies_are_normalised`` is
set true so the pipeline reports fractions of the sample rate, never fabricated Hz.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from sigscope.io.errors import CaptureError
from sigscope.io.filenames import parse_filename
from sigscope.io.raw import iter_raw_blocks, read_raw
from sigscope.io.sigmf_sidecar import find_sidecar, read_sigmf
from sigscope.io.wav import read_wav
from sigscope.types import CaptureMeta

_WAV_SUFFIXES = {".wav", ".wave"}
# libsndfile subtype -> on-disk sample format label
_WAV_SUBTYPE = {
    "PCM_S8": "int8",
    "PCM_U8": "uint8",
    "PCM_16": "int16",
    "PCM_24": "int24",
    "PCM_32": "int32",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}
_CONF_FS = {"user": 1.0, "sigmf": 1.0, "wav": 1.0, "filename": 0.6, "unknown": 0.15}
_CONF_FC = {"user": 1.0, "sigmf": 1.0, "auxi": 0.9, "filename": 0.6, "none": 0.3}


def read_capture(
    path: str | Path,
    *,
    fs: float | None = None,
    fc: float | None = None,
    dtype: str | None = None,
) -> tuple[np.ndarray, CaptureMeta]:
    """Turn any capture file into ``(iq: complex64[N], CaptureMeta)``.

    ``fs`` / ``fc`` / ``dtype`` are optional caller overrides that beat any file guess.
    Raises :class:`CaptureError` (never a bare traceback) on anything unreadable.
    """
    path = Path(path)
    if not path.is_file():
        raise CaptureError(f"{path}: no such file")
    if path.stat().st_size == 0:
        raise CaptureError(f"{path}: file is empty")

    hints = parse_filename(path.name)
    notes: list[str] = []

    sidecar = find_sidecar(path)
    if sidecar is not None:
        cap = read_sigmf(*sidecar)
        iq = cap.iq
        notes += cap.notes
        src_fmt = "sigmf"
        file_fs, fs_src = cap.sample_rate, ("sigmf" if cap.sample_rate else None)
        file_fc, fc_src = cap.center_freq, ("sigmf" if cap.center_freq is not None else None)
        file_start = cap.start_time
        dtype_label, dtype_conf, dtype_scores = cap.datatype, 1.0, {}
        iq_from_stereo = False
    elif path.suffix.lower() in _WAV_SUFFIXES:
        cap = read_wav(path)
        iq = cap.iq
        notes += cap.notes
        src_fmt = "wav-iq" if cap.iq_from_stereo else "wav-audio"
        file_fs, fs_src = cap.sample_rate, "wav"
        file_fc, fc_src = cap.center_freq, ("auxi" if cap.center_freq is not None else None)
        file_start = cap.start_time
        dtype_label = _WAV_SUBTYPE.get(cap.subtype, cap.subtype or "wav")
        dtype_conf, dtype_scores = 1.0, {}
        iq_from_stereo = cap.iq_from_stereo
    else:
        cap = read_raw(path, dtype=dtype)
        iq = cap.iq
        notes += cap.notes
        src_fmt = "raw-iq"
        file_fs, fs_src = None, None
        file_fc, fc_src = None, None
        file_start = None
        dtype_label, dtype_conf, dtype_scores = cap.dtype, cap.dtype_confidence, cap.dtype_scores
        iq_from_stereo = False

    # ---- resolve sample rate: caller > file > filename > normalised fallback ----
    if fs is not None:
        sample_rate, fs_src, normalised = float(fs), "user", False
        notes.append(f"sample rate {fs:.0f} Hz supplied by caller")
    elif file_fs:
        sample_rate, normalised = float(file_fs), False
    elif hints.sample_rate:
        sample_rate, fs_src, normalised = float(hints.sample_rate), "filename", False
    else:
        sample_rate, fs_src, normalised = 1.0, "unknown", True
        notes.append("sample rate unknown; set to 1.0 and frequencies reported normalised")

    # ---- resolve centre frequency: caller > file > filename > unknown ----
    if fc is not None:
        center_freq, fc_src = float(fc), "user"
        notes.append(f"centre frequency {fc:.0f} Hz supplied by caller")
    elif file_fc is not None:
        center_freq = float(file_fc)
    elif hints.center_freq is not None:
        center_freq, fc_src = float(hints.center_freq), "filename"
    else:
        center_freq, fc_src = None, None
        notes.append("centre frequency unknown; detections reported as offset from centre only")

    start_time = file_start or hints.start_time
    for note in hints.notes:
        if note not in notes:
            notes.append(note)

    meta_conf = round(0.5 * _CONF_FS[fs_src or "unknown"] + 0.5 * _CONF_FC[fc_src or "none"], 2)

    meta = CaptureMeta(
        sample_rate=sample_rate,
        center_freq=center_freq,
        dtype_guessed=dtype_label,
        dtype_confidence=round(float(dtype_conf), 3),
        source_format=src_fmt,
        start_time=start_time,
        n_samples=int(iq.size),
        metadata_confidence=meta_conf,
        notes=notes,
        center_freq_source=fc_src,
        frequencies_are_normalised=normalised,
        iq_from_stereo=iq_from_stereo,
        dtype_candidates={k: round(v, 3) for k, v in dtype_scores.items()},
    )
    return np.ascontiguousarray(iq, dtype=np.complex64), meta


def iter_blocks(
    path: str | Path,
    *,
    fs: float | None = None,
    fc: float | None = None,
    dtype: str | None = None,
    block_samples: int = 1 << 20,
    overlap: float = 0.25,
) -> tuple[CaptureMeta, Iterator[np.ndarray]]:
    """Stream a capture as overlapping ``complex64`` blocks (CLAUDE.md §2 Stage 2, §9 B).

    Headerless raw and stereo WAV are streamed without materialising the whole signal.
    SigMF and mono WAV (Hilbert needs the full signal) yield a single block.
    """
    path = Path(path)
    if not path.is_file():
        raise CaptureError(f"{path}: no such file")

    if find_sidecar(path) is None and path.suffix.lower() not in _WAV_SUFFIXES:
        dtype_name, n_samples, blocks = iter_raw_blocks(
            path, dtype=dtype, block_samples=block_samples, overlap=overlap
        )
        hints = parse_filename(path.name)
        notes = [f"streamed in blocks of {block_samples} samples ({overlap:.0%} overlap)"]
        if fs is not None:
            sample_rate, fs_src, normalised = float(fs), "user", False
        elif hints.sample_rate:
            sample_rate, fs_src, normalised = float(hints.sample_rate), "filename", False
        else:
            sample_rate, fs_src, normalised = 1.0, "unknown", True
            notes.append("sample rate unknown; set to 1.0 and frequencies reported normalised")
        if fc is not None:
            center_freq, fc_src = float(fc), "user"
        elif hints.center_freq is not None:
            center_freq, fc_src = float(hints.center_freq), "filename"
        else:
            center_freq, fc_src = None, None
        notes += hints.notes
        meta = CaptureMeta(
            sample_rate=sample_rate,
            center_freq=center_freq,
            dtype_guessed=dtype_name,
            dtype_confidence=1.0 if dtype else 0.9,
            source_format="raw-iq",
            start_time=hints.start_time,
            n_samples=n_samples,
            metadata_confidence=round(
                0.5 * _CONF_FS[fs_src] + 0.5 * _CONF_FC[fc_src or "none"], 2
            ),
            notes=notes,
            center_freq_source=fc_src,
            frequencies_are_normalised=normalised,
        )
        return meta, blocks

    if path.suffix.lower() in _WAV_SUFFIXES and find_sidecar(path) is None:
        import soundfile as sf

        info = sf.info(path)
        if info.channels == 2:
            _, meta = read_capture(path, fs=fs, fc=fc)
            step = max(1, int(block_samples * (1.0 - overlap)))

            def _wav_blocks() -> Iterator[np.ndarray]:
                for blk in sf.blocks(
                    path, blocksize=block_samples, overlap=block_samples - step,
                    dtype="float32", always_2d=True,
                ):
                    yield (blk[:, 0] + 1j * blk[:, 1]).astype(np.complex64)

            return meta, _wav_blocks()

    # SigMF / mono WAV: read whole, hand back one block
    iq, meta = read_capture(path, fs=fs, fc=fc, dtype=dtype)
    return meta, iter((iq,))
