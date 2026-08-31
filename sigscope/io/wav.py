"""WAV reader (CLAUDE.md §3 Stage 1, step 2).

Two channels -> IQ (I = ch0, Q = ch1), ``iq_from_stereo`` flagged. One channel -> real
audio, lifted to complex baseband with :func:`scipy.signal.hilbert`. The non-standard
``auxi`` RIFF chunk written by HDSDR and SDRuno is parsed for centre frequency and start
time.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from sigscope.io.errors import CaptureError


@dataclass
class WavCapture:
    iq: np.ndarray
    sample_rate: float
    iq_from_stereo: bool
    center_freq: float | None = None
    start_time: str | None = None
    subtype: str = ""
    notes: list[str] = field(default_factory=list)


def _iter_riff_chunks(path: Path):
    """Yield ``(chunk_id: bytes, data: bytes)`` for a RIFF/WAVE file, skipping ``data``."""
    with open(path, "rb") as fh:
        header = fh.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return
            cid, size = struct.unpack("<4sI", head)
            if cid == b"data":  # never pull the whole signal into memory
                fh.seek(size + (size & 1), 1)
                continue
            yield cid, fh.read(size)
            if size & 1:
                fh.seek(1, 1)


def _parse_auxi(blob: bytes) -> tuple[float | None, str | None]:
    """HDSDR/SDRuno ``auxi`` chunk: 2x SYSTEMTIME, then CenterFreq, then ADFrequency."""
    center = start = None
    if len(blob) >= 16:
        y, mo, _dow, d, h, mi, s, _ms = struct.unpack("<8H", blob[:16])
        if 1970 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            start = f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}Z"
    if len(blob) >= 36:
        (freq,) = struct.unpack("<I", blob[32:36])
        if 0 < freq < 0xFFFFFFFF:
            center = float(freq)
    return center, start


def read_wav(path: Path) -> WavCapture:
    """Read a WAV file as IQ (stereo) or real audio (mono)."""
    import soundfile as sf  # lazy: raw-IQ ingest must not require libsndfile

    path = Path(path)
    try:
        data, samplerate = sf.read(path, dtype="float32", always_2d=True)
        info = sf.info(path)
    except (RuntimeError, OSError, sf.SoundFileError) as exc:
        raise CaptureError(f"{path}: not a readable WAV file ({exc})") from exc
    if data.shape[0] == 0:
        raise CaptureError(f"{path}: WAV file contains no samples")
    if data.shape[1] > 2:
        raise CaptureError(f"{path}: {data.shape[1]}-channel WAV; expected mono audio or stereo IQ")

    center = start = None
    notes: list[str] = []
    for cid, blob in _iter_riff_chunks(path):
        if cid == b"auxi":
            center, start = _parse_auxi(blob)
            if center is not None:
                notes.append(f"auxi chunk: centre {center:.0f} Hz")
            if start is not None:
                notes.append(f"auxi chunk: start {start}")

    if data.shape[1] == 2:
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
        notes.insert(0, "stereo WAV read as IQ (I=ch0, Q=ch1); iq_from_stereo=True")
        stereo = True
    else:
        iq = hilbert(data[:, 0]).astype(np.complex64)
        notes.insert(0, "mono WAV: real audio lifted to complex baseband via Hilbert transform")
        stereo = False

    return WavCapture(
        iq=iq,
        sample_rate=float(samplerate),
        iq_from_stereo=stereo,
        center_freq=center,
        start_time=start,
        subtype=info.subtype or "",
        notes=notes,
    )
