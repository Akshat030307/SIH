"""Filename metadata parsers (CLAUDE.md §3 Stage 1 "Filename patterns").

Recognises the SDR#, HDSDR and SDRuno naming conventions plus the generic ``_<rate>sps_``
and ``_fc<freq>_`` tokens, and records which pattern matched so the guess is auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SI = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}


def _si(value: str, suffix: str) -> float:
    return float(value) * _SI[suffix.lower()]


def _iso(date8: str, time6: str) -> str:
    return f"{date8[0:4]}-{date8[4:6]}-{date8[6:8]}T{time6[0:2]}:{time6[2:4]}:{time6[4:6]}Z"


# SDRSharp_YYYYMMDD_HHMMSSZ_<freq>Hz_IQ.wav
_SDRSHARP = re.compile(
    r"SDRSharp_(?P<d>\d{8})_(?P<t>\d{6})Z_(?P<f>\d+)Hz_IQ", re.IGNORECASE
)
# HDSDR_YYYYMMDD_HHMMSSZ_<khz>kHz_RF.wav
_HDSDR = re.compile(
    r"HDSDR_(?P<d>\d{8})_(?P<t>\d{6})Z_(?P<f>\d+(?:\.\d+)?)kHz_RF", re.IGNORECASE
)
# SDRuno_YYYYMMDD_HHMMSSZ_<freq>Hz.wav  (SDRuno also fills the auxi chunk)
_SDRUNO = re.compile(
    r"SDRuno_(?P<d>\d{8})_(?P<t>\d{6})Z?_(?P<f>\d+(?:\.\d+)?)(?P<u>[kMG]?)Hz", re.IGNORECASE
)
_RATE = re.compile(r"(?<![0-9.])(?P<v>\d+(?:\.\d+)?)(?P<u>[kMG]?)sps", re.IGNORECASE)
_FC = re.compile(r"fc[_-]?(?P<v>\d+(?:\.\d+)?)(?P<u>[kMG]?)(?:hz)?", re.IGNORECASE)


@dataclass
class FilenameHints:
    """Metadata recovered from a file name. Every field is optional."""

    sample_rate: float | None = None
    center_freq: float | None = None
    start_time: str | None = None
    matched: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.matched)


def parse_filename(name: str) -> FilenameHints:
    """Parse ``name`` (basename, extension included) for capture metadata."""
    h = FilenameHints()

    if (m := _SDRSHARP.search(name)) is not None:
        h.center_freq = float(m["f"])
        h.start_time = _iso(m["d"], m["t"])
        h.matched.append("SDRSharp")
        h.notes.append(f"SDR# filename: centre {h.center_freq:.0f} Hz, start {h.start_time}")
    elif (m := _HDSDR.search(name)) is not None:
        h.center_freq = float(m["f"]) * 1e3
        h.start_time = _iso(m["d"], m["t"])
        h.matched.append("HDSDR")
        h.notes.append(f"HDSDR filename: centre {h.center_freq:.0f} Hz, start {h.start_time}")
    elif (m := _SDRUNO.search(name)) is not None:
        h.center_freq = _si(m["v"], m["u"])
        h.start_time = _iso(m["d"], m["t"])
        h.matched.append("SDRuno")
        h.notes.append(f"SDRuno filename: centre {h.center_freq:.0f} Hz, start {h.start_time}")

    if (m := _RATE.search(name)) is not None:
        h.sample_rate = _si(m["v"], m["u"])
        h.matched.append("sps")
        h.notes.append(f"filename '_{m['v']}{m['u']}sps' -> sample rate {h.sample_rate:.0f} Hz")

    if h.center_freq is None and (m := _FC.search(name)) is not None:
        h.center_freq = _si(m["v"], m["u"])
        h.matched.append("fc")
        h.notes.append(f"filename 'fc{m['v']}{m['u']}' -> centre {h.center_freq:.0f} Hz")

    return h
