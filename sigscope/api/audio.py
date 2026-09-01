"""Demodulate a detection to listenable audio (CLAUDE.md §7 ``/audio.wav``, §8 stretch 2).

§7 lists the endpoint as "Demodulated audio, **where demodulable**", and that qualifier is
the whole design. AM and FM demodulate to something an analyst can actually listen to; a
QPSK burst does not, and handing back a wav of digital hash would be a worse answer than
saying so. :func:`demodulate` returns ``None`` with a reason for anything it cannot honestly
render, and the API turns that into a 422 naming the problem.

CW is included because §8's first stretch goal is a Morse decoder and hearing the tone is
half of that demo: an on/off-keyed carrier is beaten against a 700 Hz offset so the keying
becomes an audible note rather than silence.

Written with numpy and ``soundfile`` only. §2 rules out anything that would need fetching a
codec at demo time.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, oaconvolve, resample_poly

__all__ = ["AudioResult", "demodulate", "AUDIO_RATE", "CW_TONE_HZ"]

AUDIO_RATE = 48_000  # what every laptop can play without resampling in the browser
CW_TONE_HZ = 700.0  # the conventional Morse sidetone
AUDIO_BAND_HZ = 5_000.0  # §4.13 puts speech content in 300 Hz - 5 kHz


@dataclass
class AudioResult:
    """Demodulated audio, or the reason there is none."""

    samples: np.ndarray | None
    sample_rate: int
    mode: str
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.samples is not None

    def to_wav_bytes(self) -> bytes:
        """16-bit PCM wav in memory -- these clips are seconds long, not gigabytes."""
        import soundfile as sf

        if self.samples is None:
            raise ValueError("no audio to write")
        buffer = io.BytesIO()
        sf.write(buffer, self.samples, self.sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


def _normalise(audio: np.ndarray) -> np.ndarray:
    """Scale to a comfortable level without clipping, and strip DC."""
    audio = np.asarray(audio, dtype=np.float64)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.89
    return audio.astype(np.float32)


def _to_audio_rate(audio: np.ndarray, fs_in: float) -> tuple[np.ndarray, int]:
    """Resample to :data:`AUDIO_RATE` with a rational factor, capped to stay cheap."""
    if fs_in <= 0 or audio.size == 0:
        return audio.astype(np.float32), AUDIO_RATE
    ratio = AUDIO_RATE / fs_in
    limit = 2000
    up, down = (
        (int(round(ratio * 100)), 100) if ratio < 1.0 else (int(round(ratio * 10)), 10)
    )
    up = max(1, min(up, limit))
    down = max(1, min(down, limit))
    if up == down:
        return audio.astype(np.float32), int(round(fs_in))
    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32), int(round(fs_in * up / down))


def _lowpass(y: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Zero-delay FIR lowpass; skipped when the cutoff is above Nyquist already."""
    if cutoff >= 0.45 * fs or y.size < 64:
        return y
    taps = min(127, (y.size - 1) | 1)
    if taps < 5:
        return y
    return oaconvolve(y, firwin(taps, cutoff, fs=fs, window="hamming"), mode="same")


def demodulate(
    y: np.ndarray,
    fs_b: float,
    *,
    label: str | None = None,
    am_depth: float | None = None,
    is_morse: bool = False,
    max_seconds: float = 30.0,
) -> AudioResult:
    """Demodulate an isolated burst to audio, or explain why it cannot be.

    The mode is chosen from what was *measured* rather than from the label alone, so a
    detection the classifier left ``unclassified`` can still be listened to when its
    envelope or its instantaneous frequency carries something audible:

    * an on/off-keyed carrier (``is_morse``) becomes a keyed 700 Hz tone;
    * a varying envelope (``am_depth`` above 0.1, or an AM/SSB label) is envelope-detected;
    * a constant envelope with a varying instantaneous frequency is FM-discriminated,
      which covers WBFM and the FSK family;
    * anything else returns ``None`` with a reason.
    """
    y = np.ascontiguousarray(y, dtype=np.complex64)
    if y.size < 256 or fs_b <= 0:
        return AudioResult(None, AUDIO_RATE, "none", "burst is too short to demodulate")

    limit = int(max_seconds * fs_b)
    if y.size > limit:
        y = y[:limit]

    power = float(np.mean(np.abs(y) ** 2))
    if power <= 0:
        return AudioResult(None, AUDIO_RATE, "none", "burst has no power")
    y = (y / math.sqrt(power)).astype(np.complex64)

    envelope = np.abs(y).astype(np.float64)
    mean_envelope = float(np.mean(envelope))
    variation = float(np.std(envelope) / mean_envelope) if mean_envelope > 0 else 0.0
    upper = (label or "").upper()

    # ---- CW / Morse: beat the keyed carrier against a sidetone so it is audible ----
    if is_morse:
        smoothed = _lowpass(envelope, fs_b, min(AUDIO_BAND_HZ, 0.4 * fs_b))
        audio, rate = _to_audio_rate(smoothed, fs_b)
        tone = np.sin(2 * np.pi * CW_TONE_HZ * np.arange(audio.size) / rate)
        return AudioResult(_normalise(audio * tone), rate, "cw")

    # ---- AM / SSB: envelope detection ----
    if "AM" in upper or (am_depth is not None and am_depth > 0.1) or variation > 0.25:
        detected = _lowpass(envelope - mean_envelope, fs_b, min(AUDIO_BAND_HZ, 0.4 * fs_b))
        audio, rate = _to_audio_rate(detected, fs_b)
        if audio.size < 64:
            return AudioResult(None, AUDIO_RATE, "am", "not enough audio after resampling")
        return AudioResult(_normalise(audio), rate, "am")

    # ---- FM: discriminate, then lowpass to the audio band ----
    if variation < 0.25:
        discriminated = np.angle(y[1:] * np.conj(y[:-1])).astype(np.float64)
        filtered = _lowpass(discriminated, fs_b, min(AUDIO_BAND_HZ, 0.4 * fs_b))
        audio, rate = _to_audio_rate(filtered, fs_b)
        if audio.size < 64:
            return AudioResult(None, AUDIO_RATE, "fm", "not enough audio after resampling")
        return AudioResult(_normalise(audio), rate, "fm")

    return AudioResult(
        None,
        AUDIO_RATE,
        "none",
        f"a {label or 'digitally modulated'} burst has no audio to recover; "
        "demodulation to bits is out of scope (CLAUDE.md §1 'Scope')",
    )
