"""Deterministic classification rules (CLAUDE.md §5.4).

Physics-based rules that override the learned models when they fire: OFDM (cyclic-prefix
autocorrelation peak), chirp-lfm (linear ridge fit), cw (on/off envelope with ~3:1
dot/dash ratio), noise (low SNR + high spectral flatness), SSB (spectral asymmetry).

Not implemented yet (Phase 6, §8).
"""

from __future__ import annotations
