"""Wideband test-scene composer (CLAUDE.md §6.2).

Builds an empty complex canvas (5-30 s, 1-10 MHz), places 1-8 RadioML examples at random
frequency offsets, powers and start times, adds complex Gaussian noise, records the
ground truth (time span, frequency span, SNR, modulation per signal), and writes the
scene as raw int16 IQ plus a ground-truth SigMF annotation file. Deterministic given a
seed. Includes awkward cases deliberately (band edge, adjacent signals, overlap, empty,
pure noise, DC spike, full-band continuous).

Not implemented yet (Phase 3, §8).
"""

from __future__ import annotations
