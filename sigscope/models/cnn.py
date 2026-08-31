"""Residual 1-D CNN on raw IQ (CLAUDE.md §5.3).

Input ``2 x 128`` float32 (I and Q as channels, power-normalised). Conv1d stem ->
2x ResidualBlock(64) -> 2x ResidualBlock(128) -> AdaptiveAvgPool -> Dropout -> Linear.
~250k parameters, trains on CPU. Sliding 128-sample window with 50% overlap at inference
on real bursts; average the softmax, report the spread as a confidence signal.

Not implemented yet (Phase 6, §8).
"""

from __future__ import annotations
