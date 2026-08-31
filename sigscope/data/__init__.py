"""Dataset loading and test-scene composition (CLAUDE.md §6).

- ``radioml`` — load/convert RadioML 2016.10a to a memmapped .npy + labels.parquet (§6.1)
- ``scene`` — compose wideband multi-signal test scenes from RadioML examples with
  ground-truth SigMF annotations, for testing detection (§6.2)

Not implemented yet (Phases 1 and 3, §8).
"""

from __future__ import annotations
