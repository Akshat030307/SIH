"""Classification models and the ensemble referee (CLAUDE.md §5).

- ``feature_clf`` — HistGradientBoostingClassifier on the §5.2 handcrafted features
- ``cnn`` — residual 1-D CNN on raw 2x128 IQ (§5.3)
- ``rules`` — deterministic physics rules that override the models when they fire (§5.4)
- ``ensemble`` — blends the votes, emits ``unknown`` on low confidence, writes evidence (§5.5-§5.6)

Not implemented yet (Phase 6, §8).
"""

from __future__ import annotations
