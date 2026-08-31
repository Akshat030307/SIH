"""Feature-based modulation classifier (CLAUDE.md §5.2).

``sklearn.ensemble.HistGradientBoostingClassifier`` (~300 iterations) over the ~30
handcrafted features, with a ``StandardScaler`` fitted on training data only. Saved with
joblib. ``permutation_importance`` is exported and drives the evidence sentences (§5.6).

Not implemented yet (Phase 6, §8).
"""

from __future__ import annotations
