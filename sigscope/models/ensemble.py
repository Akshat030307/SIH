"""Ensemble referee (CLAUDE.md §5.5-§5.6).

Takes a fired rule outright (confidence 0.9); otherwise blends the feature-classifier
and CNN softmax weighted by their validation accuracy at the measured SNR bucket.
Emits ``unknown`` below 0.45 max probability, caps confidence at 0.6 on model
disagreement, and produces 2-4 plain-language evidence sentences.

Not implemented yet (Phase 6, §8).
"""

from __future__ import annotations
