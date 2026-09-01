"""Classification: three opinions and a referee (CLAUDE.md §5).

- :mod:`~sigscope.models.feature_clf` -- gradient boosting over the §5.2 feature vector
- :mod:`~sigscope.models.cnn`         -- the §5.3 residual CNN, with sliding-window inference
- :mod:`~sigscope.models.rules`       -- the §5.4 deterministic rules, which override both
- :mod:`~sigscope.models.ensemble`    -- the §5.5 referee
- :mod:`~sigscope.models.evidence`    -- the §5.6 evidence sentences

:func:`classify` is the single entry point the pipeline uses. Both learned models work
untrained -- they abstain with a reason -- so the whole path is exercised on a fresh clone
with no checkpoints, and a rule can still fire and produce a physically-grounded label.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sigscope.models.cnn import CnnClassifier
from sigscope.models.ensemble import (
    PRIOR_WEIGHTS,
    SNR_BANDS,
    EnsembleConfig,
    EnsembleResult,
    referee,
    snr_band,
)
from sigscope.models.evidence import build_evidence
from sigscope.models.feature_clf import TRAINED_CLASSES, FeatureClassifier, Prediction
from sigscope.models.rules import RULE_LABELS, RuleConfig, RuleHit, apply_rules

__all__ = [
    "FeatureClassifier",
    "CnnClassifier",
    "Prediction",
    "TRAINED_CLASSES",
    "RULE_LABELS",
    "RuleHit",
    "RuleConfig",
    "apply_rules",
    "EnsembleConfig",
    "EnsembleResult",
    "SNR_BANDS",
    "PRIOR_WEIGHTS",
    "referee",
    "snr_band",
    "build_evidence",
    "Classifiers",
    "load_classifiers",
    "classify",
]


@dataclass
class Classifiers:
    """The two learned voters, loaded once and reused across a batch."""

    feature: FeatureClassifier
    cnn: CnnClassifier

    @property
    def any_trained(self) -> bool:
        return self.feature.is_trained or self.cnn.is_trained

    def status(self) -> list[str]:
        return [
            f"feature classifier: {'loaded' if self.feature.is_trained else 'not trained'}",
            f"cnn: {'loaded' if self.cnn.is_trained else 'not trained'}",
        ]


@lru_cache(maxsize=4)
def load_classifiers(
    feature_path: str | None = None, cnn_path: str | None = None
) -> Classifiers:
    """Load both checkpoints once. Missing checkpoints yield abstaining models."""
    return Classifiers(
        feature=FeatureClassifier.load(Path(feature_path) if feature_path else None),
        cnn=CnnClassifier.load(Path(cnn_path) if cnn_path else None),
    )


def classify(
    y: np.ndarray,
    fs_b: float,
    *,
    snr_db: float | None = None,
    sps: int | None = None,
    raw_slice: np.ndarray | None = None,
    fs_raw: float | None = None,
    box_f_lo: float | None = None,
    box_f_hi: float | None = None,
    chirp: Any = None,
    extra: dict[str, Any] | None = None,
    symbol_rate: float | None = None,
    bandwidth_hz: float | None = None,
    classifiers: Classifiers | None = None,
    cfg: EnsembleConfig | None = None,
) -> tuple[EnsembleResult, list[str]]:
    """Classify one isolated burst end to end (§5.2 -> §5.6).

    Returns the referee's result and the §5.6 evidence sentences. ``raw_slice`` is the
    un-isolated time slice, needed by the chirp rule alone.
    """
    classifiers = classifiers or load_classifiers()

    hit = apply_rules(
        y,
        fs_b,
        snr_db=snr_db,
        raw_slice=raw_slice,
        fs_raw=fs_raw,
        box_f_lo=box_f_lo,
        box_f_hi=box_f_hi,
        chirp=chirp,
    )
    feature_prediction = classifiers.feature.predict(y, sps=sps)
    cnn_prediction = classifiers.cnn.predict(y)

    result = referee(
        feature_prediction,
        cnn_prediction,
        rule_hit=hit,
        snr_db=snr_db,
        feature_metadata=classifiers.feature.metadata,
        cnn_metadata=classifiers.cnn.metadata,
        cfg=cfg,
    )

    merged_extra = dict(extra or {})
    merged_extra.update(result.extra)
    evidence = build_evidence(
        result.label,
        snr_db=snr_db,
        extra=merged_extra,
        rule_evidence=list(getattr(hit, "evidence", []) or []),
        ensemble_notes=result.notes,
        model_metadata=classifiers.feature.metadata or classifiers.cnn.metadata,
        symbol_rate=symbol_rate,
        bandwidth_hz=bandwidth_hz,
    )
    result.extra = merged_extra
    return result, evidence
