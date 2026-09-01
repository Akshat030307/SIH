"""Gradient-boosted classifier over the §5.2 features (CLAUDE.md §5.2 "Model").

``HistGradientBoostingClassifier``, ~300 iterations, saved with joblib, fitted on a
``StandardScaler`` that is itself fitted on training data only (§5.2). Also exports
``permutation_importance``, because §5.2 is explicit that that is what generates the
evidence sentences -- a feature the model does not lean on has no business appearing in an
explanation shown to an analyst.

**Untrained is a supported state.** RadioML 2016.10a is a 225 MB download that §2 forbids
fetching at runtime, so a fresh clone has no checkpoint. Every entry point here works
without one and returns an abstention carrying the reason, rather than raising or --
much worse -- guessing. That is the same rule the §4 estimators follow.

The fitted column order is stored in the checkpoint and checked on load. A model trained
against one :data:`~sigscope.features.FEATURE_NAMES` order and applied against another
would produce confident nonsense with nothing to signal the problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sigscope.features import FEATURE_NAMES, extract_features

__all__ = ["TRAINED_CLASSES", "Prediction", "FeatureClassifier", "DEFAULT_MODEL_PATH"]

# §5.1: the dataset fixes the trained label set. `ofdm`, `chirp-lfm`, `cw` and `noise` come
# from deterministic rules only (§5.4) and are never predicted here; `unknown` is never a
# trained class.
TRAINED_CLASSES: tuple[str, ...] = (
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "feature_clf.joblib"


@dataclass
class Prediction:
    """One voter's opinion (§5.5). ``label`` is ``None`` when the voter abstains."""

    label: str | None
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    method: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.label is None

    def top(self, n: int = 2) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:n]


class FeatureClassifier:
    """Scaler + gradient boosting over the §5.2 feature vector."""

    method = "feature_clf/§5.2"

    def __init__(
        self,
        model: Any = None,
        scaler: Any = None,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
        classes: tuple[str, ...] = TRAINED_CLASSES,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_names = tuple(feature_names)
        self.classes = tuple(classes)
        self.metadata = metadata or {}

    # ---------------------------------------------------------------- persistence
    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.scaler is not None

    @classmethod
    def load(cls, path: str | Path | None = None) -> FeatureClassifier:
        """Load a checkpoint, or return an untrained instance if there is none.

        A missing checkpoint is normal on a fresh clone and is not an error; the returned
        classifier abstains and says why.
        """
        path = Path(path or DEFAULT_MODEL_PATH)
        if not path.is_file():
            return cls()
        import joblib

        blob = joblib.load(path)
        names = tuple(blob.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError(
                f"{path}: checkpoint was fitted on {len(names)} features in a different "
                f"order than the current FEATURE_NAMES ({len(FEATURE_NAMES)}). Retrain "
                "with `sigscope train --model feature`."
            )
        return cls(
            model=blob["model"],
            scaler=blob["scaler"],
            feature_names=names,
            classes=tuple(blob["classes"]),
            metadata=blob.get("metadata", {}),
        )

    def save(self, path: str | Path | None = None) -> Path:
        if not self.is_trained:
            raise RuntimeError("refusing to save an untrained classifier")
        import joblib

        path = Path(path or DEFAULT_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "classes": self.classes,
                "metadata": self.metadata,
            },
            path,
        )
        return path

    # ---------------------------------------------------------------- training
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        max_iter: int = 300,
        learning_rate: float = 0.1,
        random_state: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureClassifier:
        """Fit the scaler and the classifier on the **training split only** (§5.2, §6.1)."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected features of shape (n, {len(FEATURE_NAMES)}), got {features.shape}"
            )

        self.scaler = StandardScaler().fit(features)
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ).fit(self.scaler.transform(features), labels)
        self.classes = tuple(str(c) for c in self.model.classes_)
        self.metadata = metadata or {}
        return self

    def permutation_importance(
        self, features: np.ndarray, labels: np.ndarray, *, n_repeats: int = 5, random_state: int = 0
    ) -> dict[str, float]:
        """Feature importances for the §5.6 evidence sentences (§5.2 "Also export")."""
        if not self.is_trained:
            return {}
        from sklearn.inspection import permutation_importance as sk_permutation_importance

        result = sk_permutation_importance(
            self.model,
            self.scaler.transform(np.asarray(features, dtype=np.float64)),
            np.asarray(labels),
            n_repeats=n_repeats,
            random_state=random_state,
        )
        return {
            name: float(value)
            for name, value in zip(self.feature_names, result.importances_mean, strict=True)
        }

    # ---------------------------------------------------------------- inference
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for a feature matrix, in :attr:`classes` order."""
        if not self.is_trained:
            raise RuntimeError("classifier is not trained")
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        return self.model.predict_proba(self.scaler.transform(features))

    def predict(self, y: np.ndarray, *, sps: int | None = None) -> Prediction:
        """Classify one burst. Abstains, with a reason, when there is no checkpoint."""
        if not self.is_trained:
            return Prediction(
                label=None,
                confidence=0.0,
                method=self.method,
                notes=[
                    "the feature classifier has no trained checkpoint; run "
                    "`sigscope fetch-data` then `sigscope train --model feature`"
                ],
            )
        y = np.asarray(y)
        if y.size < 16:
            return Prediction(
                label=None, confidence=0.0, method=self.method,
                notes=[f"burst is only {y.size} samples; too short to classify"],
            )

        probabilities = self.predict_proba(extract_features(y, sps=sps))[0]
        order = int(np.argmax(probabilities))
        return Prediction(
            label=str(self.classes[order]),
            confidence=float(probabilities[order]),
            probabilities={
                str(c): float(p) for c, p in zip(self.classes, probabilities, strict=True)
            },
            method=self.method,
            notes=[],
        )
