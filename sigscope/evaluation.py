"""Classification scoring against the RadioML test split (CLAUDE.md §9 D, §5.3).

Produces everything §9 D asks for and §10 says wins the pitch: accuracy per SNR for each
model and the ensemble, confusion matrices in three SNR bands, per-class precision and
recall, and the calibration check.

**Never a single overall number.** §5.3 is explicit -- "Report validation accuracy per SNR
bucket, never as a single overall number. A single number is meaningless and a judge will
say so" -- so no function here returns one, and the report renders curves and bands only.

The test split is used here and nowhere else (§6.1). ``scripts/train.py`` touches train and
val; this module touches test. Keeping that boundary in code rather than in a comment is
what makes the published numbers comparable with the literature.

One evaluation-only liberty, stated because it flatters us slightly: the §5.5 ensemble
weights by the *measured* SNR at inference, but on RadioML the true SNR label is available,
so that is what is used here. A real burst's SNR carries the §4.7 estimator's own error --
under 0.05 dB against known truth, so the difference is small, but it is not zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sigscope.models.ensemble import SNR_BANDS, referee, snr_band
from sigscope.models.feature_clf import Prediction

__all__ = [
    "ClassScore",
    "ModelScores",
    "score_predictions",
    "confusion_matrix",
    "calibration_table",
    "CALIBRATION_THRESHOLD",
    "evaluate_models",
]

CALIBRATION_THRESHOLD = 0.8  # §9 D: "among predictions with confidence above 0.8..."


@dataclass
class ClassScore:
    """Per-class precision, recall and support."""

    label: str
    precision: float
    recall: float
    support: int


@dataclass
class ModelScores:
    """Everything §9 D wants about one model (or the ensemble)."""

    name: str
    accuracy_by_snr: dict[int, float] = field(default_factory=dict)
    accuracy_by_band: dict[str, float] = field(default_factory=dict)
    per_class: list[ClassScore] = field(default_factory=list)
    confusion: dict[str, tuple[list[str], np.ndarray]] = field(default_factory=dict)
    calibration: list[tuple[float, float, int, float]] = field(default_factory=list)
    high_confidence_accuracy: float | None = None
    high_confidence_count: int = 0
    unknown_fraction_low_snr: float | None = None
    n_evaluated: int = 0


def score_predictions(
    name: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    snr_db: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    labels: list[str] | None = None,
) -> ModelScores:
    """Full §9 D score card for one model's predictions."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    snr_db = np.asarray(snr_db)
    labels = labels or sorted(set(truth.tolist()) | set(predicted.tolist()))

    scores = ModelScores(name=name, n_evaluated=int(truth.size))

    for snr in sorted(np.unique(snr_db)):
        mask = snr_db == snr
        scores.accuracy_by_snr[int(snr)] = float(np.mean(truth[mask] == predicted[mask]))

    bands = np.array([snr_band(float(s)) for s in snr_db])
    for band_name, _, _ in SNR_BANDS:
        mask = bands == band_name
        if int(np.count_nonzero(mask)) == 0:
            continue
        scores.accuracy_by_band[band_name] = float(
            np.mean(truth[mask] == predicted[mask])
        )
        scores.confusion[band_name] = confusion_matrix(
            truth[mask], predicted[mask], labels
        )

    for label in labels:
        predicted_positive = predicted == label
        actual_positive = truth == label
        true_positive = int(np.count_nonzero(predicted_positive & actual_positive))
        n_predicted = int(np.count_nonzero(predicted_positive))
        n_actual = int(np.count_nonzero(actual_positive))
        if n_actual == 0 and n_predicted == 0:
            continue
        scores.per_class.append(
            ClassScore(
                label=label,
                precision=(true_positive / n_predicted) if n_predicted else 0.0,
                recall=(true_positive / n_actual) if n_actual else 0.0,
                support=n_actual,
            )
        )

    # §9 D: at SNR < 5 dB most errors must land on `unknown` rather than a wrong label
    low = snr_db < 5.0
    if int(np.count_nonzero(low)) > 0:
        wrong = low & (truth != predicted)
        if int(np.count_nonzero(wrong)) > 0:
            scores.unknown_fraction_low_snr = float(
                np.mean(predicted[wrong] == "unknown")
            )

    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)
        scores.calibration = calibration_table(truth, predicted, confidence)
        high = confidence > CALIBRATION_THRESHOLD
        scores.high_confidence_count = int(np.count_nonzero(high))
        if scores.high_confidence_count:
            scores.high_confidence_accuracy = float(
                np.mean(truth[high] == predicted[high])
            )
    return scores


def confusion_matrix(
    truth: np.ndarray, predicted: np.ndarray, labels: list[str]
) -> tuple[list[str], np.ndarray]:
    """Row-normalised confusion matrix: rows are truth, columns are prediction."""
    index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guess in zip(truth, predicted, strict=True):
        if actual in index and guess in index:
            counts[index[actual], index[guess]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(
        counts, np.maximum(totals, 1), out=np.zeros(counts.shape), where=totals > 0
    )
    return labels, fractions


def calibration_table(
    truth: np.ndarray, predicted: np.ndarray, confidence: np.ndarray, bins: int = 5
) -> list[tuple[float, float, int, float]]:
    """Reliability table: ``(lo, hi, count, accuracy)`` per confidence bin.

    §9 D calls calibration more important than raw accuracy, and this is the table that
    shows it: a well-calibrated model's accuracy in each bin tracks the bin's confidence.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[tuple[float, float, int, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        count = int(np.count_nonzero(mask))
        accuracy = float(np.mean(truth[mask] == predicted[mask])) if count else 0.0
        out.append((float(lo), float(hi), count, accuracy))
    return out


def evaluate_models(
    iq: np.ndarray,
    truth: np.ndarray,
    snr_db: np.ndarray,
    classifiers: Any,
    *,
    progress: int = 0,
) -> dict[str, ModelScores]:
    """Score the feature classifier, the CNN and the §5.5 ensemble on one split.

    ``iq`` is ``(n, 2, 128)`` in the RadioML layout. Returns one score card per model that
    is actually trained, plus the ensemble whenever at least one of them is.
    """
    from sigscope.features import extract_feature_matrix

    iq = np.asarray(iq)
    complex_iq = (iq[:, 0, :] + 1j * iq[:, 1, :]).astype(np.complex64)
    n = complex_iq.shape[0]
    out: dict[str, ModelScores] = {}

    feature_probabilities = None
    if classifiers.feature.is_trained:
        features = extract_feature_matrix(iq, progress=progress)
        feature_probabilities = classifiers.feature.predict_proba(features)
        classes = list(classifiers.feature.classes)
        predicted = np.array(classes)[feature_probabilities.argmax(axis=1)]
        out["feature_clf"] = score_predictions(
            "feature_clf",
            truth,
            predicted,
            snr_db,
            feature_probabilities.max(axis=1),
        )

    cnn_probabilities = None
    if classifiers.cnn.is_trained:
        import torch

        network = classifiers.cnn.network
        network.eval()
        power = np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1, keepdims=True)
        normalised = (iq / np.sqrt(np.maximum(power, 1e-12))[:, None, :]).astype(np.float32)
        chunks = []
        with torch.no_grad():
            for start in range(0, n, 1024):
                batch = torch.from_numpy(normalised[start : start + 1024])
                chunks.append(torch.softmax(network(batch), dim=1).numpy())
        cnn_probabilities = np.concatenate(chunks, axis=0)
        classes = list(classifiers.cnn.classes)
        predicted = np.array(classes)[cnn_probabilities.argmax(axis=1)]
        out["cnn"] = score_predictions(
            "cnn", truth, predicted, snr_db, cnn_probabilities.max(axis=1)
        )

    if feature_probabilities is None and cnn_probabilities is None:
        return out

    # ---- the §5.5 ensemble, run example by example so the referee's real logic is what
    # gets scored rather than a reimplementation of it ----
    ensemble_labels: list[str] = []
    ensemble_confidence: list[float] = []
    feature_classes = list(classifiers.feature.classes)
    cnn_classes = list(classifiers.cnn.classes)
    for i in range(n):
        feature_prediction = Prediction(None, 0.0, {}, "feature_clf/§5.2", ["untrained"])
        if feature_probabilities is not None:
            row = feature_probabilities[i]
            feature_prediction = Prediction(
                feature_classes[int(row.argmax())],
                float(row.max()),
                dict(zip(feature_classes, map(float, row), strict=True)),
                "feature_clf/§5.2",
            )
        cnn_prediction = Prediction(None, 0.0, {}, "cnn/§5.3", ["untrained"])
        if cnn_probabilities is not None:
            row = cnn_probabilities[i]
            cnn_prediction = Prediction(
                cnn_classes[int(row.argmax())],
                float(row.max()),
                dict(zip(cnn_classes, map(float, row), strict=True)),
                "cnn/§5.3",
            )
        result = referee(
            feature_prediction,
            cnn_prediction,
            rule_hit=None,  # 128-sample windows: every §5.4 rule abstains (see module docs)
            snr_db=float(snr_db[i]),
            feature_metadata=classifiers.feature.metadata,
            cnn_metadata=classifiers.cnn.metadata,
        )
        ensemble_labels.append(result.label)
        ensemble_confidence.append(result.confidence)
        if progress and i and i % progress == 0:
            print(f"    ensemble {i:,}/{n:,}", flush=True)

    del complex_iq
    out["ensemble"] = score_predictions(
        "ensemble",
        truth,
        np.array(ensemble_labels),
        snr_db,
        np.array(ensemble_confidence),
    )
    return out
