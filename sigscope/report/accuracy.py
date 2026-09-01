"""Markdown tables for ACCURACY.md's classification half (CLAUDE.md §9 D, §10).

Renders what :mod:`sigscope.evaluation` scores: accuracy per SNR for each model and the
ensemble, accuracy by SNR band, per-class precision and recall, confusion matrices in three
bands, the calibration check, and the low-SNR failure mode.

§10 says the accuracy-vs-SNR curve with the failure region marked is "the one slide that
wins it", and §5.3 forbids collapsing any of this to a single number -- so nothing here
emits one, and the "where we fail" prose is generated from the measured numbers rather
than written by hand and left to go stale.
"""

from __future__ import annotations

from sigscope.evaluation import CALIBRATION_THRESHOLD, ModelScores

__all__ = ["render_classification", "render_unavailable"]

_BAND_TITLES = (
    ("high", "high SNR (>= 15 dB)"),
    ("mid", "mid SNR (5-15 dB)"),
    ("low", "low SNR (< 5 dB)"),
)


def render_unavailable(problems: list[str]) -> list[str]:
    """The section that appears when the models or the dataset are missing.

    Written out in full rather than omitted: a reader has to be able to tell "we measured
    this and it is bad" apart from "we did not measure this", and a silently absent table
    reads like the first.
    """
    out = [
        "## Classification accuracy (§9 D)",
        "",
        "**Not measured.** These tables need the RadioML test split and trained "
        "checkpoints, and at least one is missing:",
        "",
    ]
    out += [f"- {problem}" for problem in problems]
    out += [
        "",
        "No classification numbers are quoted anywhere else in this document, and none "
        "should be quoted in the pitch, until this section is populated. §2's rule against "
        "fabricated results applies to accuracy figures more than to anything else.",
        "",
        "Once both are present::",
        "",
        "```bash",
        "sigscope fetch-data --src <path to RML2016.10a_dict.pkl>",
        "sigscope train --model both",
        "sigscope evaluate --out ACCURACY.md",
        "```",
        "",
        "What will appear here: accuracy vs SNR for the feature classifier, the CNN and "
        "the §5.5 ensemble; accuracy by SNR band against §9 D's targets; per-class "
        "precision and recall; confusion matrices at three SNR bands; the calibration "
        "check; and the share of sub-5 dB errors that land on `unknown`.",
        "",
    ]
    return out


def _accuracy_vs_snr(scores: dict[str, ModelScores]) -> list[str]:
    snrs = sorted({s for m in scores.values() for s in m.accuracy_by_snr})
    if not snrs:
        return []
    out = [
        "### Accuracy vs SNR",
        "",
        "Each column is one SNR level of the RadioML **test** split. §5.3: *report accuracy "
        "per SNR bucket, never as a single overall number*.",
        "",
        "| Model | " + " | ".join(f"{s} dB" for s in snrs) + " |",
        "|---|" + "---|" * len(snrs),
    ]
    for name, model in scores.items():
        cells = " | ".join(
            f"{model.accuracy_by_snr[s]:.2f}" if s in model.accuracy_by_snr else "—"
            for s in snrs
        )
        out.append(f"| {name} | {cells} |")
    out.append("")
    return out


def _accuracy_by_band(scores: dict[str, ModelScores]) -> list[str]:
    out = [
        "### Accuracy by SNR band, against the §9 D targets",
        "",
        "| Model | low (< 5 dB) | mid (5-15 dB) | high (>= 15 dB) |",
        "|---|---|---|---|",
    ]
    for name, model in scores.items():
        cells = " | ".join(
            "—" if model.accuracy_by_band.get(band) is None
            else f"{model.accuracy_by_band[band]:.3f}"
            for band in ("low", "mid", "high")
        )
        out.append(f"| {name} | {cells} |")
    out.append("")

    ensemble = scores.get("ensemble")
    if ensemble:
        high = ensemble.accuracy_by_band.get("high")
        mid = ensemble.accuracy_by_band.get("mid")
        checks = []
        if high is not None:
            checks.append(
                f"§9 D wants >= 0.85 at SNR >= 15 dB; the ensemble reaches **{high:.3f}** "
                f"({'pass' if high >= 0.85 else 'FAIL'})."
            )
        if mid is not None:
            checks.append(
                f"§9 D wants >= 0.70 at 5-15 dB; the ensemble reaches **{mid:.3f}** "
                f"({'pass' if mid >= 0.70 else 'FAIL'})."
            )
        checks.append(
            "Published results on RadioML 2016.10a plateau around 0.85-0.90, so a number "
            "much above that would mean a leak between the splits, not a better model."
        )
        out += [*(f"- {c}" for c in checks), ""]
    return out


def _per_class(scores: dict[str, ModelScores]) -> list[str]:
    out: list[str] = []
    for name, model in scores.items():
        if not model.per_class:
            continue
        out += [
            f"### Per-class precision and recall — {name}",
            "",
            "| Class | Precision | Recall | Support |",
            "|---|---|---|---|",
        ]
        for entry in sorted(model.per_class, key=lambda c: c.label):
            out.append(
                f"| {entry.label} | {entry.precision:.3f} | {entry.recall:.3f} | "
                f"{entry.support:,} |"
            )
        out.append("")
    return out


def _confusion(scores: dict[str, ModelScores]) -> list[str]:
    model = scores.get("ensemble") or next(iter(scores.values()), None)
    if model is None:
        return []
    out: list[str] = []
    for band, title in _BAND_TITLES:
        entry = model.confusion.get(band)
        if entry is None:
            continue
        labels, matrix = entry
        out += [
            f"### Confusion matrix — {model.name}, {title}",
            "",
            "Rows are truth, columns are prediction; each row sums to 1. "
            "`·` is below 0.01.",
            "",
            "| truth \\ predicted | " + " | ".join(labels) + " |",
            "|---|" + "---|" * len(labels),
        ]
        for i, label in enumerate(labels):
            cells = []
            for j in range(len(labels)):
                value = matrix[i, j]
                if i == j:
                    cells.append(f"**{value:.2f}**")
                elif value >= 0.01:
                    cells.append(f"{value:.2f}")
                else:
                    cells.append("·")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def _calibration(scores: dict[str, ModelScores]) -> list[str]:
    out = [
        "### Calibration check (§9 D)",
        "",
        "§9 D: *among predictions with confidence above 0.8, actual accuracy must exceed "
        "0.8 — this matters more than raw accuracy.* A well-calibrated model's accuracy in "
        "each row tracks that row's confidence band.",
        "",
        "| Model | Confidence bin | Predictions | Actual accuracy |",
        "|---|---|---|---|",
    ]
    for name, model in scores.items():
        for lo, hi, count, accuracy in model.calibration:
            if count:
                out.append(f"| {name} | {lo:.1f}–{hi:.1f} | {count:,} | {accuracy:.3f} |")
    out += [
        "",
        f"| Model | Predictions above {CALIBRATION_THRESHOLD} | Accuracy | Passes §9 D |",
        "|---|---|---|---|",
    ]
    for name, model in scores.items():
        if model.high_confidence_accuracy is None:
            out.append(f"| {name} | 0 | — | no predictions that confident |")
            continue
        passes = model.high_confidence_accuracy > CALIBRATION_THRESHOLD
        out.append(
            f"| {name} | {model.high_confidence_count:,} | "
            f"{model.high_confidence_accuracy:.3f} | {'yes' if passes else '**NO**'} |"
        )
    out.append("")
    return out


def _low_snr(scores: dict[str, ModelScores]) -> list[str]:
    out = [
        "### Low-SNR failure mode (§9 D)",
        "",
        "§9 D: *at SNR < 5 dB, most errors must land on `unknown` rather than on a wrong "
        "confident label.* The §5.5 referee produces `unknown` whenever the blended "
        "probability falls below 0.45.",
        "",
        "| Model | Share of sub-5 dB errors reported as `unknown` |",
        "|---|---|",
    ]
    for name, model in scores.items():
        value = model.unknown_fraction_low_snr
        out.append(f"| {name} | {'—' if value is None else f'{value:.2f}'} |")
    out.append("")
    return out


def _where_we_fail(scores: dict[str, ModelScores]) -> list[str]:
    """Generated from the measured numbers, so it cannot go stale (§10)."""
    model = scores.get("ensemble") or next(iter(scores.values()), None)
    if model is None or not model.accuracy_by_snr:
        return []
    out = ["### Where the classifier fails", ""]

    ordered = sorted(model.accuracy_by_snr.items())
    below = [snr for snr, accuracy in ordered if accuracy < 0.5]
    if below:
        out.append(
            f"- Below **{max(below)} dB** the ensemble is right less than half the time. "
            "That is the failure region, and it is the part of the curve to show a judge "
            "rather than crop."
        )
    worst = sorted(model.per_class, key=lambda c: c.recall)[:3]
    if worst:
        pairs = ", ".join(f"{c.label} ({c.recall:.2f})" for c in worst)
        out.append(f"- Lowest recall by class: {pairs}.")
    confusions = model.confusion.get("high")
    if confusions:
        labels, matrix = confusions
        pairs = []
        for i, actual in enumerate(labels):
            for j, guess in enumerate(labels):
                if i != j and matrix[i, j] >= 0.10:
                    pairs.append(f"{actual} → {guess} ({matrix[i, j]:.2f})")
        if pairs:
            out.append(
                "- Most common confusions even at high SNR: " + "; ".join(pairs[:5]) + "."
            )
    out.append("")
    return out


def render_classification(scores: dict[str, ModelScores]) -> list[str]:
    """Full §9 D classification section as Markdown lines."""
    if not scores:
        return []
    n = next(iter(scores.values())).n_evaluated
    out = [
        "## Classification accuracy (§9 D)",
        "",
        f"Scored on the RadioML 2016.10a **test** split ({n:,} examples), which no model "
        "was trained or early-stopped on. The split is the deterministic 70/15/15 cut "
        "seeded with 26147 (§6.1), published so these numbers are comparable with the "
        "literature.",
        "",
    ]
    out += _accuracy_vs_snr(scores)
    out += _accuracy_by_band(scores)
    out += _per_class(scores)
    out += _confusion(scores)
    out += _calibration(scores)
    out += _low_snr(scores)
    out += _where_we_fail(scores)
    return out
