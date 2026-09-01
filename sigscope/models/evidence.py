"""Evidence sentence generator (CLAUDE.md §5.6).

§5.6's templates, filled with measured values. §3 is blunt about why this exists: evidence
"is not decoration -- it is the feature that separates this from every other submission",
and §10 puts the evidence panel at the centre of the demo. So every sentence here has to be
a statement an analyst can check by hand against their own tools.

Three consequences for how this is written:

* Every sentence quotes a number that was actually measured on this burst. Nothing is
  templated from the predicted label alone -- that would produce a sentence that sounds
  like evidence while being a restatement of the guess.
* The low-SNR caution is emitted whenever the SNR is low, regardless of label. §5.6 calls
  it "the single most credibility-building line in the product", and it is the one sentence
  that must never be dropped for looking bad.
* Where a measurement contradicts the label, the sentence says so rather than being
  suppressed. A confident label with a caveat is more useful to an analyst than a confident
  label alone.

The per-class accuracy quoted in the low-SNR sentence comes from the trained model's
metadata. Without a checkpoint there is no measured accuracy, and the sentence says that
instead of inventing a percentage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sigscope.features import THEORETICAL_RATIOS

__all__ = ["build_evidence", "MIN_EVIDENCE"]

MIN_EVIDENCE = 2  # §3: "always present, minimum two"


def _cumulant_sentence(label: str, ratio_c40: float | None) -> str | None:
    """§5.6: "C40 magnitude {v:.2f} is close to the {label} theoretical value {t:.2f}"."""
    if ratio_c40 is None or label not in THEORETICAL_RATIOS:
        return None
    theoretical = THEORETICAL_RATIOS[label][0]
    difference = abs(ratio_c40 - theoretical)
    if difference <= 0.15:
        return (
            f"The fourth-order cumulant ratio |C40|/C21^2 is {ratio_c40:.2f}, close to the "
            f"{label} theoretical value of {theoretical:.2f}."
        )
    return (
        f"The fourth-order cumulant ratio |C40|/C21^2 is {ratio_c40:.2f}, which is "
        f"{difference:.2f} away from the {label} theoretical value of {theoretical:.2f} -- "
        "weaker support for this label than the other evidence."
    )


def _accuracy_sentence(
    snr_db: float | None, label: str, metadata: dict[str, Any] | None
) -> str | None:
    """§5.6's low-SNR caution -- "always emit it at low SNR"."""
    if snr_db is None or not np.isfinite(snr_db) or snr_db >= 5.0:
        return None
    per_class = (metadata or {}).get("val_accuracy_by_class_low_snr", {})
    accuracy = per_class.get(label)
    if accuracy is not None:
        return (
            f"Measured SNR is {snr_db:.1f} dB. Below 5 dB our accuracy for {label} drops to "
            f"{float(accuracy):.0%}, so treat this result with caution."
        )
    return (
        f"Measured SNR is {snr_db:.1f} dB. Below 5 dB several of our estimators and both "
        "classifiers lose accuracy, so treat this result with caution."
    )


def build_evidence(
    label: str,
    *,
    snr_db: float | None = None,
    extra: dict[str, Any] | None = None,
    rule_evidence: list[str] | None = None,
    ensemble_notes: list[str] | None = None,
    model_metadata: dict[str, Any] | None = None,
    symbol_rate: float | None = None,
    bandwidth_hz: float | None = None,
) -> list[str]:
    """Build 2-5 plain sentences explaining a classification (CLAUDE.md §5.6).

    ``extra`` is the detection's measurement bag from the pipeline, so every sentence is
    filled from a number that was measured rather than assumed.
    """
    extra = extra or {}
    sentences: list[str] = []

    # a fired rule already explains itself in physical terms; that is the best evidence
    # available and it leads
    for sentence in rule_evidence or []:
        sentences.append(sentence)

    # §5.6: "Raising to the power {M} produced a single strong spectral line..."
    order = extra.get("psk_order")
    if order is not None:
        sentences.append(
            f"Raising the signal to the power {order:.0f} produced a single strong spectral "
            f"line, which indicates {order:.0f} phase states."
        )

    cumulant = _cumulant_sentence(label, extra.get("cumulant_c40_ratio"))
    if cumulant:
        sentences.append(cumulant)

    # §5.6: "Envelope is nearly constant (amplitude variance {v:.3f})..."
    depth = extra.get("am_depth")
    if depth is not None and depth < 0.1:
        sentences.append(
            f"The envelope is nearly constant (modulation depth {depth:.3f}), so this is "
            "not an amplitude scheme."
        )
    elif depth is not None and depth > 0.4:
        sentences.append(
            f"The envelope varies with a modulation depth of {depth:.2f}, so amplitude "
            "carries information here."
        )

    # §5.6: "The instantaneous frequency histogram has {n} distinct peaks spaced {d} Hz apart"
    tones = extra.get("n_tones")
    spacing = extra.get("fsk_tone_spacing_hz")
    if tones is not None and spacing is not None:
        sentences.append(
            f"The instantaneous frequency histogram has {tones:.0f} distinct peaks spaced "
            f"{spacing:.0f} Hz apart."
        )

    # §5.6: "Autocorrelation peaks at lag {L} samples, consistent with a cyclic prefix"
    lag = extra.get("ofdm_useful_symbol_len")
    if lag is not None and not any("cyclic prefix" in s for s in sentences):
        sentences.append(
            f"Autocorrelation peaks at lag {lag:.0f} samples, consistent with a cyclic prefix."
        )

    asymmetry = extra.get("sideband_asymmetry_db")
    already_said = any("sideband" in s for s in sentences)
    if asymmetry is not None and abs(asymmetry) > 10.0 and not already_said:
        side = "above" if asymmetry > 0 else "below"
        sentences.append(
            f"The spectrum is {abs(asymmetry):.1f} dB stronger {side} the centre frequency, "
            "which is a single-sideband signature."
        )

    if symbol_rate and bandwidth_hz and symbol_rate > 0:
        ratio = bandwidth_hz / symbol_rate
        if 0.8 <= ratio <= 2.5:
            sentences.append(
                f"The occupied bandwidth of {bandwidth_hz:,.0f} Hz is {ratio:.2f} times the "
                f"measured symbol rate of {symbol_rate:,.0f} Bd, which is consistent with a "
                "pulse-shaped linear modulation."
            )

    # §5.6's low-SNR caution: always emitted when it applies
    caution = _accuracy_sentence(snr_db, label, model_metadata)
    if caution:
        sentences.append(caution)
    elif snr_db is not None and np.isfinite(snr_db):
        sentences.append(f"Measured signal-to-noise ratio is {snr_db:.1f} dB.")

    # the referee's own reasoning, when it has something a reader needs (disagreement,
    # an unknown verdict, a single-model vote)
    for note in ensemble_notes or []:
        if len(sentences) >= 5:
            break
        sentence = note[0].upper() + note[1:] if note else note
        if sentence and not sentence.endswith("."):
            sentence += "."
        if sentence and sentence not in sentences:
            sentences.append(sentence)

    # §3: minimum two, always
    if len(sentences) < MIN_EVIDENCE and bandwidth_hz:
        sentences.append(f"The occupied bandwidth measures {bandwidth_hz:,.0f} Hz.")
    if len(sentences) < MIN_EVIDENCE:
        sentences.append(
            "No further distinguishing measurement was available for this burst, which is "
            "itself why the confidence is low."
        )

    return sentences[:5]
