"""Classification: the §5 chain end to end (CLAUDE.md §5, §9 D).

Covers the §5.2 features, the §5.4 deterministic rules, the §5.5 referee, the §5.6 evidence
sentences, and the §9 D scoring path.

RadioML 2016.10a is a 225 MB download that §2 forbids fetching at runtime, so these tests
build a small dataset in the RadioML *shape* -- ``(n, 2, 128)`` float32, per-example SNR
labels -- from ``testsignals``. That is enough to prove the training and evaluation
machinery is correct end to end, which is the part that would otherwise ship untested. It
is **not** a substitute for the real accuracy numbers, and nothing here asserts an accuracy
figure that could be mistaken for one: the real tables come from
``sigscope train`` + ``sigscope evaluate`` against the published split.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigscope import testsignals as ts
from sigscope.evaluation import (
    CALIBRATION_THRESHOLD,
    calibration_table,
    confusion_matrix,
    evaluate_models,
    score_predictions,
)
from sigscope.features import FEATURE_NAMES, N_FEATURES, extract_features
from sigscope.models import (
    Classifiers,
    CnnClassifier,
    FeatureClassifier,
    Prediction,
    apply_rules,
    build_evidence,
    classify,
    referee,
    snr_band,
)
from sigscope.models.cnn import WINDOW, build_network, sliding_windows
from sigscope.models.ensemble import PRIOR_WEIGHTS, blend_weights
from sigscope.report.accuracy import render_classification, render_unavailable

FS = 200_000.0


# --------------------------------------------------------------------------------------
# a tiny RadioML-shaped dataset
# --------------------------------------------------------------------------------------


def _example(order: int, snr_db: float, seed: int) -> np.ndarray:
    """One 128-sample example in the RadioML ``(2, 128)`` layout."""
    burst = ts.psk(order, 10_000.0, 40_000.0, 64, rng=seed)
    burst = ts.add_awgn(burst, snr_db, rng=seed + 1)[:WINDOW]
    power = float(np.mean(np.abs(burst) ** 2)) or 1.0
    burst = burst / np.sqrt(power)
    return np.stack([burst.real, burst.imag], axis=0).astype(np.float32)


@pytest.fixture(scope="module")
def tiny_dataset():
    """``(iq, modulation, snr_db)`` shaped exactly like the RadioML cache."""
    labels = {"BPSK": 2, "QPSK": 4, "8PSK": 8}
    snrs = [18, 10, 2, -6]
    iq, modulation, snr_db = [], [], []
    seed = 0
    for name, order in labels.items():
        for snr in snrs:
            for _ in range(40):
                seed += 7
                iq.append(_example(order, float(snr), seed))
                modulation.append(name)
                snr_db.append(snr)
    return (
        np.stack(iq).astype(np.float32),
        np.array(modulation, dtype=object),
        np.array(snr_db, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------
# §5.2 features
# --------------------------------------------------------------------------------------


def test_feature_vector_is_the_frozen_length_and_all_finite():
    """A NaN reaching the scaler poisons every downstream tree (§5.2)."""
    for signal in (
        ts.psk(4, 10_000.0, 40_000.0, 64, rng=1),
        ts.fsk(2, 6_000.0, 4_800.0, 96_000.0, 64, rng=2),
        ts.ofdm(64, 16, 8, 1.0, rng=3),
        np.zeros(256, dtype=np.complex64),
    ):
        values = extract_features(np.asarray(signal)[:256])
        assert values.shape == (N_FEATURES,)
        assert np.all(np.isfinite(values))


def test_feature_names_match_the_vector_length():
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(set(FEATURE_NAMES)) == N_FEATURES  # no duplicate column names


def test_cumulant_ratio_orders_the_psk_family():
    """The §5.2 table's ordering is what the classifier keys on."""
    ratios = {}
    for name, order in (("BPSK", 2), ("QPSK", 4), ("8PSK", 8)):
        values = extract_features(np.asarray(ts.psk(order, 10_000.0, 40_000.0, 128, rng=4)))
        ratios[name] = values[FEATURE_NAMES.index("ratio_c40")]
    assert ratios["BPSK"] > ratios["QPSK"] > ratios["8PSK"]


def test_constant_modulus_signal_has_no_amplitude_variation():
    values = extract_features(np.asarray(ts.fsk(2, 6_000.0, 4_800.0, 96_000.0, 128, rng=5)))
    assert values[FEATURE_NAMES.index("sigma_aa")] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# §5.4 deterministic rules -- they override the models, so precision matters most
# --------------------------------------------------------------------------------------


def test_ofdm_rule_fires_on_ofdm():
    hit = apply_rules(np.asarray(ts.ofdm(64, 16, 400, 64_000.0, rng=1)), 64_000.0, snr_db=20.0)
    assert hit is not None and hit.label == "ofdm"
    assert hit.confidence == pytest.approx(0.9)
    assert len(hit.evidence) >= 2


@pytest.mark.parametrize(
    ("name", "signal", "fs"),
    [
        ("QPSK sps=20", ts.psk(4, 10_000.0, 200_000.0, 4000, rng=2), 200_000.0),
        ("QPSK sps=4", ts.psk(4, 10_000.0, 40_000.0, 8000, rng=2), 40_000.0),
        ("BPSK sps=20", ts.psk(2, 10_000.0, 200_000.0, 4000, rng=5), 200_000.0),
        ("2FSK", ts.fsk(2, 6_000.0, 4_800.0, 240_000.0, 3000, rng=3), 240_000.0),
        ("4FSK", ts.fsk(4, 9_600.0, 4_800.0, 240_000.0, 3000, rng=8), 240_000.0),
    ],
)
def test_ofdm_rule_does_not_fire_on_single_carriers(name, signal, fs):
    """A §5.4 rule beats a correct CNN, so a false positive here is the expensive kind.

    Two things had to be true before this passed: §4.11 rejects a correlation peak too
    strong to be a cyclic prefix (2-FSK reaches R = 0.80 on its own tone periodicity), and
    the rule additionally requires a flat spectrum, which a pulse-shaped single carrier
    does not have.
    """
    hit = apply_rules(np.asarray(signal), fs, snr_db=20.0)
    assert (hit is None) or (hit.label != "ofdm"), f"{name} was wrongly called OFDM"


def test_noise_rule_fires_on_an_empty_channel():
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000)).astype(np.complex64)
    hit = apply_rules(noise, 64_000.0, snr_db=1.0)
    assert hit is not None and hit.label == "noise"


def test_noise_rule_needs_low_snr_as_well_as_flatness():
    """§5.4 requires both. Flatness alone would label any wideband signal as noise."""
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000)).astype(np.complex64)
    assert apply_rules(noise, 64_000.0, snr_db=20.0) is None


def test_ssb_rule_fires_on_single_sideband_but_not_on_a_carrier():
    from scipy.signal import hilbert

    fs = 100_000.0
    t = np.arange(int(0.3 * fs)) / fs
    audio = np.cos(2 * np.pi * 1200 * t) * (1 + 0.6 * np.sin(2 * np.pi * 4 * t))
    usb = hilbert(audio).astype(np.complex64)

    assert apply_rules(usb, fs, snr_db=20.0).label == "AM-SSB"
    assert apply_rules(np.conj(usb), fs, snr_db=20.0).label == "AM-SSB"
    # an unmodulated carrier sitting off centre is asymmetric too, but it is not SSB
    tone = apply_rules(np.asarray(ts.tone(1000.0, fs, 20_000)), fs, snr_db=20.0)
    assert tone is None or tone.label != "AM-SSB"


def test_chirp_rule_fires_only_with_the_box():
    """§4.12's ridge needs the detection box; without it the argmax follows whatever is
    loudest anywhere in the band."""
    fs = 2_000_000.0
    chirp = ts.add_awgn(ts.lfm_chirp(-500_000.0, 500_000.0, 0.01, fs), 20.0, rng=1)
    hit = apply_rules(
        chirp, fs, snr_db=20.0, raw_slice=chirp, fs_raw=fs,
        box_f_lo=-600_000.0, box_f_hi=600_000.0,
    )
    assert hit is not None and hit.label == "chirp-lfm"


# --------------------------------------------------------------------------------------
# §5.5 referee
# --------------------------------------------------------------------------------------


def _prediction(label, confidence, method="m"):
    others = {"A": 0.0, "B": 0.0, "C": 0.0}
    others[label] = confidence
    remaining = (1.0 - confidence) / 2.0
    for key in others:
        if key != label:
            others[key] = remaining
    return Prediction(label, confidence, others, method)


def test_a_fired_rule_wins_and_records_the_other_votes():
    """§5.5 step 1."""
    class Hit:
        label, rule, evidence, extra = "ofdm", "ofdm/§5.4", [], {}

    result = referee(_prediction("A", 0.9), _prediction("B", 0.8), rule_hit=Hit(), snr_db=20.0)
    assert result.label == "ofdm"
    assert result.confidence == pytest.approx(0.9)
    assert result.votes == {"feature_clf": "A", "cnn": "B", "rules": "ofdm"}


def test_low_blend_becomes_unknown_with_candidates_kept():
    """§5.5 step 4."""
    result = referee(_prediction("A", 0.40), _prediction("A", 0.40), snr_db=20.0)
    assert result.label == "unknown"
    assert result.runner_up == "A"


def test_model_disagreement_caps_confidence_and_warns():
    """§5.5 step 5."""
    result = referee(_prediction("A", 0.95), _prediction("B", 0.95), snr_db=20.0)
    assert result.confidence <= 0.60
    assert any("disagree" in w for w in result.warnings)


def test_both_abstaining_gives_unclassified_never_a_guess():
    abstain = Prediction(None, 0.0, {}, "m", ["untrained"])
    result = referee(abstain, abstain, snr_db=20.0)
    assert result.label == "unclassified"
    assert result.confidence == 0.0
    assert result.warnings


def test_snr_bands_and_prior_weights_follow_5_5():
    """§5.5: "High SNR favours the feature classifier, low SNR the CNN"."""
    assert snr_band(20.0) == "high"
    assert snr_band(10.0) == "mid"
    assert snr_band(0.0) == "low"
    assert snr_band(None) == "mid"

    high, _ = blend_weights("high", None, None)
    low, is_prior = blend_weights("low", None, None)
    assert is_prior
    assert high[0] > high[1], "high SNR should favour the feature classifier"
    assert low[1] > low[0], "low SNR should favour the CNN"
    assert high == PRIOR_WEIGHTS["high"]


def test_measured_validation_accuracy_replaces_the_prior():
    metadata_f = {"val_accuracy_by_band": {"high": 0.9}}
    metadata_c = {"val_accuracy_by_band": {"high": 0.3}}
    weights, is_prior = blend_weights("high", metadata_f, metadata_c)
    assert not is_prior
    assert weights[0] == pytest.approx(0.75)


# --------------------------------------------------------------------------------------
# §5.6 evidence
# --------------------------------------------------------------------------------------


def test_evidence_always_has_at_least_two_sentences():
    """§3: "always present, minimum two"."""
    for extra in ({}, {"psk_order": 4.0}, {"cumulant_c40_ratio": 1.0, "am_depth": 0.02}):
        assert len(build_evidence("QPSK", snr_db=20.0, extra=extra)) >= 2


def test_low_snr_caution_is_always_emitted():
    """§5.6 calls this "the single most credibility-building line in the product"."""
    sentences = build_evidence("QPSK", snr_db=2.0, extra={"psk_order": 4.0})
    assert any("caution" in s for s in sentences)


def test_cumulant_sentence_quotes_the_theoretical_value():
    sentences = build_evidence("QPSK", snr_db=20.0, extra={"cumulant_c40_ratio": 0.98})
    match = [s for s in sentences if "C40" in s]
    assert match and "1.00" in match[0] and "0.98" in match[0]


def test_cumulant_sentence_says_so_when_the_measurement_disagrees():
    """A measurement that contradicts the label is reported, not suppressed."""
    sentences = build_evidence("QPSK", snr_db=20.0, extra={"cumulant_c40_ratio": 2.0})
    match = [s for s in sentences if "C40" in s]
    assert match and "away from" in match[0]


def test_evidence_never_exceeds_five_sentences():
    sentences = build_evidence(
        "QPSK",
        snr_db=1.0,
        extra={
            "psk_order": 4.0, "cumulant_c40_ratio": 1.0, "am_depth": 0.5,
            "n_tones": 2.0, "fsk_tone_spacing_hz": 1000.0,
            "ofdm_useful_symbol_len": 64.0, "sideband_asymmetry_db": 20.0,
        },
        rule_evidence=["a", "b"],
        ensemble_notes=["c", "d"],
    )
    assert 2 <= len(sentences) <= 5


# --------------------------------------------------------------------------------------
# §5.3 CNN mechanics
# --------------------------------------------------------------------------------------


def test_sliding_windows_cover_a_long_burst_with_50_percent_overlap():
    windows = sliding_windows(np.zeros(1000, dtype=np.complex64))
    assert windows.shape[1:] == (2, WINDOW)
    assert windows.shape[0] > 1


def test_short_burst_is_padded_to_one_window():
    windows = sliding_windows(np.ones(40, dtype=np.complex64))
    assert windows.shape == (1, 2, WINDOW)


def test_network_shape_matches_5_3():
    import torch

    network = build_network(11)
    assert tuple(network(torch.randn(3, 2, WINDOW)).shape) == (3, 11)


def test_untrained_models_abstain_with_a_reason():
    """A fresh clone has no checkpoints; that is a supported state, not an error."""
    for model in (FeatureClassifier(), CnnClassifier()):
        prediction = model.predict(np.asarray(ts.psk(4, 10_000.0, 40_000.0, 64, rng=1)))
        assert prediction.abstained
        assert prediction.notes and "checkpoint" in prediction.notes[0]


def test_classify_works_with_no_checkpoints_at_all():
    """The whole §5 chain runs untrained: a rule can still produce a physical label."""
    result, evidence = classify(
        np.asarray(ts.ofdm(64, 16, 400, 64_000.0, rng=1)), 64_000.0, snr_db=20.0
    )
    assert result.label == "ofdm"
    assert len(evidence) >= 2


# --------------------------------------------------------------------------------------
# §9 D scoring
# --------------------------------------------------------------------------------------


def test_confusion_matrix_rows_sum_to_one():
    labels, matrix = confusion_matrix(
        np.array(["A", "A", "B"]), np.array(["A", "B", "B"]), ["A", "B"]
    )
    assert labels == ["A", "B"]
    assert matrix.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert matrix[0, 0] == pytest.approx(0.5)


def test_calibration_table_bins_by_confidence():
    truth = np.array(["A"] * 10)
    predicted = np.array(["A"] * 5 + ["B"] * 5)
    confidence = np.array([0.9] * 5 + [0.1] * 5)
    table = calibration_table(truth, predicted, confidence)
    top = [row for row in table if row[0] >= 0.8][0]
    assert top[2] == 5 and top[3] == pytest.approx(1.0)


def test_score_predictions_reports_per_snr_never_one_number():
    """§5.3: accuracy per SNR bucket, never a single overall number."""
    truth = np.array(["A", "A", "B", "B"])
    predicted = np.array(["A", "B", "B", "B"])
    snr = np.array([18, -6, 18, -6])
    scores = score_predictions("m", truth, predicted, snr, np.array([0.9, 0.9, 0.9, 0.2]))

    assert set(scores.accuracy_by_snr) == {18, -6}
    assert scores.accuracy_by_snr[18] == pytest.approx(1.0)
    assert not hasattr(scores, "accuracy")  # there is deliberately no overall figure
    assert scores.per_class


def test_high_confidence_accuracy_is_the_9d_calibration_gate():
    truth = np.array(["A"] * 4)
    predicted = np.array(["A", "A", "A", "B"])
    scores = score_predictions(
        "m", truth, predicted, np.array([18] * 4), np.array([0.9, 0.9, 0.9, 0.9])
    )
    assert scores.high_confidence_count == 4
    assert scores.high_confidence_accuracy == pytest.approx(0.75)
    assert CALIBRATION_THRESHOLD == 0.8


# --------------------------------------------------------------------------------------
# the training -> evaluation -> report path, end to end
# --------------------------------------------------------------------------------------


def test_train_score_and_render_end_to_end(tiny_dataset, tmp_path):
    """Fit both models on synthetic data and drive the whole §9 D reporting path.

    This is a *machinery* test, not an accuracy claim -- see the module docstring. It
    exists because the training script and the evaluation tables would otherwise ship
    having never been executed.
    """
    import torch

    from sigscope.features import extract_feature_matrix

    iq, modulation, snr_db = tiny_dataset
    n = iq.shape[0]
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    train_idx, test_idx = order[: int(0.7 * n)], order[int(0.7 * n) :]

    # §5.2 feature classifier
    features = extract_feature_matrix(iq[train_idx])
    classifier = FeatureClassifier().fit(
        features, modulation[train_idx], max_iter=30, random_state=0
    )
    assert classifier.is_trained
    saved = classifier.save(tmp_path / "feature_clf.joblib")
    assert FeatureClassifier.load(saved).is_trained

    # §5.3 CNN, two epochs -- enough to exercise the loop and the checkpoint
    classes = tuple(sorted(set(modulation.tolist())))
    index_of = {label: i for i, label in enumerate(classes)}
    network = build_network(len(classes))
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    batch = torch.from_numpy(iq[train_idx])
    labels = torch.from_numpy(
        np.array([index_of[m] for m in modulation[train_idx]], dtype=np.int64)
    )
    for _ in range(2):
        optimiser.zero_grad()
        criterion(network(batch), labels).backward()
        optimiser.step()
    network.eval()
    cnn = CnnClassifier(network=network, classes=classes, metadata={})
    cnn_path = cnn.save(tmp_path / "cnn.pt")
    assert CnnClassifier.load(cnn_path).is_trained

    # §9 D scoring across all three voters
    scores = evaluate_models(
        iq[test_idx], modulation[test_idx], snr_db[test_idx],
        Classifiers(feature=classifier, cnn=cnn),
    )
    assert set(scores) == {"feature_clf", "cnn", "ensemble"}
    for model in scores.values():
        assert model.accuracy_by_snr, "per-SNR accuracy is the whole point (§5.3)"
        assert model.per_class
        assert model.confusion
        assert model.calibration

    # the report renders and is a well-formed table
    lines = render_classification(scores)
    text = "\n".join(lines)
    assert "Accuracy vs SNR" in text
    assert "Calibration check" in text
    assert "Confusion matrix" in text
    assert "Per-class precision and recall" in text
    for row in [line for line in lines if line.startswith("|")]:
        assert row.endswith("|")


def test_render_unavailable_says_what_is_missing():
    """A silently absent table reads like a measured-and-bad one; it must not."""
    text = "\n".join(render_unavailable(["the CNN has no checkpoint."]))
    assert "Not measured" in text
    assert "the CNN has no checkpoint." in text
    assert "sigscope train" in text
