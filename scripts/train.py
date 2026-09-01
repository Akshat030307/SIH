"""Train the §5.2 feature classifier and the §5.3 CNN (CLAUDE.md §5, §8 Phase 6).

**Training data is the RadioML train split and nothing else** (§6.1). The split is the
deterministic 70/15/15 cut in ``sigscope.data.radioml`` seeded with ``SPLIT_SEED = 26147``,
which is published so our numbers are comparable with the literature. The val split tunes
early stopping and produces the per-SNR-band accuracies the §5.5 referee weights by; the
test split is never touched here -- ``scripts/evaluate.py`` owns it.

Run with::

    sigscope fetch-data --src <path to RML2016.10a_dict.pkl>
    sigscope train --model both

Each checkpoint carries the metadata the rest of §5 needs and cannot invent:

* ``val_accuracy_by_band`` -- accuracy in the §9 D SNR bands, which is what §5.5 step 2
  means by "weighted by their validation accuracy at the measured SNR bucket". Without it
  the ensemble falls back to a documented prior and says so.
* ``val_accuracy_by_class_low_snr`` -- per-class accuracy below 5 dB, which fills §5.6's
  "below 5 dB our accuracy for this class drops to {a:.0%}" sentence with a measured
  number rather than a plausible-looking one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sigscope.data import radioml  # noqa: E402
from sigscope.features import extract_feature_matrix  # noqa: E402
from sigscope.models.cnn import DEFAULT_CNN_PATH, CnnClassifier, build_network  # noqa: E402
from sigscope.models.ensemble import SNR_BANDS, snr_band  # noqa: E402
from sigscope.models.feature_clf import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FeatureClassifier,
)

LOW_SNR_DB = 5.0  # §9 D / §5.6 boundary


# --------------------------------------------------------------------------------------
# shared scoring
# --------------------------------------------------------------------------------------


def accuracy_by_band(
    truth: np.ndarray, predicted: np.ndarray, snr_db: np.ndarray
) -> dict[str, float]:
    """Accuracy in each §9 D SNR band -- never a single overall number (§5.3)."""
    out: dict[str, float] = {}
    bands = np.array([snr_band(float(s)) for s in snr_db])
    for name, _, _ in SNR_BANDS:
        mask = bands == name
        if int(np.count_nonzero(mask)) == 0:
            continue
        out[name] = float(np.mean(truth[mask] == predicted[mask]))
    return out


def accuracy_by_class_low_snr(
    truth: np.ndarray, predicted: np.ndarray, snr_db: np.ndarray
) -> dict[str, float]:
    """Per-class accuracy below 5 dB, for the §5.6 caution sentence."""
    low = snr_db < LOW_SNR_DB
    out: dict[str, float] = {}
    for label in np.unique(truth):
        mask = low & (truth == label)
        if int(np.count_nonzero(mask)) == 0:
            continue
        out[str(label)] = float(np.mean(predicted[mask] == label))
    return out


def accuracy_by_snr(
    truth: np.ndarray, predicted: np.ndarray, snr_db: np.ndarray
) -> dict[str, float]:
    """Accuracy at each individual SNR level, for the ACCURACY.md curve."""
    return {
        str(int(s)): float(np.mean(truth[snr_db == s] == predicted[snr_db == s]))
        for s in sorted(np.unique(snr_db))
    }


def _metadata(
    truth: np.ndarray, predicted: np.ndarray, snr_db: np.ndarray, **extra: object
) -> dict:
    return {
        "split_seed": radioml.SPLIT_SEED,
        "trained_on": "RadioML 2016.10a train split only (CLAUDE.md §6.1)",
        "val_accuracy_by_band": accuracy_by_band(truth, predicted, snr_db),
        "val_accuracy_by_snr": accuracy_by_snr(truth, predicted, snr_db),
        "val_accuracy_by_class_low_snr": accuracy_by_class_low_snr(truth, predicted, snr_db),
        **extra,
    }


# --------------------------------------------------------------------------------------
# feature classifier (§5.2)
# --------------------------------------------------------------------------------------


def train_features(
    data: radioml.RadioMLDataset, *, out: Path, max_iter: int, seed: int
) -> dict:
    """Fit the §5.2 scaler + gradient boosting on the train split."""
    train_idx = data.split("train")
    val_idx = data.split("val")
    print(f"  feature classifier: {len(train_idx):,} train / {len(val_idx):,} val examples")

    started = time.perf_counter()
    print("    extracting training features...", flush=True)
    x_train = extract_feature_matrix(data.iq[train_idx], progress=20000)
    y_train = data.modulation[train_idx]
    print("    extracting validation features...", flush=True)
    x_val = extract_feature_matrix(data.iq[val_idx], progress=20000)
    y_val = data.modulation[val_idx]
    snr_val = data.snr_db[val_idx]
    print(f"    features done in {time.perf_counter() - started:.0f} s", flush=True)

    classifier = FeatureClassifier().fit(
        x_train, y_train, max_iter=max_iter, random_state=seed
    )
    predicted = classifier.model.predict(classifier.scaler.transform(x_val))

    importance = classifier.permutation_importance(x_val[:4000], y_val[:4000])
    metadata = _metadata(
        y_val,
        predicted,
        snr_val,
        n_train=int(len(train_idx)),
        permutation_importance=importance,
        max_iter=max_iter,
        train_seconds=round(time.perf_counter() - started, 1),
    )
    classifier.metadata = metadata
    path = classifier.save(out)

    print(f"    saved {path}")
    for band, value in metadata["val_accuracy_by_band"].items():
        print(f"      val accuracy [{band:>4} SNR]: {value:.3f}")
    return metadata


# --------------------------------------------------------------------------------------
# CNN (§5.3)
# --------------------------------------------------------------------------------------


def train_cnn(
    data: radioml.RadioMLDataset,
    *,
    out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    patience: int,
) -> dict:
    """Train the §5.3 residual CNN on the train split.

    §5.3's recipe: Adam at 1e-3, cosine schedule, batch 256, 40 epochs, early stop on val
    loss, label smoothing 0.05. CPU-only is fine and is what the demo laptop will do.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # torch owns every source of randomness that matters here: weight init, dropout and
    # the DataLoader's shuffle. numpy is only used for deterministic indexing.
    torch.manual_seed(seed)

    train_idx = data.split("train")
    val_idx = data.split("val")
    classes = tuple(sorted(np.unique(data.modulation)))
    index_of = {label: i for i, label in enumerate(classes)}
    print(f"  cnn: {len(train_idx):,} train / {len(val_idx):,} val examples, "
          f"{len(classes)} classes")

    def tensors(idx: np.ndarray) -> TensorDataset:
        raw = np.asarray(data.iq[idx], dtype=np.float32)
        # §5.3: power-normalise each example, matching inference
        power = np.mean(raw[:, 0] ** 2 + raw[:, 1] ** 2, axis=1, keepdims=True)
        raw = raw / np.sqrt(np.maximum(power, 1e-12))[:, None, :]
        labels = np.array([index_of[m] for m in data.modulation[idx]], dtype=np.int64)
        return TensorDataset(torch.from_numpy(raw), torch.from_numpy(labels))

    train_loader = DataLoader(
        tensors(train_idx), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(tensors(val_idx), batch_size=512, shuffle=False)

    network = build_network(len(classes))
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_loss, best_state, stale = float("inf"), None, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        network.train()
        total = 0.0
        for batch, labels in train_loader:
            optimiser.zero_grad()
            loss = criterion(network(batch), labels)
            loss.backward()
            optimiser.step()
            total += float(loss) * batch.shape[0]
        schedule.step()

        network.eval()
        val_loss, correct, seen = 0.0, 0, 0
        with torch.no_grad():
            for batch, labels in val_loader:
                logits = network(batch)
                val_loss += float(criterion(logits, labels)) * batch.shape[0]
                correct += int((logits.argmax(1) == labels).sum())
                seen += batch.shape[0]
        val_loss /= max(seen, 1)
        print(
            f"    epoch {epoch:>3}/{epochs}  train {total / len(train_idx):.4f}  "
            f"val {val_loss:.4f}  acc {correct / max(seen, 1):.3f}  "
            f"({time.perf_counter() - started:.0f} s)",
            flush=True,
        )

        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"    early stop: val loss has not improved for {patience} epochs")
                break

    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()

    predicted_idx = []
    with torch.no_grad():
        for batch, _ in val_loader:
            predicted_idx.append(network(batch).argmax(1).numpy())
    predicted = np.array(classes)[np.concatenate(predicted_idx)]

    metadata = _metadata(
        data.modulation[val_idx],
        predicted,
        data.snr_db[val_idx],
        n_train=int(len(train_idx)),
        epochs_run=epoch,
        best_val_loss=best_loss,
        parameters=int(sum(p.numel() for p in network.parameters())),
        train_seconds=round(time.perf_counter() - started, 1),
    )
    path = CnnClassifier(network=network, classes=classes, metadata=metadata).save(out)

    print(f"    saved {path}")
    for band, value in metadata["val_accuracy_by_band"].items():
        print(f"      val accuracy [{band:>4} SNR]: {value:.3f}")
    return metadata


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["feature", "cnn", "both"], default="both")
    parser.add_argument("--data", default=str(radioml.DEFAULT_ROOT))
    parser.add_argument("--feature-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--cnn-out", default=str(DEFAULT_CNN_PATH))
    parser.add_argument("--epochs", type=int, default=40, help="§5.3 default")
    parser.add_argument("--batch-size", type=int, default=256, help="§5.3 default")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="§5.3 default")
    parser.add_argument("--max-iter", type=int, default=300, help="§5.2 default")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary", default=None, help="write the metadata to this JSON")
    args = parser.parse_args(argv)

    try:
        data = radioml.load(args.data)
    except FileNotFoundError as exc:
        print(f"train: {exc}")
        return 1

    print(
        f"RadioML cache: {len(data):,} examples, "
        f"{len(np.unique(data.modulation))} modulations, "
        f"{len(np.unique(data.snr_db))} SNR levels (split seed {radioml.SPLIT_SEED})"
    )

    summary: dict = {}
    if args.model in ("feature", "both"):
        summary["feature_clf"] = train_features(
            data, out=Path(args.feature_out), max_iter=args.max_iter, seed=args.seed
        )
    if args.model in ("cnn", "both"):
        summary["cnn"] = train_cnn(
            data,
            out=Path(args.cnn_out),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            patience=args.patience,
        )

    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
