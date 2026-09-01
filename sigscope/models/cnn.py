"""Residual CNN on raw IQ (CLAUDE.md §5.3).

The §5.3 architecture exactly::

    Conv1d(2, 64, k=7, pad=3) -> BN -> ReLU
    2 x ResidualBlock(64, k=5)     each: Conv-BN-ReLU-Conv-BN + skip, then MaxPool(2)
    2 x ResidualBlock(128, k=5)
    AdaptiveAvgPool1d(1) -> Dropout(0.3) -> Linear(128, n_classes)

Input is ``2 x 128`` float32, I and Q as channels, power-normalised -- matching the dataset
(§6.1). CPU-trainable in well under an hour on 220k short examples, which is what §10 means
by "it runs on the laptop in front of you".

The layer stack above is §5.3's verbatim; built as written it comes to **381,771**
parameters rather than the "~250k" §5.3 estimates. The two 128-channel blocks account for
most of it (a 128->128 convolution with kernel 5 is 82k weights on its own). The
architecture is not changed to chase the quoted figure -- it is still small, still trains on
a CPU, and §10's claim of "about 250k model parameters" is the number that should be
corrected to 380k before anyone says it to a judge who might add it up.

**Sliding-window inference** (§5.3): a real burst is far longer than 128 samples, so a
128-sample window slides across it with 50% overlap, the softmax outputs are averaged, and
the *spread* across windows is reported as an extra confidence signal -- high disagreement
between windows means low confidence. That spread is genuinely informative: a clean burst
gives near-identical windows, while a burst that is half signal and half noise does not,
and the ensemble should not treat those two cases alike.

Torch is imported lazily. §2 puts torch in the dependency set, but importing it costs about
a second, and `sigscope --help` and the whole classical DSP path have no use for it.

Like the feature classifier, an untrained instance is a supported state: it abstains with a
reason rather than raising or guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sigscope.models.feature_clf import TRAINED_CLASSES, Prediction

__all__ = ["WINDOW", "IqCnn", "CnnClassifier", "DEFAULT_CNN_PATH", "build_network"]

WINDOW = 128  # §6.1: each dataset example is 128 complex samples
DEFAULT_CNN_PATH = Path(__file__).resolve().parent / "checkpoints" / "cnn.pt"


def build_network(n_classes: int = len(TRAINED_CLASSES)) -> Any:
    """Construct the §5.3 network. Imports torch lazily."""
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        """Conv-BN-ReLU-Conv-BN + skip, then MaxPool(2) (§5.3)."""

        def __init__(self, in_channels: int, out_channels: int, kernel: int = 5) -> None:
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel, padding=pad)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, padding=pad)
            self.bn2 = nn.BatchNorm1d(out_channels)
            # 1x1 projection only where the channel count changes, so the skip stays cheap
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv1d(in_channels, out_channels, 1)
            )
            self.pool = nn.MaxPool1d(2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):  # noqa: ANN001, ANN201
            identity = self.skip(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.pool(self.relu(out + identity))

    class IqCnnModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                ResidualBlock(64, 64),
                ResidualBlock(64, 64),
                ResidualBlock(64, 128),
                ResidualBlock(128, 128),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):  # noqa: ANN001, ANN201
            return self.head(self.blocks(self.stem(x)))

    torch.manual_seed(0)
    return IqCnnModule()


# exported under the §5.3 name as well
IqCnn = build_network


@dataclass
class WindowedPrediction:
    """Averaged softmax over sliding windows, plus the disagreement between them."""

    probabilities: np.ndarray
    n_windows: int
    spread: float


def to_channels(y: np.ndarray) -> np.ndarray:
    """Complex burst -> ``(2, n)`` float32, power-normalised (§5.3 input format)."""
    y = np.asarray(y, dtype=np.complex128).ravel()
    power = float(np.mean(np.abs(y) ** 2))
    if power > 1e-30:
        y = y / np.sqrt(power)
    return np.stack([y.real, y.imag], axis=0).astype(np.float32)


def sliding_windows(y: np.ndarray, window: int = WINDOW, overlap: float = 0.5) -> np.ndarray:
    """Cut a burst into ``(n_windows, 2, window)`` with the §5.3 50% overlap.

    A burst shorter than one window is zero-padded to exactly one window rather than
    dropped -- the caller has already decided the burst is worth classifying, and the
    padding is visible to the network as low power rather than as signal.
    """
    channels = to_channels(y)
    n = channels.shape[1]
    if n < window:
        padded = np.zeros((2, window), dtype=np.float32)
        padded[:, :n] = channels
        return padded[None, ...]

    step = max(1, int(window * (1.0 - overlap)))
    starts = list(range(0, n - window + 1, step))
    if starts[-1] + window < n:  # keep the tail
        starts.append(n - window)
    return np.stack([channels[:, s : s + window] for s in starts], axis=0)


class CnnClassifier:
    """The §5.3 network with sliding-window inference and honest abstention."""

    method = "cnn/§5.3"

    def __init__(
        self,
        network: Any = None,
        classes: tuple[str, ...] = TRAINED_CLASSES,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.network = network
        self.classes = tuple(classes)
        self.metadata = metadata or {}

    @property
    def is_trained(self) -> bool:
        return self.network is not None

    @classmethod
    def load(cls, path: str | Path | None = None) -> CnnClassifier:
        """Load a checkpoint, or return an untrained instance if there is none."""
        path = Path(path or DEFAULT_CNN_PATH)
        if not path.is_file():
            return cls()
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        classes = tuple(blob["classes"])
        network = build_network(len(classes))
        network.load_state_dict(blob["state_dict"])
        network.eval()
        return cls(network=network, classes=classes, metadata=blob.get("metadata", {}))

    def save(self, path: str | Path | None = None) -> Path:
        if not self.is_trained:
            raise RuntimeError("refusing to save an untrained network")
        import torch

        path = Path(path or DEFAULT_CNN_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "classes": list(self.classes),
                "metadata": self.metadata,
            },
            path,
        )
        return path

    # ---------------------------------------------------------------- inference
    def predict_windows(self, y: np.ndarray, *, batch_size: int = 256) -> WindowedPrediction:
        """Average the softmax across sliding windows and measure their disagreement."""
        import torch

        windows = sliding_windows(y)
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, windows.shape[0], batch_size):
                batch = torch.from_numpy(windows[start : start + batch_size])
                outputs.append(torch.softmax(self.network(batch), dim=1).numpy())
        probabilities = np.concatenate(outputs, axis=0)
        mean = probabilities.mean(axis=0)
        # spread: mean per-window distance from the average distribution. 0 means every
        # window agrees; large means the burst is not homogeneous.
        spread = float(np.mean(np.abs(probabilities - mean).sum(axis=1) / 2.0))
        return WindowedPrediction(
            probabilities=mean, n_windows=int(windows.shape[0]), spread=spread
        )

    def predict(self, y: np.ndarray, **_: Any) -> Prediction:
        """Classify one burst. Abstains, with a reason, when there is no checkpoint."""
        if not self.is_trained:
            return Prediction(
                label=None,
                confidence=0.0,
                method=self.method,
                notes=[
                    "the CNN has no trained checkpoint; run `sigscope fetch-data` then "
                    "`sigscope train --model cnn`"
                ],
            )
        y = np.asarray(y)
        if y.size < 16:
            return Prediction(
                label=None, confidence=0.0, method=self.method,
                notes=[f"burst is only {y.size} samples; too short to classify"],
            )

        result = self.predict_windows(y)
        order = int(np.argmax(result.probabilities))
        notes = [
            f"averaged over {result.n_windows} sliding window(s); "
            f"window disagreement {result.spread:.2f}"
        ]
        # §5.3: high disagreement between windows means low confidence
        confidence = float(result.probabilities[order]) * float(max(0.0, 1.0 - result.spread))
        if result.spread > 0.5:
            notes.append(
                "the sliding windows disagree strongly, so this burst may not be a single "
                "homogeneous signal"
            )
        return Prediction(
            label=str(self.classes[order]),
            confidence=confidence,
            probabilities={
                str(c): float(p)
                for c, p in zip(self.classes, result.probabilities, strict=True)
            },
            method=self.method,
            notes=notes,
        )
