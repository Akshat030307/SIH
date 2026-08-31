"""RadioML 2016.10a loader and one-time converter (CLAUDE.md §6.1).

The DeepSig RML2016.10a distributable is a Python pickle: a ``dict`` keyed
``(modulation_name, snr_db)`` -> ``float32`` array of shape ``(1000, 2, 128)`` (I on
row 0, Q on row 1). 11 modulations x 20 SNR levels (-20..+18 dB, 2 dB steps) = 220
keys = 220 000 examples.

``convert()`` reads that pickle exactly once: it verifies a SHA-256, checks the
structure (220 keys, each ``(1000, 2, 128)``), stacks everything into a single
memory-mapped ``iq.npy`` and writes a ``labels.parquet`` with columns
``index, modulation, snr_db`` plus a ``meta.json`` sidecar. Training and evaluation
call ``load()``, which only ever touches the cache — the pickle is never re-parsed.

Splits (§6.1): a fixed permutation seeded with ``SPLIT_SEED`` cut 70/15/15
train/val/test. Report on the test split only.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sigscope import __version__

# ---------------------------------------------------------------------------
# Structure of the published dataset (CLAUDE.md §6.1). Referenced at call time so
# tests can shrink them via monkeypatch.
MODULATIONS: tuple[str, ...] = (
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
)
SNRS: tuple[int, ...] = tuple(range(-20, 19, 2))  # -20, -18, ..., +18  (20 levels)
N_PER_CELL = 1000
EXAMPLE_SHAPE: tuple[int, int] = (2, 128)
N_KEYS = 220
N_TOTAL = N_KEYS * N_PER_CELL

# Split (CLAUDE.md §6.1 "Splits" -- publish the seed).
SPLIT_SEED = 26147
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "radioml"

_IQ_NAME = "iq.npy"
_LABELS_NAME = "labels.parquet"
_META_NAME = "meta.json"


@dataclass
class ConvertResult:
    """Summary of a completed (or cached) conversion."""

    root: Path
    sha256: str
    n_examples: int
    modulations: tuple[str, ...]
    snrs: tuple[int, ...]
    cached: bool


@dataclass
class RadioMLDataset:
    """A loaded RadioML cache. ``iq`` is a read-only memmap of shape ``(N, 2, 128)``."""

    iq: np.ndarray
    modulation: np.ndarray  # (N,) str
    snr_db: np.ndarray  # (N,) int
    root: Path

    def __len__(self) -> int:
        return int(self.iq.shape[0])

    def where(
        self,
        *,
        modulation: str | None = None,
        snr_db: int | None = None,
    ) -> np.ndarray:
        """Row indices matching an optional modulation and/or SNR filter."""
        mask = np.ones(len(self), dtype=bool)
        if modulation is not None:
            mask &= self.modulation == modulation
        if snr_db is not None:
            mask &= self.snr_db == snr_db
        return np.nonzero(mask)[0]

    def complex_iq(self, idx) -> np.ndarray:
        """Return example(s) ``idx`` as ``complex64`` -- shape ``(..., 128)``."""
        a = np.asarray(self.iq[idx], dtype=np.float32)
        return (a[..., 0, :] + 1j * a[..., 1, :]).astype(np.complex64)

    def split(self, which: str) -> np.ndarray:
        """Sorted row indices for ``"train"`` / ``"val"`` / ``"test"`` (§6.1)."""
        return split_indices(which, len(self))


# ---------------------------------------------------------------------------
# Splits


def split_indices(
    which: str,
    n: int = N_TOTAL,
    *,
    seed: int = SPLIT_SEED,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> np.ndarray:
    """Deterministic 70/15/15 split (CLAUDE.md §6.1). Returns sorted indices."""
    if which not in ("train", "val", "test"):
        raise ValueError(f"unknown split {which!r}; expected train/val/test")
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    parts = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    return np.sort(parts[which])


# ---------------------------------------------------------------------------
# Hashing


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Conversion (runs once)


def _normalise_key(key: object) -> tuple[str, int]:
    """``(b"QPSK", 8)`` or ``("QPSK", 8)`` -> ``("QPSK", 8)``."""
    if not (isinstance(key, tuple) and len(key) == 2):
        raise ValueError(f"pickle key {key!r} is not a (modulation, snr) pair")
    mod, snr = key
    mod = mod.decode() if isinstance(mod, bytes) else str(mod)
    return mod, int(snr)


def validate_raw(obj: object) -> None:
    """Check the pickle matches the published structure (CLAUDE.md §6.1)."""
    if not isinstance(obj, dict):
        raise ValueError(f"expected a dict pickle, got {type(obj).__name__}")
    if len(obj) != N_KEYS:
        raise ValueError(f"expected {N_KEYS} keys, got {len(obj)}")
    want = (N_PER_CELL, *EXAMPLE_SHAPE)
    mods: set[str] = set()
    snrs: set[int] = set()
    for key, value in obj.items():
        mod, snr = _normalise_key(key)
        mods.add(mod)
        snrs.add(snr)
        arr = np.asarray(value)
        if arr.shape != want:
            raise ValueError(f"key {key!r}: shape {arr.shape}, expected {want}")
    if len(mods) * len(snrs) != N_KEYS:
        raise ValueError(
            f"{len(mods)} modulations x {len(snrs)} SNRs != {N_KEYS} keys "
            "(keys are not a full modulation x SNR grid)"
        )


def _resolve_source(src: str | Path, work_dir: Path) -> Path:
    """Return a local path to the ``.pkl``, downloading and/or un-tarring as needed."""
    src_str = str(src)
    if "://" in src_str:
        dest = work_dir / (Path(urllib.parse.urlparse(src_str).path).name or "radioml_download.pkl")
        if not dest.exists():
            print(f"downloading {src_str}\n       -> {dest}")
            with urllib.request.urlopen(src_str) as resp, open(dest, "wb") as out:  # noqa: S310
                shutil.copyfileobj(resp, out)
        local = dest
    else:
        local = Path(src).expanduser()
        if not local.exists():
            raise FileNotFoundError(f"no such file: {local}")

    if tarfile.is_tarfile(local):
        with tarfile.open(local) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".pkl")]
            if not members:
                raise ValueError(f"{local} is a tarball with no .pkl member")
            tf.extract(members[0], work_dir, filter="data")
            local = work_dir / members[0].name
    return local


def _load_pickle(path: Path) -> dict:
    """Load the RadioML pickle (Python-2 protocol-0 -> ``encoding='latin1'``)."""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh, encoding="latin1")
    except (pickle.UnpicklingError, EOFError, ValueError) as exc:
        raise ValueError(
            f"{path} is not a readable RadioML pickle ({exc}). "
            "Expected RML2016.10a_dict.pkl."
        ) from exc


def _stack(raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all cells in sorted (modulation, snr) order."""
    keys = sorted(raw.keys(), key=_normalise_key)
    parts: list[np.ndarray] = []
    mods: list[str] = []
    snrs: list[int] = []
    for key in keys:
        mod, snr = _normalise_key(key)
        arr = np.ascontiguousarray(raw[key], dtype=np.float32)
        parts.append(arr)
        mods.extend([mod] * len(arr))
        snrs.extend([snr] * len(arr))
    iq = np.concatenate(parts, axis=0)
    return iq, np.asarray(mods, dtype=object), np.asarray(snrs, dtype=np.int64)


def convert(
    src: str | Path,
    out_dir: str | Path = DEFAULT_ROOT,
    *,
    sha256: str | None = None,
    force: bool = False,
) -> ConvertResult:
    """Convert the RadioML pickle to the ``iq.npy`` + ``labels.parquet`` cache (§6.1).

    ``src`` is a local path or an ``http(s)://`` URL, optionally a ``.tar.bz2``.
    Pass ``sha256`` to assert the pickle digest. Idempotent: skips work if the cache
    already exists and ``force`` is false.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iq_path = out_dir / _IQ_NAME
    labels_path = out_dir / _LABELS_NAME
    meta_path = out_dir / _META_NAME

    if not force and iq_path.exists() and labels_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"cache present in {out_dir} (sha256 {meta['sha256']}); use --force to rebuild")
        return ConvertResult(
            root=out_dir,
            sha256=meta["sha256"],
            n_examples=meta["n_examples"],
            modulations=tuple(meta["modulations"]),
            snrs=tuple(meta["snrs"]),
            cached=True,
        )

    pkl_path = _resolve_source(src, out_dir)
    digest = sha256_file(pkl_path)
    print(f"sha256  {digest}  {pkl_path.name}")
    if sha256 and digest.lower() != sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {sha256}, got {digest}")

    raw = _load_pickle(pkl_path)
    validate_raw(raw)
    iq, modulation, snr_db = _stack(raw)

    np.save(iq_path, iq)
    pd.DataFrame(
        {
            "index": np.arange(len(iq), dtype=np.int64),
            "modulation": pd.array(modulation, dtype="string"),
            "snr_db": snr_db.astype(np.int16),
        }
    ).to_parquet(labels_path, index=False)

    mods = tuple(sorted({str(m) for m in modulation}))
    snrs = tuple(int(s) for s in sorted({int(s) for s in snr_db}))
    meta = {
        "sha256": digest,
        "source": str(src),
        "n_examples": int(len(iq)),
        "example_shape": list(EXAMPLE_SHAPE),
        "modulations": list(mods),
        "snrs": list(snrs),
        "split_seed": SPLIT_SEED,
        "split_fractions": list(SPLIT_FRACTIONS),
        "sigscope_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(
        f"wrote {iq_path.name} {iq.shape} {iq.dtype}, "
        f"{labels_path.name} ({len(iq)} rows), {meta_path.name}"
    )
    return ConvertResult(
        root=out_dir,
        sha256=digest,
        n_examples=int(len(iq)),
        modulations=mods,
        snrs=snrs,
        cached=False,
    )


# ---------------------------------------------------------------------------
# Loading (touches only the cache -- never the pickle)


def load(root: str | Path = DEFAULT_ROOT) -> RadioMLDataset:
    """Load the converted cache. Run ``sigscope fetch-data`` first to create it."""
    root = Path(root)
    iq_path = root / _IQ_NAME
    labels_path = root / _LABELS_NAME
    if not iq_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"no RadioML cache in {root}. Create it with:\n"
            "  sigscope fetch-data --src <path-or-url to RML2016.10a_dict.pkl>"
        )
    iq = np.load(iq_path, mmap_mode="r")
    labels = pd.read_parquet(labels_path)
    return RadioMLDataset(
        iq=iq,
        modulation=labels["modulation"].to_numpy(dtype=object),
        snr_db=labels["snr_db"].to_numpy(dtype=np.int64),
        root=root,
    )
