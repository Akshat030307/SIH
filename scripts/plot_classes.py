"""Phase 1 checkpoint (CLAUDE.md §8): eyeball the RadioML loader.

For each modulation, take N examples at a chosen SNR and save one PNG grid with the
constellation, the mean magnitude spectrum, and the instantaneous frequency. Also print
the class and SNR distribution of the whole loaded dataset.

The constellation panel shows the raw 128-sample scatter (faint) plus the symbol-centre
points after coarse carrier-offset removal (bold) -- RadioML bakes in a carrier offset
and pulse shaping, so the raw scatter is a smear and the corrected symbol-rate points are
what reveal the modulation order. QPSK at high SNR must show four clean lobes there; if it
does not, the loader is wrong -- stop and find it before building anything on top.

    python scripts/plot_classes.py [--snr 18] [--n 20] [--root data/radioml] [--out ...]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sigscope.data import radioml  # noqa: E402


def _print_distribution(ds: radioml.RadioMLDataset) -> None:
    print(f"loaded {len(ds)} examples from {ds.root}")
    print(f"  iq: shape {tuple(ds.iq.shape)} dtype {ds.iq.dtype}")

    mods = Counter(ds.modulation.tolist())
    print(f"\nclass distribution ({len(mods)} modulations):")
    for name in sorted(mods):
        print(f"  {name:<8} {mods[name]:>7}")

    snrs = Counter(int(s) for s in ds.snr_db)
    print(f"\nSNR distribution ({len(snrs)} levels, dB): "
          f"{min(snrs)}..{max(snrs)}")
    counts = {s for s in snrs.values()}
    print(f"  {sorted(snrs)}")
    print(f"  per-level count: {counts if len(counts) > 1 else counts.pop()}")


# samples per symbol for the digital modulations in RML2016.10a (GNU Radio
# transmitters.py), and the M-th power that removes carrier offset for each.
_SPS = 8
_CFO_ORDER = {"BPSK": 2, "QPSK": 4, "PAM4": 4, "QAM16": 4, "QAM64": 4, "8PSK": 8}


def _derotate(row: np.ndarray, order: int) -> np.ndarray:
    """Remove a constant carrier phase/offset via the M-th-power estimate (§4.9)."""
    return row * np.exp(-1j * np.angle(np.mean(row**order)) / order)


def _plot_class(name: str, x: np.ndarray, snr_db: int, out_dir: Path, sps: int) -> Path:
    """x: complex64 array, shape (n_examples, n_samples)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"{name}  @ {snr_db} dB SNR   ({x.shape[0]} examples)", fontsize=13)

    # constellation: raw samples (faint) + symbol-centre, CFO-corrected points (bold).
    # RadioML bakes in a carrier offset, so the raw 128-sample scatter is a smear;
    # the corrected symbol-rate points are what shows the modulation order.
    xn = x / (np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True)) + 1e-12)
    axes[0].scatter(xn.real.ravel(), xn.imag.ravel(), s=3, alpha=0.06,
                    color="0.6", edgecolors="none")
    order = _CFO_ORDER.get(name)
    if order is not None:
        pts = np.concatenate([_derotate(r, order)[::sps] for r in xn])
        axes[0].scatter(pts.real, pts.imag, s=6, alpha=0.35, color="#1f77b4", edgecolors="none")
        sub = "symbol-rate, CFO-corrected"
    else:
        sub = "raw samples (analogue / non-PSK)"
    lim = 1.05 * np.percentile(np.abs(xn), 99.5)
    axes[0].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="I", ylabel="Q",
                title=f"constellation\n{sub}")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.2)

    # mean magnitude spectrum (dB)
    spec = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1))
    spec_db = 20 * np.log10(spec.mean(axis=0) + 1e-12)
    freq = np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
    axes[1].plot(freq, spec_db - spec_db.max(), lw=1)
    axes[1].set(xlabel="normalised frequency", ylabel="dB (rel. peak)",
                title="mean spectrum", ylim=(-60, 3))
    axes[1].grid(alpha=0.2)

    # instantaneous frequency for the first few examples
    for row in x[: min(5, len(x))]:
        f_inst = np.diff(np.unwrap(np.angle(row))) / (2 * np.pi)
        axes[2].plot(f_inst, lw=0.8, alpha=0.8)
    axes[2].set(xlabel="sample", ylabel="cycles/sample", title="instantaneous frequency")
    axes[2].grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / f"{name.replace('/', '-')}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--snr", type=int, default=18, help="SNR level in dB to sample [18]")
    ap.add_argument("--n", type=int, default=20, help="examples per class [20]")
    ap.add_argument("--sps", type=int, default=_SPS, help=f"samples per symbol [{_SPS}]")
    ap.add_argument("--root", default=str(radioml.DEFAULT_ROOT), help="RadioML cache dir")
    ap.add_argument("--out", default=None, help="output dir [<root>/plots]")
    args = ap.parse_args()

    ds = radioml.load(args.root)
    _print_distribution(ds)

    out_dir = Path(args.out) if args.out else Path(args.root) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nwriting {out_dir}/  ({args.n} examples per class @ {args.snr} dB)")
    for name in sorted(set(ds.modulation.tolist())):
        idx = ds.where(modulation=name, snr_db=args.snr)[: args.n]
        if len(idx) == 0:
            print(f"  {name:<8} -- no examples at {args.snr} dB, skipped")
            continue
        path = _plot_class(name, ds.complex_iq(idx), args.snr, out_dir, args.sps)
        print(f"  {name:<8} -> {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
