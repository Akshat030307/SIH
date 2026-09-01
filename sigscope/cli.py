"""SIGSCOPE command-line entry point — the CLI surface from CLAUDE.md §3.

Live: ``analyse`` and ``batch`` (Phase 5), ``fetch-data`` (Phase 1), ``evaluate`` (Phase 4).
Still stubbed: ``make-scenes``. Every subcommand keeps its §3
signature so the rest of the project can be built against the interface.

Handlers import their dependencies lazily. ``sigscope --help`` should be instant, and it
is not instant if importing the CLI drags in torch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _not_implemented(args: argparse.Namespace) -> int:
    """Placeholder handler for every subcommand during Phase 0 (§8)."""
    print(f"sigscope {args.command}: not implemented")
    return 1


def _cmd_fetch_data(args: argparse.Namespace) -> int:
    """Convert RadioML 2016.10a to the memmap + parquet cache (CLAUDE.md §6.1)."""
    from pathlib import Path

    from sigscope.data import radioml

    if not args.src:
        print(
            "fetch-data: pass --src with a local path or URL to RML2016.10a_dict.pkl\n"
            "  sigscope fetch-data --src ~/Downloads/RML2016.10a_dict.pkl\n"
            "  sigscope fetch-data --src https://.../RML2016.10a.tar.bz2 --sha256 <hex>"
        )
        return 2
    try:
        result = radioml.convert(
            args.src, Path(args.out), sha256=args.sha256, force=args.force
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"fetch-data: {exc}")
        return 1

    tag = "cache reused" if result.cached else "converted"
    print(
        f"{tag}: {result.n_examples} examples, "
        f"{len(result.modulations)} modulations, {len(result.snrs)} SNR levels "
        f"-> {result.root}"
    )
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    """Analyse one capture and write all three §3 outputs (CLAUDE.md §3 Stage 6)."""
    from dataclasses import replace
    from pathlib import Path

    from sigscope.io import CaptureError
    from sigscope.pipeline import AnalysisConfig, analyse_file
    from sigscope.report import write_all

    cfg = AnalysisConfig()
    if args.threshold_db is not None:
        cfg = replace(cfg, detector=replace(cfg.detector, threshold_db=args.threshold_db))

    try:
        analysis = analyse_file(
            args.input, fs=args.fs, fc=args.fc, dtype=args.dtype, cfg=cfg
        )
    except CaptureError as exc:
        print(f"analyse: {exc}")
        return 1

    report = analysis.report
    out_dir = Path(args.out)
    try:
        written = write_all(
            report, out_dir, spectrogram=analysis.spectrogram, bursts=analysis.bursts
        )
    except OSError as exc:
        print(f"analyse: could not write reports to {out_dir}: {exc}")
        return 1

    _print_summary(report)
    print()
    for kind in ("json", "sigmf", "html"):
        print(f"  {kind:<6} {written[kind]}")
    return 0


def _print_summary(report) -> None:
    """Human-readable summary on stdout -- what an analyst reads before opening the HTML."""
    capture = report.capture
    normalised = capture.frequencies_are_normalised
    fs_text = "unknown" if normalised else f"{capture.sample_rate:,.0f} Hz"
    fc_text = "unknown" if capture.center_freq is None else f"{capture.center_freq:,.0f} Hz"
    time_unit = "samples" if normalised else "s"

    def freq(value: float | None) -> str:
        """Hz, or a fraction of the sample rate. A normalised value is a small fraction,
        so it needs significant figures rather than a thousands-separated integer -- an
        occupied bandwidth of 0.0166 x fs printed as "0 x fs" is worse than useless."""
        if value is None:
            return "-"
        return f"{value:+.5g} x fs" if normalised else f"{value:,.0f} Hz"

    print(f"{report.file.name}  ({report.file.bytes:,} bytes, {capture.source_format})")
    duration = (
        f"{capture.duration_s:,.0f} samples" if normalised else f"{capture.duration_s:.3f} s"
    )
    print(
        f"  sample rate {fs_text}   centre {fc_text}"
        f"   duration {duration}   dtype {capture.dtype_guessed}"
        f" ({capture.dtype_confidence:.0%})"
    )
    floor = "unknown" if report.noise_floor_dbfs is None else f"{report.noise_floor_dbfs:.1f} dBFS"
    print(f"  noise floor {floor}   {len(report.detections)} detection(s)"
          f"   analysed in {report.runtime_s:.2f} s")

    if report.detections:
        print()
        print(
            f"  {'#':>3}  {'start (' + time_unit + ')':>9}  "
            f"{'duration':>9}  {'centre':>16}  "
            f"{'OBW99':>12}  {'SNR':>8}  {'symbol rate':>20}  modulation"
        )
        for d in report.detections:
            if d.center_freq_hz is not None:
                centre = f"{d.center_freq_hz:,.0f} Hz"
            elif d.center_freq_offset_hz is not None:
                centre = freq(d.center_freq_offset_hz)
            else:
                centre = "unknown"
            obw = freq(d.bandwidth_hz.occupied_99).lstrip("+")
            if isinstance(d.snr_db, (int, float)):
                snr = f"{d.snr_db:.1f} dB"
            elif isinstance(d.snr_db, str):
                snr = d.snr_db
            else:
                snr = "-"
            if d.symbol_rate_hz is not None and d.symbol_rate_hz.value is not None:
                # "Bd" is symbols per *second*; with the sample rate unknown the only
                # honest unit is symbols per sample, i.e. a fraction of fs
                value = (
                    f"{d.symbol_rate_hz.value:.5g} x fs"
                    if normalised
                    else f"{d.symbol_rate_hz.value:,.0f} Bd"
                )
                rate = f"{value} ({d.symbol_rate_hz.confidence:.2f})"
            else:
                rate = "unknown"
            label = d.modulation.label if d.modulation else "-"
            if normalised:
                start = f"{d.time_start_s:>9,.0f}"
                length = f"{d.duration_s:>9,.0f}"
            else:
                start = f"{d.time_start_s:>8.3f}s"
                length = f"{d.duration_s:>8.3f}s"
            print(
                f"  {d.id:>3}  {start}  {length}  "
                f"{centre:>16}  {obw:>12}  {snr:>8}  {rate:>20}  {label}"
            )

    if report.warnings:
        print()
        print(f"  warnings ({len(report.warnings)}):")
        for warning in report.warnings[:8]:
            print(f"    - {warning}")
        if len(report.warnings) > 8:
            print(f"    ... and {len(report.warnings) - 8} more (see report.json)")


def _cmd_batch(args: argparse.Namespace) -> int:
    """Analyse a folder of captures into one CSV row per file (CLAUDE.md §3, §9 E)."""
    import time
    from dataclasses import replace
    from pathlib import Path

    from sigscope.batch import find_captures, run_batch, write_csv
    from sigscope.pipeline import AnalysisConfig

    folder = Path(args.input)
    if not folder.is_dir():
        print(f"batch: {folder} is not a directory")
        return 1

    paths = find_captures(folder, args.pattern)
    if not paths:
        print(f"batch: no capture files found in {folder} matching {args.pattern!r}")
        return 1

    cfg = AnalysisConfig()
    if args.threshold_db is not None:
        cfg = replace(cfg, detector=replace(cfg.detector, threshold_db=args.threshold_db))

    print(f"batch: {len(paths)} file(s) from {folder}, {args.workers} worker(s)")
    started = time.perf_counter()
    rows = run_batch(
        paths,
        workers=args.workers,
        fs=args.fs,
        fc=args.fc,
        dtype=args.dtype,
        reports_dir=args.reports,
        cfg=cfg,
    )
    elapsed = time.perf_counter() - started

    out = write_csv(rows, args.out)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    failed = len(rows) - ok
    detections = sum(int(r.get("n_detections") or 0) for r in rows)
    total_bytes = sum(p.stat().st_size for p in paths)
    rate = (total_bytes / 1e6) / elapsed if elapsed > 0 else 0.0

    print()
    print(f"  {ok} ok, {failed} failed, {detections} detection(s) total")
    print(f"  {elapsed:.1f} s for {total_bytes / 1e6:.1f} MB ({rate:.2f} MB/s)")
    print(f"  csv    {out}")
    if args.reports:
        print(f"  reports {args.reports}")
    if failed:
        print()
        for row in rows:
            if row.get("status") != "ok":
                print(f"    ! {row['file']}: {row['error']}")
    return 0


def _run_script(name: str, argv: list[str]) -> int:
    """Run one of the ``scripts/`` harnesses in-process.

    They live in ``scripts/`` so they can also be run directly without installing the
    package; these subcommands are the §3 CLI surface over them.
    """
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    if not script.exists():
        print(f"{name}: harness not found at {script}")
        return 1

    spec = importlib.util.spec_from_file_location(f"sigscope_{name}", script)
    if spec is None or spec.loader is None:
        print(f"{name}: could not load {script}")
        return 1
    module = importlib.util.module_from_spec(spec)
    # register before exec: @dataclass resolves its annotations through sys.modules, and
    # the harnesses use `from __future__ import annotations`, so an unregistered module
    # makes every dataclass in it fail to build
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return int(module.main(argv))


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the local API + dashboard (CLAUDE.md §7)."""
    from pathlib import Path

    from sigscope.api.app import WEB_ROOT, serve

    if not WEB_ROOT.is_dir() or not (WEB_ROOT / "index.html").is_file():
        print(f"serve: the dashboard is missing from {WEB_ROOT}")
        return 1
    print(f"sigscope {args.port}: dashboard on http://{args.host}:{args.port}/")
    print("  everything is served locally; no network access is required or used")
    try:
        serve(host=args.host, port=args.port)
    except OSError as exc:
        print(f"serve: could not bind {args.host}:{args.port}: {exc}")
        return 1
    _ = Path
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train the §5.2 classifier and/or the §5.3 CNN on the RadioML train split."""
    argv = ["--model", args.model, "--epochs", str(args.epochs), "--seed", str(args.seed)]
    if args.data:
        argv += ["--data", args.data]
    if args.summary:
        argv += ["--summary", args.summary]
    return _run_script("train", argv)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Build the §9 A estimator accuracy tables (CLAUDE.md §8 Phase 4).

    The harness lives in ``scripts/evaluate.py`` so it can also be run directly without
    installing the package; this subcommand is the §3 CLI surface over it.
    """
    return _run_script(
        "evaluate",
        ["--out", args.out, "--trials", str(args.trials), "--seed", str(args.seed)],
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser for all §3 subcommands."""
    parser = argparse.ArgumentParser(
        prog="sigscope",
        description="Automated analysis of .iq and .wav radio captures (SIH26147 / NTRO).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    # sigscope analyse capture.iq --fs 2e6 --fc 100e6 --out report/
    p = sub.add_parser("analyse", help="analyse a single capture file")
    p.add_argument("input", help="path to the .iq / .wav / raw capture")
    p.add_argument("--fs", type=float, default=None, help="sample rate in Hz (else guessed)")
    p.add_argument("--fc", type=float, default=None, help="centre frequency in Hz (else guessed)")
    p.add_argument("--out", default="report/", help="output directory (default: report/)")
    p.add_argument("--dtype", default=None,
                   help="force the raw sample format (int8/int16/float32) instead of guessing")
    p.add_argument("--threshold-db", type=float, default=None,
                   help="detection threshold over the noise floor in dB (default: 8)")
    p.set_defaults(func=_cmd_analyse)

    # sigscope batch ./captures/ --workers 8 --out results.csv
    p = sub.add_parser("batch", help="analyse every capture in a folder")
    p.add_argument("input", help="folder of capture files")
    p.add_argument("--workers", type=int, default=8, help="parallel workers (default: 8)")
    p.add_argument("--out", default="results.csv", help="output CSV path (default: results.csv)")
    p.add_argument("--pattern", default="*", help="glob to select files (default: *)")
    p.add_argument("--reports", default=None,
                   help="also write per-file json/sigmf/html reports into this directory")
    p.add_argument("--fs", type=float, default=None, help="sample rate in Hz for every file")
    p.add_argument("--fc", type=float, default=None, help="centre frequency in Hz for every file")
    p.add_argument("--dtype", default=None, help="force the raw sample format for every file")
    p.add_argument("--threshold-db", type=float, default=None,
                   help="detection threshold over the noise floor in dB (default: 8)")
    p.set_defaults(func=_cmd_batch)

    # sigscope fetch-data --src <path-or-url> [--sha256 <hex>] [--force]
    p = sub.add_parser("fetch-data", help="download + convert RadioML 2016.10a")
    p.add_argument("--src", default=None, help="local path or URL to the RML2016.10a pickle")
    p.add_argument("--out", default="data/radioml/", help="output directory [data/radioml/]")
    p.add_argument("--sha256", default=None, help="expected SHA-256 of the pickle (asserted)")
    p.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    p.set_defaults(func=_cmd_fetch_data)

    # sigscope make-scenes --n 300 --out data/scenes/
    p = sub.add_parser("make-scenes", help="compose wideband test scenes with ground truth")
    p.add_argument("--n", type=int, default=300, help="number of scenes (default: 300)")
    p.add_argument("--out", default="data/scenes/", help="output directory [data/scenes/]")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.set_defaults(func=_not_implemented)

    # sigscope train --model both
    p = sub.add_parser("train", help="train the feature classifier and/or the CNN")
    p.add_argument(
        "--model",
        choices=["feature", "cnn", "both"],
        default="both",
        help="which model(s) to train (default: both)",
    )
    p.add_argument("--data", default=None, help="RadioML cache directory [data/radioml/]")
    p.add_argument("--epochs", type=int, default=40, help="CNN epochs (§5.3 default: 40)")
    p.add_argument("--seed", type=int, default=0, help="training seed (default: 0)")
    p.add_argument("--summary", default=None, help="write training metadata to this JSON")
    p.set_defaults(func=_cmd_train)

    # sigscope evaluate --out ACCURACY.md
    p = sub.add_parser("evaluate", help="build the accuracy tables")
    p.add_argument("--out", default="ACCURACY.md", help="output markdown path [ACCURACY.md]")
    p.add_argument("--trials", type=int, default=8, help="noise seeds per SNR point [8]")
    p.add_argument("--seed", type=int, default=1000, help="base RNG seed [1000]")
    p.set_defaults(func=_cmd_evaluate)

    # sigscope serve --port 8000
    p = sub.add_parser("serve", help="run the local API + dashboard")
    p.add_argument("--port", type=int, default=8000, help="listen port (default: 8000)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default: 127.0.0.1; there is no auth, see §7)")
    p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
