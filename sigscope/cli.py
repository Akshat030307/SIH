"""SIGSCOPE command-line entry point — the CLI surface from CLAUDE.md §3.

Every subcommand is a stub that prints ``not implemented`` and exits non-zero until its
pipeline stage lands (phase plan in §8). ``sigscope --help`` and ``sigscope <cmd> --help``
work today so the interface can be built against.
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
    p.set_defaults(func=_not_implemented)

    # sigscope batch ./captures/ --workers 8 --out results.csv
    p = sub.add_parser("batch", help="analyse every capture in a folder")
    p.add_argument("input", help="folder of capture files")
    p.add_argument("--workers", type=int, default=8, help="parallel workers (default: 8)")
    p.add_argument("--out", default="results.csv", help="output CSV path (default: results.csv)")
    p.set_defaults(func=_not_implemented)

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
    p.set_defaults(func=_not_implemented)

    # sigscope evaluate --out ACCURACY.md
    p = sub.add_parser("evaluate", help="build the accuracy tables")
    p.add_argument("--out", default="ACCURACY.md", help="output markdown path [ACCURACY.md]")
    p.set_defaults(func=_not_implemented)

    # sigscope serve --port 8000
    p = sub.add_parser("serve", help="run the local API + dashboard")
    p.add_argument("--port", type=int, default=8000, help="listen port (default: 8000)")
    p.set_defaults(func=_not_implemented)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
