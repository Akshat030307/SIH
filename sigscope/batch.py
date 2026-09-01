"""Batch mode: a folder of captures in, one CSV row per file out (CLAUDE.md §3, §9 E).

§9 E is specific: "A batch of 100 files completes with one CSV row each and no crash."
Both halves matter. One row per file means the CSV stays a file-level index an analyst can
sort and filter (§7's batch view), with the per-detection detail living in the JSON reports;
and *no crash* means a single unreadable file must never take the run down, so every file
is wrapped and a failure becomes a row with a ``status`` of ``error`` and the reason.

Workers use processes because the work is CPU-bound DSP. If a process pool cannot start --
which happens in some frozen and embedded launchers on Windows -- the run falls back to
sequential rather than failing, because §10's demo runs on one laptop with the network off
and a batch that refuses to start is worse than a slow one.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sigscope.io import CaptureError
from sigscope.pipeline import AnalysisConfig, analyse_file

__all__ = ["CSV_COLUMNS", "BatchRow", "find_captures", "analyse_one", "run_batch", "write_csv"]

# files that are never a capture, so they are not offered to the reader at all
_SKIP_SUFFIXES = {
    ".json", ".md", ".txt", ".csv", ".html", ".png", ".jpg", ".jpeg", ".pdf",
    ".py", ".yml", ".yaml", ".toml", ".gitkeep", ".parquet", ".npy", ".pt", ".pkl",
}

CSV_COLUMNS: tuple[str, ...] = (
    "file",
    "status",
    "error",
    "source_format",
    "sample_rate_hz",
    "center_freq_hz",
    "center_freq_source",
    "frequencies_are_normalised",
    "duration_s",
    "dtype",
    "dtype_confidence",
    "noise_floor_dbfs",
    "n_detections",
    "modulations",
    "strongest_snr_db",
    "strongest_center_freq_hz",
    "strongest_bandwidth_hz",
    "strongest_symbol_rate_hz",
    "strongest_symbol_rate_confidence",
    "n_warnings",
    "runtime_s",
    "report_json",
)


@dataclass
class BatchRow:
    """One CSV row. ``status`` is ``ok`` or ``error``; an error row still carries a name."""

    data: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.data.get("status") == "ok"


def find_captures(folder: Path, pattern: str = "*") -> list[Path]:
    """Every plausible capture file in ``folder``, sorted.

    Skips obvious non-captures and, for a SigMF pair, keeps only the ``.sigmf-data`` so the
    recording is not analysed twice under two names.
    """
    files = [p for p in sorted(folder.glob(pattern)) if p.is_file()]
    out: list[Path] = []
    for path in files:
        if path.suffix.lower() in _SKIP_SUFFIXES or path.name.startswith("."):
            continue
        if path.suffix == ".sigmf-meta" and path.with_suffix(".sigmf-data").is_file():
            continue  # the .sigmf-data half carries the same pair
        out.append(path)
    return out


def analyse_one(
    path: str | Path,
    *,
    fs: float | None = None,
    fc: float | None = None,
    dtype: str | None = None,
    reports_dir: str | Path | None = None,
    cfg: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Analyse one file and flatten it to a CSV row. Never raises.

    Module-level and picklable-by-name so it can be the process-pool worker.
    """
    path = Path(path)
    row: dict[str, Any] = dict.fromkeys(CSV_COLUMNS, "")
    row["file"] = path.name

    try:
        analysis = analyse_file(path, fs=fs, fc=fc, dtype=dtype, cfg=cfg)
    except CaptureError as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        return row
    except Exception as exc:  # noqa: BLE001 -- §9 E: one bad file must not stop the batch
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    report = analysis.report
    capture = report.capture
    row.update(
        status="ok",
        source_format=capture.source_format,
        sample_rate_hz=f"{capture.sample_rate:.6g}",
        center_freq_hz="" if capture.center_freq is None else f"{capture.center_freq:.6g}",
        center_freq_source=capture.center_freq_source or "",
        frequencies_are_normalised=int(capture.frequencies_are_normalised),
        duration_s=f"{capture.duration_s:.6g}",
        dtype=capture.dtype_guessed,
        dtype_confidence=f"{capture.dtype_confidence:.3f}",
        noise_floor_dbfs=(
            "" if report.noise_floor_dbfs is None else f"{report.noise_floor_dbfs:.2f}"
        ),
        n_detections=len(report.detections),
        n_warnings=len(report.warnings),
        runtime_s=f"{report.runtime_s:.3f}",
    )

    labels = sorted(
        {d.modulation.label for d in report.detections if d.modulation is not None}
    )
    row["modulations"] = "|".join(labels)

    numeric_snr = [
        d for d in report.detections if isinstance(d.snr_db, (int, float))
    ]
    if numeric_snr:
        best = max(numeric_snr, key=lambda d: d.snr_db)
        row["strongest_snr_db"] = f"{best.snr_db:.2f}"
        if best.center_freq_hz is not None:
            row["strongest_center_freq_hz"] = f"{best.center_freq_hz:.6g}"
        elif best.center_freq_offset_hz is not None:
            row["strongest_center_freq_hz"] = f"{best.center_freq_offset_hz:+.6g} (offset)"
        if best.bandwidth_hz.occupied_99 is not None:
            row["strongest_bandwidth_hz"] = f"{best.bandwidth_hz.occupied_99:.6g}"
        if best.symbol_rate_hz is not None and best.symbol_rate_hz.value is not None:
            row["strongest_symbol_rate_hz"] = f"{best.symbol_rate_hz.value:.6g}"
            row["strongest_symbol_rate_confidence"] = f"{best.symbol_rate_hz.confidence:.2f}"

    if reports_dir is not None:
        from sigscope.report import write_all

        target = Path(reports_dir) / path.stem
        try:
            written = write_all(
                report, target, spectrogram=analysis.spectrogram, bursts=analysis.bursts
            )
            row["report_json"] = str(written["json"])
        except Exception as exc:  # noqa: BLE001 -- a write failure is a row note, not a crash
            row["error"] = f"report write failed: {type(exc).__name__}: {exc}"

    return row


def run_batch(
    paths: Sequence[Path],
    *,
    workers: int = 8,
    fs: float | None = None,
    fc: float | None = None,
    dtype: str | None = None,
    reports_dir: str | Path | None = None,
    cfg: AnalysisConfig | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Analyse every path, returning one row per file in the original order."""
    if not paths:
        return []

    kwargs = {"fs": fs, "fc": fc, "dtype": dtype, "reports_dir": reports_dir, "cfg": cfg}
    workers = max(1, min(int(workers), len(paths), (os.cpu_count() or 1) * 2))

    if workers == 1:
        return [_with_progress(analyse_one(p, **kwargs), p, i, len(paths), progress)
                for i, p in enumerate(paths, start=1)]

    try:
        results: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(analyse_one, str(p), **kwargs): i for i, p in enumerate(paths)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                results[index] = future.result()
                _with_progress(None, paths[index], done, len(paths), progress)
        return [results[i] for i in range(len(paths))]
    except (OSError, RuntimeError, ImportError) as exc:
        if progress:
            print(f"  process pool unavailable ({exc}); falling back to sequential")
        return [_with_progress(analyse_one(p, **kwargs), p, i, len(paths), progress)
                for i, p in enumerate(paths, start=1)]


def _with_progress(
    row: dict[str, Any] | None, path: Path, index: int, total: int, progress: bool
) -> Any:
    if progress:
        print(f"  [{index}/{total}] {path.name}")
    return row


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write the batch CSV, one row per file, columns fixed by :data:`CSV_COLUMNS`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
