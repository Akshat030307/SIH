"""Stage 6 -- Report (CLAUDE.md §3 "Stage 6 -- Report").

One internal ``Report`` object (see ``sigscope.types``), three outputs:

- :func:`write_json`  -- ``report.json``, the frozen §3 schema, validated on the way out
- :func:`write_sigmf` -- ``capture.sigmf-meta`` annotations, checked to load in ``sigmf``
- :func:`write_html`  -- a self-contained ``report.html`` openable with no server

:func:`write_all` writes all three into one directory, which is what ``sigscope analyse``
does.
"""

from __future__ import annotations

from pathlib import Path

from sigscope.dsp.spectrogram import Spectrogram
from sigscope.report.html_report import build_html, format_hz, render_spectrogram_png, write_html
from sigscope.report.json_report import SchemaError, validate_report_dict, write_json
from sigscope.report.sigmf_report import build_sigmf_meta, write_sigmf
from sigscope.types import Burst, Report

__all__ = [
    "write_json",
    "validate_report_dict",
    "SchemaError",
    "write_sigmf",
    "build_sigmf_meta",
    "write_html",
    "build_html",
    "render_spectrogram_png",
    "format_hz",
    "write_all",
]


def write_all(
    report: Report,
    out_dir: str | Path,
    *,
    spectrogram: Spectrogram | None = None,
    bursts: list[Burst] | None = None,
    stem: str = "report",
) -> dict[str, Path]:
    """Write all three §3 outputs into ``out_dir`` and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": write_json(report, out_dir / f"{stem}.json"),
        "sigmf": write_sigmf(report, out_dir / f"{stem}.sigmf-meta"),
        "html": write_html(report, out_dir / f"{stem}.html",
                           spectrogram=spectrogram, bursts=bursts),
    }
