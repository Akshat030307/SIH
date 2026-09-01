"""Self-contained ``report.html`` writer (CLAUDE.md §3, §7).

A single HTML file openable with no server and with the network off (§2, §9 E): the
spectrogram is an embedded base64 PNG with the detection boxes already drawn into the
pixels, and every style is inline. No CDN, no Plotly, no fonts fetched -- the interactive
Plotly dashboard is Phase 7 and a different surface.

Follows §7's design direction rather than a default dashboard look: dark ground (#0d1117)
because a spectrogram is a light-on-dark object, one accent (#4dd0c4) used only for
selection and nothing decorative, hairline #8b949e detection boxes, viridis for the
spectrogram, monospace and larger type for measured values because "Numbers are the
content of this product".

§7 asks for IBM Plex. Vendoring the woff2 files is Phase 7 work along with the dashboard;
until then this uses a system stack in the same spirit (humanist sans, monospaced figures)
so the report stays a single file with nothing to fetch.
"""

from __future__ import annotations

import base64
import html
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sigscope.dsp.spectrogram import Spectrogram
from sigscope.report.png import colormap, draw_rect, encode_png
from sigscope.types import Burst, Detection, Report

__all__ = ["write_html", "render_spectrogram_png", "format_hz"]

# §7 palette
BG = "#0d1117"
PANEL = "#161b22"
RULE = "#2d333b"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
ACCENT = "#4dd0c4"
WARN = "#d29922"

BOX_COLOUR = (139, 148, 158)  # #8b949e hairline, §7: boxes are hairline when idle
LABEL_COLOUR = (77, 208, 196)  # #4dd0c4

MAX_PNG_WIDTH = 1600
MAX_PNG_HEIGHT = 720


# --------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------


def format_hz(value: float | None, *, normalised: bool = False, digits: int = 3) -> str:
    """Human-readable frequency. Normalised captures are labelled, never faked as Hz."""
    if value is None:
        return "—"
    if normalised:
        # a normalised value is a small fraction; significant figures, not fixed decimals
        return f"{value:+.5g} × fs"
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.{digits}f} GHz"
    if magnitude >= 1e6:
        return f"{value / 1e6:.{digits}f} MHz"
    if magnitude >= 1e3:
        return f"{value / 1e3:.{digits}f} kHz"
    return f"{value:.1f} Hz"


def _format_seconds(value: float | None, *, normalised: bool = False) -> str:
    """Seconds, or a sample count when the capture has no known sample rate.

    With ``fs`` unknown the pipeline works at ``fs = 1.0``, so every "second" is really a
    sample index. §3 forbids printing a fabricated Hz value; printing a fabricated second
    is the same mistake with a different unit.
    """
    if value is None:
        return "—"
    if normalised:
        return f"{value:,.0f} samples"
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def _format_db(value: float | str | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return html.escape(value)
    return f"{value:.1f} dB"


def _format_rate(detection: Detection, *, normalised: bool = False) -> str:
    """Baud, or symbols per sample when the sample rate is unknown.

    "Bd" is symbols per *second*. With ``fs`` unknown there are no seconds to divide by,
    so quoting a baud figure would invent one.
    """
    estimate = detection.symbol_rate_hz
    if estimate is None or estimate.value is None:
        return "—"
    if normalised:
        return f"{estimate.value:.5g} × fs"
    return f"{estimate.value:,.0f} Bd"


def _esc(value: object) -> str:
    return html.escape(str(value))


# --------------------------------------------------------------------------------------
# spectrogram image
# --------------------------------------------------------------------------------------


def _max_pool(values: np.ndarray, factor_f: int, factor_t: int) -> np.ndarray:
    """Downsample by max over blocks.

    Max rather than mean on purpose: a narrow carrier one bin wide survives max pooling and
    disappears under averaging, and the whole point of the picture is to show the signals
    the detector found.
    """
    n_f, n_t = values.shape
    pad_f = (-n_f) % factor_f
    pad_t = (-n_t) % factor_t
    if pad_f or pad_t:
        values = np.pad(
            values, ((0, pad_f), (0, pad_t)), mode="constant", constant_values=values.min()
        )
    n_f, n_t = values.shape
    return values.reshape(n_f // factor_f, factor_f, n_t // factor_t, factor_t).max(axis=(1, 3))


def render_spectrogram_png(
    spec: Spectrogram,
    bursts: list[Burst],
    *,
    max_width: int = MAX_PNG_WIDTH,
    max_height: int = MAX_PNG_HEIGHT,
) -> tuple[bytes, int, int]:
    """Render the spectrogram with detection boxes drawn in, as PNG bytes.

    Time across, frequency up (§7), viridis, boxes as hairline rectangles. Renders the
    *same* spectrogram the detector used -- recomputing one with different parameters
    would put the boxes in visibly wrong places.
    """
    s_db = np.asarray(spec.S_db, dtype=np.float32)
    n_f, n_t = s_db.shape
    factor_f = max(1, math.ceil(n_f / max_height))
    factor_t = max(1, math.ceil(n_t / max_width))
    pooled = _max_pool(s_db, factor_f, factor_t)

    # frequency up: row 0 is the top of the image, so flip the frequency axis
    rgb = colormap(pooled)[::-1].copy()
    height, width = rgb.shape[:2]

    t0, t1 = float(spec.t[0]), float(spec.t[-1])
    f0, f1 = float(spec.f[0]), float(spec.f[-1])
    t_span = (t1 - t0) or 1.0
    f_span = (f1 - f0) or 1.0

    def to_col(t: float) -> int:
        return int(round((t - t0) / t_span * (width - 1)))

    def to_row(f: float) -> int:
        return int(round((1.0 - (f - f0) / f_span) * (height - 1)))

    for burst in bursts:
        x0, x1 = to_col(burst.t0), to_col(burst.t1)
        y0, y1 = to_row(burst.f_hi), to_row(burst.f_lo)
        # give a degenerate box a visible minimum size, or it renders as a dot
        if x1 - x0 < 2:
            x1 = x0 + 2
        if y1 - y0 < 2:
            y1 = y0 + 2
        draw_rect(rgb, x0, y0, x1, y1, BOX_COLOUR)

    return encode_png(rgb), width, height


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------

_CSS = f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px;
  background: {BG}; color: {TEXT};
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 14px; line-height: 1.5;
}}
h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 13px; font-weight: 600; margin: 0 0 12px;
     text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; }}
.wrap {{ max-width: 1600px; margin: 0 auto; }}
.sub {{ color: {MUTED}; font-size: 13px; margin-bottom: 24px; }}
.panel {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 6px;
         padding: 20px; margin-bottom: 20px; }}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, "SF Mono", Menlo,
        Consolas, "Liberation Mono", monospace; font-variant-numeric: tabular-nums; }}
.figure {{ margin: 0; }}
.figure img {{ width: 100%; height: auto; display: block; border: 1px solid {RULE};
              border-radius: 4px; image-rendering: auto; }}
.axes {{ display: flex; justify-content: space-between; color: {MUTED};
        font-size: 12px; margin-top: 8px; }}
.axis-y {{ display: flex; justify-content: space-between; color: {MUTED};
          font-size: 12px; margin-bottom: 6px; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ text-align: left; font-weight: 600; font-size: 11px; text-transform: uppercase;
     letter-spacing: 0.06em; color: {MUTED}; padding: 8px 12px 8px 0;
     border-bottom: 1px solid {RULE}; white-space: nowrap; }}
td {{ padding: 10px 12px 10px 0; border-bottom: 1px solid {RULE}; vertical-align: top; }}
tr:last-child td {{ border-bottom: none; }}
td.num {{ font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas,
         monospace; font-variant-numeric: tabular-nums; font-size: 15px;
         white-space: nowrap; }}
.id {{ color: {ACCENT}; font-weight: 600; }}
.label {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
         border: 1px solid {RULE}; color: {MUTED}; font-size: 12px; }}
.evidence {{ margin: 0; padding: 0 0 0 18px; color: {TEXT}; }}
.evidence li {{ margin: 4px 0; }}
.evidence-row td {{ padding-top: 0; color: {MUTED}; }}
.kv {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px 28px; }}
.kv div span {{ display: block; color: {MUTED}; font-size: 11px;
               text-transform: uppercase; letter-spacing: 0.06em; }}
.kv div strong {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
                 font-weight: 500; font-size: 15px; font-variant-numeric: tabular-nums; }}
.warn {{ color: {WARN}; }}
.warn ul, .notes ul {{ margin: 0; padding-left: 18px; }}
.warn li, .notes li {{ margin: 3px 0; }}
.notes {{ color: {MUTED}; }}
.empty {{ color: {MUTED}; padding: 8px 0; }}
footer {{ color: {MUTED}; font-size: 12px; margin-top: 28px;
         border-top: 1px solid {RULE}; padding-top: 16px; }}
"""


def _capture_panel(report: Report) -> str:
    capture = report.capture
    normalised = capture.frequencies_are_normalised
    rows = [
        ("File", _esc(report.file.name)),
        ("Size", f"{report.file.bytes:,} bytes"),
        ("Format", _esc(capture.source_format)),
        ("Sample rate", "unknown" if normalised else f"{capture.sample_rate:,.0f} Hz"),
        (
            "Centre frequency",
            "unknown" if capture.center_freq is None else format_hz(capture.center_freq),
        ),
        ("Centre from", _esc(capture.center_freq_source or "—")),
        ("Duration", _format_seconds(capture.duration_s, normalised=normalised)),
        ("Sample format", f"{_esc(capture.dtype_guessed)} ({capture.dtype_confidence:.0%})"),
        ("Noise floor", _format_db(report.noise_floor_dbfs) + "FS"),
        ("Detections", str(len(report.detections))),
        ("Metadata confidence", f"{capture.metadata_confidence:.0%}"),
        ("Analysis time", f"{report.runtime_s:.2f} s"),
    ]
    cells = "".join(
        f"<div><span>{_esc(name)}</span><strong>{value}</strong></div>" for name, value in rows
    )
    return f'<section class="panel"><h2>Capture</h2><div class="kv">{cells}</div></section>'


def _spectrogram_panel(report: Report, spec: Spectrogram | None, bursts: list[Burst]) -> str:
    if spec is None:
        return (
            '<section class="panel"><h2>Spectrogram</h2>'
            '<p class="empty">No spectrogram: the capture was too short to analyse.</p>'
            "</section>"
        )
    png, _, _ = render_spectrogram_png(spec, bursts)
    encoded = base64.b64encode(png).decode("ascii")
    normalised = report.capture.frequencies_are_normalised
    f_lo, f_hi = float(spec.f[0]), float(spec.f[-1])
    t_lo, t_hi = float(spec.t[0]), float(spec.t[-1])
    return f"""<section class="panel"><h2>Spectrogram</h2>
<figure class="figure">
  <div class="axis-y"><span>{format_hz(f_hi, normalised=normalised)}</span>
    <span>frequency (offset from capture centre)</span>
    <span>{format_hz(f_lo, normalised=normalised)}</span></div>
  <img alt="Spectrogram with {len(bursts)} detection box(es)"
       src="data:image/png;base64,{encoded}">
  <div class="axes"><span>{_format_seconds(t_lo)}</span>
    <span>time — {len(bursts)} detection(s) boxed</span>
    <span>{_format_seconds(t_hi)}</span></div>
</figure></section>"""


def _detections_panel(report: Report) -> str:
    normalised = report.capture.frequencies_are_normalised
    if not report.detections:
        return (
            '<section class="panel"><h2>Detections</h2>'
            '<p class="empty">No signals found above the detection threshold. '
            "Try lowering it with <code>--threshold-db</code>.</p></section>"
        )

    head = (
        "<tr><th>#</th><th>Start</th><th>Duration</th><th>Centre</th><th>Offset</th>"
        "<th>OBW99</th><th>SNR</th><th>Power</th><th>Symbol rate</th>"
        "<th>Modulation</th></tr>"
    )
    body: list[str] = []
    for d in report.detections:
        centre = (
            format_hz(d.center_freq_hz)
            if d.center_freq_hz is not None
            else '<span style="color:#8b949e">unknown</span>'
        )
        modulation = d.modulation.label if d.modulation else "—"
        body.append(
            "<tr>"
            f'<td class="num id">{d.id}</td>'
            f'<td class="num">{_format_seconds(d.time_start_s, normalised=normalised)}</td>'
            f'<td class="num">{_format_seconds(d.duration_s, normalised=normalised)}</td>'
            f'<td class="num">{centre}</td>'
            f'<td class="num">{format_hz(d.center_freq_offset_hz, normalised=normalised)}</td>'
            f'<td class="num">{format_hz(d.bandwidth_hz.occupied_99, normalised=normalised)}</td>'
            f'<td class="num">{_format_db(d.snr_db)}</td>'
            f'<td class="num">{_format_db(d.power_dbfs)}</td>'
            f'<td class="num">{_format_rate(d, normalised=normalised)}</td>'
            f'<td><span class="label">{_esc(modulation)}</span></td>'
            "</tr>"
        )
        if d.modulation and d.modulation.evidence:
            items = "".join(f"<li>{_esc(line)}</li>" for line in d.modulation.evidence)
            body.append(
                '<tr class="evidence-row"><td></td>'
                f'<td colspan="9"><ul class="evidence">{items}</ul></td></tr>'
            )
    return (
        '<section class="panel"><h2>Detections</h2>'
        f"<table>{head}{''.join(body)}</table></section>"
    )


def _list_panel(title: str, items: list[str], css_class: str) -> str:
    if not items:
        return ""
    entries = "".join(f"<li>{_esc(item)}</li>" for item in items)
    return (
        f'<section class="panel {css_class}"><h2>{_esc(title)}</h2>'
        f"<ul>{entries}</ul></section>"
    )


def build_html(report: Report, spec: Spectrogram | None, bursts: list[Burst]) -> str:
    """Assemble the full self-contained HTML document."""
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    unclassified = any(
        d.modulation and d.modulation.label == "unclassified" for d in report.detections
    )
    phase_note = (
        "<p>Modulation classification is not wired in yet (Phase 6), so every detection "
        "is labelled <code>unclassified</code>. Every measured parameter above is real.</p>"
        if unclassified
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIGSCOPE — {_esc(report.file.name)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>SIGSCOPE</h1>
<p class="sub">{_esc(report.file.name)} — {len(report.detections)} detection(s) —
 generated {generated}</p>
{_capture_panel(report)}
{_spectrogram_panel(report, spec, bursts)}
{_detections_panel(report)}
{_list_panel("Warnings", report.warnings, "warn")}
{_list_panel("How this file was read", report.capture.notes, "notes")}
<footer>Schema {_esc(report.schema_version)} —
 analysed in {report.runtime_s:.2f} s —
 SHA-256 <span class="mono">{_esc(report.file.sha256[:16] or "—")}</span>
{phase_note}</footer>
</div></body></html>
"""


def write_html(
    report: Report,
    path: str | Path,
    *,
    spectrogram: Spectrogram | None = None,
    bursts: list[Burst] | None = None,
) -> Path:
    """Write a self-contained ``report.html`` (CLAUDE.md §3 Stage 6)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(report, spectrogram, bursts or []), encoding="utf-8")
    return path
