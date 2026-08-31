"""Self-contained ``report.html`` writer (CLAUDE.md §3, §7).

A single HTML file openable with no server: spectrogram with detection boxes, a table of
detections, and the evidence panel per detection. Plotly is vendored locally — no CDN.

Not implemented yet (Phase 5, §8).
"""

from __future__ import annotations
