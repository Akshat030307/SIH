"""Stage 6 — Report (CLAUDE.md §3 "Stage 6 — Report").

One internal ``Report`` object (see ``sigscope.types``), three outputs:
- ``json_report`` — ``report.json``, the frozen §3 schema
- ``sigmf_report`` — ``capture.sigmf-meta`` SigMF annotations, readable by real tools
- ``html_report`` — a self-contained ``report.html`` openable with no server

Not implemented yet (Phase 5, §8).
"""

from __future__ import annotations
