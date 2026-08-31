"""``report.json`` writer (CLAUDE.md §3 "Frozen output schema").

Serialises a ``sigscope.types.Report`` via ``Report.to_dict``. Anything unmeasurable is
``null`` plus a ``warnings`` entry, never a placeholder number.

Not implemented yet (Phase 5, §8).
"""

from __future__ import annotations
