"""``python -m sigscope`` -- the CLI without installing anything.

§9 E wants a fresh clone to reach a working demo quickly on a clean machine, and §10 runs
that demo with the network off. ``pip install -e .`` is the normal route, but it still
needs a build backend and a writable environment. This module means a clone plus the
dependencies is enough::

    python -m sigscope analyse capture.iq --out report/
    python -m sigscope serve

which is the shortest path from "someone handed me a USB stick" to a report, and the
fallback if the install step is ever the thing that breaks on stage.
"""

from __future__ import annotations

import sys

from sigscope.cli import main

if __name__ == "__main__":
    sys.exit(main())
