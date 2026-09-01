# Vendored JavaScript

CLAUDE.md §7 asks for **Plotly vendored as a local file**. Plotly is **not in this
checkout**: it is not installed, no `plotly.min.js` exists on the build machine, and §2
forbids fetching it at runtime, so it could not be vendored offline.

The dashboard does not need it. Every chart is drawn on a plain 2-D canvas:

| §7 element | How it is drawn |
|---|---|
| Spectrogram | `js/viridis.js` maps the `/spectrogram.json` dB matrix to an `ImageData`, drawn to a canvas |
| Detection boxes | Absolutely-positioned DOM elements over the canvas — clickable, and styled with §7's exact hairline/accent colours |
| Constellation, spectrum, instantaneous frequency | `js/plots.js`, one canvas each |

For these four things that is not merely a substitute. Each plot is a single series with no
interaction beyond looking at it; canvas gives exact control over §7's palette and hairline
weights, adds nothing to load time, and — the part that matters for §9 E — cannot pull a
script or a font from anywhere. The interactive element that genuinely needed care, the
spectrogram with clickable boxes, is DOM and gets real hit-testing rather than a chart
library's approximation of it.

If Plotly is wanted later (zoom and pan across a long capture is the obvious reason), drop
`plotly.min.js` here and load it from `index.html`. Nothing in the current code depends on
its absence.
