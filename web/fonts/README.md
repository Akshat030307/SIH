# Fonts

CLAUDE.md §7 asks for **IBM Plex Sans** and **IBM Plex Mono**, vendored here as `woff2`,
with no font CDN. The woff2 files are **not in this checkout** — they are not present
anywhere on the build machine and §2 forbids fetching them at runtime, so they could not
be vendored offline.

`css/app.css` already declares them:

```
IBMPlexSans-Regular.woff2     (400)
IBMPlexSans-SemiBold.woff2    (600)
IBMPlexMono-Regular.woff2     (400)
IBMPlexMono-Medium.woff2      (500)
```

Drop those four files into this directory and they take effect on the next page load —
there is no build step and no code change. Until then the CSS falls through to a system
humanist sans and a monospace, both with `font-variant-numeric: tabular-nums`, so §7's
typographic intent (tabular figures, measured values set larger than their labels)
survives even though the exact typeface does not.

Get them from the IBM Plex release archive (OFL-1.1 licensed) on a machine with network
access, and copy them across. Subsetting to `latin` keeps each file near 30 kB.
