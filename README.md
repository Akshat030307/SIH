# SIGSCOPE

Automated analysis of `.iq` and `.wav` radio captures with signal-parameter extraction.

Smart India Hackathon 2026 problem **SIH26147**, sponsor **NTRO** (category: Software).
The complete build specification is in [`CLAUDE.md`](CLAUDE.md) — read it end to end
before touching code. `KICKOFF-PROMPTS.md` is just copy-paste prompts.

For **every separate signal inside a capture** the tool reports centre frequency,
bandwidth, on/off times, SNR, symbol rate, modulation (AM/FM/SSB/CW/FSK/PSK/QAM/OFDM/
chirp) with sub-parameters, and a confidence plus human-readable evidence — as a
spectrogram with boxes, a JSON report, and a SigMF annotation file. Runs fully offline,
CPU only.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton: repo layout, frozen `types.py` schema, CLI surface | done |
| 1 | RadioML 2016.10a loader + `fetch-data`, `testsignals.py` | done |
| 2 | File I/O: wav (mono/stereo-IQ), headerless raw + dtype guesser, SigMF, filename parsers | done |
| 3 | Spectrogram (§4.1), noise floor (§4.2), detector (§4.3) | done |
| 4 | Classical estimators (§4.4–§4.13) + `ACCURACY.md` estimator tables | done |
| 5 | End-to-end pipeline, JSON / SigMF / HTML reports, `analyse` + `batch` | done |
| 6 | Features, rules, ensemble, evidence, CNN, training script | code done, **not trained** |
| 7 | FastAPI app + static dashboard | done |
| 8 | Hardening: §9 E/F sweep, streaming for large captures, timing | done (demo not rehearsed) |
| — | Scene composer `make-scenes` (§6.2) | not started |

`pytest` is green and `ruff` is clean. `python scripts/hardening.py` runs the §9 E and
§9 F acceptance sweep (add `--with-2gb --with-clone` for the two slow cases);
`python scripts/evaluate.py --out ACCURACY.md` regenerates the accuracy and throughput
tables.

### Two things that are deliberately absent

**No classification accuracy numbers.** RadioML 2016.10a is a 225 MB download and §2
forbids fetching it at runtime, so no checkpoint has been trained. Both learned models
abstain with a reason, the §5.4 deterministic rules still fire (OFDM, chirp, CW, noise,
SSB), and the classification half of `ACCURACY.md` says exactly what is missing rather
than quoting a figure. To populate it:

```bash
sigscope fetch-data --src <path to RML2016.10a_dict.pkl>
sigscope train --model both
sigscope evaluate --out ACCURACY.md
```

**Plotly and IBM Plex are not vendored.** §7 asks for both as local files; neither is
present on the build machine and neither can be fetched offline. The dashboard draws
every chart on a plain 2-D canvas instead and falls back to a system font stack with
tabular figures. See [`web/vendor/README.md`](web/vendor/README.md) and
[`web/fonts/README.md`](web/fonts/README.md) — dropping the files in needs no code change.

## Setup

Offline, CPU only. Python is pinned to **3.11** (CLAUDE.md §2).

```bash
# 1. environment
conda create -n sigscope python=3.11 numpy scipy pandas pyarrow matplotlib pytest ruff soundfile
conda activate sigscope
pip install -e . --no-deps          # torch / sigmf / scikit-learn / fastapi added per-phase

# 2. sanity
sigscope --help
pytest

# 3. dataset (one-time, needs the RML2016.10a pickle obtained out-of-band)
sigscope fetch-data --src /path/to/RML2016.10a_dict.pkl      # or a .tar.bz2, or a URL
python scripts/plot_classes.py                                # eyeball the loader
```

`fetch-data` verifies the pickle structure (220 keys × `(1000, 2, 128)`), prints a
SHA-256, and writes a memory-mapped `data/radioml/iq.npy` + `labels.parquet` cache that
training never re-parses. The raw pickle and the cache are git-ignored.

## CLI

```
sigscope analyse capture.iq --fs 2e6 --fc 100e6 --out report/   # json + sigmf + html
sigscope batch ./captures/ --workers 8 --out results.csv        # one CSV row per file
sigscope serve --port 8000                                      # API + dashboard
sigscope fetch-data --src <path|url>                            # RadioML cache
sigscope train --model both                                     # needs the cache
sigscope evaluate --out ACCURACY.md                             # accuracy tables
sigscope make-scenes --n 300 --out data/scenes/                 # not implemented yet
```

`analyse` and `batch` also take `--dtype` and `--threshold-db`; `batch` takes `--pattern`
and `--reports <dir>` to write a full report set per file.

## Dashboard

```bash
sigscope serve --port 8000       # then open http://127.0.0.1:8000/
```

Or without installing anything, from a clone:

```bash
python -m sigscope serve --port 8000
```

Three screens: **drop** (drag a capture in; optional sample rate / centre frequency /
threshold), **analysis** (spectrogram with clickable detection boxes, detection list,
parameter panel, evidence panel, and constellation / spectrum / instantaneous-frequency
plots), and **batch** (sortable, filterable table with CSV export).

Everything is served from `web/` by the same process that answers the API — no npm, no
build step, no CDN, no font service. `tests/test_api.py` asserts that no asset references
an external origin, so §9 E's "works identically with wifi disabled" is a property of the
checkout rather than of a setting.

The API is **local-only and unauthenticated** by design (§7). `POST /api/batch` reads a
caller-supplied directory path, so keep it bound to `127.0.0.1`; `create_app` carries the
note marking where auth would go.

## Layout

```
sigscope/
  io/            file readers, format sniffing, metadata recovery      (Phase 2 ✅)
  dsp/           spectrogram, noise floor, detector, estimators        (Phases 3-4)
  features/      cumulants, instantaneous + spectral statistics        (Phase 6)
  models/        feature_clf, cnn, rules, ensemble, evidence           (Phase 6)
  data/          radioml.py loader, scene.py composer (not started)
  report/        json / sigmf / html / png writers                     (Phase 5)
  api/           fastapi app, job table, demodulation                  (Phase 7)
  pipeline.py    ingest -> condition -> detect -> measure -> classify  (Phase 5)
  batch.py       folder -> one CSV row per file                        (Phase 5)
  evaluation.py  §9 D classification scoring                           (Phase 6)
  testsignals.py known-truth generators, used ONLY by tests           (Phase 1 ✅)
  types.py       frozen dataclasses matching the §3 JSON schema        (Phase 0 ✅)
  cli.py
data/
  radioml/       converted RadioML cache            (git-ignored, `sigscope fetch-data`)
  samples/       small real captures                (committed)
  scenes/        composed test scenes               (git-ignored, `sigscope make-scenes`)
web/             static dashboard (no build step)
scripts/         plot_classes.py, train.py, evaluate.py
tests/           pytest suite
archive/         raw dataset drop                   (git-ignored, keep local)
```

## Conventions (CLAUDE.md §2)

- Every estimator is a pure function: numpy array in, dataclass out — value **and**
  confidence (0..1) **and** method name.
- Complex baseband is `np.complex64`, converted once at the `sigscope/io/` edge.
- Sample rate always passed explicitly as `fs: float`. Hz / seconds / dB at every boundary.
- Anything unmeasurable is `None` plus a `warnings` entry — never a fabricated number.
- `ruff` clean, type hints everywhere, tests before estimators where the answer is closed-form.
