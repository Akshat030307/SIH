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
| 0 | Skeleton: repo layout, frozen `types.py` schema, CLI surface | ✅ done |
| 1 | RadioML 2016.10a loader + `fetch-data`, `testsignals.py` | ✅ done |
| 2 | File I/O: wav (mono/stereo-IQ), headerless raw + dtype guesser, SigMF, filename parsers | ✅ done |
| 3 | Spectrogram (§4.1), noise floor (§4.2), detector (§4.3), scene composer (§6.2) | 🚧 in progress |
| 4–8 | Estimators, end-to-end, classifier, API/dashboard, hardening | ⬜ not started |

Phase 3 detector modules (`sigscope/dsp/`) are committed as work-in-progress and do
**not** yet pass the §9 C precision/recall targets; the scene composer and detection
evaluator are still being written.

`pytest` is green for everything through Phase 2 (types, CLI, RadioML loader,
`testsignals`, and the full §9 B file-I/O suite).

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
sigscope analyse capture.iq --fs 2e6 --fc 100e6 --out report/   # not implemented yet
sigscope batch ./captures/ --workers 8 --out results.csv        # not implemented yet
sigscope fetch-data --src <path|url>                            # ✅ Phase 1
sigscope make-scenes --n 300 --out data/scenes/                 # 🚧 Phase 3
sigscope train --model both                                     # not implemented yet
sigscope evaluate --out ACCURACY.md                             # not implemented yet
sigscope serve --port 8000                                      # not implemented yet
```

## Layout

```
sigscope/
  io/            file readers, format sniffing, metadata recovery      (Phase 2 ✅)
  dsp/           spectrogram, noise floor, detector, conditioner       (Phase 3 🚧)
  features/      cumulants, instantaneous statistics                   (Phase 6)
  models/        cnn.py, feature_clf.py, rules.py, ensemble.py         (Phase 6)
  data/          radioml.py loader (✅), scene.py composer (🚧)
  report/        json / sigmf / html writers                           (Phase 5)
  api/           fastapi app                                           (Phase 7)
  testsignals.py known-truth generators, used ONLY by tests           (Phase 1 ✅)
  types.py       frozen dataclasses matching the §3 JSON schema        (Phase 0 ✅)
  cli.py
data/
  radioml/       converted RadioML cache            (git-ignored, `sigscope fetch-data`)
  samples/       small real captures                (committed)
  scenes/        composed test scenes               (git-ignored, `sigscope make-scenes`)
scripts/         plot_classes.py, evaluation helpers
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
