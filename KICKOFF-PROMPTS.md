# Kickoff prompts

Put both files in an empty git repo, open Claude Code there, and paste these one at a time. Wait for
each checkpoint to pass before moving on. Every prompt assumes `CLAUDE.md` is in the repo root.

---

## Phase 0 — skeleton

> Read `CLAUDE.md` end to end before writing anything.
>
> Set up the project skeleton: `pyproject.toml` with the exact dependencies from §2, the repo layout
> from §2, ruff and pytest config, and a `.gitignore` excluding `data/radioml/`, `data/scenes/`, and
> `*.pt`.
>
> Create `sigscope/types.py` with every dataclass the pipeline passes around — `CaptureMeta`, `Burst`,
> `Estimate` (value, confidence, method, notes), `Detection`, `Report`. Match the frozen JSON schema in
> §3 exactly. That schema does not change again, and everything else is built against it.
>
> Create `sigscope/cli.py` with every subcommand listed in §3, each printing "not implemented".
>
> Stop there. No DSP yet.

---

## Phase 1 — data in

> Read `CLAUDE.md` §6.
>
> Build `sigscope/data/radioml.py` and the `sigscope fetch-data` command. It takes a local path or a
> URL to RadioML 2016.10a, verifies the pickle structure (220 keys, each array `(1000, 2, 128)`),
> converts once to a memory-mapped `.npy` plus a `labels.parquet` with columns `index`, `modulation`,
> `snr_db`, and caches that. Never re-parse the pickle at training time. Verify a SHA-256 and print it.
>
> Then build `sigscope/testsignals.py` exactly as scoped in §6.3 — six functions, about 80 lines, for
> tests only. Do not expand it into a data generator.
>
> Finally write `scripts/plot_classes.py`: 20 examples per class at 18 dB SNR, saving a grid per class
> with constellation, spectrum, and instantaneous frequency. Run it and show me the images, plus the
> class and SNR distribution of the loaded dataset.
>
> If QPSK does not show four clean dots, something is wrong with the loader — stop and find it.

---

## Phase 2 — file I/O

> Read `CLAUDE.md` §3, Stage 1.
>
> Build `sigscope/io/`: wav reading (mono real, and stereo treated as IQ), headerless raw IQ with the
> dtype-guessing histogram heuristic, SigMF, and the filename metadata parsers for the SDR#, HDSDR and
> SDRuno patterns.
>
> Populate `CaptureMeta` with honest confidences and a `notes` list explaining every guess made. When
> the sample rate is unknown, set it to 1.0, set `frequencies_are_normalised` true, and make sure that
> flag propagates into the report.
>
> Write the tests from §9 section B, including the headerless round-trip and both of the malformed-file
> cases.

---

## Phase 3 — detection

> Read `CLAUDE.md` §4.1, §4.2, §4.3 and §6.2.
>
> Build the spectrogram, the robust per-bin noise floor, and the detector. Follow the algorithm step by
> step — percentile noise estimate, threshold, closing then opening, connected components, size
> rejection, bounding boxes, then the time-gap merge. Put every constant in a `DetectorConfig`
> dataclass with the spec defaults.
>
> Then build `sigscope/data/scene.py` and `sigscope make-scenes`, composing wideband scenes out of
> RadioML examples exactly as §6.2 describes, including all the awkward cases listed there. Each scene
> writes raw int16 IQ plus a ground-truth SigMF annotation file. Generate 300.
>
> Report detection precision and recall split by SNR band against the targets in §9 section C.

---

## Phase 4 — classical estimators

> Read `CLAUDE.md` §4.4 through §4.13 in full.
>
> Build `sigscope/dsp/estimators.py`. One pure function per measurement, each returning an `Estimate`
> with value, confidence and method name. Implement burst isolation (§4.4) first — everything else runs
> on the isolated burst.
>
> For symbol rate, implement all three methods and the reconciliation logic. Do not skip the harmonic
> check; locking onto 2× or 0.5× the true rate is the most common failure in this whole project.
>
> Write the tests from §9 section A as you go, using `testsignals.py`, plus the SNR sweep for each.
> Produce the first version of `ACCURACY.md` with the estimator error tables.

---

## Phase 5 — end to end

> Wire the full pipeline: ingest → condition → detect → isolate → measure → report. Modulation stays
> `unclassified` for now.
>
> Build `sigscope/report/`: the JSON writer matching the frozen schema, the SigMF annotation writer,
> and a self-contained HTML report with an embedded spectrogram PNG and a detections table.
>
> Make `sigscope analyse` and `sigscope batch` work for real. Run them on every file in
> `data/samples/` and show me the output.
>
> Then commit and tag. This is a submittable project on its own.

---

## Phase 6 — classifier

> Read `CLAUDE.md` §5 in full.
>
> Build in this order: (1) `sigscope/features/` with the cumulants and instantaneous statistics,
> (2) the gradient boosting classifier plus its training script, (3) the deterministic rules,
> (4) the CNN, (5) the ensemble referee, (6) the evidence sentence generator.
>
> Train on the RadioML train split only. Implement the sliding-window inference described in §5.3 so
> bursts longer than 128 samples work.
>
> Then write `scripts/evaluate.py` producing the full `ACCURACY.md`: confusion matrices at three SNR
> bands, accuracy-vs-SNR curves for each model and the ensemble, per-class precision and recall, and
> the calibration check from §9 section D.
>
> Report accuracy per SNR bucket. Never quote a single overall number.

---

## Phase 7 — API and dashboard

> Read `CLAUDE.md` §7 in full, including the design direction — follow it rather than reaching for a
> default dashboard look.
>
> Build the FastAPI app with every endpoint listed. Background jobs in a thread with an in-memory job
> table. No Celery, no Redis. Stream uploads to a temp file.
>
> Then the static dashboard in `web/`: vanilla JS, Plotly vendored locally, IBM Plex vendored as woff2.
> Build the analysis view first — spectrogram with clickable detection boxes, detection list, parameter
> panel, evidence panel. Drop screen second.
>
> No npm, no build step. It must run from `sigscope serve` with the network off.

---

## Phase 8 — hardening

> Run every test in `CLAUDE.md` §9 sections E and F. Fix everything that fails.
>
> Specifically: an empty file, a text file renamed to `.iq`, a 2 GB file, pure noise, a strong DC
> spike, and a fresh clone installed from scratch with the network disabled. Every failure must produce
> a clear message naming the problem, never a traceback.
>
> Then profile the pipeline, report seconds per megabyte, and finalise `ACCURACY.md`.

---

## Useful mid-build prompts

> The symbol rate estimator is reporting double the true rate on 2FSK. Check the harmonic
> disambiguation in the reconciliation step against `CLAUDE.md` §4.8.

> Detection is splitting one bursty transmission into many boxes. Review the time-gap merge threshold
> and show me before and after on scene_042.

> Show me the accuracy-vs-SNR curve for the ensemble against each individual model. If the ensemble is
> not beating both members at every SNR, the referee weighting in §5.5 is wrong.

> Confidence calibration is failing — predictions above 0.8 confidence are only 0.7 accurate. Add
> temperature scaling fitted on the validation split, and re-check.

> Add the Morse decoder from the stretch goals in §8: envelope threshold, dot/dash clustering, lookup
> table. Only now that Phase 8 passes.
