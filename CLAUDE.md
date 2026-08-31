# SIGSCOPE — complete build spec

**Problem:** SIH26147 (Smart India Hackathon 2026) — "Automated model for analysis of .IQ and .wav
files along with signal parameter extraction". Sponsor: National Technical Research Organisation
(NTRO). Category: Software.

This one file is the whole brief. Read it end to end before writing any code. `KICKOFF-PROMPTS.md`
next to it is just copy-paste prompts; it contains no information that is not here.

---

# 1. The problem

NTRO monitors the radio spectrum. Receivers record what they hear and dump it to disk in two shapes.

**`.iq` files** — the raw wave. A radio wave has both a size and a timing, so it cannot be stored with
one number per sample. It is stored as two: I (in-phase) and Q (quadrature), together forming one
complex number per sample. This preserves the signal exactly, including which side of the centre
frequency it sits on. These files usually have **no header at all** — sample rate, number format and
centre frequency have to be supplied or guessed.

**`.wav` files** — an ordinary audio container holding one of two very different things:
- Real audio, already demodulated down to sound.
- IQ in disguise: a stereo wav where left = I and right = Q. Every common SDR program (SDR#, HDSDR,
  SDRuno, KiwiSDR) writes this. Centre frequency is often hidden in the filename or a non-standard
  header chunk.

Today an analyst opens each file, eyeballs a waterfall display, and writes the numbers down by hand.
Slow, unscalable, and two analysts will disagree.

## What we deliver

For **every separate signal inside a file**:

| Output | Plain meaning |
|---|---|
| Centre frequency | Where on the dial the signal sits |
| Bandwidth | How wide it is |
| Start, stop, duration | When it was on air |
| SNR | Strength against the background hiss |
| Symbol rate | How fast data is being sent |
| Modulation | AM, FM, SSB, CW, FSK, PSK, QAM, OFDM, chirp |
| Sub-parameters | FSK deviation and tone count, PSK order, OFDM subcarrier spacing, chirp rate |
| Confidence + evidence | How sure we are, and why |

Plus a spectrogram with boxes drawn round each detection, machine-readable output, and batch mode.

## Hard constraints

- **Runs fully offline.** Air-gapped machine, no GPU, no internet, no cloud API, no CDN links.
- **Handles unknown files.** Judges will bring their own recording. It may have no header, an odd
  sample rate, several overlapping signals, or nothing but noise.
- **Says "I don't know".** Every output carries a confidence. Low confidence prints `unknown`, never a
  guess. A confident wrong number in front of an NTRO judge is worse than an honest gap.

## What wins and what loses

Judges are signal people. They reward: correct classical estimates they can check against their own
tools; an honest accuracy-vs-SNR curve including where we fail; handling a messy real capture; output
in SigMF annotation format, which their tooling already reads; and an explanation of *why* the
classifier said QPSK.

They punish: a single neural net with no classical cross-check; accuracy quoted with no SNR attached;
a demo that only works on the file we brought; anything needing internet.

## Scope

In: detection, measurement, classification, evidence, reporting, batch, dashboard.
Out unless everything above is finished: full protocol decoding, direction finding, geolocation, live
SDR capture. One exception — a Morse decoder is cheap and demos brilliantly, so it is a stretch goal.

---

# 2. Stack and conventions

## Dependencies — do not substitute

Python 3.11. `numpy`, `scipy` (all DSP), `soundfile` (wav I/O), `sigmf` (SigMF read/write),
`scikit-learn` (feature classifier, scaler, metrics), `torch` CPU-capable (the CNN), `fastapi` +
`uvicorn` (API), `pytest`.

Frontend: plain HTML, vanilla JS, `plotly.js` vendored as a local file. No React, no npm, no build step.

**No GNU Radio.** It is a nightmare to install on a demo laptop and we do not need it.

## Repo layout

```
sigscope/
  io/            file readers, format sniffing, metadata recovery
  dsp/           spectrogram, detection, parameter estimators
  features/      cumulants, instantaneous statistics
  models/        cnn.py, feature_clf.py, rules.py, ensemble.py
  data/          radioml.py loader, scene.py composer
  report/        json, sigmf annotations, html
  api/           fastapi app
  testsignals.py tiny known-truth generators, used ONLY by tests
  cli.py
data/
  radioml/       RML2016.10a.pkl (gitignored, downloaded once)
  samples/       small real captures, committed
  scenes/        composed test scenes (gitignored)
web/             static dashboard
tests/
```

## Coding rules

- Every estimator is a **pure function**: numpy array in, dataclass out. No global state, no I/O inside
  DSP code.
- Every estimator returns a value **and** a confidence in 0..1 **and** the method name. A number with
  no confidence is useless to an analyst.
- Sample rate is always passed explicitly as `fs: float`. Never assume 1.0.
- Complex baseband is always `np.complex64`. Convert once, at the edge, inside `sigscope/io/`.
- Hz, seconds, dB at every boundary. Convert at the edge only.
- Type hints everywhere. `ruff` clean. Each docstring names the algorithm and cites the section number
  of this document that it implements.

## Hard rules

- **Never fake a result.** If an estimator cannot get a reliable answer, return `None` with a reason.
- **Never hardcode a demo.** No `if filename == "demo.wav"`. Assume the judges bring their own file.
- Do not commit datasets or model weights over 20 MB. Ship a download script and a training script,
  plus one small checkpoint for the demo.
- Write the test before the estimator wherever the expected answer is known in closed form.

## Definition of done for any module

1. Unit test passes against a known-truth signal.
2. Works on at least one real capture in `data/samples/`.
3. Appears in the JSON report.
4. Has a row in the accuracy table produced by `scripts/evaluate.py`.

---

# 3. Architecture

```
file ─► INGEST ─► CONDITION ─► DETECT ─► ISOLATE ─► MEASURE ─► CLASSIFY ─► REPORT
        sniff      dc removal   time-freq  cut out    estimate   ensemble    json
        format     normalise    blobs      + mix down numbers    + evidence  sigmf
        + metadata resample     ► bursts   ► clean y  ► values   ► label     html
```

Each stage is a package. Each runs alone from the CLI — that matters when debugging in front of judges.

## Stage 1 — Ingest (`sigscope/io/`)

Job: turn any input into `(iq: np.complex64[N], meta: CaptureMeta)`.

Sniffing order:
1. A `.sigmf-meta` sidecar exists → trust it completely. Done.
2. `.wav` → read header with `soundfile`.
   - 2 channels → IQ (I = ch0, Q = ch1). Flag `iq_from_stereo=True`.
   - 1 channel → real audio. Convert to complex baseband with `scipy.signal.hilbert`.
   - Look for an `auxi` chunk — HDSDR and SDRuno store centre frequency and timestamp there.
3. `.iq`, `.dat`, `.bin`, `.cfile`, `.cs8`, `.cs16`, `.cf32` → headerless. Resolve by:
   - **Filename patterns.** Parse and record which matched:
     - `SDRSharp_YYYYMMDD_HHMMSSZ_<freq>Hz_IQ.wav`
     - `HDSDR_YYYYMMDD_HHMMSSZ_<khz>kHz_RF.wav`
     - `*_<rate>sps_*`, `*_fc<freq>_*`, and the `.cs8` / `.cs16` / `.cf32` extension convention
   - **Dtype guessing.** Try int8, int16-LE, float32-LE. For each, histogram the sample values. The
     wrong guess looks spiky, clipped, or bimodal; the right one looks roughly Gaussian. Score each,
     pick the best, and surface all three with scores so the user can override.
   - If sample rate is still unknown, set `fs = 1.0` and report every frequency as **normalised**
     (a fraction of sample rate), clearly labelled. Never print a fabricated Hz value.

`CaptureMeta`: `sample_rate`, `center_freq` (nullable), `dtype_guessed`, `dtype_confidence`,
`source_format`, `start_time`, `n_samples`, `metadata_confidence`, `notes: list[str]`.

## Stage 2 — Condition (`sigscope/dsp/condition.py`)

- Remove DC offset: subtract the mean, then apply a very narrow notch at 0 Hz if a spike remains. SDR
  hardware leaves a strong centre spike that will otherwise be detected as a signal.
- Normalise to unit average power, keeping the original scale factor in meta.
- Large files: process in overlapping blocks of 2^20 samples with 25% overlap, so a burst crossing a
  block edge is not cut in two.

## Stage 3 — Detect

Produces `Burst(t0, t1, f_lo, f_hi)` rectangles. Algorithm in §4.3.

## Stage 4 — Isolate + measure

Cut, mix to zero, filter, decimate; run every estimator on the clean burst. Algorithms in §4.4–§4.13.

## Stage 5 — Classify

Three opinions and a referee. §5.

## Stage 6 — Report

One internal object, three outputs: `report.json` (schema below), `capture.sigmf-meta` (SigMF
annotations, so real tools can read it), and a self-contained `report.html` openable with no server.

## Frozen output schema

Freeze this at the start. Everyone builds against it. Half of all hackathon failures are integration
failures on the last night.

```json
{
  "schema_version": "1.0",
  "file": { "name": "capture.iq", "sha256": "...", "bytes": 8388608 },
  "capture": {
    "sample_rate_hz": 2000000,
    "center_freq_hz": 100000000,
    "center_freq_source": "filename",
    "duration_s": 2.097,
    "dtype": "int16",
    "dtype_confidence": 0.91,
    "frequencies_are_normalised": false,
    "notes": ["centre frequency parsed from filename pattern SDRSharp_*"]
  },
  "noise_floor_dbfs": -74.2,
  "detections": [
    {
      "id": 1,
      "time_start_s": 0.104, "time_stop_s": 0.612, "duration_s": 0.508,
      "center_freq_hz": 100250000, "center_freq_offset_hz": 250000,
      "bandwidth_hz": { "occupied_99": 48200, "minus_3db": 31000, "minus_20db": 62400 },
      "snr_db": 21.4,
      "power_dbfs": -32.8,
      "symbol_rate_hz": { "value": 31250, "confidence": 0.88, "method": "cyclostationary" },
      "modulation": {
        "label": "QPSK",
        "confidence": 0.83,
        "runner_up": "8PSK",
        "runner_up_confidence": 0.11,
        "votes": { "feature_clf": "QPSK", "cnn": "QPSK", "rules": null },
        "evidence": [
          "C40 magnitude 0.98 is close to the QPSK theoretical value 1.00",
          "4th-power spectrum has one strong line, consistent with 4 phase states",
          "constant envelope, amplitude variance 0.02"
        ]
      },
      "extra": { "psk_order": 4, "carrier_offset_hz": 1240 }
    }
  ],
  "warnings": ["sample rate not found in file; 2 MHz assumed from filename"],
  "runtime_s": 3.9
}
```

Rules: anything unmeasurable is `null` plus a `warnings` entry, never a placeholder number.
`evidence` is human sentences, always present, minimum two. It is not decoration — it is the feature
that separates this from every other submission.

## CLI

```
sigscope analyse capture.iq --fs 2e6 --fc 100e6 --out report/
sigscope batch ./captures/ --workers 8 --out results.csv
sigscope fetch-data                       # download + convert RadioML 2016.10a
sigscope make-scenes --n 300 --out data/scenes/
sigscope train --model both
sigscope evaluate --out ACCURACY.md
sigscope serve --port 8000
```

---

# 4. Signal processing spec

Notation: `x[n]` complex baseband, `fs` sample rate, `N` length. Implement as pure functions.

## 4.1 Spectrogram

```python
f, t, Zxx = scipy.signal.stft(x, fs=fs, nperseg=nfft, noverlap=nfft*3//4,
                              window="hann", return_onesided=False)
S_db = 20*np.log10(np.abs(np.fft.fftshift(Zxx, axes=0)) + 1e-12)
```

Pick `nfft` adaptively: aim for ~2000 time columns across the file, clamp to a power of two in
[256, 8192]. Too few bins and narrow signals vanish; too many and short bursts smear.

`return_onesided=False` always — the signal is complex and negative frequencies are real information,
not a mirror.

## 4.2 Noise floor

Never use the mean; one strong signal drags it up. Per frequency bin:

```
noise_per_bin = np.percentile(S_db, 25, axis=1)
noise_floor   = np.median(noise_per_bin)
```

For a tighter estimate use the median absolute deviation of magnitudes: `sigma = 1.4826 * MAD`.

## 4.3 Detection

1. Build `S_db` (§4.1) and `noise_per_bin` (§4.2).
2. `mask = S_db > noise_per_bin[:, None] + threshold_db`, default `threshold_db = 8`.
3. Morphology on the mask: `binary_closing` with a small rectangle to join a burst broken by a fade,
   then `binary_opening` to kill single-pixel specks.
4. `scipy.ndimage.label` for connected components.
5. Reject components under `min_pixels` (default 20) or thinner than 2 frequency bins.
6. Bounding box of each survivor → `Burst`.
7. Merge boxes overlapping in frequency and separated in time by fewer than 3 STFT hops. This stitches
   a bursty transmission back into one detection instead of forty.

Special case: a component covering >90% of duration and >60% of bandwidth is probably the whole
channel. Flag it as continuous/wideband and still measure it.

All constants live in a `DetectorConfig` dataclass, not scattered as literals.

## 4.4 Isolating a burst

```
1. slice samples n0..n1 from t0, t1
2. f_c = (f_lo + f_hi) / 2
3. mix down:  y = x_slice * exp(-2j*pi*f_c*n/fs)
4. lowpass FIR: cutoff = 0.6*(f_hi - f_lo), 101 taps, firwin, hamming
5. filter, then decimate by D = floor(fs / (2.5*(f_hi - f_lo))) if D >= 2
6. new rate fs_b = fs / D
```

Everything downstream works on `y` at `fs_b`. Keep `D` — multiply frequency results back up before
reporting. Isolating first is the whole trick: estimators that fail on a crowded band work fine on one
clean burst.

## 4.5 Centre frequency

Power-weighted centroid over bins inside the box: `f_c = Σ P[k]·f[k] / Σ P[k]`, P linear.

Refine with parabolic interpolation around the peak bin. With dB values `y0,y1,y2` at bins `k-1,k,k+1`:
`delta = 0.5*(y0 - y2) / (y0 - 2*y1 + y2)`, so `f_peak = f[k] + delta*binwidth`.

Report both centroid (good for wide modulated signals) and peak (good for carriers and CW).
Confidence: high if the in-box PSD is smooth and unimodal; low if multi-humped, which usually means
two signals got merged into one box.

## 4.6 Bandwidth

Welch PSD over the burst, `nperseg=1024`. Three numbers, all reported:

- **OBW99** — walk the cumulative power inward from each edge until 0.5% is passed; the remaining span
  is the 99% occupied bandwidth. This is the headline number and the one regulators use.
- **−3 dB bandwidth** — width within 3 dB of peak.
- **−20 dB bandwidth** — same at 20 dB down; reveals splatter.

They disagree in informative ways and analysts want all three.

## 4.7 SNR

```
P_sig_plus_noise = mean power inside the burst box
P_noise_density  = noise floor power per Hz (§4.2), measured outside the box
P_noise_in_band  = P_noise_density * bandwidth
P_signal         = P_sig_plus_noise - P_noise_in_band
SNR_db           = 10*log10(P_signal / P_noise_in_band)
```

If `P_signal <= 0`, report `< 0 dB` with low confidence rather than NaN. SNR must be reported next to
every other estimate, because every other estimate degrades with it.

## 4.8 Symbol rate — the hard one

Three methods, then reconcile.

**(a) Cyclostationary / spectral line — primary.** A digitally modulated signal has hidden periodicity
at the symbol rate; a nonlinearity makes it visible as a spectral line.

```
1. y = isolated burst, unit power
2. z = |y|**2
3. z = z - mean(z)            # the DC term is huge and would swamp everything
4. Z = |FFT(z * blackmanharris)|, zero-padded 8x for resolution
5. peak = argmax of Z in [fs_b/1000, fs_b/2]
6. symbol_rate = frequency of that peak, parabolically refined
```

Confidence = peak height above the local median of `Z`, through a logistic: 12 dB → ~0.9, 3 dB → ~0.2.
Fails on MSK/GMSK because `|y|²` is flat; fall back to (b).

**(b) Instantaneous frequency transitions.**

```
phi    = np.unwrap(np.angle(y))
f_inst = np.diff(phi) * fs_b / (2*pi)
d      = np.abs(np.diff(f_inst))         # spikes at symbol transitions
D      = |FFT(d - mean(d))|
symbol_rate = argmax over the same range
```

Good for FSK, MSK, and anything with phase transitions.

**(c) Envelope autocorrelation.** Correlate `z` with itself, find the first strong non-zero-lag peak
`L`, then `symbol_rate = fs_b / L`. Cheap sanity check that catches octave errors.

**Reconciliation.** If two of three agree within 5%, report that with confidence ≥ 0.85. If all three
disagree, report (a) with confidence below 0.4 plus a warning. **Always run the harmonic check:** (a)
often locks onto 2× or 0.5× the true rate, so test whether half and double also show strong lines and
prefer the lowest rate that explains all observed lines. This is the single most common failure.

## 4.9 Carrier frequency offset and PSK order

For M-PSK, raising to the M-th power collapses the modulation and leaves a tone at `M · f_offset`:

```
for M in (1, 2, 4, 8):
    W = |FFT(y ** M)|
    sharpness[M] = peak height above median
f_offset = argmax(W_best) / M_best
```

The M giving the sharpest single line is also strong evidence for the PSK order. Feed `sharpness` into
the classifier as features.

## 4.10 FSK parameters

Histogram `f_inst` (§4.8b) into 200 bins over the burst. Peaks (`scipy.signal.find_peaks` with a
prominence threshold) are the FSK tones.

- `n_tones` = peak count → M in M-FSK (2, 4, 8).
- `deviation` = half the outer peak spread for 2FSK; tone spacing for M-FSK.
- `modulation_index h = deviation / symbol_rate`. `h ≈ 0.5` means MSK.

## 4.11 OFDM

OFDM repeats the end of each symbol at its start — the cyclic prefix — so autocorrelation spikes at a
lag equal to the useful symbol length.

```
R[lag] = |Σ y[n]·conj(y[n+lag])| / Σ|y[n]|²
```

A clear peak at lag `L` with `R > 0.3` means OFDM, useful symbol duration `L/fs_b`, subcarrier spacing
`fs_b/L`. Estimate CP length from the width of the correlation plateau. This is physics, not
statistics — trust it over the CNN when it fires.

## 4.12 Chirp / LFM

A chirp draws a straight line in the spectrogram.

```
1. per time column, argmax over frequency -> ridge f_peak[t]
2. keep columns whose peak is > 6 dB above that column's median
3. fit f = a·t + b  (np.polyfit degree 1)
4. if R² > 0.9 and |a| large -> chirp; chirp_rate = a (Hz/s);
   sweep_bandwidth = |a| * duration
```

Also fit degree 2 — a better quadratic fit means a nonlinear chirp, worth reporting.

## 4.13 Analogue modulations

- **AM** — envelope `|y|` varies significantly and its spectrum holds audio-band content (300 Hz–5 kHz).
  Depth = `(max_env − min_env)/(max_env + min_env)`.
- **FM** — envelope roughly constant, `f_inst` varies. Deviation = 99th percentile of `|f_inst|`, not
  the max; one glitch ruins the max.
- **SSB** — strongly asymmetric spectrum about centre. Ratio of upper-half to lower-half power > 10 dB
  says USB or LSB, and the sign says which.
- **CW / Morse** — envelope is on/off. Threshold smoothed `|y|`, then check the on-durations cluster
  into two groups with a ratio near 3:1 (dot and dash). If they do, you have Morse — and you can decode
  it (stretch goal).

---

# 5. Classification

## 5.1 Classes

The dataset (§6) fixes the trained label set:

```
8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
```

Three more come from deterministic rules only, not from training: `ofdm`, `chirp-lfm`, `cw`.
Plus `noise` (rule) and `unknown` (low confidence). `unknown` is never a trained class.

Be explicit in the UI and the pitch about which labels are learned and which are rule-based. Do not
quietly imply the model was trained on OFDM when it was not.

## 5.2 Handcrafted features (`sigscope/features/`)

Computed on the isolated, power-normalised burst. All cheap, all explainable to a judge.

**Higher-order cumulants** — the classic tell, because each scheme has a known theoretical value:

```
M20=mean(y**2)  M21=mean(|y|**2)  M40=mean(y**4)
M41=mean(y**3*conj(y))  M42=mean(|y|**4)  M63=mean(|y|**6)

C20=M20   C21=M21
C40=M40 - 3*M20**2
C41=M41 - 3*M20*M21
C42=M42 - |M20|**2 - 2*M21**2
C63=M63 - 9*C42*C21 - 6*C21**3
```

Feed magnitudes plus the ratios `|C40|/C21²` and `|C42|/C21²`. Theoretical values, used both as
features and to fill the evidence sentences:

| Modulation | \|C40\|/C21² | \|C42\|/C21² |
|---|---|---|
| BPSK | 2.00 | 2.00 |
| QPSK | 1.00 | 1.00 |
| 8PSK | 0.00 | 1.00 |
| 16QAM | 0.68 | 0.68 |
| 64QAM | 0.62 | 0.62 |
| Gaussian noise | 0.00 | 0.00 |

**Instantaneous statistics** from `a=|y|`, `phi=unwrap(angle(y))`, `f_inst=diff(phi)`:
`gamma_max` (max PSD of the centred normalised amplitude / N), `sigma_ap`, `sigma_dp`, `sigma_aa`,
`sigma_af`, kurtosis of amplitude, kurtosis of instantaneous frequency, PAPR in dB.

**Spectral features:** upper/lower power asymmetry in dB (catches SSB); peak count in the `f_inst`
histogram (catches M-FSK order); squared- and 4th-power spectral line strengths (§4.9); cyclic-prefix
autocorrelation peak (§4.11); spectral flatness (OFDM is flat-topped, PSK is raised-cosine rounded).

About 30 features. `StandardScaler` fitted on training data only.

**Model:** `sklearn.ensemble.HistGradientBoostingClassifier`, ~300 iterations, saved with joblib.
Also export `permutation_importance` — that is what generates evidence sentences.

## 5.3 CNN on raw IQ (`sigscope/models/cnn.py`)

Input `2 × 128` float32 (matching the dataset), I and Q as channels, power-normalised.

```
Conv1d(2, 64, k=7, pad=3) -> BN -> ReLU
2 x ResidualBlock(64, k=5)    each: Conv-BN-ReLU-Conv-BN + skip, then MaxPool(2)
2 x ResidualBlock(128, k=5)
AdaptiveAvgPool1d(1) -> Dropout(0.3) -> Linear(128, n_classes)
```

~250k parameters. Trains on CPU in well under an hour on 220k short examples.

Adam, lr 1e-3, cosine schedule, batch 256, 40 epochs, early stop on val loss, label smoothing 0.05.

At inference on a real burst, the signal is longer than 128 samples: slide a 128-sample window with
50% overlap across the burst, average the softmax outputs, and report the spread across windows as an
extra confidence signal — high disagreement between windows means low confidence.

**Report validation accuracy per SNR bucket, never as a single overall number.** A single number is
meaningless and a judge will say so.

## 5.4 Deterministic rules (`sigscope/models/rules.py`)

These override the learned models when they fire, because they are physics:

| Rule | Fires when | Output |
|---|---|---|
| OFDM | CP autocorrelation peak > 0.3 at a consistent lag | `ofdm` |
| Chirp | linear ridge fit R² > 0.9 and a large sweep | `chirp-lfm` |
| CW | on/off envelope with dot/dash ratio near 3:1 | `cw` |
| Noise | SNR < 3 dB and spectral flatness > 0.9 | `noise` |
| SSB | spectral asymmetry > 10 dB | `AM-SSB`, with the side |

## 5.5 Ensemble referee

```
1. If a deterministic rule fires, take it, confidence 0.9. Record the other votes anyway.
2. Otherwise blend the softmax of feature_clf and cnn, weighted by their validation accuracy
   at the measured SNR bucket. High SNR favours the feature classifier, low SNR the CNN.
3. label = argmax of the blend.
4. If max probability < 0.45 -> "unknown", keeping the top two as candidates.
5. If the two models disagree on the top class, cap confidence at 0.6 and add a warning.
```

## 5.6 Evidence sentences

Every classification emits 2–4 plain sentences, templates filled with the measured values:

- "C40 magnitude {v:.2f} is close to the {label} theoretical value {t:.2f}."
- "Raising to the power {M} produced a single strong spectral line, indicating {M} phase states."
- "Envelope is nearly constant (amplitude variance {v:.3f}), so this is not an amplitude scheme."
- "The instantaneous frequency histogram has {n} distinct peaks spaced {d:.0f} Hz apart."
- "Autocorrelation peaks at lag {L} samples, consistent with a cyclic prefix."
- "Measured SNR is {s:.1f} dB. Below 5 dB our accuracy for this class drops to {a:.0%}, so treat this
  result with caution."

That last one is the single most credibility-building line in the product. Always emit it at low SNR.

---

# 6. Data — use an existing dataset, do not generate one

## 6.1 Primary dataset: RadioML 2016.10a

The standard public benchmark for modulation classification, generated by DeepSig and released at the
2016 GNU Radio Conference.

- 11 modulations: 8 digital (BPSK, QPSK, 8PSK, QAM16, QAM64, CPFSK, GFSK, PAM4) and 3 analogue
  (WBFM, AM-DSB, AM-SSB).
- 20 SNR levels, −20 dB to +18 dB in 2 dB steps.
- 1000 examples per (modulation, SNR) pair → 220,000 total.
- Each example is 128 complex samples, stored as shape `(2, 128)` float32.
- It already includes realistic impairments: multipath fading, carrier frequency offset, sample clock
  offset, and AWGN. That is why we do not need our own channel model.
- Format: a Python pickle dict keyed `(mod_name, snr_db)` → array of shape `(1000, 2, 128)`.

**Where to get it.** The original from `deepsig.ai/datasets` is a 640 MB pickle written with protocol
0. A re-serialised, byte-identical version at 225 MB exists on Zenodo, and Kaggle carries mirrors.
Prefer the smaller one. Whichever you use, verify the shape: 220 keys, each `(1000, 2, 128)`.

Write `sigscope fetch-data` to accept a local path or a URL, verify a SHA-256, convert once to a
memory-mapped `.npy` plus a `labels.parquet` (columns: `index`, `modulation`, `snr_db`), and cache
that. Never re-parse the pickle at training time. Add `data/radioml/` to `.gitignore`.

**Splits.** Shuffle across modulation and SNR, then split 70/15/15 train/val/test with a fixed seed.
Report on the test split only. Publish the seed.

**Known limits — say these out loud in the pitch rather than being caught on them.** 128 samples is
short; it is enough to classify but not to estimate a symbol rate precisely, which is why our symbol
rate comes from classical DSP on the real burst and not from this dataset. The dataset is baseband and
single-signal, so it cannot be used to test detection. And it is itself synthetic, which is why we
validate on real captures too.

## 6.2 Scenes for testing detection (`sigscope/data/scene.py`)

Detection needs wideband files with several signals; the dataset has none. Compose them instead of
inventing new modulators:

```
1. make an empty complex canvas: duration 5-30 s at 1-10 MHz sample rate
2. pick 1-8 examples from RadioML at chosen SNRs
3. for each: repeat/interpolate it up to the target duration, mix to a random
   frequency offset, scale to a target power, and add at a random start time
4. add complex Gaussian noise to the whole canvas
5. record the ground truth: for each placed signal, its time span, frequency span,
   SNR, and true modulation label
6. write the scene as raw int16 IQ plus a ground-truth SigMF annotation file
```

Include awkward cases deliberately: a signal at the very edge of the band; two signals one bandwidth
apart; two overlapping in both time and frequency; an empty file; pure noise; a strong DC spike; a
single continuous full-band signal.

300 scenes is plenty. `sigscope make-scenes --n 300`. Deterministic given a seed.

## 6.3 Test signals for unit tests (`sigscope/testsignals.py`)

You cannot unit-test a symbol rate estimator without a signal whose symbol rate you know. This file is
about 80 lines and exists **only** for tests — it is not a training data generator and it is never
imported by the pipeline:

```python
tone(f, fs, n)                     # complex exponential
psk(order, symbol_rate, fs, n_sym) # RRC-shaped, roll-off 0.35
fsk(order, deviation, symbol_rate, fs, n_sym)   # continuous phase
lfm_chirp(f0, f1, duration, fs)
ofdm(n_sc, cp_len, n_sym, fs)
add_awgn(x, snr_db)
```

Keep it that small. Anything more and you are rebuilding the generator we deliberately removed.

## 6.4 Real captures (`data/samples/`)

Synthetic training, real validation. Collect ten or so small files and commit them:

- Public SDR recordings: KiwiSDR captures, university SDR archives, signal identification wikis that
  publish labelled sample audio per signal type.
- If anyone on the team has an RTL-SDR, capture FM broadcast, an airband AM transmission, a pager or
  ADS-B burst, and some amateur band traffic. Ten minutes of real capture is worth a lot in the demo.

**Never train on these.** They are the honesty check, and the thing that separates us from teams who
only ever ran on RadioML.

---

# 7. API and dashboard

## API (`sigscope/api/`)

FastAPI, local only, started by `sigscope serve`. Leave a comment where auth would go.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyse` | Multipart upload plus optional `fs`, `fc`. Returns a job id. |
| GET | `/api/jobs/{id}` | `queued` / `running` / `done` / `failed`, with progress 0–1. |
| GET | `/api/jobs/{id}/report` | The full JSON report. |
| GET | `/api/jobs/{id}/spectrogram.png` | Rendered spectrogram. |
| GET | `/api/jobs/{id}/spectrogram.json` | Downsampled dB matrix for browser plotting. |
| GET | `/api/jobs/{id}/detections/{n}/audio.wav` | Demodulated audio, where demodulable. |
| GET | `/api/jobs/{id}/sigmf` | SigMF `.sigmf-meta` download. |
| POST | `/api/batch` | A folder path on disk. Returns a job id; results as CSV. |
| GET | `/api/health` | Version, model checksums, whether models are loaded. |

Background jobs in a thread with an in-memory job table. No Celery, no Redis — one more thing to fail
on the demo laptop. Files up to 2 GB: stream to a temp file, never read an upload fully into memory.

## Dashboard (`web/`)

Plain HTML, vanilla JS, Plotly vendored locally, served as static files by FastAPI. No build step.

**Screens.**

*Drop file* — a full-height drop zone, plus two optional inputs (sample rate, centre frequency) with
the note "leave blank and we will work it out from the file", and a short list of filename patterns we
can parse. When a file lands, show what we detected about it — format, sample rate, where that guess
came from — *before* the analysis finishes. That early feedback is what makes it feel fast.

*Analysis view* — the screen the judges will look at.

```
┌──────────────────────────────────────────────────────────┬───────────────┐
│                                                          │  DETECTIONS   │
│   spectrogram, time across, frequency up                 │  ┌─────────┐  │
│   detections drawn as boxes, click to select             │  │ #1 QPSK │  │
│                                                          │  │ 31.2 kBd│  │
│                                                          │  └─────────┘  │
│                                                          │  ┌─────────┐  │
│                                                          │  │ #2 GFSK │  │
├──────────────────────────────────────────────────────────┴───────────────┤
│  SELECTED: #1                                                            │
│  centre 100.250 MHz   bandwidth 48.2 kHz (99%)   SNR 21.4 dB             │
│  symbol rate 31 250 Bd (confidence 0.88)                                 │
│  modulation QPSK (confidence 0.83, runner-up 8PSK)                       │
│                                                                          │
│  Why we think so                                                         │
│  · C40 magnitude 0.98 is close to the QPSK theoretical value 1.00        │
│  · 4th-power spectrum has one strong line — four phase states            │
│  · constant envelope, amplitude variance 0.02                            │
│                                                                          │
│  [ constellation ]  [ spectrum ]  [ inst. frequency ]                    │
└──────────────────────────────────────────────────────────────────────────┘
```

*Batch view* — sortable table, one row per file, filter by modulation, export CSV.

**Design direction.** The subject is a spectrum monitoring instrument used by an analyst who stares at
it for hours. That points somewhere specific, and it is not a startup landing page. Follow this rather
than reaching for a default dashboard look.

- Dark, because a spectrogram is a light-on-dark object and a white page around it destroys the
  contrast the analyst needs. Base `#0d1117`, panel `#161b22`, rule `#2d333b`, text `#c9d1d9`.
- One accent, `#4dd0c4`, used only for the selected detection. Detection boxes are hairline `#8b949e`
  when idle. Colour never decorates here — it means "selected" or "warning", nothing else.
- Spectrogram colormap: viridis. Perceptually uniform, standard in the field, colourblind-safe. Not
  jet — jet invents features that are not there and any RF engineer will notice.
- One type family: IBM Plex Sans for the interface, IBM Plex Mono for measured values only. Numbers
  are the content of this product, so they get tabular figures and are set larger than the labels
  beside them. Vendor as woff2; no font CDN.
- The spectrogram takes two-thirds of the screen and never shrinks below it. Everything else is a
  panel around it. Left-aligned throughout.
- No motion except the selection highlight and the progress bar. An instrument that animates for fun
  is an instrument that annoys you by hour three.
- Empty and error states say what happened and what to do: "No signals found above 8 dB over the noise
  floor. Try lowering the detection threshold." Not "No results."

Spend the boldness in one place — the spectrogram with live boxes is the signature element. Keep
everything around it quiet.

---

# 8. Build plan

Eight phases. Each ends with something that runs. Do not start a phase before the previous checkpoint
passes. If time runs out, everything through Phase 5 is already a submittable project.

**Phase 0 — skeleton (1 h).** Repo layout, `pyproject.toml`, ruff and pytest config, `.gitignore`
excluding `data/radioml/`, `data/scenes/`, `*.pt`. `sigscope/types.py` with every dataclass matching
the frozen schema. CLI with all subcommands printing "not implemented".
*Checkpoint:* `sigscope --help` works, `pytest` runs clean.

**Phase 1 — data in (2 h).** `sigscope fetch-data` downloading/converting RadioML 2016.10a to memmap
`.npy` + `labels.parquet`, with shape verification. `sigscope/testsignals.py` (§6.3).
*Checkpoint:* plot 20 examples per class at 18 dB SNR — constellation, spectrum, instantaneous
frequency. Look at them. QPSK must show four dots. Print the class and SNR distribution.

**Phase 2 — file I/O (3 h).** Wav mono and stereo-IQ, headerless raw with dtype guessing, SigMF,
filename parsers. Honest confidences and a `notes` list explaining every guess.
*Checkpoint:* round-trip — write a signal as headerless int16, read it back with dtype guessed,
samples match to within quantisation.

**Phase 3 — detection (4 h).** §4.1–§4.3, all constants in `DetectorConfig`. Then the scene composer
(§6.2); generate 300 scenes with ground truth.
*Checkpoint:* precision and recall both ≥ 0.90 for signals above 10 dB SNR. Print the numbers by SNR band.

**Phase 4 — classical estimators (6 h).** §4.4–§4.13, one pure function each. Isolation first. For
symbol rate, all three methods plus reconciliation **including the harmonic check**. Tests as you go.
*Checkpoint:* `pytest` green, first version of `ACCURACY.md` with estimator error tables.

**Phase 5 — end to end (2 h).** Wire the pipeline. JSON, SigMF and HTML writers. `analyse` and `batch`
working for real. Modulation reads `unclassified` for now.
*Checkpoint:* a report you would show someone, produced from every file in `data/samples/`.
**Commit and tag. This is submittable on its own.**

**Phase 6 — classifier (5 h).** Features → gradient boosting → rules → CNN → referee → evidence
sentences, in that order. Sliding-window inference for bursts longer than 128 samples.
*Checkpoint:* accuracy-vs-SNR curve, confusion matrices at three SNR bands, calibration check.

**Phase 7 — API and dashboard (5 h).** Endpoints, then the static UI. Build the analysis view first —
it is what gets demoed. Drop screen second.
*Checkpoint:* drag a file from the desktop, see boxes, click one, read the evidence.

**Phase 8 — hardening (4 h).** Every robustness test in §9. Time it, report seconds per megabyte.
Finalise `ACCURACY.md`. Rehearse the demo three times on the actual demo laptop with wifi off.

**Stretch goals, in priority order.** (1) Morse decoder — cheap, and hearing the tool read a message
is the best demo moment available. (2) Playable demodulated audio for AM and FM, one button, huge
impact. (3) Frequency hopping detection — cluster detections by bandwidth and duration; many short
same-width bursts at different frequencies with regular spacing means a hopper, and NTRO will care a
lot. (4) Spectrogram CNN as a third ensemble member. (5) Soft symbol decisions for the clean cases.
Do not start any of these before Phase 8 is done.

**Team split.** A+B: estimators (Phases 3–4), the DSP-heavy pair. C: file I/O and hardening.
D: classifier, starting with the training loop against Phase 1's output. E: API and dashboard, and can
start on day one against mock JSON because the schema is frozen. F: evaluation harness, `ACCURACY.md`,
slides, demo script.

---

# 9. Acceptance tests

If a test here fails, the feature is not done, however good the demo looks.

**A. Estimators against known truth** (from `testsignals.py`), each run at SNR 20, 10, 5, 0 dB with the
break point recorded. Pass conditions at 15 dB:

| Test | Pass |
|---|---|
| Centre frequency, single tone at +137 kHz | within 1 FFT bin |
| Centre frequency, QPSK at +250 kHz | within 2% of bandwidth |
| OBW99 of RRC QPSK, roll-off 0.35 | within 10% of `(1+β)·Rs` |
| SNR estimate at a set SNR | within 2 dB |
| Symbol rate, QPSK at 10 kBd | within 2% |
| Symbol rate, 2FSK at 4.8 kBd | within 3% |
| FSK tone count and deviation, 4FSK | correct count, deviation within 10% |
| OFDM symbol duration and subcarrier spacing | within 2% |
| Chirp rate, 1 MHz in 10 ms | within 5% |
| PSK order via M-th power, orders 2/4/8 | correct |
| \|C40\|/C21² for BPSK/QPSK/8PSK | within 0.15 of the table |

**B. File I/O.** Stereo wav round-trip to 1e-6. Mono real wav converts without crashing. Headerless
int8/int16/float32 dtype guessed correctly ≥ 90% of the time. SigMF pair read exactly with no guessing.
SDR#-style filename parsed. 2 GB file processed in blocks with peak RSS under 2 GB. Empty file, and a
text file renamed `.iq`, both give a clean message naming the problem — never a traceback.

**C. Detection, against the scene ground truth.** Precision and recall ≥ 0.90 above 10 dB, ≥ 0.70 in
3–10 dB. Two signals one bandwidth apart reported as two. A bursty signal with sub-3-hop gaps merged
into one. Pure noise gives zero detections and says so plainly. A continuous full-band signal is one
detection flagged wideband. A strong DC spike is not reported as a signal.

**D. Classification, on the RadioML test split.** ≥ 0.85 at SNR ≥ 15 dB (published results on this
dataset plateau around 0.85–0.90, so do not claim more). ≥ 0.70 at 5–15 dB. At SNR < 5 dB, most errors
must land on `unknown` rather than on a wrong confident label. **Calibration:** among predictions with
confidence above 0.8, actual accuracy must exceed 0.8 — this matters more than raw accuracy. Every
classification carries at least two evidence sentences. On the real captures in `data/samples/`, the
modulation is correct on at least 7 of 10.

**E. End to end.** A 10 s, 2 MHz scene completes in under 30 s on a laptop CPU. A batch of 100 files
completes with one CSV row each and no crash. Report JSON validates against the schema. SigMF output
loads in the `sigmf` library. **Everything works identically with wifi disabled.** A fresh clone plus
`pip install -e .` reaches a working demo in under 10 minutes on a clean machine.

The last two bite hardest on demo day. Test them the night before, on the demo laptop.

**F. The judge test.** Someone outside the team hands you a capture you have never seen, with no
metadata, and you get five minutes. If you cannot produce a sensible report, you are not done.

---

# 10. Demo and pitch

## The six minutes

Wifi off. Everything local. Rehearsed three times on the demo machine.

**0:00 — the problem, 30 seconds.** "An analyst gets a folder of radio recordings. Today they open each
one, look at a waterfall, and write the numbers down by hand. A thousand files is a week of work. We
made that automatic."

**0:30 — drop a real capture in.** Boxes appear. "Four signals. It found them itself. We told it
nothing about the file."

**1:15 — click a detection.** Read out centre frequency, bandwidth, SNR, symbol rate, modulation.

**1:45 — the evidence panel. This is the moment.** "Here is why it says QPSK. The fourth-order cumulant
is 0.98; the theoretical value for QPSK is 1.00. The fourth-power spectrum shows one line, meaning four
phase states. An analyst can check every one of these by hand. We are not asking anyone to trust a
black box."

**2:30 — click a weak signal.** Low confidence, label `unknown`. "This one is at 4 dB. It says it does
not know, and it says why."

**3:15 — Morse, if it made it in.** Play the decoded text.

**3:45 — batch mode.** Run 200 files, show the CSV filling. Quote seconds per megabyte.

**4:15 — SigMF export.** "The standard annotation format. It drops into existing tooling. We are not
asking anyone to adopt our format."

**4:45 — the accuracy slide.** The accuracy-vs-SNR curve, the confusion matrix, and out loud: "here is
where we fail."

**5:30 — offline.** "Everything you just saw ran on this laptop with the network off."

## The one slide that wins it

Not the architecture. The accuracy-vs-SNR curve with the failure region marked and labelled. Every
other team will quote one accuracy number with no conditions. Showing the curve, including the part
where you lose, tells a signals judge you actually did the work.

## Questions they will ask

**"Where did your data come from?"** RadioML 2016.10a, the standard public benchmark — 220,000
examples, 11 modulations, −20 to +18 dB, with multipath, carrier offset and clock offset already baked
in. We use the published test split so our numbers are directly comparable to the literature. We
validate on real captures we never trained on.

**"Isn't that dataset synthetic too?"** Yes, and that is exactly why we do not stop there. Our
parameter estimates come from classical DSP, which we test against signals with known ground truth,
and we validate the whole pipeline on real recordings.

**"What about two overlapping signals?"** Separated in frequency, we resolve them. Overlapping in both
time and frequency, we report one detection and flag it as a possible blend — the spectral shape is
multi-humped and we detect that. Full source separation is genuinely hard and we are not claiming it.

**"How do you know the sample rate on a headerless file?"** We do not, unless it is in the filename or
a sidecar. When we do not know, we report frequencies as a fraction of sample rate and say so. We never
print a made-up Hz value.

**"Why not just one neural network?"** Because an analyst cannot check a neural network. Classical
estimators run alongside it, and where physics gives a deterministic answer — cyclic prefix for OFDM,
ridge fit for chirp — the rule overrides the model. The network is a tie-breaker, not the authority.

**"Would this run on our systems?"** Pure Python, CPU only, no internet, no GPU, about 250k model
parameters. It runs on the laptop in front of you.

**"What is missing?"** Demodulation to bits for most schemes, co-channel source separation, and
direction finding. We know roughly how to do the first one.

## Language

Say "raw radio recording" before you say "IQ file". Say "how fast the data is being sent" before
"symbol rate". A signals judge will not mind the plain version; a non-signals judge will be lost
without it. And never say "AI-powered" without immediately saying what the model actually does — on
this panel that phrase is a red flag, not a selling point.
