"""FastAPI app factory and route definitions (CLAUDE.md §7 "API").

Every endpoint in §7's table, backed by the :mod:`sigscope.api.jobs` thread pool. Local
only, no auth (see the note at :func:`create_app`), no Celery, no Redis.

**Uploads stream to a temp file.** §7 allows files up to 2 GB and says an upload must never
be read fully into memory; :func:`_stream_upload` copies the request body through a 1 MB
buffer straight to disk, so peak RSS is unrelated to the file size. The 2 GB ceiling is
enforced while streaming rather than from a header, because a ``Content-Length`` is a claim
by the client and not a fact.

The static dashboard is mounted at ``/`` from ``web/``. Nothing on any page is fetched from
a network: §2's offline rule is a property of the files, not of a setting.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sigscope import __version__ as SIGSCOPE_VERSION
from sigscope.api.audio import demodulate
from sigscope.api.jobs import Job, JobError, JobState, JobStore
from sigscope.batch import CSV_COLUMNS, find_captures, run_batch
from sigscope.dsp.estimators import isolate_burst
from sigscope.io import CaptureError, read_capture
from sigscope.pipeline import AnalysisConfig, analyse_file
from sigscope.report import build_sigmf_meta, render_spectrogram_png
from sigscope.report.html_report import _max_pool
from sigscope.types import Burst

__all__ = ["create_app", "MAX_UPLOAD_BYTES"]

MAX_UPLOAD_BYTES = 2 * 1024**3  # §7: "Files up to 2 GB"
UPLOAD_CHUNK = 1 << 20
WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


async def _stream_upload(upload: UploadFile, destination: Path) -> int:
    """Copy an upload to disk in fixed-size chunks. Never materialises the whole file."""
    written = 0
    with open(destination, "wb") as handle:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"upload exceeds the {MAX_UPLOAD_BYTES // 1024**3} GB limit. "
                        "Analyse it from disk with `sigscope analyse` instead."
                    ),
                )
            handle.write(chunk)
    return written


def _require(job: Job | None, job_id: str) -> Job:
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}; it may have been evicted")
    return job


def _require_done(job: Job) -> Job:
    if job.state is JobState.FAILED:
        raise HTTPException(status_code=409, detail=job.error or "the job failed")
    if job.state is not JobState.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.state.value} ({job.progress:.0%}); poll /api/jobs/{job.id}",
        )
    return job


def _analysis_of(job: Job) -> Any:
    result = job.result
    if not isinstance(result, dict) or "analysis" not in result:
        raise HTTPException(status_code=409, detail="this job has no analysis result")
    return result["analysis"]


def _model_checksums() -> dict[str, str | None]:
    """§7 ``/api/health``: "Version, model checksums, whether models are loaded"."""
    from sigscope.models.cnn import DEFAULT_CNN_PATH
    from sigscope.models.feature_clf import DEFAULT_MODEL_PATH

    out: dict[str, str | None] = {}
    for name, path in (("feature_clf", DEFAULT_MODEL_PATH), ("cnn", DEFAULT_CNN_PATH)):
        if not path.is_file():
            out[name] = None
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(UPLOAD_CHUNK), b""):
                digest.update(chunk)
        out[name] = digest.hexdigest()[:16]
    return out


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------


def create_app(*, store: JobStore | None = None, web_root: Path | None = None) -> FastAPI:
    """Build the local API (CLAUDE.md §7).

    AUTH: there is deliberately none. §7 specifies a local-only tool and says to "leave a
    comment where auth would go" -- this is that comment. If this were ever exposed beyond
    localhost it would need, at minimum: an auth dependency on every ``/api`` route, a
    same-origin or token check on ``POST /api/batch`` (which reads an arbitrary directory
    path from the request and would otherwise be a filesystem-disclosure primitive), and a
    real CORS policy. Until then, bind to 127.0.0.1 and leave it there.
    """
    store = store or JobStore()
    web_root = web_root or WEB_ROOT

    app = FastAPI(
        title="SIGSCOPE",
        version=SIGSCOPE_VERSION,
        description="Automated analysis of .iq and .wav radio captures (SIH26147 / NTRO).",
    )
    app.state.store = store

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from sigscope.models import load_classifiers

        classifiers = load_classifiers()
        return {
            "status": "ok",
            "version": SIGSCOPE_VERSION,
            "models": {
                "feature_clf_loaded": classifiers.feature.is_trained,
                "cnn_loaded": classifiers.cnn.is_trained,
                "checksums": _model_checksums(),
                "note": (
                    None
                    if classifiers.any_trained
                    else "no trained checkpoints; modulation comes from the §5.4 "
                    "deterministic rules alone"
                ),
            },
            "jobs": {
                "total": len(store.list()),
                "running": sum(1 for j in store.list() if j.state is JobState.RUNNING),
            },
            "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES},
        }

    # ---------------------------------------------------------------- analyse
    @app.post("/api/analyse")
    async def analyse(
        # B008: calling File()/Form() in a default is FastAPI's own declaration syntax,
        # not the accidental-shared-mutable-default the rule is aimed at.
        file: UploadFile = File(...),  # noqa: B008
        fs: float | None = Form(default=None),  # noqa: B008
        fc: float | None = Form(default=None),  # noqa: B008
        dtype: str | None = Form(default=None),  # noqa: B008
        threshold_db: float | None = Form(default=None),  # noqa: B008
    ) -> dict[str, Any]:
        """Upload a capture and start analysing it. Returns a job id (§7)."""
        name = Path(file.filename or "capture").name
        # deliberately not a context manager: the file has to outlive this scope, and
        # the job owns it from here (JobStore._cleanup deletes it)
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="sigscope-", suffix=f"-{name}", delete=False
        )
        destination = Path(handle.name)
        handle.close()
        try:
            size = await _stream_upload(file, destination)
        except HTTPException:
            raise
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"could not save upload: {exc}") from exc
        if size == 0:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"{name}: the uploaded file is empty")

        job = store.create("analyse", name, temp_paths=[destination])

        def work(job: Job) -> dict[str, Any]:
            from dataclasses import replace as dataclass_replace

            cfg = AnalysisConfig()
            if threshold_db is not None:
                cfg = dataclass_replace(
                    cfg, detector=dataclass_replace(cfg.detector, threshold_db=threshold_db)
                )
            store.advance(job, "reading")
            try:
                # analyse_file, not read_capture + analyse_iq: it sizes the capture first
                # and routes anything large to the segmented streaming path. §7 accepts
                # uploads up to 2 GB, and reading one of those whole would undo the
                # streaming the upload itself was careful to do.
                analysis = analyse_file(
                    destination,
                    fs=fs,
                    fc=fc,
                    dtype=dtype,
                    cfg=cfg,
                    on_progress=lambda stage, fraction: store.advance(job, stage, fraction),
                )
            except CaptureError as exc:
                # a bad file is an expected outcome, and io already names the problem;
                # swap the temp path for the name the user uploaded
                raise JobError(str(exc).replace(str(destination), name)) from exc
            analysis.report.file.name = name
            return {"analysis": analysis, "source": destination}

        store.submit(job, work)
        return {"job_id": job.id, "name": name, "bytes": size}

    # ---------------------------------------------------------------- batch
    @app.post("/api/batch")
    def batch(payload: dict[str, Any]) -> dict[str, Any]:
        """Analyse a folder on disk. Returns a job id; results as CSV (§7).

        AUTH: this reads a caller-supplied path from the local filesystem, which is exactly
        why the app must stay bound to localhost until an auth dependency exists.
        """
        raw = str(payload.get("path", "")).strip()
        if not raw:
            raise HTTPException(status_code=400, detail="pass a folder path as `path`")
        folder = Path(raw).expanduser()
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"{folder} is not a directory")

        paths = find_captures(folder, str(payload.get("pattern", "*")))
        if not paths:
            raise HTTPException(
                status_code=400, detail=f"no capture files found in {folder}"
            )

        job = store.create("batch", folder.name)

        def work(job: Job) -> dict[str, Any]:
            store.advance(job, "running", 0.05)
            rows: list[dict[str, Any]] = []
            for done, path in enumerate(paths, start=1):
                rows.append(
                    run_batch(
                        [path],
                        workers=1,
                        fs=payload.get("fs"),
                        fc=payload.get("fc"),
                        progress=False,
                    )[0]
                )
                store.advance(job, "running", 0.05 + 0.9 * done / len(paths))
            store.advance(job, "reporting")
            return {"rows": rows, "columns": list(CSV_COLUMNS), "folder": str(folder)}

        store.submit(job, work)
        return {"job_id": job.id, "n_files": len(paths), "folder": str(folder)}

    # ---------------------------------------------------------------- job state
    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in reversed(store.list())]}

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: str) -> dict[str, Any]:
        return _require(store.get(job_id), job_id).to_dict()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, Any]:
        return {"deleted": store.discard(job_id)}

    @app.get("/api/jobs/{job_id}/report")
    def job_report(job_id: str) -> JSONResponse:
        """The full §3 JSON report."""
        job = _require_done(_require(store.get(job_id), job_id))
        if job.kind == "batch":
            return JSONResponse(job.result)
        return JSONResponse(_analysis_of(job).report.to_dict())

    @app.get("/api/jobs/{job_id}/csv")
    def job_csv(job_id: str) -> Response:
        """Batch results as CSV, one row per file (§7, §9 E)."""
        import csv
        import io as _io

        job = _require_done(_require(store.get(job_id), job_id))
        if job.kind != "batch":
            raise HTTPException(status_code=409, detail="this job is not a batch")
        buffer = _io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=job.result["columns"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(job.result["rows"])
        return Response(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{job.name}-results.csv"'},
        )

    # ---------------------------------------------------------------- spectrogram
    @app.get("/api/jobs/{job_id}/spectrogram.png")
    def spectrogram_png(job_id: str) -> Response:
        """Rendered spectrogram, viridis, with the detection boxes drawn in (§7)."""
        job = _require_done(_require(store.get(job_id), job_id))
        analysis = _analysis_of(job)
        if analysis.spectrogram is None:
            raise HTTPException(status_code=404, detail="this capture has no spectrogram")
        cached = job.artefacts.get("spectrogram_png")
        if cached and Path(cached).is_file():
            return FileResponse(cached, media_type="image/png")
        png, _, _ = render_spectrogram_png(analysis.spectrogram, analysis.bursts)
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- job-owned, see above
            prefix="sigscope-spec-", suffix=".png", delete=False
        )
        handle.write(png)
        handle.close()
        job.artefacts["spectrogram_png"] = Path(handle.name)
        return Response(png, media_type="image/png")

    @app.get("/api/jobs/{job_id}/spectrogram.json")
    def spectrogram_json(job_id: str, max_width: int = 800, max_height: int = 380) -> JSONResponse:
        """Downsampled dB matrix for browser plotting (§7).

        Max-pooled rather than averaged: a carrier one bin wide survives max pooling and
        vanishes under a mean, and the point of the picture is the signals we found.
        """
        job = _require_done(_require(store.get(job_id), job_id))
        analysis = _analysis_of(job)
        spec = analysis.spectrogram
        if spec is None:
            raise HTTPException(status_code=404, detail="this capture has no spectrogram")

        s_db = np.asarray(spec.S_db, dtype=np.float32)
        factor_f = max(1, int(np.ceil(s_db.shape[0] / max(max_height, 16))))
        factor_t = max(1, int(np.ceil(s_db.shape[1] / max(max_width, 16))))
        pooled = _max_pool(s_db, factor_f, factor_t)
        finite = pooled[np.isfinite(pooled)]
        # Anchor the low end near the noise floor rather than at the 5th percentile. Most
        # cells in a typical capture *are* noise, so a 5th-percentile floor spends the
        # bottom half of viridis rendering the noise in bright purple and leaves the
        # signals with what is left. Putting the floor at the 60th percentile pushes the
        # noise to the dark end where §7 wants it and gives the signals the colour range.
        vmin = float(np.percentile(finite, 60.0)) if finite.size else -120.0
        vmax = float(np.percentile(finite, 99.8)) if finite.size else 0.0
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max()) if finite.size else (0.0, 1.0)
        return JSONResponse(
            {
                # one decimal is below the visual resolution of any colormap and roughly
                # halves the payload; the PNG endpoint is there for pixel-exact output
                "z": np.round(pooled, 1).tolist(),
                "f_lo": float(spec.f[0]),
                "f_hi": float(spec.f[-1]),
                "t_lo": float(spec.t[0]),
                "t_hi": float(spec.t[-1]),
                "vmin": vmin,
                "vmax": vmax,
                "shape": list(pooled.shape),
                "normalised_frequency": analysis.report.capture.frequencies_are_normalised,
            }
        )

    # ---------------------------------------------------------------- audio
    @app.get("/api/jobs/{job_id}/detections/{index}/audio.wav")
    def detection_audio(job_id: str, index: int) -> Response:
        """Demodulated audio for one detection, where demodulable (§7)."""
        job = _require_done(_require(store.get(job_id), job_id))
        analysis = _analysis_of(job)
        report = analysis.report
        match = [d for d in report.detections if d.id == index]
        if not match:
            raise HTTPException(status_code=404, detail=f"no detection {index}")
        detection = match[0]

        source = job.result.get("source")
        if source is None or not Path(source).is_file():
            raise HTTPException(
                status_code=409, detail="the source capture is no longer available"
            )
        iq, meta = read_capture(source)
        half = (detection.bandwidth_hz.occupied_99 or 0.0) / 2.0
        centre = detection.center_freq_offset_hz or 0.0
        burst = Burst(
            t0=detection.time_start_s,
            t1=detection.time_stop_s,
            f_lo=centre - half,
            f_hi=centre + half,
        )
        try:
            isolated = isolate_burst(iq, meta.sample_rate, burst)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result = demodulate(
            isolated.y,
            isolated.fs_b,
            label=detection.modulation.label if detection.modulation else None,
            am_depth=detection.extra.get("am_depth"),
            is_morse=detection.extra.get("morse_dash_dot_ratio") is not None,
        )
        if not result.ok:
            raise HTTPException(status_code=422, detail=result.reason or "not demodulable")
        return Response(
            result.to_wav_bytes(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": (
                    f'inline; filename="detection-{index}-{result.mode}.wav"'
                ),
                "X-Sigscope-Demod-Mode": result.mode,
            },
        )

    @app.get("/api/jobs/{job_id}/detections/{index}/iq.json")
    def detection_iq(job_id: str, index: int, max_samples: int = 4096) -> JSONResponse:
        """Decimated IQ for one detection, for the analysis view's small plots.

        Not in §7's endpoint table, but §7's analysis-view layout ends with
        ``[ constellation ] [ spectrum ] [ inst. frequency ]`` and all three are views of
        the same isolated burst. One endpoint serving all three is the smallest addition
        that makes the specified screen buildable; the alternative is shipping three
        buttons that do nothing.
        """
        job = _require_done(_require(store.get(job_id), job_id))
        analysis = _analysis_of(job)
        match = [d for d in analysis.report.detections if d.id == index]
        if not match:
            raise HTTPException(status_code=404, detail=f"no detection {index}")
        detection = match[0]

        source = job.result.get("source")
        if source is None or not Path(source).is_file():
            raise HTTPException(status_code=409, detail="the source capture is no longer available")
        iq, meta = read_capture(source)
        half = (detection.bandwidth_hz.occupied_99 or 0.0) / 2.0
        centre = detection.center_freq_offset_hz or 0.0
        try:
            isolated = isolate_burst(
                iq,
                meta.sample_rate,
                Burst(detection.time_start_s, detection.time_stop_s,
                      centre - half, centre + half),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        y = isolated.y

        # The spectrum and the instantaneous frequency want the waveform as it is, so they
        # are computed on a plainly decimated copy.
        step = max(1, y.size // max(max_samples, 64))
        wave = y[::step][:max_samples]
        wave = wave / (np.sqrt(float(np.mean(np.abs(wave) ** 2))) or 1.0)
        f_inst = np.angle(wave[1:] * np.conj(wave[:-1])) * isolated.fs_b / (2 * np.pi)
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(wave)) ** 2)
        spectrum_db = 10.0 * np.log10(spectrum + 1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(wave.size, 1.0 / isolated.fs_b))

        # The constellation does not. A constellation is a property of the symbol-rate
        # samples, so decimating the pulse-shaped waveform by an arbitrary step smears
        # every symbol across all phases and QPSK comes out as a disc. Matched-filtering
        # and sampling at the recovered timing phase is what makes §8 Phase 1's "QPSK must
        # show four dots" true on screen -- the same correction §5.2's cumulants needed.
        constellation = wave
        symbol_sampled = False
        rate = detection.symbol_rate_hz
        if rate is not None and rate.value:
            sps = isolated.fs_b / float(rate.value)
            if 2.0 <= sps <= 64.0:
                from sigscope.features.cumulants import symbol_sample

                stream, _ = symbol_sample(y, int(round(sps)))
                if stream.size >= 16:
                    stream = stream[:max_samples]
                    power = float(np.mean(np.abs(stream) ** 2)) or 1.0
                    constellation = stream / np.sqrt(power)
                    symbol_sampled = True

        return JSONResponse(
            {
                "i": np.round(constellation.real, 4).tolist(),
                "q": np.round(constellation.imag, 4).tolist(),
                "symbol_sampled": symbol_sampled,
                "f_inst": np.round(f_inst, 2).tolist(),
                "spectrum_db": np.round(spectrum_db, 2).tolist(),
                "spectrum_hz": np.round(freqs, 2).tolist(),
                "fs_b": isolated.fs_b,
                "decimation": isolated.decimation,
                "n_samples": int(wave.size),
            }
        )

    # ---------------------------------------------------------------- sigmf
    @app.get("/api/jobs/{job_id}/sigmf")
    def job_sigmf(job_id: str) -> Response:
        """SigMF ``.sigmf-meta`` download (§7, §10 "SigMF export")."""
        job = _require_done(_require(store.get(job_id), job_id))
        meta = build_sigmf_meta(_analysis_of(job).report)
        stem = Path(job.name).stem or "capture"
        return Response(
            json.dumps(meta, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{stem}.sigmf-meta"'},
        )

    # ---------------------------------------------------------------- static
    if web_root.is_dir():
        app.mount("/", StaticFiles(directory=str(web_root), html=True), name="web")
    else:  # pragma: no cover -- only when the checkout is incomplete
        @app.get("/")
        def missing_web() -> HTMLResponse:
            return HTMLResponse(
                f"<h1>SIGSCOPE</h1><p>The dashboard is missing from {web_root}.</p>",
                status_code=500,
            )

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> None:
    """Run the app with uvicorn. Bound to localhost -- see the AUTH note in ``create_app``."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, reload=reload, log_level="info")
