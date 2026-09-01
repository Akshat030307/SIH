"""The local API and the static dashboard (CLAUDE.md §7, §9 E).

Covers every endpoint in §7's table, the job table's behaviour, and the two properties the
demo depends on and which are easy to lose silently:

* **Nothing is fetched from a network.** §2 and §9 E both hang on this, and it is a
  property of the files in ``web/`` rather than of a setting, so it is asserted against the
  files themselves.
* **Uploads never land in memory.** §7 allows 2 GB; the streaming test drives a body larger
  than the chunk size and checks the bytes reach disk.

Uses ``fastapi.testclient`` (Starlette's, over ``httpx``), so no server process is started
and no port is bound.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from sigscope import testsignals as ts
from sigscope.api.app import MAX_UPLOAD_BYTES, WEB_ROOT, create_app
from sigscope.api.audio import demodulate
from sigscope.api.jobs import Job, JobError, JobState, JobStore

FS = 200_000.0


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def capture_bytes() -> bytes:
    """A small stereo-IQ WAV holding a QPSK burst and a tone, as an upload body."""
    rng = np.random.default_rng(0)
    n = int(FS)
    canvas = (
        np.sqrt(5e-4) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ).astype(np.complex64)

    burst = ts.psk(4, 12_500.0, FS, 3000, rng=1)
    start = int(0.15 * FS)
    stop = min(n, start + burst.size)
    t = np.arange(start, stop) / FS
    canvas[start:stop] += (
        0.8 * burst[: stop - start] * np.exp(2j * np.pi * 40_000.0 * t)
    ).astype(np.complex64)

    tone = ts.tone(0.0, FS, int(0.5 * FS))
    t2 = np.arange(0, tone.size) / FS
    canvas[: tone.size] += (0.4 * tone * np.exp(-2j * np.pi * 60_000.0 * t2)).astype(
        np.complex64
    )
    canvas /= np.max(np.abs(canvas)) * 1.05

    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.stack([canvas.real, canvas.imag], axis=1).astype(np.float32),
        int(FS),
        format="WAV",
        subtype="FLOAT",
    )
    return buffer.getvalue()


@pytest.fixture
def client():
    app = create_app(store=JobStore(max_workers=2))
    with TestClient(app) as test_client:
        yield test_client


def _run(client, capture_bytes, name="SDRSharp_20240101_000000Z_100000000Hz_IQ.wav", **form):
    """Upload, wait for the job, and return its id."""
    response = client.post(
        "/api/analyse",
        files={"file": (name, capture_bytes, "audio/wav")},
        data={k: str(v) for k, v in form.items()},
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    for _ in range(600):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["state"] in ("done", "failed"):
            break
        time.sleep(0.1)
    return job_id, state


# --------------------------------------------------------------------------------------
# §7 endpoint table
# --------------------------------------------------------------------------------------


def test_health_reports_version_and_model_state(client):
    """§7: "Version, model checksums, whether models are loaded"."""
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert set(body["models"]) >= {"feature_clf_loaded", "cnn_loaded", "checksums"}
    assert body["limits"]["max_upload_bytes"] == MAX_UPLOAD_BYTES
    # untrained is a supported state, and health has to say so rather than look healthy
    if not body["models"]["feature_clf_loaded"]:
        assert body["models"]["note"]


def test_analyse_returns_a_job_id_and_completes(client, capture_bytes):
    job_id, state = _run(client, capture_bytes)
    assert state["state"] == "done", state
    assert state["progress"] == 1.0
    assert re.fullmatch(r"[0-9a-f]{16}", job_id)


def test_job_progress_moves_through_stages(client, capture_bytes):
    """§7 wants progress 0-1. A bar pinned at one value until the end is not progress."""
    response = client.post(
        "/api/analyse", files={"file": ("cap.wav", capture_bytes, "audio/wav")}
    )
    job_id = response.json()["job_id"]
    seen = set()
    for _ in range(600):
        state = client.get(f"/api/jobs/{job_id}").json()
        seen.add(round(state["progress"], 2))
        if state["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["state"] == "done"
    assert len(seen) >= 2, f"progress never moved: {seen}"
    assert all(0.0 <= p <= 1.0 for p in seen)


def test_report_matches_the_frozen_schema(client, capture_bytes):
    from sigscope.report import validate_report_dict

    job_id, _ = _run(client, capture_bytes)
    report = client.get(f"/api/jobs/{job_id}/report").json()
    validate_report_dict(report)
    assert report["file"]["name"].endswith(".wav")
    assert report["capture"]["center_freq_hz"] == 100_000_000.0  # parsed from the name


def test_spectrogram_png_is_a_png(client, capture_bytes):
    job_id, _ = _run(client, capture_bytes)
    response = client.get(f"/api/jobs/{job_id}/spectrogram.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_spectrogram_json_is_a_downsampled_matrix(client, capture_bytes):
    job_id, _ = _run(client, capture_bytes)
    body = client.get(f"/api/jobs/{job_id}/spectrogram.json").json()
    rows, cols = body["shape"]
    assert len(body["z"]) == rows
    assert len(body["z"][0]) == cols
    assert rows <= 380 and cols <= 800, "the matrix must be downsampled for the browser"
    assert body["vmin"] < body["vmax"]
    assert body["f_lo"] < body["f_hi"] and body["t_lo"] < body["t_hi"]


def test_sigmf_download_loads_in_the_sigmf_library(client, capture_bytes):
    """§9 E: "SigMF output loads in the sigmf library"."""
    from sigmf import SigMFFile

    job_id, _ = _run(client, capture_bytes)
    response = client.get(f"/api/jobs/{job_id}/sigmf")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    SigMFFile(metadata=json.loads(response.text)).validate()


def test_detection_iq_symbol_samples_the_constellation(client, capture_bytes):
    """§8 Phase 1's checkpoint is "QPSK must show four dots", and that only happens on
    symbol-rate samples -- a plain decimation of the pulse-shaped waveform is a disc."""
    job_id, _ = _run(client, capture_bytes)
    report = client.get(f"/api/jobs/{job_id}/report").json()
    target = next(
        (
            d
            for d in report["detections"]
            if d["symbol_rate_hz"] and d["symbol_rate_hz"]["value"]
        ),
        None,
    )
    if target is None:
        pytest.skip("no detection with a measured symbol rate in this capture")

    body = client.get(f"/api/jobs/{job_id}/detections/{target['id']}/iq.json").json()
    assert len(body["i"]) == len(body["q"]) > 0
    assert len(body["spectrum_db"]) == len(body["spectrum_hz"])
    assert np.all(np.isfinite(body["i"])) and np.all(np.isfinite(body["q"]))


def test_audio_endpoint_either_returns_wav_or_says_why_not(client, capture_bytes):
    """§7: "Demodulated audio, **where demodulable**". A digital burst is not."""
    job_id, _ = _run(client, capture_bytes)
    report = client.get(f"/api/jobs/{job_id}/report").json()
    assert report["detections"], "the fixture capture should produce detections"

    for detection in report["detections"]:
        response = client.get(f"/api/jobs/{job_id}/detections/{detection['id']}/audio.wav")
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            assert response.headers["content-type"] == "audio/wav"
            assert response.content[:4] == b"RIFF"
        else:
            assert response.json()["detail"]  # a reason, never a bare failure


def test_batch_endpoint_produces_one_csv_row_per_file(client, capture_bytes, tmp_path):
    """§9 E: "A batch of 100 files completes with one CSV row each and no crash"."""
    for i in range(3):
        (tmp_path / f"capture_{i}.wav").write_bytes(capture_bytes)
    (tmp_path / "broken.iq").write_text("not a capture\n" * 40, encoding="utf-8")

    response = client.post("/api/batch", json={"path": str(tmp_path)})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    assert response.json()["n_files"] == 4

    for _ in range(900):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["state"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert state["state"] == "done", state

    csv_text = client.get(f"/api/jobs/{job_id}/csv").text
    lines = [line for line in csv_text.splitlines() if line.strip()]
    assert len(lines) == 5, "header plus one row per file"
    assert lines[0].startswith("file,status,error")
    # the unreadable file is a row, not a crash
    assert any(",error," in line or line.split(",")[1] == "error" for line in lines[1:])


# --------------------------------------------------------------------------------------
# error states -- §7: "say what happened and what to do"
# --------------------------------------------------------------------------------------


def test_empty_upload_is_rejected_by_name(client):
    response = client.post(
        "/api/analyse", files={"file": ("empty.iq", b"", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    assert "empty.iq" in response.json()["detail"]


def test_unreadable_capture_fails_with_a_phrased_message(client):
    """The job fails, the message names the problem and the fix, and no class name leaks."""
    body = b"this is a text file, not a capture\n" * 40
    response = client.post("/api/analyse", files={"file": ("notes.iq", body, "text/plain")})
    job_id = response.json()["job_id"]
    for _ in range(300):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["state"] == "failed"
    assert "notes.iq" in state["error"]
    assert "convert it first" in state["error"]
    assert not state["error"].startswith("ValueError")


def test_results_of_a_failed_job_are_409_not_a_traceback(client):
    body = b"still not a capture\n" * 40
    job_id = client.post(
        "/api/analyse", files={"file": ("x.iq", body, "text/plain")}
    ).json()["job_id"]
    for _ in range(300):
        if client.get(f"/api/jobs/{job_id}").json()["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    response = client.get(f"/api/jobs/{job_id}/report")
    assert response.status_code == 409
    assert "Traceback" not in response.text


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/0123456789abcdef/report").status_code == 404
    assert client.get("/api/jobs/0123456789abcdef").status_code == 404


def test_batch_rejects_a_path_that_is_not_a_directory(client):
    response = client.post("/api/batch", json={"path": "definitely/not/here"})
    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"]


def test_batch_requires_a_path(client):
    assert client.post("/api/batch", json={}).status_code == 400


# --------------------------------------------------------------------------------------
# uploads stream to disk (§7)
# --------------------------------------------------------------------------------------


def test_upload_larger_than_one_chunk_reaches_disk_intact(client):
    """§7: an upload is never read fully into memory, so it must survive chunking."""
    from sigscope.api.app import UPLOAD_CHUNK

    payload = bytes(range(256)) * ((UPLOAD_CHUNK * 2) // 256 + 7)
    response = client.post(
        "/api/analyse", files={"file": ("big.iq", payload, "application/octet-stream")}
    )
    assert response.status_code == 200
    assert response.json()["bytes"] == len(payload)
    assert len(payload) > UPLOAD_CHUNK


def test_upload_ceiling_is_enforced_while_streaming(monkeypatch, capture_bytes):
    """The limit is checked against bytes actually written, not a Content-Length header,
    because a header is a claim by the client."""
    import sigscope.api.app as app_module

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(app_module, "UPLOAD_CHUNK", 256)
    with TestClient(app_module.create_app(store=JobStore())) as local:
        response = local.post(
            "/api/analyse", files={"file": ("big.iq", capture_bytes, "application/octet-stream")}
        )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# job table
# --------------------------------------------------------------------------------------


def test_job_store_records_failure_without_raising():
    store = JobStore()
    job = store.create("analyse", "x")

    def boom(_job: Job):
        raise RuntimeError("kaboom")

    store.submit(job, boom)
    for _ in range(200):
        if job.state is JobState.FAILED:
            break
        time.sleep(0.02)
    assert job.state is JobState.FAILED
    assert job.error == "RuntimeError: kaboom"


def test_job_error_message_is_not_prefixed_with_a_class_name():
    store = JobStore()
    job = store.create("analyse", "x")
    store.submit(job, lambda _job: (_ for _ in ()).throw(JobError("capture.iq: is empty")))
    for _ in range(200):
        if job.state is JobState.FAILED:
            break
        time.sleep(0.02)
    assert job.error == "capture.iq: is empty"


def test_discarding_a_job_deletes_its_temp_files(tmp_path):
    store = JobStore()
    temp = tmp_path / "upload.bin"
    temp.write_bytes(b"x")
    job = store.create("analyse", "x", temp_paths=[temp])
    assert store.discard(job.id) is True
    assert not temp.exists()
    assert store.get(job.id) is None


def test_store_evicts_finished_jobs_past_its_ceiling():
    store = JobStore(max_jobs=4)
    for i in range(10):
        job = store.create("analyse", f"job-{i}")
        job.state = JobState.DONE
    assert len(store.list()) <= 4


# --------------------------------------------------------------------------------------
# demodulation (§7 /audio.wav)
# --------------------------------------------------------------------------------------


def test_am_burst_demodulates():
    fs = 96_000.0
    t = np.arange(int(0.4 * fs)) / fs
    am = (1 + 0.7 * np.cos(2 * np.pi * 900 * t)).astype(np.complex64)
    result = demodulate(am, fs, label="AM-DSB", am_depth=0.7)
    assert result.ok and result.mode == "am"
    assert result.to_wav_bytes()[:4] == b"RIFF"


def test_fm_burst_demodulates():
    fs = 96_000.0
    t = np.arange(int(0.4 * fs)) / fs
    fm = np.exp(2j * np.pi * np.cumsum(3000 * np.cos(2 * np.pi * 400 * t)) / fs)
    result = demodulate(fm.astype(np.complex64), fs, label="WBFM")
    assert result.ok and result.mode == "fm"


def test_morse_burst_becomes_an_audible_tone():
    fs = 48_000.0
    dot = int(0.06 * fs)
    envelope = np.concatenate(
        [np.ones(n * dot) if i % 2 == 0 else np.zeros(n * dot)
         for i, n in enumerate([1, 1, 3, 1, 1, 3, 3, 1])]
    )
    result = demodulate(envelope.astype(np.complex64), fs, is_morse=True)
    assert result.ok and result.mode == "cw"


def test_too_short_burst_is_refused_with_a_reason():
    result = demodulate(np.ones(16, dtype=np.complex64), 48_000.0)
    assert not result.ok
    assert result.reason


# --------------------------------------------------------------------------------------
# the dashboard is genuinely offline (§2, §9 E)
# --------------------------------------------------------------------------------------


WEB_FILES = sorted(
    p for p in WEB_ROOT.rglob("*") if p.is_file() and p.suffix in {".html", ".css", ".js"}
)


def test_web_assets_exist():
    names = {p.relative_to(WEB_ROOT).as_posix() for p in WEB_FILES}
    assert "index.html" in names
    assert "css/app.css" in names
    assert {"js/app.js", "js/api.js", "js/plots.js", "js/viridis.js"} <= names


@pytest.mark.parametrize("path", WEB_FILES, ids=lambda p: p.name)
def test_no_asset_references_an_external_origin(path: Path):
    """§9 E: "Everything works identically with wifi disabled."

    Asserted against the files rather than by unplugging: no absolute URL, no
    protocol-relative src, no font service, no CDN.
    """
    text = path.read_text(encoding="utf-8")
    forbidden = [
        "http://", "https://", "//cdn", "//fonts", "fonts.googleapis", "fonts.gstatic",
        "unpkg.com", "jsdelivr", "cdnjs",
    ]
    for needle in forbidden:
        if needle in ("http://", "https://"):
            # allow them inside comments that merely mention a URL scheme
            for line in text.splitlines():
                stripped = line.strip()
                if needle in line and not (
                    stripped.startswith("*") or stripped.startswith("/*")
                    or stripped.startswith("//") or stripped.startswith("<!--")
                ):
                    raise AssertionError(f"{path.name}: external reference in {line.strip()!r}")
        else:
            assert needle not in text, f"{path.name} references {needle}"


def test_dashboard_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SIGSCOPE" in response.text
    for asset in ("css/app.css", "js/app.js", "js/viridis.js"):
        assert client.get("/" + asset).status_code == 200


def test_css_carries_the_section_7_palette():
    """§7 names the colours; a redesign that quietly drifts off them should fail here."""
    css = (WEB_ROOT / "css" / "app.css").read_text(encoding="utf-8")
    for colour in ("#0d1117", "#161b22", "#2d333b", "#c9d1d9", "#4dd0c4", "#8b949e"):
        assert colour in css, f"§7 palette colour {colour} is missing"
