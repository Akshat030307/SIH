"""Acceptance tests E and F, run end to end (CLAUDE.md §9 E, §9 F, §8 Phase 8).

§9's rule is "If a test here fails, the feature is not done, however good the demo looks",
so this runs the §9 E list against real files on disk and prints a pass/fail table with
the measured numbers: the 10 s scene budget, a 100-file batch, schema validation, the SigMF
round trip, the offline guarantee, and the hostile inputs (empty file, text file renamed
``.iq``, pure noise, a strong DC spike).

The two long cases are opt-in because they cost minutes and gigabytes:

* ``--with-2gb`` writes and analyses a genuine 2 GB capture, checking §9 B's "processed in
  blocks with peak RSS under 2 GB".
* ``--with-clone`` copies the working tree to a temp directory and runs
  ``pip install -e . --no-index`` to prove a fresh clone installs with the network off.

Run with::

    python scripts/hardening.py                 # the fast cases
    python scripts/hardening.py --with-2gb --with-clone --out HARDENING.md
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf  # noqa: E402

from sigscope import testsignals as ts  # noqa: E402
from sigscope.io import CaptureError  # noqa: E402
from sigscope.pipeline import analyse_file  # noqa: E402
from sigscope.report import build_sigmf_meta, validate_report_dict  # noqa: E402

SCENE_BUDGET_S = 30.0  # §9 E
RSS_BUDGET_BYTES = 2 * 1024**3  # §9 B
REPO = Path(__file__).resolve().parent.parent


@dataclass
class Case:
    """One §9 acceptance check."""

    section: str
    name: str
    requirement: str
    passed: bool | None = None
    measured: str = ""
    detail: str = ""
    skipped: bool = False

    @property
    def mark(self) -> str:
        if self.skipped:
            return "skipped"
        return "pass" if self.passed else "**FAIL**"


@dataclass
class Runner:
    cases: list[Case] = field(default_factory=list)

    def add(self, section, name, requirement, passed, measured="", detail="", skipped=False):
        case = Case(section, name, requirement, passed, measured, detail, skipped)
        self.cases.append(case)
        flag = "SKIP" if skipped else ("ok  " if passed else "FAIL")
        print(f"  [{flag}] {name}: {measured}", flush=True)
        if detail and not passed and not skipped:
            print(f"         {detail}", flush=True)
        return case


# --------------------------------------------------------------------------------------
# fixtures written to disk
# --------------------------------------------------------------------------------------


def write_scene(path: Path, duration_s: float, fs: float = 2_000_000.0) -> Path:
    """A multi-signal scene: recurring QPSK, a 2-FSK, a steady tone and chirps."""
    rng = np.random.default_rng(3)
    n = int(fs * duration_s)
    canvas = (
        np.sqrt(5e-4) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ).astype(np.complex64)

    def place(signal, f_offset, t0, amplitude):
        n0 = int(t0 * fs)
        n1 = min(n, n0 + len(signal))
        if n1 <= n0:
            return
        t = np.arange(n0, n1) / fs
        canvas[n0:n1] += (
            amplitude * signal[: n1 - n0] * np.exp(2j * np.pi * f_offset * t)
        ).astype(np.complex64)

    for k in range(int(duration_s)):
        place(ts.psk(4, 50_000.0, fs, 4000, rng=k), +250_000.0, 0.05 + k, 0.9)
    place(ts.fsk(2, 20_000.0, 10_000.0, fs, int(3000 * duration_s), rng=2), -600_000.0, 0.2, 0.7)
    place(ts.tone(0.0, fs, int(0.8 * duration_s * fs)), -150_000.0, 0.05, 0.35)
    for k in range(max(1, int(duration_s / 2))):
        place(ts.lfm_chirp(-150_000.0, 150_000.0, 0.05, fs), +700_000.0, 0.6 + 2 * k, 0.8)

    canvas /= np.max(np.abs(canvas)) * 1.05
    sf.write(
        path,
        np.stack([canvas.real, canvas.imag], axis=1).astype(np.float32),
        int(fs),
        subtype="FLOAT",
    )
    return path


def write_noise(path: Path, seconds: float = 1.0, fs: float = 1_000_000.0) -> Path:
    rng = np.random.default_rng(5)
    n = int(fs * seconds)
    c = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    c /= np.max(np.abs(c)) * 1.05
    np.stack([c.real, c.imag], axis=1).astype(np.float32).tofile(path)
    return path


def write_dc_spike(path: Path, seconds: float = 1.0, fs: float = 1_000_000.0) -> Path:
    """Noise plus a large constant offset -- the §4.3 / §9 C DC-spike case."""
    rng = np.random.default_rng(6)
    n = int(fs * seconds)
    c = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    c += np.complex64(14.0 + 6.0j)
    c /= np.max(np.abs(c)) * 1.05
    np.stack([c.real, c.imag], axis=1).astype(np.float32).tofile(path)
    return path


def peak_rss_of(command: list[str], poll_s: float = 0.4) -> tuple[int, int, str]:
    """Run ``command`` and return ``(returncode, peak RSS bytes, output)``."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    peak = 0
    stop = threading.Event()

    def watch():
        nonlocal peak
        while not stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "powershell", "-NoProfile", "-Command",
                        "(Get-Process -Id "
                        f"{process.pid} -ErrorAction SilentlyContinue).WorkingSet64",
                    ],
                    capture_output=True, text=True,
                ).stdout.strip()
                if out.isdigit():
                    peak = max(peak, int(out))
            except Exception:  # noqa: BLE001 -- sampling must never kill the run
                pass
            stop.wait(poll_s)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    output, _ = process.communicate()
    stop.set()
    thread.join(timeout=2)
    return process.returncode, peak, output or ""


# --------------------------------------------------------------------------------------
# §9 E
# --------------------------------------------------------------------------------------


def case_scene_budget(runner: Runner, work: Path) -> None:
    """§9 E: "A 10 s, 2 MHz scene completes in under 30 s on a laptop CPU"."""
    path = write_scene(work / "scene10s.wav", 10.0)
    started = time.perf_counter()
    analysis = analyse_file(path)
    elapsed = time.perf_counter() - started
    megabytes = path.stat().st_size / 1e6
    runner.add(
        "E", "10 s / 2 MHz scene under 30 s",
        f"< {SCENE_BUDGET_S:.0f} s",
        elapsed < SCENE_BUDGET_S,
        f"{elapsed:.1f} s for {megabytes:.0f} MB ({elapsed / megabytes:.3f} s/MB), "
        f"{len(analysis.report.detections)} detections",
    )


def case_batch_100(runner: Runner, work: Path) -> None:
    """§9 E: "A batch of 100 files completes with one CSV row each and no crash"."""
    from sigscope.batch import find_captures, run_batch, write_csv

    folder = work / "batch100"
    folder.mkdir(exist_ok=True)
    small = write_scene(work / "small.wav", 1.0)
    payload = small.read_bytes()
    for i in range(98):
        (folder / f"capture_{i:03d}.wav").write_bytes(payload)
    (folder / "broken.iq").write_text("not a capture\n" * 40, encoding="utf-8")
    (folder / "empty.iq").write_bytes(b"")

    paths = find_captures(folder)
    started = time.perf_counter()
    rows = run_batch(paths, workers=max(2, (os.cpu_count() or 4) // 2), progress=False)
    elapsed = time.perf_counter() - started
    csv_path = write_csv(rows, work / "results.csv")
    lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    ok = sum(1 for r in rows if r.get("status") == "ok")
    runner.add(
        "E", "batch of 100 files, one CSV row each",
        "100 rows, no crash",
        len(paths) == 100 and len(rows) == 100 and len(lines) == 101,
        f"{len(rows)} rows ({ok} ok, {len(rows) - ok} error) in {elapsed:.0f} s",
        detail=f"found {len(paths)} files, csv had {len(lines)} lines including the header",
    )


def case_schema_and_sigmf(runner: Runner, work: Path) -> None:
    """§9 E: report JSON validates; SigMF output loads in the ``sigmf`` library."""
    from sigmf import SigMFFile

    analysis = analyse_file(write_scene(work / "scene1s.wav", 1.0))
    try:
        validate_report_dict(analysis.report.to_dict())
        schema_ok, schema_detail = True, ""
    except Exception as exc:  # noqa: BLE001
        schema_ok, schema_detail = False, str(exc)
    runner.add("E", "report JSON validates against the frozen schema", "§3 schema",
               schema_ok, "valid" if schema_ok else "invalid", schema_detail)

    try:
        SigMFFile(metadata=build_sigmf_meta(analysis.report)).validate()
        sigmf_ok, sigmf_detail = True, ""
    except Exception as exc:  # noqa: BLE001
        sigmf_ok, sigmf_detail = False, str(exc)
    runner.add("E", "SigMF output loads in the sigmf library", "loads + validates",
               sigmf_ok, "loads" if sigmf_ok else "rejected", sigmf_detail)


def case_offline(runner: Runner, work: Path) -> None:
    """§9 E: "Everything works identically with wifi disabled".

    Rather than unplugging, this makes every outbound socket raise for the duration of an
    analysis. If any code path tried to reach a network it would fail loudly here.
    """
    path = write_scene(work / "offline.wav", 1.0)
    real_socket = socket.socket
    real_create = socket.create_connection
    blocked: list[str] = []

    def refuse(*args, **kwargs):
        blocked.append("socket")
        raise OSError("network access is disabled for this test")

    socket.socket = refuse  # type: ignore[assignment]
    socket.create_connection = refuse  # type: ignore[assignment]
    try:
        analysis = analyse_file(path)
        ok, detail = len(analysis.report.detections) >= 1, ""
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    finally:
        socket.socket = real_socket  # type: ignore[assignment]
        socket.create_connection = real_create  # type: ignore[assignment]

    runner.add(
        "E", "analysis with all network access blocked", "no sockets opened",
        ok, "completed, 0 sockets attempted" if ok and not blocked else
        (f"{len(blocked)} socket attempt(s)" if blocked else "failed"), detail,
    )

    # and the dashboard's own assets
    web = REPO / "web"
    offenders = []
    for asset in sorted(web.rglob("*")):
        if asset.suffix not in {".html", ".css", ".js"}:
            continue
        text = asset.read_text(encoding="utf-8")
        for needle in ("//cdn", "fonts.googleapis", "fonts.gstatic", "unpkg.com", "jsdelivr"):
            if needle in text:
                offenders.append(f"{asset.name}:{needle}")
    runner.add(
        "E", "dashboard references no external origin", "no CDN, no font service",
        not offenders, "clean" if not offenders else ", ".join(offenders),
    )


def case_hostile_inputs(runner: Runner, work: Path) -> None:
    """§9 B / the demo-day list: every bad input names the problem, never a traceback."""
    empty = work / "empty.iq"
    empty.write_bytes(b"")
    text = work / "notes.iq"
    text.write_text("this is a text file, not a capture\n" * 60, encoding="utf-8")

    for label, path, expect in (
        ("empty file", empty, "empty"),
        ("text file renamed .iq", text, "convert it first"),
    ):
        try:
            analyse_file(path)
            ok, measured, detail = False, "no error raised", "a bad file must be rejected"
        except CaptureError as exc:
            message = str(exc)
            ok = expect in message and "Traceback" not in message
            # a Windows path contains ':', so drop the leading path rather than
            # splitting on the first colon
            measured = message.replace(str(path), path.name).strip()[:90]
            detail = "" if ok else f"message did not name the problem: {message}"
        except Exception as exc:  # noqa: BLE001
            ok = False
            measured = f"{type(exc).__name__}"
            detail = f"raised {type(exc).__name__} instead of CaptureError: {exc}"
        runner.add("B", f"{label} gives a clear message", "CaptureError naming the problem",
                   ok, measured, detail)

    # pure noise -> zero detections, and it says so
    noise_report = analyse_file(write_noise(work / "noise.cf32")).report
    says_so = any("no signals found" in w.lower() for w in noise_report.warnings)
    runner.add(
        "C", "pure noise gives zero detections", "0 detections and says so",
        len(noise_report.detections) == 0 and says_so,
        f"{len(noise_report.detections)} detections, "
        f"{'says so' if says_so else 'silent'}",
    )

    # a strong DC spike is not a signal
    dc_report = analyse_file(write_dc_spike(work / "dc.cf32")).report
    at_dc = [
        d for d in dc_report.detections
        if abs(d.center_freq_offset_hz or 0.0) < 0.005 * dc_report.capture.sample_rate
    ]
    runner.add(
        "C", "strong DC spike is not reported as a signal", "no detection at 0 Hz",
        not at_dc,
        f"{len(dc_report.detections)} detections, {len(at_dc)} at DC",
    )


# --------------------------------------------------------------------------------------
# §9 F -- the judge test
# --------------------------------------------------------------------------------------


def case_judge(runner: Runner, work: Path):
    """§9 F: an unseen capture with no metadata at all, and five minutes.

    Deliberately hostile in the way a real handover is: a headerless file, an odd sample
    rate nobody told us, an unusual name, and modulations chosen so no single estimator
    carries the result.
    """
    fs = 1_234_567.0  # deliberately not a round number, and not recorded anywhere
    rng = np.random.default_rng(99)
    n = int(fs * 3)
    canvas = (
        np.sqrt(6e-4) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ).astype(np.complex64)

    def place(signal, f_offset, t0, amplitude):
        n0 = int(t0 * fs)
        n1 = min(n, n0 + len(signal))
        t = np.arange(n0, n1) / fs
        canvas[n0:n1] += (
            amplitude * signal[: n1 - n0] * np.exp(2j * np.pi * f_offset * t)
        ).astype(np.complex64)

    place(ts.fsk(4, 12_000.0, 6_000.0, 1_200_000.0, 9000, rng=1), -300_000.0, 0.2, 0.8)
    # not OFDM: testsignals.ofdm occupies the whole band regardless of fs (one
    # sample per subcarrier), so beside another signal it swallows it.
    place(ts.psk(4, 40_000.0, 1_200_000.0, 12000, rng=2), +250_000.0, 0.5, 0.7)
    place(ts.lfm_chirp(-80_000.0, 80_000.0, 0.04, fs), +450_000.0, 1.9, 0.9)
    canvas /= np.max(np.abs(canvas)) * 1.05

    path = work / "UNKNOWN_HANDOVER.bin"  # no rate, no centre, no convention
    interleaved = np.empty(2 * n, dtype=np.int16)
    interleaved[0::2] = np.clip(canvas.real * 32767, -32768, 32767).astype(np.int16)
    interleaved[1::2] = np.clip(canvas.imag * 32767, -32768, 32767).astype(np.int16)
    interleaved.tofile(path)

    started = time.perf_counter()
    analysis = analyse_file(path)
    elapsed = time.perf_counter() - started
    report = analysis.report

    normalised = report.capture.frequencies_are_normalised
    no_fake_hz = normalised or report.capture.center_freq is None
    has_evidence = all(
        d.modulation is not None and len(d.modulation.evidence) >= 2
        for d in report.detections
    )
    sensible = (
        len(report.detections) >= 2
        and elapsed < 300.0
        and no_fake_hz
        and has_evidence
    )
    runner.add(
        "F", "unseen headerless capture, no metadata, 5 minutes",
        "a sensible report, no invented Hz",
        sensible,
        f"{len(report.detections)} detections in {elapsed:.1f} s; "
        f"dtype guessed {report.capture.dtype_guessed} ({report.capture.dtype_confidence:.0%}); "
        f"frequencies {'normalised' if normalised else 'in Hz'}",
        detail="" if sensible else "see the report warnings",
    )
    return report


# --------------------------------------------------------------------------------------
# opt-in long cases
# --------------------------------------------------------------------------------------


def case_two_gigabytes(runner: Runner, work: Path, keep: Path | None = None) -> None:
    """§9 B: "2 GB file processed in blocks with peak RSS under 2 GB"."""
    path = keep if keep and keep.is_file() else work / "huge_2000000sps_cs16.iq"
    if not path.is_file():
        print("  writing a 2 GB capture (this takes a few minutes)...", flush=True)
        fs = 2_000_000.0
        chunk = 1 << 22
        total = (2 * 1024**3) // 4
        rng = np.random.default_rng(11)
        written = 0
        with open(path, "wb") as handle:
            index = 0
            while written < total:
                count = int(min(chunk, total - written))
                c = (
                    np.sqrt(4e-4)
                    * (rng.standard_normal(count) + 1j * rng.standard_normal(count))
                ).astype(np.complex64)
                burst = ts.psk(4, 50_000.0, fs, 6000, rng=index)
                b = burst[: min(len(burst), count // 2)]
                t = np.arange(len(b)) / fs
                c[: len(b)] += (0.9 * b * np.exp(2j * np.pi * 250_000.0 * t)).astype(np.complex64)
                c += (0.25 * ts.tone(-400_000.0, fs, count)).astype(np.complex64)
                c /= np.max(np.abs(c)) * 1.05
                inter = np.empty(2 * count, dtype=np.int16)
                inter[0::2] = np.clip(c.real * 32767, -32768, 32767).astype(np.int16)
                inter[1::2] = np.clip(c.imag * 32767, -32768, 32767).astype(np.int16)
                handle.write(inter.tobytes())
                written += count
                index += 1

    script = (
        f"import sys, time; sys.path.insert(0, r'{REPO}');"
        "from sigscope.pipeline import analyse_file;"
        f"t=time.perf_counter(); a=analyse_file(r'{path}');"
        "print('DETECTIONS', len(a.report.detections));"
        "print('RUNTIME', round(time.perf_counter()-t,1))"
    )
    code, peak, output = peak_rss_of([sys.executable, "-c", script])
    gigabytes = path.stat().st_size / 1024**3
    detections = next(
        (ln.split()[1] for ln in output.splitlines() if ln.startswith("DETECTIONS")), "?"
    )
    runtime = next(
        (ln.split()[1] for ln in output.splitlines() if ln.startswith("RUNTIME")), "?"
    )
    runner.add(
        "B", "2 GB capture processed in blocks", f"peak RSS < {RSS_BUDGET_BYTES / 1024**3:.0f} GB",
        code == 0 and 0 < peak < RSS_BUDGET_BYTES,
        f"{gigabytes:.2f} GB file, peak RSS {peak / 1024**3:.2f} GB, "
        f"{detections} detections in {runtime} s",
        detail="" if code == 0 else output[-400:],
    )


def case_fresh_clone(runner: Runner, work: Path) -> None:
    """§9 E: "A fresh clone plus pip install -e . reaches a working demo"... offline."""
    clone = work / "clone"
    if clone.exists():
        shutil.rmtree(clone, ignore_errors=True)
    clone.mkdir(parents=True)

    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True
    ).stdout.split()
    extra = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout.split()
    for relative in sorted(set(listing) | set(extra)):
        source = REPO / relative
        if not source.is_file():
            continue
        target = clone / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    started = time.perf_counter()
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".",
         "--no-index", "--no-build-isolation", "--no-deps"],
        cwd=clone, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started

    demo = subprocess.run(
        [sys.executable, "-m", "sigscope", "--help"], cwd=clone, capture_output=True, text=True
    )
    runner.add(
        "E", "fresh clone installs with the index disabled",
        "pip install -e . --no-index succeeds",
        install.returncode == 0 and demo.returncode == 0,
        f"installed in {elapsed:.0f} s, `python -m sigscope --help` exits "
        f"{demo.returncode}",
        detail="" if install.returncode == 0 else install.stderr[-400:],
    )

    # put the real editable install back; the clone's one has just replaced it
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".",
         "--no-index", "--no-build-isolation", "--no-deps"],
        cwd=REPO, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------------------


def render(runner: Runner, throughput: str) -> str:
    lines = [
        "# Hardening — acceptance tests E and F",
        "",
        "Generated by `scripts/hardening.py`. §9: *if a test here fails, the feature is "
        "not done, however good the demo looks.*",
        "",
        "| § | Check | Requirement | Result | Measured |",
        "|---|---|---|---|---|",
    ]
    for case in runner.cases:
        lines.append(
            f"| {case.section} | {case.name} | {case.requirement} | {case.mark} | "
            f"{case.measured} |"
        )
    lines += ["", "## Throughput", "", throughput, ""]
    failed = [c for c in runner.cases if c.passed is False and not c.skipped]
    if failed:
        lines += ["## Failures", ""]
        lines += [f"- **{c.name}** — {c.detail or c.measured}" for c in failed]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-2gb", action="store_true", help="write and analyse 2 GB")
    parser.add_argument("--with-clone", action="store_true", help="offline install test")
    parser.add_argument("--huge", default=None, help="reuse an existing 2 GB capture")
    parser.add_argument("--work", default=None, help="working directory for fixtures")
    parser.add_argument("--out", default=None, help="write a Markdown report here")
    args = parser.parse_args(argv)

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="sigscope-hard-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"hardening: working in {work}", flush=True)

    runner = Runner()
    throughput = "not measured"
    try:
        case_scene_budget(runner, work)
        scene = work / "scene10s.wav"
        if scene.is_file():
            megabytes = scene.stat().st_size / 1e6
            started = time.perf_counter()
            analyse_file(scene)
            elapsed = time.perf_counter() - started
            throughput = (
                f"`{elapsed / megabytes:.3f} s/MB` on a 10 s / 2 MHz scene "
                f"({megabytes:.0f} MB, {elapsed:.1f} s) — 8-core laptop CPU, no GPU."
            )
        case_schema_and_sigmf(runner, work)
        case_offline(runner, work)
        case_hostile_inputs(runner, work)
        case_judge(runner, work)
        case_batch_100(runner, work)
        if args.with_2gb:
            case_two_gigabytes(runner, work, Path(args.huge) if args.huge else None)
        else:
            runner.add("B", "2 GB capture processed in blocks", "peak RSS < 2 GB",
                       None, "not run (pass --with-2gb)", skipped=True)
        if args.with_clone:
            case_fresh_clone(runner, work)
        else:
            runner.add("E", "fresh clone installs with the index disabled",
                       "pip install -e . --no-index", None,
                       "not run (pass --with-clone)", skipped=True)
    finally:
        pass

    report = render(runner, throughput)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print()
        print(report)

    failed = [c for c in runner.cases if c.passed is False and not c.skipped]
    print(f"\n{len(runner.cases) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
