"""The frozen §3 schema must serialise key-for-key (CLAUDE.md §3)."""

from __future__ import annotations

from sigscope.types import (
    SCHEMA_VERSION,
    Bandwidth,
    CaptureMeta,
    Detection,
    Estimate,
    FileInfo,
    Modulation,
    Report,
)


def _example_report() -> Report:
    return Report(
        file=FileInfo(name="capture.iq", sha256="0" * 64, bytes=8388608),
        capture=CaptureMeta(
            sample_rate=2_000_000.0,
            center_freq=100_000_000.0,
            dtype_guessed="int16",
            dtype_confidence=0.91,
            source_format="raw",
            start_time=None,
            n_samples=4_194_304,
            metadata_confidence=0.8,
            notes=["centre frequency parsed from filename pattern SDRSharp_*"],
            center_freq_source="filename",
            frequencies_are_normalised=False,
        ),
        noise_floor_dbfs=-74.2,
        detections=[
            Detection(
                id=1,
                time_start_s=0.104,
                time_stop_s=0.612,
                duration_s=0.508,
                center_freq_hz=100_250_000.0,
                center_freq_offset_hz=250_000.0,
                bandwidth_hz=Bandwidth(occupied_99=48200.0, minus_3db=31000.0, minus_20db=62400.0),
                snr_db=21.4,
                power_dbfs=-32.8,
                symbol_rate_hz=Estimate(value=31250.0, confidence=0.88, method="cyclostationary"),
                modulation=Modulation(
                    label="QPSK",
                    confidence=0.83,
                    runner_up="8PSK",
                    runner_up_confidence=0.11,
                    votes={"feature_clf": "QPSK", "cnn": "QPSK", "rules": None},
                    evidence=["C40 magnitude 0.98 close to QPSK theoretical 1.00", "constant env"],
                ),
                extra={"psk_order": 4, "carrier_offset_hz": 1240},
            )
        ],
        warnings=["sample rate not found in file; 2 MHz assumed from filename"],
        runtime_s=3.9,
    )


def test_report_top_level_keys():
    d = _example_report().to_dict()
    assert list(d) == [
        "schema_version",
        "file",
        "capture",
        "noise_floor_dbfs",
        "detections",
        "warnings",
        "runtime_s",
    ]
    assert d["schema_version"] == SCHEMA_VERSION == "1.0"


def test_capture_block_keys():
    d = _example_report().to_dict()["capture"]
    assert list(d) == [
        "sample_rate_hz",
        "center_freq_hz",
        "center_freq_source",
        "duration_s",
        "dtype",
        "dtype_confidence",
        "frequencies_are_normalised",
        "notes",
    ]


def test_detection_block_keys():
    det = _example_report().to_dict()["detections"][0]
    assert list(det) == [
        "id",
        "time_start_s",
        "time_stop_s",
        "duration_s",
        "center_freq_hz",
        "center_freq_offset_hz",
        "bandwidth_hz",
        "snr_db",
        "power_dbfs",
        "symbol_rate_hz",
        "modulation",
        "extra",
    ]
    assert list(det["bandwidth_hz"]) == ["occupied_99", "minus_3db", "minus_20db"]
    assert list(det["symbol_rate_hz"]) == ["value", "confidence", "method"]
    assert list(det["modulation"]) == [
        "label",
        "confidence",
        "runner_up",
        "runner_up_confidence",
        "votes",
        "evidence",
    ]


def test_report_json_roundtrips():
    import json

    d = _example_report().to_dict()
    assert json.loads(json.dumps(d)) == d


def test_unmeasured_values_are_none_not_placeholders():
    est = Estimate(value=None, confidence=0.0, method="cyclostationary")
    assert est.to_dict() == {"value": None, "confidence": 0.0, "method": "cyclostationary"}
