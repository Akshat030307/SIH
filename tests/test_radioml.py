"""Loader + converter checks for sigscope.data.radioml (CLAUDE.md §6.1).

The real 225 MB pickle is not in the repo, so these build a tiny pickle with the same
*structure* (monkeypatching the size constants) and exercise convert -> load round-trip,
structure validation, hashing, and the deterministic split.
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest

from sigscope.data import radioml


@pytest.fixture
def tiny(monkeypatch):
    """Shrink the expected structure to 3 modulations x 2 SNRs x 5 examples of (2, 8)."""
    monkeypatch.setattr(radioml, "N_PER_CELL", 5)
    monkeypatch.setattr(radioml, "EXAMPLE_SHAPE", (2, 8))
    monkeypatch.setattr(radioml, "N_KEYS", 6)
    mods = [b"QPSK", b"BPSK", b"WBFM"]
    snrs = [0, 18]
    rng = np.random.default_rng(0)
    return {(m, s): rng.standard_normal((5, 2, 8)).astype(np.float32) for m in mods for s in snrs}


def _write_pickle(path, obj):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return path


def test_convert_then_load_roundtrip(tmp_path, tiny):
    src = _write_pickle(tmp_path / "RML.pkl", tiny)
    out = tmp_path / "cache"

    result = radioml.convert(src, out)
    assert result.cached is False
    assert result.n_examples == 30
    assert result.sha256 == radioml.sha256_file(src)
    assert (out / "iq.npy").exists()
    assert (out / "labels.parquet").exists()
    assert (out / "meta.json").exists()

    ds = radioml.load(out)
    assert len(ds) == 30
    assert ds.iq.shape == (30, 2, 8)
    assert set(np.unique(ds.modulation)) == {"QPSK", "BPSK", "WBFM"}
    assert set(np.unique(ds.snr_db)) == {0, 18}

    # rows are grouped in sorted (modulation, snr) order
    assert ds.modulation[0] == "BPSK" and ds.snr_db[0] == 0

    cplx = ds.complex_iq(np.arange(4))
    assert cplx.shape == (4, 8)
    assert cplx.dtype == np.complex64
    assert ds.complex_iq(0).shape == (8,)


def test_convert_is_cached_on_second_call(tmp_path, tiny, capsys):
    src = _write_pickle(tmp_path / "RML.pkl", tiny)
    out = tmp_path / "cache"
    radioml.convert(src, out)
    capsys.readouterr()
    again = radioml.convert(src, out)
    assert again.cached is True
    assert "cache present" in capsys.readouterr().out


def test_convert_rejects_sha_mismatch(tmp_path, tiny):
    src = _write_pickle(tmp_path / "RML.pkl", tiny)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        radioml.convert(src, tmp_path / "cache", sha256="deadbeef")


def test_validate_raw_rejects_bad_shape(tiny):
    tiny[(b"QPSK", 0)] = np.zeros((5, 2, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        radioml.validate_raw(tiny)


def test_validate_raw_rejects_wrong_key_count(tiny):
    tiny.pop((b"QPSK", 0))
    with pytest.raises(ValueError, match="expected 6 keys"):
        radioml.validate_raw(tiny)


def test_validate_raw_accepts_str_and_bytes_keys(tiny):
    radioml.validate_raw(tiny)  # bytes keys
    radioml.validate_raw({(k[0].decode(), k[1]): v for k, v in tiny.items()})  # str keys


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"radioml" * 1000)
    assert radioml.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_split_indices_partition_and_determinism():
    n = 1000
    parts = {w: radioml.split_indices(w, n) for w in ("train", "val", "test")}
    allidx = np.concatenate(list(parts.values()))
    assert np.array_equal(np.sort(allidx), np.arange(n))  # disjoint + covering
    assert len(parts["train"]) == 700
    assert len(parts["val"]) == 150
    assert len(parts["test"]) == 150
    assert np.array_equal(parts["test"], radioml.split_indices("test", n))  # deterministic
    assert list(parts["test"]) == sorted(parts["test"])  # sorted for memmap access


def test_load_without_cache_is_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-data"):
        radioml.load(tmp_path / "nope")
