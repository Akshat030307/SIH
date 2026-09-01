"""The §9 A accuracy harness produces a valid report (CLAUDE.md §8 Phase 4, §9 A).

``scripts/evaluate.py`` is what turns the SNR sweeps into ACCURACY.md, and §10 says that
table is the slide that wins the pitch -- so it gets the same treatment as the estimators
it measures. Runs the real harness at one trial per point, into ``tmp_path``, and checks
the report is well formed, deterministic, and honest about break points.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"


@pytest.fixture(scope="module")
def harness():
    """Load ``scripts/evaluate.py`` the way the CLI does."""
    spec = importlib.util.spec_from_file_location("sigscope_evaluate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(harness, tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("accuracy") / "ACCURACY.md"
    assert harness.main(["--out", str(out), "--trials", "1", "--seed", "1000"]) == 0
    return out.read_text(encoding="utf-8")


def test_report_covers_every_9a_row(report, harness):
    for name, *_ in harness.ROWS:
        assert name.replace("|", r"\|") in report, f"{name} missing from the report"


def test_report_table_is_well_formed(report, harness):
    """Every data row must have the same cell count as the header.

    Counts *cell separators*, so ``\\|`` inside a name (as in ``|C40|/C21^2``) does not
    count -- that escape is exactly what keeps the cell intact. An unescaped pipe would
    silently split the cell and make the table ragged, which is what this catches.
    """
    lines = [line for line in report.splitlines() if line.startswith("|")]
    assert lines, "no table found"
    widths = {line.replace(r"\|", "").count("|") for line in lines}
    assert len(widths) == 1, f"ragged table: rows have {sorted(widths)} separators"
    # header + separator + one row per estimator
    assert len(lines) == len(harness.ROWS) + 2


def test_report_is_deterministic(harness, tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    harness.main(["--out", str(first), "--trials", "1", "--seed", "1000"])
    harness.main(["--out", str(second), "--trials", "1", "--seed", "1000"])

    def strip_runtime(text: str) -> list[str]:
        return [line for line in text.splitlines() if "generated in" not in line]

    assert strip_runtime(first.read_text(encoding="utf-8")) == strip_runtime(
        second.read_text(encoding="utf-8")
    )


def test_report_states_where_we_fail(report):
    """§10: the accuracy slide has to include the part where we lose."""
    assert "Where we fail" in report
    assert "Break point" in report


def test_break_point_stops_at_the_first_failure(harness):
    """A row passing at 20 and 10 dB but failing at 15 must report 20 dB, not 10 --
    otherwise the table would claim performance the estimator does not have."""
    row = harness.Row(name="x", truth="", tolerance="", unit="")
    row.errors = {20.0: [0.0], 15.0: [9.0], 10.0: [0.0]}
    row.passed = {20.0: [True], 15.0: [False], 10.0: [True]}

    assert row.break_point() == 20.0


def test_break_point_is_none_when_nothing_passes(harness):
    row = harness.Row(name="x", truth="", tolerance="", unit="")
    row.errors = {20.0: [9.0], 10.0: [9.0]}
    row.passed = {20.0: [False], 10.0: [False]}

    assert row.break_point() is None


def test_abstentions_are_reported_not_hidden(harness):
    """An abstention is a stated non-answer (§2), and the table must show it as such
    rather than quietly averaging over the trials that did answer."""
    row = harness.Row(name="x", truth="", tolerance="", unit="%")
    row.errors = {10.0: [None, None]}
    row.passed = {10.0: [False, False]}
    row.abstentions = {10.0: 2}

    assert "abstained" in harness._fmt_error(row, 10.0)
