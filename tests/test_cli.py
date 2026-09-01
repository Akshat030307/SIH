"""Phase 0 checkpoint (CLAUDE.md §8): ``sigscope --help`` works, every subcommand stubs."""

from __future__ import annotations

import pytest

from sigscope.cli import build_parser, main

ALL_COMMANDS = ["analyse", "batch", "fetch-data", "make-scenes", "train", "evaluate", "serve"]
# still stubbed: fetch-data landed in Phase 1, evaluate in Phase 4, analyse and batch in
# Phase 5, train in Phase 6, serve in Phase 7
STUBBED = ["make-scenes"]
# subcommands that take a required positional argument
POSITIONAL = {"analyse": ["dummy"], "batch": ["dummy"]}


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_no_command_errors():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code != 0


@pytest.mark.parametrize("cmd", STUBBED)
def test_subcommand_not_implemented(cmd, capsys):
    rc = main([cmd, *POSITIONAL.get(cmd, [])])
    assert rc == 1
    assert "not implemented" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ["analyse", "batch"])
def test_pipeline_commands_are_wired(cmd):
    """Phase 5: ``analyse`` and ``batch`` dispatch to the pipeline, not the stub."""
    from sigscope.cli import _cmd_analyse, _cmd_batch

    args = build_parser().parse_args([cmd, "somewhere"])
    assert args.func is {"analyse": _cmd_analyse, "batch": _cmd_batch}[cmd]


def test_analyse_reports_a_missing_file_cleanly(capsys):
    """§9 B: a bad input gives a message naming the problem, never a traceback."""
    rc = main(["analyse", "no_such_capture.iq"])
    assert rc == 1
    assert "no such file" in capsys.readouterr().out.lower()


def test_batch_reports_a_missing_folder_cleanly(capsys):
    rc = main(["batch", "no_such_folder"])
    assert rc == 1
    assert "not a directory" in capsys.readouterr().out


def test_serve_is_wired_but_never_started_by_the_suite():
    """``sigscope serve`` dispatches to the §7 app factory.

    Deliberately does not call it: running ``main(["serve"])`` binds a real port, which a
    test must never do -- an earlier revision of this file did exactly that and the suite
    started a server on :8000. The app itself is exercised through
    ``fastapi.testclient`` in tests/test_api.py, which binds nothing.
    """
    from sigscope.cli import _cmd_serve

    args = build_parser().parse_args(["serve"])
    assert args.func is _cmd_serve
    assert args.port == 8000
    assert args.host == "127.0.0.1", "§7 is a local-only tool with no auth"


def test_train_reports_a_missing_dataset_cleanly(capsys):
    """§6.1: the cache is built by ``fetch-data``. Without it, ``train`` says so and names
    the command that fixes it -- it does not traceback and does not train on nothing."""
    rc = main(["train", "--data", "no_such_cache"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "no RadioML cache" in out
    assert "fetch-data" in out


def test_evaluate_is_wired_to_the_harness():
    """``sigscope evaluate`` dispatches to the §9 A harness, not the stub (§8 Phase 4).

    Deliberately does not *run* it: ``main(["evaluate"])`` with default arguments would
    overwrite the repository's own ACCURACY.md as a side effect of running the test suite.
    The harness is exercised end to end in ``tests/test_evaluate.py``, into ``tmp_path``.
    """
    from sigscope.cli import _cmd_evaluate

    args = build_parser().parse_args(["evaluate"])
    assert args.func is _cmd_evaluate
    assert args.out == "ACCURACY.md"
    assert args.trials == 8


def test_fetch_data_needs_src(capsys):
    rc = main(["fetch-data"])
    assert rc == 2
    assert "--src" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ALL_COMMANDS)
def test_subcommand_help_exits_zero(cmd):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([cmd, "--help"])
    assert exc.value.code == 0
