"""Phase 0 checkpoint (CLAUDE.md §8): ``sigscope --help`` works, every subcommand stubs."""

from __future__ import annotations

import pytest

from sigscope.cli import build_parser, main

ALL_COMMANDS = ["analyse", "batch", "fetch-data", "make-scenes", "train", "evaluate", "serve"]
# still stubbed in Phase 0 (fetch-data landed in Phase 1)
STUBBED = ["analyse", "batch", "make-scenes", "train", "evaluate", "serve"]
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


def test_fetch_data_needs_src(capsys):
    rc = main(["fetch-data"])
    assert rc == 2
    assert "--src" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ALL_COMMANDS)
def test_subcommand_help_exits_zero(cmd):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([cmd, "--help"])
    assert exc.value.code == 0
