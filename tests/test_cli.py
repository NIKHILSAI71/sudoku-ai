import sys
import subprocess


def run_cli(args):
    proc = subprocess.run(
        [sys.executable, "-m", "cli.main", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def test_top_level_help_shows_examples_and_aliases():
    code, out = run_cli(["--help"])
    assert code == 0
    assert "Sudoku toolkit" in out
    assert "solve (slove)" in out
    assert "ai-solve (ai-slove)" in out

    # generator removed


def test_solve_alias_help_usage_and_flags():
    code, out = run_cli(["slove", "--help"])
    assert code == 0
    assert "usage: sudoku solve" in out
    assert "Solver backend to use" in out


def test_ai_solve_alias_help_usage_and_flags():
    code, out = run_cli(["ai-slove", "--help"])
    assert code == 0
    assert "usage: sudoku ai-solve" in out
    assert "Path to policy checkpoint (.pt)" in out


    # rate removed


def test_eval_help_choices_and_defaults():
    code, out = run_cli(["eval", "--help"])
    assert code == 0
    assert "Evaluate one or more solver backends" in out
    assert "Backends to evaluate" in out


def test_train_help_defaults_and_desc():
    code, out = run_cli(["train", "--help"])
    assert code == 0
    assert "Train a small policy network" in out
    assert "--epochs" in out
