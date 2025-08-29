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

    # generator/train/eval removed


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


# train/eval tests removed
