"""Smoke tests: the examples run end to end on a tiny budget."""

import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_black_scholes_pricer_runs():
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "black_scholes_pricer.py"),
            "--samples",
            "2000",
            "--epochs",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "Greeks via autodiff" in result.stdout
    assert "delta" in result.stdout
