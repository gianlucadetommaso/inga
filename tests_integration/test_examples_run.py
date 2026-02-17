from __future__ import annotations

import re
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    "cmd, expected_patterns",
    [
        (
            [
                sys.executable,
                "-u",
                "examples/causal_training_benchmark.py",
                "--num-seeds",
                "1",
                "--epochs",
                "1",
                "--train-size",
                "64",
                "--test-size",
                "32",
            ],
            [
                r"Per-seed results \(seed=00\)",
                r"Summary across seeds",
                r"\|\s*causal_consistency\s*\|",
            ],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/causal_transfer_benchmark.py",
                "--train-datasets",
                "4",
                "--test-datasets",
                "2",
                "--samples-per-dataset",
                "64",
                "--epochs",
                "1",
                "--batch-size",
                "128",
            ],
            [
                r"Test dataset results",
                r"Transfer summary \(across test datasets\)",
                r"\|\s*causal\s*\|",
            ],
        ),
    ],
)
def test_examples_run(cmd: list[str], expected_patterns: list[str]) -> None:
    """Integration test: examples should run end-to-end and produce expected output.

    We keep hyperparameters tiny to keep CI runtime reasonable.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    assert proc.returncode == 0, (
        f"Command failed: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )

    for pat in expected_patterns:
        assert re.search(pat, stdout), (
            f"Missing pattern {pat!r} in output. Output:\n{stdout}"
        )
