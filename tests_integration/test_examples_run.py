from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    "cmd, expected_patterns, expected_outputs",
    [
        (
            [
                sys.executable,
                "-u",
                "examples/causal_consistency_benchmark.py",
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
            [],
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
            [],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/draw.py",
                "--output",
                "{tmp}/dag.png",
                "--dpi",
                "72",
            ],
            [
                r"Saved DAG PNG to:",
            ],
            ["{tmp}/dag.png"],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/animate.py",
                "--output",
                "{tmp}/dag.gif",
                "--fps",
                "8",
                "--frames-per-flow",
                "8",
            ],
            [
                r"Saved animated DAG flow GIF to:",
            ],
            ["{tmp}/dag.gif"],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/explore.py",
                "--output",
                "{tmp}/datacard.html",
                "--samples",
                "8",
                "--grid-size",
                "2",
                "--max-precomputed-states",
                "8",
            ],
            [
                r"Saved interactive posterior explorer to:",
            ],
            ["{tmp}/datacard.html"],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/fixed_scm_dataset.py",
            ],
            [
                r"Generated dataset from fixed SCM",
                r"Variables:",
                r"Queries:\s*2",
            ],
            [],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/save_load_dataset.py",
                "--output",
                "{tmp}/scm_dataset_example",
                "--samples",
                "16",
            ],
            [
                r"Saved dataset to:",
                r"Loaded variables:",
                r"Loaded queries:\s*\d+",
            ],
            ["{tmp}/scm_dataset_example.json", "{tmp}/scm_dataset_example.pt"],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/random_scm_collection.py",
                "--output",
                "{tmp}/random_scm_collection",
                "--samples",
                "16",
                "--queries",
                "2",
                "--num-datasets",
                "2",
            ],
            [
                r"Generated random SCM dataset collection",
                r"Saved collection to:",
                r"Subdatasets:\s*2",
            ],
            [
                "{tmp}/random_scm_collection/manifest.json",
                "{tmp}/random_scm_collection/dataset_0000.json",
                "{tmp}/random_scm_collection/dataset_0000.pt",
            ],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/random_scm_dataset.py",
                "--output",
                "{tmp}/random_scm_collection_alias",
                "--samples",
                "16",
                "--queries",
                "2",
                "--num-datasets",
                "2",
            ],
            [
                r"\[deprecated\] examples\.random_scm_dataset is renamed to examples\.random_scm_collection",
                r"Generated random SCM dataset collection",
            ],
            [
                "{tmp}/random_scm_collection_alias/manifest.json",
                "{tmp}/random_scm_collection_alias/dataset_0000.json",
                "{tmp}/random_scm_collection_alias/dataset_0000.pt",
            ],
        ),
        (
            [
                sys.executable,
                "-u",
                "examples/analyze_scm_dataset.py",
                "--input",
                "datasets/scm_dataset_example",
                "--output-json",
                "{tmp}/scm_dataset_analysis.json",
            ],
            [
                r"SCM dataset collection analysis",
                r"Aggregate coverage \(observed/treatment/outcome/pairs\):",
                r"Worst missing keys \(effects/biases\):",
            ],
            ["{tmp}/scm_dataset_analysis.json"],
        ),
    ],
)
def test_examples_run(
    cmd: list[str],
    expected_patterns: list[str],
    expected_outputs: list[str],
    tmp_path: Path,
) -> None:
    """Integration test: examples should run end-to-end and produce expected output.

    We keep hyperparameters tiny to keep CI runtime reasonable.
    """
    materialized_cmd = [part.replace("{tmp}", str(tmp_path)) for part in cmd]

    proc = subprocess.run(materialized_cmd, capture_output=True, text=True, check=False)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    assert proc.returncode == 0, (
        f"Command failed: {materialized_cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )

    for pat in expected_patterns:
        assert re.search(pat, stdout), (
            f"Missing pattern {pat!r} in output. Output:\n{stdout}"
        )

    for out in expected_outputs:
        assert Path(out.replace("{tmp}", str(tmp_path))).exists(), (
            f"Expected output file not found: {out}"
        )
