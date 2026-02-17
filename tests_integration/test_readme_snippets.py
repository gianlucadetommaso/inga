from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest


README_PATH = Path("README.md")


def _extract_python_blocks(readme_text: str) -> list[str]:
    """Extract fenced python code blocks from README in order."""
    return re.findall(r"```python\r?\n(.*?)```", readme_text, flags=re.S)


def _make_runtime_friendly(code: str) -> str:
    """Apply tiny test-only tweaks so docs snippets stay fast in CI.

    We preserve snippet semantics while reducing expensive defaults and replacing
    placeholder output paths with temporary local files.
    """
    out = code

    # Keep expensive posterior calls lightweight.
    out = out.replace("causal_effect(\n", "causal_effect(\n    num_samples=64,\n")
    out = out.replace("causal_bias(\n", "causal_bias(\n    num_samples=64,\n")

    # Keep dataset generation lightweight.
    out = out.replace("num_samples=128", "num_samples=16")

    # Route placeholder paths to files that can be created during test execution.
    out = out.replace('output_path="YOUR_DAG.png"', 'output_path="readme_dag.png"')
    out = out.replace('output_path="YOUR_SCM.html"', 'output_path="readme_scm.html"')
    out = out.replace(
        'dataset_path = "YOUR_DATASET.json"', 'dataset_path = "readme_dataset.json"'
    )

    # Keep HTML export fast and bounded.
    out = out.replace(
        'observed_ranges={"X": (-2.0, 2.0)}',
        'observed_ranges={"X": (-2.0, 2.0)},\n    num_posterior_samples=20,\n    max_precomputed_states=32',
    )

    return out


@pytest.mark.integration
def test_readme_python_blocks_compile() -> None:
    """Every README python block should at least parse and compile."""
    readme = README_PATH.read_text(encoding="utf-8")
    blocks = _extract_python_blocks(readme)

    assert blocks, "No python code blocks found in README.md"

    for idx, block in enumerate(blocks):
        compile(block, f"README.md::python_block_{idx}", "exec")


@pytest.mark.integration
def test_readme_tutorial_python_blocks_run(tmp_path: Path) -> None:
    """Execute README tutorial snippets in order.

    Running snippets sequentially mirrors the tutorial flow, where later blocks
    intentionally rely on symbols created in earlier blocks.
    """
    readme = README_PATH.read_text(encoding="utf-8")
    blocks = _extract_python_blocks(readme)
    assert blocks, "No python code blocks found in README.md"

    module_name = "__readme_snippets__"
    module = types.ModuleType(module_name)
    namespace = module.__dict__
    namespace["__name__"] = module_name
    sys.modules[module_name] = module

    old_cwd = Path.cwd()
    try:
        # Avoid polluting repository root with generated doc artefacts.
        import os

        os.chdir(tmp_path)

        for idx, block in enumerate(blocks):
            runtime_block = _make_runtime_friendly(block)
            compiled = compile(runtime_block, f"README.md::python_block_{idx}", "exec")
            exec(compiled, namespace)
    finally:
        import os

        os.chdir(old_cwd)
        sys.modules.pop(module_name, None)
