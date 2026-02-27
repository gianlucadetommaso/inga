"""Backward-compatible alias for ``examples.random_scm_collection``.

Prefer:
    uv run python -m examples.random_scm_collection
"""

from __future__ import annotations

from examples.random_scm_collection import main


if __name__ == "__main__":
    print(
        "[deprecated] examples.random_scm_dataset is renamed to "
        "examples.random_scm_collection"
    )
    main()
