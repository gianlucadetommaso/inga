"""Generate a dataset from a user-defined (fixed) SCM.

Usage:
    uv run python -m examples.fixed_scm_dataset
"""

from __future__ import annotations

from inga.scm import CausalQueryConfig, SCM
from inga.scm.variable import LinearVariable


def main() -> None:
    scm = SCM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X",
                parent_names=["Z"],
                sigma=1.0,
                coefs={"Z": 1.0},
                intercept=0.0,
            ),
            LinearVariable(
                name="Y",
                parent_names=["X", "Z"],
                sigma=1.0,
                coefs={"X": 2.0, "Z": 1.0},
                intercept=0.0,
            ),
        ]
    )

    dataset = scm.generate_dataset(
        num_samples=128,
        seed=123,
        queries=[
            CausalQueryConfig(
                treatment_name="X",
                outcome_name="Y",
                observed_names=["X"],
            ),
            CausalQueryConfig(
                treatment_name="X",
                outcome_name="Z",
                observed_names=["X", "Y"],
            ),
        ],
    )

    print("Generated dataset from fixed SCM")
    print(f"Variables: {list(dataset.data.keys())}")
    print(f"Queries: {len(dataset.queries)}")


if __name__ == "__main__":
    main()
