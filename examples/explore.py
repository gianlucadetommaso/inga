"""Create an interactive posterior explorer HTML (multi-observation example).

Usage:
    uv run python -m examples.posterior_explorer
    uv run python -m examples.posterior_explorer --output plots/posterior_explorer.html
"""

from __future__ import annotations

import argparse

from steindag.sem import SEM
from steindag.variable import LinearVariable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="plots/posterior_explorer.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=400,
        help="Posterior predictive samples per precomputed slider state.",
    )
    args = parser.parse_args()

    sem = SEM(
        variables=[
            LinearVariable(
                name="Z1", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="Z2", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X",
                parent_names=["Z1", "Z2"],
                sigma=1.0,
                coefs={"Z1": 1.0, "Z2": -0.6},
                intercept=0.2,
            ),
            LinearVariable(
                name="M",
                parent_names=["X", "Z1"],
                sigma=1.0,
                coefs={"X": 1.4, "Z1": 0.8},
                intercept=0.0,
            ),
            LinearVariable(
                name="Y",
                parent_names=["X", "M", "Z2"],
                sigma=1.0,
                coefs={"X": 1.2, "M": 1.6, "Z2": 0.7},
                intercept=-0.1,
            ),
            LinearVariable(
                name="V",
                parent_names=["Y", "M"],
                sigma=1.0,
                coefs={"Y": 1.1, "M": 0.5},
                intercept=0.0,
            ),
        ],
        posterior_kwargs={
            "num_map_restarts": 1,
            "continuation_steps": 8,
            "num_mixture_components": 1,
            "lbfgs_max_iter": 25,
        },
    )

    output = sem.export_html(
        output_path=args.output,
        observed_ranges={
            "X": (-2.0, 2.0, 5),
            "M": (-2.5, 2.5, 5),
            "V": (-3.0, 3.0, 4),
        },
        baseline_observed={"X": 0.0, "M": 0.0, "V": 0.0},
        outcome_name="Y",
        num_posterior_samples=args.samples,
        max_precomputed_states=300,
        title="SteinDAG Posterior Explorer",
    )
    print(f"Saved interactive posterior explorer to: {output}")


if __name__ == "__main__":
    main()
