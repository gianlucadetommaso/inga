"""Create an interactive posterior explorer HTML (multi-observation example).

Usage:
    uv run python -m examples.explorer
    uv run python -m examples.explorer --output plots/explorer.html
"""

from __future__ import annotations

import argparse

from inga.scm import SCM
from inga.scm.variable import LinearVariable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="plots/datacard.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=400,
        help="Posterior predictive samples per precomputed slider state.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Grid points for X and M sliders (V uses max(2, grid-size - 1)).",
    )
    parser.add_argument(
        "--max-precomputed-states",
        type=int,
        default=300,
        help="Upper bound on precomputed slider states.",
    )
    args = parser.parse_args()

    scm = SCM(
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

    output = scm.export_html(
        output_path=args.output,
        observed_ranges={
            "X": (-2.0, 2.0, args.grid_size),
            "M": (-2.5, 2.5, args.grid_size),
            "V": (-3.0, 3.0, max(2, args.grid_size - 1)),
        },
        baseline_observed={"X": 0.0, "M": 0.0, "V": 0.0},
        outcome_name="Y",
        num_posterior_samples=args.samples,
        max_precomputed_states=args.max_precomputed_states,
        title="SteinDAG Posterior Explorer",
    )
    print(f"Saved interactive posterior explorer to: {output}")


if __name__ == "__main__":
    main()
