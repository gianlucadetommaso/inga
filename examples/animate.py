"""Create an animated DAG GIF.

Usage:
    uv run python -m examples.animate
    uv run python -m examples.animate --output examples/dag.gif
"""

from __future__ import annotations

import argparse

from inga.scm import SCM
from inga.variable import Variable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="plots/dag.gif",
        help="Output GIF path.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames-per-flow", type=int, default=36)
    args = parser.parse_args()

    scm = SCM(
        variables=[
            Variable(name="V1"),
            Variable(name="X", parent_names=["V1"]),
            Variable(name="V2", parent_names=["V1", "X"]),
            Variable(name="V3", parent_names=["V1", "X", "V2"]),
            Variable(name="Y", parent_names=["V1", "X", "V2", "V3"]),
            Variable(name="V4", parent_names=["V1", "X", "V2", "V3", "Y"]),
            Variable(name="V5", parent_names=["X", "V3", "Y"]),
        ]
    )

    treatment_name = "X"
    outcome_name = "Y"
    observed_names = ["X", "V2", "V4", "V5"]

    output = scm.animate_flow_gif(
        output_path=args.output,
        observed_names=observed_names,
        treatment_name=treatment_name,
        outcome_name=outcome_name,
        fps=args.fps,
        frames_per_flow=args.frames_per_flow,
    )
    print(f"Saved animated DAG flow GIF to: {output}")


if __name__ == "__main__":
    main()
