"""Create a static DAG PNG.

Usage:
    uv run python -m examples.draw
    uv run python -m examples.draw --output plots/dag.png
"""

from __future__ import annotations

import argparse

from inga.scm import SCM
from inga.scm.variable import Variable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="plots/dag.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
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

    observed_names = ["X", "V2", "V4", "V5"]
    output = scm.draw(
        output_path=args.output,
        observed_names=observed_names,
        dpi=args.dpi,
    )
    print(f"Saved DAG PNG to: {output}")


if __name__ == "__main__":
    main()
