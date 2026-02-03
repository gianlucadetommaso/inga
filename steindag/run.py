"""Demo script for the structural equation model.

This script demonstrates the SEM functionality by:
1. Creating a simple linear SEM with variables Z -> X -> Y and Z -> Y
2. Generating samples from the model
3. Performing MAP inference with partial observations
4. Computing approximate posterior samples
5. Comparing against analytical ground truth
"""

import torch
from torch import Tensor
from steindag.variable.linear import LinearVariable
from steindag.sem.base import SEM


def _get_coef(sem: SEM, var_name: str, parent_name: str) -> float:
    """Get a linear coefficient from a LinearVariable in the SEM.

    Args:
        sem: The structural equation model.
        var_name: Name of the variable.
        parent_name: Name of the parent variable.

    Returns:
        The linear coefficient for the specified parent.

    Raises:
        AssertionError: If the variable is not a LinearVariable.
    """
    var = sem._variables[var_name]
    assert isinstance(var, LinearVariable)
    return var._coefs[parent_name]


def main() -> None:
    """Run the demo showing posterior inference on a linear SEM.

    Creates a SEM with structure Z -> X, Z -> Y, X -> Y, generates samples,
    and demonstrates posterior inference under two scenarios:
    1. Observing only X
    2. Observing both X and Y

    Prints the absolute error between estimated and analytical posterior
    means and variances.
    """
    sem = SEM(
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
                coefs={"X": 2.0, "Z": 3.0},
                intercept=0.0,
            ),
        ]
    )

    alpha = _get_coef(sem, "X", "Z")
    beta = _get_coef(sem, "Y", "X")
    gamma = _get_coef(sem, "Y", "Z")

    values = sem.generate(10)

    # observe only X
    observed: dict[str, Tensor] = {"X": values["X"]}

    maps = sem.fit_map(observed)
    chols_cov = sem.approx_cov_chol(maps, observed)
    maps_rav = sem._ravel(maps)

    samples = sem.sample(maps_rav, chols_cov, 10000, sem._get_latent_names(maps))

    posterior_means = alpha * observed["X"] / (1 + alpha**2)
    posterior_vars = Tensor([1 / (1 + alpha**2)]).expand(10)

    print(torch.abs(posterior_means - samples["Z"].mean(1)))
    print(torch.abs(posterior_vars - samples["Z"].var(1)))

    # observe X and Y
    observed = {"X": values["X"], "Y": values["Y"]}

    maps = sem.fit_map(observed)
    chols_cov = sem.approx_cov_chol(maps, observed)
    maps_rav = sem._ravel(maps)

    samples = sem.sample(maps_rav, chols_cov, 10000, sem._get_latent_names(maps))

    posterior_means = (
        gamma * (observed["Y"] - beta * observed["X"]) + alpha * observed["X"]
    ) / (1 + alpha**2 + gamma**2)
    posterior_vars = Tensor([1 / (1 + alpha**2 + gamma**2)]).expand(10)

    print(torch.abs(posterior_means - samples["Z"].mean(1)))
    print(torch.abs(posterior_vars - samples["Z"].var(1)))


if __name__ == "__main__":
    main()
