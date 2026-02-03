"""Structural Equation Model implementation."""

from torch import Tensor, nn, vmap, no_grad
from torch.func import grad
import torch
from functools import partial
from steindag.variable.base import Variable


class SEM:
    """A Structural Equation Model (SEM).

    A SEM defines a collection of variables with causal relationships.
    It supports forward sampling, MAP estimation, and approximate posterior inference.

    Attributes:
        _variables: Dictionary mapping variable names to Variable objects.
    """

    def __init__(self, variables: list[Variable]) -> None:
        """Initialize the SEM.

        Args:
            variables: List of variables in topological order (parents before children).
        """
        self._variables = {variable.name: variable for variable in variables}

    def generate(self, num_samples: int) -> dict[str, Tensor]:
        """Generate samples from the SEM by forward sampling.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Dictionary mapping variable names to their sampled tensor values.
        """
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }
            values[name] = variable.f(parents, torch.randn(num_samples))

        return values

    def _loss_fn(
        self, u_latent: dict[str, Tensor], observed: dict[str, Tensor]
    ) -> Tensor:
        """Compute the negative log posterior (up to a constant).

        Args:
            u_latent: Dictionary of latent noise variables to optimize.
            observed: Dictionary of observed variable values.

        Returns:
            The loss value (negative log posterior).
        """
        values: dict[str, Tensor] = {}
        loss: Tensor | float = 0.0

        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }
            f_bar = variable.f_bar(parents)

            if name in observed:
                values[name] = observed[name]
                u = observed[name] - f_bar

            else:
                values[name] = variable.f(parents, u_latent[name], f_bar)
                u = u_latent[name]

            loss = loss + 0.5 * torch.sum((u / variable.sigma) ** 2)

        return loss  # type: ignore[return-value]

    def fit_map(self, observed: dict[str, Tensor]) -> dict[str, Tensor]:
        """Find the MAP estimate of latent variables given observations.

        Uses L-BFGS optimization to find the maximum a posteriori estimate
        of the latent noise variables.

        Args:
            observed: Dictionary of observed variable values.

        Returns:
            Dictionary mapping latent variable names to their MAP noise estimates.
        """
        size = len(list(observed.values())[0])
        u_latent = nn.ParameterDict(
            {
                name: torch.randn(size)
                for name in self._variables
                if name not in observed
            }
        )

        optimizer = torch.optim.LBFGS(list(u_latent.values()), lr=1, max_iter=100)

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = self._loss_fn(dict(u_latent), observed)
            loss.backward()
            return loss

        optimizer.step(closure)

        return {name: u.detach().clone() for name, u in u_latent.items()}

    @no_grad()
    def approx_cov_chol(
        self, u_latent: dict[str, Tensor], observed: dict[str, Tensor]
    ) -> Tensor:
        """Compute the Cholesky factor of the approximate posterior covariance.

        Uses a Gauss-Newton approximation to the Hessian to compute an
        approximate covariance matrix for the posterior distribution.

        Args:
            u_latent: Dictionary of latent noise variables (e.g., MAP estimates).
            observed: Dictionary of observed variable values.

        Returns:
            Cholesky factor of the approximate covariance matrix.
            Shape: (num_samples, num_latent, num_latent).
        """
        size = len(u_latent)
        num_samples = len(list(u_latent.values())[0])
        device = list(u_latent.values())[0].device

        latent_names = self._get_latent_names(u_latent)
        u_latent_rav = self._ravel(u_latent)

        gn_hessian_rav = torch.zeros((num_samples, size, size), device=device)
        gn_hessian_rav[:, range(size), range(size)] = Tensor(
            [1 / self._variables[name].sigma ** 2 for name in latent_names]
        )

        for observed_name in observed:
            f_bar_wrt_u = partial(
                self._f_bar_wrt_u,
                observed_name=observed_name,
                latent_names=latent_names,
            )

            g = vmap(lambda u, o: grad(partial(f_bar_wrt_u, observed=o))(u))(
                u_latent_rav, observed
            )
            gn_hessian_rav += g[:, None] * g[:, :, None]

        if size == 1:
            L = 1 / gn_hessian_rav.sqrt()
        else:
            Linv = torch.linalg.cholesky(gn_hessian_rav, upper=True)
            L = torch.linalg.solve_triangular(
                Linv, torch.eye(size, requires_grad=False, device=device), upper=False
            )

        return L

    @no_grad()
    def sample(
        self,
        maps_rav: Tensor,
        chol_cov: Tensor,
        num_samples: int,
        latent_names: list[str],
    ) -> dict[str, Tensor]:
        """Sample from the approximate posterior distribution.

        Generates samples from a Gaussian approximation to the posterior,
        centered at the MAP estimate with covariance given by chol_cov.

        Args:
            maps_rav: Raveled MAP estimates. Shape: (batch_size, num_latent).
            chol_cov: Cholesky factor of covariance. Shape: (batch_size, num_latent, num_latent).
            num_samples: Number of posterior samples to draw.
            latent_names: List of latent variable names in order.

        Returns:
            Dictionary mapping latent variable names to sampled values.
            Each tensor has shape (batch_size, num_samples).
        """
        samples_rav = maps_rav.unsqueeze(-1) + torch.sum(
            chol_cov.unsqueeze(-1)
            * torch.randn(size=(len(maps_rav), 1, maps_rav.shape[1], num_samples)),
            dim=1,
        )
        return vmap(lambda s: self._unravel(s, latent_names), in_dims=2, out_dims=1)(
            samples_rav
        )

    def _get_latent_names(self, u_latent: dict[str, Tensor]) -> list[str]:
        """Get the names of latent variables in topological order.

        Args:
            u_latent: Dictionary of latent noise variables.

        Returns:
            List of latent variable names in the order they appear in the SEM.
        """
        return [name for name in self._variables if name in u_latent]

    def _f_bar_wrt_u(
        self,
        u_latent_rav: Tensor,
        observed: dict[str, Tensor],
        observed_name: str,
        latent_names: list[str],
    ) -> Tensor:
        """Compute f_bar of an observed variable as a function of latent noise.

        This is used to compute gradients for the Gauss-Newton Hessian.
        The gradient path through other observed variables is blocked.

        Args:
            u_latent_rav: Raveled latent noise variables.
            observed: Dictionary of observed variable values.
            observed_name: Name of the observed variable to compute f_bar for.
            latent_names: List of latent variable names in order.

        Returns:
            The f_bar value for the specified observed variable.

        Raises:
            ValueError: If observed_name is not found in the SEM.
        """
        u_latent = self._unravel(u_latent_rav, latent_names)
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                parent_name: values[parent_name]
                for parent_name in variable.parent_names
            }

            if name in observed:
                if name == observed_name:
                    return self._variables[name].f_bar(parents)

                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Observed variable '{observed_name}' not found in SEM")

    def _ravel(self, u_latent: dict[str, Tensor]) -> Tensor:
        """Convert a dictionary of latent variables to a stacked tensor.

        Args:
            u_latent: Dictionary mapping latent names to tensors.

        Returns:
            Stacked tensor of shape (batch_size, num_latent).
        """
        latent_names = self._get_latent_names(u_latent)
        return torch.stack([u_latent[name] for name in latent_names], dim=1)

    def _unravel(
        self, u_latent_rav: Tensor, latent_names: list[str]
    ) -> dict[str, Tensor]:
        """Convert a stacked tensor back to a dictionary of latent variables.

        Args:
            u_latent_rav: Stacked tensor of shape (..., num_latent).
            latent_names: List of latent variable names in order.

        Returns:
            Dictionary mapping latent names to tensors.
        """
        return {
            name: u_latent_rav.select(dim=-1, index=i)
            for i, name in enumerate(latent_names)
        }
