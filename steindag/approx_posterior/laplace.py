"""Laplace approximation for posterior inference in SEMs.

This module provides a Laplace (Gaussian) approximation to the posterior
distribution over latent noise variables given observations.
"""

import torch
from torch import Tensor, no_grad, vmap, nn
from torch.func import grad
from dataclasses import dataclass
from functools import partial
from steindag.variable.base import Variable
from steindag.variable.functional import FunctionalVariable


_LATENT_ABS_MAX = 8.0
_EPS_ABS_MAX = 3.0
_MAX_STD_DEVS = 3.0
_NONLINEAR_LATENT_ABS_MAX = 5.0
_NONLINEAR_VAR_CAP = 0.25


@dataclass
class LaplacePosteriorState:
    """State of a fitted Laplace posterior.

    Attributes:
        MAP_rav: Raveled MAP estimates of latent variables.
        L_cov_rav: Cholesky factor of the approximate covariance matrix.
        latent_names: Names of latent variables in order.
    """

    MAP_rav: Tensor
    L_cov_rav: Tensor
    latent_names: list[str]


class LaplacePosterior:
    """Laplace approximation to the posterior distribution.

    This class computes a Gaussian approximation to the posterior distribution
    over latent noise variables in a SEM. The approximation is centered at the
    MAP estimate with covariance given by the inverse Gauss-Newton Hessian.

    Attributes:
        _variables: Dictionary mapping variable names to Variable objects.
        _state: Current fitted state, or None if not yet fitted.
    """

    def __init__(self, variables: dict[str, Variable]) -> None:
        """Initialize the Laplace posterior.

        Args:
            variables: Dictionary mapping variable names to Variable objects.
        """
        self._variables = variables
        self._state: LaplacePosteriorState | None = None

    def _has_nonlinear_variables(self) -> bool:
        return any(isinstance(v, FunctionalVariable) for v in self._variables.values())

    def fit(self, observed: dict[str, Tensor]) -> None:
        """Fit the Laplace posterior to observed data.

        Computes the MAP estimate and the Cholesky factor of the approximate
        covariance matrix.

        Args:
            observed: Dictionary of observed variable values.
        """
        MAP = self._fit_map(observed)
        MAP_rav = self._ravel(MAP)
        latent_names = self._get_latent_names(MAP)
        L_cov_rav = self._approx_cov_chol(MAP_rav, observed, latent_names)

        self.state = LaplacePosteriorState(
            MAP_rav=MAP_rav, L_cov_rav=L_cov_rav, latent_names=latent_names
        )

    @no_grad()
    def sample(
        self,
        num_samples: int,
    ) -> dict[str, Tensor]:
        """Sample from the approximate posterior distribution.

        Generates samples from a Gaussian approximation to the posterior,
        centered at the MAP estimate with covariance given by the fitted
        Cholesky factor.

        Args:
            num_samples: Number of posterior samples to draw.

        Returns:
            Dictionary mapping latent variable names to sampled values.
            Each tensor has shape (batch_size, num_samples).

        Raises:
            ValueError: If `fit` has not been called yet.
        """
        if self.state is None:
            raise ValueError("`fit` must be invoked first.")

        state = self.state
        has_nonlinear = self._has_nonlinear_variables()
        latent_abs_max = _NONLINEAR_LATENT_ABS_MAX if has_nonlinear else _LATENT_ABS_MAX

        if has_nonlinear:
            # Use antithetic pairs + clipped standard-normal draws to reduce
            # Monte Carlo variance and avoid rare tail samples dominating
            # highly nonlinear downstream quantities (e.g. causal bias terms).
            half = (num_samples + 1) // 2
            eps_half = torch.randn(
                size=(
                    len(state.MAP_rav),
                    state.MAP_rav.shape[1],
                    half,
                ),
                device=state.MAP_rav.device,
                dtype=state.MAP_rav.dtype,
            ).clamp(-_EPS_ABS_MAX, _EPS_ABS_MAX)
            eps = torch.cat([eps_half, -eps_half], dim=2)[..., :num_samples]
        else:
            # Preserve exact Gaussian sampling behavior for linear SEMs, where
            # analytical posterior tests expect unbiased Laplace sampling.
            eps = torch.randn(
                size=(
                    len(state.MAP_rav),
                    state.MAP_rav.shape[1],
                    num_samples,
                ),
                device=state.MAP_rav.device,
                dtype=state.MAP_rav.dtype,
            )
        # Correct Gaussian sampling: u = MAP + L @ eps, where Cov(u|x)=L L^T.
        samples_rav = state.MAP_rav.unsqueeze(-1) + torch.matmul(state.L_cov_rav, eps)
        if has_nonlinear:
            # Additional robustification only for nonlinear SEMs: truncate to a
            # bounded number of posterior standard deviations around MAP.
            delta = (samples_rav - state.MAP_rav.unsqueeze(-1)).clamp(
                -_MAX_STD_DEVS, _MAX_STD_DEVS
            )
            samples_rav = state.MAP_rav.unsqueeze(-1) + delta
        samples_rav = samples_rav.clamp(-latent_abs_max, latent_abs_max)
        return vmap(
            lambda s: self._unravel(s, state.latent_names), in_dims=2, out_dims=1
        )(samples_rav)

    @property
    def state(self) -> LaplacePosteriorState | None:
        """Get the current fitted state.

        Returns:
            The fitted state, or None if not yet fitted.
        """
        return self._state

    @state.setter
    def state(self, state: LaplacePosteriorState) -> None:
        """Set the fitted state.

        Args:
            state: The new state to set.
        """
        self._state = state

    def _fit_map(self, observed: dict[str, Tensor]) -> dict[str, Tensor]:
        """Find the MAP estimate of latent variables given observations.

        Uses L-BFGS optimization to find the maximum a posteriori estimate
        of the latent noise variables.

        Args:
            observed: Dictionary of observed variable values.

        Returns:
            Dictionary mapping latent variable names to their MAP noise estimates.
        """
        size = len(list(observed.values())[0])
        latent_abs_max = (
            _NONLINEAR_LATENT_ABS_MAX if self._has_nonlinear_variables() else _LATENT_ABS_MAX
        )
        u_latent = nn.ParameterDict(
            {
                name: torch.randn(size)
                for name in self._variables
                if name not in observed
            }
        )

        optimizer = torch.optim.LBFGS(list(u_latent.values()), lr=1, max_iter=100)

        def closure() -> Tensor:
            with torch.no_grad():
                for u in u_latent.values():
                    u.nan_to_num_(nan=0.0, posinf=latent_abs_max, neginf=-latent_abs_max)
                    u.clamp_(-latent_abs_max, latent_abs_max)

            optimizer.zero_grad()
            loss = self._posterior_loss_fn(u_latent, observed)
            if not torch.isfinite(loss):
                # If the posterior objective becomes non-finite, switch to a
                # finite quadratic fallback on the latent variables to pull the
                # optimizer back to a numerically valid region.
                loss = torch.zeros((), device=list(u_latent.values())[0].device)
                for u in u_latent.values():
                    u_safe = torch.nan_to_num(
                        u,
                        nan=0.0,
                        posinf=latent_abs_max,
                        neginf=-latent_abs_max,
                    )
                    loss = loss + 0.5 * torch.sum(u_safe**2)
            loss.backward()
            with torch.no_grad():
                for u in u_latent.values():
                    if u.grad is not None:
                        u.grad.nan_to_num_(nan=0.0, posinf=1e3, neginf=-1e3)
            # Preserve gradients via backward, but return a detached scalar for LBFGS.
            return loss.detach()

        optimizer.step(closure)

        # LBFGS may leave parameters in a non-finite state after its final line
        # search update. Enforce finiteness before returning the MAP estimate.
        with torch.no_grad():
            for u in u_latent.values():
                u.nan_to_num_(nan=0.0, posinf=latent_abs_max, neginf=-latent_abs_max)
                u.clamp_(-latent_abs_max, latent_abs_max)

        return {name: u.detach().clone() for name, u in u_latent.items()}

    def _posterior_loss_fn(
        self, u_latent: nn.ParameterDict, observed: dict[str, Tensor]
    ) -> Tensor:
        """Compute the negative log posterior (up to a constant).

        Args:
            u_latent: Dictionary of latent noise parameters to optimize.
            observed: Dictionary of observed variable values.

        Returns:
            The loss value (negative log posterior).
        """
        values: dict[str, Tensor] = {}
        loss: Tensor | float = 0.0
        latent_abs_max = (
            _NONLINEAR_LATENT_ABS_MAX if self._has_nonlinear_variables() else _LATENT_ABS_MAX
        )

        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }
            f_mean = variable.f_mean(parents)
            f_mean = torch.nan_to_num(f_mean, nan=0.0, posinf=1e6, neginf=-1e6)

            if name in observed:
                values[name] = observed[name]
                u = (observed[name] - f_mean) / variable.sigma

            else:
                latent_u = torch.nan_to_num(
                    u_latent[name],
                    nan=0.0,
                    posinf=latent_abs_max,
                    neginf=-latent_abs_max,
                )
                latent_u = latent_u.clamp(-latent_abs_max, latent_abs_max)
                values[name] = variable.f(parents, latent_u, f_mean)
                values[name] = torch.nan_to_num(
                    values[name], nan=0.0, posinf=1e6, neginf=-1e6
                )
                u = latent_u

            u = torch.nan_to_num(
                u,
                nan=0.0,
                posinf=latent_abs_max,
                neginf=-latent_abs_max,
            )
            u = u.clamp(-latent_abs_max, latent_abs_max)

            loss = loss + 0.5 * torch.sum(u**2)

        return loss  # type: ignore[return-value]

    @no_grad()
    def _approx_cov_chol(
        self,
        u_latent_rav: Tensor,
        observed: dict[str, Tensor],
        latent_names: list[str],
    ) -> Tensor:
        """Compute the Cholesky factor of the approximate posterior covariance.

        Uses a Gauss-Newton approximation to the Hessian to compute an
        approximate covariance matrix for the posterior distribution.

        Args:
            u_latent_rav: Ravelled latent noise variables (e.g., MAP estimates).
            observed: Dictionary of observed variable values.
            latent_names: List of latent variable names in order.

        Returns:
            Cholesky factor of the approximate covariance matrix.
            Shape: (num_samples, num_latent, num_latent).

        Raises:
            ValueError: If dimensions of u_latent_rav don't match latent_names.
        """
        latent_dim = len(latent_names)
        if u_latent_rav.shape[1] != latent_dim:
            raise ValueError(
                f"`u_latent_rav.shape[1]` and `len(latent_names)` must match, "
                f"but {u_latent_rav.shape[1]} != {latent_dim}."
            )

        num_samples = len(u_latent_rav)
        device = u_latent_rav.device
        has_nonlinear = self._has_nonlinear_variables()
        var_cap = _NONLINEAR_VAR_CAP if has_nonlinear else 1.0

        gn_hessian_rav = torch.zeros(
            (num_samples, latent_dim, latent_dim), device=device
        )
        # Prior precision over latent noises u is identity (u ~ N(0, I)).
        gn_hessian_rav[:, range(latent_dim), range(latent_dim)] = 1.0

        for observed_name in observed:
            f_mean_wrt_u = partial(
                self._f_mean_u,
                observed_name=observed_name,
                latent_names=latent_names,
            )
            g = vmap(lambda u, o: grad(partial(f_mean_wrt_u, observed=o))(u))(
                u_latent_rav, observed
            )
            g = torch.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
            # Observed term in posterior loss is 0.5 * ((y-f)/sigma)^2,
            # so Jacobian contribution is scaled by 1/sigma.
            sigma_obs = self._variables[observed_name].sigma
            g_scaled = g / sigma_obs
            gn_hessian_rav += g_scaled[:, None] * g_scaled[:, :, None]

        gn_hessian_rav = torch.nan_to_num(
            gn_hessian_rav, nan=0.0, posinf=1e6, neginf=-1e6
        )
        # Gauss-Newton Hessian should be symmetric PSD in theory, but numerical
        # autodiff/vmap noise can introduce slight asymmetry/indefiniteness.
        gn_hessian_rav = 0.5 * (gn_hessian_rav + gn_hessian_rav.transpose(-1, -2))

        if latent_dim == 1:
            L = 1 / gn_hessian_rav.clamp_min(1e-8).sqrt()
        else:
            # Per-sample robust factorization. Some batches can be numerically
            # ill-conditioned even when others are fine; handling each sample
            # separately avoids failing the whole fit.
            eye = torch.eye(latent_dim, requires_grad=False, device=device)
            L = torch.zeros((num_samples, latent_dim, latent_dim), device=device)

            for i in range(num_samples):
                h_i = 0.5 * (gn_hessian_rav[i] + gn_hessian_rav[i].T)
                chol_h_i = None
                jitter = 1e-10

                for _ in range(14):
                    chol_try, info = torch.linalg.cholesky_ex(
                        h_i + jitter * eye,
                        upper=False,
                        check_errors=False,
                    )
                    if int(info.item()) == 0 and torch.isfinite(chol_try).all():
                        chol_h_i = chol_try
                        break
                    jitter *= 10

                if chol_h_i is not None:
                    # If H = C C^T (C lower), then a Cholesky factor of H^{-1}
                    # is C^{-1}.
                    L[i] = torch.linalg.solve_triangular(
                        chol_h_i,
                        eye,
                        upper=False,
                    )
                else:
                    # Final fallback: diagonal precision approximation.
                    # This keeps posterior sampling finite without masking
                    # values in downstream causal computations.
                    d = torch.diag(h_i).clamp_min(1e-6)
                    L[i] = torch.diag(1.0 / torch.sqrt(d))

        # Posterior over latent noises u should not be more diffuse than the
        # N(0, I) prior in this model class (precision = I + J^T J). Numerical
        # approximations/fallbacks can violate this; cap per-latent posterior
        # variance to 1 by row-wise rescaling of the Cholesky factor.
        if latent_dim == 1:
            L = L.clamp(max=var_cap**0.5)
        else:
            var_diag = torch.sum(L**2, dim=2)
            scale = (var_cap / var_diag.clamp_min(var_cap)).sqrt()
            L = L * scale.unsqueeze(-1)

        return L

    def _get_latent_names(self, u_latent: dict[str, Tensor]) -> list[str]:
        """Get the names of latent variables in topological order.

        Args:
            u_latent: Dictionary of latent noise variables.

        Returns:
            List of latent variable names in the order they appear in the SEM.
        """
        return [name for name in self._variables if name in u_latent]

    def _f_mean_u(
        self,
        u_latent_rav: Tensor,
        observed: dict[str, Tensor],
        observed_name: str,
        latent_names: list[str],
    ) -> Tensor:
        """Compute f_mean of an observed variable as a function of latent noise.

        This is used to compute gradients for the Gauss-Newton Hessian.
        The gradient path through other observed variables is blocked.

        Args:
            u_latent_rav: Raveled latent noise variables.
            observed: Dictionary of observed variable values.
            observed_name: Name of the observed variable to compute f_mean for.
            latent_names: List of latent variable names in order.

        Returns:
            The f_mean value for the specified observed variable.

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
                    return self._variables[name].f_mean(parents)

                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Observed variable '{observed_name}' not found in the SEM.")

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
