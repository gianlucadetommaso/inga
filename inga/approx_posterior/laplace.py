"""Laplace approximation for posterior inference in SEMs.

This module provides a Laplace (Gaussian) approximation to the posterior
distribution over latent noise variables given observations.
"""

import torch
from torch import Tensor, no_grad, vmap, nn
from torch.func import grad, jacrev
from dataclasses import dataclass
from functools import partial
from typing import Mapping
from inga.scm.variable.base import Variable
from inga.scm.variable.gaussian import GaussianVariable
from inga.scm.variable.categorical import CategoricalVariable


@dataclass
class LaplacePosteriorState:
    """State of a fitted Laplace posterior.

    Attributes:
        MAP_components_rav: MAP means of Gaussian mixture components.
            Shape: (batch_size, num_components, latent_dim).
        L_cov_components_rav: Cholesky factors of component covariances.
            Shape: (batch_size, num_components, latent_dim, latent_dim).
        component_log_weights: Log-weights of mixture components.
            Shape: (batch_size, num_components). Rows are log-softmax normalized.
        latent_names: Names of latent variables in order.
    """

    MAP_components_rav: Tensor
    L_cov_components_rav: Tensor
    component_log_weights: Tensor
    latent_names: list[str]


class LaplacePosterior:
    """Laplace approximation to the posterior distribution.

    This class computes a Gaussian approximation to the posterior distribution
    over latent noise variables in a SCM. The approximation is centered at the
    MAP estimate with covariance given by the inverse Gauss-Newton Hessian.

    Attributes:
        _variables: Dictionary mapping variable names to Variable objects.
        _state: Current fitted state, or None if not yet fitted.
    """

    def __init__(
        self,
        variables: dict[str, Variable],
        *,
        continuation_scales: tuple[float, ...] = (3.0, 1.5, 1.0),
        continuation_steps: int = 40,
        num_map_restarts: int = 3,
        restart_init_scales: tuple[float, ...] = (0.0, 0.5, 1.0),
        num_mixture_components: int = 1,
        adam_lr: float = 5e-2,
        adam_scheduler_gamma: float | None = None,
        lbfgs_lr: float = 1.0,
        lbfgs_max_iter: int = 100,
        lbfgs_line_search_fn: str | None = "strong_wolfe",
        latent_trust_radii: tuple[float, ...] = (2.0, 4.0, 8.0),
        jacobian_norm_cap: float = 1e3,
    ) -> None:
        """Initialize the Laplace posterior.

        Args:
            variables: Dictionary mapping variable names to Variable objects.
            continuation_scales: Observation-noise scales for continuation warm starts.
            continuation_steps: Number of Adam steps per continuation scale.
            num_map_restarts: Number of MAP restart candidates.
            restart_init_scales: Std scales for restart initializations.
            num_mixture_components: Number of posterior mixture components kept.
            adam_lr: Adam learning rate for continuation optimization.
            adam_scheduler_gamma: Optional exponential LR decay factor for Adam.
                If None, no scheduler is used.
            lbfgs_lr: L-BFGS learning rate for final refinement.
            lbfgs_max_iter: Maximum number of L-BFGS iterations.
            lbfgs_line_search_fn: L-BFGS line search function.
            latent_trust_radii: Trust-region radii used to parameterize latent
                optimization as ``u = r * tanh(z)`` across continuation stages.
            jacobian_norm_cap: Smooth cap for per-row Jacobian norm used in
                Gauss-Newton standardization. Larger values recover vanilla GN.
        """
        self._variables = variables
        self._state: LaplacePosteriorState | None = None
        self._adam_scheduler_gamma: float | None = None
        self._lbfgs_line_search_fn: str | None = "strong_wolfe"
        self.configure_map_options(
            continuation_scales=continuation_scales,
            continuation_steps=continuation_steps,
            num_map_restarts=num_map_restarts,
            restart_init_scales=restart_init_scales,
            num_mixture_components=num_mixture_components,
            adam_lr=adam_lr,
            adam_scheduler_gamma=adam_scheduler_gamma,
            lbfgs_lr=lbfgs_lr,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_line_search_fn=lbfgs_line_search_fn,
            latent_trust_radii=latent_trust_radii,
            jacobian_norm_cap=jacobian_norm_cap,
        )

    def configure_map_options(
        self,
        *,
        continuation_scales: tuple[float, ...] | None = None,
        continuation_steps: int | None = None,
        num_map_restarts: int | None = None,
        restart_init_scales: tuple[float, ...] | None = None,
        num_mixture_components: int | None = None,
        adam_lr: float | None = None,
        adam_scheduler_gamma: float | None = None,
        lbfgs_lr: float | None = None,
        lbfgs_max_iter: int | None = None,
        lbfgs_line_search_fn: str | None = None,
        latent_trust_radii: tuple[float, ...] | None = None,
        jacobian_norm_cap: float | None = None,
    ) -> None:
        """Configure robust MAP and optimizer options.

        This method allows users to adjust continuation, multi-start, mixture,
        and optimizer behavior without modifying private attributes.
        """
        if continuation_scales is not None:
            if len(continuation_scales) == 0 or any(
                scale <= 0 for scale in continuation_scales
            ):
                raise ValueError(
                    "`continuation_scales` must be non-empty and all values > 0."
                )
            self._continuation_scales = continuation_scales

        if continuation_steps is not None:
            if continuation_steps <= 0:
                raise ValueError("`continuation_steps` must be > 0.")
            self._continuation_steps = continuation_steps

        if num_map_restarts is not None:
            if num_map_restarts <= 0:
                raise ValueError("`num_map_restarts` must be > 0.")
            self._num_map_restarts = num_map_restarts

        if restart_init_scales is not None:
            if len(restart_init_scales) == 0 or any(
                scale < 0 for scale in restart_init_scales
            ):
                raise ValueError(
                    "`restart_init_scales` must be non-empty and all values >= 0."
                )
            self._restart_init_scales = restart_init_scales

        if num_mixture_components is not None:
            if num_mixture_components <= 0:
                raise ValueError("`num_mixture_components` must be > 0.")
            self._num_mixture_components = num_mixture_components

        if adam_lr is not None:
            if adam_lr <= 0:
                raise ValueError("`adam_lr` must be > 0.")
            self._adam_lr = adam_lr

        if adam_scheduler_gamma is not None:
            if not (0 < adam_scheduler_gamma <= 1):
                raise ValueError("`adam_scheduler_gamma` must be in (0, 1].")
            self._adam_scheduler_gamma = adam_scheduler_gamma
        elif not hasattr(self, "_adam_scheduler_gamma"):
            self._adam_scheduler_gamma = None

        if lbfgs_lr is not None:
            if lbfgs_lr <= 0:
                raise ValueError("`lbfgs_lr` must be > 0.")
            self._lbfgs_lr = lbfgs_lr

        if lbfgs_max_iter is not None:
            if lbfgs_max_iter <= 0:
                raise ValueError("`lbfgs_max_iter` must be > 0.")
            self._lbfgs_max_iter = lbfgs_max_iter

        if lbfgs_line_search_fn is not None:
            if lbfgs_line_search_fn not in ("strong_wolfe",):
                raise ValueError("`lbfgs_line_search_fn` must be 'strong_wolfe'.")
            self._lbfgs_line_search_fn = lbfgs_line_search_fn
        elif not hasattr(self, "_lbfgs_line_search_fn"):
            self._lbfgs_line_search_fn = "strong_wolfe"

        if latent_trust_radii is not None:
            if len(latent_trust_radii) == 0 or any(r <= 0 for r in latent_trust_radii):
                raise ValueError(
                    "`latent_trust_radii` must be non-empty and all values > 0."
                )
            self._latent_trust_radii = latent_trust_radii

        if jacobian_norm_cap is not None:
            if jacobian_norm_cap <= 0:
                raise ValueError("`jacobian_norm_cap` must be > 0.")
            self._jacobian_norm_cap = jacobian_norm_cap

    def fit(self, observed: dict[str, Tensor]) -> None:
        """Fit the Laplace posterior to observed data.

        Computes the MAP estimate and the Cholesky factor of the approximate
        covariance matrix.

        Args:
            observed: Dictionary of observed variable values.
        """
        if not observed:
            raise ValueError("`observed` must contain at least one observed variable.")

        reference = next(iter(observed.values()))
        num_samples = len(reference)
        latent_names = [name for name in self._variables if name not in observed]

        if len(latent_names) == 0:
            self.state = LaplacePosteriorState(
                MAP_components_rav=torch.empty(
                    (num_samples, 1, 0), device=reference.device, dtype=reference.dtype
                ),
                L_cov_components_rav=torch.empty(
                    (num_samples, 1, 0, 0),
                    device=reference.device,
                    dtype=reference.dtype,
                ),
                component_log_weights=torch.zeros(
                    (num_samples, 1), device=reference.device, dtype=reference.dtype
                ),
                latent_names=[],
            )
            return

        candidates, candidate_losses = self._fit_map_candidates(observed, latent_names)

        maps_by_restart = torch.stack(
            [self._ravel(candidate) for candidate in candidates], dim=0
        )
        losses_by_restart = torch.stack(candidate_losses, dim=0)

        maps_by_row = maps_by_restart.permute(1, 0, 2)
        losses_by_row = losses_by_restart.transpose(0, 1)

        num_components = min(self._num_mixture_components, len(candidates))
        num_components = max(1, num_components)

        safe_losses_by_row = torch.where(
            torch.isfinite(losses_by_row),
            losses_by_row,
            torch.full_like(losses_by_row, float("inf")),
        )
        selected_idx = torch.topk(-safe_losses_by_row, k=num_components, dim=1).indices
        selected_losses = torch.gather(safe_losses_by_row, dim=1, index=selected_idx)

        latent_dim = maps_by_row.shape[-1]
        selected_idx_expanded = selected_idx.unsqueeze(-1).expand(-1, -1, latent_dim)
        MAP_components_rav = torch.gather(
            maps_by_row, dim=1, index=selected_idx_expanded
        )

        L_cov_components = [
            self._approx_cov_chol(MAP_components_rav[:, i], observed, latent_names)
            for i in range(num_components)
        ]
        L_cov_components_rav = torch.stack(L_cov_components, dim=1)
        component_log_weights = torch.log_softmax(-selected_losses, dim=1)

        self.state = LaplacePosteriorState(
            MAP_components_rav=MAP_components_rav,
            L_cov_components_rav=L_cov_components_rav,
            component_log_weights=component_log_weights,
            latent_names=latent_names,
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
        if len(state.latent_names) == 0:
            return {}

        batch_size, _, latent_dim = state.MAP_components_rav.shape
        component_idx = (
            torch.distributions.Categorical(logits=state.component_log_weights)
            .sample((num_samples,))
            .transpose(0, 1)
        )

        batch_idx = torch.arange(
            batch_size, device=state.MAP_components_rav.device
        ).unsqueeze(-1)
        means = state.MAP_components_rav[batch_idx, component_idx]
        chols = state.L_cov_components_rav[batch_idx, component_idx]
        eps = torch.randn(
            (batch_size, num_samples, latent_dim, 1),
            device=state.MAP_components_rav.device,
            dtype=state.MAP_components_rav.dtype,
        )
        samples_rav = (means + (chols @ eps).squeeze(-1)).permute(0, 2, 1)

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

    def _fit_map(
        self, observed: dict[str, Tensor], latent_names: list[str]
    ) -> dict[str, Tensor]:
        """Find the MAP estimate of latent variables given observations.

        Uses a robust two-part strategy:
        1) continuation warm-start on progressively less-smoothed objectives,
        2) second-order refinement (L-BFGS) on the true objective,
        repeated for a small set of initializations (multi-start), with
        per-observation best-candidate selection.

        Args:
            observed: Dictionary of observed variable values.
            latent_names: Names of latent variables in optimization order.

        Returns:
            Dictionary mapping latent variable names to their MAP noise estimates.
        """
        reference = next(iter(observed.values()))
        size = len(reference)
        candidates, candidate_losses = self._fit_map_candidates(observed, latent_names)

        stacked_losses = torch.stack(candidate_losses, dim=0)  # (restarts, batch)
        safe_losses = torch.where(
            torch.isfinite(stacked_losses),
            stacked_losses,
            torch.full_like(stacked_losses, float("inf")),
        )
        best_restart_idx = safe_losses.argmin(dim=0)
        row_idx = torch.arange(size, device=reference.device)

        return {
            name: torch.stack([candidate[name] for candidate in candidates], dim=0)[
                best_restart_idx, row_idx
            ]
            for name in latent_names
        }

    def _fit_map_candidates(
        self, observed: dict[str, Tensor], latent_names: list[str]
    ) -> tuple[list[dict[str, Tensor]], list[Tensor]]:
        """Generate MAP candidates and their per-sample objective values.

        Args:
            observed: Dictionary of observed variable values.
            latent_names: Names of latent variables in optimization order.

        Returns:
            Tuple ``(candidates, losses)`` where:
            - candidates is a list of latent dictionaries (one per restart),
            - losses is a list of per-sample objective tensors, aligned with
              candidates.
        """
        reference = next(iter(observed.values()))
        size = len(reference)

        candidates: list[dict[str, Tensor]] = []
        candidate_losses: list[Tensor] = []
        for i in range(self._num_map_restarts):
            init_scale = self._restart_init_scales[
                min(i, len(self._restart_init_scales) - 1)
            ]
            u_candidate = self._optimize_map_candidate(
                observed=observed,
                latent_names=latent_names,
                size=size,
                reference=reference,
                init_scale=init_scale,
            )
            candidates.append(u_candidate)
            observed_for_loss = {
                name: value.to(dtype=next(iter(u_candidate.values())).dtype)
                for name, value in observed.items()
            }
            candidate_losses.append(
                self._posterior_loss_fn(u_candidate, observed_for_loss)
            )

        return candidates, candidate_losses

    def _optimize_map_candidate(
        self,
        observed: dict[str, Tensor],
        latent_names: list[str],
        size: int,
        reference: Tensor,
        init_scale: float,
    ) -> dict[str, Tensor]:
        """Optimize one MAP candidate using continuation + L-BFGS refinement.

        Args:
            observed: Batched observed values.
            latent_names: Names of latent variables in optimization order.
            size: Batch size.
            reference: A representative observed tensor to inherit dtype/device.
            init_scale: Std scale of random latent initialization.

        Returns:
            Candidate MAP latent dictionary for all observations in the batch.
        """
        initial_radius = self._latent_trust_radii[0]
        opt_dtype = torch.float64
        observed_opt = {
            name: value.to(dtype=opt_dtype) for name, value in observed.items()
        }
        if init_scale == 0.0:
            z_init = {
                name: torch.zeros(size, device=reference.device, dtype=opt_dtype)
                for name in latent_names
            }
        else:
            z_init = {
                name: (init_scale / initial_radius)
                * torch.randn(size, device=reference.device, dtype=opt_dtype)
                for name in latent_names
            }

        z_latent = nn.ParameterDict(
            {name: nn.Parameter(value) for name, value in z_init.items()}
        )

        def bounded_u(radius: float) -> dict[str, Tensor]:
            return {name: radius * torch.tanh(z) for name, z in z_latent.items()}

        adam = torch.optim.Adam(list(z_latent.values()), lr=self._adam_lr)
        scheduler = None
        if self._adam_scheduler_gamma is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                adam, gamma=self._adam_scheduler_gamma
            )
        for idx, obs_noise_scale in enumerate(self._continuation_scales):
            radius = self._latent_trust_radii[
                min(idx, len(self._latent_trust_radii) - 1)
            ]
            for _ in range(self._continuation_steps):
                adam.zero_grad()
                loss = self._posterior_loss_fn(
                    bounded_u(radius), observed_opt, obs_noise_scale=obs_noise_scale
                ).mean()
                loss.backward()
                adam.step()
                if scheduler is not None:
                    scheduler.step()

        optimizer = torch.optim.LBFGS(
            list(z_latent.values()),
            lr=self._lbfgs_lr,
            max_iter=self._lbfgs_max_iter,
            line_search_fn=self._lbfgs_line_search_fn,
        )
        final_radius = self._latent_trust_radii[-1]
        continuation_u = {
            name: (final_radius * torch.tanh(z)).detach().clone()
            for name, z in z_latent.items()
        }

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = self._posterior_loss_fn(
                bounded_u(final_radius), observed_opt, obs_noise_scale=1.0
            ).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        refined_u = {
            name: (final_radius * torch.tanh(z)).detach().clone()
            for name, z in z_latent.items()
        }

        continuation_loss = self._posterior_loss_fn(continuation_u, observed_opt)
        refined_loss = self._posterior_loss_fn(refined_u, observed_opt)
        use_refined = torch.isfinite(refined_loss) & (
            (refined_loss <= continuation_loss) | ~torch.isfinite(continuation_loss)
        )

        return {
            name: torch.where(use_refined, refined_u[name], continuation_u[name]).to(
                dtype=reference.dtype
            )
            for name in latent_names
        }

    def _posterior_loss_fn(
        self,
        u_latent: Mapping[str, Tensor],
        observed: dict[str, Tensor],
        obs_noise_scale: float = 1.0,
    ) -> Tensor:
        """Compute per-sample negative log posterior values.

        This function always returns one loss per observation row. The caller
        can aggregate (e.g. mean) for optimization, while preserving the
        per-sample objective needed for batched multi-start selection.

        Args:
            u_latent: Dictionary of latent noise values/parameters.
            observed: Dictionary of observed variable values.
            obs_noise_scale: Multiplicative scale on observed-noise terms used
                during continuation warm-start (1.0 = true objective).

        Returns:
            Tensor of shape (batch_size,) with per-sample posterior losses.
        """
        values: dict[str, Tensor] = {}
        reference = next(iter(observed.values()))
        loss = torch.zeros(
            len(reference),
            device=reference.device,
            dtype=reference.dtype,
        )

        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }

            if name in observed:
                values[name] = observed[name]
                loss = loss - variable.log_pdf(
                    parents=parents,
                    observed=observed[name],
                    noise_scale=obs_noise_scale,
                )
                continue

            else:
                values[name] = variable.f(parents, u_latent[name])
                u = u_latent[name]

            u_sq = u**2
            while u_sq.ndim > 1:
                u_sq = u_sq.sum(dim=-1)
            loss = loss + 0.5 * u_sq

        return loss

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
        unsupported = [
            name
            for name, variable in self._variables.items()
            if not isinstance(variable, (GaussianVariable, CategoricalVariable))
        ]
        if unsupported:
            raise ValueError(
                "Hessian approximation is currently supported only for "
                "GaussianVariable/CategoricalVariable nodes. "
                f"Found unsupported variables: {sorted(unsupported)}."
            )

        latent_dim = len(latent_names)
        if u_latent_rav.shape[1] != latent_dim:
            raise ValueError(
                f"`u_latent_rav.shape[1]` and `len(latent_names)` must match, "
                f"but {u_latent_rav.shape[1]} != {latent_dim}."
            )

        num_samples = len(u_latent_rav)
        device = u_latent_rav.device
        out_dtype = u_latent_rav.dtype
        hessian_dtype = torch.float64

        u_latent_rav_h = u_latent_rav.to(dtype=hessian_dtype)
        observed_h = {
            name: value.to(dtype=hessian_dtype) for name, value in observed.items()
        }

        gn_hessian_rav = torch.zeros(
            (num_samples, latent_dim, latent_dim), device=device, dtype=hessian_dtype
        )
        latent_diag: list[float] = []
        for name in latent_names:
            sigma = self._variables[name].sigma
            latent_diag.append(1.0 if sigma is None else 1 / sigma**2)
        gn_hessian_rav[:, range(latent_dim), range(latent_dim)] = torch.tensor(
            latent_diag,
            device=device,
            dtype=hessian_dtype,
        )

        for observed_name in observed_h:
            observed_variable = self._variables[observed_name]

            if isinstance(observed_variable, CategoricalVariable):
                f_logits_wrt_u = partial(
                    self._f_logits_u,
                    observed_name=observed_name,
                    latent_names=latent_names,
                )

                logits = vmap(lambda u, o: f_logits_wrt_u(u, observed=o))(
                    u_latent_rav_h,
                    observed_h,
                ).to(dtype=hessian_dtype)
                jacobian = vmap(
                    lambda u, o: jacrev(partial(f_logits_wrt_u, observed=o))(u)
                )(u_latent_rav_h, observed_h).to(dtype=hessian_dtype)

                probs = torch.softmax(logits, dim=-1)
                fisher = torch.diag_embed(probs) - probs[:, :, None] * probs[:, None, :]
                gn_hessian_rav += torch.einsum(
                    "nci,ncd,ndj->nij", jacobian, fisher, jacobian
                )
                continue

            f_mean_wrt_u = partial(
                self._f_mean_u,
                observed_name=observed_name,
                latent_names=latent_names,
            )
            g = vmap(lambda u, o: grad(partial(f_mean_wrt_u, observed=o))(u))(
                u_latent_rav_h, observed_h
            )

            # Smooth Jacobian standardization (robust influence):
            # g_std = g / sqrt(1 + ||g||^2 / c^2)
            # This is mathematically equivalent to using an adaptive local
            # observation scale sigma_eff = sigma * sqrt(1 + ||g||^2 / c^2).
            g_norm = torch.linalg.vector_norm(g, dim=1)
            scale = torch.sqrt(1.0 + (g_norm / self._jacobian_norm_cap) ** 2)
            g = g / scale[:, None]

            sigma_obs = observed_variable.sigma
            obs_scale = 1.0 if sigma_obs is None else sigma_obs**2
            gn_hessian_rav += g[:, None] * g[:, :, None] / obs_scale

        # Enforce exact symmetry before Cholesky to avoid numerical skew.
        gn_hessian_rav = 0.5 * (gn_hessian_rav + gn_hessian_rav.transpose(-1, -2))

        Linv = torch.linalg.cholesky(gn_hessian_rav)

        eye = torch.eye(
            latent_dim, requires_grad=False, device=device, dtype=hessian_dtype
        )

        L = torch.linalg.solve_triangular(
            Linv,
            eye,
            upper=False,
        )

        return L.to(dtype=out_dtype)

    def _get_latent_names(self, u_latent: dict[str, Tensor]) -> list[str]:
        """Get the names of latent variables in topological order.

        Args:
            u_latent: Dictionary of latent noise variables.

        Returns:
            List of latent variable names in the order they appear in the SCM.
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
            ValueError: If observed_name is not found in the SCM.
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
                    value = self._variables[name].f_mean(parents)
                    return value if value.ndim == 0 else value.sum()

                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Observed variable '{observed_name}' not found in the SCM.")

    def _f_logits_u(
        self,
        u_latent_rav: Tensor,
        observed: dict[str, Tensor],
        observed_name: str,
        latent_names: list[str],
    ) -> Tensor:
        """Compute logits of a categorical observed variable as function of latent noise."""
        u_latent = self._unravel(u_latent_rav, latent_names)
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                parent_name: values[parent_name]
                for parent_name in variable.parent_names
            }

            if name in observed:
                if name == observed_name:
                    if not isinstance(variable, CategoricalVariable):
                        raise ValueError(
                            f"Observed variable '{observed_name}' is not categorical."
                        )
                    return variable.f_logits(parents)

                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Observed variable '{observed_name}' not found in the SCM.")

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
