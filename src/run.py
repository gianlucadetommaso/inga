from torch import Tensor, nn, vmap, no_grad
from torch.func import grad
import torch
from abc import abstractmethod
from typing import Iterable
from functools import partial


class Variable:
    def __init__(self, name: str, parent_names: Iterable[str], sigma: float) -> None:
        self.name = name
        self.parent_names = parent_names
        self.sigma = sigma
    
    def f(self, parents: dict[str, Tensor], u: Tensor, f_bar: Tensor | None = None) -> Tensor:
        if f_bar is None:
            f_bar = self.f_bar(parents)
        return f_bar + self.sigma * u
    
    @abstractmethod
    def f_bar(self, parents: dict[str, Tensor]) -> Tensor: ...
    

class LinearVariable(Variable):
    def __init__(self, name: str, parent_names: Iterable[str], sigma: float, coefs: dict[str, float], intercept: float) -> None:
        super().__init__(name, parent_names, sigma)
        
        self._coefs = coefs
        self._intercept = intercept
        
    def f_bar(self, parents: dict[str, Tensor]) -> Tensor:
        f_bar = self._intercept
        for parent_name, parent in parents.items():
            f_bar += self._coefs[parent_name] * parent
            
        return f_bar
            
    
class SEM:
    def __init__(self, variables: list[Variable]) -> None:
        self._variables = {variable.name: variable for variable in variables}
        
    def generate(self, num_samples: int) -> None:
        values = {}
        
        for name, variable in self._variables.items():
            parents = {pa_name: parent for pa_name, parent in values.items() if pa_name in variable.parent_names}
            values[name] = variable.f(parents, torch.randn(num_samples))
            
        return values
    
    def _loss_fn(self, u_latent: dict[str, Tensor], observed: dict[str, Tensor]) -> dict[str, Tensor]:
        values = {}
        loss = 0.
        
        for name, variable in self._variables.items():
            parents = {pa_name: parent for pa_name, parent in values.items() if pa_name in variable.parent_names}
            f_bar = variable.f_bar(parents)
            
            if name in observed:
                values[name] = observed[name]
                u = observed[name] - f_bar
                
            else:
                values[name] = variable.f(parents, u_latent[name], f_bar)
                u = u_latent[name]
                
            loss = loss + 0.5 * torch.sum((u / variable.sigma) ** 2)
            
        return loss
    
    def fit_map(self, observed: dict[str, Tensor]) -> dict[str,]:
        size = len(list(observed.values())[0])
        u_latent = nn.ParameterDict({name: torch.randn(size) for name in self._variables if name not in observed})
        
        optimizer = torch.optim.LBFGS(u_latent.values(), lr=0.01, max_iter=100)
        
        size = len(list(observed.values())[0])
        
        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = self._loss_fn(u_latent, observed)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return {name: u.detach().clone() for name, u in u_latent.items()}
    
    @no_grad
    def approx_cov_chol(self, u_latent: dict[str, Tensor], observed: dict[str, Tensor]) -> None:
        size = len(u_latent)
        num_samples = len(list(u_latent.values())[0])
        device = list(u_latent.values())[0].device
        
        latent_names = self._get_latent_names(u_latent)
        u_latent_rav = self._ravel(u_latent)
        
        gn_hessian_rav = torch.zeros((num_samples, size, size), device=device)
        gn_hessian_rav[:, range(size), range(size)] = Tensor([1 / self._variables[name].sigma ** 2 for name in latent_names])
        
        for observed_name in observed:
            f_bar_wrt_u = partial(
                self._f_bar_wrt_u,
                observed_name=observed_name,
                latent_names=latent_names
            )
                
            g = vmap(lambda u, o: grad(partial(f_bar_wrt_u, observed=o))(u))(u_latent_rav, observed)
            gn_hessian_rav += g[:, None] * g[:, :, None]
            
        if size == 1:
            L = 1 / gn_hessian_rav.sqrt()
        else:
            Linv = torch.linalg.cholesky(gn_hessian_rav, upper=True)
            L = torch.linalg.solve_triangular(Linv, torch.eye(size, requires_grad=False, device=device), upper=False)
                
        return L
    
    @no_grad
    def sample(self, map_rav: Tensor, chol_cov: Tensor, num_samples: int, latent_names: list[str]) -> dict[str, Tensor]:
        samples_rav = map_rav.unsqueeze(-1) + chol_cov @ torch.randn(size=(map_rav.shape[1], num_samples))
        return vmap(lambda s: self._unravel(s, latent_names), in_dims=2, out_dims=1)(samples_rav)
        
    def _get_latent_names(self, u_latent: dict[str, Tensor]) -> Tensor:
        return [name for name in self._variables if name in u_latent]
    
    def _f_bar_wrt_u(self, u_latent_rav: Tensor, observed: dict[str, Tensor], observed_name: str, latent_names: list[str]) -> Tensor:
        u_latent = self._unravel(u_latent_rav, latent_names)
        values = {}
        
        for name, variable in self._variables.items():
            parents = {parent_name: values[parent_name] for parent_name in variable.parent_names}
            
            if name in observed:
                if name == observed_name:
                    return self._variables[name].f_bar(parents)
                
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])
    

    def _ravel(self, u_latent: dict[str, Tensor]) -> Tensor:
        return torch.stack(list(u_latent.values()), dim=1)
                
    def _unravel(self, u_latent_rav: Tensor, latent_names: list[str]) -> dict[str, Tensor]:
        return {name: u_latent_rav.select(dim=-1, index=i) for i, name in enumerate(latent_names)}
                
        
sem = SEM(
    variables=[
        LinearVariable(name="Z", parent_names=[], sigma=1., coefs={}, intercept=0.),
        LinearVariable(name="X", parent_names=["Z"], sigma=1., coefs={"Z": 1.}, intercept=0.),
        LinearVariable(name="Y", parent_names=["X", "Z"], sigma=1., coefs={"X": 2., "Z": 3.}, intercept=0.)
    ]
)

values = sem.generate(10)
observed = {"X": values["X"]}

maps = sem.fit_map(observed)
chols_cov = sem.approx_cov_chol(maps, observed)
maps_rav = sem._ravel(maps)

samples = sem.sample(maps_rav, chols_cov, 1000, sem._get_latent_names(maps))
