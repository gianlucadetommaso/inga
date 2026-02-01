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
            f_bar += self._coefs[parent_name] * parent ** 2
            
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
        u_latent = nn.ParameterDict({name: torch.randn(1) for name in self._variables if name not in observed})
        
        optimizer = torch.optim.LBFGS(u_latent.values(), lr=0.01, max_iter=100)
        
        size = len(list(observed.values())[0])
        
        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = self._loss_fn(self._expand_dict(u_latent, size), observed)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return dict(u_latent)
    
    @no_grad
    def compute_gn_hessian(self, u_latent: dict[str, Tensor], observed: dict[str, Tensor]) -> None:
        gn_hessians = {name: 1 / variable.sigma ** 2 for name, variable in self._variables.items() if name not in observed}
        
        for observed_name in observed:
            for latent_name in gn_hessians:
                f_bar_wrt_u = partial(
                    self._f_bar_wrt_u,
                    observed_name=observed_name, 
                    latent_name=latent_name, 
                    u_latent=u_latent,
                )
                
                g = vmap(lambda o: grad(partial(f_bar_wrt_u, observed=o))(u_latent[latent_name][0]))(observed)
                gn_hessians[latent_name] += torch.outer(g, g)
                
        return gn_hessians
    
    
    def _f_bar_wrt_u(self, u: Tensor, observed_name: str, latent_name: str, u_latent: dict[str, Tensor], observed: dict[str, Tensor]) -> Tensor:
        values = {}
        for name, variable in self._variables.items():
            parents = {parent_name: values[parent_name] for parent_name in variable.parent_names}
            
            if name in observed:
                if name == observed_name:
                    return self._variables[observed_name].f_bar(parents)
                    
                values[name] = observed[name].clone()
                
            elif name == latent_name:
                values[name] = variable.f(parents, u)
                
            else:
                values[name] = variable.f(parents, u_latent[name])
                
    @staticmethod
    def _expand_dict(dct: dict[str, Tensor], size: int) -> Tensor:
        return {name: value.expand(size) for name, value in dct.items()}
                
        
sem = SEM(
    variables=[
        LinearVariable(name="Z", parent_names=[], sigma=1., coefs={}, intercept=0.),
        LinearVariable(name="X", parent_names=["Z"], sigma=1., coefs={"Z": 1.}, intercept=0.),
        LinearVariable(name="Y", parent_names=["X", "Z"], sigma=1., coefs={"X": 2., "Z": 3.}, intercept=0.)
    ]
)

values = sem.generate(10)
observed = {"X": values["X"], "Y": values["Y"]}

maps = sem.fit_map(observed)
gn_hessians = sem.compute_gn_hessian(maps, observed)
