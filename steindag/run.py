import torch
from torch import Tensor
from steindag.variable.linear import LinearVariable
from steindag.sem.base import SEM

        
sem = SEM(
    variables=[
        LinearVariable(name="Z", parent_names=[], sigma=1., coefs={}, intercept=0.),
        LinearVariable(name="X", parent_names=["Z"], sigma=1., coefs={"Z": 1.}, intercept=0.),
        LinearVariable(name="Y", parent_names=["X", "Z"], sigma=1., coefs={"X": 2., "Z": 3.}, intercept=0.)
    ]
)

alpha = sem._variables["X"]._coefs["Z"]
beta = sem._variables["Y"]._coefs["X"]
gamma = sem._variables["Y"]._coefs["Z"]

values = sem.generate(10)

# observe only X
observed = {"X": values["X"]}

maps = sem.fit_map(observed)
chols_cov = sem.approx_cov_chol(maps, observed)
maps_rav = sem._ravel(maps)

samples = sem.sample(maps_rav, chols_cov, 10000, sem._get_latent_names(maps))

posterior_means = alpha * observed["X"] / (1 + alpha ** 2)
posterior_vars = Tensor([1 / (1 + alpha ** 2)]).expand(10)

print(torch.abs(posterior_means - samples["Z"].mean(1)))
print(torch.abs(posterior_vars - samples["Z"].var(1)))

# observe X and Y
observed = {"X": values["X"], "Y": values["Y"]}

maps = sem.fit_map(observed)
chols_cov = sem.approx_cov_chol(maps, observed)
maps_rav = sem._ravel(maps)

samples = sem.sample(maps_rav, chols_cov, 10000, sem._get_latent_names(maps))

posterior_means = (gamma * (observed["Y"] - beta * observed["X"]) + alpha * observed["X"]) / (1 + alpha ** 2 + gamma ** 2)
posterior_vars = Tensor([1 / (1 + alpha ** 2 + gamma ** 2)]).expand(10)

print(torch.abs(posterior_means - samples["Z"].mean(1)))
print(torch.abs(posterior_vars - samples["Z"].var(1)))


