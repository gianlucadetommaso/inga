# Inga 因果

`inga` is a toolkit for generating and inspecting synthetic tabular datasets. It constructs arbitrarily complex Structural Causal Models (SCMs), draws samples from them, and computes causal effects and causal biases conditioned on observed variables and outcomes. All computed quantities are stored and made available for causally-informed pre-training of tabular models.

![Flow X to Y](plots/explorer_assets/flow_X_to_Y.gif)

## Causal Effect and Causal Bias

The current scope of this repository is restricted to SCMs with continuous variables. Let $V_i$ denote a generic scalar variable in the SCM, and let $U_{V_i} \sim \mathcal{N}(0,1)$ be its corresponding exogenous noise, such that

$$
V_i := f_{V_i}(\mathrm{Pa}(V_i), U_{V_i})
     := \bar{f}_{V_i}(\mathrm{Pa}(V_i)) + \sigma_{V_i} U_{V_i}.
$$

Here, $\mathrm{Pa}(V_i)$ denotes the set of parents of $V_i$ in the DAG, $\bar{f}_{V_i}$ captures the deterministic structural component, and $\sigma_{V_i}$ controls the scale of the exogenous noise.

In particular, let $X$ denote a treatment variable, $Y$ an outcome, and $\mathcal{O}$ a set of observed variables. Under mild regularity assumptions [(Detommaso et al.)](https://arxiv.org/abs/2106.09762), the causal effect and causal bias for a given treatment value $x$ and observation vector $o$ are defined as

$$
\begin{aligned}
\mathcal{C}_X(x, o) 
&:= \mathbb{E}\big[\nabla_x f_Y^x \,\big|\, x, o\big], \\
\mathcal{B}_X(x, o) 
&:= -\sum_{V_i \in \{X\}\cup\mathcal{O}}
\frac{1}{\sigma_{V_i}}\mathbb{E}\Big[
\Big(
\nabla_{V_i} f_Y^{x,o}
- (f_Y^{x,o} - \mathbb{E}[Y \mid x, o])\, U_{V_i}
\Big)
\nabla_x (f_{V_i}^{x,o} - v_i)
\,\Big|\, x, o
\Big].
\end{aligned}
$$

Here, $f_{V_i}^{a}$ denotes the structural function $f_{V_i}$ under intervention $A=a$. All expectations are taken with respect to the posterior distribution $p(U \mid x, o)$, where $U$ is the vector of all exogenous noise variables.

`inga` approximates this posterior using a robust Laplace approximation, enabling scalable computation in high-dimensional settings and across batches of observations $(x, o)$.

One can show that the association between treatment $X$ and outcome $Y$ decomposes into causal effect and causal bias:

$$
\mathcal{A}_X(x, o)
:= \nabla_x \mathbb{E}[Y \mid x, o]
= \mathcal{C}_X(x, o) + \mathcal{B}_X(x, o).
$$

## Causally Consistent Pre-Training

Causal effect and causal bias provide a granular characterization of how information propagates from observed variables to the outcome within the DAG.

Standard point-estimation models aim to approximate the conditional expectation $\mathbb{E}[Y \mid x, o]$, but they do not distinguish between contributions arising from causal pathways and those arising from non-causal (e.g., confounding or purely statistical) dependencies. As a result, the underlying data-generating process is often unidentifiable, which can lead to suboptimal generalization and brittleness under distribution shift.


Consider an encoder model $z := h(o)$ and a prediction head $\hat{y}(z)$. Introduce two additional heads, $\hat{c}_j(z)$ and $\hat{b}_j(z)$, intended to learn the causal effect and causal bias from $O_j$ (treated as the treatment variable) to $Y$. We say that the model is **causally consistent** for $O_j$ if

$$
\begin{aligned}
\nabla_{o_j} \hat{y} &= \hat{c}_j + \hat{b}_j, \\
\hat{c}_j &= \mathcal{C}_{O_j}(o_j, o), \\
\hat{b}_j &= \mathcal{B}_{O_j}(o_j, o).
\end{aligned}
$$

`inga` enables causally consistent pre-training by generating synthetic datasets that include the full set of causal effects $\mathcal{C}_{O_j}(o_j, o)$ and causal biases $\mathcal{B}_{O_j}(o_j, o)$. These quantities can be incorporated directly into training objectives, encouraging models to learn representations that respect the causal structure of the data-generating process.

### A Small Benchmark
The small benchmark [causal_consistency_benchmark.py](examples/causal_consistency_benchmark.py) demonstrates this intution. A simple MLP encoder is attached to three linear heads, respectively predicting outcomes, causal effects and causal biases. The model is trained and tested individually on splits of 30 randomly generated synthetic dataset. 

```
+--------------------+----------------+-------------------+-------------------------+
| method_type        | prediction_mae | causal_effect_mae | prediction_win_fraction |
+--------------------+----------------+-------------------+-------------------------+
| standard           | 0.7909 [0.31]  | 0.3353 [0.45]     | 0.0667                  |
| l2                 | 0.7868 [0.31]  | 0.3141 [0.46]     | 0.0667                  |
| causal_consistency | 0.7694 [0.31]  | 0.0461 [0.21]     | 0.8667                  |
+--------------------+----------------+-------------------+-------------------------+
```

The table shows that not only the model trained using causal consistency provides much more reliable causal effect estimates, but also decreases the generalization error on ~87% of the datasets. Results can be replicated by running `uv run python examples/causal_consistency_benchmark.py`.

## How To:

### Install
Clone the repository:

```bash
git clone https://github.com/gianlucadetommaso/inga.git
cd inga
```

Sync dependencies:

```bash
uv sync
```

Run scripts, for example:

```bash
uv run python -m examples.explore
```

### Create a SCM
