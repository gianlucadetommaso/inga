# Inga 因果

`inga` is a Python toolkit for **structural causal models (SCMs)** focused on building, simulating, and analyzing causal systems.

![Flow X to Y](plots/posterior_explorer_assets/flow_X_to_Y.gif)

## Purpose

This repository provides a practical workflow for causal modeling experiments. You can use it to:

- define SCMs from explicit structural equations,
- simulate observational/posterior predictive samples,
- compute causal quantities such as causal effect and causal bias,
- export visual diagnostics (DAGs, animations, and interactive HTML explorers),
- benchmark causal-consistency training against standard baselines.

In short, `inga` helps you go from **SCM specification** to **causal analysis and benchmarking** in one place.

## Installation

`inga` is currently intended to be installed from GitHub (not from PyPI).

### 1) Clone the repository

```bash
git clone https://github.com/gianlucadetommaso/steindag.git
cd steindag
```

### 2) Install and manage dependencies with `uv` (recommended)

```bash
uv sync
```

To run an example script, execute

```bash
uv run python -m examples.explore
```

## What this repo can do

### 1) Generate synthetic datasets from SCMs

Use random SCM generation and dataset tooling to create supervised/causal training corpora with query annotations.

- API: `inga.scm.random`, `inga.scm.dataset`
- Example: `examples/causal_training_benchmark.py`, `examples/causal_transfer_benchmark.py`

### 2) Compute causal effects and causal bias

Each SCM supports posterior-based estimation of:

- `causal_effect(...)`
- `causal_bias(...)`

These are available directly from `SCM` objects and are used throughout tests and benchmarks.

### 3) Explore datasets and posteriors in HTML

You can export an interactive explorer with sliders, posterior predictive histograms, DAG visuals, and causal plots.

```bash
uv run python -m examples.explore --output plots/explorer.html
```

This generates an HTML page plus assets in `plots/` for visual causal analysis.

Generated example included in this repo:

- Interactive HTML: [`plots/explorer.html`](plots/explorer.html)
- Assets folder: [`plots/explorer_assets/`](plots/explorer_assets/)

Preview from the generated page (click to open the interactive HTML):

[![Posterior explorer preview](plots/explorer_assets/dag.png)](plots/explorer.html)

> Note: GitHub README renders a static preview image; the full interactivity is available when opening `plots/explorer.html` in a browser.

### 4) Train causal-consistency models that beat baselines

The benchmark scripts compare:

1. standard MLP baseline,
2. L2-regularized MLP,
3. causal-consistency multi-head models predicting outcome + causal quantities.

Use:

- `examples/causal_training_benchmark.py` for multi-seed evaluation,
- `examples/causal_transfer_benchmark.py` for transfer across unseen SCM datasets.

These scripts are built to demonstrate regimes where causal-consistency objectives outperform conventional baselines on causal metrics.