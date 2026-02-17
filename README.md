# inga

`inga` is a Python toolkit for **structural causal models (SCMs)** focused on causal simulation, inference, and model benchmarking.

## Purpose

This repository is designed to help you:

- define and sample from SCMs,
- generate synthetic datasets with causal annotations,
- estimate causal quantities (causal effects and causal bias),
- inspect model behavior with interactive HTML explorers,
- benchmark causal-consistency training against standard baselines.

In short: it is a practical playground for building and testing causal ML workflows end-to-end.

## Installation

### With `uv` (recommended)

```bash
uv sync
```

Run scripts with:

```bash
uv run python -m examples.explore
```

### With `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
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
uv run python -m examples.explore --output plots/scm_explorer.html
```

This generates an HTML page plus assets in `plots/` for visual causal analysis.

### 4) Train causal-consistency models that beat baselines

The benchmark scripts compare:

1. standard MLP baseline,
2. L2-regularized MLP,
3. causal-consistency multi-head models predicting outcome + causal quantities.

Use:

- `examples/causal_training_benchmark.py` for multi-seed evaluation,
- `examples/causal_transfer_benchmark.py` for transfer across unseen SCM datasets.

These scripts are built to demonstrate regimes where causal-consistency objectives outperform conventional baselines on causal metrics.