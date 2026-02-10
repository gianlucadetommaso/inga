"""Benchmark and compare the two causal methods over many random seeds."""

from __future__ import annotations

import argparse
import statistics
import math

from steindag.sem.demo_compare_causal_methods import run_demo


def _summary(values: list[float]) -> tuple[float, float, float]:
    finite_vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    if not finite_vals:
        return float("nan"), float("nan"), float("nan")
    return (
        statistics.mean(finite_vals),
        statistics.median(finite_vals),
        statistics.pstdev(finite_vals),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    base_pred: list[float] = []
    base_eff: list[float] = []
    base_bias: list[float] = []
    base_cons: list[float] = []

    m1_pred: list[float] = []
    m1_eff: list[float] = []
    m1_bias: list[float] = []
    m1_cons: list[float] = []

    m2_pred: list[float] = []
    m2_eff: list[float] = []
    m2_bias: list[float] = []
    m2_cons: list[float] = []

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        out = run_demo(seed=seed, train_epochs=args.epochs, verbose=False)
        base_pred.append(out["baseline_pred_mae"])
        base_eff.append(out["baseline_effect_mae"])
        base_bias.append(out["baseline_bias_mae"])
        base_cons.append(out["baseline_consistency_mae"])

        m1_pred.append(out["method1_pred_mae"])
        m1_eff.append(out["method1_effect_mae"])
        m1_bias.append(out["method1_bias_mae"])
        m1_cons.append(out["method1_consistency_mae"])

        m2_pred.append(out["method2_pred_mae"])
        m2_eff.append(out["method2_effect_mae"])
        m2_bias.append(out["method2_bias_mae"])
        m2_cons.append(out["method2_consistency_mae"])

        print(
            f"seed={seed} | base_pred={out['baseline_pred_mae']:.4f} "
            f"m1_pred={out['method1_pred_mae']:.4f} m2_pred={out['method2_pred_mae']:.4f} "
            f"base_eff={out['baseline_effect_mae']:.4f} "
            f"m1_eff={out['method1_effect_mae']:.4f} m2_eff={out['method2_effect_mae']:.4f} "
            f"base_bias={out['baseline_bias_mae']:.4f} "
            f"m1_bias={out['method1_bias_mae']:.4f} m2_bias={out['method2_bias_mae']:.4f}",
            flush=True,
        )

    base_pred_mean, base_pred_med, base_pred_std = _summary(base_pred)
    base_eff_mean, base_eff_med, base_eff_std = _summary(base_eff)
    base_bias_mean, base_bias_med, base_bias_std = _summary(base_bias)
    base_cons_mean, base_cons_med, base_cons_std = _summary(base_cons)

    m1_pred_mean, m1_pred_med, m1_pred_std = _summary(m1_pred)
    m1_eff_mean, m1_eff_med, m1_eff_std = _summary(m1_eff)
    m1_bias_mean, m1_bias_med, m1_bias_std = _summary(m1_bias)
    m1_cons_mean, m1_cons_med, m1_cons_std = _summary(m1_cons)

    m2_pred_mean, m2_pred_med, m2_pred_std = _summary(m2_pred)
    m2_eff_mean, m2_eff_med, m2_eff_std = _summary(m2_eff)
    m2_bias_mean, m2_bias_med, m2_bias_std = _summary(m2_bias)
    m2_cons_mean, m2_cons_med, m2_cons_std = _summary(m2_cons)

    print("=== Compare Causal Methods Benchmark ===")
    print(f"Seeds: {args.start_seed}..{args.start_seed + args.num_seeds - 1}")

    print("+-----------------------+-----------+-----------------+-----------------+-----------------+-----------------+")
    print("| method                | stat      | pred_mae        | effect_mae      | bias_mae        | consistency_mae |")
    print("+-----------------------+-----------+-----------------+-----------------+-----------------+-----------------+")
    print(
        f"| baseline              | mean      | {base_pred_mean:>15.6f} | {base_eff_mean:>15.6f} |"
        f" {base_bias_mean:>15.6f} | {base_cons_mean:>15.6f} |"
    )
    print(
        f"| baseline              | median    | {base_pred_med:>15.6f} | {base_eff_med:>15.6f} |"
        f" {base_bias_med:>15.6f} | {base_cons_med:>15.6f} |"
    )
    print(
        f"| baseline              | std       | {base_pred_std:>15.6f} | {base_eff_std:>15.6f} |"
        f" {base_bias_std:>15.6f} | {base_cons_std:>15.6f} |"
    )
    print("+-----------------------+-----------+-----------------+-----------------+-----------------+-----------------+")
    print(
        f"| method1_effect_grad   | mean      | {m1_pred_mean:>15.6f} | {m1_eff_mean:>15.6f} |"
        f" {m1_bias_mean:>15.6f} | {m1_cons_mean:>15.6f} |"
    )
    print(
        f"| method1_effect_grad   | median    | {m1_pred_med:>15.6f} | {m1_eff_med:>15.6f} |"
        f" {m1_bias_med:>15.6f} | {m1_cons_med:>15.6f} |"
    )
    print(
        f"| method1_effect_grad   | std       | {m1_pred_std:>15.6f} | {m1_eff_std:>15.6f} |"
        f" {m1_bias_std:>15.6f} | {m1_cons_std:>15.6f} |"
    )
    print("+-----------------------+-----------+-----------------+-----------------+-----------------+-----------------+")
    print(
        f"| method2_three_head    | mean      | {m2_pred_mean:>15.6f} | {m2_eff_mean:>15.6f} |"
        f" {m2_bias_mean:>15.6f} | {m2_cons_mean:>15.6f} |"
    )
    print(
        f"| method2_three_head    | median    | {m2_pred_med:>15.6f} | {m2_eff_med:>15.6f} |"
        f" {m2_bias_med:>15.6f} | {m2_cons_med:>15.6f} |"
    )
    print(
        f"| method2_three_head    | std       | {m2_pred_std:>15.6f} | {m2_eff_std:>15.6f} |"
        f" {m2_bias_std:>15.6f} | {m2_cons_std:>15.6f} |"
    )
    print("+-----------------------+-----------+-----------------+-----------------+-----------------+-----------------+")


if __name__ == "__main__":
    main()
