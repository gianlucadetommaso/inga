"""High-bias stress benchmark to highlight method-2 advantages.

This benchmark filters to seeds where the *true causal bias magnitude* is high
(approximated by baseline bias MAE, since baseline predicts zero bias), then
compares prediction quality across baseline/method1/method2.

Intuition:
- Method 1 enforces d y_hat / d x ~= causal effect, effectively bias ~= 0.
- When true bias is large, this can conflict with fitting y.
- Method 2 models effect and bias separately, so it can represent
  d y_hat / d x ~= effect + bias.
"""

from __future__ import annotations

import argparse
import math
import statistics

from steindag.sem.demo_compare_causal_methods import run_demo


def _summary(values: list[float]) -> tuple[float, float, float, int]:
    finite_vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    if not finite_vals:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        statistics.mean(finite_vals),
        statistics.median(finite_vals),
        statistics.pstdev(finite_vals),
        len(finite_vals),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--min-bias-mae",
        type=float,
        default=0.1,
        help="Keep only seeds with baseline_bias_mae >= this threshold.",
    )
    parser.add_argument(
        "--lambda-effect-m1",
        type=float,
        default=2.0,
        help="Stronger method-1 effect constraint for stress-testing high-bias settings.",
    )
    args = parser.parse_args()

    base_pred: list[float] = []
    base_eff: list[float] = []
    m1_pred: list[float] = []
    m1_eff: list[float] = []
    m2_pred: list[float] = []
    m2_eff: list[float] = []
    bias_mag: list[float] = []

    selected = 0
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        out = run_demo(
            seed=seed,
            train_epochs=args.epochs,
            lambda_effect_m1=args.lambda_effect_m1,
            verbose=False,
        )

        bmag = out["baseline_bias_mae"]
        if not math.isfinite(bmag) or bmag < args.min_bias_mae:
            continue

        selected += 1
        bias_mag.append(bmag)
        base_pred.append(out["baseline_pred_mae"])
        base_eff.append(out["baseline_effect_mae"])
        m1_pred.append(out["method1_pred_mae"])
        m1_eff.append(out["method1_effect_mae"])
        m2_pred.append(out["method2_pred_mae"])
        m2_eff.append(out["method2_effect_mae"])

        better = "method2" if out["method2_pred_mae"] < out["method1_pred_mae"] else "method1"
        print(
            f"seed={seed} | bias_mag={bmag:.4f} | better_m1_vs_m2={better} "
            f"base_pred={out['baseline_pred_mae']:.4f} "
            f"m1_pred={out['method1_pred_mae']:.4f} "
            f"m2_pred={out['method2_pred_mae']:.4f} "
            f"base_eff={out['baseline_effect_mae']:.4f} "
            f"m1_eff={out['method1_effect_mae']:.4f} "
            f"m2_eff={out['method2_effect_mae']:.4f}",
            flush=True,
        )

    base_mean, base_med, base_std, base_n = _summary(base_pred)
    base_eff_mean, base_eff_med, base_eff_std, base_eff_n = _summary(base_eff)
    m1_mean, m1_med, m1_std, m1_n = _summary(m1_pred)
    m1_eff_mean, m1_eff_med, m1_eff_std, m1_eff_n = _summary(m1_eff)
    m2_mean, m2_med, m2_std, m2_n = _summary(m2_pred)
    m2_eff_mean, m2_eff_med, m2_eff_std, m2_eff_n = _summary(m2_eff)
    bias_mean, bias_med, bias_std, bias_n = _summary(bias_mag)

    wins_m2_over_m1 = sum(m2_pred[i] < m1_pred[i] for i in range(len(m1_pred)))

    print("=== High-Bias Stress Benchmark (method1 vs method2) ===")
    print(
        f"seed_range={args.start_seed}..{args.start_seed + args.num_seeds - 1} | "
        f"min_bias_mae={args.min_bias_mae} | selected={selected}"
    )
    print("+-----------------------+-----------+-----------------+")
    print("| quantity              | stat      | value           |")
    print("+-----------------------+-----------+-----------------+")
    print(f"| selected_bias_mae     | mean      | {bias_mean:>15.6f} |")
    print(f"| selected_bias_mae     | median    | {bias_med:>15.6f} |")
    print(f"| selected_bias_mae     | std       | {bias_std:>15.6f} |")
    print("+-----------------------+-----------+-----------------+")
    print(f"| baseline_pred_mae     | mean      | {base_mean:>15.6f} |")
    print(f"| baseline_pred_mae     | median    | {base_med:>15.6f} |")
    print(f"| baseline_pred_mae     | std       | {base_std:>15.6f} |")
    print("+-----------------------+-----------+-----------------+")
    print(f"| method1_pred_mae      | mean      | {m1_mean:>15.6f} |")
    print(f"| method1_pred_mae      | median    | {m1_med:>15.6f} |")
    print(f"| method1_pred_mae      | std       | {m1_std:>15.6f} |")
    print("+-----------------------+-----------+-----------------+")
    print(f"| method2_pred_mae      | mean      | {m2_mean:>15.6f} |")
    print(f"| method2_pred_mae      | median    | {m2_med:>15.6f} |")
    print(f"| method2_pred_mae      | std       | {m2_std:>15.6f} |")
    print("+-----------------------+-----------+-----------------+")

    print(f"| baseline_effect_mae   | mean      | {base_eff_mean:>15.6f} |")
    print(f"| baseline_effect_mae   | median    | {base_eff_med:>15.6f} |")
    print(f"| baseline_effect_mae   | std       | {base_eff_std:>15.6f} |")
    print("+-----------------------+-----------+-----------------+")
    print(f"| method1_effect_mae    | mean      | {m1_eff_mean:>15.6f} |")
    print(f"| method1_effect_mae    | median    | {m1_eff_med:>15.6f} |")
    print(f"| method1_effect_mae    | std       | {m1_eff_std:>15.6f} |")
    print("+-----------------------+-----------+-----------------+")
    print(f"| method2_effect_mae    | mean      | {m2_eff_mean:>15.6f} |")
    print(f"| method2_effect_mae    | median    | {m2_eff_med:>15.6f} |")
    print(f"| method2_effect_mae    | std       | {m2_eff_std:>15.6f} |")
    print("+-----------------------+-----------+-----------------+")

    print(
        f"counts: baseline/m1/m2 pred={base_n}/{m1_n}/{m2_n}, "
        f"effect={base_eff_n}/{m1_eff_n}/{m2_eff_n}, bias={bias_n} | "
        f"m2 better than m1 on {wins_m2_over_m1}/{len(m1_pred)} selected seeds"
    )


if __name__ == "__main__":
    main()
