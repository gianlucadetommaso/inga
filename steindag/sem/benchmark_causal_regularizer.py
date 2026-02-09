"""Benchmark baseline vs L2 vs causal-effect regularization."""

from __future__ import annotations

import argparse
import statistics

from steindag.sem.demo_causal_regularizer import run_demo


def main() -> None:
    """Run benchmark over many seeds and print mean/std MAE per setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    variant_names = ["baseline", "l2_regularized", "causal_regularized"]
    pred_variant_names = [
        "baseline_pred_mae",
        "l2_regularized_pred_mae",
        "causal_regularized_pred_mae",
    ]
    errors: dict[str, list[float]] = {name: [] for name in variant_names}
    pred_errors: dict[str, list[float]] = {name: [] for name in pred_variant_names}

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        out = run_demo(seed=seed, train_epochs=args.epochs, verbose=False)
        for model_name in errors:
            errors[model_name].append(out[model_name])
        for model_name in pred_errors:
            pred_errors[model_name].append(out[model_name])
        best_name = min(out, key=out.get)
        baseline_for_delta = out["baseline"]
        l2_for_delta = out["l2_regularized"]
        causal_for_delta = out["causal_regularized"]
        baseline_pred = out["baseline_pred_mae"]
        l2_pred = out["l2_regularized_pred_mae"]
        causal_pred = out["causal_regularized_pred_mae"]
        print(
            f"seed={seed} | best={best_name}:{out[best_name]:.4f} "
            f"baseline={baseline_for_delta:.4f} l2={l2_for_delta:.4f} "
            f"causal={causal_for_delta:.4f} "
            f"| pred_mae baseline={baseline_pred:.4f} l2={l2_pred:.4f} "
            f"causal={causal_pred:.4f}",
            flush=True,
        )

    baseline_key = "baseline"
    baseline_mean = statistics.mean(errors[baseline_key])

    print("=== Causal Setup Benchmark ===")
    print(f"Seeds: {args.start_seed}..{args.start_seed + args.num_seeds - 1}")
    print("+-----------------------------+--------------+--------------+--------------+--------------+")
    print("| setup                       | mean_mae     | median_mae   | std_mae      | delta_vs_base|")
    print("+-----------------------------+--------------+--------------+--------------+--------------+")
    for model_name in sorted(errors, key=lambda name: statistics.mean(errors[name])):
        mean_mae = statistics.mean(errors[model_name])
        median_mae = statistics.median(errors[model_name])
        std_mae = statistics.pstdev(errors[model_name])
        delta = mean_mae - baseline_mean
        print(
            f"| {model_name:<27} | {mean_mae:>12.6f} | {median_mae:>12.6f} |"
            f" {std_mae:>12.6f} | {delta:>12.6f} |"
        )
    print("+-----------------------------+--------------+--------------+--------------+--------------+")

    best_setup = min(errors, key=lambda name: statistics.mean(errors[name]))
    print(
        "Best setup by mean MAE: "
        f"{best_setup} (mean={statistics.mean(errors[best_setup]):.6f}, "
        f"std={statistics.pstdev(errors[best_setup]):.6f})"
    )
    print(f"Delta reference setup: {baseline_key}")

    pred_baseline_mean = statistics.mean(pred_errors["baseline_pred_mae"])
    pred_key_map = {
        "baseline": "baseline_pred_mae",
        "l2_regularized": "l2_regularized_pred_mae",
        "causal_regularized": "causal_regularized_pred_mae",
    }

    print("=== Test Prediction MAE Benchmark ===")
    print(f"Seeds: {args.start_seed}..{args.start_seed + args.num_seeds - 1}")
    print("+-----------------------------+--------------+--------------+--------------+--------------+")
    print("| setup                       | mean_mae     | median_mae   | std_mae      | delta_vs_base|")
    print("+-----------------------------+--------------+--------------+--------------+--------------+")
    for model_name in sorted(
        variant_names,
        key=lambda name: statistics.mean(pred_errors[pred_key_map[name]]),
    ):
        values = pred_errors[pred_key_map[model_name]]
        mean_mae = statistics.mean(values)
        median_mae = statistics.median(values)
        std_mae = statistics.pstdev(values)
        delta = mean_mae - pred_baseline_mean
        print(
            f"| {model_name:<27} | {mean_mae:>12.6f} | {median_mae:>12.6f} |"
            f" {std_mae:>12.6f} | {delta:>12.6f} |"
        )
    print("+-----------------------------+--------------+--------------+--------------+--------------+")

    best_pred_setup = min(
        variant_names,
        key=lambda name: statistics.mean(pred_errors[pred_key_map[name]]),
    )
    best_pred_vals = pred_errors[pred_key_map[best_pred_setup]]
    print(
        "Best setup by mean test-prediction MAE: "
        f"{best_pred_setup} (mean={statistics.mean(best_pred_vals):.6f}, "
        f"std={statistics.pstdev(best_pred_vals):.6f})"
    )


if __name__ == "__main__":
    main()
