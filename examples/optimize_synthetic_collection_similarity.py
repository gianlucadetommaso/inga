"""Iteratively optimize synthetic collections to match a real dataset.

This script performs several iterations. In each iteration it:
1) samples multiple synthetic-collection candidates,
2) scores each against a real dataset,
3) identifies key factors associated with higher realism,
4) updates the sampling center toward better candidates,
5) persists all candidate collections and reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from inga.scm import (
    RandomSCMConfig,
    SCMDatasetCollectionConfig,
    SimilarityThresholds,
    compare_real_dataset_to_collection,
    generate_scm_dataset_collection,
    infer_similarity_key_factors,
    load_scm_dataset,
)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=str, default="datasets/scm_dataset_example")
    parser.add_argument(
        "--output-root", type=str, default="datasets/similarity_optimization"
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--candidates-per-iteration", type=int, default=5)
    parser.add_argument("--num-datasets", type=int, default=8)
    parser.add_argument("--num-variables", type=int, default=6)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    real_dataset = load_scm_dataset(args.real)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = SimilarityThresholds()
    rng = random.Random(args.seed)

    center = {
        "parent_prob": 0.40,
        "nonlinear_prob": 0.50,
        "coef_abs_max": 2.0,
        "sigma_min": 0.5,
        "sigma_max": 1.5,
        "intercept_abs_max": 1.0,
    }
    spread = {
        "parent_prob": 0.15,
        "nonlinear_prob": 0.20,
        "coef_abs_max": 0.8,
        "sigma_min": 0.25,
        "sigma_max": 0.25,
        "intercept_abs_max": 0.6,
    }

    global_best: dict | None = None
    history: list[dict] = []

    for iteration in range(args.iterations):
        iter_dir = output_root / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        candidate_records: list[dict] = []
        candidate_params: list[dict[str, float]] = []
        candidate_scores: list[float] = []

        for cand_idx in range(args.candidates_per_iteration):
            params = {
                "parent_prob": _clip(
                    rng.gauss(center["parent_prob"], spread["parent_prob"]), 0.05, 0.95
                ),
                "nonlinear_prob": _clip(
                    rng.gauss(center["nonlinear_prob"], spread["nonlinear_prob"]),
                    0.0,
                    1.0,
                ),
                "coef_abs_max": _clip(
                    rng.gauss(center["coef_abs_max"], spread["coef_abs_max"]), 0.2, 5.0
                ),
                "sigma_min": _clip(
                    rng.gauss(center["sigma_min"], spread["sigma_min"]), 0.05, 2.5
                ),
                "sigma_max": _clip(
                    rng.gauss(center["sigma_max"], spread["sigma_max"]), 0.10, 3.0
                ),
                "intercept_abs_max": _clip(
                    rng.gauss(center["intercept_abs_max"], spread["intercept_abs_max"]),
                    0.1,
                    3.0,
                ),
            }
            if params["sigma_max"] < params["sigma_min"] + 0.05:
                params["sigma_max"] = params["sigma_min"] + 0.05

            scm_cfg = RandomSCMConfig(
                num_variables=args.num_variables,
                parent_prob=params["parent_prob"],
                sigma_range=(params["sigma_min"], params["sigma_max"]),
                coef_range=(-params["coef_abs_max"], params["coef_abs_max"]),
                intercept_range=(
                    -params["intercept_abs_max"],
                    params["intercept_abs_max"],
                ),
                nonlinear_prob=params["nonlinear_prob"],
                seed=args.seed + iteration * 1000 + cand_idx,
            )
            coll_cfg = SCMDatasetCollectionConfig(
                scm_config=scm_cfg,
                num_datasets=args.num_datasets,
                num_samples=args.samples,
                num_queries=args.queries,
                seed=args.seed + iteration * 1000 + cand_idx,
            )
            collection = generate_scm_dataset_collection(coll_cfg)

            cand_dir = iter_dir / f"candidate_{cand_idx:03d}"
            collection.save(cand_dir)

            sim_report = compare_real_dataset_to_collection(
                real_dataset,
                collection,
                thresholds=thresholds,
            )
            report_path = cand_dir / "similarity_report.json"
            report_path.write_text(
                json.dumps(sim_report.to_dict(), indent=2), encoding="utf-8"
            )

            record = {
                "candidate_index": cand_idx,
                "collection_dir": str(cand_dir),
                "score": sim_report.overall_score,
                "similar_enough": sim_report.similar_enough,
                "params": params,
            }
            candidate_records.append(record)
            candidate_params.append(params)
            candidate_scores.append(sim_report.overall_score)

            if global_best is None or sim_report.overall_score > float(
                global_best["score"]
            ):
                global_best = record

        factor_rank = infer_similarity_key_factors(candidate_params, candidate_scores)
        best_local = max(candidate_records, key=lambda item: float(item["score"]))

        top_n = max(1, len(candidate_records) // 2)
        top = sorted(
            candidate_records, key=lambda item: float(item["score"]), reverse=True
        )[:top_n]
        for key in center.keys():
            center[key] = float(
                sum(float(item["params"][key]) for item in top) / len(top)
            )
        for key in spread.keys():
            spread[key] = max(0.03, spread[key] * 0.90)

        iter_summary = {
            "iteration": iteration,
            "best_local": best_local,
            "factor_rank": [{"factor": k, "correlation": v} for k, v in factor_rank],
            "updated_center": dict(center),
            "updated_spread": dict(spread),
            "candidates": candidate_records,
        }
        history.append(iter_summary)
        (iter_dir / "iteration_summary.json").write_text(
            json.dumps(iter_summary, indent=2),
            encoding="utf-8",
        )

        print(
            f"[iter {iteration}] best score={best_local['score']:.3f} "
            f"candidate={best_local['candidate_index']}"
        )
        if factor_rank:
            top_factor, top_corr = factor_rank[0]
            print(f"  key factor: {top_factor} (corr={top_corr:.3f})")

    final_summary = {
        "real_dataset": args.real,
        "iterations": args.iterations,
        "candidates_per_iteration": args.candidates_per_iteration,
        "global_best": global_best,
        "history": history,
        "final_center": center,
        "final_spread": spread,
    }
    (output_root / "optimization_summary.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    print("Optimization completed.")
    if global_best is not None:
        print(
            f"Global best score={global_best['score']:.3f} at {global_best['collection_dir']}"
        )
    print(f"Saved optimization summary to: {output_root / 'optimization_summary.json'}")


if __name__ == "__main__":
    main()
