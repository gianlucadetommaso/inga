"""Run a large synthetic->real benchmark pipeline with multiple seeds.

Pipeline per seed:
1) Generate and persist a synthetic SEM corpus.
2) Train synthetic checkpoints (prediction-only + causal-consistency).
3) Evaluate few-shot transfer on real datasets using 3 regimes:
   - baseline_real
   - synthetic_finetuned
   - causal_finetuned

All artifacts are stored on disk, so runs are resumable.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, log_path: Path, skip_if_log_exists: bool) -> str:
    if skip_if_log_exists and log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        print(f"[skip] {' '.join(cmd)}")
        return text

    print(f"\n[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )
    return out


def _extract_wins(text: str) -> tuple[str, str, str]:
    def find(pat: str) -> str:
        m = re.search(pat, text)
        return m.group(0) if m else "N/A"

    a = find(r"Causal wins vs baseline_real: \d+/\d+ datasets")
    b = find(r"Causal wins vs synthetic_finetuned: \d+/\d+ datasets")
    c = find(r"Causal best of all 3 regimes: \d+/\d+ datasets")
    return a, b, c


def _corpus_cache_exists(corpus_dir: Path, *, expected_num_datasets: int) -> bool:
    """Return True if a usable synthetic corpus cache is present."""
    manifest = corpus_dir / "manifest.json"
    if not manifest.exists():
        return False
    pt_files = list(corpus_dir.glob("dataset_*.pt"))
    return len(pt_files) >= max(1, expected_num_datasets)


def _models_cache_exists(models_dir: Path) -> bool:
    """Return True if all required synthetic model checkpoints are present."""
    required = ["baseline.pt", "l2.pt", "causal.pt", "meta.json"]
    return all((models_dir / name).exists() for name in required)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=str, default=".data/synth_to_real_large")
    # Best-practice default: evaluate across multiple random seeds.
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")

    # Synthetic corpus size/config.
    p.add_argument("--num-datasets", type=int, default=120)
    p.add_argument("--num-variables", type=int, default=12)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--num-queries", type=int, default=1)
    p.add_argument("--min-observed", type=int, default=1)

    # Synthetic training config.
    p.add_argument("--train-epochs", type=int, default=4)
    p.add_argument("--train-max-datasets", type=int, default=120)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--arch", type=str, default="ft_transformer_v1")
    p.add_argument("--adapter-dim", type=int, default=32)
    p.add_argument("--meta", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--meta-epochs", type=int, default=3)
    p.add_argument("--inner-steps", type=int, default=8)
    p.add_argument("--support-frac", type=float, default=0.2)

    # Real few-shot eval config.
    p.add_argument(
        "--real-datasets",
        type=str,
        default="abalone,airfoil_self_noise,yacht_hydrodynamics,wine_quality_red,wine_quality_white",
    )
    p.add_argument("--train-frac", type=float, default=0.01)
    p.add_argument("--real-epochs", type=int, default=80)
    p.add_argument("--probe-epochs", type=int, default=20)
    p.add_argument("--finetune-epochs", type=int, default=80)
    p.add_argument("--cache-dir", type=str, default=".data/tabular_regression")
    p.add_argument(
        "--generate-retries",
        type=int,
        default=3,
        help=(
            "Retries for synthetic corpus generation per seed. "
            "On occasional Laplace posterior failures, retries use a shifted seed."
        ),
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if logs/checkpoints exist.",
    )
    args = p.parse_args()

    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    summary_rows: list[str] = []

    for seed in seeds:
        seed_dir = root / f"seed_{seed:03d}"
        corpus_dir = seed_dir / "corpus"
        models_dir = seed_dir / "models"
        logs_dir = seed_dir / "logs"

        has_corpus_cache = _corpus_cache_exists(
            corpus_dir,
            expected_num_datasets=int(args.num_datasets),
        )
        has_models_cache = _models_cache_exists(models_dir)

        # Default behavior (unless --force): prefer cache and only re-run eval.
        should_generate = bool(args.force or not has_corpus_cache)
        should_train = bool(args.force or not has_models_cache)

        # 1) Generate corpus.
        if should_generate:
            gen_error: RuntimeError | None = None
            retries = max(1, int(args.generate_retries))
            for attempt in range(retries):
                # Keep benchmark seed identity, but shift generator seed on retry
                # to avoid deterministic bad draws that can trigger occasional
                # Laplace/Cholesky failures in SEM posterior fitting.
                gen_seed = int(seed + 10_000 * attempt)
                gen_cmd = [
                    sys.executable,
                    "-u",
                    "examples/synthetic_corpus_generate.py",
                    "--out-dir",
                    str(corpus_dir),
                    "--num-datasets",
                    str(args.num_datasets),
                    "--num-variables",
                    str(args.num_variables),
                    "--num-samples",
                    str(args.num_samples),
                    "--num-queries",
                    str(args.num_queries),
                    "--min-observed",
                    str(args.min_observed),
                    "--seed",
                    str(gen_seed),
                ]
                try:
                    _run(
                        gen_cmd,
                        log_path=logs_dir / "01_generate.log",
                        skip_if_log_exists=False,
                    )
                    if attempt > 0:
                        print(
                            f"[info] generation for seed={seed} succeeded on retry {attempt + 1}/{retries} "
                            f"with generator-seed={gen_seed}"
                        )
                    gen_error = None
                    break
                except RuntimeError as err:
                    gen_error = err
                    if attempt < retries - 1:
                        print(
                            f"[warn] generation failed for seed={seed} (attempt {attempt + 1}/{retries}); retrying"
                        )
            if gen_error is not None:
                raise gen_error
        else:
            print(f"[cache] seed={seed:03d}: using existing corpus at {corpus_dir}")

        # 2) Train synthetic models.
        if should_train:
            train_cmd = [
                sys.executable,
                "-u",
                "examples/synthetic_corpus_train.py",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(models_dir),
                "--max-datasets",
                str(args.train_max_datasets),
                "--epochs",
                str(args.train_epochs),
                "--hidden-dim",
                str(args.hidden_dim),
                "--arch",
                str(args.arch),
                "--adapter-dim",
                str(args.adapter_dim),
                "--meta-epochs",
                str(args.meta_epochs),
                "--inner-steps",
                str(args.inner_steps),
                "--support-frac",
                str(args.support_frac),
                "--seed",
                str(seed),
            ]
            train_cmd.append("--meta" if args.meta else "--no-meta")
            _run(
                train_cmd,
                log_path=logs_dir / "02_train.log",
                skip_if_log_exists=False,
            )
        else:
            print(f"[cache] seed={seed:03d}: using existing models at {models_dir}")

        # 3) Evaluate few-shot transfer.
        eval_cmd = [
            sys.executable,
            "-u",
            "examples/synthetic_to_real_eval.py",
            "--models-dir",
            str(models_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--datasets",
            str(args.real_datasets),
            "--train-frac",
            str(args.train_frac),
            "--real-epochs",
            str(args.real_epochs),
            "--probe-epochs",
            str(args.probe_epochs),
            "--finetune-epochs",
            str(args.finetune_epochs),
            "--finetune-mode",
            "auto",
            "--seed",
            str(seed),
        ]
        eval_out = _run(
            eval_cmd,
            log_path=logs_dir / "03_eval.log",
            # Intentionally re-run evaluation by default so summaries reflect
            # current eval settings while reusing generate/train cache.
            skip_if_log_exists=False,
        )
        w1, w2, w3 = _extract_wins(eval_out)
        row = f"seed={seed:03d} | {w1} | {w2} | {w3}"
        summary_rows.append(row)
        print(f"\n[summary] {row}")

    summary_path = root / "summary.txt"
    summary_text = "\n".join(summary_rows) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSaved summary to: {summary_path}")
    print(summary_text)


if __name__ == "__main__":
    main()
