from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_SEEDS = (7, 11, 13)
DEFAULT_EXACT_RANKS = (1, 2, 4, 8, 16, 64)
DEFAULT_VISIBLE_CLAUSES = (5, 6, 7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the phase-4 masked ternary scale-up pilot with sane bert-large defaults."
    )
    parser.add_argument("--root-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model-name", default="bert-large-uncased")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4_scaleup_bert_large_pilot"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/features"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-examples-per-label", type=int, default=600)
    parser.add_argument("--test-examples-per-label", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--masked-visible-clauses",
        type=int,
        nargs="+",
        default=list(DEFAULT_VISIBLE_CLAUSES),
    )
    parser.add_argument("--exact-ranks", type=int, nargs="+", default=list(DEFAULT_EXACT_RANKS))
    parser.add_argument("--prefer-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--train-device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--with-random-control", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root_dir = args.root_dir.resolve()
    command = [
        sys.executable,
        "scripts/run_phase1.py",
        "--root-dir",
        str(root_dir),
        "--task-suite",
        "phase4_masked_balanced_ternary",
        "--phase3-structural-holdout",
        "--relation-ids",
        "masked_balanced_triplet",
        "--model-name",
        args.model_name,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--exact-ranks",
        *[str(rank) for rank in args.exact_ranks],
        "--masked-visible-clauses",
        *[str(count) for count in args.masked_visible_clauses],
        "--train-examples-per-label",
        str(args.train_examples_per_label),
        "--test-examples-per-label",
        str(args.test_examples_per_label),
        "--epochs",
        str(args.epochs),
        "--encoder-batch-size",
        str(args.encoder_batch_size),
        "--probe-batch-size",
        str(args.probe_batch_size),
        "--max-length",
        str(args.max_length),
        "--prefer-device",
        args.prefer_device,
        "--train-device",
        args.train_device,
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    if not args.with_random_control:
        command.append("--skip-random-control")
    if args.no_plots:
        command.append("--no-plots")

    env = os.environ.copy()
    src_path = str(root_dir / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"

    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=root_dir, env=env, check=True)


if __name__ == "__main__":
    main()
