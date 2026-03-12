from __future__ import annotations

import argparse
from pathlib import Path

from stoked_semantic.config import (
    BALANCED_TEST_TEMPLATE_IDS,
    BALANCED_TRAIN_TEMPLATE_IDS,
    DEFAULT_FEATURE_CACHE_DIR,
    DEFAULT_RELATION_IDS,
    DEFAULT_TEMPLATE_IDS,
    STRICT_TEST_TEMPLATE_IDS,
    STRICT_TRAIN_TEMPLATE_IDS,
    DataConfig,
    EncoderConfig,
    ExperimentConfig,
    ProbeConfig,
    ReportConfig,
)
from stoked_semantic.data import available_relation_ids, available_template_ids
from stoked_semantic.pipeline import MultiSeedPhaseOnePipeline, PhaseOnePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the phase-1 semantic probing pipeline.")
    parser.add_argument("--root-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--prefer-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--train-device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    parser.add_argument("--train-examples-per-label", type=int, default=600)
    parser.add_argument("--test-examples-per-label", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-rank", type=int, default=64)
    parser.add_argument("--pairwise-hidden", type=int, default=128)
    parser.add_argument("--exact-rank", type=int, default=64)
    parser.add_argument("--triadic-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--relation-ids",
        nargs="+",
        default=list(DEFAULT_RELATION_IDS),
        metavar="RELATION",
        help=f"Relation families to include. Available: {', '.join(available_relation_ids())}",
    )
    parser.add_argument(
        "--train-template-ids",
        nargs="+",
        metavar="TEMPLATE",
        help=f"Template ids for train split. Available: {', '.join(available_template_ids())}",
    )
    parser.add_argument(
        "--test-template-ids",
        nargs="+",
        metavar="TEMPLATE",
        help=f"Template ids for test split. Available: {', '.join(available_template_ids())}",
    )
    parser.add_argument(
        "--balanced-template-holdout",
        action="store_true",
        help="Use disjoint template families while keeping both forward and reverse variants on both sides.",
    )
    parser.add_argument(
        "--strict-template-holdout",
        action="store_true",
        help="Use disjoint forward/reverse template sets between train and test.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1"))
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--skip-random-control", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    train_template_ids, test_template_ids = _template_ids_from_args(args)
    _validate_choices("relation", tuple(args.relation_ids), available_relation_ids())
    _validate_choices("template", train_template_ids, available_template_ids())
    _validate_choices("template", test_template_ids, available_template_ids())

    data = DataConfig(
        relation_ids=tuple(args.relation_ids),
        train_template_ids=train_template_ids,
        test_template_ids=test_template_ids,
        train_examples_per_label=args.train_examples_per_label,
        test_examples_per_label=args.test_examples_per_label,
        seed=args.seed,
    )
    encoder = EncoderConfig(
        model_name=args.model_name,
        batch_size=args.encoder_batch_size,
        max_length=args.max_length,
        prefer_device=args.prefer_device,
        cache_dir=args.cache_dir,
    )
    train_device = "auto" if args.train_device == "auto" else args.train_device
    probe = ProbeConfig(
        exact_rank=args.exact_rank,
        pairwise_rank=args.pairwise_rank,
        pairwise_hidden=args.pairwise_hidden,
        triadic_rank=args.triadic_rank,
        epochs=args.epochs,
        batch_size=args.probe_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_device=train_device,
        seed=args.seed,
    )
    report = ReportConfig(output_dir=args.output_dir, write_plots=not args.no_plots)
    return ExperimentConfig(
        root_dir=args.root_dir,
        data=data,
        encoder=encoder,
        probe=probe,
        report=report,
        run_pretrained=not args.skip_pretrained,
        run_random_control=not args.skip_random_control,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    if args.seeds:
        run_seeds = tuple(dict.fromkeys(args.seeds))
        runner = MultiSeedPhaseOnePipeline(config=config, run_seeds=run_seeds)
        summary = runner.run()
    else:
        pipeline = PhaseOnePipeline(config=config)
        summary = pipeline.run().summary()
    print(summary)


def _template_ids_from_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if args.balanced_template_holdout and args.strict_template_holdout:
        raise ValueError("Choose either --balanced-template-holdout or --strict-template-holdout, not both.")

    if args.balanced_template_holdout:
        train_template_ids = tuple(args.train_template_ids or BALANCED_TRAIN_TEMPLATE_IDS)
        test_template_ids = tuple(args.test_template_ids or BALANCED_TEST_TEMPLATE_IDS)
        return train_template_ids, test_template_ids

    if args.strict_template_holdout:
        train_template_ids = tuple(args.train_template_ids or STRICT_TRAIN_TEMPLATE_IDS)
        test_template_ids = tuple(args.test_template_ids or STRICT_TEST_TEMPLATE_IDS)
        return train_template_ids, test_template_ids

    train_template_ids = tuple(args.train_template_ids or DEFAULT_TEMPLATE_IDS)
    test_template_ids = tuple(args.test_template_ids or train_template_ids)
    return train_template_ids, test_template_ids


def _validate_choices(kind: str, requested: tuple[str, ...], allowed: tuple[str, ...]) -> None:
    invalid = sorted(set(requested) - set(allowed))
    if invalid:
        raise ValueError(
            f"Unknown {kind} id(s): {', '.join(invalid)}. "
            f"Available: {', '.join(allowed)}"
        )


if __name__ == "__main__":
    main()
