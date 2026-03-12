from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stoked_semantic.config import (
    BALANCED_TEST_TEMPLATE_IDS,
    BALANCED_TRAIN_TEMPLATE_IDS,
    DEFAULT_FEATURE_CACHE_DIR,
    DEFAULT_RELATION_IDS,
    DEFAULT_TEMPLATE_IDS,
    FACTORIZED_TEST_TEMPLATE_IDS,
    FACTORIZED_TRAIN_TEMPLATE_IDS,
    PHASE1_TASK_SUITE,
    PHASE2_DEFAULT_RELATION_IDS,
    PHASE2_DEFAULT_TEMPLATE_IDS,
    PHASE2_TASK_SUITE,
    PHASE3_DEFAULT_RELATION_IDS,
    PHASE3_DEFAULT_TEMPLATE_IDS,
    PHASE3_STRUCTURAL_TEST_TEMPLATE_IDS,
    PHASE3_STRUCTURAL_TRAIN_TEMPLATE_IDS,
    PHASE3_TASK_SUITE,
    PHASE3_TEST_NAMES,
    PHASE3_TRAIN_NAMES,
    PHASE4_DEFAULT_RELATION_IDS,
    PHASE4_TASK_SUITE,
    STRICT_TEST_TEMPLATE_IDS,
    STRICT_TRAIN_TEMPLATE_IDS,
    STRUCTURAL_TEST_TEMPLATE_IDS,
    STRUCTURAL_TRAIN_TEMPLATE_IDS,
    DataConfig,
    EncoderConfig,
    ExperimentConfig,
    ProbeConfig,
    ReportConfig,
)
from stoked_semantic.data import available_relation_ids, available_template_ids, relation_query_arity
from stoked_semantic.pipeline import MultiSeedPhaseOnePipeline, PhaseOnePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the semantic probing pipeline.")
    parser.add_argument("--root-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--task-suite",
        choices=[PHASE1_TASK_SUITE, PHASE2_TASK_SUITE, PHASE3_TASK_SUITE, PHASE4_TASK_SUITE],
        default=PHASE1_TASK_SUITE,
    )
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
    parser.add_argument("--exact-ranks", type=int, nargs="+")
    parser.add_argument("--triadic-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--relation-ids",
        nargs="+",
        default=None,
        metavar="RELATION",
        help="Relation or task families to include. Available ids depend on --task-suite.",
    )
    parser.add_argument(
        "--train-template-ids",
        nargs="+",
        metavar="TEMPLATE",
        help="Template ids for train split. Available ids depend on --task-suite.",
    )
    parser.add_argument(
        "--test-template-ids",
        nargs="+",
        metavar="TEMPLATE",
        help="Template ids for test split. Available ids depend on --task-suite.",
    )
    parser.add_argument(
        "--structural-template-holdout",
        action="store_true",
        help=(
            "Use a structural holdout that keeps active/passive, above/below, and base mixed families "
            "on both sides while dropping the paraphrase-mixed inverse families."
        ),
    )
    parser.add_argument(
        "--factorized-template-holdout",
        action="store_true",
        help=(
            "Use complementary structural template combinations so train/test both contain "
            "direct and inverse lexical realizations, both registers, and both clause orders."
        ),
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
    parser.add_argument(
        "--phase3-structural-holdout",
        action="store_true",
        help=(
            "For phase-3 and phase-4 balanced ternary tasks, train on one set of ternary paraphrase "
            "families and test on held-out families while keeping forward and reverse clause orders "
            "on both sides."
        ),
    )
    parser.add_argument(
        "--masked-visible-clauses",
        type=int,
        nargs="+",
        metavar="K",
        help=(
            "For phase-4 masked balanced ternary tasks, use these visible positive-clause counts "
            "when sampling masked premises."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1"))
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--skip-random-control", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    train_template_ids, test_template_ids = _template_ids_from_args(args)
    relation_ids = tuple(args.relation_ids or _default_relation_ids(task_suite=args.task_suite))
    _validate_masked_visible_clauses(args)
    _validate_choices("relation", relation_ids, available_relation_ids(args.task_suite))
    _validate_choices("template", train_template_ids, available_template_ids(args.task_suite))
    _validate_choices("template", test_template_ids, available_template_ids(args.task_suite))
    _validate_query_arity(task_suite=args.task_suite, relation_ids=relation_ids)

    data = DataConfig(
        task_suite=args.task_suite,
        relation_ids=relation_ids,
        train_names=_default_train_names(args.task_suite),
        test_names=_default_test_names(args.task_suite),
        train_template_ids=train_template_ids,
        test_template_ids=test_template_ids,
        train_examples_per_label=args.train_examples_per_label,
        test_examples_per_label=args.test_examples_per_label,
        seed=args.seed,
        masked_visible_clause_counts=tuple(args.masked_visible_clauses or DataConfig.masked_visible_clause_counts),
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
        exact_rank_sweep=tuple(args.exact_ranks or ()),
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
    report = ReportConfig(
        output_dir=_resolve_output_dir(root_dir=args.root_dir, output_dir=args.output_dir),
        write_plots=not args.no_plots,
    )
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


def _resolve_output_dir(root_dir: Path, output_dir: Path) -> Path:
    rooted = output_dir if output_dir.is_absolute() else root_dir / output_dir
    if not rooted.exists():
        return output_dir

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base_name = output_dir.name
    parent = output_dir.parent
    candidate = parent / f"{base_name}_{timestamp}"
    counter = 1
    rooted_candidate = candidate if candidate.is_absolute() else root_dir / candidate
    while rooted_candidate.exists():
        candidate = parent / f"{base_name}_{timestamp}_{counter:02d}"
        rooted_candidate = candidate if candidate.is_absolute() else root_dir / candidate
        counter += 1
    return candidate


def _template_ids_from_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if args.task_suite == PHASE2_TASK_SUITE:
        selected_holdouts = [
            args.structural_template_holdout,
            args.factorized_template_holdout,
            args.balanced_template_holdout,
            args.strict_template_holdout,
            args.phase3_structural_holdout,
        ]
        if any(selected_holdouts):
            raise ValueError(
                "Phase-2 four-node tasks do not use the phase-1 or phase-3 holdout presets. "
                "Specify --train-template-ids/--test-template-ids directly if needed."
            )
        train_template_ids = tuple(args.train_template_ids or PHASE2_DEFAULT_TEMPLATE_IDS)
        test_template_ids = tuple(args.test_template_ids or train_template_ids)
        return train_template_ids, test_template_ids

    if args.task_suite == PHASE3_TASK_SUITE:
        selected_holdouts = [
            args.structural_template_holdout,
            args.factorized_template_holdout,
            args.balanced_template_holdout,
            args.strict_template_holdout,
        ]
        if any(selected_holdouts):
            raise ValueError(
                "Phase-3 balanced ternary tasks do not use the phase-1 holdout presets. "
                "Specify --train-template-ids/--test-template-ids directly if needed."
            )
        if args.phase3_structural_holdout:
            train_template_ids = tuple(args.train_template_ids or PHASE3_STRUCTURAL_TRAIN_TEMPLATE_IDS)
            test_template_ids = tuple(args.test_template_ids or PHASE3_STRUCTURAL_TEST_TEMPLATE_IDS)
            return train_template_ids, test_template_ids
        train_template_ids = tuple(args.train_template_ids or PHASE3_DEFAULT_TEMPLATE_IDS)
        test_template_ids = tuple(args.test_template_ids or train_template_ids)
        return train_template_ids, test_template_ids

    if args.task_suite == PHASE4_TASK_SUITE:
        selected_holdouts = [
            args.structural_template_holdout,
            args.factorized_template_holdout,
            args.balanced_template_holdout,
            args.strict_template_holdout,
        ]
        if any(selected_holdouts):
            raise ValueError(
                "Phase-4 masked balanced ternary tasks do not use the phase-1 holdout presets. "
                "Specify --train-template-ids/--test-template-ids directly if needed."
            )
        if args.phase3_structural_holdout:
            train_template_ids = tuple(args.train_template_ids or PHASE3_STRUCTURAL_TRAIN_TEMPLATE_IDS)
            test_template_ids = tuple(args.test_template_ids or PHASE3_STRUCTURAL_TEST_TEMPLATE_IDS)
            return train_template_ids, test_template_ids
        train_template_ids = tuple(args.train_template_ids or PHASE3_DEFAULT_TEMPLATE_IDS)
        test_template_ids = tuple(args.test_template_ids or train_template_ids)
        return train_template_ids, test_template_ids

    if args.phase3_structural_holdout:
        raise ValueError(
            "--phase3-structural-holdout is only valid with --task-suite "
            "phase3_balanced_ternary or phase4_masked_balanced_ternary."
        )

    selected_holdouts = [
        args.structural_template_holdout,
        args.factorized_template_holdout,
        args.balanced_template_holdout,
        args.strict_template_holdout,
        args.phase3_structural_holdout,
    ]
    if sum(bool(flag) for flag in selected_holdouts) > 1:
        raise ValueError(
            "Choose at most one of --structural-template-holdout, "
            "--factorized-template-holdout, --balanced-template-holdout, "
            "--strict-template-holdout, or --phase3-structural-holdout."
        )

    if args.structural_template_holdout:
        train_template_ids = tuple(args.train_template_ids or STRUCTURAL_TRAIN_TEMPLATE_IDS)
        test_template_ids = tuple(args.test_template_ids or STRUCTURAL_TEST_TEMPLATE_IDS)
        return train_template_ids, test_template_ids

    if args.factorized_template_holdout:
        train_template_ids = tuple(args.train_template_ids or FACTORIZED_TRAIN_TEMPLATE_IDS)
        test_template_ids = tuple(args.test_template_ids or FACTORIZED_TEST_TEMPLATE_IDS)
        return train_template_ids, test_template_ids

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


def _default_relation_ids(task_suite: str) -> tuple[str, ...]:
    if task_suite == PHASE1_TASK_SUITE:
        return DEFAULT_RELATION_IDS
    if task_suite == PHASE2_TASK_SUITE:
        return PHASE2_DEFAULT_RELATION_IDS
    if task_suite == PHASE3_TASK_SUITE:
        return PHASE3_DEFAULT_RELATION_IDS
    if task_suite == PHASE4_TASK_SUITE:
        return PHASE4_DEFAULT_RELATION_IDS
    raise ValueError(f"Unsupported task suite: {task_suite}")


def _default_train_names(task_suite: str) -> tuple[str, ...]:
    if task_suite in {PHASE3_TASK_SUITE, PHASE4_TASK_SUITE}:
        return PHASE3_TRAIN_NAMES
    return DataConfig.train_names


def _default_test_names(task_suite: str) -> tuple[str, ...]:
    if task_suite in {PHASE3_TASK_SUITE, PHASE4_TASK_SUITE}:
        return PHASE3_TEST_NAMES
    return DataConfig.test_names


def _validate_choices(kind: str, requested: tuple[str, ...], allowed: tuple[str, ...]) -> None:
    invalid = sorted(set(requested) - set(allowed))
    if invalid:
        raise ValueError(
            f"Unknown {kind} id(s): {', '.join(invalid)}. "
            f"Available: {', '.join(allowed)}"
        )


def _validate_query_arity(task_suite: str, relation_ids: tuple[str, ...]) -> None:
    arities = {
        relation_query_arity(task_suite=task_suite, relation_id=relation_id)
        for relation_id in relation_ids
    }
    if len(arities) > 1:
        raise ValueError(
            "Mixed query arities are not supported in a single run yet. "
            f"Selected relations have arities: {sorted(arities)}. "
            "Run pair-query and ternary-query task families separately."
        )


def _validate_masked_visible_clauses(args: argparse.Namespace) -> None:
    if args.masked_visible_clauses is None:
        return
    if args.task_suite != PHASE4_TASK_SUITE:
        raise ValueError("--masked-visible-clauses is only valid with --task-suite phase4_masked_balanced_ternary.")
    invalid = [count for count in args.masked_visible_clauses if count < 1 or count > 9]
    if invalid:
        raise ValueError(
            "Masked visible clause counts must be between 1 and 9. "
            f"Received: {invalid}"
        )


if __name__ == "__main__":
    main()
