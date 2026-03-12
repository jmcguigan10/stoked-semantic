from __future__ import annotations

import json
from collections import Counter, defaultdict

from stoked_semantic.cli import build_parser, make_config
from stoked_semantic.data import make_dataset_builder


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)
    bundle = make_dataset_builder(config.data).build()

    print("Metadata")
    print(
        json.dumps(
            {
                "task_suite": config.data.task_suite,
                "relation_ids": list(config.data.relation_ids or ()),
                "train_template_ids": list(config.data.train_template_ids or ()),
                "test_template_ids": list(config.data.test_template_ids or ()),
                "masked_visible_clause_counts": list(config.data.masked_visible_clause_counts),
                "train_examples_per_label": config.data.train_examples_per_label,
                "test_examples_per_label": config.data.test_examples_per_label,
                "seed": config.data.seed,
            },
            indent=2,
        )
    )
    print()

    for split_name, examples in (
        ("train", bundle.train_examples),
        ("test", bundle.test_examples),
    ):
        print(f"Split: {split_name}")
        print(f"  examples: {len(examples)}")
        print(f"  labels: {dict(Counter(example.label for example in examples))}")

        by_query_indices: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        by_query_type: dict[str, Counter[int]] = defaultdict(Counter)
        by_template_query: dict[tuple[str, tuple[int, ...]], Counter[int]] = defaultdict(Counter)

        for example in examples:
            by_query_indices[example.query_indices][example.label] += 1
            by_query_type[example.query_type][example.label] += 1
            by_template_query[(example.template_id, example.query_indices)][example.label] += 1

        print("  query_type counts:")
        for query_type, counts in sorted(by_query_type.items()):
            print(f"    {query_type}: {dict(counts)}")

        worst_query_indices = sorted(
            (
                abs(counts[1] - counts[0]),
                query_indices,
                dict(counts),
            )
            for query_indices, counts in by_query_indices.items()
        )[::-1][:12]
        print("  worst query_indices imbalance:")
        for delta, query_indices, counts in worst_query_indices:
            print(f"    delta={delta} query_indices={query_indices} counts={counts}")

        worst_template_query = sorted(
            (
                abs(counts[1] - counts[0]),
                key,
                dict(counts),
            )
            for key, counts in by_template_query.items()
        )[::-1][:12]
        print("  worst template+query_indices imbalance:")
        for delta, key, counts in worst_template_query:
            print(f"    delta={delta} key={key} counts={counts}")
        print()


if __name__ == "__main__":
    main()
