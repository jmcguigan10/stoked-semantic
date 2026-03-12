from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a phase-1 result bundle.")
    parser.add_argument("result_dir", type=Path, help="Path to a result directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_dir = args.result_dir
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        metadata = payload.get("metadata", {})
        if metadata:
            print("Metadata")
            print(json.dumps(metadata, indent=2))
            print()

    probe_path = _existing(
        result_dir / "probe_accuracy_aggregate.csv",
        result_dir / "probe_accuracy_by_layer.csv",
    )
    group_path = _existing(
        result_dir / "template_group_accuracy_aggregate.csv",
        result_dir / "template_group_accuracy_by_layer.csv",
    )

    probe_rows = list(csv.DictReader(probe_path.open()))
    aggregated = "test_accuracy_mean" in probe_rows[0]
    _print_probe_summary(probe_rows, aggregated=aggregated)

    if group_path is not None:
        group_rows = list(csv.DictReader(group_path.open()))
        _print_group_summary(group_rows, aggregated="test_accuracy_mean" in group_rows[0])


def _print_probe_summary(rows: list[dict[str, str]], aggregated: bool) -> None:
    raw_key = "test_accuracy_mean" if aggregated else "test_accuracy"
    pinv_key = (
        "test_accuracy_polarity_invariant_mean"
        if aggregated
        else "test_accuracy_polarity_invariant"
    )
    std_key = "test_accuracy_std" if aggregated else None

    print("Probe Summary")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant_name"]].append(row)

    for variant_name in sorted(grouped):
        print(f"  {variant_name}")
        probe_rows = grouped[variant_name]
        by_probe: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in probe_rows:
            by_probe[row["probe_name"]].append(row)
        for probe_name in ("query_only", "exact", "pairwise", "triadic"):
            vals = by_probe.get(probe_name)
            if not vals:
                continue
            avg_raw = _mean(float(row[raw_key]) for row in vals)
            avg_pinv = _mean(float(row[pinv_key]) for row in vals)
            best = max(vals, key=lambda row: float(row[raw_key]))
            line = (
                f"    {probe_name}: avg={avg_raw:.4f} "
                f"pinv_avg={avg_pinv:.4f} "
                f"best={float(best[raw_key]):.4f}@L{best['layer_index']}"
            )
            if std_key is not None:
                line += f" sd={float(best[std_key]):.4f}"
            print(line)

        exact_rows = by_probe.get("exact")
        pairwise_rows = by_probe.get("pairwise")
        triadic_rows = by_probe.get("triadic")
        if exact_rows and pairwise_rows and triadic_rows:
            pair_exact_gap = _mean(
                float(p[raw_key]) - float(e[raw_key])
                for p, e in zip(
                    sorted(pairwise_rows, key=lambda row: int(row["layer_index"])),
                    sorted(exact_rows, key=lambda row: int(row["layer_index"])),
                    strict=True,
                )
            )
            tri_pair_gap = _mean(
                float(t[raw_key]) - float(p[raw_key])
                for t, p in zip(
                    sorted(triadic_rows, key=lambda row: int(row["layer_index"])),
                    sorted(pairwise_rows, key=lambda row: int(row["layer_index"])),
                    strict=True,
                )
            )
            print(
                f"    avg gaps: pairwise-exact={pair_exact_gap:+.4f} "
                f"triadic-pairwise={tri_pair_gap:+.4f}"
            )
    print()


def _print_group_summary(rows: list[dict[str, str]], aggregated: bool) -> None:
    raw_key = "test_accuracy_mean" if aggregated else "test_accuracy"
    pinv_key = (
        "test_accuracy_polarity_invariant_mean"
        if aggregated
        else "test_accuracy_polarity_invariant"
    )
    gap_rows = [
        row
        for row in rows
        if row["group_type"] == "template_family_id"
    ]
    gap_rows.sort(
        key=lambda row: float(row[pinv_key]) - float(row[raw_key]),
        reverse=True,
    )

    if not gap_rows:
        return

    print("Largest Template-Family Polarity Gaps")
    for row in gap_rows[:12]:
        gap = float(row[pinv_key]) - float(row[raw_key])
        print(
            "  "
            f"{row['variant_name']} {row['probe_name']} "
            f"L{row['layer_index']} {row['group_name']}: "
            f"raw={float(row[raw_key]):.4f} "
            f"pinv={float(row[pinv_key]):.4f} "
            f"gap={gap:+.4f}"
        )


def _existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _mean(values: list[float] | tuple[float, ...] | object) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
