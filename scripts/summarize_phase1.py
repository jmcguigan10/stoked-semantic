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
    diagnostic_path = _existing(
        result_dir / "diagnostics_aggregate.csv",
        result_dir / "diagnostics_by_layer.csv",
    )

    if probe_path is None:
        probe_rows = _seed_rows(result_dir=result_dir, filename="probe_accuracy_by_layer.csv")
        if not probe_rows:
            raise FileNotFoundError(f"No probe results found under {result_dir}")
        _print_probe_summary(probe_rows, aggregated=False)

        diagnostic_rows = _seed_rows(result_dir=result_dir, filename="diagnostics_by_layer.csv")
        if diagnostic_rows:
            _print_diagnostic_summary(diagnostic_rows, aggregated=False)

        group_rows = _seed_rows(result_dir=result_dir, filename="template_group_accuracy_by_layer.csv")
        if group_rows:
            _print_group_summary(group_rows, aggregated=False)
        return

    probe_rows = list(csv.DictReader(probe_path.open()))
    aggregated = "test_accuracy_mean" in probe_rows[0]
    _print_probe_summary(probe_rows, aggregated=aggregated)

    if diagnostic_path is not None:
        diagnostic_rows = list(csv.DictReader(diagnostic_path.open()))
        if diagnostic_rows:
            _print_diagnostic_summary(
                diagnostic_rows,
                aggregated="exactness_std" in diagnostic_rows[0],
            )

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
        for probe_name in sorted(by_probe, key=_probe_sort_key):
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

        exact_family_rows = [
            row
            for probe_name, probe_rows in by_probe.items()
            if _probe_family(probe_name) == "exact"
            for row in probe_rows
        ]
        if exact_family_rows:
            exact_by_probe: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in exact_family_rows:
                exact_by_probe[row["probe_name"]].append(row)
            best_exact_probe, best_exact_rows = max(
                exact_by_probe.items(),
                key=lambda item: _mean(float(row[raw_key]) for row in item[1]),
            )
            print(
                "    best exact family member: "
                f"{best_exact_probe} avg={_mean(float(row[raw_key]) for row in best_exact_rows):.4f} "
                f"pinv_avg={_mean(float(row[pinv_key]) for row in best_exact_rows):.4f}"
            )

        pairwise_rows = by_probe.get("pairwise")
        pairwise_plus_query_context_rows = by_probe.get("pairwise_plus_query_context")
        pairwise_plus_triplet_mlp_rows = by_probe.get("pairwise_plus_triplet_mlp")
        pairwise_plus_triadic_rows = by_probe.get("pairwise_plus_triadic")
        triadic_rows = by_probe.get("triadic")
        if exact_family_rows and pairwise_rows and triadic_rows:
            exact_by_layer: dict[int, list[float]] = defaultdict(list)
            for row in exact_family_rows:
                exact_by_layer[int(row["layer_index"])].append(float(row[raw_key]))
            pair_exact_gap = _mean(
                float(row[raw_key]) - max(exact_by_layer.get(int(row["layer_index"]), [0.0]))
                for row in sorted(pairwise_rows, key=lambda row: int(row["layer_index"]))
            )
            tri_pair_gap = _mean(
                float(t[raw_key]) - float(p[raw_key])
                for t, p in zip(
                    sorted(triadic_rows, key=lambda row: int(row["layer_index"])),
                    sorted(pairwise_rows, key=lambda row: int(row["layer_index"])),
                    strict=True,
                )
            )
            line = f"    avg gaps: pairwise-best_exact={pair_exact_gap:+.4f} triadic-pairwise={tri_pair_gap:+.4f}"
            if pairwise_plus_query_context_rows:
                pair_ctx_pair_gap = _mean(
                    float(pc[raw_key]) - float(p[raw_key])
                    for pc, p in zip(
                        sorted(pairwise_plus_query_context_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" pair+ctx-pairwise={pair_ctx_pair_gap:+.4f}"
            if pairwise_plus_triplet_mlp_rows and pairwise_plus_query_context_rows:
                pair_mlp_ctx_gap = _mean(
                    float(pm[raw_key]) - float(pc[raw_key])
                    for pm, pc in zip(
                        sorted(pairwise_plus_triplet_mlp_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_plus_query_context_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" pair+mlp-pair+ctx={pair_mlp_ctx_gap:+.4f}"
            if pairwise_plus_triadic_rows:
                pair_tri_pair_gap = _mean(
                    float(pt[raw_key]) - float(p[raw_key])
                    for pt, p in zip(
                        sorted(pairwise_plus_triadic_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" pair+triadic-pairwise={pair_tri_pair_gap:+.4f}"
            if pairwise_plus_triadic_rows and pairwise_plus_triplet_mlp_rows:
                pair_tri_mlp_gap = _mean(
                    float(pt[raw_key]) - float(pm[raw_key])
                    for pt, pm in zip(
                        sorted(pairwise_plus_triadic_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_plus_triplet_mlp_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" pair+triadic-pair+mlp={pair_tri_mlp_gap:+.4f}"
            if pairwise_plus_triadic_rows:
                tri_pair_tri_gap = _mean(
                    float(t[raw_key]) - float(pt[raw_key])
                    for t, pt in zip(
                        sorted(triadic_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_plus_triadic_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" triadic-pair+triadic={tri_pair_tri_gap:+.4f}"
            print(line)
        elif exact_family_rows and pairwise_rows:
            exact_by_layer: dict[int, list[float]] = defaultdict(list)
            for row in exact_family_rows:
                exact_by_layer[int(row["layer_index"])].append(float(row[raw_key]))
            pair_exact_gap = _mean(
                float(row[raw_key]) - max(exact_by_layer.get(int(row["layer_index"]), [0.0]))
                for row in sorted(pairwise_rows, key=lambda row: int(row["layer_index"]))
            )
            line = f"    avg gap vs best exact@layer: pairwise-best_exact={pair_exact_gap:+.4f}"
            if pairwise_plus_query_context_rows:
                pair_ctx_pair_gap = _mean(
                    float(pc[raw_key]) - float(p[raw_key])
                    for pc, p in zip(
                        sorted(pairwise_plus_query_context_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" pair+ctx-pairwise={pair_ctx_pair_gap:+.4f}"
            if pairwise_plus_triplet_mlp_rows and pairwise_plus_query_context_rows:
                pair_mlp_ctx_gap = _mean(
                    float(pm[raw_key]) - float(pc[raw_key])
                    for pm, pc in zip(
                        sorted(pairwise_plus_triplet_mlp_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_plus_query_context_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" pair+mlp-pair+ctx={pair_mlp_ctx_gap:+.4f}"
            if pairwise_plus_triadic_rows:
                pair_tri_pair_gap = _mean(
                    float(pt[raw_key]) - float(p[raw_key])
                    for pt, p in zip(
                        sorted(pairwise_plus_triadic_rows, key=lambda row: int(row["layer_index"])),
                        sorted(pairwise_rows, key=lambda row: int(row["layer_index"])),
                        strict=True,
                    )
                )
                line += f" pair+triadic-pairwise={pair_tri_pair_gap:+.4f}"
            print(line)
    print()


def _print_group_summary(rows: list[dict[str, str]], aggregated: bool) -> None:
    raw_key = "test_accuracy_mean" if aggregated else "test_accuracy"
    pinv_key = (
        "test_accuracy_polarity_invariant_mean"
        if aggregated
        else "test_accuracy_polarity_invariant"
    )
    relation_rows = [row for row in rows if row["group_type"] == "relation_id"]
    if relation_rows:
        print("Relation Averages")
        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in relation_rows:
            grouped[(row["variant_name"], row["probe_name"], row["group_name"])].append(row)
        for variant_name in sorted({row["variant_name"] for row in relation_rows}):
            print(f"  {variant_name}")
            probe_names = {
                row["probe_name"]
                for row in relation_rows
                if row["variant_name"] == variant_name
            }
            for probe_name in sorted(probe_names, key=_probe_sort_key):
                parts = []
                relation_names = {
                    row["group_name"]
                    for row in relation_rows
                    if row["variant_name"] == variant_name and row["probe_name"] == probe_name
                }
                for relation_name in sorted(relation_names):
                    vals = grouped[(variant_name, probe_name, relation_name)]
                    avg_raw = _mean(float(row[raw_key]) for row in vals)
                    parts.append(f"{relation_name}={avg_raw:.4f}")
                if parts:
                    print(f"    {probe_name}: " + " ".join(parts))
        print()

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
        gap_rows = []

    if gap_rows:
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

    query_rows = [row for row in rows if row["group_type"] == "query_type"]
    if query_rows:
        print()
        print("Query-Type Averages")
        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in query_rows:
            grouped[(row["variant_name"], row["probe_name"], row["group_name"])].append(row)
        query_type_order = sorted(
            {row["group_name"] for row in query_rows},
            key=_query_type_sort_key,
        )
        for variant_name in sorted({row["variant_name"] for row in query_rows}):
            print(f"  {variant_name}")
            for probe_name in sorted(
                {row["probe_name"] for row in query_rows if row["variant_name"] == variant_name},
                key=_probe_sort_key,
            ):
                parts = []
                for query_type in query_type_order:
                    vals = grouped.get((variant_name, probe_name, query_type))
                    if not vals:
                        continue
                    avg_raw = _mean(float(row[raw_key]) for row in vals)
                    parts.append(f"{query_type}={avg_raw:.4f}")
                if parts:
                    print(f"    {probe_name}: " + " ".join(parts))


def _print_diagnostic_summary(rows: list[dict[str, str]], aggregated: bool) -> None:
    exact_key = "exactness_mean"
    curl_key = "curl_energy_positional_mean"
    std_key = "curl_energy_positional_std" if aggregated else None

    print("Diagnostic Summary")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant_name"]].append(row)

    for variant_name in sorted(grouped):
        print(f"  {variant_name}")
        by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in grouped[variant_name]:
            by_name[row["probe_name"]].append(row)
        for probe_name in sorted(by_name, key=_probe_sort_key):
            vals = by_name[probe_name]
            avg_exact = _mean(float(row[exact_key]) for row in vals)
            avg_curl = _mean(float(row[curl_key]) for row in vals)
            peak = max(vals, key=lambda row: float(row[curl_key]))
            line = (
                f"    {probe_name}: avg_exact={avg_exact:.4f} "
                f"avg_pos_curl={avg_curl:.4f} "
                f"peak_pos_curl={float(peak[curl_key]):.4f}@L{peak['layer_index']}"
            )
            if std_key is not None:
                line += f" sd={float(peak[std_key]):.4f}"
            print(line)
    print()


def _existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _seed_rows(result_dir: Path, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed_dir in sorted(result_dir.glob("seed_*")):
        path = seed_dir / filename
        if not path.exists():
            continue
        rows.extend(csv.DictReader(path.open()))
    return rows


def _mean(values: list[float] | tuple[float, ...] | object) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _probe_sort_key(probe_name: str) -> tuple[int, int, str]:
    if probe_name == "query_only":
        return (0, 0, probe_name)
    if probe_name.startswith("exact_r"):
        return (1, _rank_from_probe_name(probe_name), probe_name)
    if probe_name == "exact":
        return (1, 0, probe_name)
    if probe_name == "pairwise":
        return (2, 0, probe_name)
    if probe_name == "pairwise_plus_query_context":
        return (3, 0, probe_name)
    if probe_name == "pairwise_plus_triplet_mlp":
        return (4, 0, probe_name)
    if probe_name == "pairwise_plus_triadic":
        return (5, 0, probe_name)
    if probe_name == "triadic":
        return (6, 0, probe_name)
    if probe_name.startswith("raw_projection"):
        return (7, 0, probe_name)
    if probe_name.startswith("raw_skew_bilinear"):
        return (8, 0, probe_name)
    return (9, 0, probe_name)


def _rank_from_probe_name(probe_name: str) -> int:
    try:
        return int(probe_name.split("_r", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0


def _probe_family(probe_name: str) -> str:
    if probe_name.startswith("exact"):
        return "exact"
    if probe_name.startswith("raw_projection"):
        return "raw_projection"
    if probe_name.startswith("raw_skew_bilinear"):
        return "raw_skew_bilinear"
    return probe_name


def _query_type_sort_key(query_type: str) -> tuple[int, int, str]:
    preferred = {
        "adjacent_forward": (0, 0),
        "adjacent_reversed": (0, 1),
        "closure_forward": (1, 0),
        "closure_reversed": (1, 1),
    }
    if query_type in preferred:
        major, minor = preferred[query_type]
        return (major, minor, query_type)
    if query_type.startswith("path_"):
        pieces = query_type.split("_")
        if len(pieces) == 3 and pieces[1].isdigit():
            direction_rank = 0 if pieces[2] == "forward" else 1
            return (2, int(pieces[1]) * 2 + direction_rank, query_type)
    return (3, 0, query_type)


if __name__ == "__main__":
    main()
