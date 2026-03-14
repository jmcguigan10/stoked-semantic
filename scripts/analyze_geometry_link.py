from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the layerwise link between higher-order performance gains and raw curl."
    )
    parser.add_argument("result_dir", type=Path, help="Path to a result directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_dir = args.result_dir
    probe_path = _existing(
        result_dir / "probe_accuracy_all_runs.csv",
        result_dir / "probe_accuracy_by_layer.csv",
    )
    diagnostic_path = _existing(
        result_dir / "diagnostics_all_runs.csv",
        result_dir / "diagnostics_by_layer.csv",
    )
    if probe_path is None or diagnostic_path is None:
        raise FileNotFoundError(f"Expected probe and diagnostic CSVs under {result_dir}")

    probe_rows = list(csv.DictReader(probe_path.open()))
    diagnostic_rows = list(csv.DictReader(diagnostic_path.open()))
    link_rows = _build_link_rows(probe_rows=probe_rows, diagnostic_rows=diagnostic_rows)
    if not link_rows:
        raise ValueError(f"No compatible rows found under {result_dir}")

    by_seed_rows = _seed_summary_rows(link_rows)
    summary_rows = _aggregate_summary_rows(by_seed_rows)

    _write_csv(result_dir / "geometry_link_by_seed.csv", by_seed_rows)
    _write_csv(result_dir / "geometry_link_summary.csv", summary_rows)
    _write_json(
        result_dir / "geometry_link_summary.json",
        {
            "by_seed": by_seed_rows,
            "summary": summary_rows,
        },
    )
    _plot_link_curves(link_rows=link_rows, path=result_dir / "geometry_link_curves.png")
    print(json.dumps({"result_dir": str(result_dir), "rows": len(by_seed_rows)}, indent=2))


def _build_link_rows(
    probe_rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
) -> list[dict[str, float | int | str]]:
    pairwise = {}
    pairwise_plus_triadic = {}
    triadic = {}
    raw_curl = {}

    for row in probe_rows:
        key = (int(row["run_seed"]), row["variant_name"], int(row["layer_index"]))
        probe_name = row["probe_name"]
        value = float(row["test_accuracy"])
        if probe_name == "pairwise":
            pairwise[key] = value
        elif probe_name == "pairwise_plus_triadic":
            pairwise_plus_triadic[key] = value
        elif probe_name == "triadic":
            triadic[key] = value

    for row in diagnostic_rows:
        if row["probe_name"] != "raw_skew_bilinear_r8":
            continue
        key = (int(row["run_seed"]), row["variant_name"], int(row["layer_index"]))
        raw_curl[key] = float(row["curl_energy_positional_mean"])

    keys = sorted(set(pairwise) & set(pairwise_plus_triadic) & set(triadic) & set(raw_curl))
    rows: list[dict[str, float | int | str]] = []
    for run_seed, variant_name, layer_index in keys:
        pairwise_acc = pairwise[(run_seed, variant_name, layer_index)]
        rows.append(
            {
                "run_seed": run_seed,
                "variant_name": variant_name,
                "layer_index": layer_index,
                "pairwise_accuracy": pairwise_acc,
                "pairwise_plus_triadic_accuracy": pairwise_plus_triadic[(run_seed, variant_name, layer_index)],
                "triadic_accuracy": triadic[(run_seed, variant_name, layer_index)],
                "delta_pairwise_plus_triadic": pairwise_plus_triadic[(run_seed, variant_name, layer_index)]
                - pairwise_acc,
                "delta_triadic": triadic[(run_seed, variant_name, layer_index)] - pairwise_acc,
                "raw_skew_bilinear_positional_curl": raw_curl[(run_seed, variant_name, layer_index)],
            }
        )
    return rows


def _seed_summary_rows(
    link_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in link_rows:
        for delta_name in ("delta_pairwise_plus_triadic", "delta_triadic"):
            grouped[(int(row["run_seed"]), str(row["variant_name"]), delta_name)].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for (run_seed, variant_name, delta_name), rows in sorted(grouped.items()):
        xs = [float(row[delta_name]) for row in rows]
        ys = [float(row["raw_skew_bilinear_positional_curl"]) for row in rows]
        summary_rows.append(
            {
                "run_seed": run_seed,
                "variant_name": variant_name,
                "delta_name": delta_name,
                "layer_count": len(rows),
                "delta_mean": _mean(xs),
                "delta_std": _std(xs),
                "curl_mean": _mean(ys),
                "curl_std": _std(ys),
                "pearson_r": _pearson(xs, ys),
                "spearman_r": _spearman(xs, ys),
            }
        )
    return summary_rows


def _aggregate_summary_rows(
    by_seed_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in by_seed_rows:
        grouped[(str(row["variant_name"]), str(row["delta_name"]))].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for (variant_name, delta_name), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "variant_name": variant_name,
                "delta_name": delta_name,
                "seed_count": len(rows),
                "pearson_r_mean": _mean(float(row["pearson_r"]) for row in rows),
                "pearson_r_std": _std(float(row["pearson_r"]) for row in rows),
                "spearman_r_mean": _mean(float(row["spearman_r"]) for row in rows),
                "spearman_r_std": _std(float(row["spearman_r"]) for row in rows),
                "delta_mean": _mean(float(row["delta_mean"]) for row in rows),
                "curl_mean": _mean(float(row["curl_mean"]) for row in rows),
            }
        )
    return summary_rows


def _plot_link_curves(
    link_rows: list[dict[str, float | int | str]],
    path: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#000000",
            "axes.facecolor": "#000000",
            "savefig.facecolor": "#000000",
            "axes.edgecolor": "#14F1FF",
            "axes.labelcolor": "#E6E6E6",
            "xtick.color": "#E6E6E6",
            "ytick.color": "#E6E6E6",
            "grid.color": "#0A7E8C",
            "grid.linestyle": ":",
            "grid.alpha": 0.35,
            "axes.grid": True,
            "legend.facecolor": "#0b0b0b",
            "legend.edgecolor": "#FE53BB",
            "legend.framealpha": 0.65,
            "text.color": "#E6E6E6",
            "axes.titlecolor": "#E6E6E6",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in link_rows:
        grouped[str(row["variant_name"])].append(row)

    for axis, variant_name in zip(axes, sorted(grouped), strict=True):
        rows = grouped[variant_name]
        by_layer: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
        for row in rows:
            by_layer[int(row["layer_index"])].append(row)
        layers = sorted(by_layer)
        delta_curve = [
            _mean(float(row["delta_pairwise_plus_triadic"]) for row in by_layer[layer])
            for layer in layers
        ]
        curl_curve = [
            _mean(float(row["raw_skew_bilinear_positional_curl"]) for row in by_layer[layer])
            for layer in layers
        ]
        axis.plot(
            layers,
            _normalize(delta_curve),
            color="#00FF85",
            label="delta(pairwise_plus_triadic - pairwise)",
        )
        axis.plot(
            layers,
            _normalize(curl_curve),
            color="#FE53BB",
            label="raw_skew_bilinear positional curl",
        )
        axis.set_title(variant_name)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Normalized value")
        axis.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [0.5 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in centered_x) * sum(y * y for y in centered_y))
    if math.isclose(denom, 0.0):
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / denom


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and math.isclose(indexed[end][1], indexed[start][1]):
            end += 1
        avg_rank = (start + end - 1) / 2 + 1
        for position in range(start, end):
            ranks[indexed[position][0]] = avg_rank
        start = end
    return ranks


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    main()
