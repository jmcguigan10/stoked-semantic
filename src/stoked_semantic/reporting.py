from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

from stoked_semantic.diagnostics import DiagnosticSummary
from stoked_semantic.training import ProbeRunResult

CYBER_COLORS = {
    "axis": "#14F1FF",
    "cyan": "#08F7FE",
    "magenta": "#FE53BB",
    "lime": "#00FF85",
    "gold": "#F5D300",
    "orange": "#FF7A00",
    "violet": "#7A5CFF",
    "gray": "#8B949E",
}

SERIES_COLORS = {
    ("pretrained", "query_only"): CYBER_COLORS["gray"],
    ("pretrained", "exact"): CYBER_COLORS["cyan"],
    ("pretrained", "pairwise"): CYBER_COLORS["magenta"],
    ("pretrained", "triadic"): CYBER_COLORS["gold"],
    ("random_control", "query_only"): "#4D5560",
    ("random_control", "exact"): CYBER_COLORS["lime"],
    ("random_control", "pairwise"): CYBER_COLORS["orange"],
    ("random_control", "triadic"): CYBER_COLORS["violet"],
}


class ReportWriter:
    def __init__(self, output_dir: Path, write_plots: bool = True):
        self.output_dir = output_dir
        self.write_plots = write_plots
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.write_plots:
            self._configure_plot_theme()

    def write(
        self,
        probe_results: list[ProbeRunResult],
        diagnostics: list[DiagnosticSummary],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._write_probe_rows(
            probe_results=probe_results,
            path=self.output_dir / "probe_accuracy_by_layer.csv",
        )
        self._write_diagnostic_rows(
            diagnostics=diagnostics,
            path=self.output_dir / "diagnostics_by_layer.csv",
        )
        group_rows = self._group_rows(probe_results)
        self._write_group_rows(
            group_rows=group_rows,
            path=self.output_dir / "template_group_accuracy_by_layer.csv",
        )
        self._write_summary_json(
            path=self.output_dir / "summary.json",
            probe_payload=[
                {
                    "run_seed": result.run_seed,
                    "variant_name": result.variant_name,
                    "layer_index": result.layer_index,
                    "probe_name": result.probe_name,
                    "rank": result.rank,
                    "num_parameters": result.num_parameters,
                    "train_accuracy": result.train_accuracy,
                    "test_accuracy": result.test_accuracy,
                    "test_accuracy_polarity_invariant": result.test_accuracy_polarity_invariant,
                }
                for result in probe_results
            ],
            diagnostic_payload=[
                {
                    "run_seed": summary.run_seed,
                    "variant_name": summary.variant_name,
                    "layer_index": summary.layer_index,
                    "probe_name": summary.probe_name,
                    "exactness_mean": summary.exactness_mean,
                    "curl_energy_mean": summary.curl_energy_mean,
                    "exactness_positional_mean": summary.exactness_positional_mean,
                    "curl_energy_positional_mean": summary.curl_energy_positional_mean,
                }
                for summary in diagnostics
            ],
            group_payload=group_rows,
            metadata=metadata,
        )
        if self.write_plots:
            self._plot_probe_accuracy(
                probe_results=probe_results,
                path=self.output_dir / "probe_accuracy_by_layer.png",
            )
            self._plot_probe_accuracy_polarity_invariant(
                probe_results=probe_results,
                path=self.output_dir / "probe_accuracy_polarity_invariant_by_layer.png",
            )
            self._plot_diagnostics(
                diagnostics=diagnostics,
                path=self.output_dir / "diagnostics_by_layer.png",
            )

    def write_aggregate(
        self,
        probe_results: list[ProbeRunResult],
        diagnostics: list[DiagnosticSummary],
        run_summaries: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        probe_aggregate = self._aggregate_probe_results(probe_results)
        diagnostic_aggregate = self._aggregate_diagnostics(diagnostics)
        group_rows = self._group_rows(probe_results)
        group_aggregate = self._aggregate_group_rows(group_rows)

        self._write_probe_rows(
            probe_results=probe_results,
            path=self.output_dir / "probe_accuracy_all_runs.csv",
        )
        self._write_diagnostic_rows(
            diagnostics=diagnostics,
            path=self.output_dir / "diagnostics_all_runs.csv",
        )
        self._write_group_rows(
            group_rows=group_rows,
            path=self.output_dir / "template_group_accuracy_all_runs.csv",
        )
        self._write_probe_aggregate_rows(
            aggregate_rows=probe_aggregate,
            path=self.output_dir / "probe_accuracy_aggregate.csv",
        )
        self._write_diagnostic_aggregate_rows(
            aggregate_rows=diagnostic_aggregate,
            path=self.output_dir / "diagnostics_aggregate.csv",
        )
        self._write_group_aggregate_rows(
            aggregate_rows=group_aggregate,
            path=self.output_dir / "template_group_accuracy_aggregate.csv",
        )
        self._write_summary_json(
            path=self.output_dir / "summary.json",
            probe_payload=probe_aggregate,
            diagnostic_payload=diagnostic_aggregate,
            group_payload=group_aggregate,
            metadata={
                **(metadata or {}),
                "run_summaries": run_summaries,
            },
        )
        if self.write_plots:
            self._plot_probe_accuracy_aggregate(
                aggregate_rows=probe_aggregate,
                path=self.output_dir / "probe_accuracy_aggregate.png",
            )
            self._plot_probe_accuracy_polarity_invariant_aggregate(
                aggregate_rows=probe_aggregate,
                path=self.output_dir / "probe_accuracy_polarity_invariant_aggregate.png",
            )
            self._plot_diagnostics_aggregate(
                aggregate_rows=diagnostic_aggregate,
                path=self.output_dir / "diagnostics_aggregate.png",
            )

    def _write_probe_rows(self, probe_results: list[ProbeRunResult], path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_seed",
                    "variant_name",
                    "layer_index",
                    "probe_name",
                    "rank",
                    "num_parameters",
                    "train_accuracy",
                    "test_accuracy",
                    "test_accuracy_polarity_invariant",
                ],
            )
            writer.writeheader()
            for result in probe_results:
                writer.writerow(
                    {
                        "run_seed": result.run_seed,
                        "variant_name": result.variant_name,
                        "layer_index": result.layer_index,
                        "probe_name": result.probe_name,
                        "rank": result.rank,
                        "num_parameters": result.num_parameters,
                        "train_accuracy": result.train_accuracy,
                        "test_accuracy": result.test_accuracy,
                        "test_accuracy_polarity_invariant": result.test_accuracy_polarity_invariant,
                    }
                )

    def _write_diagnostic_rows(self, diagnostics: list[DiagnosticSummary], path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_seed",
                    "variant_name",
                    "layer_index",
                    "probe_name",
                    "exactness_mean",
                    "curl_energy_mean",
                    "exactness_positional_mean",
                    "curl_energy_positional_mean",
                ],
            )
            writer.writeheader()
            for summary in diagnostics:
                writer.writerow(
                    {
                        "run_seed": summary.run_seed,
                        "variant_name": summary.variant_name,
                        "layer_index": summary.layer_index,
                        "probe_name": summary.probe_name,
                        "exactness_mean": summary.exactness_mean,
                        "curl_energy_mean": summary.curl_energy_mean,
                        "exactness_positional_mean": summary.exactness_positional_mean,
                        "curl_energy_positional_mean": summary.curl_energy_positional_mean,
                    }
                )

    def _write_group_rows(self, group_rows: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_seed",
                    "variant_name",
                    "layer_index",
                    "probe_name",
                    "group_type",
                    "group_name",
                    "example_count",
                    "test_accuracy",
                    "test_accuracy_polarity_invariant",
                ],
            )
            writer.writeheader()
            writer.writerows(group_rows)

    def _write_probe_aggregate_rows(self, aggregate_rows: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "variant_name",
                    "layer_index",
                    "probe_name",
                    "rank",
                    "num_parameters_mean",
                    "num_parameters_std",
                    "run_count",
                    "train_accuracy_mean",
                    "train_accuracy_std",
                    "test_accuracy_mean",
                    "test_accuracy_std",
                    "test_accuracy_polarity_invariant_mean",
                    "test_accuracy_polarity_invariant_std",
                ],
            )
            writer.writeheader()
            writer.writerows(aggregate_rows)

    def _write_diagnostic_aggregate_rows(self, aggregate_rows: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "variant_name",
                    "layer_index",
                    "probe_name",
                    "run_count",
                    "exactness_mean",
                    "exactness_std",
                    "curl_energy_mean",
                    "curl_energy_std",
                    "exactness_positional_mean",
                    "exactness_positional_std",
                    "curl_energy_positional_mean",
                    "curl_energy_positional_std",
                ],
            )
            writer.writeheader()
            writer.writerows(aggregate_rows)

    def _write_group_aggregate_rows(self, aggregate_rows: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "variant_name",
                    "layer_index",
                    "probe_name",
                    "group_type",
                    "group_name",
                    "run_count",
                    "example_count_mean",
                    "example_count_std",
                    "test_accuracy_mean",
                    "test_accuracy_std",
                    "test_accuracy_polarity_invariant_mean",
                    "test_accuracy_polarity_invariant_std",
                ],
            )
            writer.writeheader()
            writer.writerows(aggregate_rows)

    def _write_summary_json(
        self,
        path: Path,
        probe_payload: list[dict[str, Any]],
        diagnostic_payload: list[dict[str, Any]],
        group_payload: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
    ) -> None:
        payload = {
            "metadata": metadata or {},
            "probe_results": probe_payload,
            "diagnostics": diagnostic_payload,
            "group_results": group_payload,
        }
        path.write_text(json.dumps(payload, indent=2))

    def _aggregate_probe_results(self, probe_results: list[ProbeRunResult]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int, str], list[ProbeRunResult]] = defaultdict(list)
        for result in probe_results:
            grouped[(result.variant_name, result.layer_index, result.probe_name)].append(result)

        rows: list[dict[str, Any]] = []
        for (variant_name, layer_index, probe_name), group in sorted(grouped.items()):
            rows.append(
                {
                    "variant_name": variant_name,
                    "layer_index": layer_index,
                    "probe_name": probe_name,
                    "rank": group[0].rank,
                    "num_parameters_mean": mean(result.num_parameters for result in group),
                    "num_parameters_std": self._std(result.num_parameters for result in group),
                    "run_count": len(group),
                    "train_accuracy_mean": mean(result.train_accuracy for result in group),
                    "train_accuracy_std": self._std(result.train_accuracy for result in group),
                    "test_accuracy_mean": mean(result.test_accuracy for result in group),
                    "test_accuracy_std": self._std(result.test_accuracy for result in group),
                    "test_accuracy_polarity_invariant_mean": mean(
                        result.test_accuracy_polarity_invariant for result in group
                    ),
                    "test_accuracy_polarity_invariant_std": self._std(
                        result.test_accuracy_polarity_invariant for result in group
                    ),
                }
            )
        return rows

    def _aggregate_diagnostics(self, diagnostics: list[DiagnosticSummary]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int, str], list[DiagnosticSummary]] = defaultdict(list)
        for summary in diagnostics:
            grouped[(summary.variant_name, summary.layer_index, summary.probe_name)].append(summary)

        rows: list[dict[str, Any]] = []
        for (variant_name, layer_index, probe_name), group in sorted(grouped.items()):
            rows.append(
                {
                    "variant_name": variant_name,
                    "layer_index": layer_index,
                    "probe_name": probe_name,
                    "run_count": len(group),
                    "exactness_mean": mean(item.exactness_mean for item in group),
                    "exactness_std": self._std(item.exactness_mean for item in group),
                    "curl_energy_mean": mean(item.curl_energy_mean for item in group),
                    "curl_energy_std": self._std(item.curl_energy_mean for item in group),
                    "exactness_positional_mean": mean(item.exactness_positional_mean for item in group),
                    "exactness_positional_std": self._std(item.exactness_positional_mean for item in group),
                    "curl_energy_positional_mean": mean(item.curl_energy_positional_mean for item in group),
                    "curl_energy_positional_std": self._std(
                        item.curl_energy_positional_mean for item in group
                    ),
                }
            )
        return rows

    def _group_rows(self, probe_results: list[ProbeRunResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in probe_results:
            for group_evaluation in result.test_group_evaluations:
                rows.append(
                    {
                        "run_seed": result.run_seed,
                        "variant_name": result.variant_name,
                        "layer_index": result.layer_index,
                        "probe_name": result.probe_name,
                        "group_type": group_evaluation.group_type,
                        "group_name": group_evaluation.group_name,
                        "example_count": group_evaluation.example_count,
                        "test_accuracy": group_evaluation.accuracy,
                        "test_accuracy_polarity_invariant": (
                            group_evaluation.polarity_invariant_accuracy
                        ),
                    }
                )
        return rows

    def _aggregate_group_rows(self, group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            grouped[
                (
                    row["variant_name"],
                    row["layer_index"],
                    row["probe_name"],
                    row["group_type"],
                    row["group_name"],
                )
            ].append(row)

        rows: list[dict[str, Any]] = []
        for key, group in sorted(grouped.items()):
            variant_name, layer_index, probe_name, group_type, group_name = key
            rows.append(
                {
                    "variant_name": variant_name,
                    "layer_index": layer_index,
                    "probe_name": probe_name,
                    "group_type": group_type,
                    "group_name": group_name,
                    "run_count": len(group),
                    "example_count_mean": mean(row["example_count"] for row in group),
                    "example_count_std": self._std(row["example_count"] for row in group),
                    "test_accuracy_mean": mean(row["test_accuracy"] for row in group),
                    "test_accuracy_std": self._std(row["test_accuracy"] for row in group),
                    "test_accuracy_polarity_invariant_mean": mean(
                        row["test_accuracy_polarity_invariant"] for row in group
                    ),
                    "test_accuracy_polarity_invariant_std": self._std(
                        row["test_accuracy_polarity_invariant"] for row in group
                    ),
                }
            )
        return rows

    def _plot_probe_accuracy(self, probe_results: list[ProbeRunResult], path: Path) -> None:
        grouped: dict[tuple[str, str], list[ProbeRunResult]] = defaultdict(list)
        for result in probe_results:
            grouped[(result.variant_name, result.probe_name)].append(result)

        fig, ax = plt.subplots(figsize=(9, 5))
        for (variant_name, probe_name), results in grouped.items():
            results = sorted(results, key=lambda item: item.layer_index)
            self._plot_neon_line(
                ax=ax,
                xs=[result.layer_index for result in results],
                ys=[result.test_accuracy for result in results],
                color=self._series_color(variant_name=variant_name, probe_name=probe_name),
                label=f"{variant_name}:{probe_name}",
            )
        self._style_axes(ax=ax, ylabel="Accuracy")
        ax.set_title("Probe Test Accuracy by Layer")
        ax.set_xlabel("Layer")
        ax.set_ylim(0.0, 1.05)
        self._style_legend(fig=fig, source_ax=ax, ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_diagnostics(self, diagnostics: list[DiagnosticSummary], path: Path) -> None:
        if not diagnostics:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        grouped: dict[tuple[str, str], list[DiagnosticSummary]] = defaultdict(list)
        for summary in diagnostics:
            grouped[(summary.variant_name, summary.probe_name)].append(summary)

        for (variant_name, probe_name), summaries in grouped.items():
            summaries = sorted(summaries, key=lambda item: item.layer_index)
            color = self._series_color(variant_name=variant_name, probe_name=probe_name)
            label = f"{variant_name}:{probe_name}"
            xs = [summary.layer_index for summary in summaries]
            self._plot_neon_line(
                ax=axes[0],
                xs=xs,
                ys=[summary.exactness_mean for summary in summaries],
                color=color,
                label=label,
            )
            self._plot_neon_line(
                ax=axes[1],
                xs=xs,
                ys=[summary.curl_energy_positional_mean for summary in summaries],
                color=color,
                label=label,
            )

        self._style_axes(ax=axes[0], ylabel="Exactness")
        axes[0].set_title("Exactness by Layer")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylim(0.0, 1.05)

        self._style_axes(ax=axes[1], ylabel="Curl Energy")
        axes[1].set_title("Curl Energy by Layer (Position-Corrected)")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylim(bottom=0.0)

        self._style_legend(fig=fig, source_ax=axes[1], ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_probe_accuracy_polarity_invariant(
        self,
        probe_results: list[ProbeRunResult],
        path: Path,
    ) -> None:
        grouped: dict[tuple[str, str], list[ProbeRunResult]] = defaultdict(list)
        for result in probe_results:
            grouped[(result.variant_name, result.probe_name)].append(result)

        fig, ax = plt.subplots(figsize=(9, 5))
        for (variant_name, probe_name), results in grouped.items():
            results = sorted(results, key=lambda item: item.layer_index)
            self._plot_neon_line(
                ax=ax,
                xs=[result.layer_index for result in results],
                ys=[result.test_accuracy_polarity_invariant for result in results],
                color=self._series_color(variant_name=variant_name, probe_name=probe_name),
                label=f"{variant_name}:{probe_name}",
            )
        self._style_axes(ax=ax, ylabel="Polarity-Invariant Accuracy")
        ax.set_title("Probe Polarity-Invariant Accuracy by Layer")
        ax.set_xlabel("Layer")
        ax.set_ylim(0.0, 1.05)
        self._style_legend(fig=fig, source_ax=ax, ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_probe_accuracy_aggregate(self, aggregate_rows: list[dict[str, Any]], path: Path) -> None:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in aggregate_rows:
            grouped[(row["variant_name"], row["probe_name"])].append(row)

        fig, ax = plt.subplots(figsize=(9, 5))
        for (variant_name, probe_name), rows in grouped.items():
            rows = sorted(rows, key=lambda item: item["layer_index"])
            color = self._series_color(variant_name=variant_name, probe_name=probe_name)
            xs = [row["layer_index"] for row in rows]
            ys = [row["test_accuracy_mean"] for row in rows]
            errs = [row["test_accuracy_std"] for row in rows]
            self._plot_neon_line(
                ax=ax,
                xs=xs,
                ys=ys,
                color=color,
                label=f"{variant_name}:{probe_name}",
            )
            if any(err > 0 for err in errs):
                lower = [max(0.0, y - err) for y, err in zip(ys, errs)]
                upper = [min(1.0, y + err) for y, err in zip(ys, errs)]
                self._plot_neon_band(ax=ax, xs=xs, lower=lower, upper=upper, color=color)
        self._style_axes(ax=ax, ylabel="Accuracy")
        ax.set_title("Probe Test Accuracy by Layer (Mean +/- 1 SD)")
        ax.set_xlabel("Layer")
        ax.set_ylim(0.0, 1.05)
        self._style_legend(fig=fig, source_ax=ax, ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_diagnostics_aggregate(self, aggregate_rows: list[dict[str, Any]], path: Path) -> None:
        if not aggregate_rows:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in aggregate_rows:
            grouped[(row["variant_name"], row["probe_name"])].append(row)

        for (variant_name, probe_name), rows in grouped.items():
            rows = sorted(rows, key=lambda item: item["layer_index"])
            color = self._series_color(variant_name=variant_name, probe_name=probe_name)
            xs = [row["layer_index"] for row in rows]
            exact_mean = [row["exactness_mean"] for row in rows]
            exact_std = [row["exactness_std"] for row in rows]
            curl_mean = [row["curl_energy_positional_mean"] for row in rows]
            curl_std = [row["curl_energy_positional_std"] for row in rows]
            label = f"{variant_name}:{probe_name}"
            self._plot_neon_line(ax=axes[0], xs=xs, ys=exact_mean, color=color, label=label)
            self._plot_neon_line(ax=axes[1], xs=xs, ys=curl_mean, color=color, label=label)
            if any(err > 0 for err in exact_std):
                lower = [max(0.0, y - err) for y, err in zip(exact_mean, exact_std)]
                upper = [min(1.0, y + err) for y, err in zip(exact_mean, exact_std)]
                self._plot_neon_band(ax=axes[0], xs=xs, lower=lower, upper=upper, color=color)
            if any(err > 0 for err in curl_std):
                lower = [max(0.0, y - err) for y, err in zip(curl_mean, curl_std)]
                upper = [max(0.0, y + err) for y, err in zip(curl_mean, curl_std)]
                self._plot_neon_band(ax=axes[1], xs=xs, lower=lower, upper=upper, color=color)

        self._style_axes(ax=axes[0], ylabel="Exactness")
        axes[0].set_title("Exactness by Layer (Mean +/- 1 SD)")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylim(0.0, 1.05)

        self._style_axes(ax=axes[1], ylabel="Curl Energy")
        axes[1].set_title("Curl Energy by Layer (Mean +/- 1 SD)")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylim(bottom=0.0)

        self._style_legend(fig=fig, source_ax=axes[1], ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_probe_accuracy_polarity_invariant_aggregate(
        self,
        aggregate_rows: list[dict[str, Any]],
        path: Path,
    ) -> None:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in aggregate_rows:
            grouped[(row["variant_name"], row["probe_name"])].append(row)

        fig, ax = plt.subplots(figsize=(9, 5))
        for (variant_name, probe_name), rows in grouped.items():
            rows = sorted(rows, key=lambda item: item["layer_index"])
            color = self._series_color(variant_name=variant_name, probe_name=probe_name)
            xs = [row["layer_index"] for row in rows]
            ys = [row["test_accuracy_polarity_invariant_mean"] for row in rows]
            errs = [row["test_accuracy_polarity_invariant_std"] for row in rows]
            self._plot_neon_line(
                ax=ax,
                xs=xs,
                ys=ys,
                color=color,
                label=f"{variant_name}:{probe_name}",
            )
            if any(err > 0 for err in errs):
                lower = [max(0.0, y - err) for y, err in zip(ys, errs)]
                upper = [min(1.0, y + err) for y, err in zip(ys, errs)]
                self._plot_neon_band(ax=ax, xs=xs, lower=lower, upper=upper, color=color)
        self._style_axes(ax=ax, ylabel="Polarity-Invariant Accuracy")
        ax.set_title("Probe Polarity-Invariant Accuracy by Layer (Mean +/- 1 SD)")
        ax.set_xlabel("Layer")
        ax.set_ylim(0.0, 1.05)
        self._style_legend(fig=fig, source_ax=ax, ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=180)
        plt.close(fig)

    @staticmethod
    def _std(values: Any) -> float:
        values = list(values)
        if len(values) <= 1:
            return 0.0
        return stdev(values)

    @staticmethod
    def _configure_plot_theme() -> None:
        plt.rcParams.update(
            {
                "figure.facecolor": "#000000",
                "axes.facecolor": "#000000",
                "savefig.facecolor": "#000000",
                "axes.edgecolor": CYBER_COLORS["axis"],
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

    def _series_color(self, variant_name: str, probe_name: str) -> str:
        return SERIES_COLORS.get((variant_name, probe_name), CYBER_COLORS["cyan"])

    @staticmethod
    def _plot_neon_line(
        ax: Any,
        xs: list[int],
        ys: list[float],
        color: str,
        label: str,
    ) -> None:
        ax.plot(xs, ys, color=color, linewidth=4.0, alpha=0.10, solid_capstyle="round", zorder=1)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.45,
            label=label,
            solid_capstyle="round",
            zorder=2,
        )

    @staticmethod
    def _plot_neon_band(
        ax: Any,
        xs: list[int],
        lower: list[float],
        upper: list[float],
        color: str,
    ) -> None:
        ax.fill_between(xs, lower, upper, color=color, alpha=0.10, linewidth=0.0, zorder=0)

    @staticmethod
    def _style_axes(ax: Any, ylabel: str) -> None:
        ax.set_ylabel(ylabel)
        ax.tick_params(width=0.8, length=3)
        for spine in ax.spines.values():
            spine.set_color(CYBER_COLORS["axis"])
            spine.set_linewidth(0.8)

    @staticmethod
    def _style_legend(fig: Any, source_ax: Any, ncol: int) -> None:
        handles, labels = source_ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="upper center", ncol=ncol)
        for text in legend.get_texts():
            text.set_color("#E6E6E6")
