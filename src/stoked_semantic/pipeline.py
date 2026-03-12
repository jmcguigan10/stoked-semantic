from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from stoked_semantic.config import ExperimentConfig, ReportConfig
from stoked_semantic.data import make_dataset_builder
from stoked_semantic.diagnostics import DiagnosticAnalyzer, DiagnosticSummary
from stoked_semantic.encoding import EncodedSplit, TransformerFeatureExtractor
from stoked_semantic.reporting import ReportWriter
from stoked_semantic.training import GroupEvaluation, ProbeRunResult, ProbeTrainer
from stoked_semantic.utils import set_seed


@dataclass(frozen=True)
class PhaseOneRunArtifacts:
    run_seed: int
    output_dir: Path
    train_examples: int
    test_examples: int
    variants: int
    probe_results: list[ProbeRunResult]
    diagnostic_results: list[DiagnosticSummary]

    def summary(self) -> dict[str, Any]:
        return {
            "run_seed": self.run_seed,
            "output_dir": str(self.output_dir),
            "train_examples": self.train_examples,
            "test_examples": self.test_examples,
            "variants": self.variants,
            "probe_runs": len(self.probe_results),
            "diagnostic_runs": len(self.diagnostic_results),
        }


class PhaseOnePipeline:
    """Runs the phase-1 experiment end to end."""

    def __init__(self, config: ExperimentConfig):
        self.config = config.with_rooted_paths()
        self.dataset_builder = make_dataset_builder(self.config.data)
        self.extractor = TransformerFeatureExtractor(self.config.encoder)
        self.trainer = ProbeTrainer(self.config.probe)
        self.diagnostics = DiagnosticAnalyzer()
        self.reporter = ReportWriter(
            output_dir=self.config.report.output_dir,
            write_plots=self.config.report.write_plots,
        )

    def run(self) -> PhaseOneRunArtifacts:
        set_seed(self.config.data.seed)
        bundle = self.dataset_builder.build()

        probe_results: list[ProbeRunResult] = []
        diagnostic_results: list[DiagnosticSummary] = []

        variants: list[tuple[str, bool]] = []
        if self.config.run_pretrained:
            variants.append(("pretrained", False))
        if self.config.run_random_control:
            variants.append(("random_control", True))

        for variant_name, random_weights in variants:
            train_split, test_split = self.extractor.encode_variant(
                variant_name=variant_name,
                train_examples=bundle.train_examples,
                test_examples=bundle.test_examples,
                random_weights=random_weights,
            )
            variant_probe_results, variant_diagnostics = self._run_variant(
                train_split=train_split,
                test_split=test_split,
            )
            probe_results.extend(variant_probe_results)
            diagnostic_results.extend(variant_diagnostics)

        artifacts = PhaseOneRunArtifacts(
            run_seed=self.config.data.seed,
            output_dir=self.config.report.output_dir,
            train_examples=len(bundle.train_examples),
            test_examples=len(bundle.test_examples),
            variants=len(variants),
            probe_results=probe_results,
            diagnostic_results=diagnostic_results,
        )
        self.reporter.write(
            probe_results=probe_results,
            diagnostics=diagnostic_results,
            metadata=self._metadata(artifacts=artifacts),
        )
        return artifacts

    def _run_variant(
        self,
        train_split: EncodedSplit,
        test_split: EncodedSplit,
    ) -> tuple[list[ProbeRunResult], list[DiagnosticSummary]]:
        probe_results: list[ProbeRunResult] = []
        diagnostics: list[DiagnosticSummary] = []
        for layer_index in tqdm(range(train_split.num_layers), desc=f"layers:{train_split.variant_name}"):
            layer_train = train_split.layer_view(layer_index=layer_index)
            layer_test = test_split.layer_view(layer_index=layer_index)
            layer_results = self.trainer.run_layer(
                train_features=layer_train,
                test_features=layer_test,
            )
            probe_results.extend(layer_results)
            for result in layer_results:
                summary = self.diagnostics.summarize_probe(
                    probe=result.model,
                    features=layer_test,
                    run_seed=self.config.data.seed,
                )
                if summary is not None:
                    diagnostics.append(summary)
            diagnostics.extend(
                self.diagnostics.summarize_raw(
                    features=layer_test,
                    run_seed=self.config.data.seed,
                )
            )
        return probe_results, diagnostics

    def _metadata(self, artifacts: PhaseOneRunArtifacts) -> dict[str, Any]:
        return {
            "mode": "single_seed",
            "run_seed": artifacts.run_seed,
            "task_suite": self.config.data.task_suite,
            "train_examples": artifacts.train_examples,
            "test_examples": artifacts.test_examples,
            "variants": artifacts.variants,
            "relation_ids": list(self.config.data.relation_ids or ()),
            "train_template_ids": list(self.config.data.train_template_ids or ()),
            "test_template_ids": list(self.config.data.test_template_ids or ()),
            "masked_visible_clause_counts": list(self.config.data.masked_visible_clause_counts),
            "exact_rank": self.config.probe.exact_rank,
            "exact_rank_sweep": list(self.config.probe.exact_rank_sweep),
            "pairwise_rank": self.config.probe.pairwise_rank,
            "triadic_rank": self.config.probe.triadic_rank,
            "output_dir": str(self.config.report.output_dir),
        }


class MultiSeedPhaseOnePipeline:
    """Runs the phase-1 pipeline repeatedly and writes aggregated reports."""

    def __init__(self, config: ExperimentConfig, run_seeds: tuple[int, ...]):
        self.config = config.with_rooted_paths()
        self.run_seeds = run_seeds
        self.reporter = ReportWriter(
            output_dir=self.config.report.output_dir,
            write_plots=self.config.report.write_plots,
        )

    def run(self) -> dict[str, Any]:
        artifacts_by_seed: list[PhaseOneRunArtifacts] = []
        for run_seed in tqdm(self.run_seeds, desc="seeds"):
            existing = self._load_completed_seed(run_seed=run_seed)
            if existing is not None:
                artifacts_by_seed.append(existing)
                self._write_aggregate_from_artifacts(artifacts_by_seed)
                continue

            seed_config = self._seed_config(run_seed=run_seed)
            pipeline = PhaseOnePipeline(config=seed_config)
            artifacts = pipeline.run()
            artifacts_by_seed.append(artifacts)
            self._write_aggregate_from_artifacts(artifacts_by_seed)

        return {
            "mode": "multi_seed",
            "run_seeds": list(self.run_seeds),
            "runs": len(artifacts_by_seed),
            "train_examples_per_run": artifacts_by_seed[0].train_examples if artifacts_by_seed else 0,
            "test_examples_per_run": artifacts_by_seed[0].test_examples if artifacts_by_seed else 0,
            "variants_per_run": artifacts_by_seed[0].variants if artifacts_by_seed else 0,
            "total_probe_runs": sum(len(artifacts.probe_results) for artifacts in artifacts_by_seed),
            "total_diagnostic_runs": sum(len(artifacts.diagnostic_results) for artifacts in artifacts_by_seed),
            "output_dir": str(self.config.report.output_dir),
        }

    def _seed_config(self, run_seed: int) -> ExperimentConfig:
        seed_output_dir = self.config.report.output_dir / f"seed_{run_seed}"
        return ExperimentConfig(
            root_dir=self.config.root_dir,
            data=replace(self.config.data, seed=run_seed),
            encoder=self.config.encoder,
            probe=replace(self.config.probe, seed=run_seed),
            report=ReportConfig(
                output_dir=seed_output_dir,
                write_plots=self.config.report.write_plots,
            ),
            run_pretrained=self.config.run_pretrained,
            run_random_control=self.config.run_random_control,
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "mode": "multi_seed",
            "run_seeds": list(self.run_seeds),
            "task_suite": self.config.data.task_suite,
            "relation_ids": list(self.config.data.relation_ids or ()),
            "train_template_ids": list(self.config.data.train_template_ids or ()),
            "test_template_ids": list(self.config.data.test_template_ids or ()),
            "masked_visible_clause_counts": list(self.config.data.masked_visible_clause_counts),
            "exact_rank": self.config.probe.exact_rank,
            "exact_rank_sweep": list(self.config.probe.exact_rank_sweep),
            "pairwise_rank": self.config.probe.pairwise_rank,
            "triadic_rank": self.config.probe.triadic_rank,
            "output_dir": str(self.config.report.output_dir),
        }

    def _write_aggregate_from_artifacts(
        self,
        artifacts_by_seed: list[PhaseOneRunArtifacts],
    ) -> None:
        if not artifacts_by_seed:
            return
        probe_results = [
            result
            for artifacts in artifacts_by_seed
            for result in artifacts.probe_results
        ]
        diagnostics = [
            summary
            for artifacts in artifacts_by_seed
            for summary in artifacts.diagnostic_results
        ]
        self.reporter.write_aggregate(
            probe_results=probe_results,
            diagnostics=diagnostics,
            run_summaries=[artifacts.summary() for artifacts in artifacts_by_seed],
            metadata=self._metadata(),
        )

    def _load_completed_seed(self, run_seed: int) -> PhaseOneRunArtifacts | None:
        output_dir = self.config.report.output_dir / f"seed_{run_seed}"
        required = (
            output_dir / "summary.json",
            output_dir / "probe_accuracy_by_layer.csv",
            output_dir / "diagnostics_by_layer.csv",
            output_dir / "template_group_accuracy_by_layer.csv",
        )
        if not all(path.exists() for path in required):
            return None

        summary_payload = json.loads((output_dir / "summary.json").read_text())
        metadata = summary_payload.get("metadata", {})

        group_rows = list(csv.DictReader((output_dir / "template_group_accuracy_by_layer.csv").open()))
        group_lookup: dict[tuple[int, str, int, str], list[GroupEvaluation]] = {}
        for row in group_rows:
            key = (
                int(row["run_seed"]),
                row["variant_name"],
                int(row["layer_index"]),
                row["probe_name"],
            )
            group_lookup.setdefault(key, []).append(
                GroupEvaluation(
                    group_type=row["group_type"],
                    group_name=row["group_name"],
                    example_count=int(float(row["example_count"])),
                    accuracy=float(row["test_accuracy"]),
                    polarity_invariant_accuracy=float(row["test_accuracy_polarity_invariant"]),
                )
            )

        probe_results: list[ProbeRunResult] = []
        for row in csv.DictReader((output_dir / "probe_accuracy_by_layer.csv").open()):
            key = (
                int(row["run_seed"]),
                row["variant_name"],
                int(row["layer_index"]),
                row["probe_name"],
            )
            probe_results.append(
                ProbeRunResult(
                    run_seed=int(row["run_seed"]),
                    layer_index=int(row["layer_index"]),
                    probe_name=row["probe_name"],
                    probe_family=row.get("probe_family", _probe_family_from_name(row["probe_name"])),
                    variant_name=row["variant_name"],
                    model=None,  # loaded runs only need metrics for aggregation
                    train_accuracy=float(row["train_accuracy"]),
                    test_accuracy=float(row["test_accuracy"]),
                    test_accuracy_polarity_invariant=float(row["test_accuracy_polarity_invariant"]),
                    test_group_evaluations=tuple(group_lookup.get(key, ())),
                    num_parameters=int(float(row["num_parameters"])),
                    rank=int(float(row["rank"])),
                )
            )

        diagnostic_results = [
            DiagnosticSummary(
                run_seed=int(row["run_seed"]),
                layer_index=int(row["layer_index"]),
                probe_name=row["probe_name"],
                diagnostic_family=row.get("diagnostic_family", "probe"),
                variant_name=row["variant_name"],
                exactness_mean=float(row["exactness_mean"]),
                curl_energy_mean=float(row["curl_energy_mean"]),
                exactness_positional_mean=float(row["exactness_positional_mean"]),
                curl_energy_positional_mean=float(row["curl_energy_positional_mean"]),
            )
            for row in csv.DictReader((output_dir / "diagnostics_by_layer.csv").open())
        ]

        return PhaseOneRunArtifacts(
            run_seed=run_seed,
            output_dir=output_dir,
            train_examples=int(metadata.get("train_examples", 0)),
            test_examples=int(metadata.get("test_examples", 0)),
            variants=int(metadata.get("variants", 0)),
            probe_results=probe_results,
            diagnostic_results=diagnostic_results,
        )


def _probe_family_from_name(probe_name: str) -> str:
    if probe_name.startswith("exact"):
        return "exact"
    return probe_name
