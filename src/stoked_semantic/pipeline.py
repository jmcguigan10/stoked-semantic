from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from stoked_semantic.config import ExperimentConfig, ReportConfig
from stoked_semantic.data import SyntheticConsistencyDatasetBuilder
from stoked_semantic.diagnostics import DiagnosticAnalyzer, DiagnosticSummary
from stoked_semantic.encoding import EncodedSplit, TransformerFeatureExtractor
from stoked_semantic.reporting import ReportWriter
from stoked_semantic.training import ProbeRunResult, ProbeTrainer
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
        self.dataset_builder = SyntheticConsistencyDatasetBuilder(self.config.data)
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
                summary = self.diagnostics.summarize(
                    probe=result.model,
                    features=layer_test,
                    run_seed=self.config.data.seed,
                )
                if summary is not None:
                    diagnostics.append(summary)
        return probe_results, diagnostics

    def _metadata(self, artifacts: PhaseOneRunArtifacts) -> dict[str, Any]:
        return {
            "mode": "single_seed",
            "run_seed": artifacts.run_seed,
            "train_examples": artifacts.train_examples,
            "test_examples": artifacts.test_examples,
            "variants": artifacts.variants,
            "relation_ids": list(self.config.data.relation_ids),
            "train_template_ids": list(self.config.data.train_template_ids),
            "test_template_ids": list(self.config.data.test_template_ids),
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
            seed_config = self._seed_config(run_seed=run_seed)
            pipeline = PhaseOnePipeline(config=seed_config)
            artifacts_by_seed.append(pipeline.run())

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
        return {
            "mode": "multi_seed",
            "run_seeds": list(self.run_seeds),
            "runs": len(artifacts_by_seed),
            "train_examples_per_run": artifacts_by_seed[0].train_examples if artifacts_by_seed else 0,
            "test_examples_per_run": artifacts_by_seed[0].test_examples if artifacts_by_seed else 0,
            "variants_per_run": artifacts_by_seed[0].variants if artifacts_by_seed else 0,
            "total_probe_runs": len(probe_results),
            "total_diagnostic_runs": len(diagnostics),
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
            "relation_ids": list(self.config.data.relation_ids),
            "train_template_ids": list(self.config.data.train_template_ids),
            "test_template_ids": list(self.config.data.test_template_ids),
            "output_dir": str(self.config.report.output_dir),
        }
