from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FEATURE_CACHE_DIR = Path(".cache/features")
DEFAULT_RELATION_IDS = ("outranks",)
DEFAULT_TEMPLATE_IDS = (
    "active_forward",
    "active_reverse",
    "passive_forward",
    "passive_reverse",
    "mixed_ab_inverse",
    "mixed_ab_inverse_reverse",
    "mixed_bc_inverse",
    "mixed_bc_inverse_reverse",
    "above_forward",
    "above_reverse",
    "below_forward",
    "below_reverse",
    "paraphrase_ab_inverse",
    "paraphrase_ab_inverse_reverse",
    "paraphrase_bc_inverse",
    "paraphrase_bc_inverse_reverse",
)
FACTORIZED_TRAIN_TEMPLATE_IDS = (
    "active_forward",
    "passive_reverse",
    "mixed_ab_inverse",
    "mixed_bc_inverse_reverse",
    "above_forward",
    "below_reverse",
    "paraphrase_ab_inverse",
    "paraphrase_bc_inverse_reverse",
)
FACTORIZED_TEST_TEMPLATE_IDS = (
    "active_reverse",
    "passive_forward",
    "mixed_ab_inverse_reverse",
    "mixed_bc_inverse",
    "above_reverse",
    "below_forward",
    "paraphrase_ab_inverse_reverse",
    "paraphrase_bc_inverse",
)
BALANCED_TRAIN_TEMPLATE_IDS = (
    "active_forward",
    "active_reverse",
    "mixed_ab_inverse",
    "mixed_ab_inverse_reverse",
    "above_forward",
    "above_reverse",
)
BALANCED_TEST_TEMPLATE_IDS = (
    "passive_forward",
    "passive_reverse",
    "mixed_bc_inverse",
    "mixed_bc_inverse_reverse",
    "below_forward",
    "below_reverse",
)
STRICT_TRAIN_TEMPLATE_IDS = (
    "active_forward",
    "passive_forward",
    "mixed_ab_inverse",
    "mixed_bc_inverse",
    "above_forward",
    "below_forward",
)
STRICT_TEST_TEMPLATE_IDS = (
    "active_reverse",
    "passive_reverse",
    "mixed_ab_inverse_reverse",
    "mixed_bc_inverse_reverse",
    "above_reverse",
    "below_reverse",
)

DEFAULT_TRAIN_NAMES = (
    "alice",
    "bob",
    "carol",
    "dave",
    "erin",
    "frank",
    "grace",
    "heidi",
)

DEFAULT_TEST_NAMES = (
    "ivan",
    "judy",
    "mallory",
    "nia",
)


@dataclass(frozen=True)
class DataConfig:
    relation_ids: tuple[str, ...] = DEFAULT_RELATION_IDS
    train_names: tuple[str, ...] = DEFAULT_TRAIN_NAMES
    test_names: tuple[str, ...] = DEFAULT_TEST_NAMES
    train_template_ids: tuple[str, ...] = DEFAULT_TEMPLATE_IDS
    test_template_ids: tuple[str, ...] = DEFAULT_TEMPLATE_IDS
    train_examples_per_label: int = 600
    test_examples_per_label: int = 72
    seed: int = 7


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "bert-base-uncased"
    batch_size: int = 32
    max_length: int = 64
    include_attentions: bool = False
    prefer_device: str = "auto"
    cache_dir: Path = DEFAULT_FEATURE_CACHE_DIR


@dataclass(frozen=True)
class ProbeConfig:
    exact_rank: int = 64
    pairwise_rank: int = 64
    pairwise_hidden: int = 128
    triadic_rank: int | None = None
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_classes: int = 2
    train_device: str = "cpu"
    seed: int = 13


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path = Path("results/phase1")
    write_plots: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    run_pretrained: bool = True
    run_random_control: bool = True

    def with_rooted_paths(self) -> "ExperimentConfig":
        encoder = EncoderConfig(
            model_name=self.encoder.model_name,
            batch_size=self.encoder.batch_size,
            max_length=self.encoder.max_length,
            include_attentions=self.encoder.include_attentions,
            prefer_device=self.encoder.prefer_device,
            cache_dir=self.root_dir / self.encoder.cache_dir,
        )
        report = ReportConfig(
            output_dir=self.root_dir / self.report.output_dir,
            write_plots=self.report.write_plots,
        )
        return ExperimentConfig(
            root_dir=self.root_dir,
            data=self.data,
            encoder=encoder,
            probe=self.probe,
            report=report,
            run_pretrained=self.run_pretrained,
            run_random_control=self.run_random_control,
        )
