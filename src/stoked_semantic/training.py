from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from stoked_semantic.config import ProbeConfig
from stoked_semantic.encoding import LayerFeatures
from stoked_semantic.probes import BaseProbe, ProbeFactory, ProbeSpec
from stoked_semantic.utils import parameter_count, resolve_torch_device, set_seed


@dataclass
class GroupEvaluation:
    group_type: str
    group_name: str
    example_count: int
    accuracy: float
    polarity_invariant_accuracy: float


@dataclass
class EvaluationResult:
    accuracy: float
    polarity_invariant_accuracy: float
    group_evaluations: tuple[GroupEvaluation, ...] = ()


@dataclass
class ProbeRunResult:
    run_seed: int
    layer_index: int
    probe_name: str
    probe_family: str
    variant_name: str
    model: BaseProbe
    train_accuracy: float
    test_accuracy: float
    test_accuracy_polarity_invariant: float
    test_group_evaluations: tuple[GroupEvaluation, ...]
    num_parameters: int
    rank: int


class ProbeTrainer:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.device = resolve_torch_device(config.train_device)
        self.factory = ProbeFactory(config)

    def run_layer(
        self,
        train_features: LayerFeatures,
        test_features: LayerFeatures,
    ) -> list[ProbeRunResult]:
        d_model = train_features.hidden_states.shape[-1]
        node_count = train_features.hidden_states.shape[1]
        query_arity = train_features.query_indices.shape[1]
        results: list[ProbeRunResult] = []
        for spec in self.factory.specs(d_model=d_model, query_arity=query_arity):
            set_seed(self.config.seed + train_features.layer_index)
            model = self.factory.build(
                spec=spec,
                d_model=d_model,
                node_count=node_count,
                query_arity=query_arity,
            ).to(self.device)
            self._fit(model=model, features=train_features)
            train_evaluation = self._evaluate(model=model, features=train_features)
            test_evaluation = self._evaluate(
                model=model,
                features=test_features,
                include_group_breakdowns=True,
            )
            model = model.to("cpu")
            results.append(
                ProbeRunResult(
                    run_seed=self.config.seed,
                    layer_index=train_features.layer_index,
                    probe_name=spec.name,
                    probe_family=spec.family,
                    variant_name=train_features.variant_name,
                    model=model,
                    train_accuracy=train_evaluation.accuracy,
                    test_accuracy=test_evaluation.accuracy,
                    test_accuracy_polarity_invariant=test_evaluation.polarity_invariant_accuracy,
                    test_group_evaluations=test_evaluation.group_evaluations,
                    num_parameters=parameter_count(model),
                    rank=spec.rank,
                )
            )
        return results

    def _fit(self, model: BaseProbe, features: LayerFeatures) -> None:
        dataset = TensorDataset(features.nodes, features.query_indices, features.labels)
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        model.train()
        for _ in range(self.config.epochs):
            for nodes, query_indices, labels in loader:
                nodes = nodes.to(self.device)
                query_indices = query_indices.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(nodes, query_indices)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
        model.eval()

    def _evaluate(
        self,
        model: BaseProbe,
        features: LayerFeatures,
        include_group_breakdowns: bool = False,
    ) -> EvaluationResult:
        dataset = TensorDataset(features.nodes, features.query_indices, features.labels)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total = 0
        correct = 0
        all_predictions: list[Tensor] = []
        all_labels: list[Tensor] = []
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for nodes, query_indices, labels in loader:
                nodes = nodes.to(self.device)
                query_indices = query_indices.to(self.device)
                labels = labels.to(self.device)
                logits = model(nodes, query_indices)
                predictions = logits.argmax(dim=-1)
                total += int(labels.shape[0])
                correct += int((predictions == labels).sum().item())
                if include_group_breakdowns:
                    all_predictions.append(predictions.cpu())
                    all_labels.append(labels.cpu())
        accuracy = correct / total if total else 0.0
        group_evaluations: tuple[GroupEvaluation, ...] = ()
        if include_group_breakdowns:
            predictions = torch.cat(all_predictions, dim=0)
            labels = torch.cat(all_labels, dim=0)
            group_evaluations = self._group_evaluations(
                predictions=predictions,
                labels=labels,
                features=features,
            )
        return EvaluationResult(
            accuracy=accuracy,
            polarity_invariant_accuracy=max(accuracy, 1.0 - accuracy),
            group_evaluations=group_evaluations,
        )

    def _group_evaluations(
        self,
        predictions: Tensor,
        labels: Tensor,
        features: LayerFeatures,
    ) -> tuple[GroupEvaluation, ...]:
        grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for index, (prediction, label) in enumerate(zip(predictions.tolist(), labels.tolist(), strict=True)):
            grouped.setdefault(
                ("relation_id", features.relation_ids[index]),
                [],
            ).append((prediction, label))
            grouped.setdefault(
                ("template_id", features.template_ids[index]),
                [],
            ).append((prediction, label))
            grouped.setdefault(
                ("template_family_id", features.template_family_ids[index]),
                [],
            ).append((prediction, label))
            grouped.setdefault(
                ("query_distance_type", features.query_distance_types[index]),
                [],
            ).append((prediction, label))
            grouped.setdefault(
                ("query_direction_type", features.query_direction_types[index]),
                [],
            ).append((prediction, label))
            grouped.setdefault(
                ("query_type", features.query_types[index]),
                [],
            ).append((prediction, label))

        evaluations: list[GroupEvaluation] = []
        for (group_type, group_name), pairs in sorted(grouped.items()):
            example_count = len(pairs)
            correct = sum(int(prediction == label) for prediction, label in pairs)
            accuracy = correct / example_count if example_count else 0.0
            evaluations.append(
                GroupEvaluation(
                    group_type=group_type,
                    group_name=group_name,
                    example_count=example_count,
                    accuracy=accuracy,
                    polarity_invariant_accuracy=max(accuracy, 1.0 - accuracy),
                )
            )
        return tuple(evaluations)
