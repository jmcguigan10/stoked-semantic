from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch import Tensor

from stoked_semantic.config import ProbeConfig
from stoked_semantic.utils import parameter_count


class BaseProbe(nn.Module, ABC):
    probe_name: str

    @abstractmethod
    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        raise NotImplementedError

    def oriented_edge_tensor(self, nodes: Tensor) -> Tensor | None:
        return None

    @staticmethod
    def query_pair(nodes: Tensor, query_indices: Tensor) -> tuple[Tensor, Tensor]:
        batch_indices = torch.arange(nodes.shape[0], device=nodes.device)
        source = nodes[batch_indices, query_indices[:, 0]]
        target = nodes[batch_indices, query_indices[:, 1]]
        return source, target

    @staticmethod
    def query_triplet(nodes: Tensor, query_indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_indices = torch.arange(nodes.shape[0], device=nodes.device)
        source_indices = query_indices[:, 0]
        target_indices = query_indices[:, 1]
        other_indices = 3 - source_indices - target_indices
        source = nodes[batch_indices, source_indices]
        target = nodes[batch_indices, target_indices]
        other = nodes[batch_indices, other_indices]
        return source, target, other


class QueryOnlyProbe(BaseProbe):
    probe_name = "query_only"

    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.source_embedding = nn.Embedding(3, hidden)
        self.target_embedding = nn.Embedding(3, hidden)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        source = self.source_embedding(query_indices[:, 0])
        target = self.target_embedding(query_indices[:, 1])
        return self.out(torch.cat([source, target], dim=-1))


class ExactProbe(BaseProbe):
    probe_name = "exact"

    def __init__(self, d_model: int, rank: int, n_classes: int):
        super().__init__()
        self.to_phi = nn.Linear(d_model, rank, bias=False)
        self.out = nn.Linear(rank, n_classes)

    def node_potentials(self, nodes: Tensor) -> Tensor:
        return self.to_phi(nodes)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        potentials = self.node_potentials(nodes)
        source, target = self.query_pair(potentials, query_indices)
        return self.out(target - source)

    def oriented_edge_tensor(self, nodes: Tensor) -> Tensor:
        potentials = self.node_potentials(nodes)
        return potentials[:, None, :, :] - potentials[:, :, None, :]


class PairwiseProbe(BaseProbe):
    probe_name = "pairwise"

    def __init__(self, d_model: int, rank: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        self.out = nn.Linear(rank, n_classes)

    def pair_feat(self, ha: Tensor, hb: Tensor) -> Tensor:
        return torch.cat([ha, hb, ha * hb, hb - ha], dim=-1)

    def edge_embedding(self, ha: Tensor, hb: Tensor) -> Tensor:
        return self.edge(self.pair_feat(ha, hb))

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        source, target = self.query_pair(nodes, query_indices)
        return self.out(self.edge_embedding(source, target))

    def oriented_edge_tensor(self, nodes: Tensor) -> Tensor:
        batch_size = nodes.shape[0]
        hidden_size = self.edge_embedding(nodes[:, 0, :], nodes[:, 1, :]).shape[-1]
        omega = nodes.new_zeros((batch_size, 3, 3, hidden_size))
        for source_index in range(3):
            for target_index in range(3):
                if source_index == target_index:
                    continue
                omega[:, source_index, target_index, :] = self.edge_embedding(
                    nodes[:, source_index, :],
                    nodes[:, target_index, :],
                )
        return omega


class TriadicProbe(BaseProbe):
    probe_name = "triadic"

    def __init__(self, d_model: int, rank: int, n_classes: int):
        super().__init__()
        self.ui = nn.Linear(d_model, rank, bias=False)
        self.uj = nn.Linear(d_model, rank, bias=False)
        self.uk = nn.Linear(d_model, rank, bias=False)
        self.query = nn.Linear(4 * d_model, rank, bias=False)
        self.out = nn.Linear(2 * rank, n_classes)

    def pair_feat(self, ha: Tensor, hb: Tensor) -> Tensor:
        return torch.cat([ha, hb, ha * hb, hb - ha], dim=-1)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        source, target, other = self.query_triplet(nodes, query_indices)
        triadic = self.ui(source) * self.uj(target) * self.uk(other)
        query_context = self.query(self.pair_feat(source, target))
        return self.out(torch.cat([query_context, triadic], dim=-1))


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    rank: int
    hidden: int | None


class ProbeFactory:
    def __init__(self, config: ProbeConfig):
        self.config = config

    def specs(self, d_model: int) -> list[ProbeSpec]:
        triadic_rank = self.config.triadic_rank
        if triadic_rank is None:
            triadic_rank = self._match_triadic_rank(d_model=d_model)
        return [
            ProbeSpec(name="query_only", rank=self.config.exact_rank, hidden=None),
            ProbeSpec(name="exact", rank=self.config.exact_rank, hidden=None),
            ProbeSpec(name="pairwise", rank=self.config.pairwise_rank, hidden=self.config.pairwise_hidden),
            ProbeSpec(name="triadic", rank=triadic_rank, hidden=None),
        ]

    def build(self, spec: ProbeSpec, d_model: int) -> BaseProbe:
        if spec.name == "query_only":
            return QueryOnlyProbe(hidden=spec.rank, n_classes=self.config.num_classes)
        if spec.name == "exact":
            return ExactProbe(d_model=d_model, rank=spec.rank, n_classes=self.config.num_classes)
        if spec.name == "pairwise":
            return PairwiseProbe(
                d_model=d_model,
                rank=spec.rank,
                n_classes=self.config.num_classes,
                hidden=spec.hidden or self.config.pairwise_hidden,
            )
        if spec.name == "triadic":
            return TriadicProbe(d_model=d_model, rank=spec.rank, n_classes=self.config.num_classes)
        raise ValueError(f"Unsupported probe name: {spec.name}")

    def _match_triadic_rank(self, d_model: int) -> int:
        pairwise = PairwiseProbe(
            d_model=d_model,
            rank=self.config.pairwise_rank,
            n_classes=self.config.num_classes,
            hidden=self.config.pairwise_hidden,
        )
        pairwise_params = parameter_count(pairwise)
        best_rank = 8
        best_gap = math.inf
        for rank in range(8, 513):
            triadic = TriadicProbe(d_model=d_model, rank=rank, n_classes=self.config.num_classes)
            gap = abs(parameter_count(triadic) - pairwise_params)
            if gap < best_gap:
                best_gap = gap
                best_rank = rank
        return best_rank
