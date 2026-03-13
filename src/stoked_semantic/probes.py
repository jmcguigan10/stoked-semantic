from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch import Tensor

from stoked_semantic.config import ProbeConfig
from stoked_semantic.utils import parameter_count


class BaseProbe(nn.Module, ABC):
    probe_name: str
    probe_family: str

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
    def query_nodes(nodes: Tensor, query_indices: Tensor) -> Tensor:
        batch_indices = torch.arange(nodes.shape[0], device=nodes.device).unsqueeze(1)
        return nodes[batch_indices, query_indices]

    @staticmethod
    def flattened_query_nodes(nodes: Tensor, query_indices: Tensor) -> Tensor:
        return BaseProbe.query_nodes(nodes, query_indices).reshape(nodes.shape[0], -1)

    @staticmethod
    def query_context(nodes: Tensor, query_indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_indices = torch.arange(nodes.shape[0], device=nodes.device)
        source_indices = query_indices[:, 0]
        target_indices = query_indices[:, 1]
        source = nodes[batch_indices, source_indices]
        target = nodes[batch_indices, target_indices]
        slot_indices = torch.arange(nodes.shape[1], device=nodes.device).unsqueeze(0).expand_as(nodes[..., 0])
        other_mask = (slot_indices != source_indices.unsqueeze(1)) & (
            slot_indices != target_indices.unsqueeze(1)
        )
        other_nodes = nodes[other_mask].view(nodes.shape[0], -1, nodes.shape[-1])
        context = other_nodes.mean(dim=1)
        return source, target, context

    @staticmethod
    def triadic_operands(nodes: Tensor, query_indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        query_arity = query_indices.shape[1]
        if query_arity == 2:
            return BaseProbe.query_context(nodes, query_indices)
        if query_arity == 3:
            query_nodes = BaseProbe.query_nodes(nodes, query_indices)
            return query_nodes[:, 0, :], query_nodes[:, 1, :], query_nodes[:, 2, :]
        raise ValueError(f"Triadic probes only support query arity 2 or 3, got {query_arity}")

    @staticmethod
    def pairwise_differences(query_nodes: Tensor) -> Tensor:
        pair_features = [
            query_nodes[:, right, :] - query_nodes[:, left, :]
            for left, right in combinations(range(query_nodes.shape[1]), 2)
        ]
        return torch.cat(pair_features, dim=-1)


class QueryOnlyProbe(BaseProbe):
    probe_name = "query_only"

    def __init__(self, hidden: int, n_classes: int, node_count: int, query_arity: int):
        super().__init__()
        self.query_embeddings = nn.ModuleList(
            nn.Embedding(node_count, hidden) for _ in range(query_arity)
        )
        self.out = nn.Linear(query_arity * hidden, n_classes)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        pieces = [
            embedding(query_indices[:, position])
            for position, embedding in enumerate(self.query_embeddings)
        ]
        return self.out(torch.cat(pieces, dim=-1))


class ExactProbe(BaseProbe):
    probe_name = "exact"

    def __init__(self, d_model: int, rank: int, n_classes: int, query_arity: int):
        super().__init__()
        self.query_arity = query_arity
        self.query_pairs = tuple(combinations(range(query_arity), 2))
        self.to_phi = nn.Linear(d_model, rank, bias=False)
        self.out = nn.Linear(len(self.query_pairs) * rank, n_classes)

    def node_potentials(self, nodes: Tensor) -> Tensor:
        return self.to_phi(nodes)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        potentials = self.node_potentials(nodes)
        query_potentials = self.query_nodes(potentials, query_indices)
        features = self.pairwise_differences(query_potentials)
        return self.out(features)

    def oriented_edge_tensor(self, nodes: Tensor) -> Tensor:
        potentials = self.node_potentials(nodes)
        return potentials[:, None, :, :] - potentials[:, :, None, :]


class PairwiseProbe(BaseProbe):
    probe_name = "pairwise"

    def __init__(self, d_model: int, rank: int, n_classes: int, query_arity: int, hidden: int = 128):
        super().__init__()
        self.query_arity = query_arity
        self.query_pairs = tuple(combinations(range(query_arity), 2))
        self.edge = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        self.out = nn.Linear(len(self.query_pairs) * rank, n_classes)

    def pair_feat(self, ha: Tensor, hb: Tensor) -> Tensor:
        return torch.cat([ha, hb, ha * hb, hb - ha], dim=-1)

    def edge_embedding(self, ha: Tensor, hb: Tensor) -> Tensor:
        return self.edge(self.pair_feat(ha, hb))

    def query_pairwise_features(self, query_nodes: Tensor) -> list[Tensor]:
        return [
            self.edge_embedding(query_nodes[:, left, :], query_nodes[:, right, :])
            for left, right in self.query_pairs
        ]

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        query_nodes = self.query_nodes(nodes, query_indices)
        return self.out(torch.cat(self.query_pairwise_features(query_nodes), dim=-1))

    def oriented_edge_tensor(self, nodes: Tensor) -> Tensor:
        batch_size = nodes.shape[0]
        node_count = nodes.shape[1]
        hidden_size = self.edge_embedding(nodes[:, 0, :], nodes[:, 1, :]).shape[-1]
        omega = nodes.new_zeros((batch_size, node_count, node_count, hidden_size))
        for source_index in range(node_count):
            for target_index in range(node_count):
                if source_index == target_index:
                    continue
                omega[:, source_index, target_index, :] = self.edge_embedding(
                    nodes[:, source_index, :],
                    nodes[:, target_index, :],
                )
        return omega


class TriadicProbe(BaseProbe):
    probe_name = "triadic"

    def __init__(self, d_model: int, rank: int, n_classes: int, query_arity: int):
        super().__init__()
        self.query_arity = query_arity
        self.ui = nn.Linear(d_model, rank, bias=False)
        self.uj = nn.Linear(d_model, rank, bias=False)
        self.uk = nn.Linear(d_model, rank, bias=False)
        self.query = nn.Linear(query_arity * d_model, rank, bias=False)
        self.out = nn.Linear(2 * rank, n_classes)

    def pair_feat(self, ha: Tensor, hb: Tensor) -> Tensor:
        return torch.cat([ha, hb, ha * hb, hb - ha], dim=-1)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        source, target, context = self.triadic_operands(nodes, query_indices)
        triadic = self.ui(source) * self.uj(target) * self.uk(context)
        query_context = self.query(self.flattened_query_nodes(nodes, query_indices))
        return self.out(torch.cat([query_context, triadic], dim=-1))


class PairwisePlusQueryContextProbe(PairwiseProbe):
    probe_name = "pairwise_plus_query_context"

    def __init__(
        self,
        d_model: int,
        pairwise_rank: int,
        context_rank: int,
        n_classes: int,
        query_arity: int,
        hidden: int = 128,
    ):
        super().__init__(
            d_model=d_model,
            rank=pairwise_rank,
            n_classes=n_classes,
            query_arity=query_arity,
            hidden=hidden,
        )
        self.query = nn.Linear(query_arity * d_model, context_rank, bias=False)
        self.out = nn.Linear(len(self.query_pairs) * pairwise_rank + context_rank, n_classes)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        query_nodes = self.query_nodes(nodes, query_indices)
        pairwise_features = self.query_pairwise_features(query_nodes)
        query_context = self.query(query_nodes.reshape(nodes.shape[0], -1))
        features = torch.cat(pairwise_features + [query_context], dim=-1)
        return self.out(features)

class PairwisePlusTripletMLPProbe(PairwisePlusQueryContextProbe):
    probe_name = "pairwise_plus_triplet_mlp"

    def __init__(
        self,
        d_model: int,
        pairwise_rank: int,
        context_rank: int,
        residual_hidden: int,
        n_classes: int,
        query_arity: int,
        hidden: int = 128,
    ):
        super().__init__(
            d_model=d_model,
            pairwise_rank=pairwise_rank,
            context_rank=context_rank,
            n_classes=n_classes,
            query_arity=query_arity,
            hidden=hidden,
        )
        self.triplet_residual = nn.Sequential(
            nn.Linear(query_arity * d_model, residual_hidden, bias=False),
            nn.GELU(),
            nn.Linear(residual_hidden, context_rank, bias=False),
        )
        self.out = nn.Linear(len(self.query_pairs) * pairwise_rank + 2 * context_rank, n_classes)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        query_nodes = self.query_nodes(nodes, query_indices)
        flattened = query_nodes.reshape(nodes.shape[0], -1)
        pairwise_features = self.query_pairwise_features(query_nodes)
        query_context = self.query(flattened)
        triplet_residual = self.triplet_residual(flattened)
        features = torch.cat(pairwise_features + [query_context, triplet_residual], dim=-1)
        return self.out(features)


class PairwisePlusTriadicProbe(PairwisePlusQueryContextProbe):
    probe_name = "pairwise_plus_triadic"

    def __init__(
        self,
        d_model: int,
        pairwise_rank: int,
        triadic_rank: int,
        n_classes: int,
        query_arity: int,
        hidden: int = 128,
    ):
        super().__init__(
            d_model=d_model,
            pairwise_rank=pairwise_rank,
            context_rank=triadic_rank,
            n_classes=n_classes,
            query_arity=query_arity,
            hidden=hidden,
        )
        self.ui = nn.Linear(d_model, triadic_rank, bias=False)
        self.uj = nn.Linear(d_model, triadic_rank, bias=False)
        self.uk = nn.Linear(d_model, triadic_rank, bias=False)
        self.out = nn.Linear(len(self.query_pairs) * pairwise_rank + 2 * triadic_rank, n_classes)

    def forward(self, nodes: Tensor, query_indices: Tensor) -> Tensor:
        query_nodes = self.query_nodes(nodes, query_indices)
        pairwise_features = self.query_pairwise_features(query_nodes)
        source, target, context = self.triadic_operands(nodes, query_indices)
        triadic = self.ui(source) * self.uj(target) * self.uk(context)
        query_context = self.query(query_nodes.reshape(nodes.shape[0], -1))
        features = torch.cat(pairwise_features + [query_context, triadic], dim=-1)
        return self.out(features)

@dataclass(frozen=True)
class ProbeSpec:
    name: str
    family: str
    rank: int
    hidden: int | None
    aux_rank: int | None = None
    aux_hidden: int | None = None


class ProbeFactory:
    def __init__(self, config: ProbeConfig):
        self.config = config

    def specs(self, d_model: int, query_arity: int) -> list[ProbeSpec]:
        exact_ranks = tuple(dict.fromkeys(self.config.exact_rank_sweep or (self.config.exact_rank,)))
        triadic_rank = self.config.triadic_rank
        if triadic_rank is None:
            triadic_rank = self._match_triadic_rank(d_model=d_model, query_arity=query_arity)
        triplet_mlp_hidden = self._match_triplet_mlp_hidden(
            d_model=d_model,
            query_arity=query_arity,
            output_rank=triadic_rank,
        )
        specs = [
            ProbeSpec(name="query_only", family="query_only", rank=self.config.exact_rank, hidden=None),
        ]
        for rank in exact_ranks:
            exact_name = "exact" if len(exact_ranks) == 1 else f"exact_r{rank}"
            specs.append(
                ProbeSpec(
                    name=exact_name,
                    family="exact",
                    rank=rank,
                    hidden=None,
                )
            )
        specs.extend(
            [
                ProbeSpec(
                    name="pairwise",
                    family="pairwise",
                    rank=self.config.pairwise_rank,
                    hidden=self.config.pairwise_hidden,
                ),
                ProbeSpec(
                    name="pairwise_plus_query_context",
                    family="pairwise_plus_query_context",
                    rank=self.config.pairwise_rank,
                    hidden=self.config.pairwise_hidden,
                    aux_rank=triadic_rank,
                ),
                ProbeSpec(
                    name="pairwise_plus_triplet_mlp",
                    family="pairwise_plus_triplet_mlp",
                    rank=self.config.pairwise_rank,
                    hidden=self.config.pairwise_hidden,
                    aux_rank=triadic_rank,
                    aux_hidden=triplet_mlp_hidden,
                ),
                ProbeSpec(
                    name="pairwise_plus_triadic",
                    family="pairwise_plus_triadic",
                    rank=self.config.pairwise_rank,
                    hidden=self.config.pairwise_hidden,
                    aux_rank=triadic_rank,
                ),
                ProbeSpec(name="triadic", family="triadic", rank=triadic_rank, hidden=None),
            ]
        )
        return specs

    def build(self, spec: ProbeSpec, d_model: int, node_count: int, query_arity: int) -> BaseProbe:
        if spec.family == "query_only":
            probe = QueryOnlyProbe(
                hidden=spec.rank,
                n_classes=self.config.num_classes,
                node_count=node_count,
                query_arity=query_arity,
            )
        elif spec.family == "exact":
            probe = ExactProbe(
                d_model=d_model,
                rank=spec.rank,
                n_classes=self.config.num_classes,
                query_arity=query_arity,
            )
        elif spec.family == "pairwise":
            probe = PairwiseProbe(
                d_model=d_model,
                rank=spec.rank,
                n_classes=self.config.num_classes,
                query_arity=query_arity,
                hidden=spec.hidden or self.config.pairwise_hidden,
            )
        elif spec.family == "pairwise_plus_query_context":
            probe = PairwisePlusQueryContextProbe(
                d_model=d_model,
                pairwise_rank=spec.rank,
                context_rank=spec.aux_rank or spec.rank,
                n_classes=self.config.num_classes,
                query_arity=query_arity,
                hidden=spec.hidden or self.config.pairwise_hidden,
            )
        elif spec.family == "pairwise_plus_triplet_mlp":
            probe = PairwisePlusTripletMLPProbe(
                d_model=d_model,
                pairwise_rank=spec.rank,
                context_rank=spec.aux_rank or spec.rank,
                residual_hidden=spec.aux_hidden or (spec.aux_rank or spec.rank),
                n_classes=self.config.num_classes,
                query_arity=query_arity,
                hidden=spec.hidden or self.config.pairwise_hidden,
            )
        elif spec.family == "pairwise_plus_triadic":
            probe = PairwisePlusTriadicProbe(
                d_model=d_model,
                pairwise_rank=spec.rank,
                triadic_rank=spec.aux_rank or spec.rank,
                n_classes=self.config.num_classes,
                query_arity=query_arity,
                hidden=spec.hidden or self.config.pairwise_hidden,
            )
        elif spec.family == "triadic":
            probe = TriadicProbe(
                d_model=d_model,
                rank=spec.rank,
                n_classes=self.config.num_classes,
                query_arity=query_arity,
            )
        else:
            raise ValueError(f"Unsupported probe family: {spec.family}")
        probe.probe_name = spec.name
        probe.probe_family = spec.family
        return probe

    def _match_triadic_rank(self, d_model: int, query_arity: int) -> int:
        pairwise = PairwiseProbe(
            d_model=d_model,
            rank=self.config.pairwise_rank,
            n_classes=self.config.num_classes,
            query_arity=query_arity,
            hidden=self.config.pairwise_hidden,
        )
        pairwise_params = parameter_count(pairwise)
        best_rank = 8
        best_gap = math.inf
        for rank in range(8, 513):
            triadic = TriadicProbe(
                d_model=d_model,
                rank=rank,
                n_classes=self.config.num_classes,
                query_arity=query_arity,
            )
            gap = abs(parameter_count(triadic) - pairwise_params)
            if gap < best_gap:
                best_gap = gap
                best_rank = rank
        return best_rank

    def _match_triplet_mlp_hidden(self, d_model: int, query_arity: int, output_rank: int) -> int:
        target_params = 3 * d_model * output_rank
        input_dim = query_arity * d_model
        best_hidden = max(8, output_rank)
        best_gap = math.inf
        for hidden in range(8, 1025):
            mlp_params = input_dim * hidden + hidden * output_rank
            gap = abs(mlp_params - target_params)
            if gap < best_gap:
                best_gap = gap
                best_hidden = hidden
        return best_hidden
