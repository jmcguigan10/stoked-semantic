from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
from torch import Tensor

from stoked_semantic.encoding import LayerFeatures
from stoked_semantic.probes import BaseProbe


@dataclass(frozen=True)
class DiagnosticSummary:
    run_seed: int
    layer_index: int
    probe_name: str
    diagnostic_family: str
    variant_name: str
    exactness_mean: float
    curl_energy_mean: float
    exactness_positional_mean: float
    curl_energy_positional_mean: float


class DiagnosticAnalyzer:
    def __init__(self) -> None:
        self._incidence_cache: dict[int, tuple[Tensor, Tensor, list[tuple[int, int]]]] = {}
        self.raw_projection_rank = 4
        self.raw_skew_bilinear_rank = 8

    def summarize_probe(
        self,
        probe: BaseProbe,
        features: LayerFeatures,
        run_seed: int,
    ) -> DiagnosticSummary | None:
        if probe.oriented_edge_tensor(features.nodes[:1]) is None:
            return None

        with torch.no_grad():
            omega = probe.oriented_edge_tensor(features.nodes)
        return self._summarize_tensor(
            run_seed=run_seed,
            layer_index=features.layer_index,
            probe_name=probe.probe_name,
            diagnostic_family="probe",
            variant_name=features.variant_name,
            omega=omega,
            positions=features.positions,
        )

    def summarize_raw(
        self,
        features: LayerFeatures,
        run_seed: int,
    ) -> list[DiagnosticSummary]:
        return [
            self._summarize_tensor(
                run_seed=run_seed,
                layer_index=features.layer_index,
                probe_name=f"raw_projection_r{self.raw_projection_rank}",
                diagnostic_family="raw_projection",
                variant_name=features.variant_name,
                omega=self._raw_projection_edges(features.nodes),
                positions=features.positions,
            ),
            self._summarize_tensor(
                run_seed=run_seed,
                layer_index=features.layer_index,
                probe_name=f"raw_skew_bilinear_r{self.raw_skew_bilinear_rank}",
                diagnostic_family="raw_skew_bilinear",
                variant_name=features.variant_name,
                omega=self._raw_skew_bilinear_edges(features.nodes),
                positions=features.positions,
            ),
        ]

    def _summarize_tensor(
        self,
        run_seed: int,
        layer_index: int,
        probe_name: str,
        diagnostic_family: str,
        variant_name: str,
        omega: Tensor,
        positions: Tensor,
    ) -> DiagnosticSummary:
        omega = self._antisymmetrize(omega)

        exactness_values: list[float] = []
        curl_values: list[float] = []
        for sample_index in range(omega.shape[0]):
            stats = self._exact_projection_and_curl(omega[sample_index])
            exactness_values.append(stats["exactness"])
            curl_values.append(stats["curl_energy"])

        centered = self._subtract_positional_means(omega=omega, positions=positions)
        exactness_centered: list[float] = []
        curl_centered: list[float] = []
        for sample_index in range(centered.shape[0]):
            stats = self._exact_projection_and_curl(centered[sample_index])
            exactness_centered.append(stats["exactness"])
            curl_centered.append(stats["curl_energy"])

        return DiagnosticSummary(
            run_seed=run_seed,
            layer_index=layer_index,
            probe_name=probe_name,
            diagnostic_family=diagnostic_family,
            variant_name=variant_name,
            exactness_mean=sum(exactness_values) / len(exactness_values),
            curl_energy_mean=sum(curl_values) / len(curl_values),
            exactness_positional_mean=sum(exactness_centered) / len(exactness_centered),
            curl_energy_positional_mean=sum(curl_centered) / len(curl_centered),
        )

    def _exact_projection_and_curl(self, omega: Tensor) -> dict[str, float]:
        if omega.ndim == 2:
            omega = omega.unsqueeze(-1)
        omega = self._antisymmetrize(omega)
        n = int(omega.shape[0])
        rank = int(omega.shape[-1])
        incidence, triangle_incidence, edges = self._incidence(n=n, device=omega.device, dtype=omega.dtype)
        stacked = torch.stack([omega[i, j] for i, j in edges], dim=0)

        incidence_free = incidence[:, 1:]
        solution = torch.linalg.lstsq(incidence_free, stacked).solution
        phi = torch.cat(
            [torch.zeros((1, rank), device=omega.device, dtype=omega.dtype), solution],
            dim=0,
        )
        projected = incidence @ phi
        residual = stacked - projected
        curl = triangle_incidence @ stacked

        denominator = stacked.pow(2).sum() + 1e-12
        exactness = 1.0 - residual.pow(2).sum() / denominator
        curl_energy = curl.pow(2).sum() / denominator
        return {
            "exactness": float(exactness.item()),
            "curl_energy": float(curl_energy.item()),
        }

    def _raw_projection_edges(self, nodes: Tensor) -> Tensor:
        directions = self._fixed_projection_directions(
            hidden_size=nodes.shape[-1],
            rank=self.raw_projection_rank,
            device=nodes.device,
            dtype=nodes.dtype,
        )
        projected = nodes @ directions
        return projected[:, None, :, :] - projected[:, :, None, :]

    def _raw_skew_bilinear_edges(self, nodes: Tensor) -> Tensor:
        left, right = self._fixed_skew_factors(
            hidden_size=nodes.shape[-1],
            rank=self.raw_skew_bilinear_rank,
            device=nodes.device,
            dtype=nodes.dtype,
        )
        left_proj = nodes @ left
        right_proj = nodes @ right
        return (
            left_proj[:, :, None, :] * right_proj[:, None, :, :]
            - right_proj[:, :, None, :] * left_proj[:, None, :, :]
        )

    def _subtract_positional_means(self, omega: Tensor, positions: Tensor) -> Tensor:
        centered = omega.clone()
        offsets = self._relative_rank_offsets(positions)
        unique_offsets = sorted({int(value) for value in offsets.unique().tolist() if int(value) != 0})
        for offset in unique_offsets:
            mask = offsets == offset
            if not mask.any():
                continue
            mean = centered[mask].reshape(-1, centered.shape[-1]).mean(dim=0)
            centered[mask] = centered[mask] - mean
        return self._antisymmetrize(centered)

    def _relative_rank_offsets(self, positions: Tensor) -> Tensor:
        order = torch.argsort(positions, dim=1)
        ranks = torch.empty_like(order)
        rank_template = torch.arange(positions.shape[1], dtype=order.dtype).unsqueeze(0).expand_as(order)
        ranks.scatter_(1, order, rank_template)
        return ranks[:, None, :] - ranks[:, :, None]

    def _incidence(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
        if n not in self._incidence_cache:
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            triangles = [(i, j, k) for i, j, k in combinations(range(n), 3)]
            incidence = torch.zeros((len(edges), n), dtype=torch.float32)
            edge_index: dict[tuple[int, int], tuple[int, float]] = {}
            for edge_id, (i, j) in enumerate(edges):
                incidence[edge_id, i] = -1.0
                incidence[edge_id, j] = 1.0
                edge_index[(i, j)] = (edge_id, 1.0)
                edge_index[(j, i)] = (edge_id, -1.0)
            triangle_incidence = torch.zeros((len(triangles), len(edges)), dtype=torch.float32)
            for triangle_id, (i, j, k) in enumerate(triangles):
                for source, target in ((i, j), (j, k), (k, i)):
                    edge_id, sign = edge_index[(source, target)]
                    triangle_incidence[triangle_id, edge_id] = sign
            self._incidence_cache[n] = (incidence, triangle_incidence, edges)
        incidence, triangle_incidence, edges = self._incidence_cache[n]
        return incidence.to(device=device, dtype=dtype), triangle_incidence.to(device=device, dtype=dtype), edges

    def _fixed_projection_directions(
        self,
        hidden_size: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        generator = torch.Generator().manual_seed(17_291 + hidden_size * 31 + rank)
        directions = torch.randn((hidden_size, rank), generator=generator, dtype=torch.float32)
        directions = torch.nn.functional.normalize(directions, dim=0)
        return directions.to(device=device, dtype=dtype)

    def _fixed_skew_factors(
        self,
        hidden_size: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(83_117 + hidden_size * 53 + rank)
        left = torch.randn((hidden_size, rank), generator=generator, dtype=torch.float32)
        right = torch.randn((hidden_size, rank), generator=generator, dtype=torch.float32)
        left = torch.nn.functional.normalize(left, dim=0)
        right = torch.nn.functional.normalize(right, dim=0)
        return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)

    @staticmethod
    def _antisymmetrize(omega: Tensor) -> Tensor:
        if omega.ndim == 3:
            return 0.5 * (omega - omega.transpose(0, 1))
        return 0.5 * (omega - omega.transpose(1, 2))
