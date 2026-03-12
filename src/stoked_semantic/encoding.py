from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from stoked_semantic.config import EncoderConfig
from stoked_semantic.data import PremiseQueryExample, template_family_id_for
from stoked_semantic.utils import resolve_torch_device, synchronize_if_needed


@dataclass
class LayerFeatures:
    split_name: str
    layer_index: int
    hidden_states: Tensor
    labels: Tensor
    positions: Tensor
    query_indices: Tensor
    example_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    template_family_ids: tuple[str, ...]
    presented_entities: tuple[tuple[str, str, str], ...]
    query_entities: tuple[tuple[str, str], ...]
    variant_name: str

    @property
    def nodes(self) -> Tensor:
        return self.hidden_states


@dataclass
class EncodedSplit:
    split_name: str
    variant_name: str
    example_ids: tuple[str, ...]
    premise_texts: tuple[str, ...]
    labels: Tensor
    template_ids: tuple[str, ...]
    template_family_ids: tuple[str, ...]
    presented_entities: tuple[tuple[str, str, str], ...]
    query_entities: tuple[tuple[str, str], ...]
    pooled_hidden_states: Tensor
    positions: Tensor
    query_indices: Tensor

    @property
    def num_layers(self) -> int:
        return int(self.pooled_hidden_states.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.pooled_hidden_states.shape[-1])

    def layer_view(self, layer_index: int) -> LayerFeatures:
        return LayerFeatures(
            split_name=self.split_name,
            layer_index=layer_index,
            hidden_states=self.pooled_hidden_states[:, layer_index, :, :],
            labels=self.labels,
            positions=self.positions,
            query_indices=self.query_indices,
            example_ids=self.example_ids,
            template_ids=self.template_ids,
            template_family_ids=self.template_family_ids,
            presented_entities=self.presented_entities,
            query_entities=self.query_entities,
            variant_name=self.variant_name,
        )


class FeatureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, split_name: str, variant_name: str, cache_key: str) -> Path:
        return self.cache_dir / f"{variant_name}_{split_name}_{cache_key}_features.pt"

    def load(self, split_name: str, variant_name: str, cache_key: str) -> EncodedSplit | None:
        path = self.cache_path(split_name=split_name, variant_name=variant_name, cache_key=cache_key)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        return EncodedSplit(
            split_name=payload["split_name"],
            variant_name=payload["variant_name"],
            example_ids=tuple(payload["example_ids"]),
            premise_texts=tuple(payload["premise_texts"]),
            labels=payload["labels"],
            template_ids=tuple(payload["template_ids"]),
            template_family_ids=tuple(
                payload.get(
                    "template_family_ids",
                    [template_family_id_for(template_id) for template_id in payload["template_ids"]],
                )
            ),
            presented_entities=tuple(tuple(entities) for entities in payload["presented_entities"]),
            query_entities=tuple(tuple(entities) for entities in payload["query_entities"]),
            pooled_hidden_states=payload["pooled_hidden_states"],
            positions=payload["positions"],
            query_indices=payload["query_indices"],
        )

    def save(self, split: EncodedSplit, cache_key: str) -> None:
        path = self.cache_path(
            split_name=split.split_name,
            variant_name=split.variant_name,
            cache_key=cache_key,
        )
        torch.save(
            {
                "cache_key": cache_key,
                "split_name": split.split_name,
                "variant_name": split.variant_name,
                "example_ids": list(split.example_ids),
                "premise_texts": list(split.premise_texts),
                "labels": split.labels.cpu(),
                "template_ids": list(split.template_ids),
                "template_family_ids": list(split.template_family_ids),
                "presented_entities": [list(entities) for entities in split.presented_entities],
                "query_entities": [list(entities) for entities in split.query_entities],
                "pooled_hidden_states": split.pooled_hidden_states.cpu(),
                "positions": split.positions.cpu(),
                "query_indices": split.query_indices.cpu(),
            },
            path,
        )


class TransformerFeatureExtractor:
    """Extracts span-pooled hidden states for the three tracked entities."""

    def __init__(self, config: EncoderConfig):
        self.config = config
        self.device = resolve_torch_device(config.prefer_device)
        self.cache = FeatureCache(config.cache_dir)
        self.tokenizer = self._load_tokenizer()

    def encode_split(
        self,
        split_name: str,
        examples: list[PremiseQueryExample],
        variant_name: str,
        random_weights: bool,
    ) -> EncodedSplit:
        cache_key = self._cache_key(examples=examples)
        cached = self.cache.load(
            split_name=split_name,
            variant_name=variant_name,
            cache_key=cache_key,
        )
        if cached is not None:
            return cached

        model = self._load_model(random_weights=random_weights)
        split = self._encode_examples(
            split_name=split_name,
            examples=examples,
            variant_name=variant_name,
            model=model,
        )
        self.cache.save(split, cache_key=cache_key)
        return split

    def encode_variant(
        self,
        variant_name: str,
        train_examples: list[PremiseQueryExample],
        test_examples: list[PremiseQueryExample],
        random_weights: bool,
    ) -> tuple[EncodedSplit, EncodedSplit]:
        train_cache_key = self._cache_key(examples=train_examples)
        test_cache_key = self._cache_key(examples=test_examples)
        cached_train = self.cache.load(
            split_name="train",
            variant_name=variant_name,
            cache_key=train_cache_key,
        )
        cached_test = self.cache.load(
            split_name="test",
            variant_name=variant_name,
            cache_key=test_cache_key,
        )
        if cached_train is not None and cached_test is not None:
            return cached_train, cached_test

        model = self._load_model(random_weights=random_weights)
        if cached_train is None:
            cached_train = self._encode_examples(
                split_name="train",
                examples=train_examples,
                variant_name=variant_name,
                model=model,
            )
            self.cache.save(cached_train, cache_key=train_cache_key)
        if cached_test is None:
            cached_test = self._encode_examples(
                split_name="test",
                examples=test_examples,
                variant_name=variant_name,
                model=model,
            )
            self.cache.save(cached_test, cache_key=test_cache_key)
        return cached_train, cached_test

    def _encode_examples(
        self,
        split_name: str,
        examples: list[PremiseQueryExample],
        variant_name: str,
        model: AutoModel,
    ) -> EncodedSplit:
        all_hidden: list[Tensor] = []
        all_positions: list[Tensor] = []

        for start in tqdm(
            range(0, len(examples), self.config.batch_size),
            desc=f"encode:{variant_name}:{split_name}",
            leave=False,
        ):
            batch_examples = examples[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                [example.premise_text for example in batch_examples],
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            offsets = encoded.pop("offset_mapping")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(
                    **encoded,
                    output_hidden_states=True,
                    output_attentions=self.config.include_attentions,
                )
            synchronize_if_needed(self.device)
            hidden_states = torch.stack(outputs.hidden_states, dim=0).detach().cpu()
            for batch_index, example in enumerate(batch_examples):
                pooled, positions = self._pool_example(
                    hidden_states=hidden_states[:, batch_index, :, :],
                    offsets=offsets[batch_index].tolist(),
                    example=example,
                )
                all_hidden.append(pooled)
                all_positions.append(positions)

        return EncodedSplit(
            split_name=split_name,
            variant_name=variant_name,
            example_ids=tuple(example.example_id for example in examples),
            premise_texts=tuple(example.premise_text for example in examples),
            labels=torch.tensor([example.label for example in examples], dtype=torch.long),
            template_ids=tuple(example.template_id for example in examples),
            template_family_ids=tuple(example.template_family_id for example in examples),
            presented_entities=tuple(example.presented_entities for example in examples),
            query_entities=tuple(example.query_entities for example in examples),
            pooled_hidden_states=torch.stack(all_hidden, dim=0),
            positions=torch.stack(all_positions, dim=0),
            query_indices=torch.tensor([example.query_indices for example in examples], dtype=torch.long),
        )

    def _load_model(self, random_weights: bool) -> AutoModel:
        kwargs = {"attn_implementation": "eager"} if self.config.include_attentions else {}
        if random_weights:
            config = self._load_hf_config()
            model = AutoModel.from_config(config, **kwargs)
        else:
            kwargs["use_safetensors"] = False
            model_path = self._resolve_model_snapshot()
            model = AutoModel.from_pretrained(model_path, local_files_only=True, **kwargs)
        return model.eval().to(self.device)

    def _load_tokenizer(self) -> AutoTokenizer:
        return self._from_pretrained_with_local_fallback(
            AutoTokenizer,
            self.config.model_name,
            use_fast=True,
        )

    def _load_hf_config(self) -> AutoConfig:
        return self._from_pretrained_with_local_fallback(AutoConfig, self.config.model_name)

    @staticmethod
    def _from_pretrained_with_local_fallback(loader: Any, model_name: str, **kwargs: Any) -> Any:
        try:
            return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
        except OSError:
            return loader.from_pretrained(model_name, **kwargs)

    def _resolve_model_snapshot(self) -> str:
        allow_patterns = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        ]
        try:
            return snapshot_download(
                repo_id=self.config.model_name,
                local_files_only=True,
                allow_patterns=allow_patterns,
            )
        except OSError:
            return snapshot_download(
                repo_id=self.config.model_name,
                allow_patterns=allow_patterns,
            )

    def _cache_key(self, examples: list[PremiseQueryExample]) -> str:
        digest = hashlib.sha256()
        digest.update(self.config.model_name.encode("utf-8"))
        digest.update(str(self.config.max_length).encode("utf-8"))
        digest.update(str(self.config.include_attentions).encode("utf-8"))
        for example in examples:
            digest.update(example.example_id.encode("utf-8"))
        return digest.hexdigest()[:16]

    def _pool_example(
        self,
        hidden_states: Tensor,
        offsets: list[list[int]],
        example: PremiseQueryExample,
    ) -> tuple[Tensor, Tensor]:
        pooled_by_entity: dict[str, Tensor] = {}
        positions_by_entity: dict[str, Tensor] = {}
        for entity in example.entities:
            pooled, position = self._pool_entity(
                hidden_states=hidden_states,
                offsets=offsets,
                mentions=example.entity_mentions[entity],
            )
            pooled_by_entity[entity] = pooled
            positions_by_entity[entity] = position
        pooled_entities = [pooled_by_entity[entity] for entity in example.presented_entities]
        positions = [positions_by_entity[entity] for entity in example.presented_entities]
        return torch.stack(pooled_entities, dim=1), torch.stack(positions, dim=0)

    def _pool_entity(
        self,
        hidden_states: Tensor,
        offsets: list[list[int]],
        mentions: tuple[Any, ...],
    ) -> tuple[Tensor, Tensor]:
        token_ids: list[int] = []
        for mention in mentions:
            token_ids.extend(
                self._span_token_ids(
                    offsets=offsets,
                    start_char=mention.start_char,
                    end_char=mention.end_char,
                )
            )
        unique_token_ids = sorted(set(token_ids))
        if not unique_token_ids:
            raise ValueError("No tokens found for entity mention span.")
        token_tensor = torch.tensor(unique_token_ids, dtype=torch.long)
        pooled = hidden_states[:, token_tensor, :].mean(dim=1)
        position = token_tensor.to(torch.float32).mean()
        return pooled, position

    @staticmethod
    def _span_token_ids(offsets: list[list[int]], start_char: int, end_char: int) -> list[int]:
        token_ids = []
        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if end <= start_char or start >= end_char:
                continue
            token_ids.append(token_index)
        return token_ids
