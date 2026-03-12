from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import permutations
from random import Random
from typing import Hashable

from stoked_semantic.config import DataConfig


ORDERED_QUERY_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 0),
    (2, 0),
    (2, 1),
)


@dataclass(frozen=True)
class MentionSpan:
    start_char: int
    end_char: int


@dataclass(frozen=True)
class RelationSpec:
    relation_id: str
    relation_phrase: str
    inverse_relation_phrase: str
    paraphrase_relation_phrase: str
    paraphrase_inverse_relation_phrase: str


RELATION_LIBRARY: dict[str, RelationSpec] = {
    "outranks": RelationSpec(
        relation_id="outranks",
        relation_phrase="outranks",
        inverse_relation_phrase="is outranked by",
        paraphrase_relation_phrase="ranks above",
        paraphrase_inverse_relation_phrase="ranks below",
    ),
    "older_than": RelationSpec(
        relation_id="older_than",
        relation_phrase="is older than",
        inverse_relation_phrase="is younger than",
        paraphrase_relation_phrase="is senior to",
        paraphrase_inverse_relation_phrase="is junior to",
    ),
}


@dataclass(frozen=True)
class PremiseQueryExample:
    example_id: str
    split_name: str
    premise_text: str
    label: int
    relation_id: str
    entities: tuple[str, str, str]
    entity_mentions: dict[str, tuple[MentionSpan, ...]]
    template_id: str
    template_family_id: str
    query_entities: tuple[str, str]
    query_indices: tuple[int, int]
    presentation_order: tuple[int, int, int]

    @property
    def presented_entities(self) -> tuple[str, str, str]:
        return tuple(self.entities[index] for index in self.presentation_order)


@dataclass(frozen=True)
class DatasetBundle:
    train_examples: list[PremiseQueryExample]
    test_examples: list[PremiseQueryExample]


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    template_family_id: str
    template: str

    def render(self, a: str, b: str, c: str, relation: RelationSpec) -> str:
        return self.template.format(
            a=a,
            b=b,
            c=c,
            relation=relation.relation_phrase,
            inverse_relation=relation.inverse_relation_phrase,
            paraphrase_relation=relation.paraphrase_relation_phrase,
            paraphrase_inverse_relation=relation.paraphrase_inverse_relation_phrase,
        )


TEMPLATE_LIBRARY: dict[str, TemplateSpec] = {
    "active_forward": TemplateSpec(
        template_id="active_forward",
        template_family_id="active",
        template="{a} {relation} {b}. {b} {relation} {c}.",
    ),
    "active_reverse": TemplateSpec(
        template_id="active_reverse",
        template_family_id="active",
        template="{b} {relation} {c}. {a} {relation} {b}.",
    ),
    "passive_forward": TemplateSpec(
        template_id="passive_forward",
        template_family_id="passive",
        template="{b} {inverse_relation} {a}. {c} {inverse_relation} {b}.",
    ),
    "passive_reverse": TemplateSpec(
        template_id="passive_reverse",
        template_family_id="passive",
        template="{c} {inverse_relation} {b}. {b} {inverse_relation} {a}.",
    ),
    "mixed_ab_inverse": TemplateSpec(
        template_id="mixed_ab_inverse",
        template_family_id="mixed_ab_inverse",
        template="{b} {inverse_relation} {a}. {b} {relation} {c}.",
    ),
    "mixed_ab_inverse_reverse": TemplateSpec(
        template_id="mixed_ab_inverse_reverse",
        template_family_id="mixed_ab_inverse",
        template="{b} {relation} {c}. {b} {inverse_relation} {a}.",
    ),
    "mixed_bc_inverse": TemplateSpec(
        template_id="mixed_bc_inverse",
        template_family_id="mixed_bc_inverse",
        template="{a} {relation} {b}. {c} {inverse_relation} {b}.",
    ),
    "mixed_bc_inverse_reverse": TemplateSpec(
        template_id="mixed_bc_inverse_reverse",
        template_family_id="mixed_bc_inverse",
        template="{c} {inverse_relation} {b}. {a} {relation} {b}.",
    ),
    "above_forward": TemplateSpec(
        template_id="above_forward",
        template_family_id="above",
        template="{a} {paraphrase_relation} {b}. {b} {paraphrase_relation} {c}.",
    ),
    "above_reverse": TemplateSpec(
        template_id="above_reverse",
        template_family_id="above",
        template="{b} {paraphrase_relation} {c}. {a} {paraphrase_relation} {b}.",
    ),
    "below_forward": TemplateSpec(
        template_id="below_forward",
        template_family_id="below",
        template="{b} {paraphrase_inverse_relation} {a}. {c} {paraphrase_inverse_relation} {b}.",
    ),
    "below_reverse": TemplateSpec(
        template_id="below_reverse",
        template_family_id="below",
        template="{c} {paraphrase_inverse_relation} {b}. {b} {paraphrase_inverse_relation} {a}.",
    ),
    "paraphrase_ab_inverse": TemplateSpec(
        template_id="paraphrase_ab_inverse",
        template_family_id="paraphrase_ab_inverse",
        template="{b} {paraphrase_inverse_relation} {a}. {b} {paraphrase_relation} {c}.",
    ),
    "paraphrase_ab_inverse_reverse": TemplateSpec(
        template_id="paraphrase_ab_inverse_reverse",
        template_family_id="paraphrase_ab_inverse",
        template="{b} {paraphrase_relation} {c}. {b} {paraphrase_inverse_relation} {a}.",
    ),
    "paraphrase_bc_inverse": TemplateSpec(
        template_id="paraphrase_bc_inverse",
        template_family_id="paraphrase_bc_inverse",
        template="{a} {paraphrase_relation} {b}. {c} {paraphrase_inverse_relation} {b}.",
    ),
    "paraphrase_bc_inverse_reverse": TemplateSpec(
        template_id="paraphrase_bc_inverse_reverse",
        template_family_id="paraphrase_bc_inverse",
        template="{c} {paraphrase_inverse_relation} {b}. {a} {paraphrase_relation} {b}.",
    ),
}


def available_relation_ids() -> tuple[str, ...]:
    return tuple(sorted(RELATION_LIBRARY))


def available_template_ids() -> tuple[str, ...]:
    return tuple(sorted(TEMPLATE_LIBRARY))


def template_family_id_for(template_id: str) -> str:
    try:
        return TEMPLATE_LIBRARY[template_id].template_family_id
    except KeyError as exc:
        available = ", ".join(available_template_ids())
        raise ValueError(f"Unknown template_id '{template_id}'. Available: {available}") from exc


def available_template_family_ids() -> tuple[str, ...]:
    return tuple(sorted({template.template_family_id for template in TEMPLATE_LIBRARY.values()}))


class SyntheticConsistencyDatasetBuilder:
    """Builds the lexical-holdout premise-query corpus used in phase 1."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._random = Random(config.seed)
        self._relation_specs = tuple(self._resolve_relation(relation_id) for relation_id in config.relation_ids)
        self._templates_by_split = {
            "train": tuple(self._resolve_template(template_id) for template_id in config.train_template_ids),
            "test": tuple(self._resolve_template(template_id) for template_id in config.test_template_ids),
        }
        self._presentation_orders = tuple(permutations(range(3), 3))

    def build(self) -> DatasetBundle:
        train_examples = self._build_split(
            split_name="train",
            names=self.config.train_names,
            max_examples_per_label=self.config.train_examples_per_label,
        )
        test_examples = self._build_split(
            split_name="test",
            names=self.config.test_names,
            max_examples_per_label=self.config.test_examples_per_label,
        )
        return DatasetBundle(train_examples=train_examples, test_examples=test_examples)

    def _build_split(
        self,
        split_name: str,
        names: tuple[str, ...],
        max_examples_per_label: int,
    ) -> list[PremiseQueryExample]:
        all_examples = self._enumerate_examples(
            split_name=split_name,
            names=names,
            templates=self._templates_by_split[split_name],
        )
        kept = self._stratified_sample(
            all_examples=all_examples,
            max_examples_per_label=max_examples_per_label,
        )
        self._random.shuffle(kept)
        return kept

    def _enumerate_examples(
        self,
        split_name: str,
        names: tuple[str, ...],
        templates: tuple[TemplateSpec, ...],
    ) -> list[PremiseQueryExample]:
        examples: list[PremiseQueryExample] = []
        for relation in self._relation_specs:
            for template in templates:
                for world_index, (a, b, c) in enumerate(permutations(names, 3)):
                    entities = (a, b, c)
                    premise_text = template.render(a=a, b=b, c=c, relation=relation)
                    entity_mentions = self._find_mentions(text=premise_text, entities=entities)
                    for query_index, (source_index, target_index) in enumerate(ORDERED_QUERY_PAIRS):
                        for order_index, presentation_order in enumerate(self._presentation_orders):
                            query_indices = self._query_indices(
                                presentation_order=presentation_order,
                                source_index=source_index,
                                target_index=target_index,
                            )
                            label = int(source_index < target_index)
                            query_entities = (entities[source_index], entities[target_index])
                            presentation_tag = "".join(str(index) for index in presentation_order)
                            example_id = (
                                f"{split_name}-{relation.relation_id}-{template.template_id}-{world_index}-"
                                f"{query_index}-{order_index}-{presentation_tag}-{a}-{b}-{c}"
                            )
                            examples.append(
                                PremiseQueryExample(
                                    example_id=example_id,
                                    split_name=split_name,
                                    premise_text=premise_text,
                                    label=label,
                                    relation_id=relation.relation_id,
                                    entities=entities,
                                    entity_mentions=entity_mentions,
                                    template_id=template.template_id,
                                    template_family_id=template.template_family_id,
                                    query_entities=query_entities,
                                    query_indices=query_indices,
                                    presentation_order=presentation_order,
                                )
                            )
        return examples

    def _stratified_sample(
        self,
        all_examples: list[PremiseQueryExample],
        max_examples_per_label: int,
    ) -> list[PremiseQueryExample]:
        strata = sorted(
            {
                (example.relation_id, example.template_id, example.query_indices)
                for example in all_examples
            }
        )
        per_bucket_targets = self._per_bucket_targets(
            total_examples=max_examples_per_label,
            buckets=strata,
        )

        grouped: dict[tuple[int, str, str, tuple[int, int]], list[PremiseQueryExample]] = {}
        for example in all_examples:
            grouped.setdefault(
                (example.label, example.relation_id, example.template_id, example.query_indices),
                [],
            ).append(example)

        sampled: list[PremiseQueryExample] = []
        for label in (0, 1):
            for relation_id, template_id, query_indices in strata:
                bucket = grouped[(label, relation_id, template_id, query_indices)]
                target_size = per_bucket_targets[(relation_id, template_id, query_indices)]
                if target_size == 0:
                    continue
                if len(bucket) < target_size:
                    raise ValueError(
                        "Not enough examples for "
                        f"label={label}, relation_id={relation_id}, "
                        f"template_id={template_id}, query_indices={query_indices}: "
                        f"needed {target_size}, found {len(bucket)}"
                    )
                self._random.shuffle(bucket)
                sampled.extend(bucket[:target_size])
        return sampled

    def _per_bucket_targets(
        self,
        total_examples: int,
        buckets: list[Hashable],
    ) -> dict[Hashable, int]:
        base = total_examples // len(buckets)
        remainder = total_examples % len(buckets)
        targets = {bucket: base for bucket in buckets}
        bucket_order = list(buckets)
        self._random.shuffle(bucket_order)
        for bucket in bucket_order[:remainder]:
            targets[bucket] += 1
        return targets

    @staticmethod
    def _query_indices(
        presentation_order: tuple[int, int, int],
        source_index: int,
        target_index: int,
    ) -> tuple[int, int]:
        slot_lookup = {
            canonical_index: slot_index
            for slot_index, canonical_index in enumerate(presentation_order)
        }
        return slot_lookup[source_index], slot_lookup[target_index]

    def _resolve_relation(self, relation_id: str) -> RelationSpec:
        try:
            return RELATION_LIBRARY[relation_id]
        except KeyError as exc:
            available = ", ".join(available_relation_ids())
            raise ValueError(f"Unknown relation_id '{relation_id}'. Available: {available}") from exc

    def _resolve_template(self, template_id: str) -> TemplateSpec:
        try:
            return TEMPLATE_LIBRARY[template_id]
        except KeyError as exc:
            available = ", ".join(available_template_ids())
            raise ValueError(f"Unknown template_id '{template_id}'. Available: {available}") from exc

    def _find_mentions(
        self,
        text: str,
        entities: tuple[str, str, str],
    ) -> dict[str, tuple[MentionSpan, ...]]:
        mentions: dict[str, tuple[MentionSpan, ...]] = {}
        for entity in entities:
            spans = tuple(
                MentionSpan(start_char=match.start(), end_char=match.end())
                for match in re.finditer(re.escape(entity), text)
            )
            if not spans:
                raise ValueError(f"Could not locate entity '{entity}' in text: {text}")
            mentions[entity] = spans
        return mentions
