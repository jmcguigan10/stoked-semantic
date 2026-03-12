from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations, permutations
from random import Random
from typing import Callable, Hashable

from stoked_semantic.config import (
    DataConfig,
    DEFAULT_RELATION_IDS,
    DEFAULT_TEMPLATE_IDS,
    PHASE1_TASK_SUITE,
    PHASE2_DEFAULT_RELATION_IDS,
    PHASE2_DEFAULT_TEMPLATE_IDS,
    PHASE2_TASK_SUITE,
    PHASE3_DEFAULT_RELATION_IDS,
    PHASE3_DEFAULT_TEMPLATE_IDS,
    PHASE3_TASK_SUITE,
    PHASE4_DEFAULT_RELATION_IDS,
    PHASE4_TASK_SUITE,
)


@dataclass(frozen=True)
class MentionSpan:
    start_char: int
    end_char: int


@dataclass(frozen=True)
class PremiseQueryExample:
    example_id: str
    split_name: str
    task_suite: str
    premise_text: str
    label: int
    relation_id: str
    entities: tuple[str, ...]
    entity_mentions: dict[str, tuple[MentionSpan, ...]]
    template_id: str
    template_family_id: str
    query_entities: tuple[str, ...]
    query_indices: tuple[int, ...]
    query_distance_type: str
    query_direction_type: str
    query_type: str
    presentation_order: tuple[int, ...]

    @property
    def presented_entities(self) -> tuple[str, ...]:
        return tuple(self.entities[index] for index in self.presentation_order)


@dataclass(frozen=True)
class DatasetBundle:
    train_examples: list[PremiseQueryExample]
    test_examples: list[PremiseQueryExample]


@dataclass(frozen=True)
class RelationSpec:
    relation_id: str
    relation_phrase: str
    inverse_relation_phrase: str
    paraphrase_relation_phrase: str
    paraphrase_inverse_relation_phrase: str


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


@dataclass(frozen=True)
class FourNodeTemplateVariant:
    template_id: str
    template_family_id: str
    clause_order: tuple[int, ...]


@dataclass(frozen=True)
class BalancedTripletTemplateVariant:
    template_id: str
    template_family_id: str
    clause_order: tuple[int, ...]
    clause_template: str


@dataclass(frozen=True)
class FourNodeTaskSpec:
    relation_id: str
    clause_templates: tuple[str, ...]
    template_variants: tuple[FourNodeTemplateVariant, ...]
    support_edges: tuple[tuple[int, int], ...]
    query_arity: int
    label_fn: Callable[[tuple[int, ...]], bool]
    query_metadata_fn: Callable[[tuple[int, ...]], tuple[str, str, str]] | None = None

    @property
    def template_ids(self) -> tuple[str, ...]:
        return tuple(variant.template_id for variant in self.template_variants)

    def render(self, entities: tuple[str, str, str, str], template_id: str) -> str:
        variant = self._variant(template_id)
        fields = {f"e{index}": entity for index, entity in enumerate(entities)}
        clauses = [template.format(**fields) for template in self.clause_templates]
        return " ".join(clauses[index] for index in variant.clause_order)

    def template_family_id_for(self, template_id: str) -> str:
        return self._variant(template_id).template_family_id

    def label_for(self, query_nodes: tuple[int, ...]) -> int:
        return int(self.label_fn(query_nodes))

    def query_metadata(self, query_nodes: tuple[int, ...]) -> tuple[str, str, str]:
        if self.query_metadata_fn is not None:
            return self.query_metadata_fn(query_nodes)
        if len(query_nodes) == 2:
            source_index, target_index = query_nodes
            distance = _shortest_path_distance(
                node_count=4,
                edges=self.support_edges,
                source_index=source_index,
                target_index=target_index,
            )
            distance_type = f"path_{distance}"
            direction_type = "forward" if source_index < target_index else "reversed"
            return distance_type, direction_type, f"{distance_type}_{direction_type}"
        return ("query", "ordered", "query_ordered")

    def _variant(self, template_id: str) -> FourNodeTemplateVariant:
        for variant in self.template_variants:
            if variant.template_id == template_id:
                return variant
        available = ", ".join(self.template_ids)
        raise ValueError(
            f"Template '{template_id}' is not available for task '{self.relation_id}'. "
            f"Available: {available}"
        )


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

FOUR_NODE_TASK_LIBRARY: dict[str, FourNodeTaskSpec] = {
    "chain_order": FourNodeTaskSpec(
        relation_id="chain_order",
        clause_templates=(
            "{e0} outranks {e1}.",
            "{e1} outranks {e2}.",
            "{e2} outranks {e3}.",
        ),
        template_variants=(
            FourNodeTemplateVariant(
                template_id="chain_forward",
                template_family_id="chain",
                clause_order=(0, 1, 2),
            ),
            FourNodeTemplateVariant(
                template_id="chain_reverse",
                template_family_id="chain",
                clause_order=(2, 1, 0),
            ),
        ),
        support_edges=((0, 1), (1, 2), (2, 3)),
        query_arity=2,
        label_fn=lambda query: query[0] < query[1],
    ),
    "same_side": FourNodeTaskSpec(
        relation_id="same_side",
        clause_templates=(
            "{e0} aligns with {e1}.",
            "{e2} aligns with {e3}.",
            "{e0} opposes {e2}.",
        ),
        template_variants=(
            FourNodeTemplateVariant(
                template_id="team_forward",
                template_family_id="team",
                clause_order=(0, 1, 2),
            ),
            FourNodeTemplateVariant(
                template_id="team_reverse",
                template_family_id="team",
                clause_order=(2, 1, 0),
            ),
        ),
        support_edges=((0, 1), (2, 3), (0, 2)),
        query_arity=2,
        label_fn=lambda query: (
            (query[0] in (0, 1) and query[1] in (0, 1))
            or (query[0] in (2, 3) and query[1] in (2, 3))
        ),
    ),
    "same_row": FourNodeTaskSpec(
        relation_id="same_row",
        clause_templates=(
            "{e0} is left of {e1}.",
            "{e2} is left of {e3}.",
            "{e0} is above {e2}.",
            "{e1} is above {e3}.",
        ),
        template_variants=(
            FourNodeTemplateVariant(
                template_id="grid_rows_first",
                template_family_id="grid",
                clause_order=(0, 1, 2, 3),
            ),
            FourNodeTemplateVariant(
                template_id="grid_columns_first",
                template_family_id="grid",
                clause_order=(2, 3, 0, 1),
            ),
        ),
        support_edges=((0, 1), (2, 3), (0, 2), (1, 3)),
        query_arity=2,
        label_fn=lambda query: (query[0] // 2) == (query[1] // 2),
    ),
    "same_triplet": FourNodeTaskSpec(
        relation_id="same_triplet",
        clause_templates=(
            "{e0}, {e1}, and {e2} form a circuit.",
            "{e1}, {e2}, and {e3} form a circuit.",
        ),
        template_variants=(
            FourNodeTemplateVariant(
                template_id="triplet_forward",
                template_family_id="triplet",
                clause_order=(0, 1),
            ),
            FourNodeTemplateVariant(
                template_id="triplet_reverse",
                template_family_id="triplet",
                clause_order=(1, 0),
            ),
        ),
        support_edges=((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)),
        query_arity=3,
        label_fn=lambda query: frozenset(query) in (frozenset((0, 1, 2)), frozenset((1, 2, 3))),
        query_metadata_fn=lambda query: (
            "triple",
            "ordered",
            "triple_ordered",
        ),
    ),
}

PHASE2_TEMPLATE_FAMILY_MAP = {
    variant.template_id: variant.template_family_id
    for task in FOUR_NODE_TASK_LIBRARY.values()
    for variant in task.template_variants
}

BALANCED_TRIPLET_RELATION_ID = "balanced_triplet"
MASKED_BALANCED_TRIPLET_RELATION_ID = "masked_balanced_triplet"
BALANCED_TRIPLET_POSITIVE_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 4),
    (0, 3, 5),
    (0, 4, 5),
    (1, 2, 5),
    (1, 3, 4),
    (1, 4, 5),
    (2, 3, 4),
    (2, 3, 5),
)
BALANCED_TRIPLET_POSITIVE_SET = frozenset(frozenset(triple) for triple in BALANCED_TRIPLET_POSITIVE_TRIPLES)
BALANCED_TRIPLET_POSITIVE_INDEX = {
    frozenset(triple): index for index, triple in enumerate(BALANCED_TRIPLET_POSITIVE_TRIPLES)
}
BALANCED_TRIPLET_TEMPLATE_VARIANTS: tuple[BalancedTripletTemplateVariant, ...] = (
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_link_forward",
        template_family_id="balanced_link",
        clause_order=tuple(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES))),
        clause_template="{a} {b} {c} link.",
    ),
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_link_reverse",
        template_family_id="balanced_link",
        clause_order=tuple(reversed(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES)))),
        clause_template="{a} {b} {c} link.",
    ),
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_sync_forward",
        template_family_id="balanced_sync",
        clause_order=tuple(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES))),
        clause_template="{a}, {b}, and {c} sync.",
    ),
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_sync_reverse",
        template_family_id="balanced_sync",
        clause_order=tuple(reversed(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES)))),
        clause_template="{a}, {b}, and {c} sync.",
    ),
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_fit_forward",
        template_family_id="balanced_fit",
        clause_order=tuple(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES))),
        clause_template="{a}, {b}, and {c} fit.",
    ),
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_fit_reverse",
        template_family_id="balanced_fit",
        clause_order=tuple(reversed(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES)))),
        clause_template="{a}, {b}, and {c} fit.",
    ),
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_align_forward",
        template_family_id="balanced_align",
        clause_order=tuple(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES))),
        clause_template="{a} aligns with {b} and {c}.",
    ),
    BalancedTripletTemplateVariant(
        template_id="balanced_triplet_align_reverse",
        template_family_id="balanced_align",
        clause_order=tuple(reversed(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES)))),
        clause_template="{a} aligns with {b} and {c}.",
    ),
)
PHASE3_TEMPLATE_FAMILY_MAP = {
    variant.template_id: variant.template_family_id
    for variant in BALANCED_TRIPLET_TEMPLATE_VARIANTS
}


def ordered_query_pairs(node_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (source_index, target_index)
        for source_index in range(node_count)
        for target_index in range(node_count)
        if source_index != target_index
    )


def ordered_query_tuples(node_count: int, query_arity: int) -> tuple[tuple[int, ...], ...]:
    return tuple(permutations(range(node_count), query_arity))


def available_relation_ids(task_suite: str = PHASE1_TASK_SUITE) -> tuple[str, ...]:
    if task_suite == PHASE1_TASK_SUITE:
        return tuple(sorted(RELATION_LIBRARY))
    if task_suite == PHASE2_TASK_SUITE:
        return tuple(sorted(FOUR_NODE_TASK_LIBRARY))
    if task_suite == PHASE3_TASK_SUITE:
        return PHASE3_DEFAULT_RELATION_IDS
    if task_suite == PHASE4_TASK_SUITE:
        return PHASE4_DEFAULT_RELATION_IDS
    raise ValueError(f"Unsupported task suite: {task_suite}")


def available_template_ids(task_suite: str = PHASE1_TASK_SUITE) -> tuple[str, ...]:
    if task_suite == PHASE1_TASK_SUITE:
        return tuple(sorted(TEMPLATE_LIBRARY))
    if task_suite == PHASE2_TASK_SUITE:
        return tuple(sorted(PHASE2_TEMPLATE_FAMILY_MAP))
    if task_suite in {PHASE3_TASK_SUITE, PHASE4_TASK_SUITE}:
        return tuple(sorted(PHASE3_TEMPLATE_FAMILY_MAP))
    raise ValueError(f"Unsupported task suite: {task_suite}")


def template_family_id_for(template_id: str) -> str:
    if template_id in TEMPLATE_LIBRARY:
        return TEMPLATE_LIBRARY[template_id].template_family_id
    if template_id in PHASE2_TEMPLATE_FAMILY_MAP:
        return PHASE2_TEMPLATE_FAMILY_MAP[template_id]
    if template_id in PHASE3_TEMPLATE_FAMILY_MAP:
        return PHASE3_TEMPLATE_FAMILY_MAP[template_id]
    available = ", ".join(
        sorted(
            set(TEMPLATE_LIBRARY)
            | set(PHASE2_TEMPLATE_FAMILY_MAP)
            | set(PHASE3_TEMPLATE_FAMILY_MAP)
        )
    )
    raise ValueError(f"Unknown template_id '{template_id}'. Available: {available}")


def available_template_family_ids(task_suite: str = PHASE1_TASK_SUITE) -> tuple[str, ...]:
    if task_suite == PHASE1_TASK_SUITE:
        return tuple(sorted({template.template_family_id for template in TEMPLATE_LIBRARY.values()}))
    if task_suite == PHASE2_TASK_SUITE:
        return tuple(sorted(set(PHASE2_TEMPLATE_FAMILY_MAP.values())))
    if task_suite in {PHASE3_TASK_SUITE, PHASE4_TASK_SUITE}:
        return tuple(sorted(set(PHASE3_TEMPLATE_FAMILY_MAP.values())))
    raise ValueError(f"Unsupported task suite: {task_suite}")


def relation_query_arity(task_suite: str, relation_id: str) -> int:
    if task_suite == PHASE1_TASK_SUITE:
        if relation_id not in RELATION_LIBRARY:
            available = ", ".join(available_relation_ids(task_suite))
            raise ValueError(f"Unknown relation_id '{relation_id}'. Available: {available}")
        return 2
    if task_suite == PHASE2_TASK_SUITE:
        try:
            return FOUR_NODE_TASK_LIBRARY[relation_id].query_arity
        except KeyError as exc:
            available = ", ".join(available_relation_ids(task_suite))
            raise ValueError(f"Unknown relation_id '{relation_id}'. Available: {available}") from exc
    if task_suite == PHASE3_TASK_SUITE:
        if relation_id != BALANCED_TRIPLET_RELATION_ID:
            available = ", ".join(available_relation_ids(task_suite))
            raise ValueError(f"Unknown relation_id '{relation_id}'. Available: {available}")
        return 3
    if task_suite == PHASE4_TASK_SUITE:
        if relation_id != MASKED_BALANCED_TRIPLET_RELATION_ID:
            available = ", ".join(available_relation_ids(task_suite))
            raise ValueError(f"Unknown relation_id '{relation_id}'. Available: {available}")
        return 3
    raise ValueError(f"Unsupported task suite: {task_suite}")


def make_dataset_builder(config: DataConfig) -> "_BaseDatasetBuilder":
    if config.task_suite == PHASE1_TASK_SUITE:
        return SyntheticConsistencyDatasetBuilder(config)
    if config.task_suite == PHASE2_TASK_SUITE:
        return FourNodeTaskDatasetBuilder(config)
    if config.task_suite == PHASE3_TASK_SUITE:
        return BalancedTripletDatasetBuilder(config)
    if config.task_suite == PHASE4_TASK_SUITE:
        return MaskedBalancedTripletDatasetBuilder(config)
    raise ValueError(f"Unsupported task suite: {config.task_suite}")


class _BaseDatasetBuilder:
    def __init__(self, config: DataConfig):
        self.config = config
        self._random = Random(config.seed)

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
        raise NotImplementedError

    def _per_bucket_targets(
        self,
        total_examples: int,
        buckets: list[Hashable],
    ) -> dict[Hashable, int]:
        if not buckets:
            return {}
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
        presentation_order: tuple[int, ...],
        source_index: int,
        target_index: int,
    ) -> tuple[int, int]:
        slot_lookup = {
            canonical_index: slot_index
            for slot_index, canonical_index in enumerate(presentation_order)
        }
        return slot_lookup[source_index], slot_lookup[target_index]

    def _find_mentions(
        self,
        text: str,
        entities: tuple[str, ...],
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


class SyntheticConsistencyDatasetBuilder(_BaseDatasetBuilder):
    """Builds the lexical-holdout 3-node premise-query corpus used in phase 1."""

    def __init__(self, config: DataConfig):
        super().__init__(config)
        relation_ids = config.relation_ids or DEFAULT_RELATION_IDS
        train_template_ids = config.train_template_ids or DEFAULT_TEMPLATE_IDS
        test_template_ids = config.test_template_ids or train_template_ids
        self._relation_specs = tuple(self._resolve_relation(relation_id) for relation_id in relation_ids)
        self._templates_by_split = {
            "train": tuple(self._resolve_template(template_id) for template_id in train_template_ids),
            "test": tuple(self._resolve_template(template_id) for template_id in test_template_ids),
        }
        self._presentation_orders = tuple(permutations(range(3), 3))

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
                for world_index, entities in enumerate(permutations(names, 3)):
                    a, b, c = entities
                    premise_text = template.render(a=a, b=b, c=c, relation=relation)
                    entity_mentions = self._find_mentions(text=premise_text, entities=entities)
                    for query_index, (source_index, target_index) in enumerate(ordered_query_pairs(3)):
                        for order_index, presentation_order in enumerate(self._presentation_orders):
                            query_indices = self._query_indices(
                                presentation_order=presentation_order,
                                source_index=source_index,
                                target_index=target_index,
                            )
                            query_distance_type = (
                                "closure" if abs(source_index - target_index) == 2 else "adjacent"
                            )
                            query_direction_type = (
                                "forward" if source_index < target_index else "reversed"
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
                                    task_suite=self.config.task_suite,
                                    premise_text=premise_text,
                                    label=label,
                                    relation_id=relation.relation_id,
                                    entities=entities,
                                    entity_mentions=entity_mentions,
                                    template_id=template.template_id,
                                    template_family_id=template.template_family_id,
                                    query_entities=query_entities,
                                    query_indices=query_indices,
                                    query_distance_type=query_distance_type,
                                    query_direction_type=query_direction_type,
                                    query_type=f"{query_distance_type}_{query_direction_type}",
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

    def _resolve_relation(self, relation_id: str) -> RelationSpec:
        try:
            return RELATION_LIBRARY[relation_id]
        except KeyError as exc:
            available = ", ".join(available_relation_ids(PHASE1_TASK_SUITE))
            raise ValueError(f"Unknown relation_id '{relation_id}'. Available: {available}") from exc

    def _resolve_template(self, template_id: str) -> TemplateSpec:
        try:
            return TEMPLATE_LIBRARY[template_id]
        except KeyError as exc:
            available = ", ".join(available_template_ids(PHASE1_TASK_SUITE))
            raise ValueError(f"Unknown template_id '{template_id}'. Available: {available}") from exc


@dataclass(frozen=True)
class _FourNodeQueryPlan:
    relation_id: str
    template_id: str
    template_family_id: str
    query_nodes: tuple[int, ...]
    label: int
    query_indices: tuple[int, ...]
    query_distance_type: str
    query_direction_type: str
    query_type: str
    presentation_order: tuple[int, int, int, int]


class FourNodeTaskDatasetBuilder(_BaseDatasetBuilder):
    """Builds the 4-node control and non-order task suite used in phase 2."""

    def __init__(self, config: DataConfig):
        super().__init__(config)
        relation_ids = config.relation_ids or PHASE2_DEFAULT_RELATION_IDS
        self._task_specs = tuple(self._resolve_task(relation_id) for relation_id in relation_ids)
        self._template_ids_by_split = {
            "train": tuple(config.train_template_ids or PHASE2_DEFAULT_TEMPLATE_IDS),
            "test": tuple(config.test_template_ids or config.train_template_ids or PHASE2_DEFAULT_TEMPLATE_IDS),
        }
        self._presentation_orders = tuple(permutations(range(4), 4))

    def _build_split(
        self,
        split_name: str,
        names: tuple[str, ...],
        max_examples_per_label: int,
    ) -> list[PremiseQueryExample]:
        if len(names) < 4:
            raise ValueError(
                f"Phase-2 four-node tasks require at least 4 names in the {split_name} split. "
                f"Received {len(names)}."
            )
        available_worlds = list(permutations(names, 4))
        plans_by_label = self._plans_by_label(split_name=split_name)

        sampled: list[PremiseQueryExample] = []
        for label, plan_lookup in plans_by_label.items():
            strata = sorted(plan_lookup)
            targets = self._per_bucket_targets(
                total_examples=max_examples_per_label,
                buckets=strata,
            )
            for bucket_index, bucket in enumerate(strata):
                target_size = targets[bucket]
                if target_size == 0:
                    continue
                plans = plan_lookup[bucket]
                sampled.extend(
                    self._materialize_examples(
                        split_name=split_name,
                        label=label,
                        bucket_index=bucket_index,
                        bucket=bucket,
                        plans=plans,
                        worlds=available_worlds,
                        target_size=target_size,
                    )
                )

        self._random.shuffle(sampled)
        return sampled

    def _plans_by_label(
        self,
        split_name: str,
    ) -> dict[int, dict[tuple[str, str, tuple[int, int]], list[_FourNodeQueryPlan]]]:
        selected_templates = set(self._template_ids_by_split[split_name])
        plans_by_label: dict[int, dict[tuple[str, str, tuple[int, int]], list[_FourNodeQueryPlan]]] = {
            0: defaultdict(list),
            1: defaultdict(list),
        }
        for task in self._task_specs:
            variants = [variant for variant in task.template_variants if variant.template_id in selected_templates]
            if not variants:
                available = ", ".join(task.template_ids)
                requested = ", ".join(sorted(selected_templates))
                raise ValueError(
                    f"No usable templates for task '{task.relation_id}' in split '{split_name}'. "
                    f"Requested templates: {requested}. Available for task: {available}"
                )
            for variant in variants:
                for query_nodes in ordered_query_tuples(4, task.query_arity):
                    label = task.label_for(query_nodes)
                    query_distance_type, query_direction_type, query_type = task.query_metadata(query_nodes)
                    for presentation_order in self._presentation_orders:
                        query_indices = self._query_indices(
                            presentation_order=presentation_order,
                            query_nodes=query_nodes,
                        )
                        bucket = (task.relation_id, variant.template_id, query_indices)
                        plans_by_label[label][bucket].append(
                            _FourNodeQueryPlan(
                                relation_id=task.relation_id,
                                template_id=variant.template_id,
                                template_family_id=variant.template_family_id,
                                query_nodes=query_nodes,
                                label=label,
                                query_indices=query_indices,
                                query_distance_type=query_distance_type,
                                query_direction_type=query_direction_type,
                                query_type=query_type,
                                presentation_order=presentation_order,
                            )
                        )
        return plans_by_label

    def _materialize_examples(
        self,
        split_name: str,
        label: int,
        bucket_index: int,
        bucket: tuple[str, str, tuple[int, int]],
        plans: list[_FourNodeQueryPlan],
        worlds: list[tuple[str, str, str, str]],
        target_size: int,
    ) -> list[PremiseQueryExample]:
        selected_worlds = self._sample_worlds(worlds=worlds, target_size=target_size)
        selected_plans = self._sample_plans(plans=plans, target_size=target_size)
        task_lookup = {task.relation_id: task for task in self._task_specs}

        examples: list[PremiseQueryExample] = []
        for sample_index, (entities, plan) in enumerate(zip(selected_worlds, selected_plans, strict=True)):
            task = task_lookup[plan.relation_id]
            premise_text = task.render(entities=entities, template_id=plan.template_id)
            entity_mentions = self._find_mentions(text=premise_text, entities=entities)
            query_entities = tuple(entities[index] for index in plan.query_nodes)
            presentation_tag = "".join(str(index) for index in plan.presentation_order)
            example_id = (
                f"{split_name}-{self.config.task_suite}-{plan.relation_id}-{plan.template_id}-"
                f"{label}-{bucket_index}-{sample_index}-{presentation_tag}-{'-'.join(entities)}"
            )
            examples.append(
                PremiseQueryExample(
                    example_id=example_id,
                    split_name=split_name,
                    task_suite=self.config.task_suite,
                    premise_text=premise_text,
                    label=plan.label,
                    relation_id=plan.relation_id,
                    entities=entities,
                    entity_mentions=entity_mentions,
                    template_id=plan.template_id,
                    template_family_id=plan.template_family_id,
                    query_entities=query_entities,
                    query_indices=plan.query_indices,
                    query_distance_type=plan.query_distance_type,
                    query_direction_type=plan.query_direction_type,
                    query_type=plan.query_type,
                    presentation_order=plan.presentation_order,
                )
            )
        return examples

    def _sample_worlds(
        self,
        worlds: list[tuple[str, str, str, str]],
        target_size: int,
    ) -> list[tuple[str, str, str, str]]:
        if target_size <= len(worlds):
            return self._random.sample(worlds, k=target_size)
        return [worlds[self._random.randrange(len(worlds))] for _ in range(target_size)]

    def _sample_plans(
        self,
        plans: list[_FourNodeQueryPlan],
        target_size: int,
    ) -> list[_FourNodeQueryPlan]:
        if target_size <= len(plans):
            return self._random.sample(plans, k=target_size)
        return [plans[self._random.randrange(len(plans))] for _ in range(target_size)]

    def _resolve_task(self, relation_id: str) -> FourNodeTaskSpec:
        try:
            return FOUR_NODE_TASK_LIBRARY[relation_id]
        except KeyError as exc:
            available = ", ".join(available_relation_ids(PHASE2_TASK_SUITE))
            raise ValueError(f"Unknown relation_id '{relation_id}'. Available: {available}") from exc

    @staticmethod
    def _query_indices(
        presentation_order: tuple[int, ...],
        query_nodes: tuple[int, ...],
    ) -> tuple[int, ...]:
        slot_lookup = {
            canonical_index: slot_index
            for slot_index, canonical_index in enumerate(presentation_order)
        }
        return tuple(slot_lookup[index] for index in query_nodes)


class MaskedBalancedTripletDatasetBuilder(_BaseDatasetBuilder):
    """Builds a masked 6-node ternary completion task with pairwise-balanced triples."""

    def __init__(self, config: DataConfig):
        super().__init__(config)
        relation_ids = config.relation_ids or PHASE4_DEFAULT_RELATION_IDS
        invalid = sorted(set(relation_ids) - {MASKED_BALANCED_TRIPLET_RELATION_ID})
        if invalid:
            available = ", ".join(available_relation_ids(PHASE4_TASK_SUITE))
            raise ValueError(
                f"Unknown relation_id(s) for phase 4: {', '.join(invalid)}. Available: {available}"
            )
        visible_clause_counts = tuple(
            dict.fromkeys(sorted(config.masked_visible_clause_counts))
        )
        if not visible_clause_counts:
            raise ValueError("Phase-4 masked ternary tasks require at least one visible clause count.")
        invalid_counts = [count for count in visible_clause_counts if count < 1 or count > 9]
        if invalid_counts:
            raise ValueError(
                "Phase-4 masked ternary visible clause counts must be between 1 and 9. "
                f"Received: {invalid_counts}"
            )

        self._relation_ids = tuple(relation_ids)
        self._visible_clause_counts = visible_clause_counts
        self._template_ids_by_split = {
            "train": tuple(config.train_template_ids or PHASE3_DEFAULT_TEMPLATE_IDS),
            "test": tuple(config.test_template_ids or config.train_template_ids or PHASE3_DEFAULT_TEMPLATE_IDS),
        }
        self._presentation_orders = tuple(permutations(range(6), 6))
        (
            self._positive_visible_clause_lookup,
            self._positive_signature_weights,
            self._negative_visible_clause_lookup,
        ) = self._build_visible_clause_lookups()

    def _build_split(
        self,
        split_name: str,
        names: tuple[str, ...],
        max_examples_per_label: int,
    ) -> list[PremiseQueryExample]:
        if len(names) < 6:
            raise ValueError(
                f"Phase-4 masked ternary tasks require at least 6 names in the {split_name} split. "
                f"Received {len(names)}."
            )

        worlds = list(permutations(names, 6))
        plans_by_label = self._plans_by_label(split_name=split_name)
        strata = sorted(
            set(plans_by_label[0])
            | set(plans_by_label[1])
        )
        targets = self._per_bucket_targets(
            total_examples=max_examples_per_label,
            buckets=strata,
        )

        sampled: list[PremiseQueryExample] = []
        for bucket_index, bucket in enumerate(strata):
            target_size = targets[bucket]
            if target_size == 0:
                continue
            for label in (0, 1):
                plan_lookup = plans_by_label[label]
                plans = plan_lookup.get(bucket)
                if not plans:
                    raise ValueError(
                        "Masked ternary bucket is missing for one label; cannot maintain "
                        f"query-index balance. Missing label={label}, bucket={bucket}."
                    )
                if len(plans) < target_size:
                    raise ValueError(
                        "Not enough masked ternary plans for balanced sampling: "
                        f"label={label}, bucket={bucket}, needed {target_size}, found {len(plans)}."
                    )
                sampled.extend(
                    self._materialize_examples(
                        split_name=split_name,
                        label=label,
                        bucket_index=bucket_index,
                        plans=plans,
                        worlds=worlds,
                        target_size=target_size,
                    )
                )

        self._random.shuffle(sampled)
        return sampled

    def _plans_by_label(
        self,
        split_name: str,
    ) -> dict[int, dict[tuple[str, str, tuple[int, int, int], int], list[_MaskedBalancedTripletQueryPlan]]]:
        selected_templates = set(self._template_ids_by_split[split_name])
        variants = [
            variant for variant in BALANCED_TRIPLET_TEMPLATE_VARIANTS if variant.template_id in selected_templates
        ]
        if not variants:
            available = ", ".join(sorted(PHASE3_TEMPLATE_FAMILY_MAP))
            requested = ", ".join(sorted(selected_templates))
            raise ValueError(
                f"No usable templates for phase-4 split '{split_name}'. "
                f"Requested templates: {requested}. Available: {available}"
            )

        plans_by_label: dict[
            int,
            dict[tuple[str, str, tuple[int, int, int], int], list[_MaskedBalancedTripletQueryPlan]],
        ] = {0: defaultdict(list), 1: defaultdict(list)}

        for relation_id in self._relation_ids:
            for variant in variants:
                for query_nodes in ordered_query_tuples(6, 3):
                    label = int(frozenset(query_nodes) in BALANCED_TRIPLET_POSITIVE_SET)
                    query_direction_type = "hidden_positive" if label else "matched_negative"
                    for visible_clause_count in self._visible_clause_counts:
                        for presentation_order in self._presentation_orders:
                            query_indices = self._query_indices(
                                presentation_order=presentation_order,
                                query_nodes=query_nodes,
                            )
                            bucket = (
                                relation_id,
                                variant.template_id,
                                query_indices,
                                visible_clause_count,
                            )
                            query_distance_type = f"visible_{visible_clause_count}"
                            query_type = f"{query_direction_type}_{query_distance_type}"
                            plans_by_label[label][bucket].append(
                                _MaskedBalancedTripletQueryPlan(
                                    relation_id=relation_id,
                                    template_id=variant.template_id,
                                    template_family_id=variant.template_family_id,
                                    query_nodes=query_nodes,
                                    label=label,
                                    query_indices=query_indices,
                                    query_distance_type=query_distance_type,
                                    query_direction_type=query_direction_type,
                                    query_type=query_type,
                                    presentation_order=presentation_order,
                                    visible_clause_count=visible_clause_count,
                                )
                            )
        return plans_by_label

    def _materialize_examples(
        self,
        split_name: str,
        label: int,
        bucket_index: int,
        plans: list[_MaskedBalancedTripletQueryPlan],
        worlds: list[tuple[str, ...]],
        target_size: int,
    ) -> list[PremiseQueryExample]:
        selected_worlds = self._sample_worlds(worlds=worlds, target_size=target_size)
        selected_plans = self._sample_plans(plans=plans, target_size=target_size)

        examples: list[PremiseQueryExample] = []
        for sample_index, (entities, plan) in enumerate(zip(selected_worlds, selected_plans, strict=True)):
            premise_text = self._render_premise(
                entities=entities,
                template_id=plan.template_id,
                query_nodes=plan.query_nodes,
                label=plan.label,
                visible_clause_count=plan.visible_clause_count,
            )
            entity_mentions = self._find_mentions(text=premise_text, entities=entities)
            query_entities = tuple(entities[index] for index in plan.query_nodes)
            presentation_tag = "".join(str(index) for index in plan.presentation_order)
            example_id = (
                f"{split_name}-{self.config.task_suite}-{plan.relation_id}-{plan.template_id}-"
                f"{label}-{bucket_index}-{sample_index}-vc{plan.visible_clause_count}-"
                f"{presentation_tag}-{'-'.join(entities)}"
            )
            examples.append(
                PremiseQueryExample(
                    example_id=example_id,
                    split_name=split_name,
                    task_suite=self.config.task_suite,
                    premise_text=premise_text,
                    label=plan.label,
                    relation_id=plan.relation_id,
                    entities=entities,
                    entity_mentions=entity_mentions,
                    template_id=plan.template_id,
                    template_family_id=plan.template_family_id,
                    query_entities=query_entities,
                    query_indices=plan.query_indices,
                    query_distance_type=plan.query_distance_type,
                    query_direction_type=plan.query_direction_type,
                    query_type=plan.query_type,
                    presentation_order=plan.presentation_order,
                )
            )
        return examples

    def _render_premise(
        self,
        entities: tuple[str, ...],
        template_id: str,
        query_nodes: tuple[int, int, int],
        label: int,
        visible_clause_count: int,
    ) -> str:
        variant = self._variant(template_id)
        visible_clause_indices = self._sample_visible_clause_indices(
            query_nodes=query_nodes,
            label=label,
            visible_clause_count=visible_clause_count,
        )
        ordered_visible = [index for index in variant.clause_order if index in visible_clause_indices]
        clauses = []
        for clause_index in ordered_visible:
            triple = BALANCED_TRIPLET_POSITIVE_TRIPLES[clause_index]
            a, b, c = (entities[node_index] for node_index in triple)
            clauses.append(variant.clause_template.format(a=a, b=b, c=c))
        return " ".join(clauses)

    def _build_visible_clause_lookups(
        self,
    ) -> tuple[
        dict[tuple[int, int], tuple[tuple[int, ...], ...]],
        dict[int, dict[tuple[int, int, int], int]],
        dict[tuple[frozenset[int], int], dict[tuple[int, int, int], tuple[tuple[int, ...], ...]]],
    ]:
        positive_lookup: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
        positive_signature_weights: dict[int, dict[tuple[int, int, int], int]] = defaultdict(dict)
        negative_lookup: dict[
            tuple[frozenset[int], int],
            dict[tuple[int, int, int], tuple[tuple[int, ...], ...]],
        ] = {}
        all_clause_indices = tuple(range(len(BALANCED_TRIPLET_POSITIVE_TRIPLES)))
        for visible_clause_count in self._visible_clause_counts:
            all_subsets = tuple(
                subset
                for subset in combinations(all_clause_indices, visible_clause_count)
                if self._covers_all_nodes(subset)
            )
            if not all_subsets:
                raise ValueError(
                    "No clause subsets cover all six entities for masked ternary tasks "
                    f"with visible_clause_count={visible_clause_count}."
                )
            for hidden_clause_index in all_clause_indices:
                allowed_indices = tuple(
                    clause_index
                    for clause_index in all_clause_indices
                    if clause_index != hidden_clause_index
                )
                positive_subsets = tuple(
                    subset
                    for subset in combinations(allowed_indices, visible_clause_count)
                    if self._covers_all_nodes(subset)
                )
                if not positive_subsets:
                    raise ValueError(
                        "No clause subsets cover all six entities for masked ternary positives "
                        f"with hidden_clause_index={hidden_clause_index} and "
                        f"visible_clause_count={visible_clause_count}."
                    )
                positive_lookup[(hidden_clause_index, visible_clause_count)] = positive_subsets
                hidden_query = BALANCED_TRIPLET_POSITIVE_TRIPLES[hidden_clause_index]
                for subset in positive_subsets:
                    signature = self._query_pair_signature(query_nodes=hidden_query, clause_subset=subset)
                    positive_signature_weights[visible_clause_count][signature] = (
                        positive_signature_weights[visible_clause_count].get(signature, 0) + 1
                    )

            for negative_query in combinations(range(6), 3):
                if frozenset(negative_query) in BALANCED_TRIPLET_POSITIVE_SET:
                    continue
                signature_buckets: dict[tuple[int, int, int], list[tuple[int, ...]]] = defaultdict(list)
                for subset in all_subsets:
                    signature = self._query_pair_signature(query_nodes=negative_query, clause_subset=subset)
                    if max(signature) > 1:
                        continue
                    if signature not in positive_signature_weights[visible_clause_count]:
                        continue
                    signature_buckets[signature].append(subset)
                if not signature_buckets:
                    raise ValueError(
                        "No pairwise-matched masked-negative subsets found for "
                        f"query={negative_query} and visible_clause_count={visible_clause_count}."
                    )
                negative_lookup[(frozenset(negative_query), visible_clause_count)] = {
                    signature: tuple(subsets)
                    for signature, subsets in signature_buckets.items()
                }
        return positive_lookup, positive_signature_weights, negative_lookup

    def _covers_all_nodes(self, clause_subset: tuple[int, ...]) -> bool:
        covered_nodes = {
            node_index
            for clause_index in clause_subset
            for node_index in BALANCED_TRIPLET_POSITIVE_TRIPLES[clause_index]
        }
        return len(covered_nodes) == 6

    def _query_pair_signature(
        self,
        query_nodes: tuple[int, int, int],
        clause_subset: tuple[int, ...],
    ) -> tuple[int, int, int]:
        pair_counts: list[int] = []
        for left_index, right_index in combinations(sorted(query_nodes), 2):
            count = 0
            for clause_index in clause_subset:
                triple = BALANCED_TRIPLET_POSITIVE_TRIPLES[clause_index]
                if left_index in triple and right_index in triple:
                    count += 1
            pair_counts.append(count)
        return tuple(pair_counts)

    def _sample_visible_clause_indices(
        self,
        query_nodes: tuple[int, int, int],
        label: int,
        visible_clause_count: int,
    ) -> tuple[int, ...]:
        if label == 1:
            hidden_clause_index = BALANCED_TRIPLET_POSITIVE_INDEX[frozenset(query_nodes)]
            options = self._positive_visible_clause_lookup[(hidden_clause_index, visible_clause_count)]
            return options[self._random.randrange(len(options))]

        signature_weights = self._positive_signature_weights[visible_clause_count]
        available_signatures = self._negative_visible_clause_lookup[
            (frozenset(query_nodes), visible_clause_count)
        ]
        signatures = tuple(sorted(available_signatures))
        weights = [signature_weights[signature] for signature in signatures]
        signature = self._random.choices(signatures, weights=weights, k=1)[0]
        options = available_signatures[signature]
        return options[self._random.randrange(len(options))]

    def _variant(self, template_id: str) -> BalancedTripletTemplateVariant:
        for variant in BALANCED_TRIPLET_TEMPLATE_VARIANTS:
            if variant.template_id == template_id:
                return variant
        available = ", ".join(sorted(PHASE3_TEMPLATE_FAMILY_MAP))
        raise ValueError(f"Unknown phase-4 template '{template_id}'. Available: {available}")

    def _sample_worlds(
        self,
        worlds: list[tuple[str, ...]],
        target_size: int,
    ) -> list[tuple[str, ...]]:
        if target_size <= len(worlds):
            return self._random.sample(worlds, k=target_size)
        return [worlds[self._random.randrange(len(worlds))] for _ in range(target_size)]

    def _sample_plans(
        self,
        plans: list[_MaskedBalancedTripletQueryPlan],
        target_size: int,
    ) -> list[_MaskedBalancedTripletQueryPlan]:
        if target_size <= len(plans):
            return self._random.sample(plans, k=target_size)
        return [plans[self._random.randrange(len(plans))] for _ in range(target_size)]

    @staticmethod
    def _query_indices(
        presentation_order: tuple[int, ...],
        query_nodes: tuple[int, ...],
    ) -> tuple[int, ...]:
        slot_lookup = {
            canonical_index: slot_index
            for slot_index, canonical_index in enumerate(presentation_order)
        }
        return tuple(slot_lookup[index] for index in query_nodes)


def _shortest_path_distance(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    source_index: int,
    target_index: int,
) -> int:
    if source_index == target_index:
        return 0
    adjacency: dict[int, tuple[int, ...]] = defaultdict(tuple)
    neighbors: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    adjacency = {node: tuple(neighbors[node]) for node in range(node_count)}

    frontier: deque[tuple[int, int]] = deque([(source_index, 0)])
    visited = {source_index}
    while frontier:
        node, depth = frontier.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor == target_index:
                return depth + 1
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append((neighbor, depth + 1))
    raise ValueError(
        f"Nodes {source_index} and {target_index} are disconnected in support graph {edges}."
    )


@dataclass(frozen=True)
class _BalancedTripletQueryPlan:
    relation_id: str
    template_id: str
    template_family_id: str
    query_nodes: tuple[int, int, int]
    label: int
    query_indices: tuple[int, int, int]
    query_distance_type: str
    query_direction_type: str
    query_type: str
    presentation_order: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class _MaskedBalancedTripletQueryPlan:
    relation_id: str
    template_id: str
    template_family_id: str
    query_nodes: tuple[int, int, int]
    label: int
    query_indices: tuple[int, int, int]
    query_distance_type: str
    query_direction_type: str
    query_type: str
    presentation_order: tuple[int, int, int, int, int, int]
    visible_clause_count: int


class BalancedTripletDatasetBuilder(_BaseDatasetBuilder):
    """Builds a 6-node ternary task with pairwise-balanced positive and negative triples."""

    def __init__(self, config: DataConfig):
        super().__init__(config)
        relation_ids = config.relation_ids or PHASE3_DEFAULT_RELATION_IDS
        invalid = sorted(set(relation_ids) - {BALANCED_TRIPLET_RELATION_ID})
        if invalid:
            available = ", ".join(available_relation_ids(PHASE3_TASK_SUITE))
            raise ValueError(
                f"Unknown relation_id(s) for phase 3: {', '.join(invalid)}. Available: {available}"
            )
        self._relation_ids = tuple(relation_ids)
        self._template_ids_by_split = {
            "train": tuple(config.train_template_ids or PHASE3_DEFAULT_TEMPLATE_IDS),
            "test": tuple(config.test_template_ids or config.train_template_ids or PHASE3_DEFAULT_TEMPLATE_IDS),
        }
        self._presentation_orders = tuple(permutations(range(6), 6))

    def _build_split(
        self,
        split_name: str,
        names: tuple[str, ...],
        max_examples_per_label: int,
    ) -> list[PremiseQueryExample]:
        if len(names) < 6:
            raise ValueError(
                f"Phase-3 balanced ternary tasks require at least 6 names in the {split_name} split. "
                f"Received {len(names)}."
            )

        worlds = list(permutations(names, 6))
        plans_by_label = self._plans_by_label(split_name=split_name)

        sampled: list[PremiseQueryExample] = []
        for label, plan_lookup in plans_by_label.items():
            strata = sorted(plan_lookup)
            targets = self._per_bucket_targets(total_examples=max_examples_per_label, buckets=strata)
            for bucket_index, bucket in enumerate(strata):
                target_size = targets[bucket]
                if target_size == 0:
                    continue
                sampled.extend(
                    self._materialize_examples(
                        split_name=split_name,
                        label=label,
                        bucket_index=bucket_index,
                        bucket=bucket,
                        plans=plan_lookup[bucket],
                        worlds=worlds,
                        target_size=target_size,
                    )
                )

        self._random.shuffle(sampled)
        return sampled

    def _plans_by_label(
        self,
        split_name: str,
    ) -> dict[int, dict[tuple[str, str, tuple[int, int, int]], list[_BalancedTripletQueryPlan]]]:
        selected_templates = set(self._template_ids_by_split[split_name])
        variants = [
            variant for variant in BALANCED_TRIPLET_TEMPLATE_VARIANTS if variant.template_id in selected_templates
        ]
        if not variants:
            available = ", ".join(sorted(PHASE3_TEMPLATE_FAMILY_MAP))
            requested = ", ".join(sorted(selected_templates))
            raise ValueError(
                f"No usable templates for phase-3 split '{split_name}'. "
                f"Requested templates: {requested}. Available: {available}"
            )

        plans_by_label: dict[
            int,
            dict[tuple[str, str, tuple[int, int, int]], list[_BalancedTripletQueryPlan]],
        ] = {0: defaultdict(list), 1: defaultdict(list)}

        for relation_id in self._relation_ids:
            for variant in variants:
                for query_nodes in ordered_query_tuples(6, 3):
                    label = int(frozenset(query_nodes) in BALANCED_TRIPLET_POSITIVE_SET)
                    for presentation_order in self._presentation_orders:
                        query_indices = self._query_indices(
                            presentation_order=presentation_order,
                            query_nodes=query_nodes,
                        )
                        bucket = (relation_id, variant.template_id, query_indices)
                        plans_by_label[label][bucket].append(
                            _BalancedTripletQueryPlan(
                                relation_id=relation_id,
                                template_id=variant.template_id,
                                template_family_id=variant.template_family_id,
                                query_nodes=query_nodes,
                                label=label,
                                query_indices=query_indices,
                                query_distance_type="balanced_triple",
                                query_direction_type="ordered",
                                query_type="balanced_triple_ordered",
                                presentation_order=presentation_order,
                            )
                        )
        return plans_by_label

    def _materialize_examples(
        self,
        split_name: str,
        label: int,
        bucket_index: int,
        bucket: tuple[str, str, tuple[int, int, int]],
        plans: list[_BalancedTripletQueryPlan],
        worlds: list[tuple[str, ...]],
        target_size: int,
    ) -> list[PremiseQueryExample]:
        selected_worlds = self._sample_worlds(worlds=worlds, target_size=target_size)
        selected_plans = self._sample_plans(plans=plans, target_size=target_size)

        examples: list[PremiseQueryExample] = []
        for sample_index, (entities, plan) in enumerate(zip(selected_worlds, selected_plans, strict=True)):
            premise_text = self._render_premise(entities=entities, template_id=plan.template_id)
            entity_mentions = self._find_mentions(text=premise_text, entities=entities)
            query_entities = tuple(entities[index] for index in plan.query_nodes)
            presentation_tag = "".join(str(index) for index in plan.presentation_order)
            example_id = (
                f"{split_name}-{self.config.task_suite}-{plan.relation_id}-{plan.template_id}-"
                f"{label}-{bucket_index}-{sample_index}-{presentation_tag}-{'-'.join(entities)}"
            )
            examples.append(
                PremiseQueryExample(
                    example_id=example_id,
                    split_name=split_name,
                    task_suite=self.config.task_suite,
                    premise_text=premise_text,
                    label=plan.label,
                    relation_id=plan.relation_id,
                    entities=entities,
                    entity_mentions=entity_mentions,
                    template_id=plan.template_id,
                    template_family_id=plan.template_family_id,
                    query_entities=query_entities,
                    query_indices=plan.query_indices,
                    query_distance_type=plan.query_distance_type,
                    query_direction_type=plan.query_direction_type,
                    query_type=plan.query_type,
                    presentation_order=plan.presentation_order,
                )
            )
        return examples

    def _render_premise(self, entities: tuple[str, ...], template_id: str) -> str:
        variant = self._variant(template_id)
        clauses = []
        for clause_index in variant.clause_order:
            triple = BALANCED_TRIPLET_POSITIVE_TRIPLES[clause_index]
            a, b, c = (entities[node_index] for node_index in triple)
            clauses.append(variant.clause_template.format(a=a, b=b, c=c))
        return " ".join(clauses)

    def _variant(self, template_id: str) -> BalancedTripletTemplateVariant:
        for variant in BALANCED_TRIPLET_TEMPLATE_VARIANTS:
            if variant.template_id == template_id:
                return variant
        available = ", ".join(sorted(PHASE3_TEMPLATE_FAMILY_MAP))
        raise ValueError(f"Unknown phase-3 template '{template_id}'. Available: {available}")

    def _sample_worlds(
        self,
        worlds: list[tuple[str, ...]],
        target_size: int,
    ) -> list[tuple[str, ...]]:
        if target_size <= len(worlds):
            return self._random.sample(worlds, k=target_size)
        return [worlds[self._random.randrange(len(worlds))] for _ in range(target_size)]

    def _sample_plans(
        self,
        plans: list[_BalancedTripletQueryPlan],
        target_size: int,
    ) -> list[_BalancedTripletQueryPlan]:
        if target_size <= len(plans):
            return self._random.sample(plans, k=target_size)
        return [plans[self._random.randrange(len(plans))] for _ in range(target_size)]

    @staticmethod
    def _query_indices(
        presentation_order: tuple[int, ...],
        query_nodes: tuple[int, ...],
    ) -> tuple[int, ...]:
        slot_lookup = {
            canonical_index: slot_index
            for slot_index, canonical_index in enumerate(presentation_order)
        }
        return tuple(slot_lookup[index] for index in query_nodes)
