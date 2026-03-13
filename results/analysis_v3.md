# Results Analysis V3

## Executive Summary

This file updates [analysis_v2.md](./analysis_v2.md) after the fair-control reruns and the canonical 5-seed `bert-large-uncased` bundle.

The current roadmap in [nextsteps.md](../nextsteps.md) is now effectively complete:

1. tighten the masked phase-4 setup
2. add raw hidden-state diagnostics
3. add fair residual controls
4. rerun the validated phase-4 task at larger scale

The resulting claim is stronger and cleaner than the one in V2:

> On order-like tasks, frozen BERT-family hidden states are well described by low-rank exact / latent-potential probes. On harder non-order tasks, pairwise probes beat exact probes. On masked ternary completion with held-out ternary paraphrase families, multiplicative higher-order probes beat pairwise-only probes, and they do so by more than can be explained by query context alone or by a generic matched-capacity triplet MLP.

The strongest practical headline is now:

> `pairwise_plus_triadic` is the safest winning probe family on the best current task, and the effect survives scale-up to `bert-large-uncased`.

That is the first point in the repo where the higher-order story is both predictive and defensible.

## What Is Now Settled

### Phase 1: 3-node order tasks

Reference:
- [phase1_structural](./phase1_structural)
- [phase1_structural_rank_sweep](./phase1_structural_rank_sweep)

Result:
- low-rank `exact` probes are best
- `pairwise` is not needed for the transferable part of the signal
- `triadic` is clearly not needed

Interpretation:
- on strict order-like relational tasks, frozen BERT hidden states look mostly integrable / latent-potential-like

This remains a real negative result for the strong higher-order claim on that task family.

### Phase 2: 4-node non-order tasks

Reference:
- [phase2_four_node](./phase2_four_node)

Result:
- `exact` fails on `same_side` and `same_row`
- `pairwise` beats `exact` strongly
- `triadic` does not beat `pairwise` robustly

Interpretation:
- exact latent potentials stop being enough once the task is no longer basically order reconstruction

This established the first real `pairwise > exact` regime in the repo.

### Phase 3: balanced ternary hyperedge tasks

Reference:
- [phase3_balanced_triplet](./phase3_balanced_triplet)
- [phase3_structural](./phase3_structural)

Result:
- same-template balanced ternary gave a strong `triadic > pairwise > exact` result
- ternary-family structural holdout weakened that result substantially

Interpretation:
- the project did find a real triadic regime
- but the first version of that win was still too template-bound

This phase justified building the harder masked completion setup.

## Phase 4: The Main Result

### The Base Fair-Control Run

Reference:
- [phase4_controls_base](./phase4_controls_base)
- [phase4_controls_base/probe_accuracy_aggregate.csv](./phase4_controls_base/probe_accuracy_aggregate.csv)
- [phase4_controls_base/diagnostics_aggregate.csv](./phase4_controls_base/diagnostics_aggregate.csv)

This run answered the main ablation question that remained open after V2.

Layer-averaged pretrained test accuracy:

| Probe | Avg Test |
| --- | ---: |
| best exact (`exact_r8`) | `0.5022` |
| `pairwise` | `0.5217` |
| `pairwise_plus_query_context` | `0.5177` |
| `pairwise_plus_triplet_mlp` | `0.5250` |
| `pairwise_plus_triadic` | `0.5457` |
| `triadic` | `0.5473` |

Layer-averaged random-control test accuracy:

| Probe | Avg Test |
| --- | ---: |
| best exact (`exact_r2`) | `0.5063` |
| `pairwise` | `0.5047` |
| `pairwise_plus_query_context` | `0.5042` |
| `pairwise_plus_triplet_mlp` | `0.5004` |
| `pairwise_plus_triadic` | `0.5065` |
| `triadic` | `0.5123` |

The key interpretation is:

- exact is dead on this task
- pairwise helps a little
- query context alone does not explain the gain
- a matched-capacity tuple MLP explains only a little
- the multiplicative higher-order probes are the only ones with a real lift over pairwise

The relevant gaps are:

| Gap | Pretrained | Random |
| --- | ---: | ---: |
| `pairwise_plus_query_context - pairwise` | `-0.0040` | `-0.0005` |
| `pairwise_plus_triplet_mlp - pairwise_plus_query_context` | `+0.0073` | `-0.0037` |
| `pairwise_plus_triadic - pairwise` | `+0.0240` | `+0.0018` |
| `triadic - pairwise` | `+0.0256` | `+0.0076` |

That was already a good result. The next question was whether it would survive scale-up.

### The Canonical `bert-large-uncased` Bundle

Reference:
- [phase4_controls_bert_large_full](./phase4_controls_bert_large_full)
- [phase4_controls_bert_large_full/probe_accuracy_aggregate.csv](./phase4_controls_bert_large_full/probe_accuracy_aggregate.csv)
- [phase4_controls_bert_large_full/diagnostics_aggregate.csv](./phase4_controls_bert_large_full/diagnostics_aggregate.csv)

This is the current best result in the repo.

Layer-averaged pretrained test accuracy:

| Probe | Avg Test |
| --- | ---: |
| best exact (`exact_r64`) | `0.5103` |
| `pairwise` | `0.5302` |
| `pairwise_plus_query_context` | `0.5351` |
| `pairwise_plus_triplet_mlp` | `0.5329` |
| `pairwise_plus_triadic` | `0.5706` |
| `triadic` | `0.5681` |

Layer-averaged random-control test accuracy:

| Probe | Avg Test |
| --- | ---: |
| best exact (`exact_r2`) | `0.4953` |
| `pairwise` | `0.4920` |
| `pairwise_plus_query_context` | `0.4931` |
| `pairwise_plus_triplet_mlp` | `0.4951` |
| `pairwise_plus_triadic` | `0.4931` |
| `triadic` | `0.4986` |

This run gives the cleanest version of the claim:

- exact is still dead
- pairwise still helps a little
- query context alone is not enough
- generic triplet MLP capacity is not enough
- a multiplicative higher-order residual helps substantially more

The aggregate gaps are:

| Gap | Pretrained | Random |
| --- | ---: | ---: |
| `pairwise_plus_query_context - pairwise` | `+0.0049` | `+0.0011` |
| `pairwise_plus_triplet_mlp - pairwise_plus_query_context` | `-0.0022` | `+0.0020` |
| `pairwise_plus_triadic - pairwise` | `+0.0404` | `+0.0011` |
| `pairwise_plus_triadic - pairwise_plus_triplet_mlp` | `+0.0377` | `-0.0020` |
| `triadic - pairwise` | `+0.0378` | `+0.0066` |

That is the first result in the repo where the multiplicative control is clearly doing something that context-only and generic nonlinear controls are not.

### Seed Stability

Reference:
- [phase4_controls_bert_large_full/probe_accuracy_all_runs.csv](./phase4_controls_bert_large_full/probe_accuracy_all_runs.csv)

For pretrained `bert-large`:

- `pairwise_plus_triadic > pairwise` in all 5 seeds
- `triadic > pairwise` in all 5 seeds

Per-seed pretrained `pairwise_plus_triadic - pairwise`:

| Seed | Gap |
| --- | ---: |
| `7` | `+0.0261` |
| `11` | `+0.0325` |
| `13` | `+0.0433` |
| `17` | `+0.0494` |
| `19` | `+0.0506` |

Per-seed pretrained `triadic - pairwise`:

| Seed | Gap |
| --- | ---: |
| `7` | `+0.0144` |
| `11` | `+0.0383` |
| `13` | `+0.0456` |
| `17` | `+0.0411` |
| `19` | `+0.0497` |

On the random-control side, the same differences are much smaller and mixed in sign. That is what we wanted to see.

## Raw Geometry: Now Strongly Supportive

Reference:
- [phase4_controls_bert_large_full/diagnostics_aggregate.csv](./phase4_controls_bert_large_full/diagnostics_aggregate.csv)

The raw hidden-state diagnostic remains the most important probe-free check.

| Variant | Source | Positional Exactness | Positional Curl |
| --- | --- | ---: | ---: |
| pretrained | `raw_projection_r4` | `1.0000` | `0.0022` |
| pretrained | `raw_skew_bilinear_r8` | `0.6466` | `2.1258` |
| random_control | `raw_projection_r4` | `1.0000` | `0.0043` |
| random_control | `raw_skew_bilinear_r8` | `0.9185` | `0.4945` |

Interpretation:

- the exact-by-construction projection sanity baseline behaves exactly as it should
- the skew-bilinear raw field is much less exact and much curlier in pretrained than in random control

That matters because it means the higher-order story is no longer resting only on learned probes. The representation itself shows a large pretrained-specific non-exact signature.

The probe-mediated pairwise diagnostics tell the same story:

| Variant | Probe | Positional Exactness | Positional Curl |
| --- | --- | ---: | ---: |
| pretrained | `pairwise` | `0.8500` | `0.9330` |
| pretrained | `pairwise_plus_triadic` | `0.8202` | `1.1147` |
| random_control | `pairwise` | `0.9632` | `0.2290` |
| random_control | `pairwise_plus_triadic` | `0.9305` | `0.3924` |

So both the raw geometry and the learned edge fields now move in the same direction.

## What The Large-Model Run Clarified

The canonical large-model bundle answers three questions that were still open in V2.

### 1. Does scale-up strengthen the result?

Yes.

Compared with the base fair-control run:

| Probe | `bert-base` | `bert-large` | Delta |
| --- | ---: | ---: | ---: |
| `pairwise` | `0.5217` | `0.5302` | `+0.0085` |
| `pairwise_plus_triadic` | `0.5457` | `0.5706` | `+0.0249` |
| `triadic` | `0.5473` | `0.5681` | `+0.0208` |

Scaling helped the higher-order probes more than it helped plain pairwise.

### 2. Is the gain just extra context?

No.

`pairwise_plus_query_context` is only `0.5351`, almost on top of `pairwise` at `0.5302`. The multiplicative residual is what makes the real jump.

### 3. Is the gain just extra tuple-level capacity?

No.

`pairwise_plus_triplet_mlp` is only `0.5329`, also near `pairwise`. The multiplicative residual still wins by `+0.0377` over the matched-capacity tuple MLP control.

## Remaining Caveats

This is the strongest result in the repo, but it still has limits.

### 1. The effect is narrow

The current positive result is on:

- masked ternary completion
- held-out ternary paraphrase families
- a specific balanced hypergraph construction

That is enough for a real claim, but it is not a universal claim about language representations.

### 2. Query-type asymmetry remains

Reference:
- [phase4_controls_bert_large_full/template_group_accuracy_aggregate.csv](./phase4_controls_bert_large_full/template_group_accuracy_aggregate.csv)

The higher-order advantage is still stronger on hidden-positive cases than on some matched-negative cases, even though the negatives are much improved relative to earlier runs.

For pretrained `bert-large`:

- `pairwise` hidden positives are around `0.58` to `0.61`
- `pairwise_plus_triadic` hidden positives are around `0.61` to `0.64`
- `pairwise` matched negatives fall to about `0.42` to `0.50`
- `pairwise_plus_triadic` matched negatives rise to about `0.47` to `0.54`
- `triadic` matched negatives rise further to about `0.49` to `0.55`

So the higher-order probes are more balanced than pairwise, but the task is still not completely uniform.

### 3. Attention diagnostics are still missing

The raw geometry result is already good, but the repo still does not include optional antisymmetric attention-edge diagnostics. That remains an obvious extension.

## What The Completed Roadmap Established

The plan in [nextsteps.md](../nextsteps.md) is now complete in substance.

It established:

1. a negative exact-order result in phase 1
2. a clean `pairwise > exact` result in phase 2
3. a same-template `triadic > pairwise` result in phase 3
4. a structurally harder masked ternary task in phase 4
5. a probe-free raw geometry result that supports the predictive result
6. a canonical scaled-up result where multiplicative higher-order probes beat fair controls

This is enough to say the project has crossed into a genuinely positive regime.

The right precise statement is:

> Frozen BERT-family hidden states are not uniformly higher-order. They look exact-like on order tasks, pairwise on some non-order tasks, and on masked ternary completion they support a real multiplicative higher-order advantage that survives held-out ternary paraphrase families, fair residual controls, raw geometry checks, and scale-up to `bert-large-uncased`.

## Recommended Next Moves

The next decision is no longer “is there any real effect at all?” That question is answered.

The next question is what kind of follow-up will most improve the scientific case.

### Option A: Attention-based raw diagnostics

Why:
- cheapest extension
- closest missing piece relative to the current roadmap
- would connect the representation-level story to model internals more directly

What to add:
- optional `include_attentions=True`
- antisymmetric attention-derived edge fields
- exactness/curl summaries alongside `raw_projection_r4` and `raw_skew_bilinear_r8`

### Option B: One second architecture family

Why:
- strongest external validity test
- would tell us whether the masked ternary higher-order effect is BERT-family-specific or broader

Good candidates:
- `roberta-base`
- `deberta-v3-base`

Recommendation:
- do one smaller architecture first, not a large sweep
- keep the exact same phase-4 control setup and diagnostics

### Option C: Write and package the result

Why:
- the repo now has enough evidence for a serious internal writeup or short paper draft

If choosing this option:
- push [phase4_controls_bert_large_full](./phase4_controls_bert_large_full)
- update [README.md](../README.md) to make the phase-4 control result the headline
- keep [analysis_v3.md](./analysis_v3.md) as the canonical narrative file

## Bottom Line

The last test went well.

The project now has:

- a clean negative exact result
- a clean pairwise-over-exact result
- a clean higher-order-over-pairwise result on the right task
- fair residual controls that make the result harder to explain away
- raw hidden-state geometry that supports the predictive effect
- a successful scale-up to `bert-large-uncased`

That is enough to say the current line of work is producing solid results, not just suggestive ones.
