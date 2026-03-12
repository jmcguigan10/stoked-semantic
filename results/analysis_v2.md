# Results Analysis V2

## Executive Summary

The project has now completed the roadmap in [`nextsteps.md`](../nextsteps.md):

1. tighten the masked phase-4 negatives
2. add raw hidden-state diagnostics
3. test a larger encoder on the same held-out masked ternary task
4. sanity-check the larger encoder with random control

The current evidence supports a more precise claim than the original one:

> Frozen BERT-family hidden states do not require higher-order probes on order-like tasks, do require more than exact latent potentials on non-order tasks, and on masked ternary completion with held-out ternary paraphrase families, a higher-order residual probe provides a small but real advantage over pairwise-only probes.

The strongest practical headline is now:

> `pairwise_plus_triadic` is the safest winning probe family on the best current task.

That is more defensible than saying “triadic wins” in the abstract.

## What Is Now Settled

### Phase 1: order-like 3-node tasks

Reference:
- [`results/phase1_structural`](./phase1_structural)
- [`results/phase1_structural_rank_sweep`](./phase1_structural_rank_sweep)

Result:
- low-rank exact probes are best
- pairwise is not needed for the transferable part of the signal
- triadic is clearly not needed

Interpretation:
- on strict order-like relational tasks, frozen BERT hidden states look mostly integrable / latent-potential-like

This part of the project is a real negative result for the stronger higher-order story.

### Phase 2: 4-node non-order tasks

Reference:
- [`results/phase2_four_node`](./phase2_four_node)

Result:
- exact fails on `same_side` and `same_row`
- pairwise beats exact strongly
- triadic does not beat pairwise robustly

Interpretation:
- exact latent potentials stop being enough once the task is no longer basically order reconstruction
- this established the first real `pairwise > exact` regime in the repo

### Phase 3: balanced ternary hyperedge tasks

Reference:
- [`results/phase3_balanced_triplet`](./phase3_balanced_triplet)
- [`results/phase3_structural`](./phase3_structural)

Result:
- same-template balanced ternary gave a strong `triadic > pairwise > exact` result
- ternary-family structural holdout weakened that result substantially

Interpretation:
- the repo did eventually find a regime where genuine triadic probes win
- but much of the first win was template-bound

This phase justified building the harder masked completion setup.

## Phase 4: The Current Main Result

### Reference Runs

The main base-model result is now:
- [`results/phase4_matched_negative_v3`](./phase4_matched_negative_v3)

The larger-model pilot is split across:
- pretrained-only 3-seed pilot: [`results/phase4_scaleup_bert_large_pilot`](./phase4_scaleup_bert_large_pilot)
- random-control seed 7: [`results/phase4_scaleup_bert_large_pilot_20260312_142516`](./phase4_scaleup_bert_large_pilot_20260312_142516)
- random-control seeds 11 and 13: [`results/phase4_scaleup_bert_large_pilot_20260312_143919`](./phase4_scaleup_bert_large_pilot_20260312_143919)

The split large-model random-control runs are a consequence of the new timestamped output-dir behavior, not a methodological difference.

### 1. Base Model: `phase4_matched_negative_v3`

Reference:
- [`results/phase4_matched_negative_v3/probe_accuracy_aggregate.csv`](./phase4_matched_negative_v3/probe_accuracy_aggregate.csv)
- [`results/phase4_matched_negative_v3/diagnostics_aggregate.csv`](./phase4_matched_negative_v3/diagnostics_aggregate.csv)

Layer-averaged pretrained test accuracy:

| Probe | Avg Test |
| --- | ---: |
| `query_only` | `0.5000` |
| best exact (`exact_r8`) | `0.5022` |
| `pairwise` | `0.5217` |
| `pairwise_plus_triadic` | `0.5443` |
| `triadic` | `0.5473` |

Random-control averages:

| Probe | Avg Test |
| --- | ---: |
| `query_only` | `0.5000` |
| best exact (`exact_r2`) | `0.5063` |
| `pairwise` | `0.5047` |
| `pairwise_plus_triadic` | `0.5110` |
| `triadic` | `0.5123` |

Key takeaways:

- the bias audit worked: `query_only` is exactly chance
- exact is dead on this task
- pairwise helps a bit
- higher-order probes help more
- the pretrained gain over pairwise is small but real:
  - `pairwise_plus_triadic - pairwise = +0.0226`
  - `triadic - pairwise = +0.0256`
- random control moves in the same direction but much less:
  - `pairwise_plus_triadic - pairwise = +0.0063`
  - `triadic - pairwise = +0.0076`

The strongest conservative statement here is:

> On masked ternary completion with held-out ternary-family transfer, a higher-order probe family helps beyond pairwise, but the margin is modest.

### 2. Raw Hidden-State Diagnostics

This was the major unfinished item in `nextsteps.md`, and it is now complete.

Reference:
- [`results/phase4_matched_negative_v3/diagnostics_aggregate.csv`](./phase4_matched_negative_v3/diagnostics_aggregate.csv)

Raw diagnostic comparison:

| Variant | Source | Positional Exactness | Positional Curl |
| --- | --- | ---: | ---: |
| pretrained | `raw_projection_r4` | `0.9997` | `0.0017` |
| pretrained | `raw_skew_bilinear_r8` | `0.6494` | `2.1036` |
| random_control | `raw_projection_r4` | `0.9993` | `0.0042` |
| random_control | `raw_skew_bilinear_r8` | `0.8406` | `0.9564` |

Interpretation:

- the exact-by-construction raw projection behaves exactly as a sanity check should
- the skew-bilinear raw edge field is much less exact and much curlier in pretrained than in random control

This is the first probe-free result in the repo that clearly supports the higher-order interpretation.

That matters a lot. Before this, the story depended too much on learned probes. After this, the representation itself shows a pretrained-specific non-exact signature.

### 3. Where The Base Phase-4 Gain Lives

Reference:
- [`results/phase4_matched_negative_v3/template_group_accuracy_aggregate.csv`](./phase4_matched_negative_v3/template_group_accuracy_aggregate.csv)

The gain is not uniform.

Template-family averages:

| Probe | `balanced_align` | `balanced_sync` |
| --- | ---: | ---: |
| `pairwise` | `0.5292` | `0.5140` |
| `pairwise_plus_triadic` | `0.5230` | `0.5650` |
| `triadic` | `0.5273` | `0.5665` |

Query-type averages:

| Probe | hidden positives | matched negatives |
| --- | ---: | ---: |
| `pairwise` | strong | weak |
| `pairwise_plus_triadic` | stronger | more balanced |
| `triadic` | moderate on positives | strongest on matched negatives |

So the correct interpretation is not “ternary reasoning solved.” It is:

> higher-order structure helps on masked ternary completion, especially where pairwise support is least reliable.

## Scale-Up: `bert-large-uncased`

### 1. Pretrained-Only Pilot

Reference:
- [`results/phase4_scaleup_bert_large_pilot`](./phase4_scaleup_bert_large_pilot)

3-seed pretrained averages:

| Probe | Avg Test |
| --- | ---: |
| best exact (`exact_r64`) | `0.5181` |
| `pairwise` | `0.5406` |
| `pairwise_plus_triadic` | `0.5745` |
| `triadic` | `0.5733` |

Compared with base `bert-base` phase 4:

| Probe | `bert-large` | `bert-base` | Delta |
| --- | ---: | ---: | ---: |
| `pairwise` | `0.5406` | `0.5217` | `+0.0189` |
| `pairwise_plus_triadic` | `0.5745` | `0.5443` | `+0.0302` |
| `triadic` | `0.5733` | `0.5473` | `+0.0260` |

Interpretation:

- scaling up helps
- it helps the higher-order probes more than plain pairwise
- `balanced_align`, which had been the weak family, improves as well

This is the first scale-up result in the repo that clearly moved in the right direction.

### 2. Random-Control Check For `bert-large`

Reference:
- [`results/phase4_scaleup_bert_large_pilot_20260312_142516`](./phase4_scaleup_bert_large_pilot_20260312_142516)
- [`results/phase4_scaleup_bert_large_pilot_20260312_143919`](./phase4_scaleup_bert_large_pilot_20260312_143919)

The random-control pilot is not perfectly flat, but it is still much weaker than pretrained overall.

Across the 3 checked random-control seeds:

| Probe | Approx Avg Test |
| --- | ---: |
| `pairwise` | `0.497` |
| `pairwise_plus_triadic` | `0.509` |
| `triadic` | `0.507` |

Compared with pretrained `bert-large`:

| Probe | Pretrained | Random |
| --- | ---: | ---: |
| `pairwise` | `0.5406` | `0.497` |
| `pairwise_plus_triadic` | `0.5745` | `0.509` |
| `triadic` | `0.5733` | `0.507` |

The cleanest headline is:

> `pairwise_plus_triadic` remains the safest winner on `bert-large`, and the higher-order margin is still materially larger in pretrained than in random control.

The raw geometry also stays strongly separated:

| Variant | `raw_skew_bilinear_r8` Positional Exactness | Positional Curl |
| --- | ---: | ---: |
| pretrained | about `0.648` | about `2.11` |
| random_control | about `0.91` | about `0.51` to `0.58` |

That is a strong representation-level difference, not a small classifier artifact.

## What The Completed `nextsteps.md` Plan Established

The roadmap in [`nextsteps.md`](../nextsteps.md) was:

1. fix phase-4 negatives
2. add raw hidden-state diagnostics
3. use those gates to decide whether scaling up is justified

All three steps are now complete.

The conclusion from that plan is:

- phase 4 survived the matched-negative cleanup
- the raw diagnostics strengthened the higher-order interpretation
- the first larger-encoder pilot improved the effect instead of collapsing it

So the repo is no longer in the “interesting but inconclusive” stage. It now supports a real, bounded positive claim.

## The Current Best Claim

This is the strongest version I would defend right now:

> On masked ternary completion with held-out ternary paraphrase families, frozen BERT-family hidden states support a small but repeatable higher-order advantage over pairwise-only probes, and this advantage is accompanied by a pretrained-specific non-exact signature in raw hidden-state geometry.

What I would **not** claim yet:

- that language is “fundamentally triadic”
- that standalone triadic probes are always better than residual higher-order probes
- that the result is uniform across all label types and paraphrase families
- that the effect has already been established across a broad model family

## Recommended Next Step

The next step should be disciplined rather than expansive.

### Recommendation: one clean consolidated `bert-large` run

Run a single 3-seed or 5-seed `bert-large` bundle that includes:

- pretrained
- random control
- the same phase-4 masked ternary structural holdout
- the same raw diagnostics
- one non-default output directory, so the results live in a single place instead of across timestamped pilot runs

Reason:

- right now the `bert-large` story is positive, but spread across multiple pilot directories
- the science is good enough to justify one clean archival run
- after that, you can decide whether to broaden model families or shift effort to attention-based raw diagnostics

### What I Would Do After That

If the consolidated `bert-large` run matches the current pilot:

1. archive it as the canonical large-model result
2. add optional attention-based antisymmetric raw diagnostics
3. only then consider a second architecture family

If the consolidated `bert-large` run weakens materially:

1. stop scaling
2. tighten `balanced_sync` / `balanced_align` task geometry further
3. keep the claim limited to `bert-base`

## Bottom Line

The original broad higher-order idea was too loose.

The current project is no longer that loose idea. It has become a much sharper result:

- exact wins on order-like tasks
- pairwise wins on harder non-order tasks
- higher-order residual structure wins on the hardest masked ternary task
- raw hidden-state geometry finally supports that predictive story
- a larger encoder improves the margin instead of erasing it

That is enough to justify one more serious, consolidated large-model run before changing research direction.
