# Results Analysis

## TL;DR

The current evidence does **not** support the claim that genuine triadic structure is needed to explain the frozen hidden states in this phase-1 setup.

The cleanest run so far is [`results/phase1_structural`](./phase1_structural). On that split:

- `query_only` is exactly at chance, so the obvious shortcut is gone.
- `exact` is the best probe on pretrained BERT.
- `pairwise` is consistently worse than `exact`.
- `triadic` is consistently worse than both.
- pairwise curl is present, but it does not translate into better predictive performance.

The current best interpretation is:

> the transferable signal in this synthetic consistency task is more compatible with an exact / potential-style description than with a flexible pairwise or triadic probe basis.

## Research Question

The phase-1 question was:

> On controlled relational-consistency tasks, are frozen transformer hidden states better described by exact node potentials, unconstrained pairwise relations, or genuine triadic interactions?

The intended decision rule was:

- if `triadic > pairwise > exact` on held-out data, the higher-order story becomes plausible
- if `pairwise > exact` but `triadic ~= pairwise`, then non-exact pairwise structure matters, but true triadic structure is not earning its keep
- if `exact >= pairwise >= triadic`, then the more robust signal is gradient-like / integrable rather than genuinely higher-order

## Experimental Progression

### 1. `phase1_hard`

Reference: [`results/phase1_hard`](./phase1_hard)

This was the first useful run after removing the conclusion-reading shortcut. It showed:

- pretrained `exact`: `0.8050`
- pretrained `pairwise`: `0.8392`
- pretrained `triadic`: `0.8349`

At first glance this looked mildly favorable to `pairwise`, but the random control was also very strong:

- random `exact`: `0.6421`
- random `pairwise`: `0.7372`
- random `triadic`: `0.7366`

Interpretation:

- useful as a pipeline validation
- not trustworthy as evidence for higher-order structure
- still too easy / too shortcut-friendly

### 2. `phase1_multiseed`

Reference: [`results/phase1_multiseed`](./phase1_multiseed)

This was the strict forward-vs-reverse template holdout. It collapsed:

- pretrained `exact`: `0.0213`
- pretrained `pairwise`: `0.0214`
- pretrained `triadic`: `0.0169`

Interpretation:

- not a semantics result
- a polarity/sign-flip pathology
- useful because it exposed how brittle the split design was

### 3. `phase1_balanced`

Reference: [`results/phase1_balanced`](./phase1_balanced)

This improved the holdout, but still behaved like a softer polarity-transfer test:

- pretrained `exact`: raw `0.3457`, polarity-invariant `0.6543`
- pretrained `pairwise`: raw `0.2733`, polarity-invariant `0.7267`
- pretrained `triadic`: raw `0.2141`, polarity-invariant `0.7859`

Interpretation:

- the split was still confounded
- flexible probes were learning internally consistent but directionally wrong rules
- still not evidence for a triadic advantage

### 4. `phase1_factorized`

Reference: [`results/phase1_factorized`](./phase1_factorized)

This run was only partially completed originally, but the completed seeds were enough to diagnose the issue:

- pretrained `exact`: raw `0.5110`, polarity-invariant `0.5534`
- pretrained `pairwise`: raw `0.4452`, polarity-invariant `0.5596`
- pretrained `triadic`: raw `0.3604`, polarity-invariant `0.6396`

Interpretation:

- much cleaner than the earlier balanced split
- but the paraphrase-mixed inverse families were still unstable
- `exact` was already emerging as the most robust probe

### 5. `phase1_structural`

Reference: [`results/phase1_structural`](./phase1_structural)

This is the current best result and the one that should drive the interpretation.

## Main Result: `phase1_structural`

### Setup

- 5 seeds: `7, 11, 13, 17, 19`
- 2 relation families: `outranks`, `older_than`
- 1200 train / 144 test examples per seed
- structural holdout:
  - train: `active_forward`, `passive_reverse`, `mixed_ab_inverse`, `mixed_bc_inverse_reverse`, `above_forward`, `below_reverse`
  - test: `active_reverse`, `passive_forward`, `mixed_ab_inverse_reverse`, `mixed_bc_inverse`, `above_reverse`, `below_forward`

This split keeps direct and inverse realizations on both sides while avoiding the paraphrase-mixed inverse families that were still causing polarity artifacts.

### Aggregate Probe Results

Reference: [`results/phase1_structural/probe_accuracy_aggregate.csv`](./phase1_structural/probe_accuracy_aggregate.csv)

| Variant | Probe | Avg Test | Best Layer Test | Avg Polarity-Invariant |
| --- | --- | ---: | ---: | ---: |
| pretrained | query_only | 0.5000 | 0.5000 | 0.5000 |
| pretrained | exact | 0.6062 | 0.6542 | 0.6152 |
| pretrained | pairwise | 0.5708 | 0.6000 | 0.5772 |
| pretrained | triadic | 0.4924 | 0.5361 | 0.5362 |
| random_control | query_only | 0.5000 | 0.5000 | 0.5000 |
| random_control | exact | 0.4679 | 0.4847 | 0.5385 |
| random_control | pairwise | 0.4126 | 0.4528 | 0.5874 |
| random_control | triadic | 0.4118 | 0.4431 | 0.5887 |

Key facts:

- `query_only` stays at chance.
- `exact` beats `pairwise` and `triadic` in the pretrained model.
- `triadic` is not just “not better”; it is clearly worse.
- the same ranking appears in the random control.

### Layerwise Pattern

Across pretrained layers, `exact` is the strongest probe from layer 1 onward:

- layer 1: `exact 0.6167`, `pairwise 0.5667`, `triadic 0.4819`
- layer 4: `exact 0.6542`, `pairwise 0.6000`, `triadic 0.5361`
- layer 8: `exact 0.6528`, `pairwise 0.5861`, `triadic 0.5028`
- layer 10: `exact 0.6194`, `pairwise 0.5986`, `triadic 0.5194`

So this is not a one-layer fluke. The ranking is stable.

### Seed Stability

The seed-level results show the same pattern in every run:

| Seed | Pretrained Exact Avg | Pretrained Pairwise Avg | Pretrained Triadic Avg |
| --- | ---: | ---: | ---: |
| 7 | 0.5791 | 0.5759 | 0.4984 |
| 11 | 0.5935 | 0.5588 | 0.4781 |
| 13 | 0.6346 | 0.5983 | 0.5155 |
| 17 | 0.5935 | 0.5363 | 0.4573 |
| 19 | 0.6303 | 0.5849 | 0.5128 |

That makes the current conclusion fairly robust:

- `exact` is the most reliable probe
- `pairwise` is second-best
- `triadic` is consistently worst of the three nontrivial probes

## Diagnostics

Reference: [`results/phase1_structural/diagnostics_aggregate.csv`](./phase1_structural/diagnostics_aggregate.csv)

### Aggregate Diagnostic Summary

| Variant | Probe | Avg Exactness | Avg Positional Exactness | Avg Curl | Avg Position-Corrected Curl |
| --- | --- | ---: | ---: | ---: | ---: |
| pretrained | exact | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| pretrained | pairwise | 0.9566 | 0.9614 | 0.1302 | 0.1157 |
| random_control | exact | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| random_control | pairwise | 0.9549 | 0.9481 | 0.1354 | 0.1557 |

Interpretation:

- pairwise fields are still highly exact-like
- curl is present, but not large
- pretrained pairwise fields are slightly **more exact** and slightly **less curly** than random after positional correction

This matters because the original motivating story was roughly:

> maybe pretraining creates meaningful cyclic residue that a triadic probe can exploit

The structural run does not support that story. The diagnostic signal goes the other way:

- pretraining appears to make the useful structure **more integrable**
- the probe that enforces exactness is also the best one

## Template-Family Behavior

Reference: [`results/phase1_structural/template_group_accuracy_aggregate.csv`](./phase1_structural/template_group_accuracy_aggregate.csv)

For pretrained BERT, average raw test accuracy by template family is:

| Family | Exact | Pairwise | Triadic |
| --- | ---: | ---: | ---: |
| `above` | 0.6538 | 0.6808 | 0.6410 |
| `active` | 0.6199 | 0.6622 | 0.6109 |
| `below` | 0.5378 | 0.5000 | 0.4942 |
| `mixed_ab_inverse` | 0.5987 | 0.5071 | 0.3006 |
| `mixed_bc_inverse` | 0.5750 | 0.4923 | 0.3615 |
| `passive` | 0.6519 | 0.5827 | 0.5462 |

This is useful because it shows where the failures are:

- `exact` is comparatively steady across all families
- `pairwise` degrades noticeably on the mixed inverse families
- `triadic` collapses hardest on the mixed inverse families

So the extra flexibility is not buying real compositional robustness. It is mostly buying brittleness.

## What Counts As A Real Result Here

If the higher-order story were winning, we would expect something like:

- `triadic > pairwise > exact`
- stable across seeds
- especially on the harder structural families
- with nontrivial curl in the same layers where `triadic` helps

What we actually see is:

- `exact > pairwise > triadic`
- stable across seeds
- especially clear on the mixed inverse families
- with modest curl that does not produce predictive gains

That is a meaningful negative result.

## Current Conclusion

The best-supported conclusion at this point is:

1. the shortcut-removal and holdout redesign work was necessary and successful
2. once the split is clean enough, the evidence does **not** favor a triadic probe basis
3. the most transferable signal in this synthetic setting looks closer to an exact / potential model
4. nonclosure exists diagnostically, but it is not behaving like useful higher-order predictive structure

In short:

> the current experiments argue against the grander higher-order claim, and in favor of a more conservative story where the robust signal is largely integrable.

## Recommended Next Steps

If the project continues, the next step should **not** be a larger encoder.

The better next moves are:

- relation-family breakdowns beyond the current two relations
- harder lexical perturbations within the structural holdout
- lower-capacity pairwise controls to confirm the current gap is about robustness rather than just overfitting
- natural controlled contrasts only after the synthetic result is fully stable

If a future run ever shows `triadic > pairwise > exact` under a comparably clean holdout, that would be worth taking seriously. Right now, it has not happened.
