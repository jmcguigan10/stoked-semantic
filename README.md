# Stoked Semantic

Phase 1 probing experiments for a narrow question:

> On controlled relational-consistency tasks, are frozen transformer hidden states better described by exact node potentials, unconstrained pairwise relations, or genuine triadic interactions?

The codebase is deliberately small. It generates synthetic premise-query datasets, extracts frozen BERT features, trains several probe families layer by layer, and reports both predictive accuracy and Hodge-style exactness/curl diagnostics.

## What This Repo Contains

- A controlled synthetic dataset with lexical holdout and configurable template holdouts
- Frozen-feature extraction from `bert-base-uncased`
- Seven probe families:
  - `query_only`
  - `exact`
  - `pairwise`
  - `pairwise_plus_query_context`
  - `pairwise_plus_triplet_mlp`
  - `pairwise_plus_triadic`
  - `triadic`
- Layerwise diagnostics for exactness and curl on both learned edge fields and raw hidden-state edge constructions
- Multi-seed aggregation, template-family breakdowns, and polarity-invariant evaluation
- A phase-2 four-node suite with control and non-order pair tasks
- Neon-themed plots for quick inspection

## Project Layout

```text
src/stoked_semantic/
  cli.py          Command-line entrypoint
  config.py       Experiment configuration
  data.py         Synthetic premise-query dataset builder
  encoding.py     Frozen encoder feature extraction and caching
  probes.py       Query-only, exact, pairwise, residual, and triadic probes
  training.py     Probe training and evaluation
  diagnostics.py  Exactness and curl metrics
  reporting.py    CSV/JSON outputs and plots
  pipeline.py     Single-seed and multi-seed orchestration

scripts/
  run_phase1.py       Main experiment runner
  summarize_phase1.py Result summarizer
```

## Install

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## How The Experiment Works

Each example contains:

- a premise-only text such as `alice outranks bob. bob outranks carol.`
- three pooled entity representations extracted from the premise
- a queried directed edge, represented as source/target indices
- a binary label indicating whether the queried relation is entailed by the premise world

The main comparison is:

- `exact`: forces queried predictions to come from node potentials
- `pairwise`: uses a flexible queried edge encoder
- `pairwise_plus_query_context`: adds the same whole-query linear context path used by the higher-order probes, but no multiplicative interaction
- `pairwise_plus_triplet_mlp`: adds a matched-capacity nonlinear tuple residual on top of the same pairwise branch
- `pairwise_plus_triadic`: adds a multiplicative triadic residual on top of the same pairwise branch
- `triadic`: adds a genuine three-node interaction

The exact probe can now also be swept over multiple ranks with `--exact-ranks` to test whether the task is close to a scalar or low-rank latent order.

The diagnostics are separate from prediction:

- `exactness`: how well the learned edge field can be projected onto `B phi`
- `curl`: how much cyclic residue remains on the triangle
- `raw_projection_r4`: exact-by-construction hidden-state projection sanity baseline
- `raw_skew_bilinear_r8`: probe-free antisymmetric hidden-state edge field used to test whether nonclosure is already present in the representation

## Task Suites

The runner now exposes four dataset suites:

- `phase1_order3`: the original 3-node transitivity and consistency setup
- `phase2_four_node`: a 4-node suite with:
  - `chain_order`: exact-friendly control
  - `same_side`: non-order composition from alliance/opposition statements
  - `same_row`: non-order composition from grid-layout statements
  - `same_triplet`: a true ternary hyperedge-membership task over queried triples
- `phase3_balanced_ternary`: a 6-node ternary suite with:
  - `balanced_triplet`: hyperedge membership where every pair appears equally often in positive and negative triples
- `phase4_masked_balanced_ternary`: a masked 6-node ternary completion suite with:
  - `masked_balanced_triplet`: the same balanced ternary world, but each example only shows a coverage-preserving subset of positive clauses, never reveals the queried positive triple verbatim, and samples pairwise-matched negative cases from the same visible-support regime

Phase 2 uses the same frozen-feature and probe pipeline, but with four pooled entities instead of three.
Phase 3 is designed to remove the main weakness of `same_triplet`: a pairwise probe should not be able to win purely from within-triple pair evidence.
Phase 4 goes one step further: the model has to complete masked ternary structure from partial evidence instead of recognizing a visible hyperedge.
Pair-query tasks and ternary-query tasks currently need to be run separately; mixed query arities are rejected at the CLI level.

## Holdout Modes

The dataset supports three useful evaluation regimes:

- default: train and test can share templates
- `--structural-template-holdout`: recommended next sweep; keeps active/passive, above/below, and base mixed families on both sides while dropping the paraphrase-mixed inverse families
- `--factorized-template-holdout`: disjoint structural template combinations while keeping both direct and inverse lexical realizations, both registers, and both clause orders on both sides
- `--strict-template-holdout`: train on forward templates, test on reverse templates
- `--balanced-template-holdout`: hold out template families while keeping both forward and reverse variants on both sides

The structural holdout is the recommended setting for the next sweep. The factorized holdout is still useful as a diagnostic stress test, but the paraphrase-mixed inverse templates have been the main remaining source of polarity instability. The strict holdout is intentionally harsh and is best treated as a stress test for polarity-flip failures. The older balanced holdout is still useful for debugging, but it can still hide polarity-transfer artifacts on some template families.

For phase 3 and phase 4, there is a separate ternary-family holdout:

- `--phase3-structural-holdout`: train on one set of ternary paraphrase families and test on held-out families, while keeping forward and reverse clause orders on both sides

## Running

Default run:

```bash
python scripts/run_phase1.py
```

Recommended structural holdout with two relation families:

```bash
python scripts/run_phase1.py \
  --structural-template-holdout \
  --relation-ids outranks older_than
```

Phase-2 four-node smoke run:

```bash
python scripts/run_phase1.py \
  --task-suite phase2_four_node \
  --relation-ids chain_order same_side same_row
```

Ternary-task run:

```bash
python scripts/run_phase1.py \
  --task-suite phase2_four_node \
  --relation-ids same_triplet
```

Balanced ternary run:

```bash
python scripts/run_phase1.py \
  --task-suite phase3_balanced_ternary \
  --relation-ids balanced_triplet \
  --max-length 96
```

Balanced ternary structural holdout:

```bash
python scripts/run_phase1.py \
  --task-suite phase3_balanced_ternary \
  --phase3-structural-holdout \
  --relation-ids balanced_triplet \
  --max-length 128
```

Masked balanced ternary completion run:

```bash
python scripts/run_phase1.py \
  --task-suite phase4_masked_balanced_ternary \
  --phase3-structural-holdout \
  --relation-ids masked_balanced_triplet \
  --masked-visible-clauses 5 6 7 \
  --max-length 128
```

Multi-seed run:

```bash
python scripts/run_phase1.py \
  --seeds 7 11 13 17 19 \
  --structural-template-holdout \
  --relation-ids outranks older_than \
  --train-examples-per-label 600 \
  --test-examples-per-label 72 \
  --epochs 20 \
  --encoder-batch-size 32 \
  --probe-batch-size 64 \
  --output-dir results/phase1_structural
```

Exact-rank sweep on the same structural split:

```bash
python scripts/run_phase1.py \
  --structural-template-holdout \
  --relation-ids outranks older_than \
  --exact-ranks 1 2 4 8 16 64 \
  --output-dir results/phase1_rank_sweep
```

Summarize a completed run:

```bash
python scripts/summarize_phase1.py results/phase1_structural
```

Scale-up pilot on the validated phase-4 task with `bert-large-uncased`:

```bash
python scripts/run_phase4_scaleup_pilot.py
```

That pilot defaults to:

- `phase4_masked_balanced_ternary`
- ternary-family structural holdout
- seeds `7 11 13`
- exact-rank sweep `1 2 4 8 16 64`
- `bert-large-uncased`
- encoder batch size `4`
- pretrained-only by default

Add `--with-random-control` if you want the expensive random baseline as well.

## Outputs

Single-seed runs write files such as:

- `probe_accuracy_by_layer.csv`
- `diagnostics_by_layer.csv`
- `template_group_accuracy_by_layer.csv` with template-family and query-type breakdowns
- `summary.json`
- `probe_accuracy_by_layer.png`
- `probe_accuracy_polarity_invariant_by_layer.png`
- `diagnostics_by_layer.png`

Multi-seed runs also write aggregate files:

- `probe_accuracy_aggregate.csv`
- `diagnostics_aggregate.csv`
- `template_group_accuracy_aggregate.csv`
- `probe_accuracy_aggregate.png`
- `probe_accuracy_polarity_invariant_aggregate.png`
- `diagnostics_aggregate.png`

If the requested `--output-dir` already exists, the runner now writes to a timestamped sibling instead of reusing it. For example:

```text
results/phase4_matched_negative_v3_20260312_142530
```

The resolved path is printed at the end of the run and recorded in `summary.json`.

## Caching

Frozen encoder features are cached under:

```text
.cache/features
```

Older cache directories can be kept under:

```text
.cache/legacy
```

Run outputs are written under `results/` and are ignored by git.

## Current Status

Phase 1 now looks fairly settled:

- `query_only` can be driven to chance under the hardened dataset
- on the clean structural holdout, low-rank `exact` probes beat `pairwise` and `triadic`
- that pattern is stable across all five checked seeds

Phase 2 changes the picture:

- on non-order four-node tasks, `pairwise` clearly beats `exact`
- on `same_triplet`, `exact` fails but `triadic` still does not beat `pairwise`

Phase 3 is the first strong higher-order result:

- on the same-template balanced ternary task, `triadic` beats both `pairwise` and all exact probes across all five seeds
- random-control probes stay near chance on that task
- pairwise diagnostics become less exact and curlier than the random baseline in the same regime

Phase 4 is the next serious test:

- the queried positive triple never appears verbatim in the premise
- every sampled premise still mentions all six entities, so the encoder sees a complete node set
- visible-clause count is configurable with `--masked-visible-clauses`

So the current evidence supports:

- latent-potential structure on the original order-like tasks
- genuine pairwise gains on harder non-order tasks
- a real same-template regime where triadic probes outperform pairwise ones

The next serious tests are:

- phase-3 ternary-family holdout, which asks whether that triadic gain survives held-out paraphrase families rather than only held-out names
- phase-4 masked ternary completion, which asks whether triadic structure still helps when the queried hyperedge is hidden from the premise entirely
