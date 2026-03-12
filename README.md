# Stoked Semantic

Phase 1 probing experiments for a narrow question:

> On controlled relational-consistency tasks, are frozen transformer hidden states better described by exact node potentials, unconstrained pairwise relations, or genuine triadic interactions?

The codebase is deliberately small. It generates synthetic premise-query datasets, extracts frozen BERT features, trains several probe families layer by layer, and reports both predictive accuracy and Hodge-style exactness/curl diagnostics.

## What This Repo Contains

- A controlled synthetic dataset with lexical holdout and configurable template holdouts
- Frozen-feature extraction from `bert-base-uncased`
- Four probe families:
  - `query_only`
  - `exact`
  - `pairwise`
  - `triadic`
- Layerwise diagnostics for exactness and curl on learned edge fields
- Multi-seed aggregation, template-family breakdowns, and polarity-invariant evaluation
- Neon-themed plots for quick inspection

## Project Layout

```text
src/stoked_semantic/
  cli.py          Command-line entrypoint
  config.py       Experiment configuration
  data.py         Synthetic premise-query dataset builder
  encoding.py     Frozen encoder feature extraction and caching
  probes.py       Query-only, exact, pairwise, and triadic probes
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
- `triadic`: adds a genuine three-node interaction

The diagnostics are separate from prediction:

- `exactness`: how well the learned edge field can be projected onto `B phi`
- `curl`: how much cyclic residue remains on the triangle

## Holdout Modes

The dataset supports three useful evaluation regimes:

- default: train and test can share templates
- `--strict-template-holdout`: train on forward templates, test on reverse templates
- `--balanced-template-holdout`: hold out template families while keeping both forward and reverse variants on both sides

The balanced holdout is the recommended setting for phase 1. The strict holdout is intentionally harsh and is best treated as a stress test for polarity-flip failures.

## Running

Default run:

```bash
python scripts/run_phase1.py
```

Recommended balanced holdout with two relation families:

```bash
python scripts/run_phase1.py \
  --balanced-template-holdout \
  --relation-ids outranks older_than
```

Multi-seed run:

```bash
python scripts/run_phase1.py \
  --seeds 7 11 13 17 19 \
  --balanced-template-holdout \
  --relation-ids outranks older_than \
  --train-examples-per-label 600 \
  --test-examples-per-label 72 \
  --epochs 20 \
  --encoder-batch-size 32 \
  --probe-batch-size 64 \
  --output-dir results/phase1_balanced
```

Summarize a completed run:

```bash
python scripts/summarize_phase1.py results/phase1_balanced
```

## Outputs

Single-seed runs write files such as:

- `probe_accuracy_by_layer.csv`
- `diagnostics_by_layer.csv`
- `template_group_accuracy_by_layer.csv`
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

The current phase-1 evidence is methodologically useful but not yet supportive of a strong higher-order claim. So far, the most stable pattern is:

- `query_only` can be driven to chance under the hardened dataset
- `pairwise` tends to beat `exact` slightly
- `triadic` has not shown a robust advantage over `pairwise`
- curl remains small in the most useful layers

That makes this repo a good base for controlled follow-up experiments, but not evidence yet that genuine triadic structure is necessary.
