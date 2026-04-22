# Stoked Semantic

The name is a play on Stokes' theorem: this project asks whether language-model semantics look more like exact, low-curl structure in a Stokes/Hodge sense or whether genuinely higher-order interactions are needed to explain them.

Controlled probing experiments for testing whether frozen transformer hidden states are best described by exact node potentials, flexible pairwise relations, or genuinely higher-order interactions.

This repository builds synthetic relational benchmarks, extracts frozen BERT-family features, trains matched probe families layer by layer, and reports both predictive accuracy and geometry diagnostics such as exactness and curl. The main result is deliberately narrow: higher-order structure does not win everywhere, but it does help on masked ternary completion after fair controls and held-out paraphrase transfer.

## Why This Exists

Many probing papers stop at "a classifier predicts the label." This project asks a sharper question: what interaction class is actually needed to explain the signal in frozen hidden states?

The benchmark progression is meant to separate three stories:

- order-like tasks where low-rank exact structure should be enough
- harder non-order tasks where pairwise structure should help
- masked ternary tasks where a higher-order residual has to earn its keep beyond pairwise and generic tuple-capacity controls

## Experiment Sketch

```text
synthetic premise/query tasks
        |
        v
frozen transformer encoder
        |
        v
pooled entity representations
        |
        +--> exact probes
        +--> pairwise probes
        +--> higher-order probes
        |
        v
accuracy + polarity-invariant accuracy + exactness/curl diagnostics
```

## Repository Layout

```text
src/stoked_semantic/
  cli.py          CLI entrypoint
  config.py       experiment configuration
  data.py         synthetic dataset builders
  encoding.py     frozen encoder feature extraction and caching
  probes.py       exact, pairwise, and higher-order probes
  training.py     probe training and evaluation
  diagnostics.py  exactness and curl metrics
  reporting.py    CSV/JSON summaries and plots
  pipeline.py     single-seed and multi-seed orchestration

scripts/
  run_phase1.py              main experiment runner
  summarize_phase1.py        result summarizer
  analyze_geometry_link.py   raw-geometry vs accuracy analysis
  audit_dataset_bias.py      label/bias checks
  run_phase4_scaleup_pilot.py canonical bert-large helper

results/
  analysis_v3.md             current writeup
  phase1_structural/         order-task structural-holdout bundle
  phase4_controls_base/      canonical bert-base fair-control bundle
  phase4_controls_bert_large_full/ canonical bert-large fair-control bundle
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

After installation, either of these entrypoints works:

```bash
stoked-semantic --help
python -m stoked_semantic --help
```

## Small Smoke Run

This is the smallest end-to-end experiment command worth keeping in the README:

```bash
python scripts/run_phase1.py \
  --relation-ids outranks \
  --train-examples-per-label 32 \
  --test-examples-per-label 8 \
  --epochs 1 \
  --skip-random-control \
  --no-plots \
  --output-dir results/smoke_publish_check
```

Expected outputs in the chosen `--output-dir`:

- `probe_accuracy_by_layer.csv`
- `diagnostics_by_layer.csv`
- `template_group_accuracy_by_layer.csv`
- `summary.json`

## Reproducing Published Results

The main published base-model result is the fair-control phase-4 masked ternary bundle:

```bash
python scripts/run_phase1.py \
  --task-suite phase4_masked_balanced_ternary \
  --phase3-structural-holdout \
  --relation-ids masked_balanced_triplet \
  --masked-visible-clauses 5 6 7 \
  --exact-ranks 1 2 4 8 16 64 \
  --seeds 7 11 13 17 19 \
  --train-examples-per-label 600 \
  --test-examples-per-label 72 \
  --epochs 20 \
  --max-length 128 \
  --output-dir results/phase4_controls_base
```

The canonical scale-up run on `bert-large-uncased` is:

```bash
python scripts/run_phase4_scaleup_pilot.py \
  --with-random-control \
  --seeds 7 11 13 17 19 \
  --output-dir results/phase4_controls_bert_large_full
```

If the target `--output-dir` already exists, the runner creates a timestamped sibling instead of overwriting the existing bundle. The resolved path is printed and recorded in `summary.json`.

## Current Findings

| Suite | Main takeaway | Reference |
| --- | --- | --- |
| Phase 1 order-3 structural holdout | Low-rank `exact` probes beat `pairwise` and `triadic`; higher-order structure is not needed here. | [`results/phase1_structural`](./results/phase1_structural) |
| Phase 2 four-node non-order tasks | `pairwise` beats `exact` on `same_side` and `same_row`; this is the clean `pairwise > exact` regime. | [`results/analysis_v3.md`](./results/analysis_v3.md) |
| Phase 3 balanced ternary | Same-template `triadic > pairwise > exact`, but ternary-family holdout weakens the effect. | [`results/analysis_v3.md`](./results/analysis_v3.md) |
| Phase 4 masked ternary, base model | Layer-averaged pretrained accuracy: `pairwise 0.5217`, `pairwise_plus_triadic 0.5457`, `triadic 0.5473`, best exact `0.5022`. | [`results/phase4_controls_base`](./results/phase4_controls_base) |
| Phase 4 masked ternary, `bert-large-uncased` | Layer-averaged pretrained accuracy: `pairwise 0.5302`, `pairwise_plus_triadic 0.5706`, `triadic 0.5681`, best exact `0.5103`. | [`results/phase4_controls_bert_large_full`](./results/phase4_controls_bert_large_full) |

The strongest current claim is conservative:

- frozen BERT-family states look exact-like on order tasks
- they need pairwise structure on some harder non-order tasks
- on masked ternary completion with held-out ternary paraphrase families, multiplicative higher-order probes beat pairwise-only probes
- that phase-4 gain survives query-context and matched-capacity tuple-MLP controls, raw-geometry diagnostics, and scale-up to `bert-large-uncased`

## Outputs And Caching

Frozen encoder features are cached under `.cache/features`.

This repository intentionally versions a curated set of result snapshots under `results/` so the published claims are inspectable without rerunning everything. New local runs under `results/` are still ignored by `.gitignore` unless you explicitly force-add them.

To summarize an existing run directory:

```bash
python scripts/summarize_phase1.py results/phase1_structural
```

## Where To Look First

- [`PRELIMINARY_FINDINGS.md`](./PRELIMINARY_FINDINGS.md) for the compact research-note writeup
- [`results/analysis_v3.md`](./results/analysis_v3.md) for the current in-repo interpretation
- [`nextsteps.md`](./nextsteps.md) for the next experimental steps
- [`src/stoked_semantic/data.py`](./src/stoked_semantic/data.py) for task construction
- [`src/stoked_semantic/probes.py`](./src/stoked_semantic/probes.py) for probe definitions

## Limitations

- The tasks are synthetic and controlled; they are evidence about representational structure under specific task geometries, not a blanket claim about natural language semantics.
- The project compares a focused set of probe families, not every possible structured predictor.
- The strongest positive result is task-dependent and should not be generalized into "language is triadic."

## Citation

If you use this repository, cite the software entry in [`CITATION.cff`](./CITATION.cff).

## License

This repository is released under the MIT License. See [`LICENSE`](./LICENSE).
