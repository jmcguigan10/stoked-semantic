# Preliminary Findings: Higher-Order Probes for Frozen Transformer Hidden States

## Abstract

We study when frozen transformer hidden states are best described by exact latent-potential probes, flexible pairwise probes, or genuinely higher-order probes. The answer is task-dependent. On order-like tasks, low-rank exact probes are best. On harder non-order tasks, pairwise probes beat exact probes. On a masked ternary completion benchmark with held-out ternary paraphrase families, multiplicative higher-order probes beat pairwise-only probes, and they do so by more than can be explained by whole-query context alone or by a matched-capacity triplet MLP. This effect survives scale-up from `bert-base-uncased` to `bert-large-uncased`, and raw hidden-state diagnostics show a strong pretrained-specific non-exact signature. The claim is deliberately narrow: frozen BERT-family representations support a meaningful higher-order advantage on the right task geometry.

## 1. Question

The motivating question is:

> When are frozen transformer representations adequately described by exact node potentials, when do pairwise relations suffice, and when do genuine higher-order interactions help?

The project is intentionally conservative. It does not argue that language is generically “triadic.” It asks when a higher-order probe basis earns its keep after controlling for simpler explanations.

## 2. Probe Families

Let layer-\(\ell\) node states be

\[
h_i^{(\ell)} \in \mathbb{R}^d.
\]

We compare the following probe families on span-pooled hidden states.

### Exact

Each node is mapped to a latent potential

\[
\phi_i = W h_i,
\]

and the classifier only sees pairwise differences inside the queried tuple.

### Pairwise

A learned edge encoder produces

\[
e_{ij} = f_2(h_i, h_j),
\]

and the classifier consumes the queried edges.

### Higher-order

The main higher-order term is multiplicative:

\[
t_{abc} = (U h_a) \odot (V h_b) \odot (W h_c).
\]

We use two probe variants:

- `triadic`: a direct higher-order classifier
- `pairwise_plus_triadic`: the pairwise branch plus a multiplicative triadic residual

### Fair controls

To avoid attributing gains to the wrong source, we also compare against:

- `pairwise_plus_query_context`: pairwise branch plus whole-query linear context, no multiplicative term
- `pairwise_plus_triplet_mlp`: pairwise branch plus matched-capacity nonlinear tuple residual

These controls separate:

1. extra query context
2. extra generic tuple-level capacity
3. specifically multiplicative higher-order structure

## 3. Geometry Diagnostics

For an oriented edge field \(\Omega\), we project onto exact flows:

\[
\Phi^* = \arg\min_\Phi \|\Omega - B\Phi\|_F^2
\]

and define

\[
\mathrm{Exactness}(\Omega) =
1 - \frac{\|\Omega - B\Phi^*\|_F^2}{\|\Omega\|_F^2 + \varepsilon},
\qquad
\mathrm{Curl}(\Omega) =
\frac{\|C\Omega\|_F^2}{\|\Omega\|_F^2 + \varepsilon},
\]

where \(B\) is the edge-vertex incidence matrix and \(C\) is the triangle-edge incidence matrix.

We use two probe-free hidden-state constructions:

- `raw_projection_r4`: exact-by-construction projected differences
- `raw_skew_bilinear_r8`: antisymmetric hidden-state edge field

\[
\omega^{\text{skew}}_{ij} = h_i^\top S h_j,
\qquad S^\top = -S
\]

The skew-bilinear field is the main probe-free test of whether pretrained representations already contain non-exact residue.

## 4. Benchmark Progression

### Phase 1: 3-node order tasks

Low-rank exact probes were best. This is a genuine negative result for the stronger higher-order claim on order-like tasks.

### Phase 2: 4-node non-order tasks

Pairwise probes beat exact probes strongly on `same_side` and `same_row`. This established a real `pairwise > exact` regime.

### Phase 3: balanced ternary hyperedge tasks

Same-template balanced ternary gave a strong `triadic > pairwise > exact` result, but ternary-family holdout weakened it substantially. This showed that the first triadic win was partly template-bound.

### Phase 4: masked balanced ternary completion

This is the current main benchmark. Each example shows only a subset of positive ternary clauses, never reveals the queried positive triple verbatim, and uses held-out ternary paraphrase families at test time.

## 5. Main Results

### 5.1 Base fair-control run

Reference:
- [results/phase4_controls_base](./results/phase4_controls_base)

Layer-averaged test accuracy:

| Probe | Pretrained | Random |
| --- | ---: | ---: |
| best exact | `0.5022` | `0.5063` |
| `pairwise` | `0.5217` | `0.5047` |
| `pairwise_plus_query_context` | `0.5177` | `0.5042` |
| `pairwise_plus_triplet_mlp` | `0.5250` | `0.5004` |
| `pairwise_plus_triadic` | `0.5457` | `0.5065` |
| `triadic` | `0.5473` | `0.5123` |

Interpretation:

- exact is dead on this task
- pairwise helps a little
- context-only does not explain the gain
- generic tuple MLP capacity explains only a little
- multiplicative higher-order probes are the only ones with a clear lift over pairwise

### 5.2 Canonical `bert-large-uncased` run

Reference:
- [results/phase4_controls_bert_large_full](./results/phase4_controls_bert_large_full)

Layer-averaged test accuracy:

| Probe | Pretrained | Random |
| --- | ---: | ---: |
| best exact | `0.5103` | `0.4953` |
| `pairwise` | `0.5302` | `0.4920` |
| `pairwise_plus_query_context` | `0.5351` | `0.4931` |
| `pairwise_plus_triplet_mlp` | `0.5329` | `0.4951` |
| `pairwise_plus_triadic` | `0.5706` | `0.4931` |
| `triadic` | `0.5681` | `0.4986` |

The key gaps are:

| Gap | Pretrained | Random |
| --- | ---: | ---: |
| `pairwise_plus_query_context - pairwise` | `+0.0049` | `+0.0011` |
| `pairwise_plus_triplet_mlp - pairwise_plus_query_context` | `-0.0022` | `+0.0020` |
| `pairwise_plus_triadic - pairwise` | `+0.0404` | `+0.0011` |
| `pairwise_plus_triadic - pairwise_plus_triplet_mlp` | `+0.0377` | `-0.0020` |
| `triadic - pairwise` | `+0.0378` | `+0.0066` |

This is the strongest result in the project. The higher-order gain survives scale-up, and it is not explained by context-only or generic tuple-MLP controls.

### 5.3 Seed stability

In the canonical `bert-large` bundle:

- `pairwise_plus_triadic > pairwise` in all 5 pretrained seeds
- `triadic > pairwise` in all 5 pretrained seeds

The corresponding random-control gaps are small and mixed in sign.

## 6. Raw Geometry

The probe-free hidden-state geometry now supports the predictive story.

On the canonical `bert-large` run:

| Variant | Source | Positional Exactness | Positional Curl |
| --- | --- | ---: | ---: |
| pretrained | `raw_projection_r4` | `1.0000` | `0.0022` |
| pretrained | `raw_skew_bilinear_r8` | `0.6466` | `2.1258` |
| random | `raw_projection_r4` | `1.0000` | `0.0043` |
| random | `raw_skew_bilinear_r8` | `0.9185` | `0.4945` |

This is the first probe-free result in the project that clearly supports the higher-order interpretation.

## 7. Geometry-Performance Link

We also tested whether the layerwise higher-order lift tracks the raw skew-bilinear curl by layer.

For each layer \(\ell\), define

\[
\Delta_\ell = \mathrm{Acc}_\ell(\text{pairwise_plus_triadic}) - \mathrm{Acc}_\ell(\text{pairwise}),
\qquad
C_\ell = \mathrm{Curl}_\ell(\text{raw\_skew\_bilinear}).
\]

Per-seed correlations across layers were not strongly positive. On average, they were mildly negative:

| Benchmark | Variant | Mean Pearson \(r\) | Mean Spearman \(r\) |
| --- | --- | ---: | ---: |
| base | pretrained | `-0.3191` | `-0.2617` |
| base | random | `+0.1105` | `+0.1082` |
| `bert-large` | pretrained | `-0.2843` | `-0.2966` |
| `bert-large` | random | `-0.2767` | `-0.2900` |

So the raw non-exact geometry is real, but it is not a simple same-layer predictor of higher-order probe gains. The best current interpretation is that pretrained representations contain a strong non-exact signature, and later layers make that structure usable.

## 8. Conservative Conclusion

The current evidence supports a narrow but real claim:

1. frozen BERT-family representations look exact-like on order tasks
2. they require pairwise structure on some non-order tasks
3. on masked ternary completion with held-out ternary paraphrase families, multiplicative higher-order probes beat pairwise-only probes
4. that gain survives fair residual controls, raw geometry checks, and scale-up to `bert-large-uncased`

What we do **not** claim:

- that language is generically or uniformly triadic
- that higher-order structure dominates all tasks
- that raw curl at a layer directly predicts probe gains at that same layer

## 9. Why This Is Worth Discussing

The result is now strong enough to motivate collaboration because it is:

- controlled: the higher-order gain survives fair residual baselines
- representation-level: raw hidden-state diagnostics support the story
- nontrivial: the gain is task-specific rather than universal
- reproducible: the canonical benchmark and code are already in a clean public repo

## 10. Next Technical Steps

The most sensible next steps are:

1. add antisymmetric attention-based raw diagnostics
2. run one second architecture family on the exact same phase-4 control setup
3. avoid further benchmark churn unless the new diagnostics reveal a clear pathology

At this point, the project is serious enough to send to professors for feedback or collaboration.
