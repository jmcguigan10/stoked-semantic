# Next Steps

The repo now supports a clearer phased story than it did originally.

## Current State

### Phase 1: 3-node order tasks

- Result: low-rank `exact` probes are best on the clean structural holdout.
- Interpretation: frozen BERT hidden states on strict order-like tasks look mostly integrable / latent-potential-like.
- Status: settled enough to treat as a real negative result for the strong higher-order claim on this task family.

### Phase 2: 4-node non-order tasks

- Result: `pairwise` beats `exact` on `same_side` and `same_row`.
- Interpretation: exact potentials are not enough once the task is no longer basically latent order.
- Status: useful intermediate result; confirms task geometry matters.

### Phase 3: balanced ternary hyperedge tasks

- Same-template balanced ternary:
  - Result: `triadic > pairwise > exact`
  - Interpretation: genuine triadic probes can win when pairwise evidence is globally balanced.
- Structural ternary-family holdout:
  - Result: only a weak higher-order edge
  - Interpretation: much of the phase-3 win was template-bound.

### Phase 4: masked balanced ternary completion

- Result on the current masked structural split:
  - pretrained `triadic` and `pairwise_plus_triadic` beat `pairwise`
  - random-control probes stay much flatter
  - pairwise diagnostics are less exact and curlier in pretrained than in random
- Interpretation:
  - this is the first held-out-form result that looks meaningfully positive for higher-order structure
  - but the gain is concentrated on hidden-positive masked cases; negative masked cases are still the weak point

## Immediate Priorities

### 1. Fair residual controls on phase 4

This is now implemented.

New controls:
- `pairwise_plus_query_context`: same pairwise branch plus whole-query linear context, but no multiplicative term
- `pairwise_plus_triplet_mlp`: same pairwise branch plus a matched-capacity nonlinear tuple residual

Why they matter:
- they separate three explanations that were previously entangled:
  - extra query context
  - extra nonlinear tuple capacity
  - specifically multiplicative triadic structure

Immediate next run:
- rerun the validated phase-4 matched-negative setup on `bert-base-uncased`
- then rerun the same comparison on `bert-large-uncased`
- compare `pairwise`, `pairwise_plus_query_context`, `pairwise_plus_triplet_mlp`, `pairwise_plus_triadic`, and `triadic`

Success condition:
- `pairwise_plus_triadic` stays above `pairwise_plus_query_context`
- `pairwise_plus_triadic` stays above `pairwise_plus_triplet_mlp`
- the gap is larger in pretrained than in random control

### 2. Raw hidden-state diagnostics

This is now implemented.

Current raw sources:
- `raw_projection_r4`: exact-by-construction projected difference edges
- `raw_skew_bilinear_r8`: probe-free antisymmetric skew-bilinear edges

Still missing:
- optional antisymmetric attention edges once attentions are enabled

Decision gate:
- if the raw skew-bilinear diagnostics track the probe story, the higher-order interpretation becomes much stronger

### 3. Only then decide on broader scale-up

This gate is now passed for the current phase-4 matched-negative setup:
- the refined phase-4 masked sweep still shows `triadic` or `pairwise_plus_triadic` beating `pairwise`
- the raw diagnostics show a consistent pretrained-specific non-exact / higher-order signal

Current scale-up status:
- a 3-seed `bert-large-uncased` pilot already improved the masked ternary result
- one-seed random control looked promising, and seeds `11`/`13` kept the higher-order edge pretrained-side

Current scale-up plan:
- finish the fair-control reruns on `bert-base-uncased`
- then run one canonical `bert-large-uncased` bundle with pretrained and random control together
- keep the exact-rank sweep and raw diagnostics unchanged

Decision questions:
- does the higher-order margin widen or stabilize relative to `bert-base-uncased`?
- do the raw skew-bilinear diagnostics strengthen in the same layer range?
- does `balanced_align` remain the weak family, or does the larger encoder improve transfer there too?

## Practical Sequence

1. Rerun the clean phase-4 matched-negative setup on `bert-base-uncased` with the new residual controls.
2. Compare context-only, generic tuple-MLP, triadic-residual, and standalone triadic probes.
3. Run one canonical `bert-large-uncased` bundle with pretrained and random control together.
4. Decide whether the remaining `balanced_align` weakness is task geometry or representation geometry.
5. Only then revisit attention-based analyses or broader model families.

## Decision Rule

Treat the project as having crossed into a genuinely positive regime only if:

- `query_only` stays at chance
- random-control higher-order probes stay near chance or much flatter than pretrained
- `triadic` or `pairwise_plus_triadic` beats `pairwise` across most seeds
- that gain survives held-out ternary paraphrase families
- the gain is not confined to one label type or one template family

Right now, we are close, but not finished.
