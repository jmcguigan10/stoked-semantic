# Outreach Email Draft

## Short Version

Subject: Preliminary result on higher-order probes for frozen transformer representations

Hi Professor [Name],

I’ve been working on a small interpretability project around a narrow question: when are frozen transformer hidden states better described by exact latent-potential probes, pairwise probes, or genuinely higher-order probes?

The current result is more specific than the original idea. On order-like tasks, low-rank exact probes are best. On harder non-order tasks, pairwise probes beat exact probes. On a masked ternary completion benchmark with held-out ternary paraphrase families, multiplicative higher-order probes beat pairwise-only probes, and the gain survives fair controls, raw hidden-state geometry checks, and scale-up from `bert-base-uncased` to `bert-large-uncased`.

I wrote up a concise note here:
- [PRELIMINARY_FINDINGS.md](./PRELIMINARY_FINDINGS.md)

And the code / benchmark are here:
- https://github.com/jmcguigan10/stoked-semantic

I’d value your take on whether the result looks interesting enough to push further, and in particular whether the next step should be attention-based diagnostics, a second architecture family, or a tighter writeup.

Best,
[Your Name]

## Slightly Longer Version

Subject: Preliminary higher-order probe result on frozen BERT-family representations

Hi Professor [Name],

I’ve been building a controlled probe benchmark around a simple question: when do frozen transformer hidden states look exact-like, pairwise, or genuinely higher-order?

The project ended up with a cleaner task-dependent story than I expected:

- on 3-node order-like tasks, low-rank exact probes are best
- on harder 4-node non-order tasks, pairwise probes beat exact probes
- on a masked ternary completion benchmark with held-out ternary paraphrase families, multiplicative higher-order probes beat pairwise-only probes

The result that seems most worth discussing is the last one. On the canonical benchmark, `pairwise_plus_triadic` and `triadic` beat `pairwise`, while context-only and matched-capacity triplet-MLP controls do not explain the gain. A probe-free raw skew-bilinear diagnostic also shows a strong pretrained-vs-random non-exact geometry difference. The effect survives scale-up from `bert-base-uncased` to `bert-large-uncased`.

I’m attaching a concise note here:
- [PRELIMINARY_FINDINGS.md](./PRELIMINARY_FINDINGS.md)

And the clean repo is here:
- https://github.com/jmcguigan10/stoked-semantic

If you have time, I’d really appreciate your view on whether this looks like a good basis for collaboration or a small paper, and what you think the highest-value next step is.

Best,
[Your Name]

## Suggested Attachment Set

- [PRELIMINARY_FINDINGS.md](./PRELIMINARY_FINDINGS.md)
- [results/analysis_v3.md](./results/analysis_v3.md) if they want the fuller internal writeup

## Suggested Talking Points If They Reply

- The claim is intentionally narrow; it is not “language is triadic.”
- The strongest benchmark is [results/phase4_controls_bert_large_full](./results/phase4_controls_bert_large_full).
- The main open extensions are attention diagnostics and one second architecture family.
