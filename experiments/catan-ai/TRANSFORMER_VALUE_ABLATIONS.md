# Transformer value and attention ablations — November 18, 2026

User requested measured tests of a value-logit penalty, value-branch dropout,
one/two added value layers, and Exphormer-based typed sparse attention with a
global connector graph. The six-way controlled suite includes an unchanged
Transformer control. These are isolated experiments; pinned CNN/Transformer
training sources, historical weights and deployed agents remain intact.

Run: `.conda/bin/python -m agent_apis.neural.value_ablations`.
Progress/results: `runs/catan-ai/transformer-value-ablations/`.
Log: `runs/catan-ai/transformer-value-ablations.log`.

## Predeclared comparison

All variants start from the incoming selected Transformer checkpoint for
cohort 8. Reuse its complete 2,000-game expert manifest: 1,800 train, 100
validation, 100 test, preserving whole-game splits. Reconstruct at most 64
seeded positions/game in RAM after checksum/action/chance/final-state replay
checks. No test features or labels are loaded. Retain original game journals.

Train each variant for 2,000 AdamW updates, batch 256, the original learning
rate/weight decay schedule and augmentation, identical minibatch/augmentation
RNG streams, W4 half-uniform/half-capped-family sampling, inverse-probability
outcome BCE correction exactly once. Both sampler halves draw from the same
expert training pool. Use one CPU thread and MPS training. Validate every 250
updates, including zero; choose earliest minimum unweighted policy NLL + value
BCE. Save final weights, best weights and atomic optimizer/RNG recovery states.
Dropout consumes extra stochastic randomness as part of that treatment.

Variants:
- Control: unchanged StructuralTransformer.
- Logit penalty: add 0.01 times uniform-position mean squared value logit.
  Under W4 sampling estimate that regularizer with the same single sample
  importance weight; the outcome BCE is corrected only once.
- Value dropout: p=0.2 on context passed to the value linear layer only.
- Depth +1/+2: add one/two width-128 Linear+GELU residual layers before the
  existing value output. Zero initialization makes each residual identity at
  start. Policy and value initial predictions exactly match the control.
- Typed Exphormer-style: keep four pretrained four-head attention blocks,
  but constrain interactions to board incidence, dynamic ownership and ports,
  a seeded degree-four random-regular expander overlay, four all-to-all global
  connector tokens, and self loops. A connector clique links global messages.
  Verify connected overlay with normalized spectral gap >0.05. Distinguish
  directional tile/node and node/road links, expander links, connector direction,
  connector peers, directional player/node and player/road ownership, ports,
  and self loops with learned per-head edge-type biases. Retain pretrained
  graph-distance and existing public-relation biases. This is an adaptation,
  not a reproduction of GraphGPS/Exphormer's full benchmark architecture.

Attention uses dense masked SDPA over a sparse interaction graph. It does not
claim an O(E) compute implementation or speed benefit at this 152-token scale.
The sparse variant changes initial attention and cannot preserve the exact
initial function; report its update-zero loss separately.

## Reporting

Report selected and final policy NLL, value BCE, Brier, action accuracy, selected
update and training fit on a seeded 100-training-game sample. Also evaluate all
selected variants on cohort-7 validation games as a secondary comparison.
That set was used in earlier lineage selection and inspected before this test;
it is not a fresh untouched confirmation set. This single-seed, single-training-
cohort screen cannot establish general architecture superiority or playing
strength. Compare controls, regularizers, added capacity, and sparse attention
separately; do not attribute all sparse-variant differences to one component.

References: [Exphormer paper](https://proceedings.mlr.press/v202/shirzad23a.html),
[author implementation](https://github.com/hamed1375/Exphormer).
