Current corpus generation (November 17, 2026): `data/jsettlers-friendly-250/`
contains the new friendly-robber native JSettlers dataset as games are committed.
Target: 250 completed games, all schema-3 decisions, whole-game train/validation/
test splits, independent per-game replay and checksum audits. This is separate
from the historical standard-robber corpora below. New training uses W4.
See [the current workflow](experiments/catan-ai/FRIENDLY_W4.md).

# JSettlers policy/value corpora

For the phase-2 observation-only encoder and reconstructed complete-decision
view, use [schema 3](experiments/catan-ai/OBSERVATIONS.md) and
`agent_apis.observation_dataset.iter_observed_samples`. The stored corpora remain
schema 2; the loader below continues to return their original 633 features.

The current datasets are under `data/jsettlers-corpora/`:

- `full-250/`: exactly 250 completed games, retaining every decision.
- `downsampled/`: disjoint additional games, up to **10,000 games**, bounded by
  the remaining byte budget. Retains every tenth action **per actor** (ordinals
  0, 10, 20, ...), every initial settlement/road placement, and each player's
  final ten actions. These sets are unioned without duplicate samples.

The completed, verified output contains **97,766 samples in 250 full games**
and **64,503 samples in 1,017 downsampled games** (1,267 distinct game paths).
It occupies **29,924,144 bytes stored** and **996,542,170 bytes uncompressed**,
including metadata. Generation stopped at the byte budget; one additional game
with a native API translation error was excluded. Every saved sample, sampling
rule, replay, exact tile number, action mask and outcome target passed audit.

The **combined** corpora stay below 1,000,000,000 bytes both stored and as
uncompressed JSON. Generation stops between whole games when this budget is
reached; 10,000 is an upper limit, not a promised game count. See each manifest
for counts, checksums, failures, splits and rules. Data is ignored by Git.

Both seats use the existing JSettlers SMART API. Catanatron supplies rules and
chance outcomes: BASE board, 15 victory points, discard above 9, standard
robber, bank/port trades, no player trades. Incomplete games are excluded from
outcome labels. No random policy fallback is used.

## Exact dice numbers

Schema 2 has **633 numeric features**: the original 614 Catanatron features,
followed by `TILE0_DICE_NUMBER` through `TILE18_DICE_NUMBER`, ordered by tile ID.
Numbers come directly from the authoritative public map's LAND `number`
fields; desert is 0. Production probabilities are retained as separate existing
features. Consequently 6 and 8 stay distinct, and a model can identify tiles
that produce together. Nothing is inferred from equal production probabilities.

For a fresh live Catanatron state, use the matching encoder:

```python
from agent_apis.dataset_bundle import create_numbered_sample
x = create_numbered_sample(game, game.state.current_color())
```

`manifest.json` contains the complete feature ordering and 332-action vocabulary.
The action IDs remain compatible with Catanatron's BASE Gym action space with
BLUE/RED colors. P0 features rotate to the actor; action coordinates and victim
colors remain absolute Catanatron coordinates/colors.

## Training

```python
from agent_apis.dataset import iter_samples

for sample in iter_samples('data/jsettlers-corpora/downsampled', split='train'):
    x = sample['features']                  # 633 features including exact dice
    policy_target = sample['action_id']     # demonstrated action class
    legal = sample['legal_action_ids']      # sparse mask
    value_target = sample['value_target']   # actor's outcome: +1 win / -1 loss
```

Use the same loader for `full-250`. The loader streams training columns and
excludes privileged initial/final state, seeds, chance results and replay
metadata. Train/validation/test splits follow the same seed hash across both
corpora, with whole games kept together. Games and seeds are disjoint between
corpora. Policy targets are JSettlers demonstrations, not MCTS visit counts;
value targets are empirical terminal outcomes, not calibrated state values.

`step` and `num_turns` retain their original pre-action engine indices. Sampling
counts individual decisions including rolls, discards, trades and END_TURN,
not whole turns. The final ten decisions are selected separately for each
actor. Downsampled rows are **not adjacent transitions**: do not use the next
saved row as the immediate successor for TD training. `next_actor` still means
immediately after the recorded action. `reward` is zero except at terminal;
`terminated` marks the final action. No truncated games receive labels.

## Storage and replay

Each `.jsonl.gz` file contains one header followed by retained pre-action rows.
Rows include compact actor-redacted state, features, turn/action indices, legal
mask, chosen action and outcome target. Headers include exact initial and final
Catanatron serialized states and the full compact action log. The action log
allows replay across omitted states; omitted feature vectors and observation
snapshots are not stored. Header state, RNG, hidden hands, ordered decks and
chance outcomes are **privileged replay data**, never model inputs.

```python
from agent_apis.dataset import replay_game

for game, row in replay_game('data/jsettlers-corpora/downsampled/game-0251.jsonl.gz'):
    # A functional Catanatron Game BEFORE each retained action.
    # Unsaved intervening actions are executed automatically.
    assert len(game.state.action_records) == row['step']
```

Pick an actual filename from the manifest. The iterator reuses one mutable
Game: copy/serialize it before advancing if persistence is needed. Exact replay
uses saved engine RNG and checks chance outcomes and final state. Regenerating
bots with the same seeds can change their choices because Java planning has
its own randomness.

## Build and audit

```sh
# Reuse the original complete corpus as input, generate additional unique games:
.conda/bin/python -m agent_apis.dataset_bundle build \
  --source data/jsettlers-1v1 --output data/jsettlers-corpora --sampled-games 10000

# Audit both corpora, exact numbers, sampling, replay, masks, labels and budget:
.conda/bin/python -m agent_apis.dataset_bundle verify
.conda/bin/python -m unittest -v tests.test_dataset_bundle
```

A new output must be empty; rerunning the same build resumes from its saved manifests. The previous `data/jsettlers-1v1/` is retained as legacy input; use the two new corpora for the exact-number encoding. Dependencies are the project's existing setup plus
`pip install -e '.[dataset]'`. The original dataset generator remains available
in `agent_apis.dataset`; the bundle builder adds the exact-number encoding and
sampling. See `agent_apis/README.md` for JSettlers installation and source.
