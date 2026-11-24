# Schema 3 observation and data contract

The rules are two-player BLUE/RED, BASE, 15 victory points, discard
above nine and no player trading. Friendly robber is the default for new games
as of November 13, 2026; standard robber remains supported for historical data.
The existing `RULE_FRIENDLY_ROBBER` feature records the actual boolean rule.
Original schema-2 shards
are immutable. `feature-schema-v3.json` freezes the 681 feature names and fixed
normalization scales. `phase2.json` identifies the complete audit configuration.

## Boundary and exact resource ledger

`from_web_view(web_view(game, Color[color]), color)` whitelists the observation
accepted by the learned-agent encoder. It strips seeds, random state, winners,
opponent hand identities and private opponent draws and collapses the remaining
development-deck composition to its total size. It rejects an ordered deck.
Pass a view requested for the same perspective: a missing private own field is
an error. No framework or JSettlers API redaction behavior is changed.

`encode_observation` receives this permitted dictionary, never a Game. Replay
and audit alone deserialize authoritative checkpoints and read authoritative
hands for comparison. They redact observations before calling the encoder.
Outcomes are attached to training rows only after encoding.

The bank contains 19 of each resource initially. With precisely two players and
no resources outside the bank and those hands, for each resource:

`opponent = 19 - public_bank - private_own_hand`

This is the exact bank-conservation alternative explicitly allowed by section 2
of the plan. It requires no probabilistic resource belief and never reads a
hidden opponent hand. `ResourceLedger` starts with empty hands and a full bank;
each permitted snapshot updates both reconstructed hands and records each
hand's net resource changes. Counts must be nonnegative integers, public
opponent hand size must agree, and bank/hand changes must conserve resources.
Production, setup receipts, costs, discards, bank/port trades, theft, Monopoly
and Year of Plenty all use the same accounting invariant.

The permitted history plus the latest own hand and bank snapshot is a
sufficient recovery checkpoint. Recovery does not need the missing private
theft result: infer its net effect from own-hand and bank changes. A new ledger
recovers exactly from the latest snapshot even after skipped events, restart or
history rewind. The audit compares fresh recovery with continuously updated
ledgers at every snapshot for both seats. This method assumes the current API's
exact public bank counts; a different bank-information policy requires a new
information experiment and ledger implementation.

## Feature fields and normalization

Features 0–632 reproduce the current 633 schema-2 values exactly, including
actor-relative P0/P1 player fields, 19 categorical tile-number codes, separate
production probabilities, ports, roads, intersections, public plays, private
own cards and total development-deck size. Static metadata loads without
creating a random Game. The audit checks the frozen base ordering against both
corpus manifests.

Features 633–637 append inferred opponent resources in WOOD, BRICK, SHEEP,
WHEAT, ORE order. The remaining 43 fields add absolute actor/observer color,
actor and turn-owner relationships, setup/free-road/knight/discard information,
rule constants, a five-way prompt encoding, purchases, new/older development
card counts and own card-age/playability flags. Playability flags describe the
currently supplied legal actions, including phase and resource restrictions.

`normalized_features` divides numeric values by the schema's fixed scales. No
normalization statistics are learned from validation/test data. Resource counts
use 19; aggregate hand size and discard counts use the conservative bound 95;
development counts use 25; pieces use their supply limits; score uses 20; road
length uses 15. Boolean/probability fields use 1. Exact dice codes remain integer
categories (0 for desert, 2–12 except 7), for embeddings or one-hot treatment by
every architecture. Normalization does not clip or turn dice codes into ordinal
probabilities. Rule-constant scales are recorded individually in the schema.

Own newly purchased cards are counted by type since that player's latest
END_TURN; older cards are current holdings minus these counts. Opponent newly
purchased cards are public counts with unknown identities. Opponent older-card
total is public holding total minus new purchases. Initial supply is fixed by
Catanatron: 14 knights, two each Monopoly/Year of Plenty/Road Building and five
victory-point cards. Public plays, own holdings, purchases and card ages constrain
the possible hidden allocations; no remaining hidden composition is exposed.

`development_knowledge` normalizes the prefix history into buy/play events,
with seat, completed own-turn ordinal and card identity only when visible.
Other events, including raw dice labels, are omitted. These safe constraints are
returned separately for the later development-card belief implementation;
they do not introduce raw history tokens into the initial model features.

## Data views

```python
from agent_apis.observation import normalized_features
from agent_apis.observation_dataset import iter_observed_samples

for row in iter_observed_samples(
        'data/jsettlers-corpora/downsampled', split='train', view='complete'):
    inputs = normalized_features(row['features'])
    legal = row['legal_action_ids']
    action = row['action_id']
    win_probability_target = row['value_target']  # 0 or 1
```

Use `view='retained'` to preserve the exact original retained decision indices.
Both views replay the full logs and encode only pre-action observations through
the live path. The complete view includes every omitted decision. The loader
streams one game at a time, checks shard identity and final replay state, and
returns source checksum, feature-schema version and split. It returns neither
Game nor privileged checkpoints, seed, ordered deck or future chance events.
Consume the iterator fully to perform its end-of-game replay check. Derived
rows are streamed, with no new corpus copies or changes to original shards.

Retained rows receive no tenfold sampling correction and are not treated as
adjacent transitions. Training must shuffle game order and use a bounded state
shuffle buffer in phase 5. The loader's stable order is for reproducible audit.
Select only `features` as model inputs; IDs, steps, labels and split metadata are
not model inputs. Existing `iter_samples` still reads legacy schema 2; use the
new loader for schema 3.

## Audit and field coverage

Run `.conda/bin/python -m agent_apis.observation_dataset audit` for the full gate.
`--limit-games N` is a smoke check per corpus and can never satisfy the gate.
Reports use the phase-one run-record schema and ignored `runs/catan-ai/` output.
Failures are recorded and cause a nonzero exit. Checks include:

- Every compressed shard checksum, manifest totals and shared base/action order.
- Unique game IDs/seeds across all corpus/split combinations and the original
  seed-hash split assignment.
- Deterministic full action replay, every legal demonstration and final state.
- Both ledgers against both authoritative hands at every state, including the
  terminal state, with exact fresh recovery and conservation.
- Every retained base feature, target, mask, outcome and observation field.
- Exact full schema-3 equality between saved-observation reconstruction and the
  live encoder on every retained decision.
- Corpus/split/phase/action-family/prompt counts in both data views.

Original numeric fields missing exact opponent resources, actor/turn/phase/rule
context and card age/playability are supplied by the new fields. The audit also
checks saved fields not used as numeric inputs. Public board caches, connected
components, buildable-node caches and road-color bookkeeping are redundant with
the explicit board/ownership fields and remain outside the vector. Current
trade/acceptee fields are excluded under the no-player-trading scope. Raw past
roll labels, absolute action index/turn count, game identity and random seed are
not numeric features. Complete normalized purchase/play constraints remain
available to the belief layer. Opponent private development types/age flags,
hidden deck composition/order, RNG state, future actions and outcomes are
deliberately omitted, not inferred from authoritative state.
