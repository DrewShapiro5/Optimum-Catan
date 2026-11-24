# Phase-two results, November 4, 2026

The section-2 completion gate passed on all 1,267 recorded games. No model was
trained. Original schema-2 shards were not modified. Both retained and complete
decision views now stream through the same schema-3 observation-only encoder.
See [the observation contract](OBSERVATIONS.md) for field coverage, resource
conservation, card knowledge, normalization and loader examples.

The full audit took 358.98 seconds. Its machine-readable
[run record](../../runs/catan-ai/phase2-audit-5865bc4655cb-seed11-20261104T212247282403Z/run.json)
contains source hashes, shard checksums, complete split lists and counts by
corpus/split/action family/phase/prompt. It validated against the run-record
schema with `gate_passed: true` and an empty failures list.

Configuration SHA-256:
`5865bc4655cbe31f67cba3269e0b805ce306c4812b27c53c6ce203997ca5be67`.
The implemented encoder, replay/audit code, tests, feature schema and configuration
were checked against the source hashes captured at audit start.

| Check | Verified | Mismatches |
| --- | ---: | ---: |
| Complete legal demonstration targets and deterministic replay | 489,337 | 0 |
| Ledger snapshots, both seats, including initial and terminal states | 981,208 | 0 |
| Fresh recovery versus continuously updated ledger | 981,208 | 0 |
| Retained original 633-feature vectors and legal masks/targets | 162,269 | 0 |
| Retained full 681-feature live/replay encoder comparisons | 162,269 | 0 |
| Saved observation fields and retained outcome conversions | 162,269 rows | 0 |
| Cross-split/corpus game ID overlap | 1,267 unique games | 0 |
| Cross-split/corpus seed overlap | 1,267 unique seeds | 0 |

Each ledger snapshot compares both reconstructed hands to authoritative hands
inside the audit layer, for 1,962,416 exact hand comparisons. Every omitted
decision is replayed and checked. Resource nonnegativity, public opponent hand
size and per-resource conservation hold throughout. Recovery uses the permitted
public-bank/own-hand sufficient snapshot, including when theft results are
redacted; it never reads a hidden hand in the encoder or ledger.

Separate checks of every header also verified completed-game winning scores,
manifest winner/game ID/seed/action-count/retained-count agreement. All
compressed shard checksums and manifest totals passed.

| Corpus | Split | Games | Complete decisions | Retained rows |
| --- | --- | ---: | ---: | ---: |
| full-250 | train | 220 | 85,773 | 85,773 |
| full-250 | validation | 17 | 6,864 | 6,864 |
| full-250 | test | 13 | 5,129 | 5,129 |
| downsampled | train | 898 | 345,697 | 56,950 |
| downsampled | validation | 61 | 23,866 | 3,905 |
| downsampled | test | 58 | 22,008 | 3,648 |
| Total | | 1,267 | 489,337 | 162,269 |

The complete view adds 327,068 omitted decisions from downsampled games. It is
the uniform-decision control; no tenfold reweighting of retained rows is used.
The original corpus is excluded from the bundle, avoiding duplicate games.

| Phase | Complete decisions | Retained rows |
| --- | ---: | ---: |
| Setup | 10,136 | 10,136 |
| Middle | 461,656 | 140,338 |
| Endgame: either public score at least 12 | 17,545 | 11,795 |

| Action family | Complete decisions | Retained rows |
| --- | ---: | ---: |
| BUILD_SETTLEMENT | 11,666 | 7,725 |
| BUILD_ROAD | 20,524 | 10,561 |
| BUILD_CITY | 6,231 | 2,397 |
| ROLL | 159,660 | 49,798 |
| END_TURN | 158,401 | 49,933 |
| MOVE_ROBBER | 40,329 | 12,222 |
| BUY_DEVELOPMENT_CARD | 25,598 | 7,457 |
| MARITIME_TRADE | 44,772 | 15,504 |
| DISCARD_RESOURCE | 4,520 | 1,402 |
| PLAY_KNIGHT_CARD | 13,532 | 3,995 |
| PLAY_ROAD_BUILDING | 2,100 | 632 |
| PLAY_YEAR_OF_PLENTY | 1,945 | 611 |
| PLAY_MONOPOLY | 59 | 32 |

All relevant resource-transfer action families occur in the audited data.
Ten focused tests passed: the four phase-one tests and six observation tests
covering exact resource transfers/recovery, invalid rules/counts, hidden-field
invariance, development-card ages/private-draw redaction, categorical dice/fixed
normalization, and retained/complete streaming through replay. Testing caught
and fixed missing empty building-category lists in fresh live observations.

All original retained numeric fields matched; no saved observation field
mismatches remain. Missing phase/resource/card-age context is added in schema 3.
The observation contract documents deliberately excluded redundant fields and
hidden information. Later architecture implementations must use this shared
schema and boundary. Phase 3's benchmark harness and frozen evaluation protocol
remain pending.
