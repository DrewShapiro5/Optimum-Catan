# Phase-four stochastic search contract

The provider `STOCHASTIC_REFERENCE` is a diagnostic implementation for the fixed
two-player BASE/15-point/discard-nine rules. It contains no learned model and
does not claim competitive strength. `phase4.json` identifies the correctness
experiment. The phase-five D2/W0 MLP will replace the frozen heuristic opponent;
the root evaluator is independently injectable for the architecture comparison.

## Observation boundary and sampled worlds

Search accepts only the audited schema-3 permitted observation. It never takes
or copies the authoritative Game. The exact resource ledger reconstructs both
hands from public bank counts and the root's own snapshot. Public map, pieces,
turn/actor, rule and phase fields initialize a fresh Catanatron simulation.
Own development cards remain known; opponent cards and remaining composition
come from the development belief. The remaining deck is represented in a
deterministic arbitrary order, which is never used for sampling a draw.

Every simulation begins by sampling one feasible development-card world at the
root. That world persists through the simulation. No authoritative future RNG,
ordered deck or hidden hand is consulted. The simulated engine's RNG is a
`NoFutureRandom`: an implicit roll, theft or other random draw raises an error.
Every stochastic transition must receive an explicit engine `ActionRecord`.
The framework shares RNG references on copies; the guard therefore applies to
copies too. World, chance and opponent-policy RNGs are independent named streams.

Development beliefs use the finite initial supply: 14 knights, two each Year of
Plenty/Monopoly/Road Building and five victory-point cards. A dynamic program
processes the permitted purchase/play/END_TURN prefix in chronological order.
Its states contain old and newly purchased opponent holdings. Remaining deck
counts follow from initial supply, all own purchases, opponent public plays
and these holdings. Unknown purchases branch with their remaining-count draw
probabilities. Known own purchases contribute their likelihood too. A play
requires a matching old card; END_TURN ages new cards. Backward continuation
mass weights forward posterior sampling of the entire past allocation.

The sampler checks own identities, holding totals, card-age flags, public play
counts, one-card-per-turn restrictions and remaining deck total. It replays
public pieces and knights using the engine's Board rules to derive historical
public scores. Worlds that would already have ended through hidden VP cards
are excluded, including before a later loss of road/army points. Inconsistent
observations raise errors; there is no permissive fallback sampler.

This posterior conditions on draw probabilities and observable rule/history
constraints. It does not weight historical strategic choices with an opponent
action-likelihood model. That remains a separate later experiment. It is not
independent sampling of current hidden card types. Sampled private purchase
identities reconstruct the opponent's own knowledge within each simulated world.

## Decision, action and chance structure

Root-player decision nodes use PUCT with `c_puct=1.5`. Every legal action keeps
its own prior, visit count and total root utility. Priors receive one global
normalization over legal action IDs. The visit policy uses mean value plus the
prior exploration term. Unvisited edges begin with neutral value 0.5.

Final action selection uses most visits, then highest mean value, then lowest
stable action ID. Exploration bonuses never select the played action. When a
time budget expires before any simulation, the same rule deterministically
breaks zero-visit ties; it does not invent evidence. Forced root actions return
immediately without preparing a belief or running simulations.

Each action edge owns per-world chance distributions and lazily retained chance
children. Action-edge visits/values and chance-child visits/values are separate.
Repeated visits accumulate evidence. Temporary Games are used to identify
equivalence classes before selection, but only sampled children remain in the
tree. Both merged and unmerged modes remain executable.

Elementary chance outcomes are all 36 ordered dice pairs at 1/36, positive
resource types at their exact victim-hand frequencies for theft, and positive
card types at their remaining sampled-world deck frequencies for draws.
Deterministic actions have one outcome. Every distribution must sum to one
within 1e-12 and retain every positive-probability outcome in exactly one class.

Within a given parent/action/world, outcomes with equal successor facts are
grouped and their probabilities added. Class diagnostics retain constituent
results and probabilities. Dice totals with identical effects can merge; raw
past roll labels are not policy inputs. Seven/discard/robber transitions remain
distinct. Normalized bank-shortage resource effects also remain distinct, even
when denied production happens to leave equal hands. Root-visible history
retains those public shortage effects.

The effective chance key is scoped by the initial root information, subsequent
root-visible action/observation history and action identity, then by a full
sampled-world state fingerprint. It includes resource counts, bank, board
ownership/robber, player pieces/scores/flags, turn/actor, phases/discards/free
roads, development holdings/age constraints, sampled remaining composition and
both players' normalized card knowledge. RNG state, object IDs and future deck
order are excluded. Public-history context fixes facts such as the most recent
initial-settlement anchor. This is not a global cross-history transposition key;
unscoped transposition reuse requires its own later correctness work.

World fingerprints retain joint private purchase identities and public plays.
Worlds are never merged just because their current public appearance is equal.
Two actions also never share their root action statistics, even if their
successors happen to match.

## Root information and frozen opponent

The decision tree is indexed by root action/observation history. Observation
components use the common features, legal root actions, normalized permitted
purchase/play knowledge and announced terminal result. Raw dice labels are
excluded. The callback boundary explicitly replaces every historical ROLL value
and result with null before any evaluator or opponent policy receives it;
ignoring labels is enforced by the interface. Opponent-only private draw types cannot create distinct root decision
nodes; root-visible own draws do. Private world keys exist below action edges
for chance bookkeeping, not to select a different root policy directly.

Opponent actions are sampled from a frozen heuristic policy using only that
opponent's own redacted observation. The root player's private card identities
and sampled remaining composition/order never enter that policy. A non-acting
root view also omits the opponent's private legal action list. The phase-four
heuristic uses public map production and fixed action-family weights; its
version is `public-production-build-prior-v1`. No opponent search is performed.

At each newly expanded root frontier, the deterministic reference evaluator
uses the root's permitted score/resource features and returns a win probability.
Its version is `score-resource-logistic-v1`; it is not a calibrated learned
value. Continuations across opponent-only phases reach a root frontier or
terminal result. Backups always retain root utility. The explicit conversion
from another actor's win probability is `1-p` in this two-player setting; values
are never flipped once per action. Exact terminal utility is 1/0 for root
win/loss. Diagnostic depth/action caps return unresolved value 0.5 and are
counted, never used as observed training outcomes.

This is POMCP-style root sampling/history sharing for a fixed opponent policy,
not ordinary perfect-information search on a revealed sampled state. It does
not establish optimal private-information play or implement multi-observer
ISMCTS. Architecture/opponent-model comparisons remain subsequent experiments.

## Run and validate

```sh
.conda/bin/python -m unittest -v tests.test_stochastic_search
.conda/bin/python -m agent_apis.search_validation freeze
.conda/bin/python -m agent_apis.search_validation audit
```

`freeze` deterministically selects and verifies the checked-in suite and refuses
to replace different contents. Its 1,000 positions come only from the existing
game-level validation split: 200 setup, 300 middle, 200 endgame, 100 robber,
100 discard and 100 free-road positions. Selection uses the lowest SHA-256 ranks
within predeclared strata; every reference pins its source shard checksum and
pre-action step. No terminal label or future action is a search input.

The audit runs the search unit tests, exact weighted one-step merged/unmerged
comparisons at every position, authoritative-future perturbation checks, and
both complete search variants with 16 simulations per non-forced root. It logs
chosen actions, priors/visits/values, raw/class counts, memory, depth, preparation
cost, DP-state count, simulations/second and failures. Root depth is capped at
four root actions and simulations at 128 total actions in this validation
configuration. Turn-depth telemetry uses the framework's `num_turns` delta,
including its setup-phase turn convention.

Every position/variant records cumulative process peak RSS, including interpreter
and validation overhead. It is an upper-bound telemetry value, not an isolated
tree size or a basis for attributing memory savings to a variant. The first
non-forced frozen position encountered in each stratum also gets a `tracemalloc`
probe for each variant, including belief preparation. Throughput excludes these
traced probes. These are not deployment latencies or 500 ms match results.
No minimum speedup is assumed. Per-position
artifacts allow stratified analysis of both variants and action differences.

Frequency tests make 100,000 draws from each representative dice, theft and
development distribution. Before drawing, each class gets simultaneous absolute
bound `sqrt(log(2*K/alpha)/(2*N))`, with total class count K and familywise
alpha 0.001, by Hoeffding plus a union bound. This controls the whole tested
multinomial family rather than checking arbitrary percentage tolerances.
All weighted one-step values must agree within 1e-8.

Outputs use the shared run-record schema under ignored `runs/catan-ai/` with
source/data/configuration/suite hashes, test logs and `positions.jsonl`. A
`--limit N` smoke run is explicitly partial and cannot satisfy the 1,000-position
gate. Errors are preserved and produce a failing exit. The suite and phase-three
benchmark share an exclusive lock to avoid competing timing jobs.

Python/HTTP callers can select the registered `STOCHASTIC_REFERENCE` provider
through the existing APIs. Default standalone decisions use a 500 ms admission
deadline with a reserve of max(20 ms, 5%); preparation and pending work still
count toward measured time. It is a correctness/reference provider, not a
certified time-budget winner. Benchmark calls must explicitly choose `time` or
`simulations` mode in their resolved configuration. The injectable evaluator
and frozen opponent interfaces are ready for phase-five neural components.
