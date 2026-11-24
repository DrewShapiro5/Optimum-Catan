# Phase 9 search self-play

Authorized November 13, 2026 by the user's request to begin self-play training.
The subsequent rule amendment makes **friendly robber the default** for new
app games and this experiment's collection and evaluation. Existing games,
expert corpora, checkpoints, and historical results retain their original rules.

Run/resume: `.conda/bin/python -m agent_apis.neural.selfplay`.
Protocol: [phase9-selfplay.json](phase9-selfplay.json).
Progress: `runs/catan-ai/selfplay-progress.json`.
Log: `runs/catan-ai/selfplay.log`.
Artifacts: `runs/catan-ai/phase9-selfplay/`.

Use XD-CNN D2/W0 seed11's phase-six validation-selected parent; the completed
teacher-aggregation pilot selected those same weights. Run three iterations,
each collecting 2,000 **completed** games and then training 10,000 updates.
Each shuffled block of four completed games has two current-agent opponents,
one historical checkpoint opponent, and one native JSettlers SMART opponent.
Initially the only historical checkpoint is the original parent. Add each
generation's final checkpoint to that frozen historical pool. A cut-off game
retries its mixture slot on fresh seeds; stop after 4,000 attempts per iteration
if the completed cohort cannot be filled. Report cutoffs and exclude their
entire trajectories from training. API/legality failures stop for investigation.
Generation seeds are training-only and checked against corpus, tuning, final,
and prior teacher-pilot seeds. Final evaluation seeds remain unused.

Every non-forced learned-agent action uses 512 **new** simulations. Keep the
existing root-player search, frozen seed11 D2/W0 MLP simulated opponent, c=1.5,
true-probability merged chance classes, and 128-action simulation cap.
Cross-decision subtree/state reuse is enabled per seat; current development-card
beliefs refresh on every decision and uncertain retained statistics reset.
The engine's friendly robber uses actual scores below three, including hidden
VP cards. Its observed legal mask and past permitted blocking moves therefore
constrain hidden VP counts. The belief dynamic program conditions on those
publicly supplied rule constraints; it never reads the opponent's actual hand.
Retained visits are reported separately from the new simulation budget.
No performance gain from reuse is assumed.

Exploration mixes 25% root Dirichlet noise, with per-action concentration 10/A,
into freshly evaluated current-root priors. Deeper priors and chance
probabilities receive no noise. Sample actions from visits with temperature 1
during setup/the first 20 turns, then .25. Record the action actually sampled
for the next reuse match. Training targets use the untempered normalized root
visit distribution. Only moves actually searched by the current learner supply
new targets; both seats count in current/current games. Historical opponents
supply trajectories but no new current-agent labels. Forced live moves bypass
search and supply no new rows. Outcomes are attached after actual completion,
relative to each recorded decision actor. Search receives only the audited
perspective observation, never the authoritative Game or future RNG/deck.

Train from each current checkpoint with AdamW and its supervised settings,
batch 256, 500-update warmup and restarted cosine schedule, audited D6 symmetry,
soft policy cross-entropy plus actual-outcome value BCE. Expert/new fractions
are 50/50, 25/75, and 10/90; sample uniformly over original complete D2 training
decisions and uniformly over new searched decisions across all completed
generations. The shard sampler bounds memory rather than loading the entire
corpus. New friendly-robber observations encode RULE_FRIENDLY_ROBBER=1 in the
existing schema slot; original expert rows correctly keep 0. The rule change
is an explicit transfer experiment, not a historical like-for-like result.

The compatible expert cache preserves all original array checksums. Its
provenance audit verifies exact AST equivalence of the standard-robber encoder
after reversing the two rule-flag changes, all other cache inputs, and replay
equivalence over 2,561 decisions from six corpus games. Audit and prior source
copies are under `runs/catan-ai/friendly-robber-migration/`; original cache and
source corpora remain immutable.

Common D2 validation is diagnostic every 1,000 updates. The predeclared final
update continues the experimental training lineage; supervised imitation loss
does not select it back to the parent. Preserve optimizer, sampling,
augmentation, CPU and MPS RNG state every 100 updates and on caught exceptions.
Training resumes from that state. Generation commits a checksummed shard and
record per game; an interrupted uncommitted game restarts at its same seed.
Per-decision telemetry and live progress expose work before a game completes.
An exclusive benchmark lock serializes training, collection and measurement.

After each iteration, run noise-free 500-ms search on the first 25 frozen
tuning board/seat pairs (50 games) for each of four conditions: parent versus
JSettlers, candidate versus JSettlers, candidate versus frozen prior checkpoint,
and candidate versus JSettlers with a uniform historical-policy mixture replacing
the frozen imitation search opponent. This last condition is a separate
simulated-opponent ablation; generation retains the original search baseline.
All these matches use friendly robber, including fresh parent measurements;
do not reuse standard-robber results as their controls. Report paired bootstrap
differences, cutoffs, errors, latency, common-validation Brier change, and gain
per generation hour, training hour and completed game. Completed evaluation
games resume individually. Positive paired strength intervals, no unexplained
calibration regression, and legality/cutoff/timing gates are required for
promotion eligibility. No automatic deployment, default checkpoint changes,
fourth iteration, or scale-up occurs. Single-seed tuning remains exploratory.
