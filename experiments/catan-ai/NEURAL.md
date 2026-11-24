Current execution (November 17, 2026): **visible-point friendly robber and
private human-vs-bot browser display fixed and live**. Training continues in
`runs/catan-ai/fresh-xdcnn-visible/`. Audit old expert journals under the corrected
rule, retain only exact legal replays with original splits, replace rejected
games to restore 2,000 completed games, then train a fresh W4 XD-CNN. Do not mix
old illegal-rule trajectories or trained weights into the corrected lineage.
Use `.conda/bin/python -m agent_apis.neural.fresh_cohorts --hours 3`.
Progress: `runs/catan-ai/fresh-xdcnn-visible-progress.json`.
Details: `experiments/catan-ai/VISIBLE_ROBBER_PRIVACY_FIX.md`.
The human hand stays displayed during bot search; opponent cards stay private.
User diagnostic observations: development-card buying, roads without settlements,
and occasional poor starting placement. No heuristic policy changes made.

Current execution (November 17, 2026): **fresh XD-CNN restart with large
cohorts**. Do not resume the old eight-game iterations. Initialize entirely new
XD-CNN weights, generate 2,000 new completed compact friendly-robber native expert
games, then train from scratch. Subsequent self-play cohorts also require 2,000
new completed games before updates. At most 64 seeded positions per game are
reconstructed into RAM; updates sample W4 minibatches rather than sweep every
position. All games/actions/chance events remain compressed and replayable.
Select the best validation checkpoint instead of blindly taking the final one.
Historical data and models are retained but not imported into this lineage.
Run/resume: `.conda/bin/python -m agent_apis.neural.fresh_cohorts --hours 3`.
Progress: `runs/catan-ai/fresh-xdcnn-progress.json`.
Procedure: `experiments/catan-ai/FRESH_XDCNN.md`. No automatic deployment.

Current status (November 17, 2026): **three-hour compact W4 window reviewed;
left paused**. Four completed iterations: 32 trained new games, 4,000 updates
and 80 pure-policy evaluation games. Three further completed collection games
and one interrupted prefix are preserved. Latest friendly-validation NLL and
Brier are worse than at window start, and small paired policy results do not
establish improvement. No further unchanged training window is justified.
All 116 compact journals replay-verified; total journal storage 426,308 bytes.
W4 remains the training default. No playing checkpoint deployment/change.
Review: `runs/catan-ai/compact-w4-review.md`. One-time review automation removed.

Current execution (November 17, 2026): **three-hour compact W4 iteration
window is running**. Deadline/review: November 17 at 06:11 America/Denver
(13:11 UTC). Collect eight new completed friendly-robber search games per round
at 128 simulations, train 1,000 W4 updates on cumulative completed self-play and
fresh friendly expert data, then run bounded pure-policy paired tuning checks.
New games persist gzip action/event journals and sparse visits only; features
are replay-derived in RAM. Source archive and training checkpoints are retained.
No per-move board snapshots or feature arrays are saved for newly generated games.
An app heartbeat schedules one research review at the deadline to decide whether
another bounded window is justified. The controller pauses itself even if the
review is delayed. No automatic playing-checkpoint deployment.
Procedure: `experiments/catan-ai/COMPACT_W4.md`.
Progress: `runs/catan-ai/compact-w4-progress.json`; log: `runs/catan-ai/compact-w4.log`.

Current status (November 17, 2026): **friendly-robber W4 workflow complete**.
250 native expert games (97,383 decisions), 1,000 W4 training updates, and all
40 tuning evaluations finished. Candidate won 4/10 vs JSettlers (parent 3/10)
and split 5/10 vs parent. No demonstrated gain: paired interval includes zero.
Friendly-validation policy NLL improved; final Brier was slightly worse despite
an early improvement. Timing gates failed. No playing-model promotion or new
background queue. W4 remains the explicit default for future training.
Results: `experiments/catan-ai/FRIENDLY_W4_RESULTS.md`.

Current execution (November 17, 2026): **W4 is the default for all new
training by explicit user instruction.** A new 250-completed-game native
JSettlers SMART-versus-SMART corpus is being generated with friendly robber.
After generation and replay/checksum verification, continue the latest
self-play checkpoint for 1,000 W4 updates with 50% fresh friendly-robber expert
training decisions and 50% the preserved 100-game self-play cohort. W4 includes
inverse-probability value correction. Fresh friendly-robber validation is
separate from training. No historical corpus/results are rewritten, no old
standard-robber expert rows enter this continuation, and no automatic deployment.
Procedure: `experiments/catan-ai/FRIENDLY_W4.md`.
Progress: `runs/catan-ai/friendly-w4-progress.json`; log: `runs/catan-ai/friendly-w4.log`.

Earlier execution and selection history follows.

Current status (November 16, 2026): **smaller self-play round completed**.
All 100 collection games, 1,000 training updates and 40 tuning evaluation games
are complete. Candidate won 3/10 against JSettlers versus the parent's 2/10;
paired uncertainty does not establish improvement. Directly against the parent:
4 wins, 5 losses, 1 cutoff. Original-validation calibration worsened and timing
gates failed. No model promotion, automatic next round, or background queue.
Results: `experiments/catan-ai/PHASE9_SELFPLAY_SMALL_ROUND_RESULTS.md`.

Current execution (November 16, 2026): **smaller-round evaluation resumed**
at the user's request. Collection and all 1,000 training updates are already
complete. Reuse the 25 committed evaluation games; finish the remaining 15.
The original full self-play queue remains paused. No automatic deployment.
Progress: `runs/catan-ai/selfplay-small-round-progress.json`.

Current status (November 14, 2026): **paused by user to save power for DJing**.
The smaller round completed collection (100 games, 28,299 examples) and all
1,000 training updates. Evaluation is paused with 25 completed games preserved.
The background process stopped and released the benchmark lock. No automatic
restart or default-model change. Resume only when the user requests it via
`.conda/bin/python -m agent_apis.neural.selfplay_small_round`.

Current execution (November 14, 2026): the **smaller first self-play round is running**.
Reuse 27 completed games and collect 73 additional games at 128 simulations per
searched move, then train 1,000 updates and run bounded paired tuning checks.
Friendly robber remains enabled. The original full queue remains paused.
Live progress: `runs/catan-ai/selfplay-small-round-progress.json`.
Log: `runs/catan-ai/selfplay-small-round.log`.

Current status (November 13, 2026): **self-play paused by user**. The full
2,000-game-per-iteration queue has stopped. Preserve and reuse its 27 completed
games and 5,908 examples. A **single 100-game first round** is prepared: collect
73 additional games at 128 simulations per searched move, then train 1,000
updates and run bounded paired tuning checks. Existing games retain their
512-simulation provenance. Friendly robber stays enabled. The smaller round
has passed preflight but has **not started**; no automatic scale-up/deployment.
Start/resume it only when requested, with
`.conda/bin/python -m agent_apis.neural.selfplay_small_round`.
See [the smaller-round procedure](PHASE9_SELFPLAY_SMALL_ROUND.md).

Earlier execution status and historical scope follow.

Current execution (November 13, 2026): **search self-play started**, XD-CNN
D2/W0 seed11 only, with **friendly robber enabled by default** for new app games
and this procedure. Three iterations each collect 2,000 completed games at 512
new simulations per non-forced learned decision, then train 10,000 updates.
The 50/25/25 current/historical/JSettlers mixture, root exploration, subtree
reuse, resumable training, and noise-free paired tuning checks are implemented.
Historical corpora/results retain their original rules; fresh friendly-robber
parent evaluations provide the controls. No automatic model deployment.
See [self-play procedure](PHASE9_SELFPLAY.md).
Live status: `runs/catan-ai/selfplay-progress.json`; log: `runs/catan-ai/selfplay.log`.

Previous completed work and historical scope amendments follow.

Current work (November 13, 2026): cross-decision subtree/state reuse implemented
and tested before any further self-play training. Hidden-card posteriors refresh
on each real decision; belief-weighted visits reset conservatively. Reuse is
opt-in with `search.reuse_subtree=true`; default remains unchanged pending
stronger performance evidence. See [implementation and audit](SUBTREE_REUSE.md).
No training or new long evaluation queue was started for this optimization.

Latest status: the bounded phase-nine teacher-aggregation pilot completed.
Twenty collection games, 4,213 new examples, 1,000 fine-tuning updates, and ten
tuning validation games. Common supervised validation retained the original
parent weights; no trained checkpoint was promoted or deployed. No large queue
is running. Updated final.pt and all learner-state data remain available for
subsequent investigation. See [pilot results](PHASE9_PILOT_RESULTS.md). Full phase 9 is incomplete.

Current execution: bounded phase-9 **teacher data aggregation** pilot, XD-CNN
only. Twenty learner-versus-JSettlers generation attempts, 1,000 fine-tuning
updates with 50/50 original/new training data, then 10 paired tuning validation
games. No Transformer/random-player runs, no automatic large self-play queue
or default deployment. The parent policy supplies learner actions; native
SMART supplies separately verified labels at learner-reached states.
See [the pilot protocol](PHASE9_PILOT.md).

Current execution amendment (November 12, 2026): phase 7 closed early at the
user request; remaining simulation-budget and confirmation matches are abandoned.
Use existing XD-CNN D2/W0 results, retain c=1.5/C0 provisionally, and do not
claim a demonstrated search gain. Phase 8 now uses a bounded 12-position
correctness/throughput and value/opponent diagnostic; the exhaustive chance
allocation sweep is deferred. No Transformer or random-player runs, no automatic
large match or self-play queue. See [PHASE7_EARLY_RESULTS.md](PHASE7_EARLY_RESULTS.md) for the preserved interim evidence.

Opponent amendment (November 11, 2026): **JSettlers only** from now on.
The user removed random-player testing. Completed JSettlers games are reused;
random-player history is retained but excluded from active scheduling and gates.
This halves phase-seven new-game limits to 400 screening games plus at most
100 additional confirmation games. Search settings and training are unchanged.

Current queue: **phase 7**, XD-CNN D2/W0 only. Run
`python -m agent_apis.neural.phase7`; see [PHASE7.md](PHASE7.md).
Phase-six XD-CNN work is complete; the remaining Transformer work was retired
by the user. Historical workflows below do not define the active queue.

Current scope amendment (November 11, 2026): **XD-CNN only**. The user retired
Transformer and stopped its remaining tests. All XD-CNN phase-six confirmation
runs are complete; see [phase-six results](PHASE6_RESULTS.md). The original two-architecture
comparison remains incomplete. W4 is promising, but its paired hierarchical
improvement interval touches zero, so the conservative recipe remains D2/W0.
Future experiment work uses XD-CNN; do not restart the historical Transformer queue.

Current execution scope (November 7, 2026): phase 6 continues with **XD-CNN and
Transformer only**, using **random player and JSettlers SMART** as validation
opponents. See [PHASE6.md](PHASE6.md) for the executable corpus/weighting queue.
The four-model/full-confirmation sequence below remains the historical phase-5
specification; do not launch it as the current default workflow.

# Phase 5 neural encoders and experiment runner

All four seed-11 training runs are complete. At the user’s request, the remaining
evaluation has resumed with per-run futility checks and reuse of the historical
MLP evidence. See [the loss diagnosis](PHASE5_DIAGNOSIS.md) and the versioned
[futility amendment](FUTILITY.md). Completion requires the results report.
Training and match records, rather than implementation
alone, determine completion. `PHASE5_RESULTS.json` and `PHASE5_RESULTS.md` are
written only after all required runs and comparisons complete successfully.
The complete-comparison runner’s live, atomically updated state is
`runs/catan-ai/phase5-evaluation.json`; its log is
`runs/catan-ai/phase5-evaluation.log`. The progress JSON is an interim snapshot;
`PHASE5_RESULTS.json` is the authoritative completed report.

All four models consume the same audited schema-3 observation (681 raw features)
and emit 332 globally masked policy logits plus a sigmoid win probability.
The MLP has three 256-wide GELU layers; XD-CNN has six cross-dimensional residual
blocks (64 spatial / 128 scalar channels, 5×3 convolutions); H-GNN has six typed
128-wide message blocks; Transformer has four pre-normalized 128-wide blocks,
four heads and width-512 feed-forward layers. Structured encoders share location
and scalar action heads. Parameter counts are 546,893 / 1,268,390 / 3,782,374 /
911,418 respectively; capacities are disclosed rather than assumed equal.

The 148-entity inventory comprises 19 tiles, 54 intersections, 72 road locations,
two players and global context. Dice codes are one-hot encoded while preserving
tile-number pairing and production probabilities. The original nine port slots
are not geometrically closed under all board symmetries: shared input packing
converts them losslessly to typed edge ports and adjacent-intersection features.
This changes no available information and is identical for every architecture.
The MLP's packed flat input has 1,287 values. Twelve D6 transforms preserve legal
action masks, labels and absolute victim colors. An independent physical-engine
audit checks 144 phase/position/symmetry cases, including rotated ports and dice.
See `phase5-augmentation-audit.json` and `feature-schema-v3.json`.

The immutable source corpus is replayed into a checksummed, ignored complete-D2
cache: 489,337 decisions from 1,267 games, including forced moves. Training has
431,470 decisions / 1,118 games; validation 30,730 decisions; test 27,137.
The bounded shuffle visits every training decision once per epoch. W0 means no
sampling or loss reweighting. Policy cross entropy is averaged over non-forced
rows; value BCE is averaged over every row, with coefficients 1 and 1. Use the
same 20,000 AdamW updates, batch 256, learning rate 3e-4, decay 1e-4, clipping 1,
500 warmup updates and cosine schedule to 3e-5 for every run. Full validation
runs every 1,000 updates; earliest minimum policy-NLL + value-BCE selects best.pt.
The final checkpoint and full training-resume state are preserved independently.

`phase5-devices.json` measures real forward/backward/AdamW and batch-1 / batch-32
inference with input packing and transfers. It selects MPS training for all four,
CPU inference for MLP/XD-CNN/H-GNN and MPS inference for Transformer. MPS requires
running outside the macOS sandbox on this machine; there is no silent fallback.
The runtime uses six CPU threads and float32. Each run records source/config,
corpus/cache/schema/topology hashes, named random streams, hardware, dependency
versions, actual game/phase/family consumption, validation history and test
policy/value/calibration metrics. Test results never select checkpoints.

Run the complete resumable sequence with the repository Python:

```sh
.conda/bin/python -m agent_apis.neural.evaluation \
  --cache runs/catan-ai/d2-cache-89b777ce01b9ebd7
```

The complete-comparison runner ends a futile model/mode run without extending
it, then evaluates the remaining architectures. It tests search even when the
same model’s standalone policy was futile. Additional seeds are limited to
qualifying architectures; an all-futile comparison completes with no winner.
The original `neural.pipeline` entry point retains its halt-for-diagnosis behavior.
The runner coordinates with the initial training queue using the exclusive
benchmark lock. It reuses completed checkpoints and exact configuration-matched
benchmark logs, and resumes interrupted training checkpoints. Changing relevant
implementation hashes produces distinct match configurations; changes during a
running pipeline stop it. A failed matching run requires investigation, not a
reroll or fallback. Benchmark execution is serial, without overlapping training.

All four seed-11 models receive pure-policy and 500 ms search matches against
native JSettlers SMART on the frozen tuning seeds. A 40-game smoke prefix extends
to 200 games by verifying and reusing its checksummed game/timing logs. Every
search model uses the same frozen seed-11 MLP opponent checkpoint. Feasible means
zero errors/illegal actions, cutoff rate at most 1%, and warm non-forced p95 at
most 525 ms. Timing failures are disclosed and exclude promotion; errors or
excessive cutoffs stop the pipeline for investigation.

Before seeing those match results, the ranking rule is fixed: each architecture
gets its highest feasible match score across policy and search; select the top
two, with fewer parameters breaking exact ties. Train these two on seeds 23 and
47 for all 20,000 updates and evaluate each in both modes on the same 100 pairs.
Also compare their seed-11 final and validation-selected checkpoints in both
modes on 100 tuning pairs. Preserve the original validation criterion for this
baseline comparison and report whether tuning strength agrees with it.

Report each training seed separately and bootstrap board/seat pairs, then
training seeds plus shared board pairs for aggregate results. Provisional
selection resolves overlapping aggregate intervals by fewer parameters; search
is selected only when its paired three-seed improvement interval excludes zero.
An installed default remains explicitly provisional pending phase-10 final
confirmation. `NEURAL_POLICY` and `NEURAL_SEARCH` expose the trained checkpoints
through both the existing Python player and HTTP provider interfaces.

Final evaluation seeds remain unused. A single training seed cannot establish
confirmation, and no win claim over JSettlers is made unless the lower 95% match
score bound exceeds 50% with all failure/timing/cutoff gates disclosed.

The long-history posterior recursion fix and corrected search rerun are described
in [PHASE5_SEARCH_FIX.md](PHASE5_SEARCH_FIX.md). Historical failed search runs
remain archived and cannot qualify for promotion.

Search optimization revision 1 removes the own-action cap and adds cached dice
equivalence, root-world reuse and batched threaded search. See
[SEARCH_OPTIMIZATION.md](SEARCH_OPTIMIZATION.md). Full-game evaluation remains
interrupted while the optimized implementation is validated separately.
