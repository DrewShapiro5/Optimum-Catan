Latest throughput investigation (November 19, 2026): **native CPU remains the recommended current backend for both model-model and learner-JSettlers collection at 128 simulations**. Current-CNN standalone turns/s: self-play 11.35 Rust vs 6.82 Python/GPU; learner-JSettlers 16.63 vs 9.42. Best hybrid short screen (3 Rust self-play + 6 Rust/JSettlers + 12 GPU self-play) reached 17.26 turns/s, but five-minute confirmations reached 12.50 and 13.20 versus 14.38 for 16 native workers (9 self-play / 7 JSettlers). Native completed 8 self-play + 15 JSettlers games versus 4 + 14 in the quieter hybrid recheck. Future combined controllers should enforce 1,000 completed games of each kind per 2,000-game update, reallocating new work at quotas; production migration/training is not launched here. Native search timers remain 83.5% ONNX inference, so the next GPU work should batch inference from native search. Twenty accepted diagnostic cases, 426 replay-verified journals, 5,715,456 simulations, 26 regression tests, source/model/journal hashes and current CPU/MPS numerical parity passed. Earlier flawed GPU-thread diagnostics are preserved and excluded. W4, optimizer, playing weights and historical runs remain unchanged. Report: `rust-catan-testing/hybrid-throughput-v2/RESULTS.md`.

Latest implementation (November 19, 2026): **learner-versus-JSettlers example-generation interface implemented and verified**. Persistent Rust-native CNN/sparse search and local inference; real SMART JSettlers opponent via its redacted interface. Only learner search visits become policy/value examples; every action/chance result is preserved and replay/checksum verified. Defaults retain the requested 16-worker native collection mode and 128 new simulations per non-forced learner decision. Durable pause/resume, pinned model/source/runtime identities, fixed whole-game splits, 2,000-game eligibility gate; no trainer or deployment changes. Verification: 22 tests; 22 completed integration games (four at 128 simulations, two resume games at four simulations, sixteen-worker smoke at four simulations). Production collection/training remains paused. Run `.conda/bin/python -m agent_apis.neural.learner_jsettlers --output runs/catan-ai/learner-jsettlers-native --games 2000 --workers 16 --simulations 128 --hours 3`. Documentation: `rust-catan-testing/learner-jsettlers-native/README.md`; verification: `runs/catan-ai/learner-jsettlers-interface-smoke/verification.json`.

Web opponent added by explicit user request (November 19, 2026): **CNN mixed cohort — search (1000 ms)**, key `CNN_MIXED_SEARCH`, pins mixed cohort-001 CNN final checkpoint SHA256 `0a6fd1181faa6ca4c1bf6cba84ba40651eb6eade9c46aa1d99ddedb112569a42`. Configuration `experiments/catan-ai/web-mixed-cnn.json`; provider `agent_apis/neural/mixed_web.py`. 1 CPU thread, subtree reuse, 1000 ms budget, same frozen simulated-opponent model as evaluation. New opponent is registered and available on local servers 8000 and 8005; 8005 reloaded with existing database. Four backend game/privacy tests passed. Training lineage/default checkpoints remain unchanged; this web exposure was manually authorized.

Latest evaluation (November 19, 2026): **mixed-trained final CNN, 1000 ms search, 10 JSettlers games**: 7 wins / 3 losses. Same five paired tuning starts; visible-point friendly robber; all journals replay/checksum verified. Mean 74.7 simulations/decision. No training or deployment. Report `runs/catan-ai/mixed-cnn-search1000-10games/RESULTS.md`.

Latest evaluation (November 19, 2026): **100 matched policy-only games per CNN checkpoint**. Incoming 40/100, mixed final 43/100, self-play-only final 38/100. Paired differences remain inconclusive. All 300 CNN journals replay/checksum verified, zero errors/cutoffs/illegal actions. Sparse remains at 20 games/branch. No training or deployment. Report `runs/catan-ai/selfplay-only-comparison/games/CNN_100_COMPARISON.md`.

Latest evaluation (November 19, 2026): **pre-mixed CNN baseline policy test expanded to 100 games** (50 paired tuning starts), 40 wins / 60 losses (40%). All games replay/checksum verified; no errors/cutoffs/illegal actions. Trained comparison branches remain at 50 games each; compare them on the shared first 25 paired starts. Report `runs/catan-ai/selfplay-only-comparison/games/CNN_BASELINE_100.md`. No training or deployment.

Latest controlled comparison (November 19, 2026): **self-play-only retraining and policy evaluation completed**. Both model types retrained from pre-mixed parents for 2,000 updates on the original 1,000 self-play games only (900/50/50). Corrected validation selects step 250; primary policy matches use equal-update final weights. CNN 50 games/condition: incoming 20/50 (40%), mixed final 22/50 (44%), self-only final 20/50 (40%). Sparse 20/condition: mixed 8/20 (40%), self-only 9/20 (45%). Paired primary differences include zero. Self-only value validation and expert imitation worsen; no evidence for dropping demonstrations based on this screen. No deployment or native Java-opponent connection. Report `runs/catan-ai/selfplay-only-comparison/RESULTS.md`.

Important diagnosis (November 19, 2026): **mixed-cohort validation metric mismatch found**. Training used soft MCTS targets correctly, but inherited validation NLL scored only their argmax. Reported regression/update-zero selection does not imply no learning. Full-target validation self-play CE improved CNN 2.2067 -> 1.2676 and sparse 2.2400 -> 1.2795, while expert imitation and value BCE worsened. Before more mixed training, use isolated soft-target validation plus per-source metrics and explicit source migration; preserve historical records. No resumed run or deployment. Diagnostics `runs/catan-ai/mixed-native-visible/diagnostics/FINDINGS.md`.

Latest status (November 19, 2026, 10:00 MST): **mixed native window including two-hour extension ended cleanly at 09:00 MST**. Mixed cohort 1 completed (1,000 JSettlers + 1,000 Python-replay-accepted native games), and both models ran 2,000 updates. Both selected update zero: CNN combined validation 0.9283 / accuracy 85.02%; sparse 1.0023 / 84.15%; incoming weights retained. Cohort 2 preserved at 1,000 expert + 120 accepted self-play games. No errors or deployment. Progress `runs/catan-ai/mixed-native-visible/progress.json`.

Current execution (November 19, 2026): **user authorized two extra hours for the mixed cohort**, extending end time from 07:00 to 09:00 MST. Existing native workers run to their original safe boundary; a bounded continuation supervisor resumes preserved progress for the remaining authorized time. No source or model changes. Extension control: `runs/catan-ai/mixed-native-visible/control/extension.json`; supervisor `control/extend.py`.

Current execution (November 19, 2026): **mixed 50/50 cohorts running until 07:00 MST**. User authorized 1,000 JSettlers + 1,000 Rust-native self-play games per 2,000-game update cohort, with 16 Rust workers and 128 simulations per searched decision. Random independent CNN/sparse model seats; fixed slot matchup/split across retries. First expert block reuses 1,000 completed expert-12 games with original splits. Both current models train W4 on the mixed cohort; sparse retains 0.001 logit penalty. Updates gated on 2,000 accepted completed games, no deployment. Progress `runs/catan-ai/mixed-native-visible/PROGRESS.md`; runner `agent_apis.neural.mixed_native_cohorts`. This replaces earlier expert-only continuation instructions for future cohorts. Details `experiments/catan-ai/MIXED_NATIVE_COHORTS.md`.

Latest evaluation (November 19, 2026): **80 JSettlers games completed**, latest cohort-11 selected models, identical ten paired tuning starts per condition. XD-CNN policy 9/20 (45%), 256-new-simulation search 11/20 (55%); sparse Transformer policy 4/20 (20%), search 11/20 (55%). Sparse search-minus-policy +35 points, paired bootstrap interval +20 to +50; CNN +10 points, interval -10 to +30. Search model difference 0 points, interval -25 to +30. Both search win-rate intervals include 50%; no confirmed superiority over JSettlers. All 80 journals replay/checksum verified, all searched decisions used 256 simulations, zero errors/cutoffs/illegal actions/native rejections. Both training runs remain paused and weights undeployed. Report: `runs/catan-ai/cohort11-cnn-sparse-policy-sim256-20each/RESULTS.md`.

Current execution (November 18, 2026): **XD-CNN cohorts plus sparse Transformer catch-up resumed for three hours, ending 23:48 MST**. Sparse Transformer starts from selected typed-Exphormer update 1750 and adds light value-logit penalty 0.001; replays original corrected expert corpus then all numbered complete 2,000-game cohorts. Unchanged CNN W4 trainer, 12 single-thread expert workers, both MPS optimizers, whole-game splits and validation selection. Dense Transformer stays paused. No deployment. Details: `experiments/catan-ai/SPARSE_TRANSFORMER_CATCHUP.md`.

Latest results (November 18, 2026): **six Transformer ablations complete, 2,000 updates each on cohort 8**. Logit penalty 0.01 reduces final value BCE 1.1079 -> 0.6201 but keeps selected update zero; dropout and +1/+2 value layers do not solve overfitting. Typed Exphormer-style local/expander/global-connector attention selects step 1750: primary combined loss 1.1666 vs dense 1.1818, but secondary cohort-7 validation worsens 1.1852 -> 1.2019 and accuracy declines. No confirmed replacement or deployment. Full report: `runs/catan-ai/transformer-value-ablations/RESULTS.md`. Production training remains paused.

Latest assessment (November 18, 2026): **cohort-8 CNN evaluated vs JSettlers on ten paired tuning starts**: policy 8/20 (40%), 1000 ms search 11/20 (55%); paired difference +15 points, 95% pair-bootstrap interval -10 to +40 points. No errors/cutoffs/illegal actions; all 40 journals replayed. Not a confirmed >50% strength result. Transformer cohort-8 audit confirms value overfitting: sampled train BCE 0.4794 -> 0.0846, held-out BCE 0.4161 -> 1.1890; policy held-out NLL slightly improves. Proposed value-path regularization ablations documented; pinned trainer unchanged. See `runs/catan-ai/cnn-cohort8-policy-search1000-20each/RESULTS.md` and `runs/catan-ai/transformer-value-audit/REPORT.md`. Both training runs remain paused.

Current execution (November 18, 2026, 15:58 MST): **both expert runs resumed for three hours, ending 18:58 MST**. CNN completed four additional expert cohorts; Transformer caught up through original corrected corpus and those same four cohorts. Cohort 5 starts this window at 1,017 completed games. On cohort-4 validation: CNN combined loss 1.0861 / accuracy 79.03%; Transformer 1.2270 / 73.50%. Transformer retained step-zero checkpoints for cohorts 3 and 4; no validation improvement from those blocks. Continue authorized W4 training and validation selection, no automatic deployment.

Current execution (November 18, 2026): **Transformer catch-up running simultaneously with CNN**, bounded to the same 15:32 MST deadline. Continue historical seed11 Transformer over corrected original expert corpus then each complete 2,000-game expert cohort, using unchanged W4/MPS trainer and validation-selected checkpoints. No deployment. Details: `experiments/catan-ai/TRANSFORMER_EXPERT_CATCHUP.md`.

Current execution (November 18, 2026): **user authorized another three-hour JSettlers expert-cohort window**. Resumed after two completed 2,000-game / 2,000-update cohorts; third cohort retained 986 completed games. Continue from the second cohort best-validation checkpoint using unchanged W4/MPS and 12 single-thread native workers. Progress: `runs/catan-ai/jsettlers-expert-cohorts-visible/progress.json`. No automatic deployment.

Current execution (November 18, 2026): **authorized expert-cohort recovery running for 110 minutes**, bounded to 11:23 AM MST. Reusing the complete 2,000-game expert cohort, deriving RAM samples then continuing unchanged W4/MPS training; subsequent cohorts require 2,000 newly completed games each. Original source and journals preserved with explicit migration under `runs/catan-ai/jsettlers-expert-cohorts-visible/repair/`. No automatic deployment.

Current status (November 18, 2026, 09:30 MST): **expert run stopped before training**. Collected 2,000 completed JSettlers expert games (1,800/100/100 splits), then sample derivation failed at 05:07 MST on a runner path-join typo. No model update occurred. Typo repaired and regression-tested; original journals and provenance preserved. Original window expired; no restart authorized. A future resume requires explicit source-provenance migration. See `runs/catan-ai/jsettlers-expert-cohorts-visible/repair/repair.json`.

Current execution (November 18, 2026): **user switched to additional 2,000-game JSettlers expert cohorts for three hours** after the current CNN scored 2/10 against JSettlers at 500 ms. Self-play is safely paused at 62 completed games plus 12 preserved interrupted prefixes. Continue the same fresh corrected-rule CNN using 12 native expert workers, 1,800/100/100 whole-game splits per cohort, unchanged W4/MPS training, and best-validation checkpoints. No automatic deployment or self-play restart. See `experiments/catan-ai/JSETTLERS_EXPERT_COHORTS.md`; progress: `runs/catan-ai/jsettlers-expert-cohorts-visible/progress.json`. This supersedes the older current-execution entries below for this window.

Current execution (November 18, 2026): **resumed with 12 single-thread CPU game workers by user preference**. Shared GPU inference remains experimental. The three-hour window ends November 18 at 06:06 AM MST; all active games save compressed replayable prefixes at the safe boundary. Preserved all 13 committed self-play games and the 2,000-game expert corpus/current fresh XD-CNN checkpoint. Continue collecting toward 2,000 completed self-play games before W4 updates, with whole-game expert splits and no automatic model deployment. Optimizer training retains its existing MPS configuration; inference is local CPU. Launcher: `.conda/bin/python -m agent_apis.neural.parallel_cohorts --hours 3 --workers 12`. Progress: `runs/catan-ai/fresh-xdcnn-visible-progress.json`; log: `runs/catan-ai/fresh-xdcnn-visible-parallel.log`. Procedure: `experiments/catan-ai/PARALLEL_CPU_COLLECTION.md`.

Current status (November 18, 2026): **shared GPU queue benchmark complete; training remains paused**. At twelve game workers, CPU averaged 8.83 searched moves/s versus 6.73 for the best GPU hybrid. Increasing GPU-backed concurrency to 32 workers allowed useful batches: fresh alternating confirmation averaged 9.84 moves/s for shared MPS XD-CNN + local CPU MLP versus 9.04 for twelve-worker CPU (8.8% faster). Peak summed process RSS: 14.33 versus 6.63 GB. All 1,512 timed diagnostic journals matched action/chance/visit traces and replayed successfully; live mixed-checkpoint/chunking parity passed. The queue is an opt-in benchmark implementation; no training collector default, playing model or W4 protocol was changed. Details: `experiments/catan-ai/SHARED_GPU_QUEUE_BENCHMARK.md`.

Current status (November 18, 2026): **worker scaling benchmark complete; training remains paused**. Tested 1/2/4/6/8 workers with one/two CPU threads and the current one-worker/six-thread baseline, then 8/10/12/14/16 single-thread workers on longer matched workloads. Recommend 12 single-thread workers: 9.01 searched moves/s, versus 9.04 at 14 and 8.97 at 16. All 1,488 diagnostic journals replayed and matched final checksums. MPS inference is slower for one position but faster in batches; shared GPU inference requires a separate implementation and end-to-end measurement. No controller or playing-model changes. Details: `experiments/catan-ai/GAME_WORKER_BENCHMARK.md`.

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
See [the smaller-round procedure](experiments/catan-ai/PHASE9_SELFPLAY_SMALL_ROUND.md).

Earlier execution status and historical scope follow.

Current execution (November 13, 2026): **search self-play started**, XD-CNN
D2/W0 seed11 only, with **friendly robber enabled by default** for new app games
and this procedure. Three iterations each collect 2,000 completed games at 512
new simulations per non-forced learned decision, then train 10,000 updates.
The 50/25/25 current/historical/JSettlers mixture, root exploration, subtree
reuse, resumable training, and noise-free paired tuning checks are implemented.
Historical corpora/results retain their original rules; fresh friendly-robber
parent evaluations provide the controls. No automatic model deployment.
See [self-play procedure](experiments/catan-ai/PHASE9_SELFPLAY.md).
Live status: `runs/catan-ai/selfplay-progress.json`; log: `runs/catan-ai/selfplay.log`.

Previous completed work and historical scope amendments follow.

Current work (November 13, 2026): cross-decision subtree/state reuse implemented
and tested before any further self-play training. Hidden-card posteriors refresh
on each real decision; belief-weighted visits reset conservatively. Reuse is
opt-in with `search.reuse_subtree=true`; default remains unchanged pending
stronger performance evidence. See [implementation and audit](experiments/catan-ai/SUBTREE_REUSE.md).
No training or new long evaluation queue was started for this optimization.

Latest status: the bounded phase-nine teacher-aggregation pilot completed.
Twenty collection games, 4,213 new examples, 1,000 fine-tuning updates, and ten
tuning validation games. Common supervised validation retained the original
parent weights; no trained checkpoint was promoted or deployed. No large queue
is running. Updated final.pt and all learner-state data remain available for
subsequent investigation. See [pilot results](experiments/catan-ai/PHASE9_PILOT_RESULTS.md). Full phase 9 is incomplete.

Current execution: bounded phase-9 **teacher data aggregation** pilot, XD-CNN
only. Twenty learner-versus-JSettlers generation attempts, 1,000 fine-tuning
updates with 50/50 original/new training data, then 10 paired tuning validation
games. No Transformer/random-player runs, no automatic large self-play queue
or default deployment. The parent policy supplies learner actions; native
SMART supplies separately verified labels at learner-reached states.
See [the pilot protocol](experiments/catan-ai/PHASE9_PILOT.md).

Latest status (November 12, 2026): phase 7 stopped early by user; evidence is
preserved in [PHASE7_EARLY_RESULTS.md](experiments/catan-ai/PHASE7_EARLY_RESULTS.md).
A bounded phase-eight pass completed all 12 validation positions with exact
fixed-simulation cache equivalence. Median measured throughput increased about
20%; this is a position diagnostic, not a playing-strength promotion. Full
chance-allocation sweeps remain deferred. No long evaluation queue is running.
The next substantive work is learner improvement/data collection (phase 9).
See [PHASE8_SHORT_RESULTS.md](experiments/catan-ai/PHASE8_SHORT_RESULTS.md).

Current execution amendment (November 12, 2026): phase 7 closed early at the
user request; remaining simulation-budget and confirmation matches are abandoned.
Use existing XD-CNN D2/W0 results, retain c=1.5/C0 provisionally, and do not
claim a demonstrated search gain. Phase 8 now uses a bounded 12-position
correctness/throughput and value/opponent diagnostic; the exhaustive chance
allocation sweep is deferred. No Transformer or random-player runs, no automatic
large match or self-play queue. See [PHASE7_EARLY_RESULTS.md](experiments/catan-ai/PHASE7_EARLY_RESULTS.md) for the preserved interim evidence.

Opponent amendment (November 11, 2026): **JSettlers only** from now on.
The user removed random-player testing. Completed JSettlers games are reused;
random-player history is retained but excluded from active scheduling and gates.
This halves phase-seven new-game limits to 400 screening games plus at most
100 additional confirmation games. Search settings and training are unchanged.

Phase 7 execution authorized November 11, 2026: XD-CNN D2/W0 search-budget and
exploration comparisons, with random player and JSettlers validation. Existing
phase-six checkpoints and compatible baselines are reused; no new training or
Transformer work. See [the phase-seven protocol](experiments/catan-ai/PHASE7.md).
Final confirmation seeds remain reserved.

Current scope amendment (November 11, 2026): **XD-CNN only**. The user retired
Transformer and stopped its remaining tests. All XD-CNN phase-six confirmation
runs are complete; see [phase-six results](experiments/catan-ai/PHASE6_RESULTS.md). The original two-architecture
comparison remains incomplete. W4 is promising, but its paired hierarchical
improvement interval touches zero, so the conservative recipe remains D2/W0.
Future experiment work uses XD-CNN; do not restart the historical Transformer queue.

# Catan AI experiment implementation plan

Prepared November 4, 2026. This document specifies future implementation and experiments; it does not implement an agent.

Execution status (November 4, 2026): phases 1–4 are complete. Phase 1 is implemented in
[`experiments/catan-ai/CONTRACT.md`](experiments/catan-ai/CONTRACT.md), with a
resolved baseline configuration, versioned run-record schema and executable
preflight. See the [phase-one results](experiments/catan-ai/PHASE1_RESULTS.md).
Phase 2's observation encoder, exact resource ledger and full-corpus audit are
documented in [the phase-two results](experiments/catan-ai/PHASE2_RESULTS.md).
Phase 3's frozen benchmark protocol, baseline smoke results and native
bank-trade regression fix are documented in
[the phase-three results](experiments/catan-ai/PHASE3_RESULTS.md).
Phase 4's root-sampled stochastic search, development-card belief and complete
1,000-position correctness audit are documented in
[the phase-four results](experiments/catan-ai/PHASE4_RESULTS.md).
Phase 5 is incomplete; the original long evaluation was interrupted.
User-requested shorter evaluations use a 10-game smoke and 50-game total with
the optimized search; see [the quick schedule](experiments/catan-ai/QUICK_EVALUATION.md). All four seed-11
models trained. Search optimization revision 1 adds cached dice equivalence,
world/observation reuse, batched threaded inference and no own-action depth cap.
See [the optimization report](experiments/catan-ai/SEARCH_OPTIMIZATION.md).
The shorter optimized-search evaluation completed November 7, 2026; see
[the quick results](experiments/catan-ai/PHASE5_QUICK_RESULTS.md). CNN won 12/50,
Transformer 9/50, GNN 1/40 and MLP 0/40; the latter two stopped for futility.
These are exploratory seed-11 results; full phase-5 confirmation remains incomplete.
Historical results remain preserved. Phase 6 execution is authorized November 7,
2026, using only Transformer and XD-CNN as candidate architectures, with random
player and JSettlers SMART as validation opponents. The shortened 10/50-game
schedule carries forward; full phase-5 confirmation is not claimed complete.
See [the phase-six protocol](experiments/catan-ai/PHASE6.md). Phase 6 screens
corpus/weighting recipes, then confirms uniform plus one challenger on both
encoders with seeds 11, 23 and 47. Phases 7–10 remain pending; final evaluation
seeds remain unused.

The objective is to select a policy/value architecture, pre-training distribution, and stochastic search configuration that beat the installed JSettlers SMART agent at a measured time budget. Start with two-player play. Compare a cross-dimensional CNN, a heterogeneous GNN, and a structural Transformer. Use a small MLP as the inexpensive control. Select models by playing strength at equal wall-clock budgets, with fixed-simulation comparisons to explain results.

## 1. Fixed scope and experiment contract

Use the project's current two-player rules throughout the initial experiments: BASE board, 15 victory points, discard above nine resources, standard robber, bank/port trading, and no player-to-player trades. BLUE and RED exchange seats in paired evaluation games. Four-player play, negotiation, and alternative rules are subsequent projects requiring new data and evaluation.

Resource hands are reconstructible in this two-player setting. Track every resource receipt, expenditure, discard, trade, steal, Year of Plenty choice, and Monopoly transfer. A theft always involves both players, so the victim can infer its result from their own hand change even when the opponent's action record redacts the resource. Do not use a probabilistic resource-hand model. Development-card identities and future chance outcomes remain uncertain.

All model comparisons receive the same information, use the same action vocabulary and search implementation, and train on the same game splits. Changes to the information available to a model constitute a separate experiment.

Initial implementation choices:

| Area | Decision |
| --- | --- |
| Training implementation | PyTorch; record and pin the installed version when implementation begins. Do not introduce multiple training frameworks. |
| Execution | Use the project-local environment. Detect supported accelerator hardware; retain CPU inference. Benchmark devices before selecting the evaluation device. |
| Agent location | Add a new provider under `agent_apis/`, registered through the existing provider catalog. Preserve the JSettlers integration. |
| Dynamics | Reuse Catanatron's rule engine; implement explicit chance sampling/aggregation around it. Do not train a world model. |
| Actions | Preserve the existing 332 BASE/two-player action IDs, legal masks, absolute board coordinates, and victim-color convention. |
| Policy | Mask illegal logits before softmax; use one global normalization across all legal actions. |
| Value | Predict acting-perspective win probability in [0, 1]. Convert existing labels with `(value_target + 1) / 2`. |
| Training targets | Demonstrated action plus actual completed-game outcome; later, search visit distribution plus self-play outcome. |
| Randomness | Independent RNG streams for board generation, simulator chance, search, training, and evaluation. Never search the authoritative future RNG stream or ordered deck. |
| Reproducibility | Training seeds 11, 23, and 47 for confirmed configurations; seed 11 for initial screening. |

Keep a machine-readable run record containing source revision/content hashes, dataset checksums, feature schema, split list, seed, configuration, hardware, software versions, checkpoint, elapsed time, and metrics. Store outputs under an ignored `runs/catan-ai/` directory. Keep configuration and analysis summaries in version control. No result may be described as statistically confirmed from a single training seed.

Dependency order: steps 1–4 establish the data and search scaffolding; step 5 trains the first models and connects them to that scaffolding; step 6 compares data/weighting recipes; step 7 measures and tunes the already connected search; steps 8–9 improve the finalists; step 10 locks and evaluates the final choice. Use a deterministic reference evaluator and heuristic opponent during step 4, replacing them with the step-5 neural components before learned-agent comparisons.

**Completion gate:** a written configuration contract and run-record schema exist; every later experiment can be identified by one configuration plus a seed.

## 2. Audit the current corpora and build a faithful observation pipeline

The current starting point is schema 2, documented in `DATASETS.md`:

| Corpus | Games | Retained examples | Retention |
| --- | ---: | ---: | --- |
| `data/jsettlers-corpora/full-250/` | 250 | 97,766 | Every decision |
| `data/jsettlers-corpora/downsampled/` | 1,017 | 64,503 | Every tenth decision per actor, all setup actions, final ten decisions per actor |
| Combined | 1,267 | 162,269 | Disjoint games; existing game-level splits |

Verify these counts against manifests before training. Both corpora already include 633 features, including exact tile numbers. Do not use the older 614-feature representation. The original source corpus overlaps the new bundle; do not add it again as independent data.

Implement a resource ledger initialized at the beginning of the game. Feed it the permitted event stream and own-hand snapshots. Reconstruct omitted events when loading downsampled games. Keep authoritative states available only to the audit/replay layer; derive model inputs through the same observation/ledger path used during live play.

Compare the ledger to authoritative resource hands after every action in every recorded game, including actions without retained feature rows. Require exact agreement, nonnegative counts, and resource conservation. Exercise restart recovery by rebuilding from history and comparing to the continuously updated ledger. If the current serialized history omits a victim's necessary private hand change, preserve that event information at the agent boundary or use the permitted exact bank-count conservation check; do not silently read a hidden hand. The two-player resource ledger must be exact before training proceeds.

Append the five inferred opponent resource counts in a versioned schema shared by every model. Include decision actor, turn owner, setup/free-road/discard phase, has-rolled and development-card restrictions, and rule constants where absent. Keep exact dice numbers as categorical inputs alongside production probabilities. Normalize bounded counts with documented fixed scales, using training-only statistics for any learned normalization.

Development-card inputs include own identities and playability, opponent card total, public card plays, and the public knowledge needed to constrain possible hidden holdings. Expose only the total remaining deck size, not its authoritative composition. Distinguish newly bought cards from cards eligible for play. Future recorded actions and terminal outcomes must never enter features.

Provide two data views:

1. **Retained-row view:** the current saved examples with their original retention pattern.
2. **Complete-decision view:** replay the complete action logs and reconstruct every pre-action observation, including omitted decisions. Stream or cache this as derived data with source hashes; do not modify the original shards.

The deterministic every-tenth retention rule is not independent random sampling. Do not claim that multiplying retained midgame rows by ten reconstructs an unbiased uniform-decision dataset. Use the complete-decision view for that control. Never treat adjacent retained rows as adjacent transitions for temporal-difference training.

**Measurable outputs and gate:** corpus/split/action-family/phase counts; zero cross-split game/seed overlap; 100% resource-ledger agreement; 100% legal demonstration targets; exact agreement between live and replay encoders on retained states; a report of any omitted or mismatched fields. Any mismatch blocks model comparisons.

## 3. Build the benchmark harness before model selection

Use the installed JSettlers SMART provider as the main external benchmark. Also include random legal play, the framework's heuristic value player if available, and frozen learned checkpoints as secondary opponents. Verify provider identities and revisions in each run. JSettlers errors must be reported, never replaced with a random fallback.

Generate disjoint tuning and final evaluation board/seed lists, separate from all corpus seeds. Freeze the final list before choosing the winning configuration. A board/seed pair defines two games with agent seats swapped. Shared seeds reduce some environmental variation but do not imply identical trajectories after actions diverge.

Evaluation schedule:

| Evaluation | Games per trained checkpoint | Purpose |
| --- | ---: | --- |
| Smoke | 20 seat-swapped pairs = 40 | API legality, completion, timing, and basic competence |
| Screening | 100 pairs = 200 on tuning seeds | Eliminate clearly weak/slow candidates |
| Confirmation | 500 pairs = 1,000 on final seeds | Compare shortlisted configurations; run all three training seeds |

Reserve final confirmation exclusively for step 10, after configurations and checkpoint choices are locked using tuning data. Steps 5–9 use tuning games only; expand to 500 tuning pairs when a promotion decision requires tighter intervals. Final results may support or reject a claim but must not trigger further tuning on those seeds.

The primary time budget is 500 ms per decision; also report 100 ms and 2,000 ms. Measure the complete agent decision, including observation encoding, belief preparation, search, inference, and action selection. Exclude opponent thinking time from the candidate's budget. Report cold start separately. Do not run competing benchmark jobs on the same accelerator during timing measurements.

Also compare 128, 512, and 2,048 simulations per non-forced decision. Pure-policy agents use no search. Forced moves return immediately. Start with fixed per-decision budgets; adaptive game-wide budgeting is a later ablation.

For time-budget experiments, stop admitting new work early enough to finish pending inference. Require p95 decision latency no greater than the budget plus `max(20 ms, 5% of budget)`, and report every overrun. A configuration that cannot satisfy this gate is not eligible for that budget's primary comparison.

Report wins, losses, cutoffs, completed-game win rate, overall match score, seat-specific results, turns, p50/p95/p99 decision time, throughput, and illegal actions/errors. Cap evaluation at 1,000 turns or 10,000 actions. Predeclare a cutoff as 0.5 match-score credit for both sides, separately labeled as unresolved; it is not a win/loss training target. Report sensitivity with cutoffs removed. A configuration with more than 1% cutoffs requires investigation before promotion.

Use 95% bootstrap intervals clustered by board/seat pair. For aggregate model results, report each training seed separately and use a hierarchical bootstrap over training seed and board pair; do not treat all moves as independent samples. Compare candidates on shared evaluation pairs. A performance claim over JSettlers requires a lower 95% bound above 50% match score, with cutoffs and failures disclosed.

**Completion gate:** repeatable baseline results, zero illegal actions, no silent failures, complete timing logs, and a frozen evaluation protocol. An experiment can finish with a negative performance result; promotion requires the stated evidence.

## 4. Implement and verify stochastic search and outcome merging

Implement explicit decision/action/chance structure. At decision nodes use PUCT with `c_puct = 1.5` initially. Maintain action-edge visits and values separately from chance-child statistics. At evaluation time choose the root action with the most visits, breaking ties by mean value and then stable action ID. Do not use exploration bonuses to select the final played action.

At chance nodes sample one outcome class from its true conditional probability distribution. Expand lazily, retain children, and accumulate repeated visits. Dice use exact two-dice probabilities; steals use exact reconstructed hand frequencies; development draws use remaining composition within a sampled feasible development-card world. Sample independent futures rather than cloning the authoritative RNG state.

### 4.1 Merge equivalent outcomes before expansion

For a given parent/action and sampled hidden world, group elementary chance outcomes by a canonical successor key. Add their probabilities and retain the constituent outcomes for diagnostics. Example: if neither a 3 nor a 4 produces resources or changes any relevant information, merge them into one child with probability `2/36 + 3/36 = 5/36`. Other equivalent rolls can join the same class. Distinct terminal results, robber/discard phases, bank-shortage effects, or information updates remain separate.

The key includes all dynamic facts that affect future legality, rewards, observations, or the policy: resource counts, bank, buildings/roads, robber, pieces, turn/decision actor, phase flags, development-card holdings and age constraints within the sampled world, and player knowledge. Exclude RNG state, object IDs, and raw past dice labels when they carry no remaining information.

Normalize observable history into sufficient event effects for these initial models. Do not feed irrelevant raw roll labels into an opponent policy while claiming the resulting branches are equivalent. If a later history encoder intentionally conditions on those labels, either normalize that encoder's inputs too or retain the distinction. Do not merge development-card worlds merely because they look identical to the root player: they can induce different future observations and actions. This first implementation merges outcomes within one world, not arbitrary belief states.

For each probability distribution require total mass 1 within numerical tolerance and preserve every positive-probability outcome through its class. Keep action identity separate even if two actions can reach the same state; their root action statistics are distinct.

### 4.2 Initial hidden-development-card and opponent treatment

Use root-player planning against a frozen opponent policy as the first search variant. Use the step-5 MLP trained with uniform decisions on D2 as this behavior-cloning policy; keep its checkpoint identical across architectural comparisons. Opponents sample legal actions from their own observations. They must not see the root player's private development cards or sampled future deck order.

At each simulation, sample a development-card allocation consistent with initial card supply, own knowledge, observed purchases/plays, and card-age restrictions. Use a constraint-consistent generative sampler or weighted particles with rejuvenation; reject inconsistent worlds. The baseline conditions on observable rule constraints but does not treat opponents' strategic choices as evidence about their hidden cards. Document this approximation. Do not naively sample current hidden types independently and ignore purchase/play history.

The tree is indexed by root action/observation history with normalized sufficient information. Hidden outcomes observed only by the opponent do not create distinguishable root decision nodes. Root-visible card draws do. Use the root player's value at evaluation frontiers and keep root utility fixed during backup. When calling an actor-perspective value model, convert perspectives explicitly; do not flip signs once per action.

Use POMCP-style root sampling/history sharing for this fixed-opponent baseline. A multi-observer ISMCTS variant, where each player searches for its own utility, is a later comparison. Do not claim that sampling a fully revealed state and running ordinary perfect-information search resolves private-information strategy fusion.

### 4.3 Search acceptance tests

- Handcrafted dice, theft, development-draw, immediate-win, sequential-action, discard, and free-road positions have expected legal transitions and values.
- Merged and unmerged chance enumeration have equal weighted one-step values within `1e-8` for a deterministic reference evaluator.
- Empirical merged-outcome frequencies agree with specified probabilities within predeclared multinomial confidence bounds over 100,000 draws.
- Altering the authoritative future RNG/deck order while holding permitted knowledge constant does not alter the agent's decision distribution under fixed search seeds.
- States indistinguishable to the root share root decision statistics; opponent-only information cannot choose a different root action directly.
- Record raw versus merged outcome counts, memory, simulations/second, decision depth, and resulting decisions on a fixed 1,000-position validation suite stratified by phase.

**Completion gate:** all correctness checks pass before any search optimization result is accepted. Retain unmerged search as a diagnostic ablation. No minimum speedup is assumed.

## 5. Implement four encoders with common heads and losses

All models use the audited schema and exact resource ledger. Every model emits logits in the same 332-action order and one win-probability value. For structured encoders, score location actions from the relevant entity embedding and global context; score scalar actions from global context. Map all scores back into the common vocabulary before masking and normalization. This avoids changing the action space during the architecture comparison.

| Model ID | Initial specification |
| --- | --- |
| MLP | Three hidden layers of width 256 with GELU activations; concatenate all permitted numeric/categorical features. |
| XD-CNN | Six residual cross-dimensional blocks; 64 spatial channels, scalar width 128, brick-coordinate layout, `5 x 3` convolutions; typed channels and valid-location masks. Broadcast scalar projections into spatial channels; feed spatial mean/variance summaries into the scalar stream. |
| H-GNN | Six residual typed message-passing blocks, hidden width 128. Entities: 19 tiles, 54 intersections, 72 road locations, two players, and global context. Relations: tile/intersection incidence, road/endpoints, ownership, ports, and player/global communication. Pool by entity type for value and scalar actions. |
| Transformer | Four pre-normalized Transformer blocks, hidden width 128, four attention heads, feed-forward width 512. Use the same entity inventory as H-GNN, type embeddings, and learned relation/distance attention biases. No unrestricted raw-history tokens initially. |

Specify port features on their adjacent intersections and preserve tile-number correlations. Use layer normalization and residual connections where applicable. Report parameter counts rather than assuming equal widths imply equal model sizes. Add a parameter-matched comparison only if capacity differences plausibly explain results; equal wall-clock play remains primary.

Training default: AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size 256, gradient norm clipping at 1.0, linear warmup for 500 updates followed by cosine decay to `3e-5`, maximum 20,000 updates. Use masked action cross entropy plus binary cross entropy for the value, each coefficient 1.0. Normalize each loss over its eligible samples; rows with one legal action contribute only to value. No auxiliary loss or label smoothing in the baseline.

Validate every 1,000 updates. For initial screening choose the checkpoint with lowest unweighted validation sum of policy and value losses, breaking ties in favor of the earlier checkpoint. Keep the final checkpoint too. In the later shortlist, use tuning-match results to test whether this checkpoint selection criterion agrees with playing strength; never choose checkpoints using final evaluation games.

Use board rotations/reflections with consistent transformation of board, ports, action labels, and legal masks. Use the same augmentation for every architecture. Validate mappings before enabling augmentation. Preserve actor-relative player encoding and correctly remap absolute victim colors. Randomize training game order and use a bounded shuffle buffer for state samples.

**Measurable outputs:** policy negative log likelihood/top-1 accuracy by phase and action family on non-forced decisions; value BCE, Brier score and calibration bins; parameters; training time; batch-1 and batched inference latency; pure-policy JSettlers match results. Complete all four seed-11 runs, then confirm the two strongest feasible architectures with seeds 23 and 47. Accuracy alone does not select the winner.

Use D2 with W0 for this first architecture comparison. Connect the trained models to step-4 search with its default settings and include 500 ms tuning matches when choosing the two finalists. This avoids eliminating a weaker standalone policy that supplies better values or priors for search.

## 6. Compare corpus composition and training weighting explicitly

First separate data coverage from weighting. Use the same architecture and update budget to compare:

- **D0:** complete decisions from `full-250` only.
- **D1:** retained rows from both corpora, sampled uniformly over retained rows.
- **D2:** complete reconstructed decisions from both corpora, sampled uniformly over all decisions.

D1 includes opening/endgame emphasis by construction. D2 is the reference for measuring weighting effects without that retention confound. D0 versus D2 measures the benefit of broader game coverage. Report unique games, sampled decisions, phase proportions, and action-family proportions actually consumed by each run.

On D2, compare these promising approaches. Keep expected gradient scale stable and do not apply an equivalent sampling correction twice.

| Weighting ID | Exact experiment | Value-head treatment |
| --- | --- | --- |
| W0 uniform decisions | Every decision equally likely. | Same distribution. |
| W1 uniform games | Sample a game uniformly, then a decision uniformly within it. | Same distribution. |
| W2 phase mixture | 50% uniform-decision stream; 50% split equally among setup, middle, and endgame pools. Setup takes precedence; endgame means either player's public score is at least 12; middle is the remainder. | Use inverse sampling weights to retain uniform-decision value loss. |
| W3 early/late curve | On uniform samples, policy weight `1 + 2 exp(-u/0.10) + 2 exp(-(1-u)/0.10)`, normalized to mean 1. `u` is actor decision ordinal divided by that actor's final ordinal, using complete trajectories. | Compare V0 uniform value weights against V1 weights `1 + exp(-(1-u)/0.10)`, normalized to mean 1. |
| W4 rare action families | 50% uniform stream; 50% family-balanced stream, with each row's relative oversampling capped at 5x its uniform rate. Determine families from action type, not individual location. Record the resulting normalized probabilities. | Correct back to uniform-decision value loss. |

Retrospective progress `u` is training metadata only. Never feed final game length or future endgame status into a model. Retain both winners' and losers' demonstrations; exclude censored games from terminal-outcome training.

Execution amendment (November 7, 2026): restrict candidate training to XD-CNN
and Transformer; validate each against random player and JSettlers in policy
and optimized 500 ms search. The frozen MLP remains only an internal search
opponent model. Carry forward 10-game smoke / 50-game total tuning evaluations.
The phase-six protocol fixes normalization, conservative uniform-control
confirmation, and resumable execution; it does not retroactively complete the
omitted phase-five confirmation.

Run D0/D1/D2 and W0–W4 initially with the strongest screened encoder and seed 11. Confirm the two best weighting recipes, including uniform if competitive, on both shortlisted encoders with all three seeds. Use equal optimizer updates and report wall time; do not exhaustively cross every preliminary option.

**Completion gate:** a comparison table reports imitation, calibration, and 500 ms JSettlers match results, including confidence intervals and consumed sampling proportions. Choose a weighting recipe only when it improves play or offers a clearly documented efficiency tradeoff. Keep uniform weighting if evidence is inconclusive.

## 7. Integrate neural evaluation and measure the value of search

Use policy priors at root-player decision nodes and neural value at newly expanded root-visible decision leaves. Continue across opponent-only phases with the frozen opponent policy until an appropriate root frontier or terminal state. Return exact terminal utility when known. Start with no rollout beyond the neural frontier.

Compare pure policy, neural PUCT at each fixed simulation budget, and neural PUCT at each time budget against JSettlers. Include search using a simple heuristic leaf evaluator as a control. Run a tuning sweep of `c_puct` in `{0.75, 1.5, 3.0}` for finalists only. At evaluation, use no root Dirichlet noise and deterministic root visit selection.

Record inference calls, cache hits, simulator time, encoding time, branching factor before/after merging, depth in completed turns, root visit entropy, and throughput. Report the entire strength-versus-latency curve. A larger model is worthwhile only if its improved predictions offset reduced search throughput at the target budget.

**Completion gate:** each shortlisted encoder has a pure-policy/search comparison. Search promotion requires a positive paired match-score improvement over its own pure-policy version at 500 ms, with a 95% interval excluding zero. If it fails, diagnose policy calibration, value quality, search perspective, and opponent-model error before adding complexity.

## 8. Compare chance allocation and throughput optimizations

Correctness-preserving implementation improvements come first: cache immutable topology, reduce state-copy cost, batch inference, reuse the compatible subtree after real observations, and bypass forced action selection. Cache neural outputs by permitted observation plus relevant belief summary; never reuse an output across different private knowledge. Keep action-edge visit statistics separate when sharing transposed evaluations.

For single-game search, test inference batches of 1, 8, and 32 using temporary virtual visits to avoid repeatedly selecting the same pending leaf. Measure waiting overhead and playing strength, not just GPU throughput. For data generation, batch across independent games. Profile CPU and accelerator inference separately.

Compare chance allocation at the primary 500 ms budget:

| ID | Sampling/allocation | Backup |
| --- | --- | --- |
| C0 baseline | Sample merged classes from true probabilities `p`. | Ordinary simulation return average. |
| C1 shallow stratification | At chance nodes before the root player's next turn, seed every merged class with a neural continuation estimate; allocate deeper samples by `p`. Deeper chance nodes use C0. | True-probability weighted child estimates; distinguish neural initialization from sampled evidence. |
| C2 probability concentration | In the same shallow region, sample with `q = 0.1 p + 0.9 normalize(p^1.5)`. | True-probability weighted per-class estimates. |
| C3 probability flattening | Same, with exponent `0.75`. | Same as C2. |
| C4 variance/cost allocation | After initial class coverage, allocate visits approximately in proportion to `p * sigma / sqrt(cost)` with a 10% true-probability exploration mixture and nonzero variance floor. | Same true-probability weighted estimates. |

These are deliberately separate experiments. C2 may increase depth on likely continuations; C3 may improve rare-event coverage; neither is assumed superior. When allocating differently from `p`, do not back up raw uncorrected averages. Maintain separate class estimates and aggregate with `p`. C1–C4's neural initializations introduce approximation until sampled; report this instead of claiming exact finite-budget unbiasedness. Deep importance-weighted trajectories are deferred because multiplying likelihood ratios can be unstable.

Distributional return variance is not the same as uncertainty in an estimated action value. Treat C4 as a heuristic inspired by stratified allocation, not a theorem about optimal adaptive MCTS. Measure whether its extra bookkeeping pays for itself.

Also compare action progressive widening only on nodes with more than 32 legal actions: initially admit the top eight priors; grow allowed children as `min(A, max(8, ceil(2 sqrt(N))))`, with 10% of new admissions selected uniformly among remaining legal actions. Always admit immediate winning actions. Keep the all-legal-actions implementation as the baseline.

**Measurable results:** paired win-rate differences, latency, true chance probability mass covered, depth, memory, nodes/second, and raw/merged branching counts. Promote an optimization if it improves strength with a positive paired 95% interval, or reduces median latency by at least 20% while the lower bound on match-score change exceeds -2 percentage points. Otherwise retain the simpler baseline.

## 9. Improve beyond heuristic imitation

Once supervised/search baselines are stable, compare three improvement routes on the same starting checkpoint:

1. **Teacher data aggregation:** generate 500 learner-versus-opponent games; query JSettlers at learner-reached decision states to add demonstrations. Use its live synchronization API with permitted history; verify it can reconstruct each queried state. A native translation failure excludes and reports that query, never invents a teacher label.
2. **Search reanalysis:** sample 20,000 training positions with the selected phase recipe; run 2,048-simulation search from permitted observations and development-card beliefs. Train on root visit distributions. Keep original completed-game outcomes as value targets initially; do not treat a searched value as a new observed outcome.
3. **Search self-play:** collect 2,000 completed games per iteration using 512 simulations per non-forced decision. Use a 50% current-agent, 25% historical-checkpoint, and 25% JSettlers opponent mixture. Retain search targets only for moves actually searched by the learned agent. Run three iterations before deciding whether to scale.

For self-play exploration use root noise only at decision nodes: 25% Dirichlet noise with concentration `10 / number_of_legal_actions` per action. Use visit temperature 1.0 during setup and the first 20 turns, then 0.25. Chance probabilities never receive exploration noise. Evaluation remains noise-free.

For the three self-play training iterations, use expert/new-data mixtures of 50/50, 25/75, and 10/90 respectively. Retain historical opponents and a uniform sample component. Exclude truncated outcomes from value labels and report their frequency. Train for 10,000 updates per iteration with the selected supervised optimizer settings, restarting the schedule from the current checkpoint; keep experimental compute accounting explicit.

Because root-player search depends on its opponent policy, compare the frozen imitation opponent model against a mixture of historical learned policies before interpreting self-play improvements. Later compare multi-observer ISMCTS to the fixed-opponent baseline using identical models and wall time. Action-likelihood conditioning of development-card beliefs is another separate ablation, not an unreported baseline change.

**Completion gate:** measure strength gain per generation hour, per training hour, and per generated game against JSettlers and frozen prior checkpoints. Promote a checkpoint only after its paired tuning interval shows improvement and no unexplained legality/calibration regression. Preserve failed iterations as results; do not silently select on the final evaluation set.

## 10. Optional architecture extensions and final selection

Only after the preceding comparisons, test the following independently:

- **GNN plus attention:** replace the last two GNN blocks with two global attention blocks of width 128 and four heads. Compare to the six-block GNN at equal time.
- **History memory:** add a small recurrent encoder of normalized observable events to the best encoder. Compare a history Transformer only if the recurrent addition helps. Resource accounting remains exact; memory is for development-card/opponent inference. Train on contiguous reconstructed sequences with a burn-in prefix, not adjacent retained rows.
- **Auxiliary learning:** predict development-card belief summaries or future observable production/build events. Compare one auxiliary task at a time with coefficient 0.1. Hidden-state labels may supervise an inference target in training, but never become inputs; audit leakage explicitly.

Select and lock the final configuration primarily by 500 ms tuning performance against JSettlers, supported by matches against diverse frozen agents and results at 100 ms and 2 seconds. If confidence intervals do not distinguish candidates, select the simpler/faster one and report the tie. Run the reserved final confirmation once for the locked candidate and predeclared comparison controls. Do not require a positive result from every experimental branch.

Final deliverables:

1. Registered agent runnable through the current Python and HTTP APIs.
2. Versioned encoder, resource-ledger implementation, development-card belief/search code, and executable correctness checks.
3. Reproducible configurations and three-seed results for architecture, weighting, and selected search comparisons.
4. Checkpoints with source/data/configuration hashes and supported-rule metadata.
5. A results report including JSettlers win rates and confidence intervals, latency curves, merged-branch savings, negative results, and selected defaults.

The project succeeds competitively when the selected agent's final 500 ms evaluation has a lower 95% match-score bound above 50% against JSettlers, zero illegal actions, and acceptable cutoff/failure rates. If that threshold is not reached, the implementation experiment still concludes with the strongest measured configuration and a quantified diagnosis; it must not claim to have surpassed JSettlers.

## Design references

- [Project dataset documentation](DATASETS.md) and [shared agent APIs](agent_apis/README.md).
- [Playing Catan with Cross-dimensional Neural Network](https://arxiv.org/html/2008.07079v1): brick-coordinate and spatial/scalar network design; its reported rules differ from this project's 15-point setup.
- [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261): graph-network building blocks.
- [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234): structural attention encodings.
- [Monte-Carlo Planning in Large POMDPs](https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html): root sampling and history-based planning with a generative model.
- [Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf): information-set search and determinization limitations.
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a): collecting teacher labels on learner-reached states.

Numerical model sizes, training schedules, sampling mixtures, budgets, and promotion thresholds above are proposed experiment specifications, not findings established by these papers.
