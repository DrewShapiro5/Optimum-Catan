# Fresh XD-CNN with large game cohorts

Authorized November 17, 2026: restart XD-CNN from entirely random weights, use
thousands of games per cohort, retain all game events compactly, and sample
positions rather than visiting every position in every update. W4 and friendly
robber remain the defaults. Historical runs remain preserved and retired.

Run/resume: `.conda/bin/python -m agent_apis.neural.fresh_cohorts --hours 3`.
Progress: `runs/catan-ai/fresh-xdcnn-progress.json`.
Log: `runs/catan-ai/fresh-xdcnn.log`.
Protocol: [fresh-xdcnn.json](fresh-xdcnn.json).

1. Save a new, seeded, randomly initialized XD-CNN checkpoint. Only schema and
   configuration metadata are copied from an earlier checkpoint. No earlier
   XD-CNN learned state_dict is loaded into this model. The frozen historical
   MLP remains an auxiliary simulated-opponent policy; it is not XD-CNN weights.
2. Generate 2,000 completed friendly-robber JSettlers-versus-JSettlers games.
   Keep whole-game splits: 1,800 training, 100 validation and 100 test games.
   Interrupted games and cutoffs are retained without training outcome labels.
   No weight update is allowed before the full completed cohort exists.
3. Derive at most 64 seeded, uniformly selected positions per game into RAM.
   Selection never examines outcomes. Preserve all original positions in the
   compressed action/event journals; no state or feature snapshots are stored.
   Test games are never loaded for training or checkpoint selection.
4. Train the random model for 2,000 AdamW minibatch updates of 256 positions,
   using W4 sampling and inverse-probability value correction once. Initial
   training is expert-only (both sampler halves draw from the same new expert
   training pool). This is independent stochastic sampling, not full sweeps of
   every game's positions. Validate every 250 updates and retain the earliest
   best combined policy-NLL plus value-BCE checkpoint, including update zero as
   a control. Preserve final weights and atomic optimizer/RNG recovery state.
5. Subsequent self-play cohorts contain 2,000 newly completed games each before
   another 2,000-update training block. Use the existing 50/25/25 current,
   historical and JSettlers opponent mixture, 128 new simulations per searched
   move, and friendly robber. Train on 50% new expert positions and 50% positions
   sampled from the current complete self-play cohort. No prior failed-run
   XD-CNN checkpoints or old self-play datasets are imported into this lineage.
6. Run bounded pure-policy tuning comparisons after a completed self-play
   training cohort; these do not establish fixed-search-budget playing strength.
   No automatic deployment occurs.

Every generated game uses a gzip journal containing isolated seeds, action IDs,
actual chance events, source/model references where relevant, sparse search
visit targets and initial/final state-and-RNG hashes. Replay reconstructs the
initial game, verifies each chance event and the final checksum, and redacts
actor observations before deriving features. One source archive and necessary
model/optimizer checkpoints are retained. Partial prefixes are checkpointed;
completed games are never regenerated on resume. Native randomness need not be
recreated because all chosen actions and chance events are recorded.

Execution remains in at-most-three-hour windows; an unfinished cohort pauses
and resumes without early training on a small prefix. A 2,000-game search cohort
will require much longer than one such window at this machine's measured
throughput. No scheduled review or automatic extension is created for this
restart. Keep all earlier artifacts intact.

## Phase-transition recovery

The first collection finished all 2,000 games, then stopped before any optimizer
update because the pretraining parent directory was missing. The directory was
created without changing pinned sources or generated data. Recovery re-verifies
committed journals rather than generating new games. The recovery launcher is
recorded in `runs/catan-ai/fresh-xdcnn-launch.json`; it explicitly sets
`friendly_w4.CORPUS = fresh_cohorts.OUTPUT / 'expert'` so training metadata and
resume digests refer to this new corpus. Use that setting for subsequent resumes.
The recovery window uses the original unused 3,311 seconds of compute budget.
All earlier failure details remain in
`runs/catan-ai/fresh-xdcnn/failure-before-output-folder-recovery.json`.
