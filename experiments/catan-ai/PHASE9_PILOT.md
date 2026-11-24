# Phase 9: bounded learner-state teacher aggregation pilot

Continue with XD-CNN, D2/W0 seed-11 validation-selected checkpoint. This is the
teacher data aggregation route from phase 9, not search self-play or reanalysis.
The user requested moving past long search comparisons. The versioned pilot is
[phase9-pilot.json](phase9-pilot.json): 20 generation attempts, 1,000 updates,
and one 10-game tuning validation. No automatic larger run or installation.

Generate ten new board/simulator seed pairs with the learner in both seats
against native JSettlers SMART. Seeds are checked against all corpus and
benchmark seeds; generation is training-only. The frozen learner plays its
pure policy: prior experiments have not established a search advantage. At
every non-forced learner decision, query the JSettlers teacher from that actor's
redacted live observation. Verify SMART identity, state index, current seat,
legal selected action, and unchanged native translation-rejection count. The
learner still executes its own action. Teacher consultation does not change
the authoritative game or replace the learner's trajectory.

Failed queries are explicitly excluded, never assigned a fallback label.
Stop before training if more than 10% fail or none succeed. Opponent/API or
legality failures stop the pilot for investigation. Forced learner moves can
supply value examples but contribute no policy gradient. Only completed games
supply outcome labels; cut-off games are reported and excluded entirely from
this pilot's training. Observations are encoded before actions, and outcomes
are attached afterward. Original corpora, validation splits, and action schema
are immutable. New data are only features, legality, demonstrated teacher action
and observed learner-game outcome; future game metadata are not model inputs.

Fine-tune from the parent checkpoint using 50% uniformly sampled complete D2
training decisions and 50% uniformly sampled learner-state teacher rows. Keep
batch 256, the existing optimizer settings, independent sampling/augmentation
random streams and audited D6 symmetry transforms. Reset the optimizer and
restart the schedule with 100 warmup updates over 1,000 total updates. Retain
parent step 0 as a checkpoint-selection control; evaluate common unweighted D2
validation at 250-step intervals and select the earliest minimum policy NLL +
value BCE. Preserve final.pt and full optimizer/sampling/augmentation/MPS resume
state independently. The held-out test metrics do not select checkpoints.

Evaluate the chosen checkpoint in pure policy against JSettlers on the frozen
five tuning board/seat pairs (10 games). Reuse and checksum-verify the parent's
identical first ten games from phase 6 for a paired comparison. Final seeds
remain unused. Ten games are a pilot, not robust promotion evidence; report
uncertainty, legality, cutoffs, latency and the actual update chosen. No default
model changes automatically, even if the small comparison favors the pilot.

The collection resumes completed game records with data checksums. An interrupted
uncommitted game restarts at its same generation seed; no committed game is
replayed. Training saves resumable state every 100 updates and on exceptions.
Validation uses the existing per-game committed-prefix resume mechanism.
An exclusive benchmark lock prevents overlapping timing jobs.

Run/resume: `.conda/bin/python -m agent_apis.neural.phase9`.
Status: `runs/catan-ai/phase9-progress.json`.
Log: `runs/catan-ai/phase9.log`.
Outputs: `runs/catan-ai/phase9-pilot/`.
The final `PHASE9_PILOT_RESULTS.md`/`.json` appear only after completion.
The original 500-game teacher comparison, search reanalysis and multi-iteration
self-play experiments remain unexecuted; this pilot does not complete phase 9.
